# SPDX-License-Identifier: Apache-2.0
"""PLaMo 3 long mode (iGPU prefill + NPU decode, up to 262,144 tokens): shared pure logic.

Nothing here touches a device, so every function is unit-testable on any machine. Sealed modules (plamo_errors, plamo_lock) are imported read-only
when the installed tree is present; otherwise small stand-ins with the same exit codes are used (unit tests, development).
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

CONTEXT_DEFAULT = 2048          # the default path's validated context (prompt + output)
DEFAULT_PROMPT_LIMIT = CONTEXT_DEFAULT - 3   # cli83: prompts over 2045 tokens are refused by the default path
MAX_TOTAL = 262144              # long mode: prompt + output positions
DEFAULT_CAP = 256               # the default path's implicit --num-tokens
TEMPLATE_OVERHEAD = 40          # upper bound of chat-template tokens around the text
WARN_ABOVE_TOKENS = 2048        # Phase 4: no length above the default's 2,048-token context passed the predeclared B2 bars (PHASE4.md), so the warning applies to every long-mode input
MIN_FREE_RAM_MB = 3500
EOS_IDS = (1, 16)
CHUNK = 1024                    # iGPU prefill rows per step
ENV_SIZE_PREFIX = 'PLAMO3_LONG_'

_SEALED = Path(__file__).resolve().parents[1] / 'IRON' / 'iron' / 'applications' / 'plamo3'
if _SEALED.is_dir() and str(_SEALED) not in sys.path:
    sys.path.insert(0, str(_SEALED))

try:  # sealed error model (same exit codes everywhere)
    import plamo_errors as errors
except ImportError:  # unit tests / development tree
    class _E:
        OK, INTERNAL, USAGE, INSTALL, DEGRADED, NPU, ENVIRONMENT, INTERRUPTED = 0, 1, 2, 3, 4, 5, 6, 130

        class PlamoError(Exception):
            code = 1

            def __init__(self, message, code=None, hint=None):
                super().__init__(message)
                self.message = str(message)
                if code is not None:
                    self.code = code
                self.hint = hint

        class InvalidRequest(PlamoError):
            code = 2

        class InstallProblem(PlamoError):
            code = 3

        class NpuBusy(PlamoError):
            code = 5

        class NpuUnavailable(PlamoError):
            code = 5

        class EnvironmentProblem(PlamoError):
            code = 6
    errors = _E


class LongModeError(errors.EnvironmentProblem):
    """Long mode cannot run on this machine (no iGPU runtime, too little memory ...). The default path is unaffected."""


class Cancelled(errors.PlamoError):
    code = 130


class LongRequestError(errors.InvalidRequest):
    pass


# ---------------------------------------------------------------- mode selection (design 3.2)
def needs_router(text_bytes, explicit_cap, force_long):
    """Cheap wrapper pre-filter (proof: every UTF-8 byte is at most one byte-fallback token, plus TEMPLATE_OVERHEAD template tokens).
    False = the request provably fits the default path; True = call the router, which tokenizes exactly."""
    if force_long:
        return True
    cap = DEFAULT_CAP if explicit_cap is None else explicit_cap
    return text_bytes + TEMPLATE_OVERHEAD + cap > CONTEXT_DEFAULT


def decide(prompt_tokens, explicit_cap, force_long):
    """'default' or 'long'. explicit_cap is the user's --num-tokens or None.
    Default path iff P + M <= 2048 (M the explicit cap, or nothing when it is only the implicit 256) and not forced; a prompt of 1793..2045 tokens
    with an implicit cap stays on the default path (it stops at the context, unchanged behaviour)."""
    if prompt_tokens < 1:
        raise LongRequestError('the prompt is empty')
    if prompt_tokens > MAX_TOTAL - 1:
        raise LongRequestError(f'input is {prompt_tokens} tokens; the limit is {MAX_TOTAL - 1} (prompt plus translation must fit in {MAX_TOTAL} tokens). Split the text.')
    if force_long:
        return 'long'
    if prompt_tokens > DEFAULT_PROMPT_LIMIT:
        return 'long'
    if explicit_cap is not None and prompt_tokens + explicit_cap > CONTEXT_DEFAULT:
        return 'long'
    return 'default'


def long_cap(prompt_tokens, explicit_cap):
    """Output cap in long mode: the user's value (must fit), else min(remaining, max(1024, 2P))."""
    remaining = MAX_TOTAL - prompt_tokens
    if explicit_cap is not None:
        if isinstance(explicit_cap, bool) or not isinstance(explicit_cap, int) or explicit_cap < 1:
            raise LongRequestError('--num-tokens must be a positive integer')
        if explicit_cap > remaining:
            raise LongRequestError(f'--num-tokens {explicit_cap} does not fit: the prompt has {prompt_tokens} tokens and the limit is {MAX_TOTAL} in total (at most {remaining} new tokens)')
        return explicit_cap
    return min(remaining, max(1024, 2 * prompt_tokens))


def length_warning(prompt_tokens):
    """The one stderr line printed when P is above the verified bucket (a limit of the model itself, not an error); None otherwise."""
    if prompt_tokens > WARN_ABOVE_TOKENS:
        return ('PLaMo: note: inputs this long (%d tokens) may repeat or stay untranslated; this is a limit of the model itself (it also happens with the original BF16 model). '
                'The run continues.' % prompt_tokens)
    return None


# ---------------------------------------------------------------- estimates and progress
def fit_linear(samples):
    """Least squares y = a + b x over [(x, y)]; (mean, 0) when degenerate."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    sx = sum(x for x, _ in samples); sy = sum(y for _, y in samples)
    sxx = sum(x * x for x, _ in samples); sxy = sum(x * y for x, y in samples)
    den = n * sxx - sx * sx
    if n < 3 or abs(den) < 1e-9:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / den
    return (sy - b * sx) / n, b


def prefill_eta_s(samples, next_pos, n_total, chunk=CHUNK):
    """Remaining prefill seconds. samples = [(pos0, seconds)] of finished chunks; cost per chunk is linear in pos0 (full-layer attention)."""
    if next_pos >= n_total:
        return 0.0
    if not samples:
        return None
    a, b = fit_linear(samples[-64:])
    total = 0.0
    pos = next_pos
    while pos < n_total:
        total += max(0.0, a + b * pos)
        pos += chunk
    return total


def prefill_estimate_s(n_tokens, chunk=CHUNK):
    """A priori estimate from the Phase 0/1 measurements on the 780M: used for the up-front notice only."""
    chunks = math.ceil(n_tokens / chunk)
    mean = 0.33 + 0.65 * (n_tokens / MAX_TOTAL) ** 1.5   # seconds per chunk: 0.35 @8k, 0.37 @32k, 0.52 @131k, 0.86-1.13 @262k measured (thermal spread +-30%)
    return chunks * mean + 4.0


def decode_estimate_s(prompt_tokens, new_tokens):
    """ms/token measured: 19.4 short, 21 at 8k, 22 at 16k, 24 at 32k, 35 at 130k, 40 at 171k, 50 at 261k (piecewise linear)."""
    pts = [(0, 19.4), (8192, 21.3), (16384, 22.2), (32768, 24.0), (131072, 35.0), (171000, 39.7), (261120, 49.9)]
    mid = prompt_tokens + new_tokens / 2
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if mid <= x1:
            return new_tokens * (y0 + (y1 - y0) * (mid - x0) / (x1 - x0)) / 1000.0
    return new_tokens * pts[-1][1] / 1000.0


def fmt_s(seconds):
    if seconds is None:
        return '?'
    seconds = int(round(seconds))
    return '%d:%02d' % divmod(seconds, 60) if seconds < 3600 else '%d:%02d:%02d' % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


class Progress:
    """One stderr line per interval; never raises; thread-free (called from the work loops)."""

    def __init__(self, stream=None, interval_s=5.0, enabled=True, clock=time.monotonic):
        self.stream = stream or sys.stderr
        self.interval_s, self.enabled, self.clock = interval_s, enabled, clock
        self.last = None

    def line(self, text, force=False):
        if not self.enabled:
            return
        now = self.clock()
        if force or self.last is None or now - self.last >= self.interval_s:
            self.last = now
            try:
                print('PLaMo: ' + text, file=self.stream, flush=True)
            except Exception:
                pass


# ---------------------------------------------------------------- dispatch splitting plan (design: cap each dispatch at about 200 ms)
def split_plan(rows, per_row_ms, cap_ms, block_rows):
    """Row ranges for one attention launch. Every boundary is a multiple of block_rows, so each kernel work-group sees the same rows and the same
    key range as in the single launch (bitwise-identical output; L7 proves it). Returns [(row0, nrows)]."""
    if rows <= 0:
        return []
    if per_row_ms is None or per_row_ms * rows <= cap_ms:
        return [(0, rows)]
    tile = max(block_rows, int(cap_ms / per_row_ms) // block_rows * block_rows)
    plan, r = [], 0
    while r < rows:
        plan.append((r, min(tile, rows - r)))
        r += tile
    return plan


class DispatchGovernor:
    """Measures the first full-attention launch of each chunk (sync + timer) and predicts the per-row time for the next one with a 25% margin.
    The measured launch of the FIRST deep chunk is itself unsplit; its time is bounded by chunk size and position (documented margin: 578 ms at 262k on the 780M)."""

    def __init__(self, cap_ms=200.0, margin=1.25, measure_every=1):
        self.cap_ms, self.margin, self.every = cap_ms, margin, measure_every
        self.per_row_ms = None
        self.count = 0
        self.max_launch_ms = 0.0
        self.splits = 0

    def want_measure(self):
        self.count += 1
        return self.per_row_ms is None or (self.count - 1) % self.every == 0

    def observe(self, rows, ms):
        if rows > 0:
            self.per_row_ms = ms / rows * self.margin
        self.max_launch_ms = max(self.max_launch_ms, ms)

    def plan(self, rows, block_rows):
        p = split_plan(rows, self.per_row_ms, self.cap_ms, block_rows)
        if len(p) > 1:
            self.splits += 1
        return p


# ---------------------------------------------------------------- KV handoff directory
HANDOFF_VERSION = 1
KV_LAYERS = 16


def prompt_digest(token_ids):
    h = hashlib.sha256()
    for i in range(0, len(token_ids), 65536):
        h.update(('\n'.join(str(int(t)) for t in token_ids[i:i + 65536]) + '\n').encode('ascii'))
    return h.hexdigest()


def write_handoff_record(dirpath, n_tokens, digest, extra=None):
    rec = dict(version=HANDOFF_VERSION, n_tokens=int(n_tokens), prompt_sha256=digest, created=time.time(), pid=os.getpid())
    rec.update(extra or {})
    tmp = Path(dirpath) / 'handoff.json.tmp'
    tmp.write_text(json.dumps(rec), encoding='ascii')
    os.replace(tmp, Path(dirpath) / 'handoff.json')


def check_handoff(dirpath, n_tokens, digest):
    """Verifies the iGPU -> NPU handoff directory (record, sizes of all 32 KV files, logits file). Raises LongModeError with a one-line reason."""
    d = Path(dirpath)
    try:
        rec = json.loads((d / 'handoff.json').read_text(encoding='ascii'))
        meta = json.loads((d / 'meta.json').read_text())
    except (OSError, ValueError):
        raise LongModeError('the prefill result is missing or unreadable (handoff record); the iGPU step probably failed')
    if rec.get('version') != HANDOFF_VERSION or rec.get('n_tokens') != n_tokens or rec.get('prompt_sha256') != digest or meta.get('n_tokens') != n_tokens:
        raise LongModeError('the prefill result does not belong to this request (stale or foreign handoff directory)')
    layers = meta.get('layers') or []
    if len(layers) != KV_LAYERS:
        raise LongModeError('the prefill result is incomplete (layer count)')
    for item in layers:
        for kind in ('k', 'v'):
            f = d / 'kv' / f'{kind}_{item["layer"]}.bin' if (d / 'kv').is_dir() else d / f'{kind}_{item["layer"]}.bin'
            try:
                size = f.stat().st_size
            except OSError:
                raise LongModeError('the prefill result is incomplete (missing %s)' % f.name)
            if size != int(item['rows']) * 256:
                raise LongModeError('the prefill result is truncated (%s)' % f.name)
    try:
        if (d / 'last_logits.bin').stat().st_size != 107520 * 4:
            raise LongModeError('the prefill result is truncated (last_logits.bin)')
    except OSError:
        raise LongModeError('the prefill result is incomplete (last_logits.bin)')
    return rec, meta


def stale_handoff_dirs(state_dir, max_age_s=3600, now=None):
    """Directories under state\\long older than max_age_s (candidates for cleanup by the process that holds the lock)."""
    now = time.time() if now is None else now
    out = []
    try:
        for child in Path(state_dir).iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_s:
                out.append(child)
    except OSError:
        pass
    return out


# ---------------------------------------------------------------- environment checks
def free_ram_mb():
    """Free physical memory in MB (Windows: GlobalMemoryStatusEx via ctypes; elsewhere /proc/meminfo); None when unknown."""
    try:
        if os.name == 'nt':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            s = MS(); s.dwLength = ctypes.sizeof(MS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                return None
            return int(s.ullAvailPhys // (1 << 20))
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def check_ram(free_mb=None, need_mb=MIN_FREE_RAM_MB):
    free = free_ram_mb() if free_mb is None else free_mb
    if free is not None and free < need_mb:
        raise LongModeError('not enough free memory for long mode: %d MB free, %d MB needed (the iGPU step peaks at about 2.7 GB of shared memory). Close other programs and try again.' % (free, need_mb))
    return free


def on_battery():
    """True when running on battery (Windows GetSystemPowerStatus); None when unknown. Used for a notice only."""
    try:
        if os.name != 'nt':
            return None
        import ctypes

        class SPS(ctypes.Structure):
            _fields_ = [('ACLineStatus', ctypes.c_ubyte), ('BatteryFlag', ctypes.c_ubyte), ('BatteryLifePercent', ctypes.c_ubyte), ('SystemStatusFlag', ctypes.c_ubyte),
                        ('BatteryLifeTime', ctypes.c_ulong), ('BatteryFullLifeTime', ctypes.c_ulong)]
        s = SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        return s.ACLineStatus == 0
    except Exception:
        return None


# ---------------------------------------------------------------- install layout
def install_root():
    """<root> from this file's location: <root>\\long\\long_core.py."""
    return Path(__file__).resolve().parents[1]


def sealed_paths(root=None):
    root = Path(root) if root else install_root()
    plamo = root / 'IRON' / 'iron' / 'applications' / 'plamo3'
    return plamo


def finish_text(tokenizer, tokens, capped):
    """Same hold-back rule as the default CLI (cli83.finish_text): an incomplete trailing UTF-8 sequence of byte-fallback tokens is not printed when the output was cut by the cap."""
    text = tokenizer.decode(tokens)
    if not capped or not text.endswith('�'):
        return text, 0
    table = getattr(tokenizer, 'tokens', None)
    if table is None:
        return text, 0
    import codecs
    pending = []
    for index in reversed(tokens):
        token = table[index] if 0 <= index < len(table) else ''
        if isinstance(token, str) and token.startswith('<0x') and token.endswith('>') and len(token) == 6:
            pending.append(int(token[3:5], 16))
        else:
            break
    pending.reverse()
    if not pending:
        return text, 0
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    decoder.decode(bytes(pending), final=False)
    held = len(decoder.getstate()[0])
    if not held:
        return text, 0
    return tokenizer.decode(tokens[:len(tokens) - held]), held


def stop_reason(tokens, cap, prompt_tokens, repeat=None):
    if repeat:
        return 'repeat'
    if tokens and tokens[-1] in EOS_IDS:
        return 'eos'
    if prompt_tokens + len(tokens) >= MAX_TOTAL:
        return 'context'
    return 'cap'


# ---------------------------------------------------------------- repetition stop (on by default; --no-repeat-stop / -NoRepeatStop)
REPEAT_MIN_PERIOD = 20
REPEAT_TIMES = 3
REPEAT_MAX_PERIOD = 4096


class RepeatDetector:
    """Incremental loop detector: at least REPEAT_TIMES consecutive repeats of a period of at least REPEAT_MIN_PERIOD tokens (the Phase 4 loop definition).
    feed(tokens) after each new token returns None or dict(start, period, keep): `keep` = number of tokens to output, i.e. the end of the FIRST occurrence of the
    repeated period, so the stopped output is an exact prefix of the greedy output. Candidate periods are the distances to earlier occurrences of the newest token
    (bounded memory per token), so a check costs a few slice comparisons."""

    def __init__(self, min_period=REPEAT_MIN_PERIOD, times=REPEAT_TIMES, max_period=REPEAT_MAX_PERIOD, per_token=64):
        self.min_period, self.times, self.max_period, self.per_token = min_period, times, max_period, per_token
        self.positions = {}

    def feed(self, tokens):
        i = len(tokens) - 1
        t = tokens[i]
        lst = self.positions.setdefault(t, [])
        hit = None
        n = self.times
        for j in reversed(lst):
            p = i - j
            if p > self.max_period:
                break
            if p < self.min_period or i + 1 < n * p:
                continue
            if all(tokens[i - k * p] == tokens[i - (k + 1) * p] for k in range(n - 1)) and tokens[i - (n - 1) * p + 1:i + 1] == tokens[i - (n - 1) * p + 1 - p:i + 1 - p]:
                hit = p
                break
        lst.append(i)
        if len(lst) > self.per_token:
            del lst[0]
        if hit is None:
            return None
        p = hit
        start = i - n * p + 1
        while start > 0 and tokens[start - 1] == tokens[start - 1 + p]:
            start -= 1
        return dict(start=start, period=p, keep=start + p)


def find_first_loop(tokens, **kw):
    """Reference (offline) use of RepeatDetector on a full token list: the first detection or None. Used by tests and the Phase 4 gates."""
    d = RepeatDetector(**kw)
    buf = []
    for t in tokens:
        buf.append(t)
        r = d.feed(buf)
        if r:
            r['at'] = len(buf) - 1
            return r
    return None
