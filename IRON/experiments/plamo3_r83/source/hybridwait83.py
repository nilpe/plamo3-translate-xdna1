"""R83 hybrid wait: process-local, exact, a completion HINT only.

Why: after a command is submitted the decode thread sleeps in runlist.wait() for 2-18 ms. The thread's core then drops into a deep idle state and the wake-up costs
0.2-0.5 ms per wait (R82: 1.28 ms of the 1.38 ms in-loop excess per M2 cycle). This module sleeps on a per-timer high-resolution waitable timer until shortly before the
expected completion (delta ms early), then polls a device-written sentinel until the command has produced its last output, and then ALWAYS calls the ordinary
runlist.wait() exactly as before. Synchronisation, output syncs and every byte of every command are unchanged; the poll only decides WHEN the ordinary wait is entered.

Exactness: the sentinel is written only into a view that the command overwrites completely before the host reads it (static proof: sentinel83_proof / sentinel-proof gate), flushed
BO_TO_DEVICE before execute (an unflushed sentinel can be written back over the device result by the CPU cache), and the poll reads with BO_FROM_DEVICE. If the poll cannot
decide (timeout 3 x estimate + 5 ms, any exception, a valid output equal to the sentinel pattern) the ordinary wait is called at once and a fallback is counted.

Scope: DefaultRuntime w4-m2 only, keep-warm off; graphs decode / r37_decode_bulk_taps / verify2 (token views) and r41_proposal_join / r48_proposal_join_en / r41_exact (drafter host
buffers). r41_refresh_join / r40_refresh_join keep the ordinary wait (no host-visible output). Not installed for the w4-serial / w8-serial options, --service of other modes, or with keep-warm.
Integration: the runtime INSTANCE is wrapped (rt.execute, rt.xrt, existing runlists in rt._graphs); pyxrt itself is never patched. Off switch: PLAMO3_HYBRID_WAIT=0 (the ordinary path, nothing wrapped).
Forbidden here and absent: timeBeginPeriod, NtSetTimerResolution, power/scheduler/priority/affinity calls, helper threads or processes."""
import os, sys, time, ctypes, collections

ENV = 'PLAMO3_HYBRID_WAIT'
DEFAULT_DELTA_MS = 1.5          # pre-declared sweep result (PREDECLARED_DELTA.md; ROUND83.md section 3): lowest median cycle among eligible deltas
WARMUP_CALLS = 4                # per graph: ordinary waits (their timings seed the estimate)
HISTORY = 16
TIMEOUT_FACTOR = 3.0
TIMEOUT_SLACK_MS = 5.0
MAIN_GRAPHS = ('decode', 'r37_decode_bulk_taps', 'verify2')
DRAFT_GRAPHS = ('r41_proposal_join', 'r48_proposal_join_en', 'r41_exact')
SENTINEL_WORD = b'\xff\xff\xff\xff'
BREAKER_THRESHOLD = 4        # H1 circuit breaker: consecutive poll timeouts of ONE graph after which that graph uses the ordinary wait only (R83 data: 19 timeouts in 93,070 waits, never consecutive)
_OFF = ('0', 'off', 'false', 'no')
_ON = ('1', 'on', 'true', 'yes')


def enabled(environ=None):
    v = (os.environ if environ is None else environ).get(ENV)
    if v is None:
        return True
    low = v.strip().lower()
    if low not in _OFF and low not in _ON and low != '':
        # H1 (BUG 12): a value that is neither on nor off was silently treated as 'on'
        try:
            from plamo_errors import note
            note('PLaMo: note: %s=%r is not recognised (use 0 or 1); the hybrid wait stays on.' % (ENV, v), key='hybrid-env')
        except Exception:
            pass
    return low not in _OFF


class _Timer:
    """Per-timer high-resolution waitable timer (kernel32; no system-wide timer resolution change)."""
    def __init__(self):
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateWaitableTimerExW.restype = ctypes.c_void_p
        k.CreateWaitableTimerExW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong]
        k.SetWaitableTimer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.WaitForSingleObject.restype = ctypes.c_ulong
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k.CreateWaitableTimerExW(None, None, 0x2, 0x1F0003)   # CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS
        if not h:
            raise OSError('CreateWaitableTimerExW(high resolution) failed: %d' % ctypes.get_last_error())
        self._k = k
        self._h = h

    def sleep_ns(self, n):
        if n <= 0:
            return
        due = ctypes.c_longlong(-max(1, n // 100))
        if not self._k.SetWaitableTimer(self._h, ctypes.byref(due), 0, None, None, 0):
            raise OSError('SetWaitableTimer failed')
        self._k.WaitForSingleObject(self._h, int(n / 1e6) + 50)   # bounded: never an unbounded sleep

    def close(self):
        h, self._h = self._h, None
        if h:
            self._k.CloseHandle(h)


class ViewSentinel:
    """Token/status view of the runtime (rt.views[view] host mapping + its BO)."""
    def __init__(self, rt, view):
        self.rt = rt
        self.view = view
        self.mv = rt.views[view]
        self.n = len(self.mv)
        assert 0 < self.n <= 64 and self.n % 4 == 0, ('sentinel view size', view, self.n)
        self.bo = rt._bo(view)
        d = rt.xrt.xclBOSyncDirection
        self.to_dev, self.from_dev = d.XCL_BO_SYNC_BO_TO_DEVICE, d.XCL_BO_SYNC_BO_FROM_DEVICE

    def prepare(self):
        self.mv[:] = SENTINEL_WORD * (self.n // 4)
        self.bo.sync(self.to_dev)                       # flush: no dirty CPU line may survive the device write

    def done(self):
        self.bo.sync(self.from_dev)
        b = bytes(self.mv)
        return all(b[i:i + 4] != SENTINEL_WORD for i in range(0, self.n, 4))


class ArraySentinel:
    """Drafter host buffer (numpy uint16) with its own sync callable; sentinel = 0xFFFF in every element."""
    def __init__(self, drafter, array_name, sync_name):
        self.d = drafter
        self.an = array_name
        self.sn = sync_name

    def prepare(self):
        d = self.d
        getattr(d, self.an)[:] = 0xFFFF
        getattr(d, self.sn)(d._to)

    def done(self):
        d = self.d
        getattr(d, self.sn)(d._from)
        return not (getattr(d, self.an) == 0xFFFF).any()


class _HybridRunlist:
    """Instance-level wrapper of one pyxrt runlist. add() and every other attribute are forwarded unchanged."""
    __slots__ = ('_real', '_hw', '_pending')

    def __init__(self, hw, real):
        self._real = real
        self._hw = hw
        self._pending = None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def add(self, run):
        return self._real.add(run)

    def execute(self):
        hw = self._hw
        pending = hw._begin() if (hw is not None and hw.active) else None
        self._real.execute()
        if pending is not None:
            pending['t1'] = time.perf_counter_ns()
            self._pending = pending

    def wait(self, *args):
        if args:
            return self._real.wait(*args)
        p, self._pending = self._pending, None
        hw = self._hw
        if p is not None and hw is not None:
            hw._hybrid(p)                                # never raises
        out = self._real.wait()                          # ALWAYS: the ordinary synchronisation point
        if p is not None and hw is not None and p.get('record'):
            hw._record(p, time.perf_counter_ns())
        return out


class _XrtProxy:
    """rt.xrt for the runtime instance: runlist() returns wrapped runlists, everything else is forwarded."""
    def __init__(self, hw, real):
        object.__setattr__(self, '_hw', hw)
        object.__setattr__(self, '_real', real)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def runlist(self, *a, **k):
        return _HybridRunlist(self._hw, self._real.runlist(*a, **k))


class HybridWait:
    def __init__(self, rt, drafter=None, delta_ms=DEFAULT_DELTA_MS, log=None):
        self.rt = rt
        self.drafter = drafter
        self.delta_ns = int(delta_ms * 1e6)
        self.log = log
        self.active = False
        self.installed = False
        self.current = None
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.stats = collections.Counter()
        self.consec = collections.Counter()   # H1: consecutive timeouts per graph
        self.disabled = {}                    # H1: graph -> reason; a disabled graph only ever uses the ordinary wait
        self.breaker_threshold = BREAKER_THRESHOLD
        self.timer = _Timer()
        self.specs = {}
        M = rt.manifest['graphs']
        for g in MAIN_GRAPHS:
            so = M[g].get('sync_outputs', [])
            assert len(so) == 1, ('graph without a single sync output view', g, so)
            self.specs[g] = ViewSentinel(rt, so[0])
        if drafter is not None:
            for g, an, sn in (('r41_proposal_join', '_coarse_host', '_coarse_sync'), ('r48_proposal_join_en', '_coarse_host', '_coarse_sync'), ('r41_exact', '_exact_host', '_exact_sync')):
                if hasattr(drafter, an) and hasattr(drafter, sn):
                    self.specs[g] = ArraySentinel(drafter, an, sn)

    # ---- installation (instance only) ----
    def install(self):
        rt = self.rt
        assert not self.installed
        self._orig_execute = rt.execute
        self._had_execute_attr = 'execute' in rt.__dict__
        hw = self

        def execute(graph, values):
            hw.current = graph
            return hw._orig_execute(graph, values)
        rt.execute = execute
        self._wrapped = []
        for state in getattr(rt, '_graphs', {}).values():
            for seg in state['segments']:
                if not isinstance(seg['runlist'], _HybridRunlist):
                    w = _HybridRunlist(self, seg['runlist'])
                    seg['runlist'] = w
                    self._wrapped.append((seg, w))
        self._orig_xrt = rt.xrt
        rt.xrt = _XrtProxy(self, rt.xrt)
        self.installed = True
        self.active = True
        return self

    def uninstall(self):
        if not self.installed:
            return
        self.active = False
        rt = self.rt
        for seg, w in self._wrapped:
            if seg.get('runlist') is w:
                seg['runlist'] = w._real
        for state in getattr(rt, '_graphs', {}).values():
            for seg in state['segments']:
                if isinstance(seg['runlist'], _HybridRunlist):
                    seg['runlist']._hw = None
        rt.xrt = self._orig_xrt
        if self._had_execute_attr:
            rt.execute = self._orig_execute
        else:
            del rt.__dict__['execute']
        self.installed = False
        self.timer.close()

    # ---- hot path ----
    def _begin(self):
        spec = self.specs.get(self.current)
        if spec is None or self.current in self.disabled:
            return None
        try:
            spec.prepare()
        except Exception:
            self.stats['prepare_errors'] += 1
            self.stats['fallbacks'] += 1
            return None
        h = self.hist[self.current]
        return dict(graph=self.current, spec=spec, record=len(h) < WARMUP_CALLS)

    def _record(self, p, t_ns):
        self.hist[p['graph']].append(t_ns - p['t1'])
        self.stats['warmup_waits'] += 1

    def _hybrid(self, p):
        try:
            if p['record']:
                return
            g = p['graph']
            h = self.hist[g]
            if not h:                                    # H1: nothing to estimate from (cannot happen after the warm-up; never min([]))
                self.stats['fallbacks'] += 1
                return
            est = min(h)
            t1 = p['t1']
            now = time.perf_counter_ns()
            target = t1 + est - self.delta_ns
            if target > now:
                self.timer.sleep_ns(target - now)
                self.stats['sleeps'] += 1
            deadline = t1 + int(TIMEOUT_FACTOR * est + TIMEOUT_SLACK_MS * 1e6)
            spec = p['spec']
            polls = 0
            while True:
                polls += 1
                if spec.done():
                    break
                if time.perf_counter_ns() > deadline:
                    self.stats['timeouts'] += 1
                    self.stats['fallbacks'] += 1
                    self.stats['polls'] += polls
                    self._timeout(g)
                    return
            t3 = time.perf_counter_ns()
            h.append(t3 - t1)
            self.consec[g] = 0
            self.stats['hybrid_waits'] += 1
            self.stats['polls'] += polls
        except Exception:
            self.stats['errors'] += 1
            self.stats['fallbacks'] += 1

    def _timeout(self, g):
        """H1 circuit breaker: BREAKER_THRESHOLD consecutive timeouts of one graph disable the hint for that graph (ordinary wait only). Each timeout costs at most
        3 x estimate + 5 ms, so the added latency per graph is bounded by BREAKER_THRESHOLD x that. Logged once; exposed in report()."""
        self.consec[g] += 1
        if self.consec[g] >= self.breaker_threshold and g not in self.disabled:
            self.disabled[g] = 'consecutive timeouts: %d' % self.consec[g]
            self.stats['breaker_trips'] += 1
            if self.log is not None:
                try:
                    self.log('PLaMo: hybrid wait disabled for %s after %d consecutive timeouts' % (g, self.consec[g]))
                except Exception:
                    pass

    def report(self):
        r = dict(self.stats)
        r['disabled_graphs'] = sorted(self.disabled)
        return r


def install(rt, drafter=None, delta_ms=DEFAULT_DELTA_MS, log=None):
    """Create and install; returns the HybridWait or None (disabled by PLAMO3_HYBRID_WAIT=0, or any construction error: the ordinary path is then untouched)."""
    if not enabled():
        return None
    try:
        hw = HybridWait(rt, drafter, delta_ms, log)
        hw.install()
        return hw
    except Exception as error:
        if log is not None:
            log('PLaMo: hybrid wait unavailable (' + type(error).__name__ + ': ' + str(error)[:200] + '); using the ordinary wait')
        return None
