# SPDX-License-Identifier: Apache-2.0
r"""PLaMo 3 long mode CLI: translate texts that do not fit the normal 2,048-token path (up to 262,144 tokens in total).

  python long_main.py [--text-stdin | TEXT] --source-lang ja --target-lang en [--num-tokens N] [--json] [--quiet] [--wait S] [--no-lock]
  python long_main.py --route-only ...    print the mode verdict ('default' or 'long') and the token counts, then exit (used by the wrapper's router)

Pipeline: tokenize (sealed fast tokenizer, same chat template as the default) -> router rule -> NPU lock -> iGPU prefill child process (progress, Ctrl+C) ->
KV handoff under <root>\state\long\<id> -> NPU decode (serial W4) -> text. Exit codes are the default path's table (USAGE.md): 2 usage, 3 installation, 5 NPU busy,
6 environment (no iGPU runtime, too little memory ...), 130 Ctrl+C. No traceback ever reaches the user (PLAMO3_DEBUG=1 shows it).
"""
import argparse
import base64
import binascii
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from long_core import (CHUNK, RepeatDetector, Cancelled, LongModeError, LongRequestError, MAX_TOTAL, Progress, check_handoff, check_ram, decide, decode_estimate_s, errors,
                       finish_text, fmt_s, install_root, length_warning, long_cap, on_battery, prefill_estimate_s, prefill_eta_s, prompt_digest, sealed_paths,
                       stale_handoff_dirs, stop_reason, write_handoff_record)

DELEGATE_EXIT = 90
MAX_LONG_TEXT_CHARS = 3_000_000
LANGUAGES = ('ja', 'en')


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(errors.USAGE, 'PLaMo: error: %s\n' % message)


def build_parser():
    p = _Parser(prog='translate_long', add_help=False, allow_abbrev=False, description='PLaMo 3 long-document translation (iGPU prefill + NPU decode)')
    p.add_argument('text', nargs='?')
    p.add_argument('--text-stdin', action='store_true')
    p.add_argument('--text-b64', default=None)
    p.add_argument('--source-lang', choices=LANGUAGES, default='ja')
    p.add_argument('--target-lang', choices=LANGUAGES, default='en')
    p.add_argument('--num-tokens', type=int, default=None, help='maximum number of output tokens (default: min(remaining, max(1024, 2 x prompt)))')
    p.add_argument('--json', action='store_true')
    p.add_argument('--quiet', action='store_true', help='no progress lines')
    p.add_argument('--stats', action='store_true')
    p.add_argument('--delegate-if-default', action='store_true', help='exit %d without doing anything when the request belongs to the normal path (used by translate_long.ps1)' % DELEGATE_EXIT)
    p.add_argument('--route-only', action='store_true', help='print the mode verdict and exit')
    p.add_argument('--long', action='store_true', help='force long mode (used by the wrapper for -Long)')
    p.add_argument('--wait', type=float, default=10.0)
    p.add_argument('--no-lock', action='store_true')
    p.add_argument('--keep-handoff', action='store_true', help='keep the iGPU result directory (gates)')
    p.add_argument('--prefill-only', action='store_true', help='run the iGPU prefill, keep its result, print the directory (gates)')
    p.add_argument('--no-repeat-stop', action='store_true', help='keep decoding when the output enters a repeat loop (default: stop at the end of the first occurrence of the repeated period)')
    p.add_argument('--no-split', action='store_true', help='disable dispatch splitting (L7 comparison)')
    p.add_argument('--dispatch-cap-ms', type=float, default=200.0)
    p.add_argument('--state-dir', default=None)
    p.add_argument('--help', '-h', action='store_true', dest='help')
    return p


def read_text(a, p, stdin=None):
    stdin = stdin if stdin is not None else sys.stdin.buffer
    if a.text_stdin:
        if a.text is not None or a.text_b64 is not None:
            p.error('--text-stdin cannot be combined with a text argument or --text-b64')
        data = stdin.read(MAX_LONG_TEXT_CHARS * 4 + 16)
        if len(data) > MAX_LONG_TEXT_CHARS * 4:
            p.error('text is too long (the limit is %d characters)' % MAX_LONG_TEXT_CHARS)
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            p.error('the text on stdin is not valid UTF-8')
        for tail in ('\r\n', '\n'):
            if text.endswith(tail):
                text = text[:-len(tail)]
                break
    elif a.text_b64 is not None:
        if a.text is not None:
            p.error('text and --text-b64 are mutually exclusive')
        try:
            text = base64.b64decode(a.text_b64, validate=True).decode('utf-8')
        except (binascii.Error, ValueError):
            p.error('--text-b64 is not valid base64 of UTF-8')
    else:
        text = a.text
        if text is None:
            p.error('text is required')
    if not text.strip():
        p.error('text is empty')
    if '<|plamo:' in text:
        p.error('text must not contain reserved PLaMo tokens (<|plamo:...)')
    if len(text) > MAX_LONG_TEXT_CHARS:
        p.error('text is too long (%d characters; the limit is %d)' % (len(text), MAX_LONG_TEXT_CHARS))
    if a.source_lang == a.target_lang:
        p.error('--source-lang and --target-lang must differ')
    if a.num_tokens is not None and a.num_tokens < 1:
        p.error('--num-tokens must be positive')
    return text


def make_tokenizer(root):
    plamo = sealed_paths(root)
    if str(plamo) not in sys.path:
        sys.path.insert(0, str(plamo))
    try:
        import fast_tokenizer
        manifest = Path(root) / 'plamo3-runtime' / 'r34-w4' / 'tokenizer'
        return fast_tokenizer.FastPlamo3Tokenizer(manifest, checkpoint=Path(root) / 'model', validate_sources=True)
    except errors.PlamoError:
        raise
    except Exception as error:
        raise errors.InstallProblem('the tokenizer could not be loaded (%s: %s). Run install.ps1 -Verify.' % (type(error).__name__, ' '.join(str(error).split())[:120]))


def model_weights_dir(root):
    """Directory of the BF16 base-model weights: <root>\\model (fixed; the file is hash-pinned in long-manifest.json and attested before use). The v1.0.0 install keeps only the 5 tokenizer files
    there; install_long.py --with-model adds model.safetensors (1.78 GB, official upstream file, pinned revision and sha256)."""
    d = Path(root) / 'model'
    if not (d / 'model.safetensors').is_file():
        raise LongModeError('long mode needs the base model weights (model.safetensors, 1.78 GB), which are not installed. Run: python install_long.py --with-model (the normal translation does not need them).')
    return d


# ---------------------------------------------------------------- iGPU child
def run_prefill_child(job_path, argv_prefix, progress, state, cancel_file, *, popen=subprocess.Popen, poll_s=0.2):
    """Runs the prefill child, relays its JSON events as progress lines, returns the 'done' event. Ctrl+C in this process creates the cancel file and waits for the child.
    Raises the child's error as a PlamoError with its exit code."""
    proc = popen(list(argv_prefix) + [str(job_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
    done, last_error = None, None
    samples = []
    t_start = time.monotonic()
    try:
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get('ev')
            if kind == 'device':
                state['device'] = ev
            elif kind == 'chunk':
                samples.append(((ev['pos'] - min(CHUNK, ev['pos'])), ev['ms'] / 1000.0))
                eta = prefill_eta_s(samples, ev['pos'], ev['n'])
                progress.line('prefill %d / %d tokens (%d%%), elapsed %s, about %s left' % (ev['pos'], ev['n'], 100 * ev['pos'] // ev['n'], fmt_s(time.monotonic() - t_start), fmt_s(eta)))
            elif kind == 'done':
                done = ev
            elif kind == 'error':
                last_error = ev
            elif kind == 'cancelled':
                last_error = dict(code=errors.INTERRUPTED, message='cancelled')
        proc.wait()
    except KeyboardInterrupt:
        Path(cancel_file).write_text('1')
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        raise Cancelled('cancelled')
    if done is not None and proc.returncode == 0:
        return done
    if last_error:
        if last_error.get('code') == errors.INTERRUPTED:
            raise Cancelled('cancelled')
        raise errors.PlamoError(last_error.get('message', 'the iGPU prefill failed'), code=int(last_error.get('code', errors.INTERNAL)))
    raise LongModeError('the iGPU prefill step stopped unexpectedly (exit code %s). Update the AMD Radeon driver and try again; the normal translation path is not affected.' % proc.returncode)


# ---------------------------------------------------------------- main flow
def translate(a, text, root, *, stderr=None, tokenizer=None, decoder_factory=None, child_runner=run_prefill_child, attest_fn=None):
    stderr = stderr or sys.stderr
    progress = Progress(stderr, enabled=not a.quiet)
    t_start = time.perf_counter()
    tok = tokenizer or make_tokenizer(root)
    prompt = tok.chat([dict(role='user', content=text)], source_lang=a.source_lang, target_lang=a.target_lang)
    P = len(prompt)
    verdict = decide(P, a.num_tokens, a.long)
    if a.route_only:
        return dict(verdict=verdict, prompt_tokens=P, cap=long_cap(P, a.num_tokens) if verdict == 'long' else None)
    if a.delegate_if_default and verdict == 'default':
        return dict(verdict='default', delegate=True, prompt_tokens=P)
    cap = long_cap(P, a.num_tokens)
    warn = length_warning(P)
    if warn:
        print(warn, file=stderr, flush=True)
    state_root = Path(a.state_dir) if a.state_dir else Path(root) / 'state' / 'long'
    state_root.mkdir(parents=True, exist_ok=True)
    for old in stale_handoff_dirs(state_root):
        shutil.rmtree(old, ignore_errors=True)
    if not str(root).isascii() and not a.prefill_only:
        # XRT (pyxrt.xclbin) opens files through the ANSI API: a folder name with non-ASCII characters makes the NPU step fail after minutes of prefill, so refuse up front (same rule and wording as the default CLI).
        raise errors.InstallProblem('the installation folder contains non-ASCII characters; the NPU library (XRT) cannot open files there. Reinstall into an ASCII-only folder, e.g. C:\\PLaMo3-NPU.')
    check_ram()
    if attest_fn is None:
        import long_attest
        attest_fn = long_attest.attest
    retained, _manifest = attest_fn(root)           # raises AttestationError (exit 3) / LongModeError (weights missing); handles stay open until the end of the run
    if on_battery():
        print('PLaMo: note: running on battery; a long prefill keeps the GPU busy for minutes. Plug in the charger for the best speed.', file=stderr, flush=True)
    pre_est = prefill_estimate_s(P)
    dec_est = decode_estimate_s(P, cap)
    progress.line('long mode: %d prompt tokens, up to %d new tokens. Estimated prefill %s (iGPU), decode up to %s (NPU, serial). Ctrl+C cancels.' % (P, cap, fmt_s(pre_est), fmt_s(dec_est)), force=True)
    work = state_root / ('%d-%d' % (int(time.time()), os.getpid()))
    work.mkdir()
    keep = a.keep_handoff or a.prefill_only
    result = {}
    try:
        import numpy as np
        np.save(work / 'tokens.npy', np.array(prompt, dtype=np.int64))
        digest = prompt_digest(prompt)
        model_dir = model_weights_dir(root)
        job = dict(tokens_npy=str(work / 'tokens.npy'), out=str(work / 'kv-out'), model_dir=str(model_dir), co_dir=str(Path(root) / 'plamo3-long' / 'prefill'),
                   cancel_file=str(work / 'cancel'), split=not a.no_split, cap_ms=a.dispatch_cap_ms, ktime=os.environ.get('H2L_KTIME') or None)
        (work / 'job.json').write_text(json.dumps(job))
        child = [sys.executable, '-I', str(HERE / 'long_engine.py')]
        state = {}
        t_pre = time.perf_counter()
        done = child_runner(work / 'job.json', child, progress, state, work / 'cancel')
        prefill_s = time.perf_counter() - t_pre
        write_handoff_record(work / 'kv-out', P, digest)
        check_handoff(work / 'kv-out', P, digest)
        progress.line('prefill done in %s (%s); first token ready' % (fmt_s(prefill_s), (state.get('device') or {}).get('name', 'iGPU')), force=True)
        if a.prefill_only:
            result = dict(prefill_dir=str(work / 'kv-out'), prompt_tokens=P, prefill_s=prefill_s, engine=done)
            return result
        first = int(done['argmax_last'])
        dec = (decoder_factory or (lambda: __import__('long_decode').LongDecoder(root)))()
        t_dec = time.perf_counter()
        dec.open()
        try:
            cancelled = lambda: (work / 'cancel').exists()
            last_report = [time.monotonic()]

            def on_token(tokens, steps):
                now = time.monotonic()
                if now - last_report[0] >= 5.0 and not a.quiet:
                    last_report[0] = now
                    med = sorted(steps[-200:])[len(steps[-200:]) // 2]
                    progress.line('decoded %d / %d tokens (%.1f ms/token)' % (len(tokens), cap, med), force=True)
            detector = None if a.no_repeat_stop else RepeatDetector()
            out = dec.run(work / 'kv-out', P, first, cap, on_token=on_token, cancelled=cancelled, stop=(detector.feed if detector else None))
        except KeyboardInterrupt:
            raise Cancelled('cancelled')
        finally:
            dec.close()
        tokens = out['tokens']
        repeat = out.get('repeat')
        stop = stop_reason(tokens, cap, P, repeat)
        if repeat:
            print('PLaMo: stopped because the output started to repeat (a loop of %d tokens); the text ends before the repetition. This is a limit of the model; split long inputs, or use -NoRepeatStop to keep decoding.' % repeat['period'], file=stderr, flush=True)
        text_out, held = finish_text(tok, tokens, stop not in ('eos', 'repeat'))
        wall = time.perf_counter() - t_start
        steps = sorted(out['steps_ms'])
        result = dict(mode='long', text=text_out, tokens=tokens, prompt_tokens=P, new_tokens=len(tokens), stop=stop, ttft_ms=(prefill_s + 0.0) * 1000.0,
                      prefill_s=prefill_s, repeat=repeat, decode_ms_median=steps[len(steps) // 2] if steps else None, wall_ms=wall * 1000.0, npu_open_s=dec.open_s, fallbacks=[])
        if held:
            result['held_back_bytes'] = held
        return result
    finally:
        try:
            retained.close()
        except Exception:
            pass
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None, stdin=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = build_parser()
    a = p.parse_args(argv)
    if a.help:
        print(p.format_help())
        return errors.OK
    lock = None
    try:
        text = read_text(a, p, stdin)
        root = install_root()
        if not a.route_only and not a.no_lock:
            try:
                sys.path.insert(0, str(sealed_paths(root)))
                import plamo_lock
                lock = plamo_lock.acquire('long', wait_s=a.wait, argv0='translate_long')
            except ImportError:
                lock = None
        result = translate(a, text, root)
        if result.get('delegate'):
            return DELEGATE_EXIT
        if a.route_only or a.prefill_only or a.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
            print(result['text'].replace('\r\n', '\n'))
        if a.stats and not (a.route_only or a.prefill_only):
            print('PLaMo: long: prompt %d tokens, %d new tokens (%s), prefill %.1f s, decode %.1f ms/token, wall %.1f s' % (result['prompt_tokens'], result['new_tokens'], result['stop'],
                  result['prefill_s'], result['decode_ms_median'] or 0.0, result['wall_ms'] / 1000.0), file=sys.stderr)
        return errors.OK
    except (Cancelled, KeyboardInterrupt):
        print('PLaMo: cancelled.', file=sys.stderr)
        return errors.INTERRUPTED
    except errors.PlamoError as error:
        print('PLaMo: error: %s' % error.message, file=sys.stderr)
        return error.code
    except SystemExit:
        raise
    except Exception as error:
        if os.environ.get('PLAMO3_DEBUG') == '1':
            import traceback
            traceback.print_exc()
        classified = errors.classify(error) if hasattr(errors, 'classify') else errors.PlamoError('internal error: %s' % type(error).__name__)
        print('PLaMo: error: %s' % classified.message, file=sys.stderr)
        return classified.code
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())
