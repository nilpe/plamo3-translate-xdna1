"""Cached translation entry point; W4 + exact M2 is the default (release build, H1).

User-facing contract (documented in USAGE.md):
* No traceback ever reaches the user (PLAMO3_DEBUG=1 shows it). Every failure is one line `PLaMo: error: ...` and an exit code from plamo_errors.
* Arguments and the request text are validated BEFORE the runtime is constructed.
* One PLaMo process at a time owns the NPU (plamo_lock); a second one gets exit 5 with a one-line message.
* A failed speculative initialization is refused (exit 3) unless --allow-serial-fallback; with it the run continues in W4 serial with a warning block.
* The stats line goes to stderr only on a terminal, or with --stats / PLAMO3_STATS=1 (never breaks `2>&1` callers); --quiet / PLAMO3_STATS=0 suppress it.
* --service answers one JSON object per input line and never dies on a bad request.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from settings83 import DEFAULT_MODE, DEFAULT_KEEP_WARM, paths
paths()
import plamo_errors as errors
import turbo83

_LOCK = None  # the NPU lock is held for the whole process lifetime (released by the OS at exit)
SERVICE_MAX_LINE_BYTES = 8 * 1024 * 1024
LANGUAGES = ('ja', 'en')
MAX_NEW_TOKENS = errors.CONTEXT_TOKENS - 1


def real_process_entry():
    """R83: True only when the CLI runs as the real process entry (python cli83.py / translate.py -> cli45 shim), i.e. when nothing else lives in this process after main() returns. False when any
    runpy frame is on the stack (gate/bridge harnesses run scripts in-process via runpy.run_path(..., run_name='__main__')) or when not on the main thread."""
    import threading
    if threading.current_thread() is not threading.main_thread():return False
    f=sys._getframe(1)
    while f is not None:
        if f.f_globals.get('__name__')=='runpy' or f.f_code.co_filename.endswith('runpy.py'):return False
        f=f.f_back
    return True


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        if message.startswith('unrecognized arguments') and any(x.startswith('-') for x in message.split()[2:]):
            message += " (text that starts with '-' must follow '--')"
        self.exit(errors.USAGE, 'PLaMo: error: %s\n' % message)


def build_parser():
    p = _Parser(prog='translate_plamo3', description='PLaMo 3 translation on the NPU (exact W4 M2 by default)', add_help=False, allow_abbrev=False)
    p.add_argument('text', nargs='?')
    p.add_argument('--text-stdin', action='store_true', help='read the text from stdin (UTF-8; one trailing newline is dropped). Not limited by the 32 KB Windows command-line length')
    p.add_argument('--text-b64', default=None, metavar='BASE64', help='the text as base64 of UTF-8 (used by the PowerShell wrapper: quoting-proof)')
    p.add_argument('--source-lang', choices=LANGUAGES, default='ja')
    p.add_argument('--target-lang', choices=LANGUAGES, default='en')
    p.add_argument('--mode', choices=('w4-m2', 'w4-serial', 'w8-serial'), default=None, help='default: %s' % DEFAULT_MODE)
    p.add_argument('--spec-off', action='store_true', help='select W4 serial unless W8 serial was explicitly selected')
    p.add_argument('--num-tokens', type=int, default=256, help='maximum number of output tokens (1..%d)' % MAX_NEW_TOKENS)
    p.add_argument('--json', action='store_true')
    p.add_argument('--service', action='store_true', help='read JSON requests (one per line) from stdin in one long-lived process')
    p.add_argument('--turbo', action='store_true', help='opt-in: set the NPU power mode to turbo for this run and restore the previous mode afterwards (system-wide; also PLAMO3_NPU_TURBO=1; see TURBO.md)')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--keep-warm', dest='keep_warm', action='store_const', const=True, default=None, help='keep the decode CPU core warm with an idle-priority helper process during generation (costs one logical CPU at 100%%; also PLAMO3_KEEP_WARM=1; default policy: DEFAULT_KEEP_WARM, on AC power only)')
    g.add_argument('--no-keep-warm', dest='keep_warm', action='store_const', const=False, help='never run the keep-warm helper (also PLAMO3_KEEP_WARM=0)')
    p.add_argument('--allow-serial-fallback', action='store_true', help='if speculative decoding cannot start, continue in W4 serial with a warning instead of refusing (also PLAMO3_ALLOW_FALLBACK=1)')
    p.add_argument('--strict', action='store_true', help='exit 4 when the run completed in a degraded way (a fallback happened)')
    p.add_argument('--stats', action='store_true', help='print the timing line on stderr even when stderr is not a terminal (also PLAMO3_STATS=1)')
    p.add_argument('--quiet', action='store_true', help='never print the timing line (also PLAMO3_STATS=0)')
    p.add_argument('--wait', type=float, default=None, metavar='SECONDS', help='how long to wait for another PLaMo process to release the NPU (default 10)')
    p.add_argument('--no-lock', action='store_true', help='do not take the NPU lock (escape hatch; two processes on the NPU can fail)')
    p.add_argument('--version', action='store_true', help='print the runtime version and exit')
    p.add_argument('--env-report', action='store_true', help='print the environment record (versions of the Windows/driver/Python components) and exit')
    p.add_argument('--diag', action='store_true', help='print diagnostics for a bug report (no user text) and exit')
    p.add_argument('--help', '-h', action='store_true', dest='help')
    return p


def release_info():
    """release.json (written by the exporter), or None in a development tree."""
    try:
        import plamo_root
        return json.loads((plamo_root.ROOT / 'release.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, ImportError):
        return None


def version_text():
    info = release_info() or {}
    return 'PLaMo 3 Translate NPU runtime %s (Built with PLaMo; unofficial; see LICENSE-PLaMo-en.md)' % info.get('release', 'development')


def validate_text(text, parser):
    if not isinstance(text, str):
        parser.error('text must be a string')
    if not text.strip():
        parser.error('text is empty')
    if '<|plamo:' in text:
        parser.error('text must not contain reserved PLaMo tokens (<|plamo:...)')
    if len(text) > errors.MAX_TEXT_CHARS:
        parser.error('text is too long (%d characters; the limit is %d). Split it into paragraphs.' % (len(text), errors.MAX_TEXT_CHARS))


def validate_args(a, p):
    """Everything that can be checked without a runtime. Exits 2 through p.error()."""
    if a.text_stdin:
        if a.text is not None or a.text_b64 is not None or a.service:
            p.error('--text-stdin cannot be combined with a text argument, --text-b64 or --service')
        data = sys.stdin.buffer.read(errors.MAX_TEXT_CHARS * 4 + 16)
        if len(data) > errors.MAX_TEXT_CHARS * 4:
            p.error('text is too long (the limit is %d characters). Split it into paragraphs.' % errors.MAX_TEXT_CHARS)
        try:
            a.text = data.decode('utf-8')
        except UnicodeDecodeError:
            p.error('the text on stdin is not valid UTF-8')
        for tail in ('\r\n', '\n'):
            if a.text.endswith(tail):
                a.text = a.text[:-len(tail)]
                break
    if a.text_b64 is not None:
        if a.text is not None:
            p.error('text and --text-b64 are mutually exclusive')
        import base64, binascii
        try:
            a.text = base64.b64decode(a.text_b64, validate=True).decode('utf-8')
        except (binascii.Error, ValueError):
            p.error('--text-b64 is not valid base64 of UTF-8')
    if a.service:
        if a.text is not None:
            p.error('text must not be given together with --service (requests are read from stdin)')
    else:
        if a.text is None:
            p.error('text is required unless --service is used')
        validate_text(a.text, p)
    if a.source_lang == a.target_lang:
        p.error('--source-lang and --target-lang must differ (translation into the same language is not supported)')
    if not 1 <= a.num_tokens <= MAX_NEW_TOKENS:
        p.error('--num-tokens must be between 1 and %d' % MAX_NEW_TOKENS)
    if a.mode is not None and a.spec_off and a.mode == 'w4-m2':
        p.error('--spec-off conflicts with --mode w4-m2')
    if a.wait is not None and a.wait < 0:
        p.error('--wait must not be negative')
    if a.stats and a.quiet:
        p.error('--stats conflicts with --quiet')


def stats_wanted(a, environ=None):
    environ = os.environ if environ is None else environ
    if a.quiet or environ.get('PLAMO3_STATS') == '0':
        return False
    if a.stats or environ.get('PLAMO3_STATS') == '1':
        return True
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def allow_fallback_wanted(a, environ=None):
    environ = os.environ if environ is None else environ
    return bool(a.allow_serial_fallback or environ.get('PLAMO3_ALLOW_FALLBACK') == '1')


def finish_text(tokenizer, tokens, capped):
    """Decode `tokens`. When the output was cut by the token cap, a trailing incomplete UTF-8 sequence (byte-fallback tokens) is held back instead of being printed as U+FFFD.
    The token list itself is never changed. Returns (text, held_back_bytes)."""
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


def printable(text):
    """Text for stdout. The text layer writes the platform newline for every \\n, so a CRLF in the model output (printed as is) would become CR CR LF (BUG 8): normalise CRLF to LF first."""
    return text.replace('\r\n', '\n')


def check_language(value, name):
    if not isinstance(value, str) or value not in LANGUAGES:
        raise errors.InvalidRequest('%s must be "ja" or "en"' % name)
    return value


def stop_reason(got, max_seq):
    if got.get('stopped_eos'):
        return 'eos'
    return 'context' if got.get('logical_length', 0) >= max_seq else 'cap'


def make_translate(app, tokenizer, a):
    def translate(request):
        if not isinstance(request, dict):
            raise errors.InvalidRequest('a request must be a JSON object')
        text = request.get('text')
        if not isinstance(text, str):
            raise errors.InvalidRequest('text must be a string')
        if not text.strip():
            raise errors.InvalidRequest('text is empty')
        if '<|plamo:' in text:
            raise errors.InvalidRequest('text must not contain reserved PLaMo tokens (<|plamo:...)')
        if len(text) > errors.MAX_TEXT_CHARS:
            raise errors.InvalidRequest('text is too long (%d characters; the limit is %d)' % (len(text), errors.MAX_TEXT_CHARS))
        source = check_language(request.get('source_lang', a.source_lang), 'source_lang')
        target = check_language(request.get('target_lang', a.target_lang), 'target_lang')
        if source == target:
            raise errors.InvalidRequest('source_lang and target_lang must differ')
        limit = request.get('max_new_tokens', a.num_tokens)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_NEW_TOKENS:
            raise errors.InvalidRequest('max_new_tokens must be an integer between 1 and %d' % MAX_NEW_TOKENS)
        rid = request.get('id')
        if rid is not None and (isinstance(rid, bool) or not isinstance(rid, (str, int))):
            rid = None
        prompt = tokenizer.chat([dict(role='user', content=text)], source_lang=source, target_lang=target)
        if len(prompt) > app.session.max_seq:
            raise errors.PromptTooLong(len(prompt), app.session.max_seq)
        got = app.generate(prompt, limit, target_language=target)
        text_out, held = finish_text(tokenizer, got['tokens'], not got.get('stopped_eos'))
        result = dict(id=rid, mode=app.mode, text=text_out, tokens=got['tokens'], prompt_tokens=len(prompt), ttft_ms=got['ttft_s'] * 1000, wall_ms=got['wall_s'] * 1000,
                      spec_ready_ms=app.spec_ready_ms, stop=stop_reason(got, app.session.max_seq), requested_mode=app.requested_mode, fallbacks=[dict(f) for f in app.fallbacks])
        if held:
            result['held_back_bytes'] = held
        return result
    return translate


def error_object(rid, error, kind):
    return dict(id=rid, error=error.message if isinstance(error, errors.PlamoError) else str(error), kind=kind)


def serve(app, translate, out=None, source=None):
    """JSON-lines service loop. One response object per non-blank input line; a bad request yields an error object and the loop continues. Returns the exit code."""
    out = out or sys.stdout
    source = source if source is not None else sys.stdin.buffer
    first = True

    def emit(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + '\n')
        out.flush()
    while True:
        try:
            raw = source.readline(SERVICE_MAX_LINE_BYTES + 1)
        except KeyboardInterrupt:
            return errors.INTERRUPTED
        if not raw:
            return errors.OK
        if first:
            first = False
            if raw.startswith(b'\xef\xbb\xbf'):
                raw = raw[3:]
        if len(raw) > SERVICE_MAX_LINE_BYTES:
            while raw and not raw.endswith(b'\n'):  # discard the rest of the oversized line
                raw = source.readline(SERVICE_MAX_LINE_BYTES)
            emit(dict(id=None, error='request line is too long (the limit is %d bytes)' % SERVICE_MAX_LINE_BYTES, kind='invalid'))
            continue
        if not raw.strip():
            continue
        rid = None
        try:
            try:
                line = raw.decode('utf-8')
            except UnicodeDecodeError:
                raise errors.InvalidRequest('the request line is not valid UTF-8')
            try:
                request = json.loads(line)
            except RecursionError:
                raise errors.InvalidRequest('the request is nested too deeply')
            except ValueError as error:
                raise errors.InvalidRequest('the request is not valid JSON (%s)' % error)
            if isinstance(request, dict):
                value = request.get('id')
                rid = value if (isinstance(value, (str, int)) and not isinstance(value, bool)) else None
            emit(translate(request))
        except KeyboardInterrupt:
            return errors.INTERRUPTED
        except BrokenPipeError:
            return errors.OK
        except errors.InvalidRequest as error:
            emit(error_object(rid, error, 'invalid'))
        except (ValueError, TypeError, KeyError) as error:  # tokenizer / runtime validation of this one request
            emit(error_object(rid, errors.InvalidRequest(' '.join(str(error).split())[:200] or 'invalid request'), 'invalid'))
        except Exception as error:
            classified = errors.classify(error)
            if classified.code in (errors.NPU, errors.INSTALL, errors.ENVIRONMENT):
                emit(error_object(rid, classified, 'fatal'))
                errors.report(error)
                return classified.code
            if errors.debug_enabled():
                errors.report(error)
            emit(error_object(rid, classified, 'internal'))


def run(a, p, process_entry, turbo_on):
    import plamo_lock, plamo_root
    if not str(plamo_root.ROOT).isascii():
        # XRT (pyxrt.xclbin) opens files through the ANSI API: a folder name with non-ASCII characters makes every xclbin load fail with an undecodable error.
        raise errors.InstallProblem('the installation folder contains non-ASCII characters; the NPU library (XRT) cannot open files there. Reinstall into an ASCII-only folder, e.g. C:\\PLaMo3-NPU.')
    global _LOCK
    if not a.no_lock:
        _LOCK = plamo_lock.acquire('service' if a.service else (a.mode or DEFAULT_MODE), wait_s=10.0 if a.wait is None else a.wait, argv0='translate')
    with turbo83.session(turbo_on) as turbo_session:
        from default83 import DefaultRuntime
        from fast_tokenizer import FastPlamo3Tokenizer  # noqa: F401 (import order of the R83 CLI is kept)
        if turbo_on and a.mode != 'w4-m2':turbo_session.wait_switched()  # serial modes have no barrier inside the runtime
        import keepwarm83
        keep_warm=keepwarm83.resolve_mode(a.keep_warm,os.environ,DEFAULT_KEEP_WARM)  # CLI > env PLAMO3_KEEP_WARM > default; invalid env value -> off + one stderr line
        allow = allow_fallback_wanted(a)
        with DefaultRuntime(a.mode, keep_warm=keep_warm, allow_fallback=allow) as app:
            if app.fallbacks:  # only reachable with --allow-serial-fallback (or the harness default)
                reason = app.fallbacks[0]['reason']
                print('PLaMo WARNING: running in W4 serial (slower; results are identical).', file=sys.stderr)
                print('PLaMo WARNING: reason: ' + reason, file=sys.stderr)
                print('PLaMo WARNING: run install.ps1 -Verify to check the installation.', file=sys.stderr, flush=True)
            import environment
            environment.check()  # once, after pyxrt is loaded: records versions, warns about untested driver/Windows/Python (never refuses)
            tokenizer=app.make_tokenizer()  # R83: same pinned tokenizer, cheaper construction (tokfast83); R77_TOK=off -> FastPlamo3Tokenizer(manifest)
            translate = make_translate(app, tokenizer, a)
            if a.service:
                print(json.dumps(dict(service_ready=True,mode=app.mode,spec_ready_ms=app.spec_ready_ms)),flush=True)
                rc = serve(app, translate)
                if rc == errors.OK and a.strict and app.fallbacks:
                    rc = errors.DEGRADED
                return rc
            try:
                got=translate(dict(text=a.text))
            except errors.PlamoError:
                raise
            except (ValueError,TypeError,KeyError) as error:
                raise errors.InvalidRequest(' '.join(str(error).split())[:300] or 'invalid request') from error
            if a.json:
                if turbo_on:got['turbo']={k:round(v,1)for k,v in turbo_session.stats.items()}
                if app.keepwarm is not None:got['keep_warm']={k:v for k,v in app.keepwarm.stats().items()if k!='helper'}
                if getattr(app, 'hybrid', None) is not None:got['hybrid']=app.hybrid.report()
                if os.environ.get('R77_REPORT_STARTUP')=='1':got['startup_events']=[dict(phase=e['phase'],ms=e['ms'])for e in app.startup.get('events',[])]  # soak gate report only; inert unless set
                print(json.dumps(got,ensure_ascii=False),flush=True)
            else:
                print(printable(got['text']),flush=True)
                if stats_wanted(a):
                    info={k:got[k]for k in('mode','prompt_tokens','ttft_ms','wall_ms','spec_ready_ms')}
                    if turbo_on:info['turbo']={k:round(v,1)for k,v in turbo_session.stats.items()}
                    print('PLaMo: '+json.dumps(info),file=sys.stderr,flush=True)
            rc = errors.DEGRADED if (a.strict and app.fallbacks) else errors.OK
            if process_entry and app.fast_exit_allowed(turbo_on=turbo_on,service=a.service):app.fast_exit(rc)  # R83: result printed and flushed; skip the in-process teardown (returns if unusual -> normal close below)
            return rc


def main(process_entry=False, argv=None):
    p = build_parser()
    try:
        a = p.parse_args(argv)
        if a.help:
            p.print_help()
            return errors.OK
        if a.version:
            print(version_text())
            return errors.OK
        if a.env_report:
            import environment
            print(environment.report_text())
            return errors.OK
        if a.diag:
            return diag()
        validate_args(a, p)
        import plamo_root
        plamo_root.check_platform(errors.EnvironmentProblem)  # strict ABI facts of our compiled files: CPython 3.13, 64-bit (exit 6 with the found version)
        if a.mode is None:
            a.mode = 'w4-serial' if a.spec_off else DEFAULT_MODE
        for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'4')
        turbo_on=a.turbo or turbo83.enabled_by_env()
        if not turbo_on:turbo83.stale_check()  # one os.path.exists when no marker: no xrt-smi call; restores a dead turbo session's recorded mode otherwise
        # R83: with the switch on, the acquire sequence (stale marker handling included) runs in a background thread started right here, so its xrt-smi calls overlap the imports and
        # the runtime init; the runtime waits for it (startup-event barrier before device_open) so turbo is in effect when the hw_contexts are created.
        return run(a, p, process_entry, turbo_on)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        return errors.INTERRUPTED
    except BrokenPipeError:
        return errors.OK
    except BaseException as error:
        return errors.report(error)


def diag():
    """Diagnostics for a bug report: release record, environment record, receipt status. No user text."""
    import environment
    import plamo_root
    print('version: ' + version_text())
    print('install root: ' + str(plamo_root.ROOT))
    print('release.json: ' + json.dumps(release_info()))
    index, reason = plamo_root._load_receipt()
    print('install receipt: ' + ('ok (%d files)' % len(index) if index is not None else reason))
    print(environment.report_text())
    return errors.OK


if __name__=='__main__':sys.exit(main(process_entry=real_process_entry()))
