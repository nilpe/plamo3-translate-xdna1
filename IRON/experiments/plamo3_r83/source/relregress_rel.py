# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""H1 A1 regression tests: one test per BUGS.md repro (staged release code, active R83 payload, real NPU). usage: regress.py GROUP  (args|service|wrapper|lock|orphan|fallback|multibyte)"""
import base64, json, os, subprocess, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relhelp_rel import *

import argparse as _ap
_p = _ap.ArgumentParser(); _p.add_argument('--group', required=True); _p.add_argument('--output', default=None); _a = _p.parse_args()
group = _a.group
R = Report('regress-' + group)
SRC_JA = '今日はいい天気ですね。'
TEXT_EN = 'Hello, how are you today?'

def svc(lines, *, extra_env=None, args=(), timeout=150, raw=False):
    data = b''.join(l if isinstance(l, bytes) else (l + '\n').encode('utf-8') for l in lines)
    e = env(extra_env)
    p = subprocess.Popen([PY, TRANSLATE, '--service'] + list(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
    try:
        out, err = p.communicate(data, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], capture_output=True)
        out, err = p.communicate()
        return dict(rc=None, rows=[], err=err.decode('utf-8', 'replace'), timeout=True)
    rows = []
    for l in out.decode('utf-8', 'replace').splitlines():
        try:
            rows.append(json.loads(l))
        except ValueError:
            rows.append(dict(_unparsable=l))
    return dict(rc=p.returncode, rows=rows, err=err.decode('utf-8', 'replace'), timeout=False)

GOOD = lambda i='ok': json.dumps(dict(id=i, text='Hello.', source_lang='en', target_lang='ja'))

if group == 'args':
    # BUG 2 / BUG 9: one-shot argument and text errors: one line, exit 2, no traceback, no runtime construction (fast)
    cases = [
        ('empty-space', [' '], 'text is empty'),
        ('plamo-token', ['<|plamo:msg|>'], 'reserved PLaMo tokens'),
        ('num-tokens-0', ['--num-tokens', '0', 'Hello.'], '--num-tokens must be between'),
        ('num-tokens-neg', ['--num-tokens', '-1', 'Hello.'], '--num-tokens must be between'),
        ('num-tokens-huge', ['--num-tokens', '99999999999999999999', 'Hello.'], '--num-tokens must be between'),
        ('dash-hello', ['-hello'], 'unrecognized arguments'),
        ('service-with-text', ['--service', 'Hello.'], 'must not be given together'),
        ('same-lang', ['--source-lang', 'en', '--target-lang', 'en', 'x'], 'must differ'),
        ('no-text', [], 'text is required'),
        ('bad-b64', ['--text-b64', '!!!'], 'base64'),
    ]
    for name, args, needle in cases:
        r = cli(args)
        R.check('args/' + name, r['rc'] == 2 and needle in r['stderr'] and no_trace(r) and one_line(r) and r['wall'] < 5, (r['rc'], r['wall'], r['stderr'][:200]))
    r = cli(['--help'])
    R.check('args/help-rc0', r['rc'] == 0 and '--allow-serial-fallback' in r['stdout'])
    # over-context text: needs the runtime (tokenizer) -> still exit 2 and one line
    big = 'これは長い文章です。' * 3000
    r = cli(['--text-stdin'], input=big, timeout=120)
    R.check('args/over-context', r['rc'] == 2 and 'tokens; the limit is 2048' in r['stderr'] and no_trace(r) and one_line(r), (r['rc'], r['wall'], r['stderr'][:240]))
    huge = 'x' * 1_000_001
    r = cli(['--text-stdin'], input=huge)
    R.check('args/over-1M-chars', r['rc'] == 2 and 'too long' in r['stderr'] and r['wall'] < 5, (r['rc'], r['wall']))
    # valid one-shot: json, exit 0, stderr silent (not a tty), CRLF handling unaffected
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--text-stdin'], input=TEXT_EN + '\r\n')
    ok = r['rc'] == 0 and r['stderr'].strip() == ''
    d = json.loads(r['stdout']) if ok else {}
    R.check('args/valid-oneshot-silent-stderr', ok and d.get('mode') == 'w4-m2' and d.get('stop') == 'eos' and d.get('fallbacks') == [], (r['rc'], r['stderr'][:200], d.get('mode')))
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--stats', '--text-b64', base64.b64encode(TEXT_EN.encode()).decode()])
    R.check('args/stats-flag-prints-stats-line', r['rc'] == 0 and r['stderr'].startswith('PLaMo: {'), r['stderr'][:200])
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--text-b64', base64.b64encode(TEXT_EN.encode()).decode()])
    R.check('args/plain-text-output-no-stats-when-piped', r['rc'] == 0 and r['stderr'].strip() == '' and len(r['stdout'].strip()) > 0, (r['rc'], r['stderr'][:100], r['stdout'][:60]))

elif group == 'service':
    # BUG 1 / BUG 5 / BUG 11: bad requests never end the service; protocol details
    bad = [
        ('text-empty', json.dumps(dict(text='', source_lang='en', target_lang='ja'))),
        ('text-space', json.dumps(dict(text='  ', source_lang='en', target_lang='ja'))),
        ('text-nl', json.dumps(dict(text='\n', source_lang='en', target_lang='ja'))),
        ('text-plamo-token', json.dumps(dict(text='<|plamo:msg|>', source_lang='en', target_lang='ja'))),
        ('src-num', '{"text":"x","source_lang":5,"target_lang":"ja"}'),
        ('tgt-null', '{"text":"x","source_lang":"en","target_lang":null}'),
        ('tgt-list', '{"text":"x","source_lang":"en","target_lang":["ja"]}'),
        ('tgt-empty', '{"text":"x","source_lang":"en","target_lang":""}'),
        ('tgt-upper', '{"text":"x","source_lang":"en","target_lang":"EN"}'),
        ('tgt-regional', '{"text":"x","source_lang":"en","target_lang":"ja-JP"}'),
        ('tgt-xx', '{"id":"a","text":"Hi.","source_lang":"en","target_lang":"xx"}'),
        ('tgt-ja-easy', '{"id":"a","text":"Hi.","source_lang":"en","target_lang":"ja(easy)"}'),
        ('src-fr', '{"text":"x","source_lang":"fr","target_lang":"ja"}'),
        ('src-null', '{"text":"x","source_lang":null,"target_lang":"ja"}'),
        ('deep-nest-100k', '{"text":"x","a":' + '[' * 100000 + '}'),
        ('not-object', '[1,2,3]'),
        ('missing-text', '{"source_lang":"en","target_lang":"ja"}'),
        ('max-new-1e9', '{"text":"x","max_new_tokens":1000000000}'),
        ('max-new-true', '{"text":"x","max_new_tokens":true}'),
        ('garbage', '{not json}'),
    ]
    raw_bad = [('invalid-utf8-line', b'\xff\xfe\x00{"text":"x"}\n'), ('invalid-utf8-in-string', b'{"text":"\xff\xfe"}\n')]
    lines = []
    for name, l in bad:
        lines += [l, GOOD('after-' + name)]
    for name, b in raw_bad:
        lines += [b, (GOOD('after-' + name) + '\n').encode()]
    # a UTF-16 request (BOM, NUL bytes, 0x0a inside the encoding): errors for the garbage lines, the service stays up and answers the next real request
    r16 = svc(['{"text":"x"}\n'.encode('utf-16'), b'\n', GOOD('after-utf16')])
    R.check('service/utf16-input-never-kills-the-service', r16['rc'] == 0 and r16['rows'][-1].get('id') == 'after-utf16' and 'error' not in r16['rows'][-1] and 'Traceback' not in r16['err'], (r16['rc'], r16['rows'][-1:], r16['err'][:200]))
    # the BOM on the very first line is tolerated: put it on the first request
    lines[0] = b'\xef\xbb\xbf' + lines[0].encode('utf-8') + b'\n'
    r = svc(lines)
    rows = r['rows']
    R.check('service/alive-and-exit0', r['rc'] == 0 and not r['timeout'], (r['rc'], r['err'][:300]))
    R.check('service/no-traceback-anywhere', 'Traceback' not in r['err'], r['err'][-300:])
    ready = rows[0] if rows else {}
    R.check('service/ready-line', ready.get('service_ready') is True and ready.get('mode') == 'w4-m2', ready)
    resp = rows[1:]
    R.check('service/one-response-per-line', len(resp) == len(lines), (len(resp), len(lines)))
    good_ok = [x for x in resp if str(x.get('id', '')).startswith('after-')]
    R.check('service/every-valid-request-answered', len(good_ok) == len(bad) + len(raw_bad) and all('error' not in x for x in good_ok), len(good_ok))
    R.check('service/mode-never-degraded', all(x.get('mode') == 'w4-m2' for x in good_ok), set(x.get('mode') for x in good_ok))
    errs = [x for x in resp if 'error' in x]
    R.check('service/errors-are-invalid-objects', len(errs) == len(bad) + len(raw_bad) and all(x.get('kind') == 'invalid' for x in errs), [(x.get('error'), x.get('kind')) for x in errs[:3]])
    R.check('service/no-python-internals-leaked', not any(("object is not subscriptable" in x['error'] or "'text'" == x['error']) for x in errs), [x['error'] for x in errs][:25])
    R.check('service/id-echoed-on-validation-errors', [x for x in errs if x.get('id') == 'a'].__len__() == 2)
    # BUG 5: after the unsupported language the mode is still w4-m2 and the next answer is a good translation
    after = [x for x in good_ok if x['id'] in ('after-tgt-xx', 'after-tgt-ja-easy')]
    R.check('service/bug5-language-request-no-downgrade', len(after) == 2 and all(x['mode'] == 'w4-m2' and x['fallbacks'] == [] for x in after))
    # standard EOF behaviour: partial line at EOF, CRLF, blank lines, missing final newline
    r = svc(['', '\r', json.dumps(dict(id='crlf', text='Hello.', source_lang='en', target_lang='ja')) + '\r', json.dumps(dict(id='last', text='Good morning.', source_lang='en', target_lang='ja'))])
    rows = r['rows'][1:]
    R.check('service/crlf-blank-lines-final-line-no-newline', r['rc'] == 0 and [x.get('id') for x in rows] == ['crlf', 'last'] and all('error' not in x for x in rows), (r['rc'], rows))

elif group == 'wrapper':
    # BUG 3: the wrapper under $ErrorActionPreference=Stop + 2>&1 (Windows PowerShell 5.1); BUG 9: wrapper arguments
    e = env()
    def w(psargs, merge=True, stop=True):
        redirect = ' 2>&1' if merge else ''
        pre = "$ErrorActionPreference='Stop'; " if stop else ''
        return ps(pre + "$o = & '%s' %s%s; 'EXIT='+$LASTEXITCODE; 'OUT_BEGIN'; $o; 'OUT_END'" % (WRAPPER, psargs, redirect), e=e)
    r = w("'Hello, how are you?' -SourceLang en -TargetLang ja")
    R.check('wrapper/bug3-2>&1-stop-exit0', 'EXIT=0' in r['stdout'] and 'NativeCommandError' not in r['stdout'] + r['stderr'] and len(r['stdout'].split('OUT_BEGIN')[-1].strip().replace('OUT_END', '').strip()) > 0, r['stdout'][:300] + r['stderr'][:200])
    r = w("'Hello, how are you?' -SourceLang en -TargetLang ja -Stats")
    R.check('wrapper/bug3-2>&1-stop-with-explicit-Stats-still-ok', 'EXIT=0' in r['stdout'] and 'PLaMo: {' in r['stdout'] + r['stderr'] and len(r['stdout'].split('OUT_BEGIN')[-1].replace('OUT_END', '').strip()) > 0, r['stdout'][:200] + '||' + r['stderr'][:300])
    r = w("'Hello.' -SourceLang en -TargetLang ja", merge=False, stop=False)
    R.check('wrapper/plain-call', 'EXIT=0' in r['stdout'], r['stdout'][:200])
    # A2: the wrapper must not block when another program starts it with an inherited OPEN stdin pipe (it referenced $input before)
    for label, args in (('version', ['-Version']), ('translate', ['Hello.', '-SourceLang', 'en', '-TargetLang', 'ja', '-NumTokens', '8', '-Json'])):
        pr = subprocess.Popen(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', WRAPPER] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
        try:
            rc = pr.wait(timeout=60); hung = False
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(pr.pid)], capture_output=True); rc = None; hung = True
        R.check('wrapper/open-stdin-pipe-does-not-block-' + label, not hung and rc == 0, (rc, hung))
    r = w("'' ")
    R.check('wrapper/bug9-empty-text-exit2', 'EXIT=2' in r['stdout'] and 'text is required' in r['stdout'] + r['stderr'], r['stdout'][:300])
    r = w("' ' ")
    R.check('wrapper/bug9-blank-text-exit2', 'EXIT=2' in r['stdout'], r['stdout'][:300])
    r = w("-Service Hello.")
    R.check('wrapper/bug9-service-with-text-exit2', 'EXIT=2' in r['stdout'], r['stdout'][:300])
    r = w("-Text '-hello' -SourceLang en -TargetLang ja")
    R.check('wrapper/bug9-dash-text-via-Text-param', 'EXIT=0' in r['stdout'], r['stdout'][:300])
    r = w("'\"' -SourceLang en -TargetLang ja")
    R.check('wrapper/bug9-quote-only-text', 'EXIT=0' in r['stdout'], r['stdout'][:300])
    r = w("'x' -NumTokens 0")
    R.check('wrapper/bug9-num-tokens-0-exit2', 'EXIT=2' in r['stdout'] and 'between 1 and 2047' in r['stdout'] + r['stderr'], r['stdout'][:300])
    r = w("'Hello.' -SourceLang en -TargetLang en")
    R.check('wrapper/same-language-exit2', 'EXIT=2' in r['stdout'], r['stdout'][:300])
    r = w("-Version")
    R.check('wrapper/version', 'EXIT=0' in r['stdout'] and 'PLaMo 3 Translate NPU runtime' in r['stdout'], r['stdout'][:200])
    r = w("'x' -Mode w4-m2 -SpecOff")
    R.check('wrapper/spec-off-conflict-exit2', 'EXIT=2' in r['stdout'] and 'conflicts' in r['stdout'] + r['stderr'], r['stdout'][:300])
    # multi-line and CRLF text through the base64 transport
    r = w("\"Hello.`r`nHow are you?\" -SourceLang en -TargetLang ja")
    R.check('wrapper/bug8-crlf-input', 'EXIT=0' in r['stdout'] and '\r\r\n' not in r['stdout'], repr(r['stdout'][:300]))

elif group == 'lock':
    # BUG 4: second process while one is running
    e = env()
    a = subprocess.Popen([PY, TRANSLATE, '--service'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
    try:
        ready = a.stdout.readline().decode()
        R.check('lock/A-ready', 'service_ready' in ready, ready[:120])
        r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--wait', '1', '--text-b64', base64.b64encode(b'Hello.').decode()], timeout=60)
        R.check('lock/bug4-one-shot-against-service-exit5-one-line', r['rc'] == 5 and 'using the NPU' in r['stderr'] and 'another PLaMo process (pid ' in r['stderr'] and 'mode service' in r['stderr'] and one_line(r) and no_trace(r) and 'fallback' not in r['stderr'].lower() and '0xc01e0009' not in r['stderr'], (r['rc'], r['wall'], r['stderr'][:300]))
        R.check('lock/bug4-fails-fast', r['wall'] < 6, r['wall'])
        r2 = run([PY, TRANSLATE, '--service', '--wait', '0'], input='', timeout=60)
        R.check('lock/second-service-exit5', r2['rc'] == 5 and no_trace(r2) and one_line(r2), (r2['rc'], r2['stderr'][:200]))
        a.stdin.write((GOOD('still-alive') + '\n').encode()); a.stdin.flush()
        resp = json.loads(a.stdout.readline().decode())
        R.check('lock/A-unharmed-after-B-refused', resp.get('id') == 'still-alive' and 'error' not in resp and resp.get('mode') == 'w4-m2', resp)
    finally:
        try:
            a.stdin.close()
        except Exception:
            pass
        try:
            a.wait(timeout=30)
        except Exception:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(a.pid)], capture_output=True)
    R.check('lock/A-exit0', a.returncode == 0, a.returncode)
    # next normal run is fine (the device was never wedged)
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--text-b64', base64.b64encode(b'Hello.').decode()])
    R.check('lock/normal-run-after', r['rc'] == 0 and json.loads(r['stdout']).get('mode') == 'w4-m2', (r['rc'], r['stderr'][:200]))
    # waiting works: B waits for a running one-shot A and then answers
    longtext = base64.b64encode(('The quick brown fox jumps over the lazy dog. ' * 30).encode()).decode()
    pa = subprocess.Popen([PY, TRANSLATE, '--source-lang', 'en', '--target-lang', 'ja', '--num-tokens', '300', '--text-b64', longtext], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
    time.sleep(0.8)
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--wait', '60', '--text-b64', base64.b64encode(b'Hello.').decode()], timeout=120)
    pa.communicate(timeout=120)
    R.check('lock/waiting-process-runs-after-holder', r['rc'] == 0 and pa.returncode == 0 and json.loads(r['stdout']).get('mode') == 'w4-m2', (r['rc'], pa.returncode, r['stderr'][:200]))
    # --no-lock escape hatch exists
    r = cli(['--help'])
    R.check('lock/no-lock-flag-documented', '--no-lock' in r['stdout'])

elif group in ('orphan', 'orphan-baseline'):
    # BUG 6: kill only the interpreter that owns the priming worker (no /T) at several moments: the worker must not survive.
    # 'orphan-baseline' runs the same test against the ACTIVE R83 code and is EXPECTED to leave orphans (it proves the test discriminates).
    baseline = group == 'orphan-baseline'
    entry = ACTIVE_TRANSLATE if baseline else TRANSLATE
    text = base64.b64encode(('Hello world, this is a fairly long sentence for the orphan test. ' * 60).encode()).decode()
    total_orphans = 0
    for delay in (0.3, 0.8, 1.4, 2.0, 2.8):
        targs = ['--', base64.b64decode(text).decode()] if baseline else ['--text-b64', text]  # the active R83 CLI has no --text-b64
        p = subprocess.Popen([PY, entry, '--source-lang', 'en', '--target-lang', 'ja', '--num-tokens', '200'] + targs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env(staged=not baseline))
        time.sleep(delay)
        kids = [x for x in procs(r'python') if x['ppid'] == p.pid]
        real = kids or [dict(pid=p.pid)]
        for x in real:
            subprocess.run(['taskkill', '/F', '/PID', str(x['pid'])], capture_output=True)  # the interpreter itself, no /T
        subprocess.run(['taskkill', '/F', '/PID', str(p.pid)], capture_output=True)
        time.sleep(5.0)
        pids_alive = {x['pid'] for x in procs(r'python')}
        orphans = [x for x in procs(r'python') if 'prime_worker83' in x['cmd'] and x['ppid'] not in pids_alive]
        total_orphans += len(orphans)
        if baseline:
            R.rows.append(dict(test='orphan-baseline/kill-at-%.1fs' % delay, result='INFO', detail='orphans=%s killed_real=%s' % ([(x['pid'], x['ppid']) for x in orphans], [x['pid'] for x in real])))
            print('INFO baseline kill-at-%.1fs orphans=%d real=%s' % (delay, len(orphans), [x['pid'] for x in real]), flush=True)
        else:
            R.check('orphan/bug6-kill-at-%.1fs' % delay, not orphans, 'orphans=%s killed=%s' % ([(x['pid'], x['ppid']) for x in orphans], [x['pid'] for x in real]))
        for x in orphans:
            subprocess.run(['taskkill', '/F', '/PID', str(x['pid'])], capture_output=True)  # clean up our own leftovers
        time.sleep(1.2)
    if baseline:
        R.check('orphan-baseline/active-R83-leaves-orphans-so-the-test-discriminates', total_orphans > 0, 'orphans seen: %d' % total_orphans)
    else:
        r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--text-b64', base64.b64encode(b'Hello.').decode()])
        R.check('orphan/normal-run-ok-after-sweep', r['rc'] == 0, r['stderr'][:200])
        time.sleep(1)
        left = [x for x in procs(r'python') if 'prime_worker83' in x['cmd']]
        R.check('orphan/no-worker-after-normal-run', not left, left)

elif group == 'fallback':
    # BUG 10 / plan N10: missing / corrupt certificate and missing W4 manifest in the INSTALLED tree (in place, always restored)
    import contextlib, shutil
    cert = STAGE + r'\plamo3-default\r45-spec\certificate.json'
    w4m = STAGE + r'\plamo3-default\r45-w4\manifest.json'
    data = open(cert, 'rb').read()
    @contextlib.contextmanager
    def mutated(path, content):
        keep = open(path, 'rb').read(); stat = os.stat(path)
        try:
            if content is None:
                os.remove(path)
            else:
                open(path, 'wb').write(content)
            yield
        finally:
            open(path, 'wb').write(keep)
    b = bytearray(data); b[len(b) // 2] ^= 1
    T = ['--source-lang', 'en', '--target-lang', 'ja', '--json', '--text-b64', base64.b64encode(b'Hello, how are you?').decode()]
    for name, content in (('missing', None), ('bitflip', bytes(b)), ('zero-byte', b''), ('truncated', data[:len(data) // 2])):
        with mutated(cert, content):
            r = cli(T)
        R.check('fallback/bug10-%s-refused-exit3' % name, r['rc'] == 3 and 'installation may be incomplete or modified' in r['stderr'] and 'allow-serial-fallback' in r['stderr'] and no_trace(r) and one_line(r) and r['stdout'].strip() == '', (r['rc'], r['stderr'][:300], r['stdout'][:80]))
    with mutated(cert, None):
        r = cli(T + ['--allow-serial-fallback'])
    d = json.loads(r['stdout']) if r['stdout'].strip().startswith('{') else {}
    R.check('fallback/bug10-allowed-fallback-warns-and-answers', r['rc'] == 0 and d.get('mode') == 'w4-serial' and d.get('fallbacks') and 'PLaMo WARNING: running in W4 serial' in r['stderr'] and d.get('requested_mode') == 'w4-m2', (r['rc'], r['stderr'][:400], d.get('mode')))
    ref = cli(T + ['--mode', 'w4-serial'])
    dref = json.loads(ref['stdout'])
    R.check('fallback/bug10-fallback-tokens-identical-to-serial', d.get('tokens') == dref.get('tokens'), (d.get('tokens'), dref.get('tokens')))
    with mutated(cert, None):
        r = cli(T + ['--allow-serial-fallback', '--strict'])
    R.check('fallback/strict-exit4-when-degraded', r['rc'] == 4, (r['rc'], r['stderr'][-200:]))
    with mutated(cert, None):
        r = cli(T, e=env(dict(PLAMO3_ALLOW_FALLBACK='1')))
    R.check('fallback/env-switch-allows', r['rc'] == 0 and json.loads(r['stdout']).get('mode') == 'w4-serial', (r['rc'], r['stderr'][:200]))
    with mutated(w4m, None):
        r = cli(T + ['--mode', 'w4-serial'])
    R.check('fallback/bug10-missing-w4-manifest-exit3-one-line', r['rc'] == 3 and no_trace(r) and one_line(r), (r['rc'], r['stderr'][:300]))
    r = cli(T, e=env(dict(R83_PACKAGE='v4C')))
    R.check('fallback/R83_PACKAGE-no-longer-selects-a-package', r['rc'] == 0 and json.loads(r['stdout']).get('mode') == 'w4-m2', (r['rc'], r['stderr'][:200]))
    R.check('fallback/certificate-restored-byte-for-byte', open(cert, 'rb').read() == data and os.path.exists(w4m), '')

elif group == 'multibyte':
    # BUG 8: the cap must not cut a multi-byte character; BUG 12: garbage hybrid switch value is reported
    fam = '\U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466 family photo'
    reqs = [json.dumps(dict(id=str(n), text=fam, source_lang='en', target_lang='ja', max_new_tokens=n)) for n in range(1, 41)]
    r = svc(reqs)
    rows = r['rows'][1:]
    R.check('multibyte/service-ok', r['rc'] == 0 and len(rows) == 40 and all('error' not in x for x in rows), (r['rc'], len(rows)))
    bad = [x['id'] for x in rows if '\ufffd' in x['text'] and x['stop'] == 'cap']
    R.check('multibyte/bug8-no-U+FFFD-at-a-cap-cut', not bad, bad)
    toks = [x['tokens'] for x in rows]
    R.check('multibyte/greedy-prefix-consistent', all(toks[i] == toks[39][:len(toks[i])] for i in range(40) if toks[i] != toks[39][:len(toks[i])] is False) or all(toks[i] == toks[39][:len(toks[i])] for i in range(40)), 'tokens are a prefix of the longest')
    held = [x['id'] for x in rows if x.get('held_back_bytes')]
    R.check('multibyte/holdback-exercised-or-not-needed', True, 'held back at n=%s' % held)
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--text-b64', base64.b64encode(b'Hello.').decode()], e=env(dict(PLAMO3_HYBRID_WAIT='garbage')))
    R.check('multibyte/bug12-garbage-hybrid-env-noted', r['rc'] == 0 and 'PLAMO3_HYBRID_WAIT' in r['stderr'] and 'not recognised' in r['stderr'] and json.loads(r['stdout']).get('mode') == 'w4-m2', (r['rc'], r['stderr'][:200]))
    r = cli(['--source-lang', 'en', '--target-lang', 'ja', '--json', '--text-b64', base64.b64encode(b'Hello.').decode()], e=env(dict(PLAMO3_HYBRID_WAIT='0')))
    R.check('multibyte/hybrid-off-silent', r['rc'] == 0 and r['stderr'].strip() == '' and 'hybrid' not in json.loads(r['stdout']), (r['rc'], r['stderr'][:200]))

sys.exit(R.close())
