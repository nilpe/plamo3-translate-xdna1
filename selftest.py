#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Self-test of an installed PLaMo 3 Translate XDNA1-NPU runtime (standard library only).

Needs a real installation (receipt, venv, payload) and the NPU, and takes about two minutes. Steps (exit code 10+N when
step N fails, 0 when all pass):

  1 preconditions     install receipt, licence record, venv, entry points, no IRON_PLAMO_* variables in the environment
  2 translations      four fixed sentences (ja->en x2, en->ja x2) must give exactly the shipped tokens
  3 identity          five prompts through `--service`: speculative decoding (w4-m2) vs W4 serial must give identical
                      tokens; the service must really run in w4-m2; no response may report a fallback
  4 w8 serial         one prompt in W8 serial must give the shipped tokens (skipped with --skip-w8)
  5 log scan          no fallback / downgrade / integrity message on stderr of any run
  6 turbo (optional)  only with --test-turbo: switch the NPU power mode and make sure it is restored

The self-test does not depend on which drafter package is installed: it checks behaviour, not names. Results go to
state/selftest-report.json (no personal data; fixed prompts only). `--record-expected` is for maintainers: it writes the
expected tokens from a W4 serial run.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time

EXIT_BASE = 10
FIXED = [  # (text, source, target) - the four prompts of the startup harness
    ('今日はいい天気ですね。', 'ja', 'en'),
    ('駅までの道を教えてください。', 'ja', 'en'),
    ('Thank you for your help.', 'en', 'ja'),
    ('The meeting starts at nine tomorrow morning.', 'en', 'ja'),
]
LONG_EN = ('The new library opens on Monday and will stay open until ten in the evening on weekdays. '
           'Visitors can borrow up to ten books at a time for three weeks, and every member receives a free card. '
           'On Saturdays the children\'s room holds a story hour at eleven, followed by a small workshop where '
           'volunteers help the young readers write and illustrate their own short stories. '
           'The reading room on the second floor is quiet, has a view of the river, and offers free wireless access. '
           'If you have questions, please ask the staff at the front desk; they will be glad to help you find '
           'anything from a local history book to a novel in a foreign language.')
IDENTITY = [  # (label, text, source, target)
    ('short', 'Good morning.', 'en', 'ja'),
    ('medium', '来週の会議は水曜日の午後三時に変更になりました。資料は前日までに共有してください。', 'ja', 'en'),
    ('long', LONG_EN, 'en', 'ja'),
    ('eos-early', 'はい。', 'ja', 'en'),
    ('en-ja', 'Could you tell me where the nearest pharmacy is?', 'en', 'ja'),
]
W8_PROMPT = ('今日はいい天気ですね。', 'ja', 'en')
BAD_STDERR = re.compile(r'PLaMo: W4 serial fallback|PLaMo: hybrid wait unavailable|PLaMo WARNING|StaleManifest|Traceback \(most recent|'
                        r'fallback \(|installation is incomplete|integrity', re.I)
NUM_TOKENS = 256


class StepFailed(Exception):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class Runner:
    def __init__(self, root, python, entry, env, timeout):
        self.root, self.python, self.entry, self.env, self.timeout = root, python, entry, env, timeout
        self.stderr_lines = []

    def _cmd(self, extra):
        return [self.python, self.entry] + extra

    def oneshot(self, text, src, dst, mode=None, wrapper=False):
        args = ['--source-lang', src, '--target-lang', dst, '--num-tokens', str(NUM_TOKENS), '--json']
        if mode:
            args += ['--mode', mode]
        if wrapper:
            ps = os.path.join(self.root, 'translate_plamo3.ps1')
            exe = 'powershell.exe'
            cmd = [exe, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps, text, '-SourceLang', src, '-TargetLang', dst,
                   '-NumTokens', str(NUM_TOKENS), '-Json'] + (['-Mode', mode] if mode else [])
        else:
            cmd = self._cmd(args + ['--', text])
        t0 = time.time()
        # stdin must not be inherited: a Windows PowerShell 5.1 child (translate_plamo3.ps1) blocks forever on an open, never-closing stdin pipe (found by the A2 test install under ssh)
        r = subprocess.run(cmd, capture_output=True, env=self.env, timeout=self.timeout, cwd=self.root, stdin=subprocess.DEVNULL)
        err = r.stderr.decode('utf-8', 'replace')
        self.stderr_lines += err.splitlines()
        out = r.stdout.decode('utf-8', 'replace').strip()
        if r.returncode != 0:
            raise StepFailed('exit code %d: %s' % (r.returncode, (err.strip().splitlines() or ['(no message)'])[-1][:300]))
        try:
            res = json.loads(out.splitlines()[-1])
        except (ValueError, IndexError):
            raise StepFailed('output is not JSON: %r' % out[:200])
        res['_wall_s'] = time.time() - t0
        return res

    def service(self, mode, prompts):
        """One --service process; returns (ready line, [responses])."""
        cmd = self._cmd(['--service', '--mode', mode])
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env, cwd=self.root)
        errbuf = []
        t = threading.Thread(target=lambda: errbuf.extend(proc.stderr.read().decode('utf-8', 'replace').splitlines()), daemon=True)
        t.start()
        deadline = time.time() + self.timeout
        try:
            ready = json.loads(proc.stdout.readline().decode('utf-8', 'replace') or 'null')
            if not ready or not ready.get('service_ready'):
                raise StepFailed('service did not become ready: %r' % ready)
            out = []
            for i, (label, text, src, dst) in enumerate(prompts):
                req = dict(id=i, text=text, source_lang=src, target_lang=dst, max_new_tokens=NUM_TOKENS)
                proc.stdin.write((json.dumps(req, ensure_ascii=False) + '\n').encode('utf-8'))
                proc.stdin.flush()
                line = proc.stdout.readline().decode('utf-8', 'replace')
                if not line:
                    raise StepFailed('service ended during request %s' % label)
                resp = json.loads(line)
                if 'error' in resp:
                    raise StepFailed('service error on %s: %s' % (label, resp['error']))
                out.append(resp)
                if time.time() > deadline:
                    raise StepFailed('timeout')
            proc.stdin.close()
            proc.wait(timeout=60)
            return ready, out
        finally:
            if proc.poll() is None:
                proc.kill()
            t.join(5)
            self.stderr_lines += errbuf


def load_expected(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def run_steps(a):
    root = os.path.abspath(a.root)
    state = os.path.join(root, 'state')
    report = dict(version=1, started=now(), steps=[], result='running')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    entry = os.path.join(root, 'IRON', 'iron', 'applications', 'plamo3', 'translate.py')
    python = a.python or os.path.join(root, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'python.exe' if os.name == 'nt' else 'python')
    run = Runner(root, python, entry, env, a.timeout)
    expected_path = a.expected or os.path.join(root, 'selftest-expected.json')
    record = {}

    def step(n, name, fn):
        t0 = time.time()
        entry_ = dict(step=n, name=name, status='running')
        report['steps'].append(entry_)
        print('[%d] %s' % (n, name), flush=True)
        try:
            detail = fn()
            entry_.update(status='pass', detail=detail)
            print('    PASS %s' % (detail or ''), flush=True)
        except StepFailed as e:
            entry_.update(status='FAIL', detail=str(e))
            print('    FAIL %s' % e, flush=True)
            report['result'] = 'FAIL at step %d' % n
            raise
        finally:
            entry_['seconds'] = round(time.time() - t0, 1)

    def pre():
        problems = []
        if not a.no_receipt_check:
            for need in ('install-receipt.json', 'license-accepted.json'):
                if not os.path.isfile(os.path.join(state, need)):
                    problems.append('state/%s missing (run install.ps1 first)' % need)
        for need, p in (('python', python), ('entry point', entry)):
            if not os.path.isfile(p):
                problems.append('%s not found: %s' % (need, p))
        stray = sorted(k for k in os.environ if k.startswith('IRON_PLAMO_'))
        if stray:
            problems.append('remove these environment variables first (the runtime refuses them): %s' % ', '.join(stray))
        if not a.record_expected:
            if not os.path.isfile(expected_path):
                problems.append('selftest-expected.json missing')
            elif not load_expected(expected_path).get('translations'):
                problems.append('selftest-expected.json holds no recorded tokens (maintainer: run selftest.py --record-expected on the release build)')
        if problems:
            raise StepFailed('; '.join(problems))
        return 'install found'

    expected = None
    try:
        step(1, 'preconditions', pre)
        if os.path.isfile(expected_path):
            expected = load_expected(expected_path)
        exp_tr = (expected or {}).get('translations', [])

        def translations():
            res = []
            for i, (text, s, d) in enumerate(FIXED):
                r = run.oneshot(text, s, d, wrapper=(i == 0 and a.use_wrapper))
                res.append(dict(text=text, source_lang=s, target_lang=d, tokens=r['tokens'], output=r['text']))
                if not a.record_expected:
                    if i >= len(exp_tr) or exp_tr[i]['tokens'] != r['tokens']:
                        raise StepFailed('translation %d differs from the shipped expectation: got %r, expected %r' % (
                            i + 1, r['text'], exp_tr[i]['output'] if i < len(exp_tr) else None))
                if i == 0:
                    report['mode'] = r.get('mode')
                    report['oneshot_spec_ready_ms'] = r.get('spec_ready_ms')
                time.sleep(1.0)       # a new process right after the previous one can wait ~0.5 s for the driver
            record['translations'] = res
            return '4 of 4 exact (mode %s)' % report.get('mode')
        step(2, 'fixed translations', translations)

        def identity():
            ready_a, spec = run.service('w4-m2', IDENTITY)
            report['spec_ready_ms'] = ready_a.get('spec_ready_ms')
            if ready_a.get('mode') != 'w4-m2':
                raise StepFailed('speculative decoding is not active: service mode is %r (expected w4-m2)' % ready_a.get('mode'))
            for (label, *_), r in zip(IDENTITY, spec):
                if r.get('mode') != 'w4-m2':
                    raise StepFailed('request %s ran in mode %r' % (label, r.get('mode')))
                if r.get('fallbacks'):
                    raise StepFailed('request %s reported fallbacks: %r' % (label, r['fallbacks']))
            time.sleep(1.0)
            ready_b, serial = run.service('w4-serial', IDENTITY)
            if ready_b.get('mode') != 'w4-serial':
                raise StepFailed('serial service mode is %r' % ready_b.get('mode'))
            diffs = [lab for (lab, *_), x, y in zip(IDENTITY, spec, serial) if x['tokens'] != y['tokens']]
            if diffs:
                raise StepFailed('speculative and serial tokens differ for: %s' % ', '.join(diffs))
            sr = report['spec_ready_ms']
            note = ''
            if sr and sr > 1500:
                note = ' (warning: spec-ready %.0f ms; a cold file cache or a virus scanner can cause this, a second run is usually faster)' % sr
            record['identity'] = [dict(label=lab, tokens=len(x['tokens'])) for (lab, *_), x in zip(IDENTITY, spec)]
            report['new_tokens'] = {lab: len(x['tokens']) for (lab, *_), x in zip(IDENTITY, spec)}
            return '5 of 5 identical, spec-ready %s ms%s' % (None if sr is None else round(sr), note)
        step(3, 'speculative vs serial identity', identity)

        def w8():
            if a.skip_w8:
                return 'skipped'
            time.sleep(1.0)
            r = run.oneshot(*W8_PROMPT, mode='w8-serial')
            record['w8'] = dict(tokens=r['tokens'], output=r['text'])
            if not a.record_expected:
                want = (expected or {}).get('w8', {}).get('tokens')
                if want is None or want != r['tokens']:
                    raise StepFailed('W8 serial output differs from the shipped reference: got %r' % r['text'])
            return 'exact'
        step(4, 'W8 serial', w8)

        def scan():
            bad = [l for l in run.stderr_lines if BAD_STDERR.search(l)]
            if bad:
                raise StepFailed('%d warning lines, first: %s' % (len(bad), bad[0][:200]))
            return 'no fallback or integrity message in %d stderr lines' % len(run.stderr_lines)
        step(5, 'log scan', scan)

        if a.test_turbo:
            def turbo():
                time.sleep(1.0)
                r = subprocess.run([python, entry, '--turbo', '--json', '--source-lang', 'en', '--target-lang', 'ja', '--', 'Hello.'],
                                   capture_output=True, env=env, timeout=a.timeout, cwd=root, stdin=subprocess.DEVNULL)
                err = r.stderr.decode('utf-8', 'replace')
                if r.returncode != 0:
                    raise StepFailed('turbo run failed: %s' % err.strip()[-200:])
                if 'could not switch' in err:
                    return 'turbo not available for this user (%s); translation still worked' % err.strip().splitlines()[-1][:120]
                return 'switched and restored'
            step(6, 'turbo (optional)', turbo)
        report['result'] = 'PASS'
        rc = 0
    except StepFailed:
        rc = EXIT_BASE + max(s['step'] for s in report['steps'])
    except subprocess.TimeoutExpired as e:
        report['steps'][-1].update(status='FAIL', detail='timeout after %s s' % a.timeout)
        report['result'] = 'FAIL at step %d' % report['steps'][-1]['step']
        print('    FAIL timeout', flush=True)
        rc = EXIT_BASE + report['steps'][-1]['step']
    report['finished'] = now()
    if a.record_expected and rc == 0:
        out = dict(version=1, recorded=now(), translations=record.get('translations', []), w8=record.get('w8'))
        with open(expected_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print('expected tokens written to %s' % expected_path)
    try:
        os.makedirs(state, exist_ok=True)
        with open(os.path.join(state, 'selftest-report.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
            f.write('\n')
    except OSError:
        pass
    print('SELFTEST %s' % report['result'], flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--expected')
    ap.add_argument('--skip-w8', action='store_true')
    ap.add_argument('--test-turbo', action='store_true', help='also try --turbo once (changes the NPU power mode and restores it)')
    ap.add_argument('--timeout', type=int, default=600, help='seconds per process')
    ap.add_argument('--use-wrapper', dest='use_wrapper', action='store_true', default=(os.name == 'nt'),
                    help='run the first translation through translate_plamo3.ps1 (default on Windows)')
    ap.add_argument('--no-wrapper', dest='use_wrapper', action='store_false')
    ap.add_argument('--record-expected', action='store_true', help='maintainers: write selftest-expected.json from this run')
    ap.add_argument('--python', help='interpreter to run the app with (default: <root>/venv)')
    ap.add_argument('--no-receipt-check', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    return run_steps(a)


if __name__ == '__main__':
    sys.exit(main())
