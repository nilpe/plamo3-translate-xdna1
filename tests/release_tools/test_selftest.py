# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
import json
import os
import sys
import tempfile
import textwrap
import unittest

from common import *   # noqa: F401,F403
import selftest as st

STUB = textwrap.dedent('''
    import json, os, sys
    mode_env = os.environ.get('STUB', 'ok')
    a = sys.argv[1:]
    def get(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default
    def tokens(text, mode):
        t = [ord(c) % 997 for c in text][:int(get('--num-tokens', 256))]
        if mode == 'w4-serial' and mode_env == 'serial-diff':
            t = t[:-1] + [t[-1] + 1]
        return t
    def reply(req, mode):
        text = req['text']
        out = dict(id=req.get('id'), mode=mode, tokens=tokens(text, mode), text='T:' + text, prompt_tokens=3, ttft_ms=1.0, wall_ms=2.0, spec_ready_ms=700.0)
        if mode_env == 'fallbacks-field' and mode == 'w4-m2':
            out['fallbacks'] = [dict(reason='x')]
        if mode_env == 'w8-diff' and mode == 'w8-serial':
            out['tokens'] = [1, 2, 3]
        return out
    mode = get('--mode', 'w4-m2')
    if mode_env == 'no-spec' and mode == 'w4-m2':
        mode = 'w4-serial'
    if mode_env == 'stderr-fallback':
        print('PLaMo: W4 serial fallback (initialization): RuntimeError: x', file=sys.stderr)
    if '--service' in a:
        print(json.dumps(dict(service_ready=True, mode=mode, spec_ready_ms=700.0)), flush=True)
        for line in sys.stdin:
            if line.strip():
                print(json.dumps(reply(json.loads(line), mode), ensure_ascii=False), flush=True)
    else:
        text = a[a.index('--') + 1] if '--' in a else ''
        print(json.dumps(reply(dict(text=text), mode), ensure_ascii=False), flush=True)
        print('PLaMo: {"mode": "%s"}' % mode, file=sys.stderr)
''')


class SelfTestTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        app = os.path.join(self.root, 'IRON', 'iron', 'applications', 'plamo3')
        os.makedirs(app)
        open(os.path.join(app, 'translate.py'), 'w').write(STUB)
        os.makedirs(os.path.join(self.root, 'state'))
        for n in ('install-receipt.json', 'license-accepted.json'):
            open(os.path.join(self.root, 'state', n), 'w').write('{}')
        st.time.sleep = lambda s: None
        os.environ.pop('STUB', None)

    def run_st(self, *extra, stub='ok'):
        os.environ['STUB'] = stub
        return st.main(['--root', self.root, '--python', sys.executable, '--no-wrapper', *extra])

    def record(self):
        self.assertEqual(self.run_st('--record-expected'), 0)
        exp = json.load(open(os.path.join(self.root, 'selftest-expected.json')))
        self.assertEqual(len(exp['translations']), 4)

    def report(self):
        return json.load(open(os.path.join(self.root, 'state', 'selftest-report.json')))

    def test_refuses_without_expected_tokens(self):
        self.assertEqual(self.run_st(), st.EXIT_BASE + 1)

    def test_pass(self):
        self.record()
        self.assertEqual(self.run_st(), 0)
        rep = self.report()
        self.assertEqual(rep['result'], 'PASS')
        self.assertEqual([s['status'] for s in rep['steps']], ['pass'] * 5)

    def test_requires_install_records(self):
        os.remove(os.path.join(self.root, 'state', 'install-receipt.json'))
        self.assertEqual(self.run_st('--record-expected'), st.EXIT_BASE + 1)

    def test_translation_differs_from_expectation(self):
        self.record()
        p = os.path.join(self.root, 'selftest-expected.json')
        exp = json.load(open(p))
        exp['translations'][2]['tokens'] = [9]
        json.dump(exp, open(p, 'w'))
        self.assertEqual(self.run_st(), st.EXIT_BASE + 2)
        self.assertIn('translation 3', self.report()['steps'][-1]['detail'])

    def test_spec_vs_serial_mismatch(self):
        self.record()
        self.assertEqual(self.run_st(stub='serial-diff'), st.EXIT_BASE + 3)

    def test_speculation_not_active(self):
        self.record()
        self.assertEqual(self.run_st(stub='no-spec'), st.EXIT_BASE + 3)

    def test_fallback_field_fails(self):
        self.record()
        self.assertEqual(self.run_st(stub='fallbacks-field'), st.EXIT_BASE + 3)

    def test_w8_mismatch_and_skip(self):
        self.record()
        self.assertEqual(self.run_st(stub='w8-diff'), st.EXIT_BASE + 4)
        self.assertEqual(self.run_st('--skip-w8', stub='w8-diff'), 0)

    def test_stderr_fallback_line_fails(self):
        self.record()
        self.assertEqual(self.run_st(stub='stderr-fallback'), st.EXIT_BASE + 5)

    def test_stray_environment_variable_is_reported(self):
        os.environ['IRON_PLAMO_X'] = '1'
        try:
            self.assertEqual(self.run_st('--record-expected'), st.EXIT_BASE + 1)
            self.assertIn('IRON_PLAMO_X', self.report()['steps'][0]['detail'])
        finally:
            del os.environ['IRON_PLAMO_X']

    def test_five_identity_prompts_and_four_fixed(self):
        self.assertEqual(len(st.IDENTITY), 5)
        self.assertEqual(len(st.FIXED), 4)
        self.assertEqual(sorted({(s, d) for _, s, d in st.FIXED}), [('en', 'ja'), ('ja', 'en')])


if __name__ == '__main__':
    unittest.main()
