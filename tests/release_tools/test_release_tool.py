# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest

os.environ['PLAMO3_INSTALL_TESTING'] = '1'
from common import *   # noqa: F401,F403
import release_tool as rt
import install as ins


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tree = os.path.join(self.tmp, 'tree')
        self.repo = os.path.join(self.tmp, 'repo')
        for base, files in ((self.tree, {'plamo3-runtime/a.bin': b'aaa', 'README.md': b'readme', 'xrt/python/pyxrt.pyd': b'MZ', 'venv/skip.txt': b'x', 'model/skip.txt': b'x', 'state/s.json': b'{}', 'SHA256SUMS': b'old'}),
                            (self.repo, {'README.md': b'readme'})):
            for rel, d in files.items():
                p = os.path.join(base, *rel.split('/'))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, 'wb').write(d)
        self.out = os.path.join(self.tmp, 'out')

    def test_stage_outputs(self):
        models = os.path.join(self.tmp, 'm.json')
        json.dump([dict(name='config.json', size=3, sha256='ab' * 32, url='https://huggingface.co/x/resolve/r/config.json')], open(models, 'w'))
        rc, out = quiet(rt.main, ['stage', '--tree', self.tree, '--out', self.out, '--repo-tree', self.repo, '--model-files', models, '--release', 'T1', '--hf-repo', 'o/PLaMo-x'])
        self.assertEqual(rc, 0)
        sums = open(os.path.join(self.out, 'SHA256SUMS')).read().splitlines()
        self.assertEqual([l.split('  ')[1] for l in sums], ['README.md', 'plamo3-runtime/a.bin', 'xrt/python/pyxrt.pyd'])   # excludes venv/model/state and the old SHA256SUMS
        man = json.load(open(os.path.join(self.out, 'PAYLOAD-MANIFEST.json')))
        src = {e['path']: e['source'] for e in man['files']}
        self.assertEqual(rt.classify('gate-data/plamo3-w4-r32/x.bin', False), ('hf', 'gate-data'))
        self.assertEqual(src, {'README.md': 'git', 'plamo3-runtime/a.bin': 'hf', 'xrt/python/pyxrt.pyd': 'hf', 'model/config.json': 'external'})
        self.assertEqual(man['hf']['repo'], 'o/PLaMo-x')
        self.assertNotIn(b'\r', open(os.path.join(self.out, 'SHA256SUMS'), 'rb').read())
        # the installer can read what the stage wrote
        self.assertEqual(len(ins.parse_sums(os.path.join(self.out, 'SHA256SUMS'))), 3)
        self.assertEqual(open(os.path.join(self.out, 'UPLOAD-LIST.txt')).read().count('\n'), 2)

    def test_check_detects_problems(self):
        quiet(rt.main, ['stage', '--tree', self.tree, '--out', self.out])
        sums = os.path.join(self.out, 'SHA256SUMS')
        self.assertEqual(quiet(rt.main, ['check', '--tree', self.tree, '--sums', sums])[0], 0)
        open(os.path.join(self.tree, 'README.md'), 'wb').write(b'changed')
        os.remove(os.path.join(self.tree, 'xrt', 'python', 'pyxrt.pyd'))
        rc, out = quiet(rt.main, ['check', '--tree', self.tree, '--sums', sums])
        self.assertEqual(rc, 1)
        self.assertIn('MISMATCH README.md', out)
        self.assertIn('MISSING  xrt/python/pyxrt.pyd', out)

    def test_plan_is_text_only_and_never_runs_anything(self):
        quiet(rt.main, ['stage', '--tree', self.tree, '--out', self.out, '--repo-tree', self.repo])
        rc, out = quiet(rt.main, ['plan', '--stage', self.out, '--hf-repo', 'o/PLaMo-x', '--github-repo', 'o/r', '--tag', 'v1.0.0'])
        self.assertEqual(rc, 0)
        self.assertIn('nothing has been uploaded', out)
        self.assertIn('hf upload-large-folder o/PLaMo-x', out)
        self.assertIn('tag -s v1.0.0', out)
        self.assertIn('gpg.format=ssh', out)
        self.assertFalse(os.path.exists(os.path.join(self.out, 'UPLOAD-PLAN.md')))
        quiet(rt.main, ['plan', '--stage', self.out, '--hf-repo', 'o/PLaMo-x', '--github-repo', 'o/r', '--tag', 'v1.0.0', '--write'])
        self.assertTrue(os.path.exists(os.path.join(self.out, 'UPLOAD-PLAN.md')))

    def test_sign_cmd_prints_but_does_not_sign(self):
        quiet(rt.main, ['stage', '--tree', self.tree, '--out', self.out])
        rc, out = quiet(rt.main, ['sign-cmd', '--sums', os.path.join(self.out, 'SHA256SUMS'), '--pubkey', 'KEY.pub'])
        self.assertEqual(rc, 0)
        self.assertIn('ssh-keygen -Y sign -n file -f KEY.pub', out)
        self.assertNotIn('gpg', out.lower().replace('gpg.format', ''))
        self.assertFalse(os.path.exists(os.path.join(self.out, 'SHA256SUMS.sig')))

    def test_no_publishing_code_paths(self):
        src = open(os.path.join(B1, 'tools', 'release_tool.py')).read()
        for forbidden in ('upload_file', 'create_repo', 'git push', 'requests.post', 'urlopen', 'huggingface_hub', "'push'"):
            self.assertNotIn(forbidden, src.replace('git -C <public repo> push', ''), forbidden)


class FindSshKeygenTests(unittest.TestCase):
    """The order of ssh-keygen candidates and the size rule for the Windows built-in client (it hangs on -Y verify input over ~8 KB). Fake paths only."""
    ENV = {'SystemRoot': 'C:\\Windows', 'ProgramFiles': 'C:\\Program Files'}
    SYS = 'C:\\Windows\\System32\\OpenSSH\\ssh-keygen.exe'
    GIT = 'C:\\Program Files\\Git\\usr\\bin\\ssh-keygen.exe'

    def pick(self, mod, present, which):
        return mod.find_ssh_keygen(which=lambda n: which.get(n), isfile=lambda p: p in present, environ=self.ENV)

    def test_order_git_first_then_path_then_system32(self):
        for mod in (rt, ins):
            for gitexe in ('C:\\Program Files\\Git\\cmd\\git.exe', 'C:\\Program Files\\Git\\bin\\git.exe', 'C:\\Program Files\\Git\\mingw64\\bin\\git.exe'):
                self.assertEqual(self.pick(mod, {self.GIT, self.SYS}, {'git': gitexe, 'ssh-keygen': self.SYS}), self.GIT, gitexe)
            self.assertEqual(self.pick(mod, {self.GIT, self.SYS}, {}), self.GIT)                       # via %ProgramFiles%
            other = 'D:\\tools\\ssh-keygen.exe'
            self.assertEqual(self.pick(mod, {self.SYS, other}, {'ssh-keygen': other}), other)         # PATH, not git
            self.assertEqual(self.pick(mod, {self.SYS}, {'ssh-keygen': self.SYS}), self.SYS)           # system32 last
            self.assertEqual(self.pick(mod, {self.SYS, other}, {'ssh-keygen': self.SYS.lower()}), self.SYS)   # the System32 client found on PATH (any case) is still the built-in one
            self.assertIsNone(self.pick(mod, set(), {}))
            relocated = 'E:\\Dev\\Git\\cmd\\git.exe'
            self.assertEqual(self.pick(mod, {'E:\\Dev\\Git\\usr\\bin\\ssh-keygen.exe', self.SYS}, {'git': relocated}), 'E:\\Dev\\Git\\usr\\bin\\ssh-keygen.exe')

    def test_builtin_client_is_not_used_for_large_files(self):
        tmp = tempfile.mkdtemp()
        big, small = os.path.join(tmp, 'big'), os.path.join(tmp, 'small')
        open(big, 'wb').write(b'x' * 9000); open(small, 'wb').write(b'x' * 100)
        signing = dict(public_key='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ', principal='who', namespace='file')
        builtin = ins.builtin_ssh_keygen_path()
        st, msg = ins.ssh_verify(big, small, signing, ssh_keygen=builtin)
        self.assertEqual(st, 'unavailable')
        self.assertIn('Git for Windows', msg)
        ok, msg2 = rt.ssh_verify(big, small, signing, ssh_keygen=builtin)
        self.assertFalse(ok)
        self.assertIn('Git for Windows', msg2)
        self.assertEqual(ins.SSH_BUILTIN_MAX_BYTES, 8000)
        # a small file is handed to the tool (here a path that does not exist: 'unavailable' for another reason, not the size rule)
        st, msg = ins.ssh_verify(small, small, signing, ssh_keygen=builtin)
        self.assertNotIn('8 KB', msg)


def _ssh_keygen_ok():
    exe = rt.find_ssh_keygen()
    if not exe:
        return False
    _, _, err = ssh_run([exe, '-Y', 'verify'], check=False)
    return b'unknown option' not in err.lower() and b'illegal option' not in err.lower()


@unittest.skipUnless(_ssh_keygen_ok(), 'ssh-keygen with -Y support not installed')
class SshSignatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp0 = tempfile.mkdtemp()
        for name in ('k1', 'k2'):
            ssh_run([rt.find_ssh_keygen(), '-q', '-t', 'ed25519', '-N', '', '-C', 'throwaway-comment-' + name, '-f', os.path.join(cls.tmp0, name)])
        cls.pub1 = open(os.path.join(cls.tmp0, 'k1.pub')).read().strip()
        cls.fpr = rt.ssh_key_fingerprint(cls.pub1)
        cls.signing = dict(public_key=' '.join(cls.pub1.split()[:2]), fingerprint=cls.fpr, principal=HANDLE, namespace='file')

    def sign(self, path, key='k1', ns='file'):
        ssh_run([rt.find_ssh_keygen(), '-q', '-Y', 'sign', '-n', ns, '-f', os.path.join(self.tmp0, key + '.pub'), path])

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sums = os.path.join(self.tmp, 'SHA256SUMS')
        open(self.sums, 'w').write('%s  a.txt\n' % ('a' * 64))
        self.sign(self.sums)
        self.sig = self.sums + '.sig'

    def test_fingerprint_matches_ssh_keygen(self):
        out = ssh_run([rt.find_ssh_keygen(), '-l', '-f', os.path.join(self.tmp0, 'k1.pub')])[1].decode()
        self.assertEqual(out.split()[1], self.fpr)
        self.assertNotIn('comment', self.signing['public_key'])

    def test_good_with_and_without_external_pin(self):
        self.assertTrue(rt.ssh_verify(self.sums, self.sig, self.signing)[0])
        self.assertTrue(rt.ssh_verify(self.sums, self.sig, self.signing, self.fpr)[0])
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, self.signing)[0], 'ok')
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, self.signing, self.fpr)[0], 'ok')

    def test_wrong_external_pin_and_inconsistent_release_json(self):
        other = rt.ssh_key_fingerprint(open(os.path.join(self.tmp0, 'k2.pub')).read())
        self.assertFalse(rt.ssh_verify(self.sums, self.sig, self.signing, other)[0])
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, self.signing, other)[0], 'bad')
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, dict(self.signing, fingerprint=other))[0], 'bad')

    def test_tampered_file_wrong_key_wrong_namespace_missing_and_garbage_signature(self):
        for fn in (lambda a: rt.ssh_verify(*a)[0], lambda a: ins.ssh_verify(*a)[0] == 'ok'):
            self.assertTrue(fn((self.sums, self.sig, self.signing)))
        other = dict(self.signing, public_key=' '.join(open(os.path.join(self.tmp0, 'k2.pub')).read().split()[:2]), fingerprint=None)
        self.assertFalse(rt.ssh_verify(self.sums, self.sig, other)[0])            # signed by k1, pinned k2
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, other)[0], 'bad')
        ns = os.path.join(self.tmp, 'ns'); open(ns, 'w').write(open(self.sums).read()); self.sign(ns, ns='git')
        self.assertFalse(rt.ssh_verify(ns, ns + '.sig', self.signing)[0])        # signed in another namespace
        self.assertEqual(ins.ssh_verify(ns, ns + '.sig', self.signing)[0], 'bad')
        self.assertEqual(ins.ssh_verify(self.sums, os.path.join(self.tmp, 'nope.sig'), self.signing)[0], 'bad')
        open(os.path.join(self.tmp, 'junk.sig'), 'w').write('not a signature')
        self.assertEqual(ins.ssh_verify(self.sums, os.path.join(self.tmp, 'junk.sig'), self.signing)[0], 'bad')
        open(self.sums, 'a').write('%s  b.txt\n' % ('b' * 64))
        self.assertFalse(rt.ssh_verify(self.sums, self.sig, self.signing)[0])
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, self.signing)[0], 'bad')

    def test_large_file_verifies_without_the_builtin_windows_client(self):
        exe = rt.find_ssh_keygen()
        if rt.is_builtin_ssh_keygen(exe):
            self.skipTest('only the built-in Windows ssh-keygen is available (it cannot verify large files)')
        big = os.path.join(self.tmp, 'BIG')
        open(big, 'w').write(''.join('%064x  file-%d.bin\n' % (i, i) for i in range(1500)))      # about 100 KB, like the real SHA256SUMS
        self.assertGreater(os.path.getsize(big), 100000)
        self.sign(big)
        self.assertTrue(rt.ssh_verify(big, big + '.sig', self.signing)[0])
        self.assertEqual(ins.ssh_verify(big, big + '.sig', self.signing)[0], 'ok')
        open(big, 'a').write('x')
        self.assertFalse(rt.ssh_verify(big, big + '.sig', self.signing)[0])

    def test_no_key_pinned_is_not_a_pass(self):
        self.assertFalse(rt.ssh_verify(self.sums, self.sig, None)[0])
        self.assertEqual(ins.ssh_verify(self.sums, self.sig, None)[0], 'unavailable')

    def test_cli_verify_sig(self):
        rj = os.path.join(self.tmp, 'release.json')
        json.dump(dict(signing=self.signing), open(rj, 'w'))
        self.assertEqual(quiet(rt.main, ['verify-sig', '--sums', self.sums, '--sig', self.sig, '--release-json', rj, '--fingerprint', self.fpr])[0], 0)
        self.assertEqual(quiet(rt.main, ['verify-sig', '--sums', self.sums, '--sig', self.sig, '--release-json', rj, '--fingerprint', 'SHA256:' + 'A' * 43])[0], 1)

    def test_installer_signature_step(self):
        class A: pass
        class L:
            def __init__(self): self.msgs = []
            def ok(self, m): self.msgs.append(m)
            def info(self, m): self.msgs.append(m)
        root = self.tmp
        a = A(); a.sums_sig = None; a.fingerprint = None; a.require_signature = False
        release = dict(signing=self.signing)
        self.assertEqual(ins.signature_step(root, release, a, L()), 'ok')                       # default SHA256SUMS.sig next to SHA256SUMS
        a.fingerprint = 'SHA256:' + 'A' * 43
        with self.assertRaises(ins.Fail):
            ins.signature_step(root, release, a, L())
        a.fingerprint = None; os.remove(self.sig)
        self.assertEqual(ins.signature_step(root, release, a, L()), 'absent')
        a.require_signature = True
        with self.assertRaises(ins.Fail):
            ins.signature_step(root, release, a, L())


class NoticeTests(unittest.TestCase):
    def test_lineage_blocks(self):
        import make_notice as mn
        tpl = open(os.path.join(B1, 'NOTICE.template'), encoding='utf-8').read()
        clean = mn.render(tpl, 'clean', 'DRAFTER-A')
        self.assertIn('DRAFTER-A', clean)
        self.assertNotIn('BEGIN:', clean)
        self.assertNotIn('@DRAFTER_NAME@', clean)
        self.assertIn('Built with PLaMo', clean)
        self.assertIn('pybind11', clean)
        for credit in ('OpenAssistant', 'Tatoeba', 'CC BY 2.0 FR', 'https://tatoeba.org', 'PLaMo-ATTRIBUTION-D20.jsonl.gz', 'Article 13', 'Aozora', 'Gutenberg'):
            self.assertIn(credit, clean)
        for banned in ('ShareAlike', 'CC BY-SA', 'KFTT', 'WikiMatrix', 'Dolly', 'Wikipedia'):
            self.assertNotIn(banned, clean)
        with self.assertRaises(ValueError):
            mn.render(tpl, 'other', 'x')
        with self.assertRaises(ValueError):
            mn.render(tpl, 'legacy', 'x')

    def test_shipped_notice_matches_a_lineage(self):
        import make_notice as mn
        tpl = open(os.path.join(B1, 'NOTICE.template'), encoding='utf-8').read()
        shipped = open(os.path.join(B1, 'NOTICE'), encoding='utf-8').read()
        self.assertTrue(any(shipped == mn.render(tpl, l, 'the PLaMo-port drafter package named in release.json') for l in mn.LINEAGES))


if __name__ == '__main__':
    unittest.main()
