# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
import base64
import hashlib
import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
import zipfile

os.environ['PLAMO3_INSTALL_TESTING'] = '1'
from common import *   # noqa: F401,F403
import install as ins


def sha(b):
    return hashlib.sha256(b).hexdigest()


class RangeHandler(http.server.BaseHTTPRequestHandler):
    files = {}
    hits = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = self.files.get(self.path.split('?')[0])
        if data is None:
            self.send_response(404); self.end_headers(); return
        rng = self.headers.get('Range')
        RangeHandler.hits.append((self.path, rng))
        if rng and rng.startswith('bytes='):
            start = int(rng[6:].split('-')[0])
            if start >= len(data):
                self.send_response(416); self.end_headers(); return
            body = data[start:]
            self.send_response(206)
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, len(data) - 1, len(data)))
        else:
            body = data
            self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Silent:
    def info(self, *a): pass
    warn = ok = step = info


def fake_release(tmp, with_model=True, with_hf=True):
    """A minimal release tree: code files (git), two payload files, two model files; SHA256SUMS and manifests."""
    root = os.path.join(tmp, 'root dir u')            # spaces on purpose; non-ASCII roots are refused (XRT opens files through the ANSI API), see test_non_ascii_root_is_refused
    src = os.path.join(tmp, 'payload')
    os.makedirs(root)
    code = {'install.py': b'# x\n', 'requirements-release.txt': open(os.path.join(B1, 'requirements-release.txt'), 'rb').read(),
            'selftest.py': b'# st\n', 'known-good-env.json': json.dumps({'windows_build': ['10.0.26300'], 'xrt_coreutil': ['1']}).encode(),
            'LICENSE-PLaMo-en.md': b'licence text\n', 'LICENSE-PLaMo-ja.md': 'ライセンス\n'.encode(),
            'plamo3-default/r45-spec/certificate.json': b'{"cert": 1}\n'}
    payload = {'plamo3-runtime/a.bin': os.urandom(5000), 'plamo3-runtime/sub dir/b.bin': os.urandom(7000), 'xrt/python/pyxrt.pyd': b'MZ' * 100}
    model = {'config.json': b'{"m": 1}\n', 'model.safetensors': os.urandom(9000)}
    for rel, d in code.items():
        p = os.path.join(root, *rel.split('/')); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'wb').write(d)
    for rel, d in payload.items():
        p = os.path.join(src, *rel.split('/')); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'wb').write(d)
    msrc = os.path.join(tmp, 'modelsrc'); os.makedirs(msrc)
    for n, d in model.items():
        open(os.path.join(msrc, n), 'wb').write(d)
    files = []
    for rel, d in code.items():
        files.append(dict(path=rel, size=len(d), sha256=sha(d), source='git', **{'class': 'code'}))
    for rel, d in payload.items():
        files.append(dict(path=rel, size=len(d), sha256=sha(d), source='hf', **{'class': 'payload'}))
    for n, d in model.items():
        files.append(dict(path='model/' + n, size=len(d), sha256=sha(d), source='external', **{'class': 'model'}, url='', optional=n == 'model.safetensors'))
    sums = ''.join('%s  %s\n' % (sha(d), rel) for rel, d in sorted({**code, **payload}.items()))
    open(os.path.join(root, 'SHA256SUMS'), 'w', newline='\n').write(sums)
    man = dict(version=1, release='test', hf=dict(repo='owner/PLaMo-test', revision='abc123'), files=files)
    json.dump(man, open(os.path.join(root, 'PAYLOAD-MANIFEST.json'), 'w'))
    json.dump(dict(release='test', certificate='plamo3-default/r45-spec/certificate.json'), open(os.path.join(root, 'release.json'), 'w'))
    return root, src, msrc, payload, model


class HelperTests(unittest.TestCase):
    def test_parse_sums_and_unsafe(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, 'S')
        open(p, 'w').write('%s  a/b.txt\n%s *c.bin\n' % ('a' * 64, 'b' * 64))
        self.assertEqual(ins.parse_sums(p), {'a/b.txt': 'a' * 64, 'c.bin': 'b' * 64})
        for bad in ('../x', '/abs', 'C:/x', 'a/../b'):
            open(p, 'w').write('%s  %s\n' % ('a' * 64, bad))
            with self.assertRaises(ins.Fail):
                ins.parse_sums(p)
        open(p, 'w').write('nonsense\n')
        with self.assertRaises(ins.Fail):
            ins.parse_sums(p)

    def test_requirements_file_is_pinned_with_hashes(self):
        pins = ins.parse_requirements(os.path.join(B1, 'requirements-release.txt'))
        self.assertEqual([(n, v) for n, v, h in pins], [('numpy', '2.4.6'), ('ml_dtypes', '0.6.0')])
        self.assertTrue(all(len(h) == 1 and len(h[0]) == 64 for _, _, h in pins))

    def test_verify_record(self):
        tmp = tempfile.mkdtemp()
        di = os.path.join(tmp, 'pkg-1.0.dist-info'); os.makedirs(di)
        data = b'hello'
        open(os.path.join(tmp, 'pkg.py'), 'wb').write(data)
        h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        open(os.path.join(di, 'RECORD'), 'w').write('pkg.py,sha256=%s,5\npkg-1.0.dist-info/RECORD,,\n' % h)
        self.assertEqual(ins.verify_record(di, tmp), (1, []))
        open(os.path.join(tmp, 'pkg.py'), 'wb').write(b'HELLO')
        self.assertEqual(ins.verify_record(di, tmp)[1], ['changed pkg.py'])
        os.remove(os.path.join(tmp, 'pkg.py'))
        self.assertEqual(ins.verify_record(di, tmp)[1], ['missing pkg.py'])

    def test_ssh_fingerprint_pure_python(self):
        key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ'
        self.assertEqual(ins.ssh_key_fingerprint(key), 'SHA256:cdzAIU0HiCtE3KIO2DOTvv66tknD8jUPANG20hSleNU')
        self.assertEqual(ins.ssh_key_fingerprint(key + ' some comment'), 'SHA256:cdzAIU0HiCtE3KIO2DOTvv66tknD8jUPANG20hSleNU')
        for bad in ('', 'ssh-ed25519', 'rsa AAAA', 'ssh-ed25519 !!!'):
            with self.assertRaises(ValueError):
                ins.ssh_key_fingerprint(bad)

    def test_ssh_verify_without_pinned_key_or_tool_is_unavailable(self):
        self.assertEqual(ins.ssh_verify('x', 'y', None)[0], 'unavailable')
        key = dict(public_key='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ', principal='who', namespace='file')
        tmp = tempfile.mkdtemp(); f = os.path.join(tmp, 'f'); open(f, 'w').write('x')
        self.assertEqual(ins.ssh_verify(f, f, key, ssh_keygen=os.path.join(tmp, 'no-such-ssh-keygen'))[0], 'unavailable')

    def test_hf_url_quotes_non_ascii_and_spaces(self):
        u = ins.hf_url('o/PLaMo-x', 'abc', 'a b/ü.bin')
        self.assertEqual(u, 'https://huggingface.co/o/PLaMo-x/resolve/abc/a%20b/%C3%BC.bin')

    def test_blob_sha1_matches_git(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, 'f'); open(p, 'wb').write(b'abc\n')
        self.assertEqual(ins.blob_sha1(p), git(tmp if False else tmp, '--version') and subprocess_hash(p))

    def test_crlf_damage_detection(self):
        tmp = tempfile.mkdtemp()
        good = b'a\nb\n'
        p = os.path.join(tmp, 'f'); open(p, 'wb').write(b'a\r\nb\r\n')
        self.assertTrue(ins.looks_like_crlf_damage(p, sha(good)))
        open(p, 'wb').write(b'zzz\r\n')
        self.assertFalse(ins.looks_like_crlf_damage(p, sha(good)))


def subprocess_hash(p):
    import subprocess
    return subprocess.run(['git', 'hash-object', p], capture_output=True).stdout.decode().strip()


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), RangeHandler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_download_resume_with_range(self):
        data = os.urandom(100000)
        RangeHandler.files['/f.bin'] = data
        RangeHandler.hits.clear()
        tmp = tempfile.mkdtemp()
        dest = os.path.join(tmp, 'd', 'f.bin')
        os.makedirs(os.path.dirname(dest))
        open(dest + '.part', 'wb').write(data[:40000])
        ins.Downloader(Silent()).fetch('http://127.0.0.1:%d/f.bin' % self.port, dest, len(data))
        self.assertEqual(open(dest, 'rb').read(), data)
        self.assertEqual(RangeHandler.hits[0][1], 'bytes=40000-')
        self.assertFalse(os.path.exists(dest + '.part'))

    def test_complete_part_and_garbage_part(self):
        data = os.urandom(5000)
        RangeHandler.files['/g.bin'] = data
        tmp = tempfile.mkdtemp()
        dest = os.path.join(tmp, 'g.bin')
        open(dest + '.part', 'wb').write(data)           # already complete: server answers 416
        ins.Downloader(Silent()).fetch('http://127.0.0.1:%d/g.bin' % self.port, dest, len(data))
        self.assertEqual(open(dest, 'rb').read(), data)
        open(dest + '.part', 'wb').write(b'x' * 9000)    # longer than the file: restart
        os.remove(dest)
        ins.Downloader(Silent(), retries=3).fetch('http://127.0.0.1:%d/g.bin' % self.port, dest, len(data))
        self.assertEqual(open(dest, 'rb').read(), data)

    def test_404_is_a_network_failure_with_exit_code(self):
        with self.assertRaises(ins.Fail) as c:
            ins.Downloader(Silent()).fetch('http://127.0.0.1:%d/missing' % self.port, os.path.join(tempfile.mkdtemp(), 'x'), 1)
        self.assertEqual(c.exception.code, ins.EXIT_NETWORK)

    def test_unreachable_server(self):
        with self.assertRaises(ins.Fail) as c:
            d = ins.Downloader(Silent(), retries=1, timeout=1)
            d.fetch('http://127.0.0.1:9/x', os.path.join(tempfile.mkdtemp(), 'x'), 1)
        self.assertEqual(c.exception.code, ins.EXIT_NETWORK)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root, self.src, self.msrc, self.payload, self.model = fake_release(self.tmp)
        self.calls = []
        self.addCleanup(setattr, ins, 'venv_step', ins.venv_step)
        self.addCleanup(setattr, ins, 'selftest_step', ins.selftest_step)
        ins.venv_step = lambda root, args, log: self.calls.append('venv') or {'numpy': '2.4.6', 'ml_dtypes': '0.6.0'}
        ins.selftest_step = lambda root, args, log: self.calls.append('selftest')

    def go(self, *extra):
        return ins.main(['--root', self.root, '--quiet', '--no-npu-check', *extra])

    def test_non_ascii_root_is_refused(self):
        # found by the A2 test install: pyxrt (XRT) opens xclbin files through the ANSI API, so every start fails below a folder with non-ASCII characters
        args = ins.build_parser().parse_args(['--no-npu-check'])
        bad = os.path.join(tempfile.mkdtemp(), 'root dir \u00fc')
        os.makedirs(bad)
        with self.assertRaises(ins.Fail) as c:
            ins.preflight(bad, False, args, Silent(), need_bytes=1)
        self.assertEqual(c.exception.code, ins.EXIT_ENV)
        self.assertIn('path', c.exception.message)

    def test_plan_changes_nothing(self):
        before = sorted(os.listdir(self.root))
        self.assertEqual(self.go('--plan', '--payload-source', self.src), 0)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_licence_required_noninteractive(self):
        self.assertEqual(self.go('--payload-source', self.src, '--model-source', self.msrc), ins.EXIT_LICENSE)
        self.assertFalse(os.path.exists(os.path.join(self.root, 'plamo3-runtime')))

    def test_full_install_verify_and_uninstall(self):
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc), 0)
        self.assertEqual(self.calls, ['venv', 'selftest'])
        for rel in self.payload:
            self.assertEqual(open(os.path.join(self.root, *rel.split('/')), 'rb').read(), self.payload[rel])
        st = os.path.join(self.root, 'state')
        for f in ('license-accepted.json', 'install-receipt.json', 'env-record.json', 'install-manifest.json', 'install-summary.json'):
            self.assertTrue(os.path.isfile(os.path.join(st, f)), f)
        receipt = json.load(open(os.path.join(st, 'install-receipt.json')))
        self.assertEqual(receipt['version'], 1)
        self.assertEqual(receipt['install_root'], os.path.realpath(self.root))
        self.assertEqual(receipt['certificate_sha256'], sha(b'{"cert": 1}\n'))
        # verify passes, then fails when one payload byte changes
        self.assertEqual(self.go('--verify'), 0)
        p = os.path.join(self.root, 'plamo3-runtime', 'a.bin')
        data = bytearray(open(p, 'rb').read()); data[10] ^= 1
        open(p, 'wb').write(bytes(data))
        self.assertEqual(self.go('--verify'), ins.EXIT_VERIFY)
        # a normal run notices only size mismatches; --payload-source repairs after deleting the damaged file
        os.remove(p)
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--skip-selftest'), 0)
        self.assertEqual(open(p, 'rb').read(), self.payload['plamo3-runtime/a.bin'])
        # uninstall: dry run keeps everything, real run removes created files but not the code
        self.assertEqual(self.go('--uninstall', '--dry-run'), 0)
        self.assertTrue(os.path.exists(p))
        self.assertEqual(self.go('--uninstall', '--yes', '--keep-model'), 0)
        self.assertFalse(os.path.exists(p))
        self.assertTrue(os.path.isfile(os.path.join(self.root, 'model', 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.root, 'install.py')))
        self.assertTrue(os.path.isfile(os.path.join(self.root, 'SHA256SUMS')))
        self.assertFalse(os.path.exists(os.path.join(self.root, 'state')))
        self.assertEqual(self.go('--uninstall', '--yes'), ins.EXIT_USAGE)       # nothing recorded any more

    def test_modified_code_file_fails_verification_with_exit_3(self):
        self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--skip-selftest')
        open(os.path.join(self.root, 'selftest.py'), 'ab').write(b'tamper\n')
        self.assertEqual(self.go('--verify'), ins.EXIT_VERIFY)

    def test_model_hash_mismatch_is_rejected(self):
        open(os.path.join(self.msrc, 'config.json'), 'wb').write(b'{"m": 2}\n')       # same size, other content
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--skip-selftest'), ins.EXIT_VERIFY)
        self.assertTrue(os.path.exists(os.path.join(self.root, 'model', 'config.json.rejected')))

    def test_required_only_choice_is_remembered_for_verify(self):
        os.remove(os.path.join(self.msrc, 'model.safetensors'))
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--model-files', 'required', '--skip-selftest'), 0)
        self.assertEqual(self.go('--verify'), 0)

    def test_gate_data_is_skipped_unless_requested(self):
        man = json.load(open(os.path.join(self.root, 'PAYLOAD-MANIFEST.json')))
        data = b'g' * 500
        p = os.path.join(self.src, 'plamo3-w4-r31', 'reference', 'g.bin')
        os.makedirs(os.path.dirname(p)); open(p, 'wb').write(data)
        rel = 'plamo3-w4-r31/reference/g.bin'
        man['files'].append(dict(path=rel, size=len(data), sha256=sha(data), source='hf', **{'class': 'gate-data'}))
        json.dump(man, open(os.path.join(self.root, 'PAYLOAD-MANIFEST.json'), 'w'))
        open(os.path.join(self.root, 'SHA256SUMS'), 'a').write('%s  %s\n' % (sha(data), rel))
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--skip-selftest'), 0)
        self.assertFalse(os.path.exists(os.path.join(self.root, *rel.split('/'))))
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--skip-selftest', '--with-gate-data'), 0)
        self.assertTrue(os.path.exists(os.path.join(self.root, *rel.split('/'))))

    def test_optional_model_file_can_be_skipped(self):
        os.remove(os.path.join(self.msrc, 'model.safetensors'))
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--model-files', 'required', '--skip-selftest'), 0)

    def test_missing_release_files(self):
        os.remove(os.path.join(self.root, 'PAYLOAD-MANIFEST.json'))
        self.assertEqual(self.go('--plan'), ins.EXIT_VERIFY)

    def test_signature_required_but_absent(self):
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc, '--require-signature'), ins.EXIT_VERIFY)

    def test_selftest_failure_code_is_passed_through(self):
        def boom(root, args, log):
            raise ins.Fail(ins.EXIT_SELFTEST_BASE + 3, 'step 3 failed')
        ins.selftest_step = boom
        self.assertEqual(self.go('--accept-license', '--payload-source', self.src, '--model-source', self.msrc), 13)

    def test_space_root_worked(self):
        self.assertIn(' ', self.root)


class PreflightTests(unittest.TestCase):
    def test_disk_and_path_checks(self):
        tmp = tempfile.mkdtemp()
        args = ins.build_parser().parse_args(['--no-npu-check'])
        class L(Silent):
            pass
        with self.assertRaises(ins.Fail) as c:
            ins.preflight(tmp, False, args, L(), need_bytes=10 ** 18)
        self.assertEqual(c.exception.code, ins.EXIT_ENV)
        long_root = os.path.join(tmp, 'x' * 160)
        with self.assertRaises(ins.Fail) as c:
            ins.preflight(long_root, False, args, L(), need_bytes=1)
        self.assertIn('path', c.exception.message)
        facts = ins.preflight(tmp, False, args, L(), need_bytes=1)
        self.assertIn('python', facts)

    def test_python_must_be_313_outside_testing(self):
        old = ins.TESTING
        ins.TESTING = False
        try:
            args = ins.build_parser().parse_args(['--no-npu-check'])
            if not ins.python_ok():
                with self.assertRaises(ins.Fail) as c:
                    ins.preflight(tempfile.mkdtemp(), False, args, Silent(), need_bytes=1)
                self.assertEqual(c.exception.code, ins.EXIT_ENV)
        finally:
            ins.TESTING = old


def build_wheel(path, name, version, body):
    """A tiny pure-Python wheel with a valid RECORD."""
    dist = '%s-%s.dist-info' % (name, version)
    files = {'%s.py' % name: body.encode(),
             dist + '/METADATA': ('Metadata-Version: 2.1\nName: %s\nVersion: %s\n' % (name, version)).encode(),
             dist + '/WHEEL': b'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    rec = ''.join('%s,sha256=%s,%d\n' % (n, base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode(), len(d)) for n, d in files.items())
    files[dist + '/RECORD'] = (rec + dist + '/RECORD,,\n').encode()
    with zipfile.ZipFile(path, 'w') as z:
        for n, d in files.items():
            z.writestr(n, d)


import importlib.util


@unittest.skipIf(importlib.util.find_spec('ensurepip') is None, 'this interpreter cannot create venvs with pip')
class VenvStepTests(unittest.TestCase):
    def test_real_venv_with_hashed_local_wheel(self):
        tmp = tempfile.mkdtemp()
        root = os.path.join(tmp, 'r s')
        os.makedirs(os.path.join(root, 'wheels'))
        whl = os.path.join(root, 'wheels', 'fakepkg_plamo-1.0-py3-none-any.whl')
        build_wheel(whl, 'fakepkg_plamo', '1.0', 'X = 1\n')
        open(os.path.join(root, 'requirements-release.txt'), 'w').write('fakepkg_plamo==1.0 \\\n    --hash=sha256:%s\n' % sha(open(whl, 'rb').read()))
        args = ins.build_parser().parse_args([])
        have = ins.venv_step(root, args, Silent())
        self.assertEqual(have, {'fakepkg_plamo': '1.0'})
        self.assertTrue(os.path.isfile(ins.venv_python(root)))
        # a modified installed file is caught by the RECORD check
        import subprocess
        purelib = subprocess.run([ins.venv_python(root), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'], capture_output=True).stdout.decode().strip()
        open(os.path.join(purelib, 'fakepkg_plamo.py'), 'w').write('X = 2\n')
        with self.assertRaises(ins.Fail) as c:
            ins.venv_step(root, args, Silent())
        self.assertEqual(c.exception.code, ins.EXIT_VERIFY)

    def test_wrong_hash_is_refused_by_pip(self):
        tmp = tempfile.mkdtemp()
        root = os.path.join(tmp, 'r')
        os.makedirs(os.path.join(root, 'wheels'))
        build_wheel(os.path.join(root, 'wheels', 'fakepkg_plamo-1.0-py3-none-any.whl'), 'fakepkg_plamo', '1.0', 'X = 1\n')
        open(os.path.join(root, 'requirements-release.txt'), 'w').write('fakepkg_plamo==1.0 --hash=sha256:%s\n' % ('0' * 64))
        with self.assertRaises(ins.Fail) as c:
            ins.venv_step(root, ins.build_parser().parse_args([]), Silent())
        self.assertEqual(c.exception.code, ins.EXIT_NETWORK)


if __name__ == '__main__':
    unittest.main()
