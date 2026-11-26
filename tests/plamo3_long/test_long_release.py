"""Phase 3 unit tests: repetition stop, attestation (mutants), exporter, install_long, router delegate. python -m unittest discover -s rel_long/tests"""
import hashlib, io, json, os, random, shutil, stat, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'long'))
import long_core as C
import long_attest as A
import long_main
import export_long as X
import install_long as I
import freeze_long as F
import gate_index_long as G


def rnd(n, lo=0, hi=1000, seed=1):
    r = random.Random(seed); return [r.randrange(lo, hi) for _ in range(n)]


class RepeatStop(unittest.TestCase):
    def test_detects_and_keeps_exact_prefix(self):
        base, loop = rnd(300), rnd(25, 1000, 2000, 2)
        t = base + loop * 4
        r = C.find_first_loop(t)
        self.assertEqual((r['start'], r['period'], r['keep']), (300, 25, 325))
        self.assertEqual(t[:r['keep']], base + loop)
        # detected exactly when the third repeat completes
        self.assertEqual(r['at'], 300 + 3 * 25 - 1)

    def test_loop_start_extends_backwards_to_first_occurrence(self):
        loop = rnd(30, 1000, 2000, 3)
        t = rnd(100) + loop[10:] + loop * 3                 # the first period starts 20 tokens before the aligned boundary
        r = C.find_first_loop(t)
        self.assertLessEqual(r['keep'], r['start'] + r['period'])
        self.assertEqual(t[r['start']:r['start'] + r['period']], t[r['start'] + r['period']:r['start'] + 2 * r['period']])

    def test_not_triggered(self):
        loop = rnd(25, 1000, 2000, 4)
        self.assertIsNone(C.find_first_loop(rnd(300) + loop + rnd(5, seed=11) + loop + rnd(7, seed=12) + loop + rnd(6, seed=13) + loop))     # legal-text-like: the phrase recurs, never consecutively
        self.assertIsNone(C.find_first_loop(rnd(300) + loop * 2))                                                  # only 2 consecutive repeats
        short = rnd(19, 1000, 2000, 5)
        self.assertIsNone(C.find_first_loop(rnd(100) + short * 4))                                                # period below 20
        # table-like: identical row layout, one cell differs in every row -> never an exact repeat
        row = rnd(24, 1000, 2000, 6)
        rows = []
        for i in range(12):
            r = list(row); r[7] = 5000 + i; rows += r
        self.assertIsNone(C.find_first_loop(rnd(50) + rows))
        # near-repeat: 3 repeats where the third differs by one token
        near = list(loop); near[-1] = 9999
        self.assertIsNone(C.find_first_loop(rnd(50) + loop + loop + near))

    def test_table_with_identical_rows_is_a_loop(self):
        row = rnd(24, 1000, 2000, 7)
        r = C.find_first_loop(rnd(50) + row * 3)
        self.assertEqual(r['period'], 24)                                                                          # three identical consecutive rows ARE the defined loop

    def test_incremental_equals_offline_on_random_streams(self):
        for seed in range(40):
            rg = random.Random(seed)
            t = []
            for _ in range(rg.randrange(1, 8)):
                t += rnd(rg.randrange(5, 120), 0, 60, rg.randrange(1 << 30))
                if rg.random() < 0.5:
                    seg = rnd(rg.randrange(20, 40), 0, 60, rg.randrange(1 << 30)); t += seg * rg.randrange(1, 5)
            off = None
            for end in range(1, len(t) + 1):                                                                        # brute force: first end where a loop exists
                x = t[:end]
                for p in range(20, min(end // 3, 4096) + 1):
                    if x[end - p:end] == x[end - 2 * p:end - p] == x[end - 3 * p:end - 2 * p]:
                        off = end - 1; break
                if off is not None: break
            got = C.find_first_loop(t)
            self.assertEqual(got['at'] if got else None, off, seed)

    def test_prefix_property(self):
        t = rnd(500) + rnd(31, 1000, 2000, 9) * 6
        r = C.find_first_loop(t)
        self.assertEqual(t[:r['keep']], t[:r['keep']])
        self.assertTrue(r['keep'] < r['at'] + 1)

    def test_decoder_loop_stops_and_cuts(self):
        import long_decode
        class S:
            def __init__(self, seq): self.seq, self.i, self.length = seq, 0, 0
            def decode(self, cur):
                self.i += 1; return dict(token=self.seq[self.i - 1])
        loop = rnd(22, 1000, 2000, 10); seq = rnd(200) + loop * 10
        class RT:
            def inject_rows(self, *a): pass
        dec = long_decode.LongDecoder(root=tempfile.gettempdir()); dec.runtime = RT()
        class Mod:
            @staticmethod
            def RuntimeSession(rt): return S(seq[1:])
        dec._session_mod = Mod
        with mock.patch.object(long_decode, 'load_dump', lambda d, n: {}):
            out = dec.run('x', 10, seq[0], 5000, stop=C.RepeatDetector().feed)
        self.assertTrue(out['repeat'])
        self.assertEqual(out['tokens'], seq[:out['repeat']['keep']])
        self.assertLess(len(out['steps_ms']), 200 + 22 * 3 + 2)


def make_tree(root, weights=b'W' * 1000):
    root = Path(root)
    (root / 'long').mkdir(parents=True); (root / 'plamo3-long' / 'prefill').mkdir(parents=True); (root / 'plamo3-runtime' / 'r34-w4').mkdir(parents=True)
    (root / 'plamo3-default' / 'r45-spec').mkdir(parents=True); (root / 'model').mkdir()
    (root / 'long' / 'a.py').write_text('print(1)\n'); (root / 'long' / 'b.py').write_text('print(2)\n')
    (root / 'plamo3-long' / 'prefill' / 'k.co').write_bytes(b'\x01' * 500)
    (root / 'plamo3-runtime' / 'r34-w4' / 'initial.bin').write_bytes(b'I' * 3000)
    (root / 'plamo3-default' / 'r45-spec' / 'certificate.json').write_text('{"base": true}')
    (root / 'model' / 'model.safetensors').write_bytes(weights)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (root / 'long' / 'long_attest.py').write_text('z')
    files = [dict(path=p.relative_to(root).as_posix(), size=p.stat().st_size, sha256=sha(p)) for base in ('long', 'plamo3-long') for p in sorted((root / base).rglob('*')) if p.is_file()]
    shared = [dict(path='plamo3-runtime/r34-w4/initial.bin', size=3000, sha256=sha(root / 'plamo3-runtime/r34-w4/initial.bin'))]
    man = dict(version=1, files=files, shared=shared, weights=dict(path='model/model.safetensors', size=len(weights), sha256=hashlib.sha256(weights).hexdigest()))
    mb = json.dumps(man, indent=1, sort_keys=True).encode()
    cert = dict(version=1, manifest_sha256=hashlib.sha256(mb).hexdigest(), manifest_bytes=len(mb), base=dict(path='plamo3-default/r45-spec/certificate.json', sha256=sha(root / 'plamo3-default/r45-spec/certificate.json')))
    (root / 'long-manifest.json').write_bytes(mb); (root / 'long-certificate.json').write_text(json.dumps(cert))
    return man


class Attestation(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=str(HERE)); make_tree(self.d)
    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def reason(self, **kw):
        with self.assertRaises(C.errors.PlamoError) as cm:
            ret, _ = A.attest(self.d, **kw); ret.close()
        return cm.exception

    def test_ok_counts_and_retains(self):
        ret, m = A.attest(self.d); self.assertEqual(ret.stats['files'], 4 + 1 + 1); self.assertEqual(len(ret.handles), 6); ret.close()
        ret, m = A.attest(self.d, retain=False, check_weights=False); self.assertEqual(ret.stats['files'], 5)

    def mutate(self, rel, how='flip'):
        p = Path(self.d) / rel; b = bytearray(p.read_bytes())
        if how == 'flip': b[len(b) // 2] ^= 1
        elif how == 'trunc': b = b[:-1]
        elif how == 'grow': b += b'x'
        p.write_bytes(bytes(b))

    def test_mutants_all_refuse(self):
        for rel in ('long/a.py', 'plamo3-long/prefill/k.co', 'plamo3-runtime/r34-w4/initial.bin', 'model/model.safetensors', 'plamo3-default/r45-spec/certificate.json', 'long-manifest.json', 'long-certificate.json'):
            for how in ('flip', 'trunc', 'grow'):
                d = tempfile.mkdtemp(dir=str(HERE))
                try:
                    make_tree(d); old = self.d; self.d = d; self.mutate(rel, how)
                    e = self.reason()
                    self.assertEqual(e.code, 3, (rel, how)); self.assertIn('Run install_long.py --verify', e.message)
                finally:
                    self.d = old; shutil.rmtree(d, ignore_errors=True)

    def test_inventory_and_missing(self):
        (Path(self.d) / 'long' / 'extra.py').write_text('x'); self.assertIn('unlisted', self.reason().message)
        os.remove(Path(self.d) / 'long' / 'extra.py'); (Path(self.d) / 'long' / '__pycache__').mkdir(); (Path(self.d) / 'long' / '__pycache__' / 'a.pyc').write_bytes(b'x')
        A.attest(self.d, retain=False)[0].close()                            # pycache ignored
        os.remove(Path(self.d) / 'long' / 'b.py'); self.assertIn('missing', self.reason().message)

    def test_missing_weights_is_a_clear_environment_error(self):
        os.remove(Path(self.d) / 'model' / 'model.safetensors')
        e = self.reason(); self.assertEqual(e.code, 6); self.assertIn('install_long.py --with-model', e.message)

    def test_unsafe_manifest_paths(self):
        p = Path(self.d) / 'long-manifest.json'; m = json.loads(p.read_text()); m['files'][0]['path'] = '../evil'
        mb = json.dumps(m, indent=1, sort_keys=True).encode(); p.write_bytes(mb)
        c = json.loads((Path(self.d) / 'long-certificate.json').read_text()); c['manifest_sha256'] = hashlib.sha256(mb).hexdigest(); c['manifest_bytes'] = len(mb)
        (Path(self.d) / 'long-certificate.json').write_text(json.dumps(c)); self.assertIn('unsafe', self.reason().message)


class Exporter(unittest.TestCase):
    def test_build_and_scrub(self):
        d = tempfile.mkdtemp(dir=str(HERE))
        try:
            root = Path(d) / 'root'; repo = Path(d) / 'repo'; out = Path(d) / 'out'
            (root / 'plamo3-runtime' / 'r34-w4').mkdir(parents=True); (root / 'plamo3-default' / 'r45-spec').mkdir(parents=True)
            (repo / 'long').mkdir(parents=True); (repo / 'plamo3-long' / 'decode').mkdir(parents=True)
            (root / 'plamo3-runtime/r34-w4/initial.bin').write_bytes(b'I' * 100); (root / 'plamo3-runtime/r34-w4/transactions.bin').write_bytes(b'T' * 50)
            (root / 'plamo3-default/r45-spec/certificate.json').write_text('{}')
            sums = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in ('plamo3-runtime/r34-w4/initial.bin', 'plamo3-runtime/r34-w4/transactions.bin', 'plamo3-default/r45-spec/certificate.json')}
            (root / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (h, p) for p, h in sums.items()))
            (repo / 'long/x.py').write_text('1'); (repo / 'plamo3-long/decode/final.xclbin').write_bytes(b'X' * 10)
            (repo / 'plamo3-long/decode/artifact-paths.json').write_text(json.dumps({'context0': '@ROOT@\\plamo3-long\\decode\\final.xclbin', 'initial': '@ROOT@\\plamo3-runtime\\r34-w4\\initial.bin'}))
            tpl = Path(d) / 'tpl.json'
            tpl.write_text(json.dumps({'version': 1, 'artifacts': {'context0': {'nbytes': 1, 'ntfs_identity': {'x': 1}, 'path': 'C:\\Users\\ykogi\\x'}, 'initial': {'nbytes': 9, 'path': 'C:\\Users\\ykogi\\y'}},
                                       'sources': ['C:\\Users\\ykogi\\z'], 'origins': {}, 'model': {}}))
            m, c = X.build(root, repo, tpl, out, 'ab' * 32)
            pd = json.loads((repo / 'plamo3-long/decode/pd-manifest.json').read_text())
            self.assertEqual(pd['artifacts']['initial']['nbytes'], 100); self.assertEqual(pd['artifacts']['context0']['nbytes'], 10)
            self.assertNotIn('ntfs_identity', pd['artifacts']['context0']); self.assertNotIn('sources', pd)
            self.assertNotIn('ykogi', json.dumps(pd))
            self.assertIn('plamo3-long/decode/pd-manifest.json', [f['path'] for f in m['files']])
            self.assertEqual(c['base']['sha256'], sums['plamo3-default/r45-spec/certificate.json'])
            self.assertEqual(hashlib.sha256((out / 'long-manifest.json').read_bytes()).hexdigest(), c['manifest_sha256'])
            # a shared file that differs from the signed SHA256SUMS is refused
            (root / 'plamo3-runtime/r34-w4/initial.bin').write_bytes(b'J' * 100)
            with self.assertRaises(SystemExit):
                X.build(root, repo, tpl, out, 'ab' * 32)
            (root / 'plamo3-runtime/r34-w4/initial.bin').write_bytes(b'I' * 100)
            (root / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (h, p) for p, h in sums.items() if 'initial' not in p))
            with self.assertRaises(SystemExit):
                X.build(root, repo, tpl, out, 'ab' * 32)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_scrub(self):
        with self.assertRaises(SystemExit):
            X.scrub_check({'a': 'C:\\Users\\x'})
        X.scrub_check({'a': '@ROOT@\\plamo3-long'})


class InstallLong(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=str(HERE))
        self.root = Path(self.d) / 'root'; self.root.mkdir()
        self.src = Path(self.d) / 'src'; make_tree(self.src, weights=b'W' * 1000)
        shutil.rmtree(self.src / 'model')
        (self.src / 'model').mkdir()
        self.w = Path(self.d) / 'w.bin'; self.w.write_bytes(b'W' * 1000)
        # root = base install (default files only); the v1.1 SHA256SUMS (default + long entries) is in the source and, for the in-place case, copied into root by the tests that need it
        for rel in ('plamo3-runtime/r34-w4/initial.bin', 'plamo3-default/r45-spec/certificate.json'):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(self.src / rel, self.root / rel)
        (self.src / 'translate_long.ps1').write_text('wrapper'); (self.src / 'install_long.py').write_text('inst')
        names = ['long/a.py', 'long/b.py', 'long/long_attest.py', 'plamo3-long/prefill/k.co', 'long-manifest.json', 'long-certificate.json', 'translate_long.ps1', 'install_long.py', 'plamo3-runtime/r34-w4/initial.bin', 'plamo3-default/r45-spec/certificate.json']
        (self.src / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (hashlib.sha256((self.src / n).read_bytes()).hexdigest(), n) for n in names)); shutil.copyfile(self.src / 'SHA256SUMS', self.root / 'SHA256SUMS')
        self.patch = mock.patch.multiple(I, MODEL_SIZE=1000, MODEL_SHA256=hashlib.sha256(b'W' * 1000).hexdigest()); self.patch.start()
    def tearDown(self):
        self.patch.stop()
        for p in Path(self.d).rglob('*'):
            try:
                if p.is_file(): os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            except OSError: pass
        shutil.rmtree(self.d, ignore_errors=True)

    def args(self, *a):
        return I.build_parser().parse_args(['--root', str(self.root), '--quiet', '--skip-base-verify'] + list(a))

    def test_full_flow_with_model_source(self):
        rc = I.run(self.args('--source', str(self.src), '--with-model', '--yes', '--model-source', str(self.w)))
        self.assertEqual(rc, 0)
        wf = self.root / 'model' / 'model.safetensors'
        self.assertTrue(wf.is_file()); self.assertFalse(os.access(wf, os.W_OK) and os.name == 'nt')
        self.assertTrue(json.loads((self.root / 'state' / 'long-install.json').read_text())['weights'])
        self.assertEqual(I.run(self.args('--verify')), 0)
        self.assertEqual((self.root / 'translate_long.ps1').read_text(), 'wrapper')

    def test_without_model_note_and_state(self):
        self.assertEqual(I.run(self.args('--source', str(self.src))), 0)
        self.assertFalse(json.loads((self.root / 'state' / 'long-install.json').read_text())['weights'])

    def test_requires_base_install_and_sums(self):
        a = I.build_parser().parse_args(['--root', str(self.root), '--quiet'])
        with self.assertRaises(I.Fail) as cm:
            I.run(a)
        self.assertEqual(cm.exception.code, 3)
        os.remove(self.src / 'SHA256SUMS')
        with self.assertRaises(I.Fail):
            I.run(self.args('--source', str(self.src)))

    def test_hash_mismatch_of_long_files(self):
        (self.src / 'long' / 'a.py').write_text('tampered\n')
        with self.assertRaises(I.Fail) as cm:
            I.run(self.args('--source', str(self.src)))
        self.assertIn('differs from SHA256SUMS', cm.exception.message)
        self.assertFalse((self.root / 'long' / 'a.py').exists())            # the tampered source file was never copied into the installation

    def test_download_wrong_hash_deleted_and_disk_check(self):
        bad = Path(self.d) / 'bad.bin'; bad.write_bytes(b'X' * 1000)
        with self.assertRaises(I.Fail) as cm:
            I.run(self.args('--source', str(self.src), '--with-model', '--yes', '--model-source', str(bad)))
        self.assertEqual(cm.exception.code, 3); self.assertFalse((self.root / 'model' / 'model.safetensors').exists())
        with self.assertRaises(I.Fail) as cm:
            I.check_disk(self.d, 10 ** 12, free=10 ** 9)
        self.assertEqual(cm.exception.code, 6); self.assertIn('not enough free disk space', cm.exception.message)
        I.check_disk(self.d, 10, free=10 ** 9)

    def test_licence_gate_and_network_downloader(self):
        a = self.args('--source', str(self.src), '--with-model')
        with mock.patch('sys.stdin', io.StringIO('')):
            with self.assertRaises(I.Fail) as cm:
                I.run(a)
        self.assertEqual(cm.exception.code, 8)
        class DL:
            def fetch(self, url, dest, size):
                self.url = url; Path(dest).write_bytes(b'W' * 1000)
        dl = DL()
        I.run(self.args('--source', str(self.src), '--with-model', '--yes'), downloader=dl)
        self.assertIn('pfnet/plamo-3-translate-0.8b/resolve/8b4d42855622093e610ba9b90c2d2b59f8d5c15d/model.safetensors', dl.url)

    def test_notice_text(self):
        buf = io.StringIO()
        with mock.patch('sys.stdout', buf):
            I.licence_notice()
        t = buf.getvalue(); self.assertIn('huggingface.co/pfnet/plamo-3-translate-0.8b', t); self.assertIn('official upstream', t)


class Delegate(unittest.TestCase):
    def test_delegate_exit_for_default_verdict(self):
        class Tok:
            def chat(self, m, source_lang=None, target_lang=None): return list(range(50))
        a = long_main.build_parser().parse_args(['--delegate-if-default', 'abc'])
        r = long_main.translate(a, 'x' * 10, tempfile.gettempdir(), tokenizer=Tok())
        self.assertTrue(r['delegate'])
        a = long_main.build_parser().parse_args(['--delegate-if-default', '--long', 'abc'])
        with mock.patch.object(long_main, 'check_ram', side_effect=C.LongModeError('stop here')):
            with self.assertRaises(C.LongModeError):
                long_main.translate(a, 'x' * 10, tempfile.gettempdir(), tokenizer=Tok())
        self.assertEqual(long_main.DELEGATE_EXIT, 90)


class FreezeIndex(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=str(HERE)); make_tree(self.d); r = Path(self.d)
        (r / 'translate_plamo3.ps1').write_text('default wrapper'); (r / 'install.py').write_text('base installer')
        (r / 'translate_long.ps1').write_text('x'); (r / 'install_long.py').write_text('y'); (r / 'tools').mkdir()
        for n in ('export_long.py', 'freeze_long.py', 'gate_index_long.py'): (r / 'tools' / n).write_text('t')
        self.v100 = r / 'v100.sums'
        self.v100.write_text(''.join('%s  %s\n' % (hashlib.sha256((r / p).read_bytes()).hexdigest(), p) for p in ('translate_plamo3.ps1', 'install.py', 'plamo3-default/r45-spec/certificate.json', 'plamo3-runtime/r34-w4/initial.bin')) + '0' * 64 + '  SHA256SUMS\n')
    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def freeze(self, **kw):
        rep = F.freeze(self.d, self.v100, **kw)
        (Path(self.d) / 'state' / 'gates-long').mkdir(parents=True, exist_ok=True); (Path(self.d) / 'state' / 'gates-long' / 'freeze.json').write_text(json.dumps(rep)); return rep

    def test_default_unchanged_detects_each_default_edit(self):
        rep = self.freeze(); self.assertEqual((rep['result'], rep['default_unchanged']), ('PASS', True)); self.assertEqual(rep['required_sources_missing'], []); self.assertIn('long/a.py', rep['source_hashes']); self.assertTrue(rep['attested']['weights'])
        for rel in ('translate_plamo3.ps1', 'install.py', 'plamo3-default/r45-spec/certificate.json', 'plamo3-runtime/r34-w4/initial.bin'):
            p = Path(self.d) / rel; old = p.read_bytes(); p.write_bytes(old + b'!')
            try:
                rep = F.freeze(self.d, self.v100, attest=False); self.assertEqual((rep['result'], rep['default_unchanged'], rep['default_differing']), ('FAIL', False, [rel]))
            finally:
                p.write_bytes(old)
        (Path(self.d) / 'install.py').unlink(); self.assertFalse(F.freeze(self.d, self.v100, attest=False)['default_unchanged'])

    def test_missing_required_source_fails_freeze(self):
        for rel in ('translate_long.ps1', 'install_long.py', 'tools/export_long.py', 'long/long_attest.py', 'tools/gate_index_long.py'):
            p = Path(self.d) / rel; data = p.read_bytes(); p.unlink()
            try:
                rep = F.freeze(self.d, self.v100, attest=False); self.assertEqual(rep['result'], 'FAIL'); self.assertEqual(rep['required_sources_missing'], [rel])
            finally:
                p.write_bytes(data)

    def test_source_hash_tracks_every_required_file(self):
        base = F.freeze(self.d, self.v100, attest=False)['source_hashes']
        for rel in ('translate_long.ps1', 'install_long.py', 'tools/export_long.py', 'long/long_attest.py', 'tools/gate_index_long.py'):
            p = Path(self.d) / rel; data = p.read_bytes(); p.write_bytes(data + b'1')
            try:
                self.assertNotEqual(F.freeze(self.d, self.v100, attest=False)['source_hashes'][rel], base[rel])
            finally:
                p.write_bytes(data)

    def test_index_needs_every_step_fresh_and_pass(self):
        rep = self.freeze(attest=False)
        with self.assertRaises(AssertionError) as cm:
            G.build_index(self.d)
        self.assertIn('l4-compaction-bitwise', str(cm.exception))                    # L4 is a required step
        time_ok = rep['started_ns'] / 1e9 + 1
        for step in G.LONG_SUITE:
            G.record(self.d, step, 'PASS', 0)
        self.assertEqual(G.build_index(self.d)['steps'], len(G.LONG_SUITE))
        G.record(self.d, 'l4-compaction-bitwise', 'FAIL', 1)
        with self.assertRaises(AssertionError):
            G.build_index(self.d)
        G.record(self.d, 'l4-compaction-bitwise', 'PASS', 0)
        fr = json.loads((Path(self.d) / 'state' / 'gates-long' / 'freeze.json').read_text()); fr['epoch'] = 'other'
        (Path(self.d) / 'state' / 'gates-long' / 'freeze.json').write_text(json.dumps(fr))                      # a new epoch voids all old records
        with self.assertRaises(AssertionError):
            G.build_index(self.d)

    def test_index_refuses_when_default_changed(self):
        self.freeze(attest=False); fr = json.loads((Path(self.d) / 'state' / 'gates-long' / 'freeze.json').read_text()); fr['default_unchanged'] = False
        (Path(self.d) / 'state' / 'gates-long' / 'freeze.json').write_text(json.dumps(fr))
        for step in G.LONG_SUITE:
            G.record(self.d, step, 'PASS', 0)
        with self.assertRaises(AssertionError):
            G.build_index(self.d)


if __name__ == '__main__':
    unittest.main()
