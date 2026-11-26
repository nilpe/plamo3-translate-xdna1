"""Unit tests for the long-mode runtime (no device needed). Run: python -m unittest discover -s long/tests  (with numpy available)."""
import io, json, os, shutil, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'long'))
import long_core as C
import long_hip
import long_decode
import long_main
import long_engine as E


class Tok:
    """Fake sealed tokenizer: one token per character plus 10 template tokens."""
    tokens = ['x'] * 400

    def chat(self, messages, source_lang=None, target_lang=None):
        return list(range(10)) + [100 + (ord(c) % 50) for c in messages[0]['content']]

    def decode(self, ids):
        return ''.join(chr(97 + i % 26) for i in ids)


def make_handoff(d, n, digest, rows_full=None, sizes_wrong=None, logits=True, record=True):
    d = Path(d); (d / 'kv').mkdir(parents=True, exist_ok=True)
    layers = []
    for i in range(16):
        full = (i + 1) % 8 == 0
        rows = n if full else min(n, 2048)
        layers.append(dict(layer=i, kind='full' if full else 'sliding', first_row=0 if full else n - rows, rows=rows))
        for kind in 'kv':
            size = rows * 256
            if sizes_wrong == (i, kind):
                size -= 2
            (d / 'kv' / f'{kind}_{i}.bin').write_bytes(b'\0' * size)
    (d / 'meta.json').write_text(json.dumps(dict(n_tokens=n, layers=layers, argmax_last=7)))
    if logits:
        (d / 'last_logits.bin').write_bytes(b'\0' * (107520 * 4))
    if record:
        C.write_handoff_record(d, n, digest)


class ModeRule(unittest.TestCase):
    def test_router_prefilter(self):
        self.assertFalse(C.needs_router(100, None, False))
        self.assertTrue(C.needs_router(100, None, True))
        self.assertFalse(C.needs_router(1700, None, False))      # 1700 + 40 + 256 = 1996 <= 2048
        self.assertTrue(C.needs_router(1800, None, False))       # 2096 > 2048
        self.assertTrue(C.needs_router(100, 2000, False))
        self.assertFalse(C.needs_router(100, 1500, False))

    def test_decide(self):
        self.assertEqual(C.decide(300, None, False), 'default')
        self.assertEqual(C.decide(1800, None, False), 'default')      # implicit cap only: unchanged default behaviour (Q2)
        self.assertEqual(C.decide(1800, 400, False), 'long')          # explicit cap overflows 2048
        self.assertEqual(C.decide(1800, 248, False), 'default')       # exactly 2048
        self.assertEqual(C.decide(1800, 249, False), 'long')
        self.assertEqual(C.decide(2045, None, False), 'default')
        self.assertEqual(C.decide(2046, None, False), 'long')          # default refuses > 2045
        self.assertEqual(C.decide(10, None, True), 'long')
        self.assertEqual(C.decide(262143, None, False), 'long')
        with self.assertRaises(C.LongRequestError):
            C.decide(262144, None, False)
        with self.assertRaises(C.LongRequestError):
            C.decide(0, None, False)

    def test_long_cap(self):
        self.assertEqual(C.long_cap(300, None), 1024)
        self.assertEqual(C.long_cap(5000, None), 10000)
        self.assertEqual(C.long_cap(200000, None), 62144)            # remaining binds
        self.assertEqual(C.long_cap(1000, 5000), 5000)
        with self.assertRaises(C.LongRequestError):
            C.long_cap(262000, 200)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(C.LongRequestError):
                C.long_cap(1000, bad)

    def test_warning(self):
        self.assertIsNone(C.length_warning(C.WARN_ABOVE_TOKENS))
        self.assertIn('limit of the model itself', C.length_warning(C.WARN_ABOVE_TOKENS + 1))


class Dispatch(unittest.TestCase):
    def test_split_plan_boundaries_and_cap(self):
        for rows in (1, 7, 16, 1000, 1024, 1025, 4096):
            for per_row in (None, 0.05, 0.5, 2.0, 50.0):
                plan = C.split_plan(rows, per_row, 200.0, 16)
                self.assertEqual(sum(n for _, n in plan), rows)
                pos = 0
                for r0, n in plan:
                    self.assertEqual(r0, pos); pos += n
                    self.assertEqual(r0 % 16, 0)                    # boundaries on block rows: work-groups unchanged
                for r0, n in plan[:-1]:
                    self.assertEqual(n % 16, 0)
                if per_row:
                    for r0, n in plan:
                        self.assertTrue(n * per_row <= 200.0 + 1e-6 or n <= 16)

    def test_governor(self):
        g = C.DispatchGovernor(cap_ms=200.0, margin=1.25)
        self.assertEqual(g.plan(1024, 16), [(0, 1024)])               # nothing known yet: one launch
        self.assertTrue(g.want_measure())
        g.observe(1024, 400.0)                                         # 0.39 ms/row, x1.25
        plan = g.plan(1024, 16)
        self.assertGreater(len(plan), 1)
        self.assertEqual(g.splits, 1)
        self.assertEqual(g.max_launch_ms, 400.0)
        g.observe(1024, 50.0)
        self.assertEqual(g.plan(1024, 16), [(0, 1024)])

    def test_attention_tiles_have_exact_pointer_offsets(self):
        class Rec:
            def __init__(self): self.calls = []
            def launch(self, name, grid, block, args): self.calls.append((name, grid, block, args))
            def sync(self): pass
        eng = E.Engine.__new__(E.Engine)
        eng.hip = Rec(); eng.A = {'q': 1000, 'att': 5000}; eng.vct = {7: 9000}
        eng.gov = C.DispatchGovernor(cap_ms=100.0, margin=1.0); eng.gov.observe(64, 64 * 1.0)   # 1 ms/row -> 96-row tiles (multiple of 16)
        ah = Rec()
        eng._attention(ah, 16, 7, 1024, 4096, True, 262143, 3000)
        rows = 0
        for name, grid, block, args in ah.calls:
            q, kc, vct, out, n, pos0 = (a[1] for a in args[:6])
            r0 = (q - 1000) // (16 * 128 * 2)
            self.assertEqual(out - 5000, q - 1000)
            self.assertEqual(pos0, 4096 + r0)
            self.assertEqual(grid, ((n + 15) // 16, 1))
            rows += n
        self.assertEqual(rows, 1024)
        self.assertGreater(len(ah.calls), 5)

    def test_sliding_layers_never_split(self):
        class Rec:
            def __init__(self): self.calls = []
            def launch(self, *a): self.calls.append(a)
            def sync(self): pass
        eng = E.Engine.__new__(E.Engine)
        eng.hip = Rec(); eng.A = {'q': 0, 'att': 0}; eng.vct = {0: 1}
        eng.gov = C.DispatchGovernor(); eng.gov.observe(8, 1e6)
        ah = Rec()
        eng._attention(ah, 8, 0, 1024, 0, False, 4095, 1)
        self.assertEqual(len(ah.calls), 1)

    def test_device_need(self):
        self.assertLess(E.device_need_mb(2048), E.device_need_mb(262144))
        self.assertGreaterEqual(E.device_need_mb(262144), 2650)


class Estimates(unittest.TestCase):
    def test_eta(self):
        samples = [(i * 1024, 0.3 + i * 0.01) for i in range(10)]
        self.assertAlmostEqual(C.prefill_eta_s(samples, 10240, 10240), 0.0)
        eta = C.prefill_eta_s(samples, 10240, 20480)
        self.assertGreater(eta, 10 * 0.3)
        self.assertIsNone(C.prefill_eta_s([], 0, 1024))

    def test_estimates_in_measured_range(self):
        self.assertTrue(150 < C.prefill_estimate_s(262144) < 420)
        self.assertTrue(40 < C.prefill_estimate_s(131072) < 200)
        self.assertAlmostEqual(C.decode_estimate_s(1000, 1000), 19.4 * 1.0 + 0.2 * 1, delta=1.0)
        self.assertTrue(C.decode_estimate_s(250000, 1000) > 45)
        self.assertEqual(C.fmt_s(None), '?'); self.assertEqual(C.fmt_s(75), '1:15'); self.assertEqual(C.fmt_s(3725), '1:02:05')


class Handoff(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=str(HERE))
    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_ok_and_mutants(self):
        dg = C.prompt_digest(list(range(5000)))
        make_handoff(self.d, 5000, dg)
        C.check_handoff(self.d, 5000, dg)
        for kw in (dict(sizes_wrong=(7, 'k')), dict(sizes_wrong=(0, 'v')), dict(logits=False), dict(record=False)):
            d = tempfile.mkdtemp(dir=str(HERE))
            try:
                make_handoff(d, 5000, dg, **kw)
                with self.assertRaises(C.LongModeError):
                    C.check_handoff(d, 5000, dg)
            finally:
                shutil.rmtree(d, ignore_errors=True)
        with self.assertRaises(C.LongModeError):
            C.check_handoff(self.d, 5001, dg)
        with self.assertRaises(C.LongModeError):
            C.check_handoff(self.d, 5000, C.prompt_digest(list(range(5001))))
        os.remove(Path(self.d) / 'kv' / 'k_3.bin')
        with self.assertRaises(C.LongModeError):
            C.check_handoff(self.d, 5000, dg)

    def test_stale(self):
        old = Path(self.d) / 'old'; old.mkdir(); new = Path(self.d) / 'new'; new.mkdir()
        t = time.time() - 7200
        os.utime(old, (t, t))
        self.assertEqual(C.stale_handoff_dirs(self.d), [old])

    def test_digest_is_content_sensitive(self):
        self.assertNotEqual(C.prompt_digest([1, 2, 3]), C.prompt_digest([1, 2, 4]))
        self.assertEqual(C.prompt_digest([1, 2, 3]), C.prompt_digest((1, 2, 3)))


class Environment(unittest.TestCase):
    def test_ram(self):
        self.assertEqual(C.check_ram(4000), 4000)
        with self.assertRaises(C.LongModeError) as cm:
            C.check_ram(1000)
        self.assertIn('1000 MB free', cm.exception.message)
        free = C.free_ram_mb()
        self.assertTrue(free is None or free > 0)

    def test_hip_detect(self):
        class Lib:
            def __init__(self, count=1, load=0, free=4 << 30, setdev=0): self.count, self.load, self.free, self.setdev = count, load, free, setdev
            def hipSetDevice(self, i): return self.setdev
            def hipGetDeviceCount(self, ref): ref._obj.value = self.count; return 0
            def hipDeviceGetName(self, buf, n, d): buf.value = b'AMD Radeon 780M Graphics'; return 0
            def hipModuleLoadData(self, ref, buf): return self.load
            def hipMemGetInfo(self, f, t): f._obj.value = self.free; t._obj.value = 8 << 30; return 0
        co = tempfile.mkdtemp(dir=str(HERE))
        try:
            (Path(co) / 'attn2.co').write_bytes(b'x')
            info = long_hip.detect(co, hip=long_hip.Hip(lib=Lib()))
            self.assertEqual(info['name'], 'AMD Radeon 780M Graphics')
            for lib, text in ((Lib(count=0), 'no GPU'), (Lib(load=1), 'gfx1103'), (Lib(free=100 << 20), 'only 100 MB')):
                with self.assertRaises(C.LongModeError) as cm:
                    long_hip.detect(co, hip=long_hip.Hip(lib=lib))
                self.assertIn(text, cm.exception.message)
            with self.assertRaises(C.LongModeError) as cm:
                long_hip.Hip(lib=Lib(setdev=2))
            self.assertIn('out of memory', cm.exception.message)
            with self.assertRaises(C.LongModeError):
                long_hip.Hip(lib=Lib(setdev=719))
            (Path(co) / 'attn2.co').unlink()
            with self.assertRaises(C.errors.InstallProblem):
                long_hip.detect(co, hip=long_hip.Hip(lib=Lib()))
        finally:
            shutil.rmtree(co, ignore_errors=True)

    def test_missing_dll_message(self):
        with mock.patch.object(long_hip, 'system_dll', return_value=str(HERE / 'no-such.dll')):
            with self.assertRaises(C.LongModeError) as cm:
                long_hip.Hip()
        self.assertIn('Adrenalin', cm.exception.message)


class Paths(unittest.TestCase):
    def test_rebase(self):
        out = long_decode.rebase_paths({'a': '@ROOT@\\plamo3-long\\decode\\final.xclbin'}, 'C:\\PLaMo3')
        self.assertTrue(out['a'].endswith('final.xclbin'))
        for bad in ('C:\\Users\\x\\y', '@ROOT@\\..\\evil', 'relative', '@ROOT@/../x'):
            with self.assertRaises(C.errors.InstallProblem):
                long_decode.rebase_paths({'a': bad}, 'C:\\PLaMo3')
        with self.assertRaises(C.errors.InstallProblem):
            long_decode.rebase_paths({'a': 5}, 'C:\\PLaMo3')

    def test_dev_payload_paths_all_rebased(self):
        p = Path(__file__).resolve().parents[1] / 'plamo3-long' / 'plamo3-long' / 'decode' / 'artifact-paths.json'
        if p.exists():
            r = long_decode.rebase_paths(json.loads(p.read_text()), 'C:\\PLaMo3')
            self.assertEqual(len(r), 34)


class Text(unittest.TestCase):
    def parse(self, argv, stdin=b''):
        p = long_main.build_parser(); a = p.parse_args(argv)
        return long_main.read_text(a, p, io.BytesIO(stdin)), a

    def test_valid(self):
        t, a = self.parse(['--text-stdin'], 'こんにちは\n'.encode()); self.assertEqual(t, 'こんにちは')
        t, a = self.parse(['--text-stdin'], b'\xef\xbb\xbfabc\r\n'); self.assertEqual(t, 'abc')
        import base64
        t, a = self.parse(['--text-b64', base64.b64encode('日本語'.encode()).decode()]); self.assertEqual(t, '日本語')

    def test_invalid_exit_2(self):
        cases = [(['--text-stdin'], b'   \n'), (['--text-stdin'], b'\xff\xfe\x00'), (['x', '--text-stdin'], b'a'), (['--text-b64', '@@@'], b''), ([], b''),
                 (['<|plamo:x|>'], b''), (['abc', '--source-lang', 'en', '--target-lang', 'en'], b''), (['abc', '--num-tokens', '0'], b'')]
        for argv, stdin in cases:
            with self.assertRaises(SystemExit) as cm, mock.patch('sys.stderr', io.StringIO()):
                self.parse(argv, stdin)
            self.assertEqual(cm.exception.code, 2, argv)

    def test_finish_text_holds_back_partial_utf8(self):
        class T:
            tokens = ['a', '<0xE3>', '<0x81>']
            def decode(self, ids):
                b = bytearray()
                for i in ids:
                    t = self.tokens[i]
                    b += bytes([int(t[3:5], 16)]) if t.startswith('<0x') else t.encode()
                return b.decode('utf-8', 'replace')
        text, held = C.finish_text(T(), [0, 1, 2], capped=True)
        self.assertEqual((text, held), ('a', 2))
        text, held = C.finish_text(T(), [0, 1, 2], capped=False)
        self.assertEqual(held, 0)


class FakeProc:
    def __init__(self, lines, rc=0, block=False):
        self.stdout = iter(lines); self.returncode = rc; self.killed = False; self.block = block
    def wait(self, timeout=None):
        return self.returncode
    def kill(self): self.killed = True


class Child(unittest.TestCase):
    def run_child(self, lines, rc=0):
        prog = C.Progress(io.StringIO(), interval_s=0)
        return long_main.run_prefill_child('job.json', ['py'], prog, {}, 'cancel', popen=lambda *a, **k: FakeProc([json.dumps(l) + '\n' for l in lines], rc)), prog

    def test_done(self):
        lines = [dict(ev='device', name='X', free_mb=1, total_mb=2), dict(ev='chunk', pos=1024, n=2048, ms=300.0), dict(ev='chunk', pos=2048, n=2048, ms=310.0), dict(ev='done', argmax_last=5)]
        done, prog = self.run_child(lines)
        self.assertEqual(done['argmax_last'], 5)
        out = prog.stream.getvalue()
        self.assertIn('prefill 1024 / 2048 tokens (50%)', out); self.assertIn('prefill 2048 / 2048 tokens (100%)', out)

    def test_error_codes_propagate(self):
        with self.assertRaises(C.errors.PlamoError) as cm:
            self.run_child([dict(ev='error', code=6, message='no gpu')], rc=6)
        self.assertEqual((cm.exception.code, cm.exception.message), (6, 'no gpu'))
        with self.assertRaises(C.Cancelled):
            self.run_child([dict(ev='cancelled')], rc=130)
        with self.assertRaises(C.LongModeError):
            self.run_child([dict(ev='chunk', pos=1, n=2, ms=1)], rc=1)                      # died without an event
        with self.assertRaises(C.LongModeError):
            self.run_child([], rc=0)                                                          # exit 0 but no 'done' record
        done, _ = self.run_child(['garbage'] and [dict(ev='done', argmax_last=1)])
        self.assertEqual(done['argmax_last'], 1)


class Orchestration(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=str(HERE)); self.err = io.StringIO()
        pm = mock.patch.object(long_main, 'model_weights_dir', lambda root: Path(root)); pm.start(); self.addCleanup(pm.stop)
        import long_attest
        pa = mock.patch.object(long_attest, 'attest', lambda root, **k: (long_attest.Retained(), {})); pa.start(); self.addCleanup(pa.stop)
    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def args(self, *argv):
        a = long_main.build_parser().parse_args(['--state-dir', self.root, '--quiet', '--no-lock'] + list(argv) + ['x']); return a

    def fake_child(self, argmax=7):
        def runner(job_path, argv, progress, state, cancel):
            job = json.loads(Path(job_path).read_text())
            import numpy as np
            n = len(np.load(job['tokens_npy']))
            make_handoff(job['out'], n, C.prompt_digest(np.load(job['tokens_npy']).tolist()), record=False)
            state['device'] = dict(name='fake')
            return dict(argmax_last=argmax)
        return runner

    class Dec:
        def __init__(self, tokens, open_s=0.5): self.tokens, self.open_s, self.closed = tokens, open_s, False
        def open(self): return self
        def run(self, d, n, first, cap, on_token=None, cancelled=None, stop=None):
            self.stop_passed = stop is not None
            out = [first] + self.tokens[:cap - 1]
            return dict(tokens=out, steps_ms=[20.0] * (len(out) - 1), inject_s=0.01, repeat=None)
        def close(self): self.closed = True

    def test_route_only(self):
        a = long_main.build_parser().parse_args(['--route-only', 'abc'])
        r = long_main.translate(a, 'a' * 100, self.root, tokenizer=Tok())
        self.assertEqual(r['verdict'], 'default')
        a = long_main.build_parser().parse_args(['--route-only', '--long', 'abc'])
        self.assertEqual(long_main.translate(a, 'a' * 100, self.root, tokenizer=Tok())['verdict'], 'long')
        a = long_main.build_parser().parse_args(['--route-only', '--num-tokens', '2000', 'abc'])
        self.assertEqual(long_main.translate(a, 'a' * 100, self.root, tokenizer=Tok())['verdict'], 'long')

    def test_eos_and_cleanup(self):
        dec = self.Dec([5, 6, 1])
        r = long_main.translate(self.args('--long'), 'a' * 50, self.root, tokenizer=Tok(), decoder_factory=lambda: dec, child_runner=self.fake_child(), stderr=self.err)
        self.assertEqual(r['stop'], 'eos'); self.assertEqual(r['tokens'], [7, 5, 6, 1]); self.assertTrue(dec.closed)
        self.assertEqual(r['prompt_tokens'], 60)
        self.assertEqual([p for p in Path(self.root).iterdir()], [])           # work dir removed

    def test_repeat_stop_reason_message_and_flag(self):
        class RDec(self.Dec):
            def run(self, d, n, first, cap, on_token=None, cancelled=None, stop=None):
                self.stop_passed = stop is not None
                return dict(tokens=[7, 5, 6], steps_ms=[20.0, 20.0], inject_s=0.0, repeat=dict(start=1, period=25, keep=3))
        d = RDec([])
        r = long_main.translate(self.args('--long'), 'a' * 50, self.root, tokenizer=Tok(), decoder_factory=lambda: d, child_runner=self.fake_child(), stderr=self.err)
        self.assertEqual(r['stop'], 'repeat'); self.assertTrue(d.stop_passed); self.assertEqual(r['tokens'], [7, 5, 6])
        self.assertIn('started to repeat', self.err.getvalue()); self.assertIn('-NoRepeatStop', self.err.getvalue())
        d = RDec([]); err = io.StringIO()
        long_main.translate(self.args('--long', '--no-repeat-stop'), 'a' * 50, self.root, tokenizer=Tok(), decoder_factory=lambda: d, child_runner=self.fake_child(), stderr=err)
        self.assertFalse(d.stop_passed)

    def test_cap_stop_and_cap_value(self):
        dec = self.Dec(list(range(2, 5000)))
        r = long_main.translate(self.args('--long', '--num-tokens', '30'), 'a' * 50, self.root, tokenizer=Tok(), decoder_factory=lambda: dec, child_runner=self.fake_child(), stderr=self.err)
        self.assertEqual((r['stop'], r['new_tokens']), ('cap', 30))

    def test_prefill_only_keeps_dir(self):
        r = long_main.translate(self.args('--long', '--prefill-only'), 'a' * 50, self.root, tokenizer=Tok(), child_runner=self.fake_child(), stderr=self.err)
        self.assertTrue(Path(r['prefill_dir'], 'handoff.json').exists())

    def test_child_failure_cleans_up_and_propagates(self):
        def bad(*a, **k): raise C.LongModeError('no gpu')
        with self.assertRaises(C.LongModeError):
            long_main.translate(self.args('--long'), 'a' * 50, self.root, tokenizer=Tok(), child_runner=bad, stderr=self.err)
        self.assertEqual(list(Path(self.root).iterdir()), [])

    def test_decoder_cancel_cleans_up(self):
        class D(self.Dec):
            def run(self, *a, **k): raise C.Cancelled('cancelled')
        d = D([])
        with self.assertRaises(C.Cancelled):
            long_main.translate(self.args('--long'), 'a' * 50, self.root, tokenizer=Tok(), decoder_factory=lambda: d, child_runner=self.fake_child(), stderr=self.err)
        self.assertTrue(d.closed); self.assertEqual(list(Path(self.root).iterdir()), [])

    def test_attestation_refusal_exit3_before_any_work(self):
        import long_attest
        def refuse(root, **k): raise long_attest.AttestationError('content of long/a.py')
        called = []
        with mock.patch.object(long_attest, 'attest', refuse):
            with self.assertRaises(C.errors.PlamoError) as cm:
                long_main.translate(self.args('--long'), 'a' * 50, self.root, tokenizer=Tok(), child_runner=lambda *a, **k: called.append(1), stderr=self.err)
        self.assertEqual(cm.exception.code, 3); self.assertEqual(called, []); self.assertEqual(list(Path(self.root).iterdir()), [])

    def test_non_ascii_install_folder_refused_before_any_work(self):
        called = []
        with self.assertRaises(C.errors.PlamoError) as cm:
            long_main.translate(self.args('--long'), 'a' * 50, self.root + '\u30c6\u30b9\u30c8', tokenizer=Tok(), child_runner=lambda *a, **k: called.append(1), stderr=self.err)
        self.assertEqual(cm.exception.code, 3); self.assertIn('non-ASCII', cm.exception.message); self.assertEqual(called, [])
        r = long_main.translate(self.args('--long', '--prefill-only'), 'a' * 50, self.root, tokenizer=Tok(), child_runner=self.fake_child(), stderr=self.err)     # prefill-only (gates) is allowed
        self.assertTrue(r['prefill_dir'])

    def test_length_warning_printed_and_ram_refusal(self):
        class BigTok(Tok):
            def chat(self, *a, **k): return list(range(C.WARN_ABOVE_TOKENS + 5))
        with mock.patch.object(long_main, 'check_ram', side_effect=C.LongModeError('low ram')):
            with self.assertRaises(C.LongModeError):
                long_main.translate(self.args('--long'), 'abc', self.root, tokenizer=BigTok(), child_runner=self.fake_child(), stderr=self.err)
        self.assertIn('limit of the model itself', self.err.getvalue())

    def test_main_exit_codes(self):
        with mock.patch('sys.stderr', io.StringIO()) as se:
            self.assertEqual(long_main.main(['--no-lock', '--text-stdin', '--route-only'], stdin=io.BytesIO(b'abc')) in (0, 3, 1), True)
        with mock.patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as cm:
            long_main.main(['--source-lang', 'fr', 'x'])
        self.assertEqual(cm.exception.code, 2)
        def boom(*a, **k): raise C.LongModeError('no iGPU')
        with mock.patch.object(long_main, 'translate', boom), mock.patch('sys.stderr', io.StringIO()) as se:
            self.assertEqual(long_main.main(['--no-lock', 'abc']), 6)
            self.assertIn('PLaMo: error: no iGPU', se.getvalue())
        def intr(*a, **k): raise KeyboardInterrupt()
        with mock.patch.object(long_main, 'translate', intr), mock.patch('sys.stderr', io.StringIO()):
            self.assertEqual(long_main.main(['--no-lock', 'abc']), 130)
        def odd(*a, **k): raise ZeroDivisionError('x')
        with mock.patch.object(long_main, 'translate', odd), mock.patch('sys.stderr', io.StringIO()) as se:
            self.assertEqual(long_main.main(['--no-lock', 'abc']), 1)
            self.assertNotIn('Traceback', se.getvalue())


class ModelDir(unittest.TestCase):
    def test_missing_weights_message(self):
        d = tempfile.mkdtemp(dir=str(HERE))
        try:
            with self.assertRaises(C.LongModeError) as cm:
                long_main.model_weights_dir(d)
            self.assertIn('install_long.py --with-model', cm.exception.message)
            (Path(d) / 'model').mkdir(); (Path(d) / 'model' / 'model.safetensors').write_bytes(b'x')
            self.assertEqual(long_main.model_weights_dir(d), Path(d) / 'model')
            with mock.patch.dict(os.environ, {'PLAMO3_LONG_MODEL_DIR': 'C:/elsewhere'}):          # the old development override is gone
                self.assertEqual(long_main.model_weights_dir(d), Path(d) / 'model')
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Compaction(unittest.TestCase):
    def test_fix_present_in_long_runtime(self):
        s = (HERE.parents[1] / 'long' / 'npu' / 'long_pdrt.py').read_text()
        self.assertIn('for _v in list(self._dirty):', s)
        self.assertIn('self.sync_to(view)', s)


if __name__ == '__main__':
    unittest.main()
