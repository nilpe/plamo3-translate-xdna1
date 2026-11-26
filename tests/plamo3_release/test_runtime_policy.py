"""Loud initialization fallback, language policy, hybrid-wait circuit breaker, worker parent watch (logic only; no device)."""
import collections
import contextlib
import io
import os
import subprocess
import sys
import textwrap
import time
import unittest

from _common import R83, IS_WINDOWS, TmpDirMixin
import default83
import hybridwait83
import plamo_errors as E


class StaleManifest(RuntimeError):
    pass


class Stub:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def bare_runtime(allow):
    rt = object.__new__(default83.DefaultRuntime)
    rt.allow_fallback = allow
    rt.primer, rt.env, rt.drafter = Stub(), Stub(), object()
    rt._closed = False
    rt.mode = rt.requested_mode = "w4-m2"
    rt.fallbacks = []
    rt.log = lambda line: None
    return rt


class LoudFallback(unittest.TestCase):
    def test_default_refuses_install_class_errors(self):
        rt = bare_runtime(False)
        primer, env = rt.primer, rt.env
        with self.assertRaises(E.InstallProblem) as cm:
            rt._refuse_initialization(StaleManifest("runtime manifest content changed; rebuild the bundle"))
        self.assertEqual(cm.exception.code, E.INSTALL)
        self.assertIn("--allow-serial-fallback", cm.exception.message)
        self.assertIn("install.ps1 -Verify", cm.exception.message)
        self.assertEqual((primer.closed, env.closed), (1, 1))  # nothing left running
        self.assertTrue(rt._closed)

    def test_allowed_fallback_does_not_raise(self):
        rt = bare_runtime(True)
        rt._refuse_initialization(StaleManifest("x"))
        rt._refuse_initialization(FileNotFoundError("gone"))
        rt._refuse_initialization(ValueError("configured package manifest hash is not approved"))
        self.assertFalse(rt._closed)

    def test_npu_and_environment_errors_never_fall_back(self):
        for allow in (True, False):
            rt = bare_runtime(allow)
            with self.assertRaises(E.NpuBusy):
                rt._refuse_initialization(RuntimeError("Failed to create context virtual (0xc01e0009)"))
            rt = bare_runtime(allow)
            with self.assertRaises(E.EnvironmentProblem):
                rt._refuse_initialization(ImportError("DLL load failed while importing pyxrt: xrt_coreutil"))

    def test_harness_default_is_allow(self):
        import inspect

        sig = inspect.signature(default83.DefaultRuntime.__init__)
        self.assertIs(sig.parameters["allow_fallback"].default, True)  # gates that build DefaultRuntime directly keep their R83 semantics

    def test_bad_language_is_an_error_not_a_downgrade(self):
        class Drafter:
            def select_language(self, lang):
                raise ValueError("target_language must explicitly be ja or en")

        class Prim:
            def check(self):
                pass

            def quiesce(self):
                pass

        for allow, expect_raise in ((False, True), (True, False)):
            rt = bare_runtime(allow)
            rt.drafter = Drafter()
            rt.session = type("S", (), {"max_seq": 2048, "_token": staticmethod(lambda t: None)})()
            rt.env = type("En", (), {"lease": type("L", (), {"require_validated": staticmethod(lambda: None)})()})()
            rt.primer = Prim()
            try:
                rt._generate([1, 2, 3], 4, None, target_language="xx")
            except ValueError as e:
                self.assertTrue(expect_raise)
                self.assertEqual(rt.mode, "w4-m2")  # the mode is never changed by a refused request
                self.assertIn("ja or en", str(e))
            except Exception:
                # allow=True keeps the R83 sticky fallback semantics and then runs the serial path, which needs a device
                self.assertFalse(expect_raise)
                self.assertEqual(rt.mode, "w4-serial")


class FakeTimer:
    def sleep_ns(self, n):
        pass

    def close(self):
        pass


class Spec:
    def __init__(self, done=False):
        self.d = done

    def prepare(self):
        pass

    def done(self):
        return self.d


def bare_hybrid(threshold=hybridwait83.BREAKER_THRESHOLD):
    hw = object.__new__(hybridwait83.HybridWait)
    hw.delta_ns = 1_500_000
    hw.log = None
    hw.logs = []
    hw.active = True
    hw.hist = collections.defaultdict(lambda: collections.deque(maxlen=16))
    hw.stats = collections.Counter()
    hw.consec = collections.Counter()
    hw.disabled = {}
    hw.breaker_threshold = threshold
    hw.timer = FakeTimer()
    hw.specs = {"verify2": Spec(False)}
    hw.current = "verify2"
    hw.log = hw.logs.append
    return hw


def pending(hw, g="verify2", est_ns=1000):
    hw.hist[g].extend([est_ns] * 4)
    return dict(graph=g, spec=hw.specs[g], record=False, t1=time.perf_counter_ns())


class Breaker(unittest.TestCase):
    def test_four_consecutive_timeouts_disable_the_graph(self):
        hw = bare_hybrid()
        for i in range(3):
            hw._hybrid(pending(hw))
            self.assertNotIn("verify2", hw.disabled)
        self.assertIsNotNone(hw._begin())
        hw._hybrid(pending(hw))
        self.assertIn("verify2", hw.disabled)
        self.assertIsNone(hw._begin())  # ordinary wait only from now on
        self.assertEqual(hw.stats["breaker_trips"], 1)
        self.assertEqual(len(hw.logs), 1)
        self.assertIn("disabled for verify2 after 4 consecutive timeouts", hw.logs[0])
        self.assertEqual(hw.report()["disabled_graphs"], ["verify2"])

    def test_success_resets_the_counter(self):
        hw = bare_hybrid()
        for i in range(3):
            hw._hybrid(pending(hw))
        hw.specs["verify2"].d = True
        hw._hybrid(pending(hw))
        self.assertEqual(hw.consec["verify2"], 0)
        hw.specs["verify2"].d = False
        for i in range(3):
            hw._hybrid(pending(hw))
        self.assertNotIn("verify2", hw.disabled)

    def test_other_graphs_unaffected(self):
        hw = bare_hybrid()
        hw.specs["decode"] = Spec(True)
        for i in range(4):
            hw._hybrid(pending(hw))
        hw.current = "decode"
        self.assertIsNotNone(hw._begin())

    def test_empty_history_does_not_raise(self):
        hw = bare_hybrid()
        hw._hybrid(dict(graph="verify2", spec=hw.specs["verify2"], record=False, t1=time.perf_counter_ns()))
        self.assertEqual(hw.stats["errors"], 0)
        self.assertEqual(hw.stats["fallbacks"], 1)

    def test_threshold_variants(self):
        for threshold, trips_after in ((1, 1), (2, 2)):
            hw = bare_hybrid(threshold)
            for i in range(trips_after):
                hw._hybrid(pending(hw))
            self.assertIn("verify2", hw.disabled, threshold)
        hw = bare_hybrid(1000)
        for i in range(8):
            hw._hybrid(pending(hw))
        self.assertNotIn("verify2", hw.disabled)

    def test_added_latency_is_bounded(self):
        hw = bare_hybrid()
        t = time.perf_counter()
        for i in range(10):
            p = pending(hw)
            if hw._begin() is None:
                break
            hw._hybrid(p)
        elapsed = time.perf_counter() - t
        # <= 4 timeouts x (3 x 1 us + 5 ms) plus scheduling slack
        self.assertLess(elapsed, 0.2)

    def test_env_garbage_is_noted_not_silent(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertTrue(hybridwait83.enabled({"PLAMO3_HYBRID_WAIT": "garbage-xyz"}))
        self.assertIn("not recognised", err.getvalue())
        for v in ("0", "off", "false", "no"):
            self.assertFalse(hybridwait83.enabled({"PLAMO3_HYBRID_WAIT": v}))
        for v in ("1", "on", "true", "yes"):
            self.assertTrue(hybridwait83.enabled({"PLAMO3_HYBRID_WAIT": v}))
        self.assertTrue(hybridwait83.enabled({}))


@unittest.skipUnless(IS_WINDOWS, "parent watch uses kernel32")
class WorkerParentWatch(TmpDirMixin, unittest.TestCase):
    def function_source(self):
        import ast

        src = (R83 / "prime_worker83.py").read_text(encoding="utf-8")
        node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "parent_watch"][0]
        return ast.get_source_segment(src, node)

    def test_child_exits_when_launcher_is_killed(self):
        fn = self.function_source()
        child = self.tmp / "child.py"
        child.write_text("import os,sys,time,ctypes\n" + fn + textwrap.dedent("""
            alive=parent_watch()
            open(sys.argv[1],'w').write(str(os.getpid()))
            t0=time.time()
            while alive() and time.time()-t0<30:time.sleep(0.05)
            open(sys.argv[1]+'.done','w').write('stopped after %.2f'%(time.time()-t0))
        """))
        launcher = self.tmp / "launcher.py"
        launcher.write_text(textwrap.dedent("""
            import os,subprocess,sys,time
            subprocess.Popen([getattr(sys,"_base_executable",sys.executable),sys.argv[1],sys.argv[2]])  # as primer83 does: the base interpreter, no launcher stub in between
            print(os.getpid(),flush=True)
            time.sleep(60)
        """))
        marker = self.tmp / "m"
        p = subprocess.Popen([sys.executable, str(launcher), str(child), str(marker)], stdout=subprocess.PIPE, text=True)
        launcher_pid = int(p.stdout.readline())  # the interpreter's own pid (the venv python.exe is only a launcher stub)
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.05)
        self.assertTrue(marker.exists())
        kill = subprocess.run(["taskkill", "/F", "/PID", str(launcher_pid)], capture_output=True, text=True)  # launcher only, no /T
        self.assertEqual(kill.returncode, 0, kill.stdout + kill.stderr)
        done = self.tmp / "m.done"
        for _ in range(300):
            if done.exists():
                break
            time.sleep(0.05)
        self.assertTrue(done.exists(), "child outlived its launcher")
        self.assertLess(float(done.read_text().split()[-1]), 15.0)


if __name__ == "__main__":
    unittest.main()
