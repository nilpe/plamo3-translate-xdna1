"""Turbo robustness (BUG 7) with the fake xrt-smi of the turbo gates: no real power-mode change is ever made (PLAMO3_TURBO_TEST=1 + fake)."""
import json
import os
import subprocess
import sys
import textwrap
import time
import unittest

from _common import R83, IS_WINDOWS, TmpDirMixin

FAKE = str(R83 / "fake_xrt_smi83.py")


class Base(TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.fake = self.tmp / "fake"
        self.state = self.tmp / "state"
        self.fake.mkdir()
        self.env = dict(
            os.environ,
            PLAMO3_TURBO_TEST="1",
            PLAMO3_XRT_SMI=FAKE,
            FAKE_XRT_DIR=str(self.fake),
            PLAMO3_TURBO_STATE_DIR=str(self.state),
        )
        for k in ("FAKE_XRT_DELAY_MS", "FAKE_XRT_FAIL_CONFIGURE", "FAKE_XRT_FAIL_EXAMINE", "PLAMO3_TURBO_WATCHDOG", "PLAMO3_TURBO_TEST_TIMEOUT_S"):
            self.env.pop(k, None)

    def fake_mode(self):
        try:
            return (self.fake / "mode.txt").read_text().strip()
        except OSError:
            return "default"

    def script(self, body, env=None, timeout=60, **kw):
        code = "import sys,time,os\nsys.path.insert(0,%r)\nimport turbo83\n" % str(R83) + textwrap.dedent(body)
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env or self.env, timeout=timeout, **kw)

    def wait_watchdog(self, real_pid, timeout=30):
        """wait until the restore watchdog of `real_pid` is running (its command line carries --watch <pid>)"""
        ps1 = self.tmp / "w.ps1"
        ps1.write_text("$n=0; Get-CimInstance Win32_Process | ForEach-Object { if ($_.CommandLine -and $_.CommandLine -like '*turbo83.py*--watch*%s*') { $n++ } }; Write-Output $n" % real_pid)
        end = time.time() + timeout
        while time.time() < end:
            out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], capture_output=True, text=True).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return True
            time.sleep(0.5)
        return False

    def wait_mode(self, want, timeout=12):
        end = time.time() + timeout
        while time.time() < end:
            if self.fake_mode() == want:
                return True
            time.sleep(0.1)
        return False


class TurboLogic(Base):
    def test_normal_run_restores(self):
        r = self.script("""
            with turbo83.session(True) as s:
                s.wait_switched()
                print('mode-in-session', open(os.path.join(os.environ['FAKE_XRT_DIR'],'mode.txt')).read())
            """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("mode-in-session turbo", r.stdout)
        self.assertTrue(self.wait_mode("default"), "pmode not restored")

    def test_hung_xrt_smi_does_not_delay_exit(self):
        env = dict(self.env, FAKE_XRT_DELAY_MS="60000", PLAMO3_TURBO_TEST_TIMEOUT_S="1")
        t = time.time()
        r = self.script("""
            with turbo83.session(True) as s:
                s.wait_switched(0.5)
                print('translation done',flush=True)
            """, env=env, timeout=60)
        elapsed = time.time() - t
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertLess(elapsed, 12.0, "exit took %.1f s (was 45 s)" % elapsed)
        self.assertIn("translation done", r.stdout)

    def test_slow_xrt_smi_is_reported_as_too_late(self):
        env = dict(self.env, FAKE_XRT_DELAY_MS="1200", PLAMO3_TURBO_WATCHDOG="off")
        r = self.script("""
            with turbo83.session(True) as s:
                ok = s.wait_switched(0.3)
                print('barrier-ok', ok, 'too_late', s.too_late)
                time.sleep(4)
                print('stats', s.stats.get('too_late'))
            """, env=env, timeout=90)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("barrier-ok False too_late True", r.stdout)
        self.assertIn("this run uses the current power mode", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertTrue(self.wait_mode("default", 30), "mode must be restored after a too-late switch")

    def test_unable_to_switch_message_mentions_elevation(self):
        env = dict(self.env, FAKE_XRT_FAIL_CONFIGURE="1", PLAMO3_TURBO_WATCHDOG="off")
        r = self.script("""
            with turbo83.session(True) as s:
                s.wait_switched()
            """, env=env)
        self.assertIn("could not set NPU power mode turbo", r.stderr)
        self.assertIn("elevated (administrator) PowerShell", r.stderr)
        self.assertEqual(self.fake_mode(), "default")

    def test_xrt_smi_missing_message(self):
        env = dict(self.env, PLAMO3_XRT_SMI=str(self.tmp / "does-not-exist.exe"))
        r = self.script("""
            with turbo83.session(True) as s:
                s.wait_switched()
            """, env=env)
        self.assertIn("xrt-smi.exe not found at", r.stderr)
        self.assertIn("driver is missing or too old", r.stderr)

    def write_marker(self, holders):
        self.state.mkdir(exist_ok=True)
        (self.state / "turbo-state.json").write_text(json.dumps(dict(version=1, previous_mode="default", created=time.time(), holders=holders)))
        (self.fake / "mode.txt").write_text("turbo")

    def test_corrupt_marker_holders_are_healed(self):
        self.write_marker(["x", {"pid": "a"}, 5, None, {"pid": 99999999, "start": "1"}])
        r = self.script("print(turbo83.stale_check())")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("True", r.stdout)
        self.assertNotIn("stale-marker check failed", r.stderr)
        self.assertEqual(self.fake_mode(), "default")
        self.assertFalse((self.state / "turbo-state.json").exists(), "marker must be removed (no warning forever)")

    def test_garbage_marker_file_is_removed(self):
        self.state.mkdir(exist_ok=True)
        (self.state / "turbo-state.json").write_text("{{{ not json")
        r = self.script("print(turbo83.stale_check())")
        self.assertNotIn("failed", r.stderr)
        self.assertFalse((self.state / "turbo-state.json").exists())

    def test_marker_of_non_dict_holders_container(self):
        self.state.mkdir(exist_ok=True)
        (self.state / "turbo-state.json").write_text(json.dumps(dict(holders="oops", previous_mode="default")))
        r = self.script("print(turbo83.stale_check())")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.state / "turbo-state.json").exists())

    def test_default_xrt_smi_from_system_root(self):
        self.assertTrue(turbo83_default().lower().endswith("system32\\amd\\xrt-smi.exe") or not IS_WINDOWS)


def turbo83_default():
    import turbo83

    return turbo83.DEFAULT_XRT_SMI


@unittest.skipUnless(IS_WINDOWS, "process-tree kill test is Windows only")
class TurboKill(Base):
    PARENT = """
        import subprocess
        # the app process: a child python that holds the turbo session and then waits
        child = subprocess.Popen([sys.executable,'-c',%r])
        print(child.pid, flush=True)
        time.sleep(120)
    """
    APP = (
        "import sys,time,os\n"
        "sys.path.insert(0,%r)\n"
        "import turbo83\n"
        "s=turbo83.session(True).start()\n"
        "s.wait_switched()\n"
        "print('SWITCHED',os.getpid(),flush=True)\n"
        "time.sleep(120)\n"
    )

    def test_tree_kill_still_restores_the_mode(self):
        app = self.APP % str(R83)
        p = subprocess.Popen([sys.executable, "-c", app], env=self.env, stdout=subprocess.PIPE, text=True)
        try:
            word, real = p.stdout.readline().split()
            self.assertEqual(word, "SWITCHED")
            self.assertEqual(self.fake_mode(), "turbo")
            self.assertTrue(self.wait_watchdog(real), "watchdog was not started")
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)  # tree kill of the launcher stub: the interpreter and everything below it
            subprocess.run(["taskkill", "/F", "/T", "/PID", real], capture_output=True)
            self.assertTrue(self.wait_mode("default", 25), "pmode stayed turbo after a tree kill")
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)

    def test_single_process_kill_restores_too(self):
        app = self.APP % str(R83)
        p = subprocess.Popen([sys.executable, "-c", app], env=self.env, stdout=subprocess.PIPE, text=True)
        try:
            word, real = p.stdout.readline().split()
            self.assertEqual(word, "SWITCHED")
            self.assertTrue(self.wait_watchdog(real), "watchdog was not started")
            subprocess.run(["taskkill", "/F", "/PID", real], capture_output=True)  # the interpreter only, no /T
            self.assertTrue(self.wait_mode("default", 25), "pmode stayed turbo after a kill")
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)

    def test_watchdog_exits_when_marker_gone(self):
        env = dict(self.env)
        r = self.script("""
            with turbo83.session(True) as s:
                s.wait_switched()
                time.sleep(1.5)
            """, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self.wait_mode("default", 15))
        time.sleep(2.5)
        ps1 = self.tmp / "q.ps1"
        ps1.write_text("$n=0; Get-CimInstance Win32_Process | ForEach-Object { if ($_.CommandLine -and $_.CommandLine -like '*turbo83.py*--watch*' -and $_.CommandLine -like '*" + str(self.tmp).replace("\\", "\\") + "*') { $n++ } }; Write-Output $n")
        ps1.write_text("$n=0; Get-CimInstance Win32_Process | ForEach-Object { if ($_.CommandLine -and $_.CommandLine -like '*turbo83.py*--watch*') { $n++ } }; Write-Output $n")
        res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], capture_output=True, text=True)
        self.assertEqual(res.stdout.strip(), "0", "watchdog process left behind (rc %s, stderr %r)" % (res.returncode, res.stderr[:300]))


if __name__ == "__main__":
    unittest.main()
