"""NPU lock: named mutex, clean busy error, abandoned mutex after a kill, wait semantics (Windows; other systems get the no-op lock)."""
import json
import os
import subprocess
import sys
import textwrap
import time
import unittest
import uuid

from _common import IS_WINDOWS, APP, TmpDirMixin
import plamo_lock
import plamo_errors as E

HOLDER = textwrap.dedent("""
    import sys,time
    sys.path.insert(0,sys.argv[1])
    import plamo_lock
    lock=plamo_lock.acquire(sys.argv[4],name=sys.argv[2],sidecar=sys.argv[3],argv0='holder-test')
    import os
    print('HELD',os.getpid(),flush=True)
    time.sleep(float(sys.argv[5]))
""")
TRY = textwrap.dedent("""
    import sys,time
    sys.path.insert(0,sys.argv[1])
    import plamo_lock,plamo_errors
    t=time.time()
    try:
        lock=plamo_lock.acquire('one-shot',name=sys.argv[2],sidecar=sys.argv[3],wait_s=float(sys.argv[4]))
        print('ACQUIRED abandoned=%s waited=%.2f'%(lock.abandoned,time.time()-t))
    except plamo_errors.NpuBusy as e:
        print('BUSY code=%d waited=%.2f msg=%s'%(e.code,time.time()-t,e.message))
        sys.exit(e.code)
""")


@unittest.skipUnless(IS_WINDOWS, "named mutex test is Windows only")
class Lock(TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.name = "PLaMo3-test-" + uuid.uuid4().hex[:12]
        self.side = str(self.tmp / "npu.lock.json")
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        super().tearDown()

    def hold(self, seconds=30, mode="w4-m2"):
        p = subprocess.Popen([sys.executable, "-c", HOLDER, str(APP), self.name, self.side, mode, str(seconds)], stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        line = p.stdout.readline().split()
        self.assertEqual(line[0], "HELD")
        p.real_pid = int(line[1])  # on Windows the venv python.exe is a launcher: Popen.pid is not the interpreter's pid
        return p

    def attempt(self, wait):
        return subprocess.run([sys.executable, "-c", TRY, str(APP), self.name, self.side, str(wait)], capture_output=True, text=True, timeout=60)

    def test_free_lock_is_acquired_fast(self):
        r = self.attempt(0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ACQUIRED abandoned=False", r.stdout)

    def test_second_process_gets_clean_busy_error(self):
        holder = self.hold(mode="service")
        r = self.attempt(0.5)
        self.assertEqual(r.returncode, E.NPU, r.stdout + r.stderr)
        self.assertIn("BUSY code=5", r.stdout)
        self.assertIn("another PLaMo process (pid %d" % holder.real_pid, r.stdout)
        self.assertIn("mode service", r.stdout)
        self.assertNotIn("Traceback", r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("\n"), 1)
        waited = float(r.stdout.split("waited=")[1].split()[0])
        self.assertGreaterEqual(waited, 0.45)
        self.assertLess(waited, 3.0)

    def test_wait_then_run_when_holder_exits(self):
        self.hold(seconds=1.5)
        r = self.attempt(15)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        waited = float(r.stdout.split("waited=")[1])
        self.assertGreater(waited, 0.8)
        self.assertLess(waited, 10)

    def test_killed_holder_leaves_abandoned_mutex_that_is_acquired(self):
        holder = self.hold()
        subprocess.run(["taskkill", "/F", "/PID", str(holder.real_pid)], capture_output=True)
        holder.wait()
        r = self.attempt(5)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ACQUIRED", r.stdout)

    def test_sidecar_describes_holder_and_is_replaced(self):
        holder = self.hold(mode="w4-serial")
        info = json.loads(open(self.side, encoding="utf-8").read())
        self.assertEqual(info["pid"], holder.real_pid)
        self.assertEqual(info["mode"], "w4-serial")

    def test_stale_sidecar_does_not_block(self):
        with open(self.side, "w", encoding="utf-8") as f:
            json.dump(dict(pid=999999, started="00:00:00", mode="x"), f)
        r = self.attempt(0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_busy_message_names_dead_holder_honestly(self):
        info = dict(pid=999999, started="00:00:00", mode="x", argv0="")
        self.assertIn("not running any more", plamo_lock.holder_text(info))
        self.assertEqual(plamo_lock.holder_text(None), "another PLaMo process")

    def test_release(self):
        lock = plamo_lock.acquire("m", name=self.name, sidecar=self.side, wait_s=0)
        lock.release()
        r = self.attempt(0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


@unittest.skipIf(IS_WINDOWS, "non-Windows: no-op lock")
class NoopLock(unittest.TestCase):
    def test_noop(self):
        self.assertIsNotNone(plamo_lock.acquire("m"))


if __name__ == "__main__":
    unittest.main()
