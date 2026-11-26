"""Early idle worker, authorized package loading, bounded health and shutdown."""
import mmap
import ctypes
import struct
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from settings45 import paths
paths()
from primer import Primer as PreviousPrimer
from prime_worker import SIZE, REQUEST, SUBMIT, put

from primer45 import WorkerUnavailable  # shared loop exception identity

class Primer(PreviousPrimer):
    def __init__(self, bundle):
        self.bundle = str(bundle)
        self.tag = 'Local\\plamo3-r45-' + str(os.getpid()) + '-' + uuid.uuid4().hex
        self.arena = mmap.mmap(-1, SIZE, tagname=self.tag)
        self.pending = None
        self.job = self.toggle = self._health_ns = 0
        self.ready_record = self.last_taps = None
        self.records = []
        self.started_ns = time.perf_counter_ns()
        self.authorized = self.closed = False
        env = os.environ.copy()
        env['R42_BOOTSTRAP'] = 'fastnorm'
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            env[key] = '2'
        try:
            self.child = subprocess.Popen([getattr(sys, '_base_executable', sys.executable), '-u', str(Path(__file__).with_name('prime_worker83.py')),
                self.bundle, self.tag], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env, bufsize=0)
        except BaseException:
            self.arena.close()
            raise

    def healthy(self):
        return not self.closed and self.child.poll() is None

    def check(self):
        if not self.healthy():
            raise WorkerUnavailable('priming worker exited (code=' + str(self.child.poll()) + ')')

    def authorize(self, runtime):
        # No package reference code or weights are read by the child before this
        # barrier. The parent retains write/delete-excluded package handles.
        runtime._artifact_lease.require_validated()
        actual = Path(runtime.manifest['r40_cycle']['bundle'])
        if actual != Path(self.bundle):
            raise WorkerUnavailable('configured drafter package differs from validated package')
        from settings83 import PACKAGE_MANIFEST_SHA256
        if runtime.manifest['r40_cycle']['package_manifest_sha256'] != PACKAGE_MANIFEST_SHA256:
            raise WorkerUnavailable('validated package manifest is not the approved package')
        self.check()
        from prime_worker import send
        send(self.child.stdin, dict(kind='validated-package', manifest_sha256=runtime.manifest['r40_cycle']['package_manifest_sha256']))
        self.authorized = True
        self.authorized_ns = time.perf_counter_ns()

    def available(self):
        # Peek the complete frame before calling the inherited framed reader;
        # a child that exits mid-read cannot leave readiness blocked forever.
        import msvcrt
        head=ctypes.create_string_buffer(4)
        read=ctypes.c_ulong();total=ctypes.c_ulong()
        api=ctypes.windll.kernel32
        api.PeekNamedPipe.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p]
        ok=api.PeekNamedPipe(ctypes.c_void_p(msvcrt.get_osfhandle(self.child.stdout.fileno())),head,4,ctypes.byref(read),ctypes.byref(total),None)
        if not ok:raise WorkerUnavailable('priming worker readiness pipe closed')
        if read.value<4:return False
        size=struct.unpack('<I',head.raw)[0]
        if not 0<size<=16384:raise WorkerUnavailable('invalid worker readiness frame extent')
        return total.value>=4+size

    def ready(self, wait=True):
        if not self.authorized:
            raise WorkerUnavailable('package has not passed the retained validation barrier')
        if self.ready_record is not None:
            self.check()
            return True
        deadline = time.perf_counter() + 15
        while True:
            self.check()
            try:
                available = self.available()
            except (OSError, RuntimeError) as error:
                raise WorkerUnavailable(str(error)) from error
            if available:
                try:
                    result = super().ready(wait=False)
                except (EOFError, OSError, AssertionError, TypeError) as error:
                    raise WorkerUnavailable('invalid worker readiness response: ' + str(error)) from error
                if self.ready_record['pid'] != self.child.pid:
                    raise WorkerUnavailable('worker PID differs from directly owned process')
                return result
            if not wait:
                return False
            if time.perf_counter() >= deadline:
                raise WorkerUnavailable('priming worker initialization timed out')
            time.sleep(.001)

    def start(self, rows, tokens):
        self.check()
        return super().start(rows, tokens)

    def done(self):
        self.check()
        return super().done()

    def result(self):
        self.check()
        try:
            return super().result()
        except (RuntimeError, AssertionError, ValueError, EOFError, OSError) as error:
            raise WorkerUnavailable('priming worker result failed: ' + str(error)) from error

    def close(self):
        if self.closed:
            return
        # Closing never waits for a slow/failed priming job. No device work is
        # owned by the CPU-only child; process termination releases its maps.
        if self.child.poll() is None:
            self.child.terminate()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=5)
        for stream in (self.child.stdin, self.child.stdout):
            stream.close()
        self.pending = None
        self.arena.close()
        self.closed = True
