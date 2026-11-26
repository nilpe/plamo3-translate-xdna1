# SPDX-License-Identifier: Apache-2.0
"""One PLaMo process at a time on the NPU: a named kernel mutex (ctypes kernel32, no new executable).

The mutex is held for the whole process lifetime (one-shot or --service) and released by the OS at exit, including after a kill
(the next owner then sees WAIT_ABANDONED, which counts as acquired). A second process waits up to `wait_s` seconds and then raises NpuBusy
with a one-line message naming the holder (read from a best-effort sidecar file). Helper processes (srccheck, priming worker, keep-warm,
turbo release) never take the lock: none of them touches the device.
Non-Windows: no-op lock (CPU tests).
"""
import json
import os
import sys
import time
from pathlib import Path

from plamo_errors import NpuBusy

NAME = "PLaMo3-XDNA1-NPU"
WAIT_OBJECT_0 = 0
WAIT_ABANDONED = 0x80
WAIT_TIMEOUT = 0x102
WAIT_FAILED = 0xFFFFFFFF
ERROR_ACCESS_DENIED = 5
DEFAULT_WAIT_S = 10.0


def sidecar_path():
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "plamo3-npu" / "npu.lock.json"
    return Path.home() / ".plamo3-npu" / "npu.lock.json"


def _api():
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    k.CreateMutexW.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.ReleaseMutex.argtypes = [wintypes.HANDLE]
    k.ReleaseMutex.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    return ctypes, k


def pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        ctypes, k = _api()
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            from ctypes import wintypes

            code = wintypes.DWORD()
            k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            if k.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == 259  # STILL_ACTIVE
            return True
        finally:
            k.CloseHandle(h)
    except Exception:
        return True


def read_holder(path=None):
    try:
        info = json.loads(Path(path or sidecar_path()).read_text(encoding="utf-8-sig"))
        if isinstance(info, dict) and isinstance(info.get("pid"), int):
            return info
    except (OSError, ValueError):
        pass
    return None


def holder_text(info):
    if not info:
        return "another PLaMo process"
    started = info.get("started", "?")
    mode = info.get("mode", "?")
    argv = info.get("argv0", "")
    alive = "" if pid_alive(info["pid"]) else " (not running any more?)"
    return f"another PLaMo process (pid {info['pid']}, started {started}, mode {mode}{' ' + argv if argv else ''}){alive}"


class NpuLock:
    def __init__(self, handle, scope, abandoned, sidecar):
        self.handle, self.scope, self.abandoned, self.sidecar = handle, scope, abandoned, sidecar
        self.owner_thread = None

    def release(self):
        h, self.handle = self.handle, None
        if h is not None:
            try:
                ctypes, k = _api()
                k.ReleaseMutex(h)
                k.CloseHandle(h)
            except Exception:
                pass
        try:
            if self.sidecar is not None:
                info = read_holder(self.sidecar)
                if info and info.get("pid") == os.getpid():
                    Path(self.sidecar).unlink()
        except OSError:
            pass


class _NullLock(NpuLock):
    def __init__(self):
        super().__init__(None, "none", False, None)


def _write_sidecar(path, mode, argv0):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = dict(pid=os.getpid(), started=time.strftime("%H:%M:%S"), mode=mode, argv0=argv0)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def acquire(mode="", *, wait_s=DEFAULT_WAIT_S, name=NAME, sidecar=None, argv0="", poll_s=0.05):
    """Acquire the NPU lock for this process. Raises NpuBusy after `wait_s` seconds. Must be called on the thread that lives as long as the
    process (the main thread): a mutex is owned by its acquiring thread."""
    if os.name != "nt":
        return _NullLock()
    sidecar = sidecar if sidecar is not None else sidecar_path()
    ctypes, k = _api()
    handle, scope = None, "Global"
    for prefix in ("Global\\", "Local\\"):
        handle = k.CreateMutexW(None, False, prefix + name)
        if handle:
            scope = prefix[:-1]
            break
        if ctypes.get_last_error() != ERROR_ACCESS_DENIED:
            break
    if not handle:
        # The lock itself is unavailable: never block a translation because of that, but say so once.
        from plamo_errors import note

        note(f"PLaMo: note: the NPU lock could not be created (error {ctypes.get_last_error()}); continuing without it.")
        return _NullLock()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    abandoned = False
    while True:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        rc = k.WaitForSingleObject(handle, min(remaining_ms, 200))
        if rc in (WAIT_OBJECT_0, WAIT_ABANDONED):
            abandoned = rc == WAIT_ABANDONED
            break
        if rc == WAIT_FAILED:
            k.CloseHandle(handle)
            return _NullLock()
        if time.monotonic() >= deadline:
            holder = holder_text(read_holder(sidecar))
            k.CloseHandle(handle)
            raise NpuBusy(
                f"{holder} is using the NPU. Wait for it to finish, close it, or use --wait <seconds>."
                if wait_s
                else f"{holder} is using the NPU."
            )
    _write_sidecar(sidecar, mode, argv0)
    lock = NpuLock(handle, scope, abandoned, sidecar)
    return lock
