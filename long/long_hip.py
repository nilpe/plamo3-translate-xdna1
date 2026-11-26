# SPDX-License-Identifier: Apache-2.0
"""HIP access for the iGPU prefill step: the driver's own amdhip64_6.dll through ctypes (no new executable), device detection, error mapping."""
import ctypes
import os
from pathlib import Path

from long_core import LongModeError, errors

HIP_ERR_OOM = 2


def system_dll():
    root = os.environ.get('SystemRoot') or os.environ.get('SYSTEMROOT') or 'C:\\Windows'
    return str(Path(root) / 'System32' / 'amdhip64_6.dll')


class Hip:
    """Thin ctypes wrapper. `lib` may be injected (unit tests)."""

    def __init__(self, dll=None, lib=None):
        if lib is not None:
            self.h = lib
        else:
            path = dll or system_dll()
            if not os.path.exists(path):
                raise LongModeError('long mode needs the AMD display driver\'s HIP runtime (%s), which was not found. Install or update the AMD Radeon driver (Adrenalin). The normal translation path is not affected.' % path)
            try:
                self.h = ctypes.WinDLL(path)
            except OSError as error:
                raise LongModeError('the AMD HIP runtime could not be loaded (%s). Install or update the AMD Radeon driver.' % ' '.join(str(error).split())[:120])
        self.mod = None
        self.funcs = {}
        self.check(self.h.hipSetDevice(0), 'hipSetDevice')

    @staticmethod
    def check(rc, what):
        if rc == 0:
            return
        if rc == HIP_ERR_OOM:
            raise LongModeError('the iGPU ran out of memory (%s). Close other programs that use the GPU or memory and try again.' % what)
        raise LongModeError('the iGPU reported an error (%s, code %d). Update the AMD Radeon driver; the normal translation path is not affected.' % (what, rc))

    def device_count(self):
        n = ctypes.c_int(0)
        self.check(self.h.hipGetDeviceCount(ctypes.byref(n)), 'hipGetDeviceCount')
        return n.value

    def device_name(self):
        buf = ctypes.create_string_buffer(256)
        self.check(self.h.hipDeviceGetName(buf, 256, 0), 'hipDeviceGetName')
        return buf.value.decode('ascii', 'replace')

    def mem_info(self):
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        self.check(self.h.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)), 'hipMemGetInfo')
        return free.value, total.value

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.h.hipMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)), 'hipMalloc')
        return p.value

    def h2d(self, dst, src_ptr, nbytes):
        self.check(self.h.hipMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src_ptr), ctypes.c_size_t(nbytes), 1), 'h2d')

    def d2h(self, dst_ptr, src, nbytes):
        self.check(self.h.hipMemcpy(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), 2), 'd2h')

    def memset(self, dst, val, nbytes):
        self.check(self.h.hipMemset(ctypes.c_void_p(dst), ctypes.c_int(val), ctypes.c_size_t(nbytes)), 'hipMemset')

    def sync(self):
        self.check(self.h.hipDeviceSynchronize(), 'hipDeviceSynchronize')

    def load_module(self, path):
        data = Path(path).read_bytes()
        buf = ctypes.create_string_buffer(data, len(data))
        self.mod = ctypes.c_void_p()
        rc = self.h.hipModuleLoadData(ctypes.byref(self.mod), buf)
        if rc != 0:
            raise LongModeError('the iGPU rejected the shipped GPU code (%s, code %d): long mode supports the Radeon 780M/760M/740M (gfx1103) only.' % (Path(path).name, rc))
        self.funcs = {}

    def func(self, name):
        if name not in self.funcs:
            f = ctypes.c_void_p()
            self.check(self.h.hipModuleGetFunction(ctypes.byref(f), self.mod, name.encode()), 'hipModuleGetFunction(%s)' % name)
            self.funcs[name] = f
        return self.funcs[name]

    def launch(self, name, grid, block, args):
        objs = []
        for kind, val in args:
            if kind == 'p':
                objs.append(ctypes.c_void_p(val))
            elif kind == 'i':
                objs.append(ctypes.c_int(int(val)))
            elif kind == 'f':
                objs.append(ctypes.c_float(float(val)))
            else:
                raise ValueError(kind)
        ptrs = (ctypes.c_void_p * len(objs))(*[ctypes.cast(ctypes.pointer(o), ctypes.c_void_p) for o in objs])
        gx, gy = grid
        self.check(self.h.hipModuleLaunchKernel(self.func(name), gx, gy, 1, block, 1, 1, 0, None, ptrs, None), 'launch ' + name)


def detect(co_dir, need_mb=3200, hip=None):
    """Device check done before any large allocation: HIP runtime loads, one device, the smallest shipped gfx1103 code object loads (this is the
    architecture test: any other GPU architecture rejects it), and enough free device memory. Returns dict(name, free_mb, total_mb)."""
    hip = hip or Hip()
    if hip.device_count() < 1:
        raise LongModeError('no GPU was found by the HIP runtime. Long mode needs the integrated Radeon graphics (780M/760M/740M); check that it is enabled and the driver is installed.')
    name = hip.device_name()
    smoke = Path(co_dir) / 'attn2.co'
    if not smoke.exists():
        raise errors.InstallProblem('the installation is incomplete (missing %s). Run install.ps1 -Verify.' % smoke)
    hip.load_module(smoke)
    free, total = hip.mem_info()
    if free // (1 << 20) < need_mb:
        raise LongModeError('the iGPU has only %d MB of memory available, %d MB are needed. Close other programs that use the GPU.' % (free // (1 << 20), need_mb))
    return dict(name=name, free_mb=free // (1 << 20), total_mb=total // (1 << 20))
