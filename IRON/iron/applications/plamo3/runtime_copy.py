# SPDX-License-Identifier: Apache-2.0
"""Bounded CPython buffer copies that release the GIL inside libc memmove."""

import ctypes as c


class Buffer(c.Structure):
    _fields_ = [
        ("buf", c.c_void_p),
        ("obj", c.c_void_p),
        ("length", c.c_ssize_t),
        ("itemsize", c.c_ssize_t),
        ("readonly", c.c_int),
        ("ndim", c.c_int),
        ("format", c.c_void_p),
        ("shape", c.c_void_p),
        ("strides", c.c_void_p),
        ("suboffsets", c.c_void_p),
        ("internal", c.c_void_p),
    ]


_get = c.pythonapi.PyObject_GetBuffer
_get.argtypes = [c.py_object, c.POINTER(Buffer), c.c_int]
_get.restype = c.c_int
_release = c.pythonapi.PyBuffer_Release
_release.argtypes = [c.POINTER(Buffer)]
_release.restype = None


def copy_into(destination, source):
    """Copy equal-sized contiguous buffers; keep both exports until C returns."""
    dst, src = Buffer(), Buffer()
    _get(destination, c.byref(dst), 1)  # PyBUF_WRITABLE
    try:
        _get(source, c.byref(src), 0)  # PyBUF_SIMPLE, supports readonly mmap
        try:
            if dst.length != src.length:
                raise ValueError("copy buffer lengths differ")
            if src.length:
                c.memmove(dst.buf, src.buf, src.length)
        finally:
            _release(c.byref(src))
    finally:
        _release(c.byref(dst))
