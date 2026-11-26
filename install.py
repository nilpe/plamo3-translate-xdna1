#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Installer for the PLaMo 3 Translate XDNA1-NPU runtime (standard library only).

Run it through install.ps1 (which finds a suitable Python) or directly with a 64-bit CPython 3.13:

    python install.py [--root DIR] [--accept-license] [--payload-source DIR] [--model-source DIR]
                      [--verify | --reattest | --selftest-only | --uninstall | --plan]

Every run resumes: files that already verify are not downloaded again. No administrator rights are needed. The
installer touches no registry, PATH, scheduled task or service; it only writes inside the install root. Network
access goes to exactly three places: the payload repository on huggingface.co, the base model repository on
huggingface.co, and PyPI (through pip, with pinned hashes).

Exit codes: 0 ok | 1 internal error | 2 usage | 3 installation does not verify | 5 no NPU / driver |
            6 environment (Windows, Python, disk, path) | 7 network | 8 licence not accepted |
            10+N self-test step N failed | 130 interrupted
"""
import argparse
import base64
import csv
import datetime
import hashlib
import json
import ntpath
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VERSION = '1.0.0'
EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_VERIFY, EXIT_NPU, EXIT_ENV, EXIT_NETWORK, EXIT_LICENSE = 0, 1, 2, 3, 5, 6, 7, 8
EXIT_SELFTEST_BASE = 10

MODEL_REPO = 'pfnet/plamo-3-translate-0.8b'
MODEL_REVISION = '8b4d42855622093e610ba9b90c2d2b59f8d5c15d'
MODEL_FILES_DEFAULT = ('config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.jsonl',
                       'tokenizer_config.json', 'special_tokens_map.json', 'tokenization_plamo.py',
                       'modeling_plamo.py', 'model.safetensors')
HF_HOST = 'https://huggingface.co'
GB = 1024 ** 3
MIN_WINDOWS_BUILD = 22000          # Windows 11
MAX_ROOT_LEN = 150
CHUNK = 512 * 1024 * 1024
TESTING = os.environ.get('PLAMO3_INSTALL_TESTING') == '1'     # lets the unit tests run the Windows-only steps on Linux

LICENSE_SUMMARY = """\
PLaMo 3 Translate runtime for the AMD XDNA1 NPU  -  licence summary (not a substitute for the full texts)

  * This software runs PLaMo 3 Translate 0.8B, a model of Preferred Networks, Inc. (PFN). The model and everything
    derived from it (the quantised weight images, embeddings, tokenizer table, drafter and compiled streams that this
    installer puts on your disk) are licensed under the PLaMo Community License. You must read it and agree to it:
        {license_en}
        {license_ja}   (Japanese original)
  * Personal, academic and other non-commercial use is free. If you use it for a COMMERCIAL PURPOSE (this includes
    use inside a company, products, services, hosting or APIs) YOU must register with PFN yourself before you start
    and you are subject to the licence's revenue condition. This project does not register you and does not grant
    commercial rights.
  * "Built with PLaMo". Unofficial community project: not affiliated with, endorsed or supported by Preferred
    Networks, Inc. or Advanced Micro Devices, Inc.
  * The code is Apache-2.0 (see LICENSE and NOTICE). Everything is provided "AS IS", without warranty of any kind.
  * No telemetry: this software never sends usage data anywhere.
"""


class Fail(Exception):
    def __init__(self, code, message, hint=None):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


# ------------------------------------------------------------------ logging

class Log:
    def __init__(self, quiet=False, path=None):
        self.quiet, self.path, self.lock = quiet, path, threading.Lock()
        self.step_no = 0

    def _write(self, text):
        if self.path:
            try:
                with self.lock:
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    with open(self.path, 'a', encoding='utf-8') as f:
                        f.write('%s %s\n' % (datetime.datetime.now().strftime('%H:%M:%S'), text))
            except OSError:
                pass        # logging must never stop an installation

    def step(self, name):
        self.step_no += 1
        self.info('[%d] %s' % (self.step_no, name))

    def info(self, text):
        self._write(text)
        if not self.quiet:
            print(text, flush=True)

    def warn(self, text):
        self._write('WARNING ' + text)
        print('  warning: ' + text, file=sys.stderr, flush=True)

    def ok(self, text):
        self.info('    ok: ' + text)


# ------------------------------------------------------------------ small helpers

def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def blob_sha1(path):
    """git blob id: lets small model files be checked against the Hugging Face tree listing."""
    size = os.path.getsize(path)
    h = hashlib.sha1(b'blob %d\0' % size)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fmt_gb(n):
    return '%.2f GB' % (n / 1e9)


def posix(p):
    return p.replace('\\', '/')


def native(root, rel):
    return os.path.join(root, *rel.split('/'))


def read_json(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp, path)


def parse_sums(path):
    sums = {}
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip('\r\n')
            if not line:
                continue
            m = re.match(r'^([0-9a-f]{64}) [ *](.+)$', line)
            if not m:
                raise Fail(EXIT_VERIFY, '%s line %d is not a sha256sum line' % (os.path.basename(path), n))
            rel = posix(m.group(2))
            if rel.startswith('/') or '..' in rel.split('/') or re.match(r'^[A-Za-z]:', rel):
                raise Fail(EXIT_VERIFY, 'unsafe path in SHA256SUMS: %s' % rel)
            sums[rel] = m.group(1)
    return sums


# ------------------------------------------------------------------ Windows facts (ctypes; stubs elsewhere)

def windows_build():
    if os.name == 'nt':
        v = sys.getwindowsversion()
        ubr = None
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows NT\CurrentVersion') as k:
                ubr = winreg.QueryValueEx(k, 'UBR')[0]
        except OSError:
            pass
        return dict(major=v.major, minor=v.minor, build=v.build, ubr=ubr)
    return None


def file_version(path):
    """Version resource (a.b.c.d) of a Windows binary, or None."""
    if os.name != 'nt' or not os.path.isfile(path):
        return None
    import ctypes
    from ctypes import wintypes
    ver = ctypes.WinDLL('version', use_last_error=True)
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(path, 0, size, buf):
        return None
    ptr, length = ctypes.c_void_p(), wintypes.UINT()
    if not ver.VerQueryValueW(buf, '\\', ctypes.byref(ptr), ctypes.byref(length)) or length.value < 52:
        return None
    ms, ls = struct.unpack_from('<II', ctypes.string_at(ptr.value, 16), 8)
    return '%02d.%02d.%d.%d' % (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def memory_gb():
    if os.name != 'nt':
        return None
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    st = MEMSTAT()
    st.dwLength = ctypes.sizeof(MEMSTAT)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullTotalPhys / GB, st.ullAvailPhys / GB


def volume_fs(path):
    if os.name != 'nt':
        return None
    import ctypes
    root = os.path.splitdrive(os.path.abspath(path))[0] + '\\'
    name = ctypes.create_unicode_buffer(64)
    if ctypes.windll.kernel32.GetVolumeInformationW(root, None, 0, None, None, None, name, 64):
        return name.value
    return None


def system_dir():
    return os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')


def npu_service_present():
    if os.name != 'nt':
        return False
    import winreg
    for svc in ('IpuMcdmDriver', 'Npu2McdmDriver'):
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\\' + svc))
            return True
        except OSError:
            continue
    return False


def run_powershell_json(script, timeout=60):
    exe = shutil.which('powershell') or shutil.which('pwsh')
    if not exe:
        return None
    try:
        r = subprocess.run([exe, '-NoProfile', '-NonInteractive', '-Command', script], capture_output=True, timeout=timeout)
        out = r.stdout.decode('utf-8', 'replace').strip()
        return json.loads(out) if out else None
    except Exception:
        return None


def npu_driver_info():
    """Driver name/version of the NPU device, or None (best effort, never raises)."""
    data = run_powershell_json(
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        "Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -match '(^|\\s)NPU(\\s|$)' } |"
        " Select-Object DeviceName,DriverVersion,DriverDate | ConvertTo-Json -Compress")
    if isinstance(data, dict):
        data = [data]
    return data


# ------------------------------------------------------------------ NTFS identity (port of the runtime's file_identity)

_NT_API = None


def _api():
    global _NT_API
    if _NT_API is None:
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt
        k = c.WinDLL('kernel32', use_last_error=True)
        k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
        k.CreateFileW.restype = w.HANDLE
        k.GetFileInformationByHandle.argtypes = [w.HANDLE, c.c_void_p]
        k.GetFileInformationByHandle.restype = w.BOOL
        k.GetFileInformationByHandleEx.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        k.GetFileInformationByHandleEx.restype = w.BOOL
        k.DeviceIoControl.argtypes = [w.HANDLE, w.DWORD, c.c_void_p, w.DWORD, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p]
        k.DeviceIoControl.restype = w.BOOL
        k.CloseHandle.argtypes = [w.HANDLE]
        _NT_API = c, w, k, msvcrt
    return _NT_API


def open_retained(path):
    """Read handle that denies concurrent writers and deleters (like the runtime's own retained handle)."""
    if os.name != 'nt':
        return open(path, 'rb')
    c, w, k, msvcrt = _api()
    handle = k.CreateFileW(path, 0x80000000, 0x00000001, None, 3, 0x80, None)    # GENERIC_READ, SHARE_READ, OPEN_EXISTING
    if handle == c.c_void_p(-1).value:
        raise c.WinError(c.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        k.CloseHandle(handle)
        raise
    return os.fdopen(fd, 'rb')


def _ioctl(handle, code):
    c, w, k, _ = _api()
    out, count = c.create_string_buffer(4096), w.DWORD()
    if not k.DeviceIoControl(handle, code, None, 0, out, len(out), c.byref(count), None):
        raise c.WinError(c.get_last_error())
    return out.raw[:count.value]


def file_identity(stream):
    """Same fields, same meaning as the runtime's runtime_identity.file_identity (see SECURITY.md)."""
    c, _, k, msvcrt = _api()
    handle = msvcrt.get_osfhandle(stream.fileno())
    info, fileid = c.create_string_buffer(52), c.create_string_buffer(24)
    if not k.GetFileInformationByHandle(handle, info):
        raise c.WinError(c.get_last_error())
    if not k.GetFileInformationByHandleEx(handle, 18, fileid, 24):
        raise c.WinError(c.get_last_error())
    fields = struct.unpack('<13I', info.raw)
    record = _ioctl(handle, 0x900EB)          # FSCTL_READ_FILE_USN_DATA
    if len(record) < 60:
        raise OSError('truncated USN record')
    version = struct.unpack_from('<H', record, 4)[0]
    if version not in (2, 3) or (version == 3 and len(record) < 76):
        raise OSError('unsupported USN record version %d' % version)
    usn = struct.unpack_from('<q', record, 24 if version == 2 else 40)[0]
    journal = _ioctl(handle, 0x900F4)         # FSCTL_QUERY_USN_JOURNAL
    if len(journal) < 56 or usn <= 0:
        raise OSError('absent/invalid USN journal identity')
    journal_id, _, next_usn, lowest = struct.unpack_from('<Qqqq', journal)
    if not journal_id or usn >= next_usn or usn < lowest:
        raise OSError('file USN outside valid journal epoch')
    return dict(volume_serial=struct.unpack_from('<Q', fileid.raw)[0], file_id_128=fileid.raw[8:24].hex(),
                size=(fields[8] << 32) | fields[9], last_write_time=(fields[6] << 32) | fields[5],
                usn=usn, journal_id=journal_id)


def maybe_identity(stream):
    if os.name != 'nt':
        return None
    try:
        return file_identity(stream)
    except OSError:
        return None


IDENTITY_ROW = ('volume_serial', 'file_id_128', 'size', 'last_write_time', 'usn', 'journal_id')


# ------------------------------------------------------------------ SSH signature of SHA256SUMS

SSH_TIMEOUT_S = 60


def ssh_key_fingerprint(public_key):
    """'SHA256:<base64 without padding>' of the key blob of an OpenSSH public key line (pure Python; the comment field is ignored)."""
    parts = public_key.split()
    if len(parts) < 2 or not parts[0].startswith(('ssh-', 'ecdsa-', 'sk-')):
        raise ValueError('not an OpenSSH public key')
    blob = base64.b64decode(parts[1], validate=True)
    return 'SHA256:' + base64.b64encode(hashlib.sha256(blob).digest()).decode('ascii').rstrip('=')


SSH_BUILTIN_MAX_BYTES = 8000


def builtin_ssh_keygen_path(environ=os.environ):
    return ntpath.join(environ.get('SystemRoot', r'C:\\Windows'), 'System32', 'OpenSSH', 'ssh-keygen.exe')


def is_builtin_ssh_keygen(path, environ=os.environ):
    return ntpath.normcase(ntpath.abspath(path)) == ntpath.normcase(ntpath.abspath(builtin_ssh_keygen_path(environ)))


def find_ssh_keygen(which=shutil.which, isfile=os.path.isfile, environ=os.environ):
    """ssh-keygen to use. Order: Git for Windows' usr\\bin\\ssh-keygen.exe, any other ssh-keygen on PATH, the built-in Windows OpenSSH client last
    (that one hangs on -Y verify when the signed file is larger than ~8 KB, see SSH_BUILTIN_MAX_BYTES)."""
    cands = []
    git = which('git')
    if git:
        d = ntpath.dirname(git)
        root = ntpath.dirname(d) if ntpath.basename(d).lower() in ('cmd', 'bin') else d
        if ntpath.basename(root).lower() == 'mingw64' or ntpath.basename(d).lower() == 'mingw64':
            root = ntpath.dirname(root)
        cands.append(ntpath.join(root, 'usr', 'bin', 'ssh-keygen.exe'))
    pf = environ.get('ProgramFiles')
    if pf:
        cands.append(ntpath.join(pf, 'Git', 'usr', 'bin', 'ssh-keygen.exe'))
    other = which('ssh-keygen')
    if other and not is_builtin_ssh_keygen(other, environ):
        cands.append(other)
    cands.append(builtin_ssh_keygen_path(environ))
    for c in cands:
        if c and isfile(c):
            return c
    return None


def ssh_verify(sums, sig, signing, fingerprint=None, ssh_keygen=None):
    """Verify the detached SSH signature `sig` of the file `sums` against the key pinned in release.json ('signing': public_key, principal, namespace).
    Returns (status, message): status is 'ok', 'bad' or 'unavailable'. No network access; the only program run is ssh-keygen -Y verify."""
    signing = signing or {}
    key, principal, namespace = signing.get('public_key'), signing.get('principal'), signing.get('namespace')
    if not (key and principal and namespace):
        return 'unavailable', 'release.json pins no signing key'
    if not (re.fullmatch(r'[A-Za-z0-9._-]+', principal) and re.fullmatch(r'[A-Za-z0-9._-]+', namespace)):
        return 'bad', 'release.json: invalid principal or namespace'
    try:
        got = ssh_key_fingerprint(key)
    except ValueError as e:
        return 'bad', 'release.json: the pinned signing key is not valid (%s)' % e
    if signing.get('fingerprint') and signing['fingerprint'] != got:
        return 'bad', 'release.json: the pinned fingerprint %s does not match the pinned key (%s)' % (signing['fingerprint'], got)
    if fingerprint and re.sub(r'\s+', '', fingerprint) != got:
        return 'bad', 'the signing key pinned in release.json (%s) is not the one you pinned (%s)' % (got, re.sub(r'\s+', '', fingerprint))
    ssh_keygen = ssh_keygen or find_ssh_keygen()
    if not ssh_keygen:
        return 'unavailable', 'ssh-keygen not found (Windows optional feature "OpenSSH Client")'
    if is_builtin_ssh_keygen(ssh_keygen) and os.path.getsize(sums) > SSH_BUILTIN_MAX_BYTES:
        return 'unavailable', ('the Windows built-in ssh-keygen hangs on files larger than about 8 KB, so SHA256SUMS cannot be checked with it; '
                               'install Git for Windows (its ssh-keygen is used automatically) to check the signature')
    tmp = tempfile.mkdtemp(prefix='plamo-sig-')
    try:
        allowed = os.path.join(tmp, 'allowed_signers')
        with open(allowed, 'w', encoding='ascii', newline='\n') as f:
            f.write('%s namespaces="%s" %s\n' % (principal, namespace, ' '.join(key.split()[:2])))
        # stdin is the file itself and stdout/stderr are files: ssh-keygen for Windows can hang on pipes
        try:
            with open(sums, 'rb') as fin, open(os.path.join(tmp, 'out'), 'wb') as fo, open(os.path.join(tmp, 'err'), 'wb') as fe:
                r = subprocess.run([ssh_keygen, '-Y', 'verify', '-f', allowed, '-I', principal, '-n', namespace, '-s', sig], stdin=fin, stdout=fo, stderr=fe, timeout=SSH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return 'unavailable', 'ssh-keygen did not answer within %d s' % SSH_TIMEOUT_S
        except OSError as e:
            return 'unavailable', 'ssh-keygen could not be run (%s)' % e
        out = open(os.path.join(tmp, 'out'), 'rb').read().decode('utf-8', 'replace').strip()
        err = open(os.path.join(tmp, 'err'), 'rb').read().decode('utf-8', 'replace').strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode == 0 and out.startswith('Good "%s" signature for %s ' % (namespace, principal)):
        return 'ok', 'good signature by key %s (%s)' % (got, principal)
    if 'unknown option' in err.lower() or 'illegal option' in err.lower() or err.lower().startswith('usage:'):
        return 'unavailable', 'this ssh-keygen does not support -Y verify (OpenSSH 8.0 or newer is needed)'
    return 'bad', 'BAD signature on SHA256SUMS (%s)' % (' '.join((err or out or 'ssh-keygen exit %d' % r.returncode).split())[:200])


# ------------------------------------------------------------------ preflight

class Env:
    """Facts gathered by the preflight (also written to state/env-record.json)."""


def python_ok():
    return sys.implementation.cache_tag == 'cpython-313' and sys.maxsize > 2 ** 32


def preflight(root, plan, args, log, need_bytes):
    facts = dict(installer=VERSION, python=sys.version.split()[0], python_build=sys.version, python_cache_tag=sys.implementation.cache_tag,
                 python_executable=sys.executable, machine=os.environ.get('PROCESSOR_ARCHITECTURE', ''), checks=[])
    problems = []

    def check(name, ok, detail, code=None, hint=None):
        facts['checks'].append(dict(name=name, ok=ok, detail=detail))
        log.info('    %s %s: %s' % ('ok  ' if ok else 'FAIL', name, detail))
        if not ok:
            problems.append(Fail(code, '%s: %s' % (name, detail), hint))

    # Python
    check('python', python_ok() or TESTING, '%s (%s, %s-bit)' % (sys.version.split()[0], sys.implementation.cache_tag, 64 if sys.maxsize > 2 ** 32 else 32),
          EXIT_ENV, 'A 64-bit CPython 3.13 is required (the NPU module is built for it). Install one of: the python.org installer '
                    '(https://www.python.org/downloads/windows/, "Windows installer (64-bit)", no admin needed with "Install for me only"), '
                    '`winget install Python.Python.3.13`, or conda-forge `python=3.13`; then run install.ps1 again or pass -Python <path>.')
    # Windows
    wb = windows_build()
    facts['windows'] = wb
    if wb:
        check('windows', wb['build'] >= MIN_WINDOWS_BUILD, 'Windows build %d.%d.%d%s' % (wb['major'], wb['minor'], wb['build'], '' if wb['ubr'] is None else '.%d' % wb['ubr']),
              EXIT_ENV, 'Windows 11 (build 22000 or newer) is required.')
    elif TESTING:
        facts['checks'].append(dict(name='windows', ok=True, detail='skipped (testing)'))
    else:
        check('windows', False, 'not Windows', EXIT_ENV, 'This runtime is for Windows 11 on an AMD XDNA1 NPU PC.')
    if os.name == 'nt' and platform_machine() not in ('AMD64', 'x86_64'):
        check('cpu-arch', False, platform_machine(), EXIT_ENV, 'x64 required.')
    # NPU + driver
    sysdir = system_dir()
    coreutil = os.path.join(sysdir, 'xrt_coreutil.dll')
    facts['xrt_coreutil'] = dict(path=coreutil, version=file_version(coreutil))
    if not args.no_npu_check and not TESTING:
        have_dll = os.path.isfile(coreutil)
        have_svc = npu_service_present()
        check('npu-driver', have_dll and have_svc,
              'xrt_coreutil.dll %s (%s), NPU service %s' % ('found' if have_dll else 'MISSING', facts['xrt_coreutil']['version'] or '-',
                                                          'found' if have_svc else 'MISSING'),
              EXIT_NPU, 'No AMD NPU (XDNA) driver was detected. Install the AMD NPU driver (part of the AMD Ryzen AI Software / '
                        'NPU driver package from AMD), reboot, and check that "NPU Compute Accelerator Device" shows in Device Manager. '
                        'To install the files on a PC without an NPU anyway, pass -NoNpuCheck.')
        facts['npu_driver'] = npu_driver_info()
    # root path
    r = os.path.abspath(root)
    nonascii = any(ord(ch) > 127 for ch in r)
    check('path', len(r) < MAX_ROOT_LEN and not r.startswith('\\\\') and 'onedrive' not in r.lower() and not nonascii,
          '%s (%d characters%s%s)' % (r, len(r), ', spaces' if ' ' in r else '', ', non-ASCII' if nonascii else ''),
          EXIT_ENV, 'Choose a local, short, ASCII-only path (under %d characters; the NPU library XRT cannot open files below a folder with non-ASCII characters), not a network share and not a OneDrive-synced folder, e.g. C:\\PLaMo3-NPU.' % MAX_ROOT_LEN)
    fs = volume_fs(r)
    facts['volume_fs'] = fs
    if fs and fs.upper() != 'NTFS':
        log.warn('the install volume is %s, not NTFS: no file identity cache, every start hashes ~5.4 GB (10-20 s).' % fs)
    # disk space
    probe = r
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    check('disk', free >= need_bytes, '%.1f GB free, %.1f GB needed for what is still missing (plus margin)' % (free / GB, need_bytes / GB),
          EXIT_ENV, 'Free some disk space or choose another -Root.')
    # memory (advice only)
    mem = memory_gb()
    if mem:
        facts['memory_gb'] = dict(total=round(mem[0], 1), available=round(mem[1], 1))
        if mem[0] < 14.5:
            log.warn('%.1f GB RAM installed: the runtime is sized for 16 GB machines.' % mem[0])
        elif mem[1] < 6:
            log.warn('only %.1f GB RAM free right now; close other programs before running a translation (about 6 GB are needed).' % mem[1])
    if problems:
        raise problems[0]
    return facts


def platform_machine():
    import platform
    return platform.machine()


# ------------------------------------------------------------------ licence

def licence_step(root, args, log):
    en, ja = os.path.join(root, 'LICENSE-PLaMo-en.md'), os.path.join(root, 'LICENSE-PLaMo-ja.md')
    if not os.path.isfile(en):
        raise Fail(EXIT_VERIFY, 'LICENSE-PLaMo-en.md is missing from the install root', 'Run this from the cloned repository.')
    print(LICENSE_SUMMARY.format(license_en=en, license_ja=ja))
    state = os.path.join(root, 'state', 'license-accepted.json')
    digest = sha256_file(en)
    if os.path.isfile(state):
        try:
            if read_json(state).get('license_sha256') == digest:
                log.ok('licence already accepted (same text)')
                return
        except (ValueError, OSError):
            pass
    if not args.accept_license:
        if not sys.stdin or not sys.stdin.isatty():
            raise Fail(EXIT_LICENSE, 'the licence has not been accepted', 'Read the licence, then pass -AcceptLicense (install.ps1) / --accept-license.')
        answer = input('Type  I ACCEPT  to accept the PLaMo Community License and continue: ').strip()
        if answer != 'I ACCEPT':
            raise Fail(EXIT_LICENSE, 'licence not accepted; nothing was installed')
    write_json(state, dict(version=1, license_sha256=digest, accepted=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                           installer=VERSION, how='flag' if args.accept_license else 'typed'))
    log.ok('licence accepted (recorded in state/license-accepted.json)')


# ------------------------------------------------------------------ release metadata and signature

def load_release(root):
    for need in ('release.json', 'SHA256SUMS', 'PAYLOAD-MANIFEST.json'):
        if not os.path.isfile(os.path.join(root, need)):
            raise Fail(EXIT_VERIFY, '%s not found in %s' % (need, root),
                       'Run the installer from a clone of the release repository (it ships release.json, SHA256SUMS and PAYLOAD-MANIFEST.json).')
    return read_json(os.path.join(root, 'release.json')), read_json(os.path.join(root, 'PAYLOAD-MANIFEST.json')), parse_sums(os.path.join(root, 'SHA256SUMS'))


def signature_step(root, release, args, log):
    sums = os.path.join(root, 'SHA256SUMS')
    sig = args.sums_sig or os.path.join(root, 'SHA256SUMS.sig')
    if not os.path.isfile(sig):
        if args.require_signature:
            raise Fail(EXIT_VERIFY, 'SHA256SUMS.sig not found but a signature is required',
                       'Download SHA256SUMS.sig from the GitHub release page of this version into the install root.')
        log.info('    note: no SHA256SUMS.sig next to SHA256SUMS; the signature was not checked (optional, see INSTALL.md)')
        return 'absent'
    status, msg = ssh_verify(sums, sig, release.get('signing'), args.fingerprint)
    if status == 'ok':
        log.ok(msg + ('' if args.fingerprint else ' (compare the fingerprint with the one published by the maintainer, see SECURITY.md)'))
    elif status == 'bad':
        raise Fail(EXIT_VERIFY, msg, 'Do not continue. Download SHA256SUMS and SHA256SUMS.sig again from the official release page.')
    else:
        if args.require_signature:
            raise Fail(EXIT_VERIFY, 'signature required but cannot be checked: ' + msg)
        log.info('    note: signature present but not checked (%s)' % msg)
    return status


# ------------------------------------------------------------------ downloads

class Downloader:
    def __init__(self, log, retries=3, timeout=60):
        self.log, self.retries, self.timeout = log, retries, timeout

    def fetch(self, url, dest, size=None, headers=None):
        """Download url to dest through dest.part with Range resume; returns when dest exists with the expected size."""
        part = dest + '.part'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                have = os.path.getsize(part) if os.path.exists(part) else 0
                req = urllib.request.Request(url, headers=dict(headers or {}, **{'User-Agent': 'plamo3-npu-installer/%s' % VERSION}))
                if have:
                    req.add_header('Range', 'bytes=%d-' % have)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = getattr(resp, 'status', 200)
                    mode = 'ab'
                    if have and status != 206:
                        have, mode = 0, 'wb'          # server ignored the range: start over
                    with open(part, mode) as out:
                        shutil.copyfileobj(resp, out, 4 * 1024 * 1024)
                got = os.path.getsize(part)
                if size is not None and got != size:
                    if got > size:
                        os.remove(part)
                    raise OSError('size %d, expected %d' % (got, size))
                os.replace(part, dest)
                return
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 416 and os.path.exists(part):        # range beyond end: part is complete or corrupt
                    if size is not None and os.path.getsize(part) == size:
                        os.replace(part, dest)
                        return
                    os.remove(part)
                elif e.code in (401, 403, 404):
                    raise Fail(EXIT_NETWORK, 'HTTP %d for %s' % (e.code, url),
                               'The file is not available at this address (wrong revision, repository not public yet, or the file was removed).')
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last = e
            time.sleep(min(2 ** attempt, 10))
        raise Fail(EXIT_NETWORK, 'download failed after %d attempts: %s (%s)' % (self.retries, url, last),
                   'Check your internet connection (or proxy settings) and run the installer again; it resumes.')


def hf_url(repo, revision, path):
    return '%s/%s/resolve/%s/%s' % (HF_HOST, repo, urllib.parse.quote(revision, safe=''), urllib.parse.quote(path, safe='/'))


def file_ok(path, size, digest):
    try:
        return os.path.getsize(path) == size and sha256_file(path) == digest
    except OSError:
        return False


def payload_entries(manifest, kinds):
    return [e for e in manifest['files'] if e['source'] in kinds]


def missing_bytes(root, entries):
    n = 0
    for e in entries:
        p = native(root, e['path'])
        if not (os.path.isfile(p) and os.path.getsize(p) == e['size']):
            n += e['size']
    return n


def fetch_payload(root, manifest, args, log):
    entries = payload_entries(manifest, ('hf',))
    todo = [e for e in entries if not (os.path.isfile(native(root, e['path'])) and os.path.getsize(native(root, e['path'])) == e['size'])]
    if not todo:
        log.ok('all %d payload files present' % len(entries))
        return
    hf = manifest.get('hf') or {}
    dl = Downloader(log)
    log.info('    %d of %d payload files to fetch, %s' % (len(todo), len(entries), fmt_gb(sum(e['size'] for e in todo))))
    done, lock = [0], threading.Lock()

    def one(e):
        dest = native(root, e['path'])
        if args.payload_source:
            src = native(args.payload_source, e['path'])
            if not os.path.isfile(src):
                raise Fail(EXIT_VERIFY, 'payload file missing in --payload-source: %s' % e['path'])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest + '.part')
            os.replace(dest + '.part', dest)
        else:
            if not hf.get('repo') or '<' in hf['repo']:
                raise Fail(EXIT_NETWORK, 'the payload repository is not set in PAYLOAD-MANIFEST.json', 'Use --payload-source DIR for a local copy.')
            dl.fetch(hf_url(hf['repo'], hf.get('revision') or 'main', e['path']), dest, e['size'])
        with lock:
            done[0] += 1
            if len(todo) < 40 or done[0] % 10 == 0 or done[0] == len(todo):
                log.info('    fetched %d/%d  %s' % (done[0], len(todo), e['path']))

    with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
        for f in [pool.submit(one, e) for e in todo]:
            f.result()


def model_entries(manifest, args):
    ents = [e for e in manifest['files'] if e['source'] == 'external' and e['class'] == 'model']
    if not ents:
        ents = [dict(path='model/' + n, size=None, sha256=None, source='external', url=hf_url(MODEL_REPO, MODEL_REVISION, n),
                     **{'class': 'model'}, optional=False) for n in MODEL_FILES_DEFAULT]
    if (args.model_files or 'all') == 'required':
        bad = [e['path'] for e in ents if not e.get('optional')]
        ents = [e for e in ents if not e.get('optional')]
        if not ents:
            raise Fail(EXIT_USAGE, '--model-files required selects nothing')
    return ents


def hf_tree_listing(log):
    """oid map of the pinned model revision from the HF tree API (LFS sha256 / git blob sha1); {} when unreachable."""
    url = '%s/api/models/%s/tree/%s' % (HF_HOST, MODEL_REPO, MODEL_REVISION)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'plamo3-npu-installer/%s' % VERSION}), timeout=30) as r:
            items = json.load(r)
        return {i['path']: i for i in items if i.get('type') == 'file'}
    except Exception as e:
        log.info('    note: could not read the Hugging Face file listing (%s); relying on the shipped hashes only' % e)
        return {}


def fetch_model(root, manifest, args, log):
    ents = model_entries(manifest, args)
    if args.no_model:
        log.info('    model download skipped (--no-model); verification only')
    listing = {} if (args.model_source or args.no_model) else None
    dl = Downloader(log)
    for e in ents:
        name = e['path'].split('/', 1)[1]
        dest = native(root, e['path'])
        if e.get('size') is not None and os.path.isfile(dest) and os.path.getsize(dest) == e['size']:
            continue
        if os.path.isfile(dest) and e.get('size') is None:
            continue
        if args.no_model:
            raise Fail(EXIT_VERIFY, 'base model file missing: %s' % e['path'], 'Run without --no-model or pass --model-source.')
        if args.model_source:
            src = os.path.join(args.model_source, name)
            if not os.path.isfile(src):
                raise Fail(EXIT_VERIFY, 'file not found in --model-source: %s' % name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest + '.part')
            os.replace(dest + '.part', dest)
            log.info('    copied %s' % name)
        else:
            log.info('    downloading %s (%s)' % (name, fmt_gb(e['size']) if e.get('size') else 'size from server'))
            dl.fetch(e.get('url') or hf_url(MODEL_REPO, MODEL_REVISION, name), dest, e.get('size'))
    # verify
    listing = listing if listing is not None else hf_tree_listing(log)
    for e in ents:
        name = e['path'].split('/', 1)[1]
        dest = native(root, e['path'])
        if not os.path.isfile(dest):
            raise Fail(EXIT_VERIFY, 'base model file missing: %s' % e['path'])
        digest = sha256_file(dest)
        if e.get('sha256') and digest != e['sha256']:
            os.replace(dest, dest + '.rejected')
            raise Fail(EXIT_VERIFY, 'base model file %s does not match the pinned hash (expected %s..., got %s...)' % (name, e['sha256'][:12], digest[:12]),
                       'The file was set aside as %s.rejected; run the installer again to download it afresh.' % name)
        item = listing.get(name)
        if item:
            lfs = (item.get('lfs') or {}).get('oid')
            if lfs and lfs != digest:
                raise Fail(EXIT_VERIFY, 'base model file %s differs from the Hugging Face listing (LFS sha256)' % name)
            if not lfs and item.get('oid') and item['oid'] != blob_sha1(dest):
                raise Fail(EXIT_VERIFY, 'base model file %s differs from the Hugging Face listing (git blob id)' % name)
        elif not e.get('sha256'):
            raise Fail(EXIT_VERIFY, 'no reference hash for %s (offline and none shipped)' % name)
    log.ok('%d base model files verified (revision %s)' % (len(ents), MODEL_REVISION[:8]))


# ------------------------------------------------------------------ verification and attestation

def verify_sums(root, sums, log, workers=4):
    bad, missing = [], []

    def one(item):
        rel, want = item
        p = native(root, rel)
        if not os.path.isfile(p):
            return ('missing', rel, None)
        got = sha256_file(p)
        return ('ok' if got == want else 'bad', rel, got)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, rel, got in pool.map(one, sorted(sums.items())):
            if status == 'missing':
                missing.append(rel)
            elif status == 'bad':
                bad.append(rel)
    # the sealed source inventory of the runtime lists exactly the shipped code files: an UNLISTED source file in a bound root makes every start refuse, so it is an installation error now
    extra = []
    for top in ('IRON', 'lib'):
        for dirpath, dirs, files in os.walk(os.path.join(root, top)):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fn in files:
                if os.path.splitext(fn)[1].lower() in ('.py', '.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.inc', '.toml', '.yaml', '.yml'):
                    rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, '/')
                    if rel not in sums:
                        extra.append(rel)
    if extra:
        for r in extra[:10]:
            log.info('    EXTRA   %s' % r)
        raise Fail(EXIT_VERIFY, '%d source file(s) in the code folders are not part of the release' % len(extra), 'Remove them (the runtime refuses to start with unlisted source files) or run the installer again.')
    if bad or missing:
        eol = [r for r in bad if looks_like_crlf_damage(native(root, r), sums[r])]
        msg = '%d files differ from SHA256SUMS, %d are missing' % (len(bad), len(missing))
        for r in (bad + missing)[:10]:
            log.info('    %s %s' % ('MISSING' if r in missing else 'DIFFERS', r))
        hint = None
        if eol:
            hint = ('%d of the differing files differ only by line endings: your git converted them. Run '
                    '`git config core.autocrlf false`, delete the clone and clone again (the repository sets "* -text").' % len(eol))
        raise Fail(EXIT_VERIFY, msg, hint or 'Run the installer again to fetch damaged or missing files; delete a file to force a new download.')
    log.ok('%d files match SHA256SUMS (%.0f s)' % (len(sums), time.time() - t0))


def looks_like_crlf_damage(path, want):
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return False
    if b'\r\n' not in data or len(data) > 64 * 1024 * 1024:
        return False
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == want


def attest(root, sums, manifest, release, log, workers=4):
    """One read pass under a deny-write handle: whole-file SHA-256 (SHA256SUMS / PAYLOAD-MANIFEST) and NTFS identity
    before and after. Writes state/install-receipt.json only if every file verified."""
    rows = {}
    for rel, digest in sums.items():
        rows[rel] = dict(sha256=digest, size=None)
    for e in manifest['files']:
        if e['source'] == 'external':
            if os.path.isfile(native(root, e['path'])):
                rows[e['path']] = dict(sha256=e.get('sha256'), size=e.get('size'))
        else:
            rows.setdefault(e['path'], dict(sha256=e['sha256'], size=e['size']))
    ids, problems = {}, []

    def one(item):
        rel, ref = item
        p = native(root, rel)
        try:
            with open_retained(p) as stream:
                before = maybe_identity(stream)
                h = hashlib.sha256()
                n = 0
                for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(b)
                    n += len(b)
                after = maybe_identity(stream)
        except OSError as e:
            return rel, 'unreadable: %s' % e, None
        if ref['sha256'] and h.hexdigest() != ref['sha256']:
            return rel, 'content differs', None
        if ref['size'] is not None and n != ref['size']:
            return rel, 'size differs', None
        if before != after:
            return rel, 'changed while being verified', None
        return rel, None, after

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, problem, ident in pool.map(one, sorted(rows.items())):
            if problem:
                problems.append((rel, problem))
            elif ident:
                ids[rel] = ident
    if problems:
        for rel, pr in problems[:10]:
            log.info('    %s: %s' % (rel, pr))
        raise Fail(EXIT_VERIFY, 'attestation failed for %d files' % len(problems), 'Run install.ps1 again; damaged files are fetched anew.')
    cert_rel = release.get('certificate', 'plamo3-default/r45-spec/certificate.json')
    cert = native(root, cert_rel)
    receipt = dict(version=1, install_root=os.path.realpath(root), created=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                   release_sha256=sha256_file(os.path.join(root, 'release.json')),
                   certificate_sha256=sha256_file(cert) if os.path.isfile(cert) else None,
                   ids=[[rel] + [ident[k] for k in IDENTITY_ROW] for rel, ident in sorted(ids.items())])
    write_json(os.path.join(root, 'state', 'install-receipt.json'), receipt)
    if not ids and os.name == 'nt':
        log.warn('no NTFS identity could be read (is this an NTFS volume with a USN journal?): every start will hash all files.')
    log.ok('%d files verified, %d identities recorded in state/install-receipt.json (%.0f s)' % (len(rows), len(ids), time.time() - t0))
    return len(ids)


# ------------------------------------------------------------------ venv

def parse_requirements(path):
    """-> [(name, version, [hashes])] from a hashed requirements file."""
    text = open(path, encoding='utf-8').read().replace('\\\n', ' ')
    out = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        m = re.match(r'^([A-Za-z0-9_.\-]+)==([0-9A-Za-z.\-+]+)(.*)$', line)
        if not m:
            raise Fail(EXIT_USAGE, 'unsupported requirement line: %s' % line)
        out.append((m.group(1), m.group(2), re.findall(r'--hash=sha256:([0-9a-f]{64})', m.group(3))))
    return out


def venv_python(root):
    scripts = 'Scripts' if os.name == 'nt' else 'bin'
    return os.path.join(root, 'venv', scripts, 'python.exe' if os.name == 'nt' else 'python')


def norm(name):
    return re.sub(r'[-_.]+', '_', name).lower()


def verify_record(dist_info, base):
    """Re-hash every file listed in a wheel's RECORD. Returns (checked, [problems])."""
    checked, problems = 0, []
    with open(os.path.join(dist_info, 'RECORD'), newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 3 or not row[1]:
                continue
            algo, _, b64 = row[1].partition('=')
            if algo != 'sha256':
                continue
            p = os.path.normpath(os.path.join(base, row[0]))
            if not os.path.isfile(p):
                problems.append('missing %s' % row[0])
                continue
            want = base64.urlsafe_b64decode(b64 + '=' * (-len(b64) % 4)).hex()
            checked += 1
            if sha256_file(p) != want:
                problems.append('changed %s' % row[0])
    return checked, problems


def venv_step(root, args, log):
    req = os.path.join(root, 'requirements-release.txt')
    pins = parse_requirements(req)
    vpy = venv_python(root)
    if not os.path.isfile(vpy):
        log.info('    creating venv')
        r = subprocess.run([sys.executable, '-m', 'venv', os.path.join(root, 'venv')], capture_output=True)
        if r.returncode:
            raise Fail(EXIT_ENV, 'venv creation failed: %s' % r.stderr.decode('utf-8', 'replace').strip()[-300:],
                       'Make sure the Python installation includes the venv and pip components.')
    def installed():
        code = 'import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in %r}))' % ([p[0] for p in pins],)
        r = subprocess.run([vpy, '-c', code], capture_output=True)
        try:
            return json.loads(r.stdout.decode()) if r.returncode == 0 else {}
        except ValueError:
            return {}
    have = installed()
    if any(have.get(n) != v for n, v, _ in pins):
        log.info('    installing pinned wheels (hash-checked)')
        cmd = [vpy, '-m', 'pip', 'install', '--isolated', '--disable-pip-version-check', '--no-input', '--require-hashes',
               '--only-binary=:all:', '--no-deps', '-r', req]
        wheels = os.path.join(root, 'wheels')
        if os.path.isdir(wheels):
            cmd[-2:-2] = ['--find-links', wheels]
        r = subprocess.run(cmd, capture_output=True)
        log.info(r.stdout.decode('utf-8', 'replace')[-400:].strip()) if r.returncode == 0 else None
        if r.returncode:
            raise Fail(EXIT_NETWORK, 'pip could not install the pinned wheels:\n' + r.stderr.decode('utf-8', 'replace').strip()[-600:],
                       'Check the internet connection. A hash mismatch means the download was altered: do not retry blindly.')
        have = installed()
    for n, v, _ in pins:
        if have.get(n) != v:
            raise Fail(EXIT_VERIFY, '%s %s expected in the venv, found %s' % (n, v, have.get(n)))
    # re-hash the installed files against the wheels' RECORD
    r = subprocess.run([vpy, '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'], capture_output=True)
    purelib = r.stdout.decode().strip()
    total = 0
    for n, v, _ in pins:
        dist = [d for d in os.listdir(purelib) if d.lower().endswith('.dist-info') and norm(d[:-len('.dist-info')].rsplit('-', 1)[0]) == norm(n)]
        if not dist:
            raise Fail(EXIT_VERIFY, 'no dist-info for %s in the venv' % n)
        checked, problems = verify_record(os.path.join(purelib, dist[0]), purelib)
        if problems:
            raise Fail(EXIT_VERIFY, '%s differs from its wheel RECORD: %s' % (n, '; '.join(problems[:3])),
                       'Delete the venv folder and run the installer again.')
        total += checked
    log.ok('venv ready: %s; %d installed files match the wheel RECORDs' % (', '.join('%s %s' % (n, v) for n, v, _ in pins), total))
    return have


# ------------------------------------------------------------------ environment record

def privacy(path):
    prof = os.environ.get('USERPROFILE')
    if prof and path and path.lower().startswith(prof.lower()):
        return '%USERPROFILE%' + path[len(prof):]
    return path


def env_record(root, facts, versions, log):
    pyxrt = native(root, 'xrt/python/pyxrt.pyd')
    rec = dict(version=1, created=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), installer=VERSION,
               windows=facts.get('windows'), python=facts['python'], python_build=facts['python_build'], python_cache_tag=facts['python_cache_tag'],
               python_executable=privacy(facts['python_executable']), packages=versions, volume_fs=facts.get('volume_fs'),
               memory_gb=facts.get('memory_gb'), xrt_coreutil=dict(version=facts['xrt_coreutil']['version']),
               npu_driver=facts.get('npu_driver'),
               pyxrt=dict(sha256=sha256_file(pyxrt) if os.path.isfile(pyxrt) else None))
    cu = facts['xrt_coreutil']['path']
    if os.path.isfile(cu):
        rec['xrt_coreutil']['sha256'] = sha256_file(cu)
        rec['xrt_coreutil']['size'] = os.path.getsize(cu)
    write_json(os.path.join(root, 'state', 'env-record.json'), rec)
    kg_path = os.path.join(root, 'known-good-env.json')
    warnings = []
    if os.path.isfile(kg_path):
        kg = read_json(kg_path)
        wb = facts.get('windows')
        if wb and 'windows_build' in kg and '%d.%d.%d' % (wb['major'], wb['minor'], wb['build']) not in kg['windows_build']:
            warnings.append('Windows %d.%d.%d is not in the tested list (%s)' % (wb['major'], wb['minor'], wb['build'], ', '.join(kg['windows_build'])))
        v = facts['xrt_coreutil']['version']
        if v and v not in kg.get('xrt_coreutil', [v]):
            warnings.append('xrt_coreutil.dll %s is not in the tested list (%s)' % (v, ', '.join(kg['xrt_coreutil'])))
        if facts['python'] not in kg.get('python', [facts['python']]):
            warnings.append('Python %s is not in the tested list (%s)' % (facts['python'], ', '.join(kg['python'])))
        if kg.get('pyxrt_sha256') and rec['pyxrt']['sha256'] and rec['pyxrt']['sha256'] != kg['pyxrt_sha256']:
            warnings.append('pyxrt.pyd differs from the tested build')
    for w in warnings:
        log.warn(w + '. Continuing; if you see crashes or device errors please report them with `state\\env-record.json`.')
    log.ok('environment recorded in state/env-record.json (%d warnings)' % len(warnings))
    return rec


# ------------------------------------------------------------------ self test

def selftest_step(root, args, log):
    st = os.path.join(root, 'selftest.py')
    if not os.path.isfile(st):
        raise Fail(EXIT_VERIFY, 'selftest.py missing')
    cmd = [venv_python(root), st, '--root', root]
    log.info('    running the self-test (about 2 minutes; the NPU is used by this test only)')
    r = subprocess.run(cmd)
    if r.returncode == 0:
        log.ok('self-test passed')
        return
    if r.returncode >= EXIT_SELFTEST_BASE:
        raise Fail(r.returncode, 'self-test step %d failed (see state\\selftest-report.json)' % (r.returncode - EXIT_SELFTEST_BASE),
                   'The installation is in place. Send state\\selftest-report.json when you report the problem.')
    raise Fail(EXIT_INTERNAL, 'self-test exited with code %d' % r.returncode)


# ------------------------------------------------------------------ uninstall

def uninstall(root, args, log):
    mpath = os.path.join(root, 'state', 'install-manifest.json')
    if not os.path.isfile(mpath):
        raise Fail(EXIT_USAGE, 'state/install-manifest.json not found: nothing recorded as installed here',
                   'Only files the installer created are ever removed.')
    m = read_json(mpath)
    root_real = os.path.realpath(root)
    victims = []
    for rel in m.get('files', []):
        victims.append(('file', rel))
    for d in m.get('dirs', []):
        if args.keep_model and d == 'model':
            continue
        victims.append(('dir', d))
    safe = []
    for kind, rel in victims:
        if rel.startswith('/') or '..' in rel.split('/') or re.match(r'^[A-Za-z]:', rel):
            raise Fail(EXIT_VERIFY, 'unsafe path in install-manifest: %s' % rel)
        p = native(root, rel)
        if args.keep_model and rel.startswith('model/'):
            continue
        if not os.path.realpath(os.path.dirname(p)).startswith(root_real):
            raise Fail(EXIT_VERIFY, 'refusing to touch %s (outside the install root)' % rel)
        safe.append((kind, rel, p))
    size = 0
    for kind, rel, p in safe:
        if kind == 'file' and os.path.isfile(p):
            size += os.path.getsize(p)
    log.info('would remove %d files, %d directories (%s) created by the installer; code from the git clone is kept.' % (
        sum(k == 'file' for k, _, _ in safe), sum(k == 'dir' for k, _, _ in safe), fmt_gb(size)))
    if args.dry_run:
        for kind, rel, p in safe[:20]:
            log.info('    %s %s' % (kind, rel))
        log.info('dry run: nothing removed')
        return
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            raise Fail(EXIT_USAGE, 'confirmation required', 'Pass -Yes to remove without asking.')
        if input('Type YES to remove them: ').strip() != 'YES':
            raise Fail(EXIT_USAGE, 'aborted; nothing removed')
    for kind, rel, p in safe:
        if kind == 'file':
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
    for kind, rel, p in sorted((s for s in safe if s[0] == 'dir'), key=lambda s: -len(s[1])):
        if os.path.islink(p) or is_junction(p):
            os.rmdir(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    # empty payload directories left behind by removed files
    for dp, dn, fn in os.walk(root, topdown=False):
        if dp != root and not os.listdir(dp) and not os.path.isdir(os.path.join(dp, '.git')):
            if os.path.relpath(dp, root).split(os.sep)[0] in {d.split('/')[0] for d in m.get('dirs', [])} | {f.split('/')[0] for f in m.get('files', [])}:
                os.rmdir(dp)
    for name in ('plamo3-npu-turbo', 'plamo3-npu'):
        la = os.environ.get('LOCALAPPDATA')
        if la and os.path.isdir(os.path.join(la, name)):
            shutil.rmtree(os.path.join(la, name), ignore_errors=True)
    log.info('removed. Delete the cloned repository folder yourself if you no longer want the code.')


def is_junction(p):
    try:
        return bool(os.stat(p, follow_symlinks=False).st_file_attributes & 0x400) if os.name == 'nt' else False
    except OSError:
        return False


def write_install_manifest(root, manifest, args):
    files = [e['path'] for e in manifest['files'] if e['source'] in ('hf', 'external') and os.path.isfile(native(root, e['path']))]
    dirs = ['venv', 'state', 'model']
    write_json(os.path.join(root, 'state', 'install-manifest.json'),
               dict(version=1, files=sorted(files), dirs=dirs, installer=VERSION, model_files=args.model_files or 'all'))


# ------------------------------------------------------------------ main flow

def build_parser():
    p = argparse.ArgumentParser(description='Install the PLaMo 3 Translate XDNA1-NPU runtime.', add_help=True)
    p.add_argument('--root', help='install root (default: the folder of this script = the cloned repository)')
    p.add_argument('--accept-license', action='store_true', help='accept the PLaMo Community License without typing')
    p.add_argument('--payload-source', help='local directory holding the payload files (same relative paths) instead of Hugging Face')
    p.add_argument('--model-source', help='existing snapshot of the base model to copy from instead of downloading')
    p.add_argument('--model-files', choices=('all', 'required'), default=None, help='"required" skips files the release marks optional')
    p.add_argument('--no-model', action='store_true', help='do not download the base model (verify only)')
    p.add_argument('--with-gate-data', action='store_true', help='also fetch the optional gate reference data (about 359 MB)')
    p.add_argument('--verify', action='store_true', help='verify an existing installation (no download, no self-test)')
    p.add_argument('--reattest', action='store_true', help='verify and rewrite state/install-receipt.json')
    p.add_argument('--selftest-only', action='store_true')
    p.add_argument('--uninstall', action='store_true')
    p.add_argument('--keep-model', action='store_true')
    p.add_argument('--yes', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--plan', action='store_true', help='print what would be done and run only the read-only checks')
    p.add_argument('--no-npu-check', action='store_true')
    p.add_argument('--skip-selftest', action='store_true')
    p.add_argument('--sums-sig', help='detached SSH signature of SHA256SUMS (default: SHA256SUMS.sig next to it)')
    p.add_argument('--fingerprint', help='expected signing key fingerprint, SHA256:... (must equal the key pinned in release.json signing)')
    p.add_argument('--require-signature', action='store_true', help='refuse to continue unless ssh-keygen verifies SHA256SUMS.sig')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--log', help='log file (default: state/logs/install-<time>.log)')
    p.add_argument('--resume', action='store_true', help='accepted for clarity: every run resumes')
    p.add_argument('--version', action='version', version='install.py %s' % VERSION)
    return p


def run(args):
    root = os.path.abspath(args.root or os.path.dirname(os.path.abspath(__file__)))
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = args.log or os.path.join(root, 'state', 'logs', 'install-%s.log' % stamp)
    log = Log(args.quiet, None if (args.plan or args.dry_run or args.uninstall) else log_path)
    log.info('PLaMo 3 Translate XDNA1-NPU runtime - installer %s' % VERSION)
    log.info('install root: %s' % root)
    if args.uninstall:
        uninstall(root, args, log)
        return EXIT_OK
    release, manifest, sums = load_release(root)
    if args.model_files is None:      # explicit option > what the installation was made with > everything
        try:
            args.model_files = read_json(os.path.join(root, 'state', 'install-manifest.json')).get('model_files', 'all')
        except (OSError, ValueError):
            args.model_files = 'all'
    if args.payload_source:
        args.payload_source = os.path.abspath(args.payload_source)
    if not args.with_gate_data:
        skip = {e['path'] for e in manifest['files'] if e.get('class') == 'gate-data'}
        sums = {k: v for k, v in sums.items() if k not in skip}
    hf_entries = [e for e in payload_entries(manifest, ('hf',)) if e.get('class') != 'gate-data' or args.with_gate_data]
    manifest['files'] = [e for e in manifest['files'] if e['source'] != 'hf' or e in hf_entries]
    model_ents = model_entries(manifest, args)
    need = missing_bytes(root, hf_entries) + sum((e['size'] or 0) for e in model_ents if e.get('size') and not os.path.isfile(native(root, e['path']))) + 1.5 * GB
    log.info('release %s%s' % (release.get('release', '?'), ' (plan only)' if args.plan else ''))
    if args.plan:
        log.info('plan: licence -> signature check -> preflight -> payload (%s) -> verify SHA256SUMS -> venv (numpy, ml_dtypes, hashed) -> '
                 'base model (%s@%s) -> attest + receipt -> env record -> self-test' % (fmt_gb(missing_bytes(root, hf_entries)), MODEL_REPO, MODEL_REVISION[:8]))
        log.info('network use: huggingface.co (payload repo, base model), pypi.org via pip. Nothing else, no telemetry.')
    log.step('Preflight')
    facts = preflight(root, args.plan, args, log, need)
    if args.plan:
        log.info('plan complete: nothing was changed.')
        return EXIT_OK
    only_verify = args.verify or args.reattest
    if not only_verify and not args.selftest_only:
        log.step('Licence')
        licence_step(root, args, log)
    log.step('Signature of SHA256SUMS')
    signature_step(root, release, args, log)
    if not only_verify and not args.selftest_only:
        log.step('Payload')
        fetch_payload(root, {'files': manifest['files'], 'hf': manifest.get('hf')}, args, log)
    log.step('Verify against SHA256SUMS')
    verify_sums(root, sums, log)
    versions = {}
    if not only_verify:
        log.step('Python environment')
        versions = venv_step(root, args, log)
        log.step('Base model')
        fetch_model(root, manifest, args, log)
    else:
        log.step('Base model')
        fetch_model(root, manifest, argparse.Namespace(**dict(vars(args), no_model=True)), log)
    log.step('Attestation and install receipt')
    attest(root, sums, manifest, release, log)
    if not only_verify:
        write_install_manifest(root, manifest, args)
    log.step('Environment record')
    env_record(root, facts, versions, log)
    if only_verify:
        log.info('verification finished: OK')
        return EXIT_OK
    if not args.skip_selftest:
        log.step('Self-test')
        selftest_step(root, args, log)
    summary = dict(version=1, installer=VERSION, release=release.get('release'), finished=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                   selftest=('skipped' if args.skip_selftest else 'passed'), warnings=[])
    write_json(os.path.join(root, 'state', 'install-summary.json'), summary)
    log.info('')
    log.info('Installed. Try:   .\\translate_plamo3.ps1 "Hello." -SourceLang en -TargetLang ja')
    log.info('Docs: USAGE.md (options, exit codes), INSTALL.md (updates, uninstall). Built with PLaMo. Unofficial; see README.md.')
    return EXIT_OK


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest_only:
        args.verify = False
    try:
        if args.selftest_only:
            root = os.path.abspath(args.root or os.path.dirname(os.path.abspath(__file__)))
            return _selftest_only(root, args)
        return run(args)
    except Fail as e:
        print('\nERROR: %s' % e.message, file=sys.stderr)
        if e.hint:
            print('  -> %s' % e.hint, file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print('\ninterrupted; run again to resume.', file=sys.stderr)
        return 130
    except Exception as e:     # never a bare traceback for a user
        if os.environ.get('PLAMO3_DEBUG') == '1':
            raise
        print('\nERROR: internal error (%s: %s). Run again with PLAMO3_DEBUG=1 for details and report it.' % (type(e).__name__, e), file=sys.stderr)
        return EXIT_INTERNAL


def _selftest_only(root, args):
    log = Log(args.quiet, os.path.join(root, 'state', 'logs', 'selftest-%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    log.step('Self-test')
    selftest_step(root, args, log)
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
