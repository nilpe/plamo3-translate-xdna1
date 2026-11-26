# SPDX-License-Identifier: Apache-2.0
"""Environment record / warn layer (Windows and driver files are RECORDED, never attested).

Our own files (weights, streams, kernels, code, pyxrt.pyd) stay strictly hash-verified elsewhere. Components that belong to Windows, the AMD NPU
driver or the Python interpreter are only recorded here and compared with the shipped known-good-env.json; an unknown version is a warning
(once per process, on stderr), never a refusal. Strict facts (CPython 3.13, 64-bit) live in plamo_root.check_platform.
Stdlib + ctypes only. Everything here is best effort: no function raises.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import plamo_root
from plamo_errors import note

KNOWN_GOOD = plamo_root.ROOT / "known-good-env.json"
STATE = plamo_root.STATE
RECORD = STATE / "env-record.json"
CHANGES = STATE / "logs" / "env-changes.jsonl"


def system_root():
    return Path(os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or "C:/Windows")


def watched_files():
    s32 = system_root() / "System32"
    return {
        "xrt_coreutil": s32 / "xrt_coreutil.dll",
        "xrt_smi": s32 / "AMD" / "xrt-smi.exe",
        "dxcore": s32 / "DXCore.dll",
    }


def file_version(path):
    """Four-part file version from the version resource (ctypes version.dll); None when unavailable."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        version = ctypes.WinDLL("version", use_last_error=True)
        version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
        version.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        version.GetFileVersionInfoW.restype = wintypes.BOOL
        version.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
        version.VerQueryValueW.restype = wintypes.BOOL
        size = version.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(str(path), 0, size, buf):
            return None
        ptr, length = ctypes.c_void_p(), wintypes.UINT()
        if not version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or length.value < 52:
            return None
        v = (ctypes.c_uint32 * 13).from_address(ptr.value)
        return "%02d.%02d.%05d.%d" % (v[4] >> 16, v[4] & 0xFFFF, v[5] >> 16, v[5] & 0xFFFF)
    except Exception:
        return None


def loaded_module_path(name):
    """Full path of an already loaded DLL (GetModuleHandleW + GetModuleFileNameW); None when it is not loaded."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        k.GetModuleHandleW.restype = wintypes.HMODULE
        k.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        k.GetModuleFileNameW.restype = wintypes.DWORD
        handle = k.GetModuleHandleW(name)
        if not handle:
            return None
        buf = ctypes.create_unicode_buffer(1024)
        n = k.GetModuleFileNameW(handle, buf, 1024)
        return buf.value if n else None
    except Exception:
        return None


def windows_build():
    try:
        v = sys.getwindowsversion()
        build = f"{v.major}.{v.minor}.{v.build}"
        ubr = None
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                ubr = winreg.QueryValueEx(key, "UBR")[0]
        except OSError:
            pass
        return build, ubr
    except Exception:
        return None, None


def _stat(path):
    try:
        st = Path(path).stat()
        return dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    except OSError:
        return None


def collect(mode="cheap"):
    """The environment record. `cheap`: versions + stat (about 1-2 ms). `full`: plus sha256 of the watched system files."""
    rec = dict(version=1, collected=time.strftime("%Y-%m-%dT%H:%M:%S"), mode=mode)
    try:
        files = {}
        for name, path in watched_files().items():
            entry = dict(path=str(path), file_version=file_version(path))
            entry.update(_stat(path) or dict(missing=True))
            if mode == "full" and not entry.get("missing"):
                try:
                    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    pass
            files[name] = entry
        rec["files"] = files
        build, ubr = windows_build()
        rec["windows"] = dict(build=build, ubr=ubr)
        rec["python"] = dict(
            version=".".join(map(str, sys.version_info[:3])),
            cache_tag=sys.implementation.cache_tag,
            build=" ".join(sys.version.split())[:120],
            executable=sys.executable,
        )
        np = sys.modules.get("numpy")
        md = sys.modules.get("ml_dtypes")
        rec["packages"] = dict(numpy=getattr(np, "__version__", None), ml_dtypes=getattr(md, "__version__", None))
        pyxrt = sys.modules.get("pyxrt")
        rec["pyxrt"] = dict(path=getattr(pyxrt, "__file__", None))
        rec["xrt_coreutil_loaded"] = loaded_module_path("xrt_coreutil.dll")
    except Exception as error:  # a record is diagnostics; never fatal
        rec["error"] = f"{type(error).__name__}: {error}"
    return rec


def load_known_good(path=None):
    try:
        return json.loads(Path(path or KNOWN_GOOD).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def compare(record, known):
    """Warning lines for components not in the tested list. Empty when everything is known or no list exists."""
    warnings = []
    if not known:
        return warnings
    try:
        coreutil = record.get("files", {}).get("xrt_coreutil", {}).get("file_version")
        tested = known.get("xrt_coreutil", [])
        if coreutil and tested and coreutil not in tested:
            warnings.append(
                f"PLaMo: note: xrt_coreutil.dll {coreutil} is not in the tested list (tested: {', '.join(tested)}). "
                "Continuing; if you see crashes or device_open errors please report."
            )
        loaded = record.get("xrt_coreutil_loaded")
        expected = str(watched_files()["xrt_coreutil"])
        if loaded and os.path.normcase(loaded) != os.path.normcase(expected):
            warnings.append(f"PLaMo: note: xrt_coreutil.dll was loaded from {loaded}, not from {expected}. Continuing.")
        build = record.get("windows", {}).get("build")
        if build and known.get("windows_build") and build not in known["windows_build"]:
            warnings.append(f"PLaMo: note: Windows build {build} is not in the tested list (tested: {', '.join(known['windows_build'])}). Continuing.")
        py = record.get("python", {}).get("version")
        if py and known.get("python") and py not in known["python"]:
            warnings.append(f"PLaMo: note: Python {py} is not in the tested list (tested: {', '.join(known['python'])}). Continuing.")
        for pkg in ("numpy", "ml_dtypes"):
            v = record.get("packages", {}).get(pkg)
            if v and known.get(pkg) and v not in known[pkg]:
                warnings.append(f"PLaMo: note: {pkg} {v} is not in the tested list (tested: {', '.join(known[pkg])}). Continuing.")
    except Exception:
        pass
    return warnings


def _signature(record):
    files = record.get("files", {})
    return json.dumps(
        {k: [v.get("file_version"), v.get("size"), v.get("mtime_ns")] for k, v in sorted(files.items())}
        | {"win": record.get("windows"), "py": record.get("python", {}).get("version"), "coreutil": record.get("xrt_coreutil_loaded")},
        sort_keys=True,
    )


def check(known_good=None, *, stream=None, write_state=True):
    """Startup check, called once after the runtime was constructed (pyxrt is imported by then). Prints at most a few warning lines and, when the
    cheap record differs from the stored one, appends one line to state/logs/env-changes.jsonl and refreshes state/env-record.json. Never raises."""
    try:
        record = collect("cheap")
        for line in compare(record, load_known_good(known_good)):
            note(line, key=line, stream=stream)
        if write_state:
            _log_change(record)
        return record
    except Exception:
        return None


def _log_change(record):
    try:
        if not STATE.is_dir():
            return
        previous = None
        try:
            previous = json.loads(RECORD.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if previous is not None and _signature(previous) == _signature(record):
            return
        full = collect("full")
        CHANGES.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(time=full["collected"], files={k: [v.get("file_version"), v.get("sha256")] for k, v in full.get("files", {}).items()}, windows=full.get("windows"), python=full.get("python", {}).get("version")))
        with open(CHANGES, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        tmp = RECORD.with_suffix(".tmp")
        tmp.write_text(json.dumps(full, indent=1), encoding="utf-8")
        os.replace(tmp, RECORD)
    except Exception:
        pass


def report_text():
    """Text for --env-report: the full record with the user profile directory replaced by %USERPROFILE%."""
    record = collect("full")
    text = json.dumps(record, indent=1, ensure_ascii=False)
    home = os.environ.get("USERPROFILE")
    if home:
        text = text.replace(home.replace("\\", "\\\\"), "%USERPROFILE%").replace(home, "%USERPROFILE%")
    return text
