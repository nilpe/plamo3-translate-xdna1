# SPDX-License-Identifier: Apache-2.0
"""User-facing error model of the PLaMo 3 NPU runtime: exit-code table, one-line messages, no tracebacks.

Exit codes (documented in USAGE.md):
  0 OK | 1 internal error | 2 usage / invalid request | 3 installation problem or attestation refused
  4 degraded run completed (--strict only) | 5 NPU unavailable / busy | 6 environment problem | 130 Ctrl+C
Stdlib only; safe to import at the very start of a process.
"""
import os
import sys

OK = 0
INTERNAL = 1
USAGE = 2
INSTALL = 3
DEGRADED = 4
NPU = 5
ENVIRONMENT = 6
INTERRUPTED = 130

CONTEXT_TOKENS = 2048  # validated context capacity (prompt + output)
MAX_TEXT_CHARS = 1_000_000


class PlamoError(Exception):
    """An error that is shown to the user as one line `PLaMo: error: <message>` and ends the process with `code`."""

    code = INTERNAL

    def __init__(self, message, code=None, hint=None):
        super().__init__(message)
        self.message = str(message)
        if code is not None:
            self.code = code
        self.hint = hint


class InvalidRequest(PlamoError):
    code = USAGE


class PromptTooLong(InvalidRequest):
    def __init__(self, tokens, limit):
        self.tokens, self.limit = tokens, limit
        super().__init__(
            f"input is {tokens} tokens; the limit is {limit} (the context holds {limit} tokens including the translation). "
            "Split the text into paragraphs."
        )


class InstallProblem(PlamoError):
    code = INSTALL


class NpuBusy(PlamoError):
    code = NPU


class NpuUnavailable(PlamoError):
    code = NPU


class EnvironmentProblem(PlamoError):
    code = ENVIRONMENT


def debug_enabled():
    return os.environ.get("PLAMO3_DEBUG") == "1"


_noted = set()


def note(message, *, key=None, stream=None):
    """One diagnostic line on stderr, once per process per key. Never raises."""
    k = key or message
    if k in _noted:
        return
    _noted.add(k)
    try:
        print(message, file=stream or sys.stderr, flush=True)
    except Exception:
        pass


def _clean(text, limit=300):
    return " ".join(str(text).split())[:limit]


# StaleManifest / OSError texts are mapped to a short reason class (plan section 5.4).
_STALE_REASONS = (
    ("source inventory changed", "extra or missing source file"),
    ("size changed", "file size changed"),
    ("content changed", "file changed"),
    ("content hash mismatch", "file changed"),
    ("manifest content changed", "file changed"),
    ("configuration changed", "environment variable set by the user (IRON_PLAMO_*)"),
    ("python abi changed", "unsupported Python build"),
    ("file changed during", "file changed"),
    ("dependency changed", "file changed"),
    ("short read", "file truncated"),
    ("must be absolute", "manifest path"),
)


def stale_reason(text):
    low = str(text).lower()
    for needle, reason in _STALE_REASONS:
        if needle in low:
            return reason
    return "attestation refused"


def classify(error):
    """Map an arbitrary exception to a PlamoError (never raises). Unknown exceptions become INTERNAL."""
    if isinstance(error, PlamoError):
        return error
    name = type(error).__name__
    text = _clean(error)
    low = text.lower()
    if name == "StaleManifest":
        hint = ""
        if "configuration changed" in low:
            names = sorted(k for k in os.environ if k.startswith("IRON_PLAMO_"))
            if names:
                hint = " [environment variable(s) set that change the certified configuration: " + ", ".join(names) + "; remove them]"
        return InstallProblem(
            f"the installation is incomplete or modified ({stale_reason(text)}: {text}){hint}. Run install.ps1 -Verify."
        )
    if "0xc01e0009" in low or "failed to create context" in low:
        return NpuBusy(
            "the NPU could not be opened (0xc01e0009): another program may be using it, or the driver stopped. "
            "Close other NPU programs and try again."
        )
    if name in ("FileNotFoundError", "PermissionError") and getattr(error, "filename", None):
        return InstallProblem(
            f"the installation is incomplete or modified (file missing or unreadable: {error.filename}). Run install.ps1 -Verify."
        )
    if name == "ImportError" and ("xrt_coreutil" in low or "pyxrt" in low or "dll load failed" in low):
        return EnvironmentProblem("xrt_coreutil.dll or pyxrt could not be loaded: install or update the AMD NPU driver. " + text)
    if name == "WorkerUnavailable":
        return InstallProblem("the priming worker is unavailable: " + text + ". Run install.ps1 -Verify.")
    return PlamoError(f"internal error ({name}); run translate_plamo3.ps1 -Diag and report it (set PLAMO3_DEBUG=1 for details)", INTERNAL)


def report(error, *, stream=None):
    """Print the one-line message for `error` and return its exit code. Traceback only with PLAMO3_DEBUG=1."""
    e = classify(error)
    try:
        print(f"PLaMo: error: {e.message}", file=stream or sys.stderr, flush=True)
        if e.hint:
            print(f"PLaMo: {e.hint}", file=stream or sys.stderr, flush=True)
        if debug_enabled():
            import traceback

            traceback.print_exception(type(error), error, error.__traceback__, file=stream or sys.stderr)
    except Exception:
        pass
    return e.code
