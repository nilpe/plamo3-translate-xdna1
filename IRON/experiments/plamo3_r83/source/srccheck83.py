"""R83 out-of-process source attestation helper.

Why: the early-XRT calls (pyxrt import, device_open, xclbin load, hw_context_create) hold the GIL for ~200 ms of the startup. In R61 the 2208
file attestation (2 worker threads) therefore could not make progress during that time and ended ~160 ms after the last hw_context.
The 1404 bound *source-only* files (never retained, never loaded by the device path) are attested in a separate Python process instead.
Nothing about the attestation changes: the same functions (runtime_identity.maybe_identity / check_content, runtime_manifest._open_retained /
source_files / _record_path) decide for the same records; a mismatch, a missing file, a malformed record or an inventory difference refuses
(StaleManifest) in the parent BEFORE the identity commit (IdentityLease.wait) and therefore before any validated state is used.
If the helper cannot be started, dies, times out or returns something unintelligible, the parent attests the same records in-process (never skipped).

check_group() is the single per-file decision function shared by the helper and the in-process fallback; it is a verbatim restatement of the
non-retained branch of identity43.IdentityLease.check (open deny-write/delete, identity query, identity-equality or full chunk SHA-256 fallback,
identity re-query after fallback, close).
"""
import json
import marshal
import os
import subprocess
import sys
import threading
import time


def check_group(name, group, *, version_ok, force_audit, open_retained, identity, content, stale):
    stream = open_retained(name)
    try:
        ident = identity(stream)
        match = (
            not force_audit
            and version_ok
            and ident is not None
            and all(r.get("ntfs_identity") == ident for r in group)
        )
        hashed = 0 if match else content(stream, group)
        if not match and identity(stream) != ident:
            raise stale(f"dependency changed during fallback: {name}")
        return match, hashed
    finally:
        stream.close()


def check_items(items, **kw):
    matches = hashed = 0
    fallback = []
    for name, group in items:
        match, h = check_group(name, group, **kw)
        matches += int(match)
        hashed += h
        if not match:
            fallback.append(name)
    return matches, hashed, fallback


class SourceChecker:
    """Parent side: spawn early, send the source-only records after the manifest is loaded, collect one result."""

    TIMEOUT_S = 30.0

    def __init__(self, proc):
        self.proc = proc
        self.out = None
        self.error = None
        self.sent = False
        self.done = threading.Event()
        self.sent_at = None
        self.thread = None

    @classmethod
    def spawn(cls, app_dir):
        try:
            exe = getattr(sys, "_base_executable", sys.executable)
            proc = subprocess.Popen(
                [exe, "-S", "-E", "-u", os.path.abspath(__file__), str(app_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except (OSError, ValueError):
            return None
        return cls(proc)

    def send(self, request):
        try:
            data = marshal.dumps(request, 4)
            self.proc.stdin.write(len(data).to_bytes(8, "little") + data)
            self.proc.stdin.close()
            self.sent = True
            self.sent_at = time.perf_counter()
        except (OSError, ValueError):
            self.close()
            return False
        self.thread = threading.Thread(target=self._read, daemon=True, name="r83-source-checker")
        self.thread.start()
        return True

    def _read(self):
        try:
            self.out = self.proc.stdout.read()
        except BaseException as exc:  # pragma: no cover - defensive
            self.error = exc
        finally:
            self.done.set()

    def poll(self):
        """None while pending; otherwise the decoded result, or False if unusable (caller must attest in-process)."""
        if not self.sent or not self.done.is_set():
            return None
        return self._decode()

    def result(self):
        if not self.sent:
            return False
        remaining = max(0.0, self.TIMEOUT_S - (time.perf_counter() - self.sent_at))
        if not self.done.wait(remaining):
            self.close()
            return False
        return self._decode()

    def _decode(self):
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.close()
            return False
        if self.error is not None or self.proc.returncode != 0 or not self.out:
            return False
        try:
            result = json.loads(self.out.decode())
        except (ValueError, UnicodeError):
            return False
        return result if isinstance(result, dict) and "ok" in result else False

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except OSError:
            pass
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass


def _main(app_dir):
    sys.path.insert(0, app_dir)
    import runtime_identity as ri
    import runtime_manifest as base

    started = time.perf_counter()
    size = int.from_bytes(sys.stdin.buffer.read(8), "little")
    request = marshal.loads(sys.stdin.buffer.read(size))
    result = dict(ok=True, files=0, matches=0, hashed_bytes=0, fallback_paths=[], inventory_files=0)
    try:
        for name, group in request["items"]:
            for record in group:
                base._record_path(record)
        walked = {}

        def walk():
            t = time.perf_counter()
            try:
                walked["files"] = base.source_files(request["roots"])
            except BaseException as exc:
                walked["error"] = exc
            walked["ms"] = (time.perf_counter() - t) * 1000

        thread = threading.Thread(target=walk)
        thread.start()
        try:
            items = request["items"]
            matches, hashed, fallback = check_items(
                items,
                version_ok=request["version_ok"],
                force_audit=request["force_audit"],
                open_retained=base._open_retained,
                identity=ri.maybe_identity,
                content=ri.check_content,
                stale=base.StaleManifest,
            )
        finally:
            thread.join()
        if "error" in walked:
            raise walked["error"]
        if walked["files"] != request["expected_inventory"]:
            raise base.StaleManifest("runtime source inventory changed; rebuild the bundle")
        result.update(
            files=len(items),
            matches=matches,
            hashed_bytes=hashed,
            fallback_paths=fallback,
            inventory_files=len(walked["files"]),
            inventory_ms=walked["ms"],
        )
    except base.StaleManifest as exc:
        result = dict(ok=False, kind="stale", message=str(exc))
    except OSError as exc:
        result = dict(ok=False, kind="os", message=f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        result = dict(ok=False, kind="error", message=f"{type(exc).__name__}: {exc}")
    result["child_ms"] = (time.perf_counter() - started) * 1000
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    _main(sys.argv[1])
