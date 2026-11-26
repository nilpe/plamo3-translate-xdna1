# SPDX-License-Identifier: Apache-2.0
"""Content-validated runtime bundles; a stale dependency is always a cache miss."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time

import plamo_root

VERSION = 1
CHUNK = 512 * 1024**2
SMALL_SNAPSHOT = 64 * 1024
HASH_BATCH_BYTES = 32 * 1024**2
HASH_BATCH_JOBS = 256
SOURCE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".toml",
    ".yaml",
    ".yml",
}


class StaleManifest(RuntimeError):
    pass


def canonical(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def configuration():
    # Diagnostic checks do not change inference; compile limits remain recorded.
    # XRT_ROOT, XILINX_XRT and SYSTEMROOT are no longer part of the compared set (environment.py records them; plamo_root.config_equal
    # ignores them in older manifests too).
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith("IRON_PLAMO_")
        and not k.startswith(("IRON_PLAMO_VERIFY_", "IRON_PLAMO_STREAM_CACHE"))
        or k
        in (
            "PLAMO3_RUNLIST",
            "IRON_COMPILE_JOBS",
            "AIE_KERNEL_COMPILE_JOBS",
            "IRON_DESIGN_COMPILE_JOBS",
            "NPU2",
        )
    }


def python_abi():
    return {
        "cache_tag": sys.implementation.cache_tag,
        "version": list(sys.version_info[:3]),
    }


def source_files(roots):
    """Canonical inventory without resolving every ordinary file separately.

    scandir supplies each entry's actual name and cached file type. Resolve a
    directory root once, then only resolve links/reparse points encountered
    below it. As with Path.rglob's default, directory symlinks are not followed;
    directory junctions follow DirEntry.is_dir(follow_symlinks=False) semantics.
    File symlinks still contribute the resolved target path. JSON inclusion and
    __pycache__ filtering use the original logical root/entry names.
    """
    found = set()

    def is_reparse(entry):
        # On Windows this stat is served from scandir's WIN32_FIND_DATA cache;
        # unlike Path.resolve(), it does not open each ordinary file by handle.
        return os.name == "nt" and bool(
            entry.stat(follow_symlinks=False).st_file_attributes & 0x400
        )

    try:
        for entry in roots:
            root = Path(entry)
            if root.is_file():
                found.add(str(root.resolve()))
            elif root.is_dir():
                # Explicit profile roots include configuration additions/removals.
                # General JSON stays excluded: generated result reports are not code.
                suffixes = SOURCE_SUFFIXES | (
                    {".json"} if root.name == "profiles" else set()
                )
                if "__pycache__" in root.parts:
                    continue
                stack = [str(root.resolve())]
                visited = set()
                while stack:
                    directory = stack.pop()
                    if directory in visited:
                        continue
                    visited.add(directory)
                    with os.scandir(directory) as entries:
                        for child in entries:
                            if child.name == "__pycache__":
                                continue
                            if child.is_dir(follow_symlinks=False):
                                nested = child.path
                                if is_reparse(child):
                                    nested = str(Path(nested).resolve())
                                stack.append(nested)
                            elif (
                                os.path.splitext(child.name)[1].lower() in suffixes
                                and child.is_file()
                            ):
                                path = child.path
                                if child.is_symlink() or is_reparse(child):
                                    path = str(Path(path).resolve())
                                found.add(path)
            else:
                raise StaleManifest(
                    f"runtime source root missing or not a file/directory: {root}"
                )
    except (OSError, TypeError, ValueError) as exc:
        raise StaleManifest(f"runtime source inventory unavailable: {exc}") from exc
    return sorted(found)


def _identity(stat):
    # CPython 3.13 Windows fstat reports modification time in st_ctime while
    # Path.stat reports creation time. Volume/file ID still detects replacement;
    # every byte is hashed, including mutations preserving these timestamps.
    return (
        stat.st_size,
        stat.st_mtime_ns,
        None if os.name == "nt" else stat.st_ctime_ns,
        stat.st_dev,
        stat.st_ino,
    )


def digest_file(path):
    path = Path(path)
    try:
        with path.open("rb") as f:
            before = os.fstat(f.fileno())
            size = before.st_size
            hashes = []
            if size:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    view = memoryview(mm)
                    try:
                        for at in range(0, size, CHUNK):
                            hashes.append(
                                hashlib.sha256(view[at : at + CHUNK]).hexdigest()
                            )
                    finally:
                        view.release()
            if _identity(os.fstat(f.fileno())) != _identity(before) or _identity(
                path.stat()
            ) != _identity(before):
                raise StaleManifest(f"file changed while fingerprinting: {path}")
        return {
            "path": str(path.resolve()),
            "nbytes": size,
            "chunk_bytes": CHUNK,
            "chunks_sha256": hashes,
        }
    except OSError as exc:
        raise StaleManifest(f"cannot fingerprint dependency {path}: {exc}") from exc


def fingerprint_many(paths, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(digest_file, paths))


def seal(manifest):
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    return manifest


_ENVELOPE_PREFIX = b'{"version":2,"sha256":"'
_ENVELOPE_MIDDLE = b'","payload":'
_ENVELOPE_SUFFIX = b"}\n"


def write_manifest(path, manifest):
    """Write a byte-sealed JSON envelope; retain the version-1 runtime schema.

    The fixed framing lets readers verify the exact payload bytes before their
    sole JSON parse, without recreating a canonical serialization at startup.
    The digest is an integrity check, with the same trust model as ``seal``.
    """
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise StaleManifest("cannot write an unsupported runtime manifest schema")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = canonical(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    envelope = (
        _ENVELOPE_PREFIX
        + digest.encode("ascii")
        + _ENVELOPE_MIDDLE
        + encoded
        + _ENVELOPE_SUFFIX
    )
    Path(path).write_bytes(envelope)
    manifest["manifest_sha256"] = digest
    return manifest


def _read_manifest(path):
    """Read and digest-check a manifest, then resolve @ROOT@/@MODEL@ placeholders and overlay the install receipt (plamo_root.prepare).
    Both happen strictly after the raw envelope digest has passed."""
    manifest = _read_manifest_raw(path)
    return plamo_root.prepare(manifest, StaleManifest, _notify)


def _notify(message, key=None):
    try:
        from plamo_errors import note

        note(message, key=key)
    except Exception:
        pass


def _read_manifest_raw(path):
    encoded = Path(path).read_bytes()
    if encoded.startswith(_ENVELOPE_PREFIX):
        hash_start = len(_ENVELOPE_PREFIX)
        hash_end = hash_start + 64
        payload_start = hash_end + len(_ENVELOPE_MIDDLE)
        claimed = encoded[hash_start:hash_end]
        if (
            len(encoded) <= payload_start + len(_ENVELOPE_SUFFIX)
            or len(claimed) != 64
            or any(c not in b"0123456789abcdef" for c in claimed)
            or encoded[hash_end:payload_start] != _ENVELOPE_MIDDLE
            or not encoded.endswith(_ENVELOPE_SUFFIX)
        ):
            raise StaleManifest("malformed version-2 manifest envelope")
        payload_end = len(encoded) - len(_ENVELOPE_SUFFIX)
        digest = hashlib.sha256(
            memoryview(encoded)[payload_start:payload_end]
        ).hexdigest()
        if claimed.decode("ascii") != digest:
            raise StaleManifest("runtime manifest content changed; rebuild the bundle")
        # No parser consumes payload bytes until their raw digest has passed.
        manifest = json.loads(encoded[payload_start:payload_end])
        if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
            raise StaleManifest("version-2 envelope has an unsupported payload schema")
        if "manifest_sha256" in manifest:
            raise StaleManifest("version-2 payload must not contain its own digest")
        manifest["manifest_sha256"] = digest
        return manifest

    # Existing bundles remain readable. An envelope with changed framing must
    # never be interpreted as a legacy manifest, even if it is valid JSON.
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict):
        raise StaleManifest("runtime manifest must be an object")
    if manifest.get("version") == 2 or "sha256" in manifest or "payload" in manifest:
        raise StaleManifest("noncanonical version-2 manifest envelope")
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != hashlib.sha256(canonical(manifest)).hexdigest():
        raise StaleManifest("runtime manifest content changed; rebuild the bundle")
    manifest["manifest_sha256"] = claimed
    return manifest


def _record_path(item):
    if not isinstance(item, dict):
        raise StaleManifest("digest record must be an object")
    try:
        path = Path(item["path"])
        nbytes, chunk_bytes, hashes = (
            item["nbytes"],
            item["chunk_bytes"],
            item["chunks_sha256"],
        )
        if not path.is_absolute():
            raise StaleManifest(f"dependency path must be absolute: {path}")
        if (
            type(nbytes) is not int
            or nbytes < 0
            or type(chunk_bytes) is not int
            or chunk_bytes <= 0
        ):
            raise StaleManifest(f"invalid digest size or chunk width: {path}")
        if (
            not isinstance(hashes, list)
            or len(hashes) != (nbytes + chunk_bytes - 1) // chunk_bytes
        ):
            raise StaleManifest(f"bad digest table: {path}")
        if any(
            not isinstance(h, str)
            or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h)
            for h in hashes
        ):
            raise StaleManifest(f"bad SHA256 digest: {path}")
        return path
    except (KeyError, TypeError, ValueError) as exc:
        raise StaleManifest(f"malformed dependency digest record: {exc}") from exc


def load(path, *, event=None):
    path = Path(path)
    started = time.perf_counter()
    try:
        manifest = _read_manifest(path)
        if manifest.get("version") != VERSION:
            raise StaleManifest("runtime manifest schema changed; rebuild the bundle")
        if not plamo_root.config_equal(manifest["configuration"], configuration()):
            raise StaleManifest("runtime configuration changed; rebuild the bundle (differing keys: " + ", ".join(plamo_root.config_diff(manifest["configuration"], configuration())) + ")")
        if "python_abi" in manifest and not plamo_root.abi_equal(manifest["python_abi"], python_abi()):
            raise StaleManifest("runtime Python ABI changed; rebuild the bundle")
        roots, sources = manifest["source_roots"], manifest["sources"]
        if not isinstance(roots, list) or not all(
            isinstance(p, str) and p for p in roots
        ):
            raise StaleManifest("runtime source roots must be a list of paths")
        if not isinstance(sources, list) or not isinstance(
            manifest.get("origins", []), list
        ):
            raise StaleManifest("runtime source and origin records must be lists")
        if not isinstance(manifest["artifacts"], dict):
            raise StaleManifest("runtime artifact records must be an object")
        for record in [
            *sources,
            *manifest.get("origins", []),
            *manifest["artifacts"].values(),
        ]:
            _record_path(record)
        if event is not None:
            event(
                "validation_manifest_json_schema_config", time.perf_counter() - started
            )
        started = time.perf_counter()
        if source_files(roots) != [r["path"] for r in sources]:
            raise StaleManifest("runtime source inventory changed; rebuild the bundle")
        if event is not None:
            event("validation_source_inventory", time.perf_counter() - started)
        return manifest
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise StaleManifest(
            f"runtime manifest unavailable or malformed: {path}: {exc}"
        ) from exc


def validate_records(records, workers=8):
    """Hash all bytes, including large payloads; timestamps are never trusted."""
    owners, views, jobs, before = [], [], [], []
    try:
        for item in records:
            path = _record_path(item)
            stat = path.stat()
            if stat.st_size != item["nbytes"]:
                raise StaleManifest(f"file size changed: {path}")
            f = path.open("rb")
            owners.append(f)
            if _identity(os.fstat(f.fileno())) != _identity(stat):
                raise StaleManifest(f"file changed while opening: {path}")
            before.append((path, f, _identity(stat)))
            if not stat.st_size:
                continue
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            owners.append(mm)
            view = memoryview(mm)
            views.append(view)
            for index, expected in enumerate(item["chunks_sha256"]):
                lo = index * item["chunk_bytes"]
                jobs.append(
                    (
                        view,
                        lo,
                        min(lo + item["chunk_bytes"], stat.st_size),
                        expected,
                        str(path),
                    )
                )

        def check(job):
            view, lo, hi, expected, path = job
            if hashlib.sha256(view[lo:hi]).hexdigest() != expected:
                raise StaleManifest(
                    f"content changed: {path} at byte {lo}; rebuild bundle"
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(check, jobs))
        for path, stream, identity in before:
            if (
                _identity(path.stat()) != identity
                or _identity(os.fstat(stream.fileno())) != identity
            ):
                raise StaleManifest(f"file changed during validation: {path}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise StaleManifest(f"dependency validation failed: {exc}") from exc
    finally:
        jobs.clear()
        for view in views:
            view.release()
        for obj in reversed(owners):
            obj.close()


def validated(path, workers=8):
    manifest = load(path)
    records = [
        *manifest["sources"],
        *manifest.get("origins", []),
        *manifest["artifacts"].values(),
    ]
    validate_records(records, workers)
    return manifest, {
        name: Path(item["path"]) for name, item in manifest["artifacts"].items()
    }


_WINDOWS_FILE_API = None


def _open_retained(path):
    """Deny concurrent writes/deletes on Windows, including through hard links.

    Read-only mmap alone does not prevent writes to its underlying file. The
    Windows runtime therefore retains a non-share-write/non-share-delete handle
    until every consumer has finished. The POSIX branch supports CPU tests; its
    final stat checks detect ordinary mutations but are not a Windows lease.
    """
    if os.name != "nt":
        return Path(path).open("rb")
    global _WINDOWS_FILE_API
    if _WINDOWS_FILE_API is None:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        _WINDOWS_FILE_API = ctypes, msvcrt, create, close
    ctypes, msvcrt, create, close = _WINDOWS_FILE_API
    # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL.
    handle = create(str(path), 0x80000000, 0x00000001, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    try:
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


class ValidationLease:
    """Pin dependencies, hash in parallel, and explicitly authorize execution.

    Context xclbins are checked synchronously before their paths can reach XRT.
    Other hashes run concurrently with buffer/context/tokenizer preparation.
    ``wait`` is the sole commit point: all hashes, identities, configuration,
    and the complete source inventory must pass. ``require_validated`` never
    waits and rejects pending or failed validation before device submission.

    Consumers may read ``artifact(name)`` during preparation but must retain
    this lease until their last mapping use. Only Windows has mandatory
    deny-write/delete sharing; the non-Windows path exists for CPU tests.
    """

    def __init__(self, path, workers=8, *, event=None):
        if type(workers) is not int or workers < 1:
            raise ValueError("validation workers must be a positive integer")
        self.path = Path(path).resolve()
        self.event = event
        self._owners = {}
        self._pool = None
        self._futures = []
        self._completion = None
        self._validated = False
        self._failure = None
        self._closed = False
        self._manifest_stream = None
        self._manifest_identity = None
        self._hash_started = None
        self._hash_finished = None
        self.stats = {
            "mandatory_write_delete_exclusion": os.name == "nt",
            "validated": False,
            "hash_workers": workers,
        }
        started = time.perf_counter()
        self._started = started
        try:
            # Lock before parsing: manifest path cannot be swapped underneath
            # the parsed structure while preparation is using that structure.
            self._manifest_stream = _open_retained(self.path)
            self._manifest_identity = _identity(
                os.fstat(self._manifest_stream.fileno())
            )
            self._span("validation_manifest_lock", started)
            self.manifest = load(self.path, event=self.event)
            self.paths = {
                name: Path(item["path"])
                for name, item in self.manifest["artifacts"].items()
            }
            records = [
                *self.manifest["sources"],
                *self.manifest.get("origins", []),
                *self.manifest["artifacts"].values(),
            ]
            pin_started = time.perf_counter()
            for item in records:
                self._pin(item)
            self._span("validation_dependency_open_and_map", pin_started)
            self.stats.update(
                dependency_records=len(records),
                unique_files=len(self._owners),
                unique_bytes=sum(item[2][0] for item in self._owners.values()),
                hashed_bytes=sum(item["nbytes"] for item in records),
                source_files=len(self.manifest["sources"]),
                source_bytes=sum(item["nbytes"] for item in self.manifest["sources"]),
                origin_files=len(self.manifest.get("origins", [])),
                origin_bytes=sum(
                    item["nbytes"] for item in self.manifest.get("origins", [])
                ),
                artifact_files=len(self.manifest["artifacts"]),
                artifact_bytes=sum(
                    item["nbytes"] for item in self.manifest["artifacts"].values()
                ),
                snapshot_limit_bytes=SMALL_SNAPSHOT,
                snapshot_files=sum(
                    isinstance(item[1], bytes) for item in self._owners.values()
                ),
                snapshot_bytes=sum(
                    len(item[1])
                    for item in self._owners.values()
                    if isinstance(item[1], bytes)
                ),
                mapped_files=sum(
                    isinstance(item[1], mmap.mmap) for item in self._owners.values()
                ),
                mapped_bytes=sum(
                    item[2][0]
                    for item in self._owners.values()
                    if isinstance(item[1], mmap.mmap)
                ),
            )
            self._span("validation_manifest_inventory_and_pin", started)

            eager = {
                item["artifact"] for item in self.manifest.get("contexts", {}).values()
            }
            eager_records = [self.manifest["artifacts"][name] for name in sorted(eager)]
            self.stats["eager_xclbin_bytes"] = sum(
                item["nbytes"] for item in eager_records
            )
            started = time.perf_counter()
            for item in eager_records:
                for job in self._jobs(item):
                    self._check_chunk(job)
            self._span("validation_xclbins_before_context", started)

            eager_ids = {id(record) for record in eager_records}
            remaining = [record for record in records if id(record) not in eager_ids]
            self._hash_started = time.perf_counter()
            # Start long C/hashlib calls first so they can run with the GIL
            # released while XRT setup enters native calls. Record chunk tables
            # remain authoritative, including older bundles using smaller chunks.
            jobs = [job for record in remaining for job in self._jobs(record)]
            jobs.sort(key=lambda job: job[2] - job[1], reverse=True)
            self.stats["background_hash_jobs"] = len(jobs)
            self.stats["largest_hash_job_bytes"] = (
                jobs[0][2] - jobs[0][1] if jobs else 0
            )
            self.stats["record_chunk_bytes"] = sorted(
                {record["chunk_bytes"] for record in records}
            )
            batches = list(self._batch_jobs(jobs))
            self.stats["background_hash_pool_tasks"] = len(batches)
            self.stats["hash_batch_limit_bytes"] = HASH_BATCH_BYTES
            self.stats["hash_batch_limit_jobs"] = HASH_BATCH_JOBS
            self._pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="runtime-validation"
            )
            self._futures.extend(
                self._pool.submit(self._check_batch, batch) for batch in batches
            )
            # Submitted after all hash jobs: this cannot occupy every worker
            # while waiting for unstarted jobs from the same pool.
            self._completion = self._pool.submit(self._finish_hashes)
            self._span("validation_hash_task_submission", self._hash_started)
        except BaseException:
            self.close()
            raise

    def _span(self, name, start):
        if self.event is not None:
            self.event(name, time.perf_counter() - start)

    def _pin(self, item):
        path = _record_path(item)
        key = str(path)
        if key in self._owners:
            if self._owners[key][2][0] != item["nbytes"]:
                raise StaleManifest(f"inconsistent digest records for {path}")
            return
        stream = None
        mapping = None
        try:
            stream = _open_retained(path)
            identity = _identity(os.fstat(stream.fileno()))
            if identity[0] != item["nbytes"] or _identity(path.stat()) != identity:
                raise StaleManifest(
                    f"file changed while acquiring validation lease: {path}"
                )
            if identity[0] <= SMALL_SNAPSHOT:
                # Keep the sharing lock even after this bounded immutable copy.
                # Avoid a file-mapping object for each tiny source/header/config.
                mapping = stream.read(identity[0])
                if (
                    len(mapping) != identity[0]
                    or _identity(os.fstat(stream.fileno())) != identity
                ):
                    raise StaleManifest(
                        f"file changed while taking readonly snapshot: {path}"
                    )
            else:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self._owners[key] = (stream, mapping, identity)
        except BaseException:
            if isinstance(mapping, mmap.mmap):
                mapping.close()
            if stream is not None:
                stream.close()
            raise

    def _jobs(self, item):
        mapping = self._owners[item["path"]][1]
        for index, expected in enumerate(item["chunks_sha256"]):
            lo = index * item["chunk_bytes"]
            yield (
                mapping,
                lo,
                min(lo + item["chunk_bytes"], item["nbytes"]),
                expected,
                item["path"],
            )

    @staticmethod
    def _check_chunk(job):
        mapping, lo, hi, expected, path = job
        with memoryview(mapping)[lo:hi] as view:
            actual = hashlib.sha256(view).hexdigest()
        if actual != expected:
            raise StaleManifest(f"content changed: {path} at byte {lo}; rebuild bundle")

    @staticmethod
    def _batch_jobs(jobs):
        """Keep large hashes independent; bound small batches by bytes and count."""
        batch, size = [], 0
        for job in jobs:
            nbytes = job[2] - job[1]
            if nbytes > HASH_BATCH_BYTES:
                if batch:
                    yield tuple(batch)
                    batch, size = [], 0
                yield (job,)
                continue
            if batch and (
                len(batch) >= HASH_BATCH_JOBS or size + nbytes > HASH_BATCH_BYTES
            ):
                yield tuple(batch)
                batch, size = [], 0
            batch.append(job)
            size += nbytes
        if batch:
            yield tuple(batch)

    def _check_batch(self, jobs):
        # Every original record/chunk keeps its own expected digest and error
        # location; batching changes scheduler overhead, not validation scope.
        for job in jobs:
            self._check_chunk(job)

    def _finish_hashes(self):
        try:
            for future in self._futures:
                future.result()
        except BaseException as exc:
            self._failure = exc
            raise
        finally:
            self._hash_finished = time.perf_counter()
            self._span("validation_all_remaining_content_hashes", self._hash_started)

    def artifact(self, name):
        if self._closed:
            raise StaleManifest("validation lease is closed")
        if self._failure is not None:
            raise StaleManifest("validation lease failed") from self._failure
        mapping = self._owners[str(self.paths[name])][1]
        return b"" if mapping is None else mapping

    def join_hashes(self):
        """Always join outstanding work, including when setup failed first."""
        try:
            if self._completion is not None:
                self._completion.result()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

    def _check_stable(self):
        started = time.perf_counter()
        if (
            _identity(self.path.stat()) != self._manifest_identity
            or _identity(os.fstat(self._manifest_stream.fileno()))
            != self._manifest_identity
        ):
            raise StaleManifest("manifest changed during validation")
        for path, (stream, _, identity) in self._owners.items():
            if (
                _identity(Path(path).stat()) != identity
                or _identity(os.fstat(stream.fileno())) != identity
            ):
                raise StaleManifest(f"file changed during validation: {path}")
        self._span("validation_final_identity_checks", started)
        started = time.perf_counter()
        if source_files(self.manifest["source_roots"]) != [
            r["path"] for r in self.manifest["sources"]
        ]:
            raise StaleManifest("runtime source inventory changed during validation")
        if not plamo_root.config_equal(self.manifest["configuration"], configuration()):
            raise StaleManifest("runtime configuration changed during validation")
        self._span("validation_final_inventory_and_config", started)

    def wait(self):
        if self._closed:
            raise StaleManifest("validation lease is closed")
        if self._validated:
            return self.manifest, self.paths
        started = time.perf_counter()
        try:
            self.join_hashes()
            self._span("validation_hash_join_wait", started)
            if self._failure is not None:
                raise self._failure
            self._check_stable()
            self._validated = True
            self.stats["validated"] = True
            self.stats["validation_wall_ms"] = (
                time.perf_counter() - self._started
            ) * 1000
        except BaseException as exc:
            self._failure = exc
            raise
        finally:
            self._span("validation_execution_barrier", started)
        return self.manifest, self.paths

    def require_validated(self):
        if self._closed:
            raise StaleManifest("validation lease is closed")
        if self._failure is not None:
            raise StaleManifest("validation lease failed") from self._failure
        if not self._validated:
            raise StaleManifest("validation is pending; NPU execution is prohibited")

    def close(self):
        if self._closed:
            return
        # Do not mask an earlier validation/setup error during cleanup.
        try:
            self.join_hashes()
        except BaseException:
            pass
        self._futures.clear()
        self._completion = None
        for stream, mapping, _ in reversed(list(self._owners.values())):
            if isinstance(mapping, mmap.mmap):
                mapping.close()
            stream.close()
        self._owners.clear()
        if self._manifest_stream is not None:
            self._manifest_stream.close()
            self._manifest_stream = None
        self._closed = True
