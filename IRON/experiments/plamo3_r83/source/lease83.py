"""R83 identity lease: same attestation, shorter and less jitter-sensitive startup critical path.

Unchanged (verbatim semantics of identity43/lease61): the set of attested files (sources + origins + artifacts), every compared identity field,
the content-hash fallback, retention of runtime inputs with write/delete exclusion, the commit re-query of every retained handle, the commit
configuration check, the commit source-inventory check, the order "all validation passed -> _validated=True -> require_validated()".

Changed (scheduling only, nothing is removed or narrowed):
 1. The 1404 source-only files (not retained, never sent to the device) are attested in a helper process (srccheck83) that starts before the manifest is
    parsed. It also performs the record-schema check (_record_path) for those records and the load-time source-inventory comparison. The parent attests
    the 804 retained runtime inputs + 3 executables itself. A helper failure/timeout makes the parent attest the same records in-process.
 2. The ctor returns once every retained runtime input (and the executables) is attested; the helper result is joined at the START of the commit
    (IdentityLease.wait) and a refusal there happens before _validated is set, i.e. before on_validated (worker authorization, package preload) and before
    any device execution (DirectRuntime.execute calls validation_barrier = require_validated). BO allocation/upload of already-attested artifacts may
    proceed meanwhile (the R46 design already overlapped them with the commit).
 3. The commit's re-queries and its source-inventory walk run concurrently.
 4. The manifest's second (expanded-manifest) SHA-256 is computed in a thread concurrently with the catalog decode; both must pass before load returns.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import mmap
import os
from pathlib import Path
import threading
import time

import identity43
import plamo_root
import srccheck83

I = identity43
base = identity43.base

# Where the retained-input attestation runs relative to the early-XRT calls (device open holds the GIL): True = before them, False = right after
# device_open, concurrently with the hw_context thread (R61 order). Fixed by the Phase B measurement (ROUND65.md: False was ~17 ms faster).
# Plain constants (no environment switches in the production path); gates and measurements patch the module attributes.
INPUTS_FIRST = False
USE_CHECKER = True
APP_DIR = str(Path(base.__file__).resolve().parent)


def _read_certificate(path, owners):
    """compact_catalog.read_certificate with the expanded-manifest hash overlapped with the catalog decode (same checks, same refusal)."""
    import compact_catalog as cc

    c = base._read_manifest(path)
    if c.get("kind") != cc.MAGIC:
        raise base.StaleManifest("not an R43 compact certificate")
    streams = {}
    for name in ("catalog", "expanded_manifest"):
        record = c[name]
        file = base._record_path(record)
        stream = base._open_retained(file)
        owners.append(stream)
        streams[name] = (record, stream)
    box = {}

    def hash_stream(name):
        record, stream = streams[name]
        h = hashlib.sha256()
        chunks, total = [], 0
        try:
            while True:
                b = stream.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
                total += len(b)
                if name == "catalog":
                    chunks.append(b)
            box[name] = (total, h.hexdigest(), chunks)
        except BaseException as exc:
            box[name] = exc

    thread = threading.Thread(target=hash_stream, args=("expanded_manifest",), name="r83-expanded-hash")
    thread.start()
    try:
        hash_stream("catalog")
        record = streams["catalog"][0]
        got = box["catalog"]
        if isinstance(got, BaseException):
            raise got
        if got[0] != record["nbytes"] or got[1] != record["sha256"]:
            raise base.StaleManifest("compact content hash mismatch: catalog")
        # The catalog digest has passed before its bytes are decoded (the expanded manifest is only hash-checked, never parsed).
        m = cc.decode(b"".join(got[2]))  # compact_catalog.decode resolves placeholders and overlays the receipt (plamo_root.prepare)
    finally:
        thread.join()
    record = streams["expanded_manifest"][0]
    got = box["expanded_manifest"]
    if isinstance(got, BaseException):
        raise got
    if got[0] != record["nbytes"] or got[1] != record["sha256"]:
        raise base.StaleManifest("compact content hash mismatch: expanded_manifest")
    m["manifest_sha256"] = c["expanded_payload_sha256"]
    return m


def load_deferred(path, *, event=None, owners=None, source_record_check=False):
    """manifest43.load without the source-inventory walk and (optionally) without the per-record schema check of the *source* records.
    Both are performed by the source helper (or in-process by the fallback) before the identity commit."""
    path = Path(path)
    started = time.perf_counter()
    try:
        manifest = _read_certificate(path, owners)
        if manifest.get("version") != base.VERSION:
            raise base.StaleManifest("runtime manifest schema changed; rebuild the bundle")
        if not plamo_root.config_equal(manifest["configuration"], base.configuration()):
            raise base.StaleManifest("runtime configuration changed; rebuild the bundle")
        if "python_abi" in manifest and not plamo_root.abi_equal(manifest["python_abi"], base.python_abi()):
            raise base.StaleManifest("runtime Python ABI changed; rebuild the bundle")
        roots, sources = manifest["source_roots"], manifest["sources"]
        if not isinstance(roots, list) or not all(isinstance(p, str) and p for p in roots):
            raise base.StaleManifest("runtime source roots must be a list of paths")
        if not isinstance(sources, list) or not isinstance(manifest.get("origins", []), list):
            raise base.StaleManifest("runtime source and origin records must be lists")
        if not isinstance(manifest["artifacts"], dict):
            raise base.StaleManifest("runtime artifact records must be an object")
        runtime_records = [*manifest.get("origins", []), *manifest["artifacts"].values()]
        for record in runtime_records if source_record_check else [*sources, *runtime_records]:
            base._record_path(record)
        if source_record_check:
            # Only the minimum the parent needs to group/hand over the source records; the full schema check runs in the helper/fallback.
            for record in sources:
                if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                    raise base.StaleManifest("digest record must be an object")
        if event is not None:
            event("validation_manifest_json_schema_config", time.perf_counter() - started)
        return manifest
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise base.StaleManifest(f"runtime manifest unavailable or malformed: {path}: {exc}") from exc


class IdentityLease(identity43.IdentityLease):
    def __init__(self, path, workers=4, *, event=None, force_audit=False, executable_ready=None):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("identity workers must be between 1 and 16")
        self.path, self.event = Path(path), event
        self._owners, self._failure = {}, None
        self._catalog_owners = []
        self._validated, self._closed = False, False
        self._executables_validated = False
        self._setup_join = None
        self._checker = None
        self._sources_joined = False
        self._force_audit = force_audit
        self._workers = workers
        self.stats = dict(
            mode="ntfs-identity",
            validated=False,
            identity_matches=0,
            fallback_files=0,
            hashed_bytes=0,
            fallback_paths=[],
            mandatory_write_delete_exclusion=os.name == "nt",
            source_attestation="pending",
        )
        self._manifest_stream = base._open_retained(self.path)
        try:
            if USE_CHECKER:
                self._checker = srccheck83.SourceChecker.spawn(APP_DIR)
            self.manifest = load_deferred(
                self.path, event=event, owners=self._catalog_owners, source_record_check=self._checker is not None
            )
            self.paths = {name: Path(r["path"]) for name, r in self.manifest["artifacts"].items()}
            records = [
                *self.manifest["sources"],
                *self.manifest.get("origins", []),
                *self.manifest["artifacts"].values(),
            ]
            groups = {}
            for record in records:
                groups.setdefault(record["path"], []).append(record)
            runtime_paths = {
                r["path"] for r in [*self.manifest.get("origins", []), *self.manifest["artifacts"].values()]
            }
            artifact_paths = {str(p) for p in self.paths.values()}
            started = time.perf_counter()
            version_ok = self.manifest.get("filesystem_identity_version") == 1

            def check(item):
                name, group = item
                stream, mapping = base._open_retained(name), None
                try:
                    identity = I.maybe_identity(stream)
                    match = (
                        not force_audit
                        and version_ok
                        and identity is not None
                        and all(r.get("ntfs_identity") == identity for r in group)
                    )
                    hashed = 0 if match else I.check_content(stream, group)
                    if not match and I.maybe_identity(stream) != identity:
                        raise base.StaleManifest(f"dependency changed during fallback: {name}")
                    if name in artifact_paths and group[0]["nbytes"]:
                        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                    owner = (stream, mapping, identity) if name in runtime_paths else None
                    if owner is None:
                        stream.close()
                    return name, owner, match, hashed
                except BaseException:
                    if mapping is not None:
                        mapping.close()
                    stream.close()
                    raise

            items = list(groups.items())
            self._source_items = [(n, g) for n, g in items if n not in runtime_paths]
            runtime_items = [(n, g) for n, g in items if n in runtime_paths]
            self._source_request = dict(
                roots=self.manifest["source_roots"],
                expected_inventory=[r["path"] for r in self.manifest["sources"]],
                items=self._source_items,
                version_ok=version_ok,
                force_audit=force_audit,
            )
            if self._checker is not None and not self._checker.send(self._source_request):
                self._checker = None
            if self._checker is not None:
                self._source_request = None  # only needed for the in-process fallback
            if self._checker is None:
                runtime_items = items  # helper unavailable: attest everything in the ctor (R61 behaviour)

            def merge(name, owner, match, hashed):
                if owner:
                    self._owners[name] = owner
                self.stats["identity_matches"] += int(match)
                self.stats["fallback_files"] += int(not match)
                self.stats["hashed_bytes"] += hashed
                if not match:
                    self.stats["fallback_paths"].append(name)

            def batch(batch_items):
                result = []
                try:
                    for item in batch_items:
                        result.append(check(item))
                    return result
                except BaseException:
                    for _, owner, _, _ in result:
                        if owner:
                            if owner[1] is not None:
                                owner[1].close()
                            owner[0].close()
                    raise

            def attest_runtime(todo):
                batches = [todo[i::workers] for i in range(workers)]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(batch, b) for b in batches]
                    error = None
                    for future in futures:
                        try:
                            for row in future.result():
                                merge(*row)
                        except BaseException as exc:
                            error = error or exc
                    if error:
                        raise error

            if executable_ready is not None:
                executable_paths = {str(self.paths[c["artifact"]]) for c in self.manifest["contexts"].values()}
                for name in sorted(executable_paths):
                    row = check((name, groups[name]))
                    if row[1] is None:
                        raise base.StaleManifest("executable input must remain pinned")
                    merge(*row)
                self._executables_validated = True
                if event:
                    event("validation_executables_ready", time.perf_counter() - started)
                runtime_items = [(n, g) for n, g in runtime_items if n not in executable_paths]
                if INPUTS_FIRST:
                    attest_runtime(runtime_items)
                    self._setup_join = executable_ready(self)
                else:
                    self._setup_join = executable_ready(self)
                    attest_runtime(runtime_items)
            else:
                attest_runtime(runtime_items)
            self.stats.update(
                unique_files=len(groups),
                retained_files=len(self._owners),
                identity_workers=workers,
            )
            if self._checker is None:
                self.stats["source_attestation"] = "in-process (ctor)"
                self._sources_joined = True
                self._inprocess_record_and_inventory()
            if event:
                event("validation_ntfs_identity_and_fallback", time.perf_counter() - started)
            self._poll_sources()
        except BaseException as exc:
            self._failure = exc
            if self._checker is not None:
                self._checker.close()
            if self._setup_join is not None:
                try:
                    self._setup_join()
                except BaseException:
                    pass
            self.close()
            raise

    # ---- source attestation (helper result or in-process fallback) -------------------------------------------------
    def _inprocess_record_and_inventory(self):
        for record in self.manifest["sources"]:
            base._record_path(record)
        if base.source_files(self.manifest["source_roots"]) != [r["path"] for r in self.manifest["sources"]]:
            raise base.StaleManifest("runtime source inventory changed; rebuild the bundle")

    def _apply_helper(self, result):
        if not result.get("ok"):
            kind, message = result.get("kind"), result.get("message", "source attestation failed")
            if kind == "os":
                raise OSError(message)
            raise base.StaleManifest(message)
        if result["files"] != len(self._source_items) or result["inventory_files"] != len(self.manifest["sources"]):
            return False  # unintelligible: caller attests in-process
        self.stats["identity_matches"] += result["matches"]
        self.stats["fallback_files"] += result["files"] - result["matches"]
        self.stats["hashed_bytes"] += result["hashed_bytes"]
        self.stats["fallback_paths"].extend(result["fallback_paths"])
        self.stats["source_attestation"] = "helper-process"
        self.stats["source_helper_ms"] = result["child_ms"]
        self.stats["source_helper_inventory_ms"] = result.get("inventory_ms")
        if self.event:
            self.event("validation_source_inventory", result.get("inventory_ms", 0.0) / 1000)
        return True

    def _poll_sources(self):
        """End of the ctor: if the helper already finished with a refusal, refuse now (earliest possible)."""
        if self._checker is None or self._sources_joined:
            return
        result = self._checker.poll()
        if result and not result.get("ok"):
            self._join_sources(result)

    def _join_sources(self, result=None):
        if self._sources_joined:
            return
        if self._checker is not None:
            if result is None:
                result = self._checker.result()
            if result:
                if self._apply_helper(result):
                    self._sources_joined = True
                    return
        # helper unusable: attest the same records in-process (never skipped)
        self._attest_sources_in_process()
        self._sources_joined = True

    def _attest_sources_in_process(self):
        items = self._source_items
        version_ok = self.manifest.get("filesystem_identity_version") == 1
        self._inprocess_record_and_inventory()
        workers = self._workers
        batches = [items[i::workers] for i in range(workers)]

        def run(batch_items):
            return srccheck83.check_items(
                batch_items,
                version_ok=version_ok,
                force_audit=self._force_audit,
                open_retained=lambda n: base._open_retained(n),
                identity=lambda s: I.maybe_identity(s),
                content=lambda s, g: I.check_content(s, g),
                stale=base.StaleManifest,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, b) for b in batches]
            error = None
            for future in futures:
                try:
                    matches, hashed, fallback = future.result()
                    self.stats["identity_matches"] += matches
                    self.stats["fallback_files"] += len(fallback)
                    self.stats["hashed_bytes"] += hashed
                    self.stats["fallback_paths"].extend(fallback)
                except BaseException as exc:
                    error = error or exc
            if error:
                raise error
        self.stats["source_attestation"] = "in-process (fallback)"

    # ---- commit ----------------------------------------------------------------------------------------------------
    def wait(self):
        if self._failure:
            raise self._failure
        if self._closed:
            raise base.StaleManifest("closed identity lease")
        if self._validated:
            return
        started = time.perf_counter()
        try:
            owners = list(self._owners.items())

            def check(batch):
                for name, (stream, _, before) in batch:
                    if I.maybe_identity(stream) != before:
                        raise base.StaleManifest("runtime input identity changed during setup: " + name)

            # Distinct retained handles; these queries never seek/read their bytes. The source-inventory walk (a directory enumeration,
            # no handle shared with these) runs concurrently on this thread. Every future completes (including on failure) before returning.
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="r83-commit") as pool:
                checks = [pool.submit(check, owners[i::3]) for i in range(3)]
                inventory_error = None
                try:
                    inventory = base.source_files(self.manifest["source_roots"])
                except BaseException as exc:
                    inventory, inventory_error = None, exc
                for future in checks:
                    future.result()
            if not plamo_root.config_equal(self.manifest["configuration"], base.configuration()):
                raise base.StaleManifest("runtime configuration changed during setup")
            if inventory_error is not None:
                raise inventory_error
            if inventory != [r["path"] for r in self.manifest["sources"]]:
                raise base.StaleManifest("source inventory changed during setup")
            # Source attestation (helper or fallback) must have passed before the commit.
            self._join_sources()
            self._validated = True
            self.stats["validated"] = True
            self.stats["commit_identity_workers"] = 3
            if self.event:
                self.event("validation_identity_commit", time.perf_counter() - started)
        except BaseException as exc:
            self._failure = exc
            raise

    def close(self):
        if self._checker is not None:
            self._checker.close()
        super().close()
