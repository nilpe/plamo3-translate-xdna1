# SPDX-License-Identifier: Apache-2.0
"""R21 NTFS change identities, with full SHA-256 fallback and retained inputs.

A matching identity is an export-time content attestation, not an mtime cache.
Every lookup includes the file ID, volume serial, EOF, last-write time, USN and
journal ID. Unsupported/absent journals always take the content path. Sources
are locked while checked, then released; runtime artifacts/origins stay pinned.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import mmap
import os
from pathlib import Path
import struct
import time

import runtime_manifest as base

_API = None


def api():
    global _API
    if _API is None:
        if os.name != "nt":
            raise OSError("NTFS identities require Windows")
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt

        k = c.WinDLL("kernel32", use_last_error=True)
        k.GetFileInformationByHandle.argtypes = [w.HANDLE, c.c_void_p]
        k.GetFileInformationByHandle.restype = w.BOOL
        k.GetFileInformationByHandleEx.argtypes = [
            w.HANDLE,
            c.c_int,
            c.c_void_p,
            w.DWORD,
        ]
        k.GetFileInformationByHandleEx.restype = w.BOOL
        k.DeviceIoControl.argtypes = [
            w.HANDLE,
            w.DWORD,
            c.c_void_p,
            w.DWORD,
            c.c_void_p,
            w.DWORD,
            c.POINTER(w.DWORD),
            c.c_void_p,
        ]
        k.DeviceIoControl.restype = w.BOOL
        _API = c, w, k, msvcrt
    return _API


def ioctl(handle, code):
    c, w, k, _ = api()
    out, count = c.create_string_buffer(4096), w.DWORD()
    if not k.DeviceIoControl(
        handle, code, None, 0, out, len(out), c.byref(count), None
    ):
        raise c.WinError(c.get_last_error())
    return out.raw[: count.value]


def file_identity(stream):
    """Read all identity fields through the same deny-write/delete handle."""
    c, _, k, msvcrt = api()
    handle = msvcrt.get_osfhandle(stream.fileno())
    info, fileid = c.create_string_buffer(52), c.create_string_buffer(24)
    if not k.GetFileInformationByHandle(handle, info):
        raise c.WinError(c.get_last_error())
    if not k.GetFileInformationByHandleEx(handle, 18, fileid, 24):
        raise c.WinError(c.get_last_error())
    fields = struct.unpack("<13I", info.raw)
    record = ioctl(handle, 0x900EB)  # FSCTL_READ_FILE_USN_DATA
    if len(record) < 60:
        raise OSError("truncated USN record")
    version = struct.unpack_from("<H", record, 4)[0]
    if version not in (2, 3) or (version == 3 and len(record) < 76):
        raise OSError(f"unsupported USN record version {version}")
    usn = struct.unpack_from("<q", record, 24 if version == 2 else 40)[0]
    # NTFS accepts QUERY_USN_JOURNAL on a file/directory handle. Opening the raw
    # volume with GENERIC_READ unnecessarily requires administrator membership.
    journal = ioctl(handle, 0x900F4)
    if len(journal) < 56 or usn <= 0:
        raise OSError("absent/invalid USN journal identity")
    journal_id, _, next_usn, lowest = struct.unpack_from("<Qqqq", journal)
    if not journal_id or usn >= next_usn or usn < lowest:
        raise OSError("file USN outside valid journal epoch")
    return dict(
        volume_serial=struct.unpack_from("<Q", fileid.raw)[0],
        file_id_128=fileid.raw[8:24].hex(),
        size=(fields[8] << 32) | fields[9],
        last_write_time=(fields[6] << 32) | fields[5],
        usn=usn,
        journal_id=journal_id,
    )


def maybe_identity(stream):
    try:
        return file_identity(stream)
    except OSError:
        return None


def check_content(stream, records):
    """Hash exact export chunks from the retained handle, never a reopened path."""
    size = os.fstat(stream.fileno()).st_size
    total = 0
    for record in records:
        if size != record["nbytes"]:
            raise base.StaleManifest(f"file size changed: {record['path']}")
        stream.seek(0)
        for expected in record["chunks_sha256"]:
            digest = hashlib.sha256()
            remaining = min(record["chunk_bytes"], size - stream.tell())
            while remaining:
                data = stream.read(min(4 * 1024**2, remaining))
                if not data:
                    raise base.StaleManifest(f"short read: {record['path']}")
                digest.update(data)
                remaining -= len(data)
                total += len(data)
            if digest.hexdigest() != expected:
                raise base.StaleManifest(
                    f"content changed: {record['path']}; export required"
                )
    return total


def seal_filesystem(manifest, workers=4):
    """Export-only: bind full content hashes and native identities under locks."""
    records = [
        *manifest["sources"],
        *manifest.get("origins", []),
        *manifest["artifacts"].values(),
    ]
    groups = {}
    for record in records:
        groups.setdefault(record["path"], []).append(record)

    def seal(group):
        with base._open_retained(group[0]["path"]) as stream:
            before = maybe_identity(stream)
            check_content(stream, group)
            after = maybe_identity(stream)
            if before != after:
                raise base.StaleManifest("dependency changed during export attestation")
            for record in group:
                record["ntfs_identity"] = after

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(seal, groups.values()))
    manifest["filesystem_identity_version"] = 1


class IdentityLease:
    """Validate all dependencies and retain every runtime input through use."""

    def __init__(self, path, workers=4, *, event=None, force_audit=False, executable_ready=None):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("identity workers must be between 1 and 16")
        self.path, self.event = Path(path), event
        self._owners, self._failure = {}, None
        self._validated, self._closed = False, False
        self._executables_validated = False
        self._setup_join = None
        self.stats = dict(
            mode="ntfs-identity",
            validated=False,
            identity_matches=0,
            fallback_files=0,
            hashed_bytes=0,
            fallback_paths=[],
            mandatory_write_delete_exclusion=os.name == "nt",
        )
        self._manifest_stream = base._open_retained(self.path)
        try:
            self.manifest = base.load(self.path, event=event)
            self.paths = {
                name: Path(r["path"]) for name, r in self.manifest["artifacts"].items()
            }
            records = [
                *self.manifest["sources"],
                *self.manifest.get("origins", []),
                *self.manifest["artifacts"].values(),
            ]
            groups = {}
            for record in records:
                groups.setdefault(record["path"], []).append(record)
            runtime_paths = {
                r["path"]
                for r in [
                    *self.manifest.get("origins", []),
                    *self.manifest["artifacts"].values(),
                ]
            }
            artifact_paths = {str(p) for p in self.paths.values()}
            started = time.perf_counter()

            def check(item):
                name, group = item
                stream, mapping = base._open_retained(name), None
                try:
                    identity = maybe_identity(stream)
                    match = (
                        not force_audit
                        and self.manifest.get("filesystem_identity_version") == 1
                        and identity is not None
                        and all(r.get("ntfs_identity") == identity for r in group)
                    )
                    hashed = 0 if match else check_content(stream, group)
                    if not match and maybe_identity(stream) != identity:
                        raise base.StaleManifest(
                            f"dependency changed during fallback: {name}"
                        )
                    if name in artifact_paths and group[0]["nbytes"]:
                        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                    owner = (
                        (stream, mapping, identity) if name in runtime_paths else None
                    )
                    if owner is None:
                        stream.close()
                    return name, owner, match, hashed
                except BaseException:
                    if mapping is not None:
                        mapping.close()
                    stream.close()
                    raise

            # A few batches avoid one Future/thread handoff per tiny source file.
            items = list(groups.items())
            if executable_ready is not None:
                executable_paths = {str(self.paths[c["artifact"]]) for c in self.manifest["contexts"].values()}
                for name in sorted(executable_paths):
                    checked_name, owner, match, hashed = check((name, groups[name]))
                    if owner is None:
                        raise base.StaleManifest("executable input must remain pinned")
                    self._owners[checked_name] = owner
                    self.stats["identity_matches"] += int(match)
                    self.stats["fallback_files"] += int(not match)
                    self.stats["hashed_bytes"] += hashed
                    if not match:self.stats["fallback_paths"].append(name)
                self._executables_validated = True
                if event:event("validation_executables_ready", time.perf_counter()-started)
                self._setup_join = executable_ready(self)
                items = [(n, g) for n, g in items if n not in executable_paths]
            batches = [items[i::workers] for i in range(workers)]

            def batch(items):
                result = []
                try:
                    for item in items:
                        result.append(check(item))
                    return result
                except BaseException:
                    for _, owner, _, _ in result:
                        if owner:
                            if owner[1] is not None:
                                owner[1].close()
                            owner[0].close()
                    raise

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(batch, items) for items in batches]
                error = None
                for future in futures:
                    try:
                        for name, owner, match, hashed in future.result():
                            if owner:
                                self._owners[name] = owner
                            self.stats["identity_matches"] += int(match)
                            self.stats["fallback_files"] += int(not match)
                            self.stats["hashed_bytes"] += hashed
                            if not match:
                                self.stats["fallback_paths"].append(name)
                    except BaseException as exc:
                        error = error or exc
                if error:
                    raise error
            self.stats.update(
                unique_files=len(groups),
                retained_files=len(self._owners),
                identity_workers=workers,
            )
            if event:
                event(
                    "validation_ntfs_identity_and_fallback",
                    time.perf_counter() - started,
                )
        except BaseException as exc:
            self._failure = exc
            if self._setup_join is not None:
                try:self._setup_join()
                except BaseException:pass
            self.close()
            raise

    def require_executables_validated(self):
        if self._closed or self._failure or not self._executables_validated:
            raise base.StaleManifest("executable validation has not committed")

    def wait(self):
        if self._failure:
            raise self._failure
        if self._closed:
            raise base.StaleManifest("closed identity lease")
        if self._validated:
            return
        started = time.perf_counter()
        try:
            for name, (stream, _, before) in self._owners.items():
                after = maybe_identity(stream)
                if after != before:
                    raise base.StaleManifest(
                        f"runtime input identity changed during setup: {name}"
                    )
            if not base.plamo_root.config_equal(self.manifest["configuration"], base.configuration()):
                raise base.StaleManifest("runtime configuration changed during setup")
            if base.source_files(self.manifest["source_roots"]) != [
                r["path"] for r in self.manifest["sources"]
            ]:
                raise base.StaleManifest("source inventory changed during setup")
            self._validated = True
            self.stats["validated"] = True
            if self.event:
                self.event("validation_identity_commit", time.perf_counter() - started)
        except BaseException as exc:
            self._failure = exc
            raise

    def require_validated(self):
        if self._closed or self._failure or not self._validated:
            raise base.StaleManifest("identity validation has not committed")

    def artifact(self, name):
        mapping = self._owners[str(self.paths[name])][1]
        return memoryview(mapping) if mapping is not None else memoryview(b"")

    def read_into(self, name, offset, destination):
        """Read directly into a mapped BO, avoiding per-page source-map faults."""
        view = memoryview(destination)
        record = self.manifest["artifacts"][name]
        if self._closed or self._failure:
            raise base.StaleManifest("unavailable input lease")
        if (
            view.readonly
            or not view.c_contiguous
            or view.ndim != 1
            or view.itemsize != 1
            or type(offset) is not int
            or offset < 0
            or offset + view.nbytes > record["nbytes"]
        ):
            raise ValueError("invalid direct file-to-BO slice")
        stream = self._owners[str(self.paths[name])][0]
        stream.seek(offset)
        done = 0
        while done < view.nbytes:
            count = stream.readinto(view[done:])
            if not count:
                raise base.StaleManifest("short direct BO read")
            done += count

    def join_hashes(self):
        if self._failure:
            raise self._failure

    def close(self):
        self._closed = True
        for stream, mapping, _ in self._owners.values():
            if mapping is not None:
                mapping.close()
            stream.close()
        self._owners.clear()
        self._manifest_stream.close()
