"""Bind an exported PLaMo graph using only Python's stdlib and pyxrt.

The caller must validate manifest/config/source/artifact identities before
execution. A retained validation lease permits setup while hashes are pending;
its xclbins must already be validated before ``open``. ``artifact_paths`` is the
validator's resolved map, not paths from an untrusted manifest. No compiler,
IRON, MLIR, tokenizer, tensor framework, native dispatch DLL, or pickle is used.

Instruction catalogs contain exact generated streams for a bounded scalar
domain.  A missing case fails before touching the device; no BD word is guessed.
All BO views are formed directly from their root, avoiding nested XRT sub-BOs.
"""

from __future__ import annotations

import importlib
import mmap
import operator
from pathlib import Path
import threading
import time


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _bytes_view(value):
    result = memoryview(value)
    if not result.c_contiguous:
        raise ValueError("buffer must be contiguous")
    return result.cast("B")


class DirectRuntime:
    """A synchronous runtime over a previously identity-validated manifest.

    Public host views are byte-oriented and retain their root BO.  Writes made
    through ``write``/``zero`` are flushed before the next execute.  Direct
    mutations of ``root_host`` require an explicit ``sync_to`` call.

    ``event(name, seconds)`` receives completed setup/dispatch spans.  It may
    record a wall/process timestamp itself to build a process-start timeline.
    ``concurrent_setup`` is experimental and disabled by default: it overlaps
    hw-context creation with data allocation/copy using the same XRT device.
    An external ``artifact_lease`` owns its retained file handles and mappings:
    close this runtime before closing that lease. ``validation_barrier`` is
    called before each execute and must reject pending or failed validation.
    """

    def __init__(
        self,
        manifest,
        artifact_paths,
        *,
        xrt_module=None,
        event=None,
        artifact_lease=None,
        validation_barrier=None,
        copy_mode="python",
    ):
        if copy_mode not in ("python", "memmove", "readinto"):
            raise ValueError("unsupported BO copy mode")
        if copy_mode == "readinto" and not callable(
            getattr(artifact_lease, "read_into", None)
        ):
            raise ValueError("readinto requires a retained file read_into lease")
        self.copy_mode = copy_mode
        self.manifest = manifest
        self.artifact_paths = {k: Path(v) for k, v in artifact_paths.items()}
        self._artifact_lease = artifact_lease
        if artifact_lease is not None:
            if (
                manifest is not artifact_lease.manifest
                and manifest != artifact_lease.manifest
            ):
                raise ValueError(
                    "runtime manifest differs from retained validation lease"
                )
            if self.artifact_paths != {
                k: Path(v) for k, v in artifact_lease.paths.items()
            }:
                raise ValueError(
                    "runtime artifact paths differ from retained validation lease"
                )
            if validation_barrier is None:
                validation_barrier = artifact_lease.require_validated
        if validation_barrier is not None and not callable(validation_barrier):
            raise TypeError("validation barrier must be callable")
        self.validation_barrier = validation_barrier
        self.xrt = xrt_module
        self.event = event
        self.device = None
        self.contexts = {}
        self._early_context_future = None
        self.roots = {}
        self.root_host = {}
        self.views = {}
        self._view_bos = {}
        self._maps = {}
        self._files = {}
        self._graphs = {}
        self._dirty = set()
        self._lock = threading.RLock()
        self._opened = False
        self._failed = False
        self._completion_uncertain = False
        self._closed = False
        self._active = False
        self.last_elapsed = None
        self._validate_structure()

    def _span(self, name, start):
        elapsed = time.perf_counter() - start
        if self.event is not None:
            self.event(name, elapsed)
        return elapsed

    def _validate_structure(self):
        m = self.manifest
        if m.get("version") != 1:
            raise ValueError("unsupported direct runtime manifest version")
        for field in (
            "artifacts",
            "contexts",
            "roots",
            "views",
            "instructions",
            "graphs",
        ):
            if not isinstance(m.get(field), dict):
                raise ValueError(f"manifest {field} must be an object")
        if set(m["artifacts"]) != set(self.artifact_paths):
            raise ValueError("resolved artifact identity set differs from manifest")
        for name, item in m["artifacts"].items():
            size = _integer(item["nbytes"], f"artifact {name} nbytes")
            if (
                self._artifact_lease is None
                and self.artifact_paths[name].stat().st_size != size
            ):
                raise ValueError(f"artifact {name} size changed after validation")
        for name, item in m["contexts"].items():
            if item["artifact"] not in self.artifact_paths:
                raise ValueError(f"context {name} references unknown artifact")
            if not isinstance(item.get("kernel", "MLIR_AIE"), str):
                raise ValueError(f"context {name} kernel must be a string")
        for name, item in m["roots"].items():
            size = _integer(item["nbytes"], f"root {name} nbytes", minimum=1)
            if item.get("flags", "host_only") not in (
                "host_only",
                "cacheable",
                "normal",
            ):
                raise ValueError(f"root {name} has unsupported BO flags")
            _integer(item.get("group_id", 0), f"root {name} group_id")
            spans = []
            for part in item.get("initial", []):
                if part.get("codec", "raw") != "raw":
                    raise ValueError("BO initial slices must be raw bytes")
                offset = _integer(part.get("offset", 0), "initial destination offset")
                length = self._check_slice(part, initial=True)
                if offset + length > size:
                    raise ValueError(f"root {name} initial data exceeds allocation")
                spans.append((offset, offset + length))
            previous_end = 0
            for start, end in sorted(spans):
                if start < previous_end:
                    raise ValueError(f"root {name} has overlapping initial spans")
                previous_end = end
        for name, item in m["views"].items():
            root = m["roots"].get(item["root"])
            if root is None:
                raise ValueError(f"view {name} references unknown root")
            offset = _integer(item.get("offset", 0), f"view {name} offset")
            size = _integer(item["nbytes"], f"view {name} nbytes", minimum=1)
            if offset + size > root["nbytes"]:
                raise ValueError(f"view {name} exceeds root")
        for name, item in m["instructions"].items():
            params = item.get("params", [])
            if len(set(params)) != len(params) or not all(
                isinstance(p, str) for p in params
            ):
                raise ValueError(f"instruction {name} parameters must be unique names")
            cases = item["cases"]
            if not isinstance(cases, dict) or not cases:
                raise ValueError(f"instruction {name} needs exact cases")
            for key, part in cases.items():
                values = key.split(",") if key else []
                if len(values) != len(params):
                    raise ValueError(f"instruction {name} case {key!r} has wrong arity")
                try:
                    if any(str(int(v)) != v for v in values):
                        raise ValueError()
                except ValueError:
                    raise ValueError(
                        f"instruction {name} case {key!r} is not canonical"
                    ) from None
                self._check_slice(part)
                length = self._instruction_size(part)
                if not length or length % 4:
                    raise ValueError(f"instruction {name} must contain uint32 words")
        for name, item in m["graphs"].items():
            params = item.get("parameters", [])
            if len(set(params)) != len(params) or not all(
                isinstance(p, str) for p in params
            ):
                raise ValueError(f"graph {name} parameters must be unique names")
            for step in item["steps"]:
                if step["context"] not in m["contexts"]:
                    raise ValueError(f"graph {name} references unknown context")
                desc = m["instructions"].get(step["instructions"])
                if desc is None or not set(desc.get("params", [])) <= set(params):
                    raise ValueError(
                        f"graph {name} instruction parameters are not graph parameters"
                    )
                for view in step["args"]:
                    if view not in m["views"]:
                        raise ValueError(f"graph {name} references unknown view {view}")
            for field in ("sync_inputs", "sync_outputs"):
                if any(v not in m["views"] for v in item.get(field, [])):
                    raise ValueError(f"graph {name} {field} references unknown view")

    def _check_slice(self, part, *, initial=False):
        artifact = self.manifest["artifacts"].get(part["artifact"])
        if artifact is None:
            raise ValueError("data slice references unknown artifact")
        offset = _integer(
            part.get("artifact_offset", 0 if initial else part.get("offset", 0)),
            "artifact offset",
        )
        length = _integer(part["nbytes"], "artifact slice nbytes")
        if offset + length > artifact["nbytes"]:
            raise ValueError("artifact slice exceeds file")
        return length

    def _artifact(self, name):
        if self._artifact_lease is not None:
            return self._artifact_lease.artifact(name)
        if name not in self._maps:
            stream = self.artifact_paths[name].open("rb")
            try:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            except BaseException:
                stream.close()
                raise
            self._files[name] = stream
            self._maps[name] = mapping
        return self._maps[name]

    def _slice(self, part, *, initial=False):
        offset = part.get("artifact_offset", 0 if initial else part.get("offset", 0))
        return memoryview(self._artifact(part["artifact"]))[
            offset : offset + part["nbytes"]
        ]

    @staticmethod
    def _instruction_size(part):
        codec = part.get("codec", "raw")
        if codec not in ("raw", "zlib"):
            raise ValueError(f"unsupported instruction codec: {codec}")
        if codec == "zlib":
            if not part["nbytes"]:
                raise ValueError("compressed instruction stream is empty")
            size = _integer(
                part["decoded_nbytes"], "decoded instruction nbytes", minimum=1
            )
        else:
            size = part["nbytes"]
            if part.get("decoded_nbytes", size) != size:
                raise ValueError("raw instruction decoded size differs from slice")
        # An explicit allocation ceiling bounds corrupt-metadata expansion.
        # Exported NPU1 per-operator streams are far below this 64-MiB limit.
        if size > 64 * 1024 * 1024 or size % 4:
            raise ValueError(
                "decoded instruction stream exceeds bounds or uint32 alignment"
            )
        return size

    def _instruction_bytes(self, part):
        size = self._instruction_size(part)
        source = self._slice(part)
        if part.get("codec", "raw") == "raw":
            return source
        import zlib

        try:
            decoder = zlib.decompressobj()
            result = decoder.decompress(source, size + 1)
            if (
                len(result) != size
                or not decoder.eof
                or decoder.unconsumed_tail
                or decoder.unused_data
            ):
                raise ValueError(
                    "compressed instruction stream size/termination differs from manifest"
                )
            return result
        except zlib.error as exc:
            raise ValueError(f"invalid compressed instruction stream: {exc}") from exc

    def begin_contexts(self, pool, *, device_index=0):
        """Setup only, after executable validation; execute still needs full commit."""
        with self._lock:
            if self._opened or self.device is not None or self._closed:
                raise RuntimeError("runtime can only be opened once")
            if self._artifact_lease is None:
                raise RuntimeError("early contexts require a pinned validation lease")
            self._artifact_lease.require_executables_validated()
            self._open_device(device_index)
            self._early_context_future = pool.submit(self._open_contexts)
            return self._early_context_future

    def _open_device(self, device_index):
        start = time.perf_counter()
        if self.xrt is None:
            self.xrt = importlib.import_module("pyxrt")
        self._span("pyxrt_import", start)
        start = time.perf_counter()
        self.device = self.xrt.device(_integer(device_index, "device index"))
        self._span("device_open", start)

    def open(self, *, device_index=0, concurrent_setup=False):
        with self._lock:
            if self._opened or (self.device is not None and self._early_context_future is None) or self._closed:
                raise RuntimeError("runtime can only be opened once")
            if self._early_context_future is None:
                self._open_device(device_index)
            try:
                if self._early_context_future is not None:
                    self._open_roots()
                    self._early_context_future.result()
                elif concurrent_setup:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="xrt-context"
                    ) as pool:
                        future = pool.submit(self._open_contexts)
                        self._open_roots()
                        future.result()
                else:
                    self._open_roots()
                    self._open_contexts()
                self._opened = True
            except BaseException:
                self._failed = True
                raise
            return self

    def _open_contexts(self):
        start = time.perf_counter()
        for name, item in self.manifest["contexts"].items():
            phase = time.perf_counter()
            image = self.xrt.xclbin(str(self.artifact_paths[item["artifact"]]))
            uuid = image.get_uuid()
            self._span(f"xclbin_load:{name}", phase)
            phase = time.perf_counter()
            self.device.register_xclbin(image)
            self._span(f"xclbin_register:{name}", phase)
            phase = time.perf_counter()
            context = self.xrt.hw_context(self.device, uuid)
            self._span(f"hw_context_create:{name}", phase)
            phase = time.perf_counter()
            kernel_name = item.get("kernel", "MLIR_AIE")
            kernel = self.xrt.kernel(context, kernel_name)
            nargs = None
            for candidate in image.get_kernels():
                if candidate.get_name() == kernel_name:
                    nargs = candidate.get_num_args()
                    break
            if nargs is None or nargs < 3:
                raise ValueError(f"context {name} has no supported NPU1 kernel ABI")
            self.contexts[name] = (image, context, kernel, nargs)
            self._span(f"kernel_binding:{name}", phase)
        self._span("xclbin_hw_contexts", start)

    @staticmethod
    def _zero_region(host, start, end):
        # Do not construct a weights-sized temporary allocation.
        zero = bytes(min(65536, end - start))
        while start < end:
            size = min(len(zero), end - start)
            host[start : start + size] = memoryview(zero)[:size]
            start += size

    def _open_roots(self):
        start = time.perf_counter()
        for name, item in self.manifest["roots"].items():
            size = item["nbytes"]
            phase = time.perf_counter()
            bo = self.xrt.bo(
                self.device,
                size,
                getattr(self.xrt.bo, item.get("flags", "host_only")),
                item.get("group_id", 0),
            )
            self._span(f"bo_allocate:{name}", phase)
            phase = time.perf_counter()
            host = _bytes_view(bo.map())
            self._span(f"bo_map:{name}", phase)
            phase = time.perf_counter()
            if len(host) != size:
                raise RuntimeError(
                    f"root {name} mapping length differs from allocation"
                )
            self.roots[name] = bo
            self.root_host[name] = host
            # Only uncovered bytes are zeroed. Weight-covered spans are copied once.
            previous = 0
            for part in sorted(
                item.get("initial", []), key=lambda p: p.get("offset", 0)
            ):
                offset = part.get("offset", 0)
                self._zero_region(host, previous, offset)
                target = host[offset : offset + part["nbytes"]]
                if self.copy_mode == "readinto":
                    self._artifact_lease.read_into(
                        part["artifact"], part.get("artifact_offset", 0), target
                    )
                elif self.copy_mode == "memmove":
                    source = self._slice(part, initial=True)
                    from runtime_copy import copy_into

                    copy_into(target, source)
                else:
                    target[:] = self._slice(part, initial=True)
                previous = offset + part["nbytes"]
            self._zero_region(host, previous, size)
            self._span(f"mmap_initial_copy_zero:{name}", phase)
            phase = time.perf_counter()
            bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._span(f"bo_upload:{name}", phase)
        for name, item in self.manifest["views"].items():
            start_at = item.get("offset", 0)
            self.views[name] = self.root_host[item["root"]][
                start_at : start_at + item["nbytes"]
            ]
        self._span("bo_allocate_map_initialize_sync", start)

    def _ready(self):
        if not self._opened or self._failed:
            raise RuntimeError("runtime is not open and healthy")
        if self._active:
            raise RuntimeError("runtime already has an active submission")

    def _bo(self, name):
        item = self.manifest["views"][name]
        key = (item["root"], item.get("offset", 0), item["nbytes"])
        if key not in self._view_bos:
            root = self.roots[key[0]]
            if key[1] == 0 and key[2] == self.manifest["roots"][key[0]]["nbytes"]:
                self._view_bos[key] = root
            else:
                self._view_bos[key] = self.xrt.bo(root, key[2], key[1])
        return self._view_bos[key]

    def _cases(self, graph_name, values):
        graph = self.manifest["graphs"][graph_name]
        if set(values) != set(graph.get("parameters", [])):
            raise ValueError(
                f"{graph_name} scalar keys differ from exported graph parameters"
            )
        integers = {
            name: _integer(value, name, minimum=-(1 << 63))
            for name, value in values.items()
        }
        for name, bounds in graph.get("parameter_ranges", {}).items():
            if not bounds[0] <= integers[name] <= bounds[1]:
                raise ValueError(
                    f"{graph_name} scalar {name} is outside exported range"
                )
        result = []
        for step in graph["steps"]:
            desc = self.manifest["instructions"][step["instructions"]]
            key = ",".join(str(integers[p]) for p in desc.get("params", []))
            try:
                part = desc["cases"][key]
            except KeyError:
                raise ValueError(
                    f"{graph_name} has no exported instruction case {step['instructions']}({key})"
                ) from None
            result.append(part)
        return result

    def prepare(self, graph_name, values):
        with self._lock:
            self._ready()
            parts = self._cases(graph_name, values)
            start = time.perf_counter()
            graph = self.manifest["graphs"][graph_name]
            state = self._graphs.get(graph_name)
            if state is None:
                state = {"steps": [], "segments": [], "instructions": {}}
                for entry in graph["steps"]:
                    _, context, kernel, nargs = self.contexts[entry["context"]]
                    if len(entry["args"]) > nargs - 3:
                        raise ValueError(
                            f"{graph_name} has too many NPU buffer arguments"
                        )
                    if nargs > 3 and not entry["args"]:
                        raise ValueError(
                            f"{graph_name} has no valid BO for unused pointer slots"
                        )
                    key = (entry["context"], entry["instructions"])
                    storage = state["instructions"].setdefault(
                        key,
                        {
                            "bo": None,
                            "host": None,
                            "part": None,
                            "kernel": kernel,
                            "generation": 0,
                        },
                    )
                    current = {
                        "run": None,
                        "storage": storage,
                        "generation": -1,
                        "context": entry["context"],
                        "kernel": kernel,
                        "args": list(entry["args"]),
                        "nargs": nargs,
                    }
                    state["steps"].append(current)
                    if (
                        not state["segments"]
                        or state["segments"][-1]["context_name"] != entry["context"]
                    ):
                        state["segments"].append(
                            {
                                "context_name": entry["context"],
                                "context": context,
                                "steps": [],
                                "runlist": None,
                            }
                        )
                    state["segments"][-1]["steps"].append(current)
                self._graphs[graph_name] = state
            rebound = False
            updated = 0
            uploaded = 0
            read_bytes = 0
            for current, part in zip(state["steps"], parts):
                storage = current["storage"]
                if storage["part"] != part:
                    payload = self._instruction_bytes(part)
                    size = self._instruction_size(part)
                    if storage["host"] is None or len(storage["host"]) != size:
                        bo = self.xrt.bo(
                            self.device,
                            size,
                            self.xrt.bo.cacheable,
                            storage["kernel"].group_id(1),
                        )
                        storage["bo"] = bo
                        storage["host"] = _bytes_view(bo.map())
                        storage["generation"] += 1
                    storage["host"][:] = payload
                    storage["bo"].sync(
                        self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
                    )
                    storage["part"] = part
                    updated += 1
                    uploaded += size
                    read_bytes += part["nbytes"]
                if current["generation"] != storage["generation"]:
                    rebound = True
            if rebound:
                # A run belongs to exactly one XRT runlist for its lifetime.
                # Rebinding instruction BO sizes must use fresh runs for every
                # member. Keep all old, completed lists/runs alive until every
                # replacement is fully bound; only then swap ownership.
                old_plans = [
                    (
                        segment["runlist"],
                        tuple(current["run"] for current in segment["steps"]),
                    )
                    for segment in state["segments"]
                ]
                replacements = []
                for segment in state["segments"]:
                    runlist = self.xrt.runlist(segment["context"])
                    runs = []
                    for current in segment["steps"]:
                        storage = current["storage"]
                        run = self.xrt.run(current["kernel"])
                        runs.append(run)
                        run.set_arg(0, 3)
                        run.set_arg(1, storage["bo"])
                        run.set_arg(2, len(storage["host"]))
                        buffers = [self._bo(view) for view in current["args"]]
                        # The resident NPU1 image declares five pointer slots.
                        # Unused slots receive a valid existing BO; exact exported
                        # streams never address those padding arguments.
                        if len(buffers) < current["nargs"] - 3:
                            buffers.extend(
                                [buffers[0]] * (current["nargs"] - 3 - len(buffers))
                            )
                        for index, bo in enumerate(buffers, start=3):
                            run.set_arg(index, bo)
                        runlist.add(run)
                    replacements.append((segment, runlist, runs))
                for segment, runlist, runs in replacements:
                    segment["runlist"] = runlist
                    for current, run in zip(segment["steps"], runs):
                        current["run"] = run
                        current["generation"] = current["storage"]["generation"]
                old_plans.clear()
            state["last_prepare"] = {
                "prepare_s": self._span(f"prepare:{graph_name}", start),
                "instruction_updates": updated,
                "instruction_upload_bytes": uploaded,
                "instruction_read_bytes": read_bytes,
                "instruction_bos": len(state["instructions"]),
                "runlist_rebuilt": rebound,
            }
            return state

    def execute(self, graph_name, values):
        with self._lock:
            self._ready()
            if self.validation_barrier is not None:
                self.validation_barrier()
            start = time.perf_counter()
            state = self.prepare(graph_name, values)
            graph = self.manifest["graphs"][graph_name]
            sync_start = time.perf_counter()
            for view in set(graph.get("sync_inputs", [])) | self._dirty:
                self._bo(view).sync(
                    self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
                )
            self._dirty.clear()
            sync_input_s = self._span(f"sync_inputs:{graph_name}", sync_start)
            self._active = True
            try:
                submit_start = time.perf_counter()
                for segment in state["segments"]:
                    segment["runlist"].execute()
                    # Only the runlist owns submission/completion. Never wait
                    # or poll individual runs after adding them to a runlist.
                    segment["runlist"].wait()
                submit_wait_s = self._span(f"submit_wait:{graph_name}", submit_start)
                sync_start = time.perf_counter()
                for view in graph.get("sync_outputs", []):
                    self._bo(view).sync(
                        self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
                    )
                sync_output_s = self._span(f"sync_outputs:{graph_name}", sync_start)
            except BaseException:
                # Completion may be uncertain. Never reuse buffers/runlists.
                self._failed = True
                self._completion_uncertain = True
                raise
            finally:
                self._active = False
            self.last_elapsed = time.perf_counter() - start
            self._span(f"execute:{graph_name}", start)
            return {
                **state["last_prepare"],
                "elapsed_s": self.last_elapsed,
                "sync_input_s": sync_input_s,
                "submit_wait_s": submit_wait_s,
                "sync_output_s": sync_output_s,
                "segments": len(state["segments"]),
                "runs": len(state["steps"]),
            }

    def write(self, view, data, *, offset=0):
        with self._lock:
            self._ready()
            source = _bytes_view(data)
            offset = _integer(offset, "write offset")
            target = self.views[view]
            if offset + len(source) > len(target):
                raise ValueError("write exceeds view")
            target[offset : offset + len(source)] = source
            self._dirty.add(view)

    def zero(self, view):
        with self._lock:
            self._ready()
            self._zero_region(self.views[view], 0, len(self.views[view]))
            self._dirty.add(view)

    def sync_to(self, view):
        with self._lock:
            self._ready()
            self._bo(view).sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._dirty.discard(view)

    def sync_from(self, view):
        with self._lock:
            self._ready()
            item = self.manifest["views"][view]
            start, end = item.get("offset", 0), item.get("offset", 0) + item["nbytes"]
            overlap = any(
                dirty["root"] == item["root"]
                and dirty.get("offset", 0) < end
                and start < dirty.get("offset", 0) + dirty["nbytes"]
                for dirty in (self.manifest["views"][name] for name in self._dirty)
            )
            if overlap:
                raise RuntimeError("cannot overwrite unsynchronized host changes")
            self._bo(view).sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def read(self, view, *, offset=0, nbytes=None, sync=True):
        with self._lock:
            self._ready()
            offset = _integer(offset, "read offset")
            nbytes = (
                len(self.views[view]) - offset
                if nbytes is None
                else _integer(nbytes, "read nbytes")
            )
            if offset > len(self.views[view]) or offset + nbytes > len(
                self.views[view]
            ):
                raise ValueError("read exceeds view")
            if sync:
                self.sync_from(view)
            return bytes(self.views[view][offset : offset + nbytes])

    def close(self):
        """Release a quiescent runtime in dependency order.

        After uncertain device completion, releasing BOs could free in-flight
        DMA targets. Such a runtime is deliberately retained until process
        termination; callers must not catch that error and resume execution.
        """
        with self._lock:
            if self._closed:
                return
            if self._active or self._completion_uncertain:
                raise RuntimeError(
                    "cannot release runtime with uncertain device completion; terminate the worker"
                )
            if self._early_context_future is not None:
                try:
                    self._early_context_future.result()
                except BaseException:
                    self._failed = True
                self._early_context_future = None
            for state in self._graphs.values():
                for segment in state["segments"]:
                    segment["runlist"] = None
                    segment["steps"].clear()
                    segment.clear()
                for step in state["steps"]:
                    step["run"] = None
                    step["storage"] = None
                    step["kernel"] = None
                for storage in state["instructions"].values():
                    if storage["host"] is not None:
                        storage["host"].release()
                    storage.clear()
                state["instructions"].clear()
                state["steps"].clear()
                state["segments"].clear()
            self._graphs.clear()
            self._view_bos.clear()
            for view in self.views.values():
                view.release()
            self.views.clear()
            for host in self.root_host.values():
                host.release()
            self.root_host.clear()
            self.roots.clear()
            self.contexts.clear()
            self.device = None
            for mapping in self._maps.values():
                mapping.close()
            self._maps.clear()
            for stream in self._files.values():
                stream.close()
            self._files.clear()
            self._dirty.clear()
            self._opened = False
            self._closed = True
