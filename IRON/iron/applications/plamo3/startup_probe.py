# SPDX-License-Identifier: Apache-2.0
"""Measure a fresh W8 process with cached builds; never compile a missing artifact.

Run this file directly for the supervisor to avoid importing IRON in the harness.
The child includes Python/sitecustomize, all application imports and first-use work.
"""

import time

ENTRY_NS = time.perf_counter_ns()

import argparse
import ctypes
import functools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def process_origin():
    """Anchor monotonic samples to actual Windows process creation, not main()."""
    if os.name != "nt":
        return int(os.environ.get("PLAMO_STARTUP_LAUNCH_NS", ENTRY_NS)), "parent launch"
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(
        kernel.GetCurrentProcess(),
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(system),
        ctypes.byref(user),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    unix_ns = ((created.dwHighDateTime << 32) | created.dwLowDateTime) * 100
    unix_ns -= 11644473600000000000
    before = time.perf_counter_ns()
    now = time.time_ns()
    after = time.perf_counter_ns()
    return (before + after) // 2 - (now - unix_ns), "Windows GetProcessTimes creation"


def steady_state(samples, window=5, tail=8, tolerance=0.10):
    """Retrospective earliest stable window, with subsequent window medians stable."""
    values = [r["wall_ms"] for r in samples]
    baseline = statistics.median(values[-tail:])
    lo, hi = baseline * (1 - tolerance), baseline * (1 + tolerance)
    start = None
    for i in range(len(values) - window + 1):
        if all(lo <= v <= hi for v in values[i : i + window]) and all(
            lo <= statistics.median(values[j : j + window]) <= hi
            for j in range(i, len(values) - window + 1)
        ):
            start = i
            break
    return dict(
        baseline_ms=baseline,
        tail_samples=min(tail, len(samples)),
        window=window,
        tolerance=tolerance,
        criterion="five consecutive tokens within 10% of final-eight median; all later five-token medians also within 10%",
        first_token=None if start is None else start + 1,
        reached_ms=None if start is None else samples[start]["end_ms"],
        confirmed_ms=None if start is None else samples[start + window - 1]["end_ms"],
        floor_ms=18.75,
        gap_ms=baseline - 18.75,
        floor_ratio=baseline / 18.75,
        effective_weight_GB_s=0.750059520 / (baseline / 1000),
    )


class Timeline:
    def __init__(self, origin):
        self.origin = origin
        self.events = []
        self.spans = []

    def now(self):
        return (time.perf_counter_ns() - self.origin) / 1e6

    def mark(self, name):
        event = dict(phase=name, end_ms=self.now())
        previous = self.events[-1]["end_ms"] if self.events else 0
        event.update(
            duration_ms=event["end_ms"] - previous, within_1s=event["end_ms"] <= 1000
        )
        self.events.append(event)
        print(json.dumps({"startup_phase": event}), flush=True)
        return event

    def wrap(self, label, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            start = self.now()
            try:
                return function(*args, **kwargs)
            finally:
                end = self.now()
                self.spans.append(
                    dict(
                        kind=label, start_ms=start, end_ms=end, duration_ms=end - start
                    )
                )

        return timed


def read_only_readobj(event, values):
    """Allow only the configured reader and its exact object-inspection arguments."""
    configured = os.environ.get("AIE_TOOL_readobj")
    if not configured or event not in ("subprocess.Popen", "_winapi.CreateProcess"):
        return None
    executable, command = values[:2]
    if isinstance(command, str):
        if os.name != "nt":
            return None
        from ctypes import wintypes

        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.CommandLineToArgvW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_int),
        ]
        shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        count = ctypes.c_int()
        parsed = shell.CommandLineToArgvW(command, ctypes.byref(count))
        if not parsed:
            return None
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [wintypes.HLOCAL]
        kernel.LocalFree.restype = wintypes.HLOCAL
        try:
            argv = [parsed[i] for i in range(count.value)]
        finally:
            kernel.LocalFree(ctypes.cast(parsed, wintypes.HLOCAL))
    elif isinstance(command, (list, tuple)):
        argv = [os.fspath(a) for a in command]
    else:
        return None
    normalized = lambda p: os.path.normcase(os.path.abspath(os.fspath(p)))
    if not argv or normalized(argv[0]) != normalized(configured):
        return None
    if executable and normalized(executable) != normalized(configured):
        return None
    options = [
        "--elf-output-style=JSON",
        "--sections",
        "--symbols",
        "--relocations",
        "--stack-sizes",
    ]
    if len(argv) != 7 or argv[1:-1] != options:
        return None
    obj = Path(argv[-1])
    if obj.suffix.lower() != ".o" or not obj.is_file():
        return None
    return argv


def worker(args):
    origin, clock = process_origin()
    timeline = Timeline(origin)
    timeline.events.append(
        dict(
            phase="python_entry",
            end_ms=(ENTRY_NS - origin) / 1e6,
            duration_ms=(ENTRY_NS - origin) / 1e6,
            within_1s=(ENTRY_NS - origin) <= 1e9,
        )
    )
    blocked = []
    allowed_readonly = []
    global _failure_state
    _failure_state = dict(
        clock=clock,
        phases=timeline.events,
        load_spans=timeline.spans,
        blocked_attempts=blocked,
        allowed_readonly_subprocess_events=allowed_readonly,
    )
    from contextvars import ContextVar

    native_launch = ContextVar("startup_checked_native_launch", default=None)

    def refuse(event, values):
        row = dict(event=event, command=repr(values[:2]), at_ms=timeline.now())
        blocked.append(row)
        raise RuntimeError("startup is cache-only: launch refused: " + repr(row))

    if os.name == "nt":
        import _winapi

        original_create_process = _winapi.CreateProcess

        def checked_create_process(*values, **kwargs):
            # This Python build's native audit event has emitted a corrupt
            # command-line field. Check the real API arguments before entering
            # it, and authorize only this call's matching native audit event.
            allowed = read_only_readobj("_winapi.CreateProcess", values)
            if allowed is None:
                refuse("_winapi.CreateProcess.arguments", values)
            token = native_launch.set(allowed)
            try:
                return original_create_process(*values, **kwargs)
            finally:
                native_launch.reset(token)

        _winapi.CreateProcess = checked_create_process

    def audit(event, values):
        if event in (
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.fork",
            "os.forkpty",
            "_winapi.CreateProcess",
        ):
            allowed = (
                native_launch.get()
                if event == "_winapi.CreateProcess"
                else read_only_readobj(event, values)
            )
            if allowed is not None:
                allowed_readonly.append(
                    dict(event=event, command=allowed, at_ms=timeline.now())
                )
                return
            refuse(event, values)

    # Cached frame validation legitimately invokes the configured llvm-readobj.
    # Keep that read-only work in the measured startup; reject every other
    # process, tool probe, compiler and dispatch-bridge build.
    sys.addaudithook(audit)
    import numpy as np
    import aie.utils as au
    from aie.iron.device import from_name
    from aie.utils.compile import utils as compile_utils
    from aie.utils.compile import readobj
    from aie.utils.hostruntime.xrtruntime import hostruntime
    from aie.utils.hostruntime.xrtruntime import device as xrt_device
    from aie.utils.hostruntime.xrtruntime.tensor import XrtTransport
    import iron.common.graph.compiled as compiled
    from iron.applications.plamo3.model import Plamo3, REPOSITORY
    from iron.lm.checkpoint import Checkpoint
    from iron.applications.plamo3.w8_cache import PackedCheckpoint
    from iron.applications.plamo3.tokenizer import Plamo3Tokenizer
    from iron.applications.plamo3.validation import sequences
    from iron.applications.plamo3.worker_state import (
        execution_identity,
        assert_graph_choice,
    )
    from iron.applications.plamo3.probe_runtime import NativeOwners
    from iron.applications.plamo3.benchmark import recorded_golden, accuracy_passes
    from iron.lm.generation import divergence, kl_stats
    from unittest.mock import patch
    from multiprocessing.process import BaseProcess

    def deny_native_build(label, original):
        # The JIT follows each module's global functions by __module__.
        # Preserve identity so instrumentation cannot rekey valid artifacts.
        @functools.wraps(original)
        def refused(*call_args, **call_kwargs):
            row = dict(
                event="native_compile",
                function=label,
                args=repr(call_args[:1])[:512],
                source=str(call_kwargs.get("source_path", "")),
                cwd=str(call_kwargs.get("cwd", "")),
                at_ms=timeline.now(),
            )
            blocked.append(row)
            raise RuntimeError(
                "startup is cache-only: native compile refused: " + repr(row)
            )

        return refused

    @functools.wraps(BaseProcess.start)
    def no_process(*_args, **_kwargs):
        blocked.append(dict(event="multiprocessing.Process.start"))
        raise RuntimeError("startup is cache-only: multiprocessing launch refused")

    # Timers wrap real calls and preserve their values/ordering. They do not
    # replace the loader or pre-open an NPU context. Native constructors are
    # wrapped only during loading and restored before inference.
    guards = [
        patch.object(
            compile_utils,
            "_run_aiecc",
            deny_native_build(
                "aie.utils.compile.utils._run_aiecc", compile_utils._run_aiecc
            ),
        ),
        patch.object(
            compile_utils,
            "compile_cxx_core_function",
            deny_native_build(
                "aie.utils.compile.utils.compile_cxx_core_function",
                compile_utils.compile_cxx_core_function,
            ),
        ),
        patch.object(BaseProcess, "start", no_process),
    ]
    load_hooks = [
        patch.object(
            Checkpoint,
            "__init__",
            timeline.wrap("checkpoint_file_mapping", Checkpoint.__init__),
        ),
        patch.object(
            PackedCheckpoint,
            "__init__",
            timeline.wrap("w8_cache_metadata", PackedCheckpoint.__init__),
        ),
        patch.object(
            PackedCheckpoint,
            "packed",
            timeline.wrap("w8_tensor_mapping", PackedCheckpoint.packed),
        ),
        patch.object(
            Plamo3Tokenizer,
            "__init__",
            timeline.wrap("tokenizer_construction", Plamo3Tokenizer.__init__),
        ),
        patch.object(
            readobj, "linked", timeline.wrap("cached_object_readobj", readobj.linked)
        ),
        patch.object(
            compiled.CompiledGraph,
            "upload",
            timeline.wrap("weight_upload", compiled.CompiledGraph.upload),
        ),
        patch.object(compiled, "_store", timeline.wrap("weight_copy", compiled._store)),
        patch.object(
            XrtTransport,
            "__init__",
            timeline.wrap("BO_allocate_map", XrtTransport.__init__),
        ),
        patch.object(
            xrt_device,
            "_open_device",
            timeline.wrap("device_open", xrt_device._open_device),
        ),
        patch.object(
            hostruntime.pyxrt,
            "xclbin",
            timeline.wrap("xclbin_parse", hostruntime.pyxrt.xclbin),
        ),
        patch.object(
            hostruntime.pyxrt,
            "hw_context",
            timeline.wrap("hw_context_create", hostruntime.pyxrt.hw_context),
        ),
        patch.object(
            hostruntime.CachedXRTRuntime,
            "load",
            timeline.wrap("runtime_load", hostruntime.CachedXRTRuntime.load),
        ),
    ]
    for p in [*guards, *load_hooks]:
        p.start()
    au.set_current_device(from_name("npu1", n_cols=4))
    timeline.mark("imports_and_probe_setup")
    model = Plamo3.from_checkpoint(REPOSITORY, max_seq_len=2048)
    timeline.mark("checkpoint_mapping_model_tokenizer")
    model.load_versions(release=model.checkpoint.release, full_elf=False)
    NativeOwners().keep((model, model.decode_step.callable, au.DefaultNPURuntime))
    timeline.mark("cached_graph_runtime_buffers_weights")
    # Contexts and instruction BOs are lazy: keep hooks through the first
    # prefill and decode, then remove them before steady-state sampling.
    identity = execution_identity(model)
    assert_graph_choice(identity)
    expected = dict(
        w8="1",
        w8_layout="row64",
        w8_math="row",
        w8_l2="4",
        w8_row64_fast="1",
        alias_kv="1",
        resident_prefill="1",
        prefill_dma_pack="1",
        prefill_service_batch="1",
    )
    if any(identity.get(k) != v for k, v in expected.items()):
        raise RuntimeError("startup requires the R17 w8-all configuration")
    if not model.runlist_status()["active"]:
        raise RuntimeError("startup requires the fast serial runlist")
    history = list(sequences(model)[args.prompt])
    prompt_tokens = list(history)
    model.forget(0)
    timeline.mark("prompt_ready")
    began = timeline.now()
    logits = model.logits(history)
    token = int(logits.argmax())
    prefill = dict(wall_ms=timeline.now() - began, token=token)
    timeline.mark("first_prefill_token")
    observed = [(list(history), np.asarray(logits).copy())]
    samples = []
    for i in range(args.tokens):
        history.append(token)
        began = timeline.now()
        logits = model.logits(history)
        token = int(logits.argmax())
        end = timeline.now()
        samples.append(
            dict(
                token=i + 1,
                position=len(history) - 1,
                wall_ms=end - began,
                end_ms=end,
                token_id=token,
                runlist=dict(model.decode_step.callable.last_runlist),
            )
        )
        observed.append((list(history), np.asarray(logits).copy()))
        if i == 0:
            timeline.mark("first_decode_token")
            for p in reversed(load_hooks):
                p.stop()
    timeline.mark("decode_sampling_complete")
    steady = steady_state(samples)
    # Accuracy runs only after timed startup and sampling, from existing recorded
    # HF rows. It cannot load a second model. Decode sampling continues past EOS
    # solely to characterize the device; live rows keep the established W8 gate.
    golden = recorded_golden(model)
    rows = []
    for tokens, actual in observed:
        if 16 in tokens[len(prompt_tokens) :]:
            break
        kl, match = divergence(golden.logits(tokens), actual)
        rows.append(dict(kl=kl, top1_match=bool(match), past_eos=False))
    stats = {
        k: float(v)
        for k, v in kl_stats([(r["kl"], r["top1_match"]) for r in rows]).items()
    }
    passed = accuracy_passes(rows, stats, w8=True) and not blocked
    grouped = {}
    for span in timeline.spans:
        group = grouped.setdefault(
            span["kind"],
            dict(
                calls=0,
                total_ms=0,
                first_start_ms=span["start_ms"],
                last_end_ms=span["end_ms"],
            ),
        )
        group["calls"] += 1
        group["total_ms"] += span["duration_ms"]
        group["last_end_ms"] = span["end_ms"]
    result = dict(
        result="PASS" if passed else "FAIL",
        clock=clock,
        config="R17 w8-all",
        process_origin_ns=origin,
        parent_launch_to_process_creation_ms=(
            origin - int(os.environ["PLAMO_STARTUP_LAUNCH_NS"])
        )
        / 1e6,
        phases=timeline.events,
        nested_load_timings=grouped,
        load_spans=timeline.spans,
        timing_note="phase durations partition process startup; nested load timers overlap and must not be added to phases or each other; shell/venv setup and supervisor excluded; first-use load hooks remain through first decode for lazy context creation, then are removed",
        cache_only=True,
        blocked_attempts=blocked,
        allowed_readonly_subprocess_events=allowed_readonly,
        native_compiles=0,
        prompt=args.prompt,
        prompt_tokens=len(prompt_tokens),
        prefill=prefill,
        decode_samples=samples,
        steady_state=steady,
        startup_within_1s=steady["reached_ms"] is not None
        and steady["reached_ms"] <= 1000,
        first_token_within_1s=next(
            r["end_ms"] for r in timeline.events if r["phase"] == "first_prefill_token"
        )
        <= 1000,
        accuracy=dict(
            result="PASS" if passed else "FAIL",
            rows=rows,
            KL=stats,
            note="live rows against recorded HF; after EOS sampling is performance-only",
        ),
        execution=identity,
        weight_upload=model.upload_report,
    )
    save(args.directory / "startup.json", result)
    save(args.directory / "worker-result.json", result)
    print(
        json.dumps(
            dict(
                result=result["result"],
                artifact=str(args.directory / "startup.json"),
                steady_state=steady,
            )
        ),
        flush=True,
    )
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--prompt", type=int, choices=range(4), default=0)
    parser.add_argument("--tokens", type=int, choices=range(16, 25), default=24)
    args = parser.parse_args()
    if args.directory is None:
        from datetime import datetime

        args.directory = (
            ROOT / "plamo3-startup" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            code = worker(args)
        except Exception as exc:
            failure = dict(
                globals().get("_failure_state", {}),
                result="FAIL",
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            save(args.directory / "failure.json", failure)
            save(args.directory / "worker-result.json", failure)
            traceback.print_exc()
            print(
                json.dumps(
                    dict(result="FAIL", artifact=str(args.directory / "failure.json"))
                ),
                flush=True,
            )
            code = 1
        # Preserve the existing Windows XRT lifetime policy on success and error.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--worker",
        "--directory",
        str(args.directory),
        "--prompt",
        str(args.prompt),
        "--tokens",
        str(args.tokens),
    ]
    log = args.directory / "startup.log"
    env = dict(os.environ)
    with log.open("wb") as stream:
        launch = time.perf_counter_ns()
        env["PLAMO_STARTUP_LAUNCH_NS"] = str(launch)
        proc = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        try:
            code = proc.wait(timeout=540)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = 1
    artifact = args.directory / "startup.json"
    result = dict(
        result="PASS" if code == 0 and artifact.exists() else "FAIL",
        directory=str(args.directory),
        artifact=str(artifact),
        log=str(log),
        exit_code=code,
        seconds=(time.perf_counter_ns() - launch) / 1e9,
        command=command,
    )
    save(args.directory / "summary.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
