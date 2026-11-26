# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline progress and Windows stack/termination diagnostics, without uvx."""

import contextlib
import ctypes
import faulthandler
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def progress(label, interval=20, tree_limit=None):
    """Keep cold compilation visible; dump stacks periodically without signals."""
    stopped = threading.Event()
    start = time.monotonic()

    def heartbeat():
        while not stopped.wait(interval):
            print(
                json.dumps(
                    {
                        "progress": label,
                        "elapsed_s": round(time.monotonic() - start),
                        "pid": os.getpid(),
                    }
                ),
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    thread.start()
    from iron.applications.plamo3.resources import MemoryWatch

    try:
        with MemoryWatch(label, tree_limit=tree_limit) as memory:
            yield memory
    finally:
        stopped.set()
        thread.join(timeout=1)
        faulthandler.cancel_dump_traceback_later()


@contextlib.contextmanager
def compiler_jobs():
    """Limit existing parallel compilers to fit available RAM, without rekeying.

    Only scheduling is changed. aiecc already parallelizes cores, and Peano
    external kernels already use AIE_KERNEL_COMPILE_JOBS. Compiling complete
    chained xclbins concurrently would violate their input-image dependency.
    """
    from aie.utils.compile import utils

    jobs = max(1, min(2, int(os.environ.get("IRON_COMPILE_JOBS", "1"))))
    os.environ["AIE_KERNEL_COMPILE_JOBS"] = str(
        min(jobs, max(1, int(os.environ.get("AIE_KERNEL_COMPILE_JOBS", str(jobs)))))
    )
    original = utils._run_aiecc
    original_peano = utils.compile_external_kernels
    timing = {"aiecc_s": 0.0, "aiecc_calls": 0, "peano_s": 0.0}

    def peano(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_peano(*args, **kwargs)
        finally:
            timing["peano_s"] += time.perf_counter() - start

    def run(mlir_file, args, **kwargs):
        options = []
        tokens = iter(args)
        for option in tokens:
            if option in ("-j", "--nthreads"):
                next(tokens, None)
            elif not option.startswith(("-j", "--nthreads=")):
                options.append(option)
        options.append(f"--nthreads={jobs}")
        start = time.monotonic()
        print(
            json.dumps(
                {
                    "compile": "begin",
                    "directory": str(kwargs.get("cwd", mlir_file)),
                    "jobs": jobs,
                }
            ),
            flush=True,
        )
        try:
            return original(mlir_file, options, **kwargs)
        finally:
            timing["aiecc_s"] += time.monotonic() - start
            timing["aiecc_calls"] += 1
            print(
                json.dumps(
                    {"compile": "end", "seconds": round(time.monotonic() - start, 2)}
                ),
                flush=True,
            )

    with (
        patch.object(utils, "_run_aiecc", run),
        patch.object(utils, "compile_external_kernels", peano),
    ):
        yield timing


def stop_worker(proc, directory, label):
    """Stop the Windows process tree; stacks already arrive in the worker log."""
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "plamo_process_tree", root / "pyobjtools/verify.py"
    )
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    tree = watchdog.process_tree(proc.pid)
    (directory / f"{label}.stop.json").write_text(
        json.dumps(
            {
                "processes": tree,
                "stacks": "periodic faulthandler dumps in worker log; no network or py-spy required",
            }
        ),
        encoding="utf-8",
    )
    terminate_pid(proc.pid)
    proc.wait(timeout=30)


def terminate_pid(pid):
    """Terminate only a supervised worker and its descendants, without exes."""
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "plamo_tree_stop", root / "pyobjtools/verify.py"
    )
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    tree = watchdog.process_tree(pid)
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    for pid, _ in reversed(tree):
        handle = kernel.OpenProcess(1, False, pid)
        if handle:
            try:
                kernel.TerminateProcess(handle, 1)
            finally:
                kernel.CloseHandle(handle)


def memory_from_log(path):
    """Read the final peak-RSS record without depending on worker success."""
    result = None
    if not Path(path).exists():
        return result
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith('{"step_memory"') or line.startswith('{"memory_abort"'):
            try:
                result = json.loads(line)
            except ValueError:
                pass
    return result


def external_compile_gate(
    directory, universal=False, resident_decode=False, model_options=None
):
    """Fail before allocating inputs if the selected designs cannot compile."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "gate.log"
    env = os.environ.copy()
    audit = directory / "subprocess.jsonl"
    env["PYOBJTOOLS_SUBPROCESS_LOG"] = str(audit)
    command = [
        sys.executable,
        "-u",
        "-m",
        "iron.applications.plamo3.compile_tools",
        "--output-dir",
        str(directory),
    ]
    deadline = 480 if resident_decode else 180 if universal else 60
    command += ["--deadline", str(deadline)]
    if universal:
        command.append("--universal")
    if resident_decode:
        command.append("--resident-decode")
        for name, value in (model_options or {}).items():
            if value is not None:
                command += ["--" + name.replace("_", "-"), str(value)]
    with log.open("wb") as out:
        proc = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        try:
            proc.wait(timeout=deadline + 5)
        except subprocess.TimeoutExpired:
            stop_worker(proc, directory, "compile-gate")
            raise RuntimeError(
                f"compile-only gate exceeded its deadline; see {log}"
            ) from None
    blocked = (
        any(
            json.loads(line).get("blocked")
            for line in audit.read_text(encoding="utf-8").splitlines()
        )
        if audit.exists()
        else False
    )
    path = directory / "summary.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if proc.returncode or blocked or result.get("result") != "PASS":
        raise RuntimeError(f"compile-only gate failed; see {log}")
    return result
