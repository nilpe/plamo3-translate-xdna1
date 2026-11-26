# SPDX-License-Identifier: Apache-2.0
"""Fresh-process cached direct runtime timing, plus an explicitly resident service."""

import time

ENTRY_NS = time.perf_counter_ns()

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

PROMPTS = (
    ("今日はいい天気ですね。", "ja", "en"),
    ("駅までの道を教えてください。", "ja", "en"),
    ("Thank you for your help.", "en", "ja"),
    ("The meeting starts at nine tomorrow morning.", "en", "ja"),
)
_DLL_DIRECTORIES = []
_FAILURE = {}
_RUNTIME_OWNERS = []


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def xrt_dir():
    """Directory of our pyxrt build: <install root>\\xrt."""
    import plamo_root

    return plamo_root.ROOT / "xrt"


def system_root():
    return Path(os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or "C:/Windows")


def bootstrap():
    """Explicit -I -S paths; do not execute sitecustomize or import IRON."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    script = str(Path(__file__).resolve().parent)
    if script not in sys.path:
        sys.path.insert(0, script)
    if os.name == "nt":
        # Preserve the venv path even when its executable is a symlink.
        venv = Path(sys.executable).parent.parent
        packages = str(venv / "Lib" / "site-packages")
        if packages not in sys.path:
            sys.path.append(packages)
        # H1: pyxrt.pyd is ours and lives at <install root>\\xrt\\python. XRT_ROOT / XILINX_XRT from the user's environment are NOT consulted.
        # XILINX_XRT is REMOVED from this process (never set): XRT itself resolves xrt_core.dll from it when it is set (measured: setting it to a directory
        # without xrt_core.dll gives "No such library ...\\xrt_core.dll"); unset, XRT finds the NPU driver's copy, as in every certified run.
        # xrt_coreutil.dll is the NPU driver's (System32); it is never shipped and never searched for in user XRT directories.
        xrt = xrt_dir()
        for name in ("XILINX_XRT", "XRT_ROOT"):
            other = os.environ.get(name)
            if other:
                from plamo_errors import note

                note(f"PLaMo: note: {name}={other} is ignored; the runtime uses its own pyxrt in {xrt}.", key="xrt-env-" + name)
        os.environ.pop("XILINX_XRT", None)
        sys.path.append(str(xrt / "python"))
        system = system_root() / "System32" / "AMD"
        # Deterministic DLL search: our xrt directory first; the driver's AMD directory only when it does not carry an xrt_coreutil.dll of its own.
        directories = [xrt]
        if system.is_dir() and not (system / "xrt_coreutil.dll").exists():
            directories.append(system)
        for directory in directories:
            if directory.is_dir():
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))


def timing_helpers(path):
    spec = importlib.util.spec_from_file_location("_r20_timing_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_guards(timeline):
    """No worker process launch or heavyweight model/compiler imports."""
    blocked = []
    imports = []
    forbidden = (
        "iron",
        "aie",
        "mlir",
        "mlir_aie",
        "transformers",
        "torch",
        "numpy",
        "numba",
        "multiprocessing",
        "benchmark",
        "aiecc",
    )
    launches = {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "_winapi.CreateProcess",
    }

    def refuse(event, description):
        row = {"event": event, "description": str(description), "at_ms": timeline.now()}
        blocked.append(row)
        raise RuntimeError("direct cached runtime refused: " + repr(row))

    def forbidden_name(name):
        return any(
            name == prefix or name.startswith(prefix + ".") for prefix in forbidden
        )

    def audit(event, values):
        if event in launches:
            # Windows' native audit command field has been unreliable on this
            # Python build. Do not inspect it; direct CreateProcess is blocked too.
            refuse(event, "worker process creation is prohibited")
        if event == "import" and values:
            name = values[0]
            imports.append({"module": name, "at_ms": timeline.now()})
            if forbidden_name(name):
                refuse("import", name)

    class ImportGate:
        def find_spec(self, fullname, path=None, target=None):
            if forbidden_name(fullname):
                refuse("import.find_spec", fullname)
            return None

    for name in tuple(sys.modules):
        if forbidden_name(name):
            refuse("preexisting import", name)
    sys.meta_path.insert(0, ImportGate())
    sys.addaudithook(audit)
    if os.name == "nt":
        import _winapi

        def create_process(*args, **kwargs):
            refuse(
                "_winapi.CreateProcess.arguments",
                "worker process creation is prohibited",
            )

        _winapi.CreateProcess = create_process
    return blocked, imports, forbidden


def _resolve(base, value):
    result = Path(value)
    return result if result.is_absolute() else Path(base).parent / result


def initialize(args):
    helper = timing_helpers(args.timing_helper)
    origin, clock = helper.process_origin()
    timeline = helper.Timeline(origin)
    timeline.events.append(
        {
            "phase": "python_entry",
            "end_ms": (ENTRY_NS - origin) / 1e6,
            "duration_ms": (ENTRY_NS - origin) / 1e6,
            "within_1s": ENTRY_NS - origin <= 1e9,
        }
    )
    blocked, imports, forbidden = install_guards(timeline)
    _FAILURE.update(
        clock=clock,
        phases=timeline.events,
        load_spans=timeline.spans,
        blocked_attempts=blocked,
        imports=imports,
    )
    bootstrap()
    from runtime_manifest import ValidationLease

    if getattr(args, "validation", "content") == "identity":
        from runtime_identity import IdentityLease as ValidationLease
    from direct_runtime import DirectRuntime
    from runtime_session import RuntimeSession
    from fast_tokenizer import FastPlamo3Tokenizer

    timeline.mark("python_stdlib_runtime_imports")

    def event(name, seconds):
        end = timeline.now()
        timeline.spans.append(
            {
                "kind": name,
                "start_ms": end - seconds * 1000,
                "end_ms": end,
                "duration_ms": seconds * 1000,
            }
        )

    owners = {"lease": None, "runtime": None}
    _RUNTIME_OWNERS.append(owners)
    early_pool = None
    # R34 starts only pinned executable artifacts. Full validation remains the
    # mandatory execution barrier; a failed remaining dependency joins setup.
    early = args.overlap and getattr(args, "validation", "content") == "identity"
    def executable_ready(lease):
        owners["lease"] = lease
        rt = DirectRuntime(lease.manifest, lease.paths, event=event,
            artifact_lease=lease, validation_barrier=lease.require_validated,
            copy_mode=getattr(args, "copy_mode", "python"))
        owners["runtime"] = rt
        future = rt.begin_contexts(early_pool)
        return future.result
    try:
        if early:
            from concurrent.futures import ThreadPoolExecutor
            early_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="early-xrt")
            lease = ValidationLease(args.manifest, workers=args.hash_workers, event=event,
                                    executable_ready=executable_ready)
        else:
            lease = ValidationLease(args.manifest, workers=args.hash_workers, event=event)
    except BaseException:
        if early_pool is not None:early_pool.shutdown(wait=True)
        raise
    _FAILURE["validation"] = lease.stats
    manifest, paths = lease.manifest, lease.paths
    owners["lease"] = lease
    timeline.mark("manifest_inventory_pinned_xclbins_validated")

    def load_tokenizer():
        started = time.perf_counter()
        import plamo_root

        tokenizer = FastPlamo3Tokenizer(
            _resolve(args.manifest, manifest["tokenizer_manifest"]),
            checkpoint=str(plamo_root.MODEL),
        )
        event("fast_tokenizer_load_validate", time.perf_counter() - started)
        started = time.perf_counter()
        text, source, target = PROMPTS[args.prompt]
        tokens = tokenizer.chat(
            [{"role": "user", "content": text}], source_lang=source, target_lang=target
        )
        if tokens != manifest["model"]["prompts"][args.prompt]:
            raise AssertionError(
                "actual user prompt tokenization differs from exported reference"
            )
        event("actual_prompt_tokenization", time.perf_counter() - started)
        return tokenizer, tokens

    try:
        runtime = owners["runtime"] or DirectRuntime(
            manifest,
            paths,
            event=event,
            artifact_lease=lease,
            validation_barrier=lease.require_validated,
            copy_mode=getattr(args, "copy_mode", "python"),
        )
        owners["runtime"] = runtime
        if args.overlap:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="startup-tokenizer"
            ) as pool:
                token_future = pool.submit(load_tokenizer)
                runtime.open(concurrent_setup=True)
                tokenizer, tokens = token_future.result()
            # Final identity/inventory checks happen after every setup consumer.
            # Device execution is also guarded inside DirectRuntime.execute.
            lease.wait()
            timeline.mark("runtime_tokenizer_and_strict_validation_ready")
        else:
            lease.wait()
            timeline.mark("strict_manifest_source_config_artifact_validation")
            tokenizer, tokens = load_tokenizer()
            timeline.mark("fast_tokenizer_and_actual_prompt_ready")
            runtime.open(concurrent_setup=False)
            timeline.mark("device_weights_xclbins_contexts_ready")
    except BaseException:
        # Joining may itself report the original content failure; preserve the
        # setup error if there was a separate one and retain pinned resources.
        try:
            lease.join_hashes()
        except BaseException:
            pass
        raise
    finally:
        if early_pool is not None:early_pool.shutdown(wait=True)
    session = RuntimeSession(runtime)
    # Binding is setup only: no runlist is executed and no hidden warmup occurs.
    first_rows = min(int(manifest["model"].get("prefill_rows", 32)), len(tokens))
    runtime.prepare(
        "prefill", {"start": 0, "rows": first_rows, "position": first_rows - 1}
    )
    runtime.prepare("decode", {"position": len(tokens)})
    timeline.mark("instruction_bos_runs_runlists_prepared")
    return {
        "helper": helper,
        "clock": clock,
        "timeline": timeline,
        "runtime": runtime,
        "session": session,
        "tokenizer": tokenizer,
        "tokens": tokens,
        "blocked": blocked,
        "imports": imports,
        "forbidden": forbidden,
        "manifest": manifest,
        "validation_lease": lease,
    }


def worker(args):
    state = initialize(args)
    timeline, session = state["timeline"], state["session"]
    first = session.prefill(state["tokens"])
    timeline.mark("first_prefill_first_output_token")
    generated = [first["token"]]
    samples = []
    for index in range(args.decode_tokens):
        start = timeline.now()
        result = session.decode(generated[-1])
        end = timeline.now()
        generated.append(result["token"])
        samples.append(
            {
                "token": index + 1,
                "position": result["position"],
                "start_ms": start,
                "end_ms": end,
                "wall_ms": end - start,
                "control": result["stats"],
            }
        )
        if index == 0:
            timeline.mark("first_decode_token")
    timing_done = timeline.now()
    steady = state["helper"].steady_state(samples)
    measured_spans = list(timeline.spans)
    measured_phases = list(timeline.events)
    # All output comparison and repeated inference occur after timing is frozen.
    from runtime_session import check_references

    accuracy_start = timeline.now()
    accuracy = check_references(
        session, generated=generated, generated_prompt=args.prompt
    )
    accuracy_elapsed = timeline.now() - accuracy_start
    prohibited_loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in state["forbidden"]
        )
    )
    if prohibited_loaded or state["blocked"]:
        raise AssertionError("cached runtime import/process guard was violated")
    result = {
        "status": "PASS",
        "mode": "fresh-process",
        "pid": os.getpid(),
        "clock": state["clock"],
        "argv": sys.argv,
        "python_flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site},
        "prompt": args.prompt,
        "prompt_tokens": len(state["tokens"]),
        "decode_tokens": args.decode_tokens,
        "overlap_setup": args.overlap,
        "overlap_validation_tokenizer_setup": args.overlap,
        "hash_workers": args.hash_workers,
        "manifest_sha256": state["manifest"]["manifest_sha256"],
        "validation": dict(state["validation_lease"].stats),
        "phases": measured_phases,
        "load_spans": measured_spans,
        "prefill": first,
        "decode_samples": samples,
        "steady_state": steady,
        "timing_done_ms": timing_done,
        "generated_tokens": generated,
        "steady_reached_within_1s": steady["reached_ms"] is not None
        and steady["reached_ms"] <= 1000,
        "steady_confirmed_within_1s": steady["confirmed_ms"] is not None
        and steady["confirmed_ms"] <= 1000,
        "accuracy_after_timing": accuracy,
        "accuracy_elapsed_ms": accuracy_elapsed,
        "blocked_attempts": state["blocked"],
        "forbidden_loaded": prohibited_loaded,
        "imports": state["imports"],
        "compile_process_launches": 0,
    }
    save(args.output, result)
    print(
        json.dumps(
            {
                "startup_result": {
                    "status": "PASS",
                    "output": str(args.output),
                    "steady_state": steady,
                    "accuracy_rows": accuracy["rows"],
                }
            }
        ),
        flush=True,
    )
    return 0


def service(args):
    state = initialize(args)
    timeline = state["timeline"]
    warm_start = timeline.now()
    first = state["session"].prefill(state["tokens"])
    warm_tokens = [first["token"]]
    warm_samples = []
    for index in range(args.decode_tokens):
        start = timeline.now()
        result = state["session"].decode(warm_tokens[-1])
        end = timeline.now()
        warm_tokens.append(result["token"])
        warm_samples.append(
            {
                "token": index + 1,
                "start_ms": start,
                "end_ms": end,
                "wall_ms": end - start,
                "control": result["stats"],
            }
        )
    warm_steady = state["helper"].steady_state(warm_samples)
    if warm_steady["confirmed_ms"] is None:
        raise RuntimeError(
            "resident service warmup did not establish the declared steady-state criterion"
        )
    timeline.mark("resident_service_warmup_complete")
    warmup = {
        "prompt": args.prompt,
        "prefill": first,
        "decode_tokens": len(warm_samples),
        "start_ms": warm_start,
        "end_ms": timeline.now(),
        "duration_ms": timeline.now() - warm_start,
        "decode_samples": warm_samples,
        "steady_state": warm_steady,
        "fresh_process_1s_goal": False,
    }
    ready = state["timeline"].now()
    print(
        json.dumps(
            {
                "service_ready": {
                    "pid": os.getpid(),
                    "process_ready_ms": ready,
                    "mode": "resident-service",
                    "fresh_process_1s_goal": False,
                    "warmup_decode_tokens": len(warm_samples),
                    "warmup_steady_state": warm_steady,
                }
            }
        ),
        flush=True,
    )
    responses = []
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        request_start = time.perf_counter()
        messages = request.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": request["text"]}]
        tokens = state["tokenizer"].chat(
            messages,
            source_lang=request.get("source_lang"),
            target_lang=request["target_lang"],
        )
        count = int(request.get("max_new_tokens", 25))
        if not 1 <= count <= state["session"].max_seq - len(tokens) + 1:
            raise ValueError("service generation count exceeds exported context")
        first = state["session"].prefill(tokens)
        first_ms = (time.perf_counter() - request_start) * 1000
        generated = [first["token"]]
        eos = frozenset(state["manifest"]["model"]["eos"])
        decode_samples = []
        while len(generated) < count and generated[-1] not in eos:
            start = time.perf_counter()
            result = state["session"].decode(generated[-1])
            end = time.perf_counter()
            generated.append(result["token"])
            decode_samples.append(
                {
                    "token": len(generated) - 1,
                    "wall_ms": (end - start) * 1000,
                    "start_ms": (start - request_start) * 1000,
                    "end_ms": (end - request_start) * 1000,
                    "control": result["stats"],
                }
            )
        reference_prompt = None
        reference_equal = None
        for reference in state["manifest"]["references"]["prompts"]:
            index = int(reference["prompt"])
            if tokens == state["manifest"]["model"]["prompts"][index]:
                reference_prompt = index
                live = int(reference["live_rows"])
                wanted = min(count, live)
                reference_equal = (
                    len(generated) == wanted
                    and generated == reference["actual_tokens"][:wanted]
                )
                if not reference_equal:
                    raise AssertionError(
                        f"resident service greedy response differs from reference prompt {index}"
                    )
                break
        response = {
            "id": request.get("id"),
            "mode": "resident-service",
            "fresh_process_1s_goal": False,
            "request_ttft_ms": first_ms,
            "request_total_ms": (time.perf_counter() - request_start) * 1000,
            "prompt_tokens": len(tokens),
            "generated_tokens": generated,
            "text": state["tokenizer"].decode(generated),
            "decode_samples": decode_samples,
            "decode_median_ms": (
                statistics.median(row["wall_ms"] for row in decode_samples)
                if decode_samples
                else None
            ),
            "reference_prompt": reference_prompt,
            "greedy_reference_equal": reference_equal,
        }
        responses.append(response)
        print(
            json.dumps({"service_response": response}, ensure_ascii=False), flush=True
        )
    # A service probe still verifies its runtime; this occurs after all requests.
    from runtime_session import check_references

    accuracy = check_references(state["session"])
    prohibited_loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in state["forbidden"]
        )
    )
    if prohibited_loaded or state["blocked"]:
        raise AssertionError("cached runtime import/process guard was violated")
    result = {
        "status": "PASS",
        "mode": "resident-service",
        "fresh_process_1s_goal": False,
        "process_ready_ms": ready,
        "phases": state["timeline"].events,
        "warmup": warmup,
        "responses": responses,
        "accuracy_after_requests": accuracy,
        "blocked_attempts": state["blocked"],
        "imports": state["imports"],
        "validation": dict(state["validation_lease"].stats),
    }
    save(args.output, result)
    return 0


def _child_command(args, output, *, service_mode=False):
    command = [
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        "--service" if service_mode else "--worker",
        "--manifest",
        str(args.manifest),
        "--output",
        str(output),
        "--prompt",
        str(args.prompt),
        "--decode-tokens",
        str(args.decode_tokens),
        "--hash-workers",
        str(args.hash_workers),
        "--validation",
        getattr(args, "validation", "content"),
        "--copy-mode",
        getattr(args, "copy_mode", "python"),
        "--timing-helper",
        str(args.timing_helper),
    ]
    if args.overlap:
        command.append("--overlap")
    return command


def supervisor(args):
    directory = args.output.parent
    directory.mkdir(parents=True, exist_ok=True)
    children = []
    for index in range(args.repeats):
        output = directory / f"startup-{index + 1}.json"
        log = directory / f"startup-{index + 1}.log"
        command = _child_command(args, output)
        env = dict(os.environ, PLAMO_STARTUP_LAUNCH_NS=str(time.perf_counter_ns()))
        with log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=env
            )
            try:
                code = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                code = 124
        print(
            json.dumps(
                {
                    "startup_child": index + 1,
                    "exit_code": code,
                    "log": str(log),
                    "output": str(output),
                }
            ),
            flush=True,
        )
        if code != 0 or not output.exists():
            save(
                args.output,
                {
                    "status": "FAIL",
                    "mode": "fresh-process",
                    "failed_child": index + 1,
                    "exit_code": code,
                    "log": str(log),
                    "children": children,
                },
            )
            return code or 1
        record = json.loads(output.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            raise AssertionError("child exited successfully without a passing report")
        children.append(record)
    names = [phase["phase"] for phase in children[0]["phases"]]
    phases = []
    for name in names:
        rows = [
            next(phase for phase in child["phases"] if phase["phase"] == name)
            for child in children
        ]
        phases.append(
            {
                "phase": name,
                "median_end_ms": statistics.median(row["end_ms"] for row in rows),
                "median_duration_ms": statistics.median(
                    row["duration_ms"] for row in rows
                ),
            }
        )
    steady = {}
    for field in (
        "baseline_ms",
        "gap_ms",
        "floor_ratio",
        "effective_weight_GB_s",
        "reached_ms",
        "confirmed_ms",
    ):
        values = [child["steady_state"][field] for child in children]
        steady[field] = (
            statistics.median(values)
            if all(value is not None for value in values)
            else None
        )
    result = {
        "status": "PASS",
        "mode": "fresh-process",
        "repeats": len(children),
        "phases_median": phases,
        "steady_state_median": steady,
        "steady_reached_within_1s_all": all(
            child["steady_reached_within_1s"] for child in children
        ),
        "steady_confirmed_within_1s_all": all(
            child["steady_confirmed_within_1s"] for child in children
        ),
        "accuracy_pass_all": all(
            child["accuracy_after_timing"]["passed"] for child in children
        ),
        "overlap_setup": args.overlap,
        "children": children,
    }
    save(args.output, result)
    print(
        json.dumps(
            {
                "startup_summary": {
                    key: value for key, value in result.items() if key != "children"
                }
            }
        ),
        flush=True,
    )
    return 0


def service_test(args):
    import queue
    import threading

    output = args.output.with_name(args.output.stem + "-worker.json")
    log = args.output.with_suffix(".log")
    requests = []
    for index in (args.prompt, 0):
        text, source, target = PROMPTS[index]
        requests.append(
            {
                "id": f"request-{len(requests) + 1}",
                "text": text,
                "source_lang": source,
                "target_lang": target,
                "max_new_tokens": 25,
            }
        )
    command = _child_command(args, output, service_mode=True)
    env = dict(os.environ, PLAMO_STARTUP_LAUNCH_NS=str(time.perf_counter_ns()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client_requests = []
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
        )
        lines = queue.Queue()

        def receive():
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(
            target=receive, name="resident-service-output", daemon=True
        )
        reader.start()
        deadline = time.monotonic() + args.timeout

        def next_message(kind):
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, args.timeout)
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    raise subprocess.TimeoutExpired(command, args.timeout) from None
                if line is None:
                    raise RuntimeError(f"service ended before {kind}")
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if kind in message:
                    return message[kind]

        try:
            next_message("service_ready")
            for request in requests:
                started = time.perf_counter()
                child.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                child.stdin.flush()
                response = next_message("service_response")
                elapsed = (time.perf_counter() - started) * 1000
                if response.get("id") != request["id"]:
                    raise AssertionError("service response id differs from request")
                client_requests.append(
                    {"id": request["id"], "client_response_ms": elapsed}
                )
            child.stdin.close()
            child.wait(timeout=max(0.001, deadline - time.monotonic()))
            code = child.returncode
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            code = 124
        except BaseException:
            child.kill()
            child.wait()
            raise
        finally:
            reader.join(timeout=5)
    if code or not output.exists():
        save(
            args.output,
            {
                "status": "FAIL",
                "mode": "resident-service",
                "exit_code": code,
                "log": str(log),
            },
        )
        return code or 1
    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or len(report.get("responses", [])) != 2:
        raise AssertionError("resident service did not finish two verified requests")
    if not all(
        response["greedy_reference_equal"] is True for response in report["responses"]
    ):
        raise AssertionError(
            "resident service test responses were not checked against exported references"
        )
    report["command"] = command
    report["log"] = str(log)
    report["requests_sent_after_ready"] = True
    report["client_requests"] = client_requests
    save(args.output, report)
    print(
        json.dumps(
            {
                "resident_service_summary": {
                    key: report[key]
                    for key in (
                        "status",
                        "mode",
                        "fresh_process_1s_goal",
                        "process_ready_ms",
                        "responses",
                    )
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--worker", action="store_true")
    group.add_argument("--service", action="store_true")
    group.add_argument("--service-test", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument("--prompt", type=int, choices=range(4), default=3)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument(
        "--validation", choices=("content", "identity"), default="content"
    )
    parser.add_argument(
        "--copy-mode", choices=("python", "memmove", "readinto"), default="python"
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--timing-helper",
        type=Path,
        default=Path(__file__).with_name("startup_probe.py"),
    )
    args = parser.parse_args()
    if (
        args.repeats < 3
        or args.decode_tokens < 24
        or args.hash_workers < 1
        or args.timeout <= 0
    ):
        parser.error(
            "require >=3 fresh processes, >=24 decode tokens, positive hash workers and timeout"
        )
    return args


def main():
    args = arguments()
    try:
        code = (
            service(args)
            if args.service
            else (
                worker(args)
                if args.worker
                else service_test(args) if args.service_test else supervisor(args)
            )
        )
    except BaseException as exc:
        traceback.print_exc()
        save(args.output, {"status": "FAIL", "exception": repr(exc), **_FAILURE})
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt" and (args.worker or args.service):
        # Known Windows XRT object-destruction hazard. All reports are persisted
        # and native submissions have completed before successful termination.
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
