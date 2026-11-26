# SPDX-License-Identifier: Apache-2.0
"""Bound every R20 worker; preserve its complete log and actual exit status."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--log", type=Path, required=True)
p.add_argument("--timeout", type=int, default=540)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = [
    sys.executable,
    "-u",
    *(a.command[1:] if a.command[:1] == ["--"] else a.command),
]
start = time.monotonic()
timed_out = False
with a.log.open("w", encoding="utf-8") as output:
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    while child.poll() is None:
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "progress": "r20-worker",
                        "elapsed_s": round(elapsed),
                        "pid": child.pid,
                    }
                ),
                flush=True,
            )
            if elapsed >= a.timeout:
                timed_out = True
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    child.kill()
                child.wait()
                break
code = 124 if timed_out else child.returncode
print(
    json.dumps(
        {
            "result": "PASS" if code == 0 else "FAIL",
            "exit_code": code,
            "seconds": time.monotonic() - start,
            "log": str(a.log),
            "command": command,
            "timeout": timed_out,
        }
    ),
    flush=True,
)
sys.exit(code)
