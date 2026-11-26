# Usage

Built with PLaMo. Unofficial; not affiliated with Preferred Networks, Inc. or Advanced Micro Devices, Inc.
Provided "AS IS", without warranty. No telemetry.

Run these from the install folder after [INSTALL.md](INSTALL.md). Examples use Windows PowerShell 5.1 or PowerShell 7.

## One-shot translation (unified CLI, up to 262,143 tokens)

Use `translate_long.ps1` for inputs of any length: short requests (prompt + output cap ≤ 2,048 tokens, without
`-Long`) are passed unchanged to the pure-NPU speculative runtime (`translate_plamo3.ps1`), while longer requests use
**256k Long mode** (Radeon iGPU prefill + NPU W4 serial decode; see [LONG.md](LONG.md)).

```powershell
.\translate_long.ps1 -InputFile report.txt -SourceLang ja -TargetLang en
.\translate_long.ps1 "今日はいい天気ですね。" -SourceLang ja -TargetLang en
.\translate_long.ps1 "Thank you for your help." -SourceLang en -TargetLang ja
.\translate_long.ps1 "Hello." -SourceLang en -TargetLang ja -Json
```

*(If you installed only the NPU runtime without `install_long.py --with-model`, use `.\translate_plamo3.ps1` directly
for requests up to 2,048 total tokens.)*

The translation is printed to standard output (UTF-8). Diagnostics go to standard error and are silent unless you ask
for them or standard error is a terminal (see `-Stats`).

| Option | Default | Meaning |
|---|---|---|
| `<text>` (first argument) | | The text to translate. Must not be empty. Use quotes. |
| `-SourceLang ja\|en` | `ja` | Language of the text |
| `-TargetLang ja\|en` | `en` | Language to translate into. Same as the source language is an error. |
| `-NumTokens N` | 256 | Maximum number of generated tokens, 1 to 2047. Output is cut at this length. |
| `-Mode w4-m2\|w4-serial\|w8-serial` | `w4-m2` | Decoding mode (see below) |
| `-SpecOff` | | Same as `-Mode w4-serial` (unless `-Mode w8-serial` was given) |
| `-Json` | | Print one JSON object: `text`, `tokens`, `mode`, `fallbacks`, `prompt_tokens`, `ttft_ms`, `wall_ms`, `spec_ready_ms` |
| `-Service` | | Long-running JSON-lines service (below) |
| `-Stats` / `-Quiet` | tty only | Print (or suppress) the one-line timing summary on standard error. Also `PLAMO3_STATS=1\|0`. Without a terminal on standard error nothing is printed, so `2>&1` captures only the translation. |
| `-Strict` | | A completed but degraded run (for example a fallback to serial decoding) exits with code 4 instead of 0 |
| `-Turbo` | | Opt-in NPU turbo power mode (below) |
| `-KeepWarm` | | Opt-in CPU keep-warm helper (below) |
| `-Wait SECONDS` | 10 | How long to wait for another PLaMo process to release the NPU |

The same switches exist on the Python entry point (`venv\Scripts\python.exe IRON\iron\applications\plamo3\translate.py`)
as `--source-lang`, `--target-lang`, `--num-tokens`, `--mode`, `--spec-off`, `--json`, `--service`, `--stats`,
`--quiet`, `--strict`, `--turbo`, `--keep-warm` / `--no-keep-warm`, `--wait`. `--version` prints the version,
"Built with PLaMo" and the licence pointer. `--env-report` prints the environment record (paths with `%USERPROFILE%`
in place of your profile directory). `--diag` collects release information, the environment record, the receipt status
and the last log lines, without any translated text.

Input limits: the prompt plus the output must fit in 2048 tokens; a prompt of more than 2045 tokens is refused with
exit code 2 and a message telling you to split the text into paragraphs. Text that contains the reserved marker
`<|plamo:` is refused.

### Modes

| Mode | Notes |
|---|---|
| `w4-m2` | Default. W4 body, exact speculative decoding with the shipped drafter. Identical output to `w4-serial`. If the speculative path cannot start, the program **refuses to continue silently in a slower mode**: it exits with code 3 and says why. `--allow-serial-fallback` (or `PLAMO3_ALLOW_FALLBACK=1`) lets it answer in serial W4 with a three-line warning instead. Failures in the middle of a request (for example the helper process dies) are handled exactly and automatically: the output stays token-identical, one warning line is printed, and `fallbacks` in `-Json` lists them. |
| `w4-serial` | Same quantised model, one token per step, about 25% slower. Certified bitwise against an independent CPU reference. |
| `w8-serial` | 8-bit weights, one token per step. Not claimed identical to W4. |

## Service mode (many requests, model stays loaded)

```powershell
.\translate_plamo3.ps1 -Service
```

Standard output carries only JSON lines. The first line announces readiness:
`{"service_ready": true, "mode": "w4-m2", "spec_ready_ms": 770.0}`. Then send one JSON object per line:

```json
{"id": 1, "text": "今日はいい天気ですね。", "source_lang": "ja", "target_lang": "en", "max_new_tokens": 256}
```

Each request gets one reply line, in order: `id`, `text`, `tokens`, `mode`, `fallbacks`, `prompt_tokens`, `ttft_ms`,
`wall_ms`, `spec_ready_ms`. A bad request (empty text, unsupported language, too long) gets
`{"id": ..., "error": "...", "kind": "invalid"}` and the service keeps running. A fatal problem (device error,
integrity failure) gets `{"error": "...", "kind": "fatal"}` and the process exits with 5 or 3. Closing standard input
ends the service cleanly (exit 0). Diagnostics are on standard error only. The service holds the NPU for its lifetime.

Use the service for scripted batches: processes started back to back (less than about 0.3 s apart) can hit a driver
stall of about 0.5 s on their first NPU context.

## Exit codes

| Code | Meaning | Typical message |
|---:|---|---|
| 0 | OK | |
| 1 | Internal error | `PLaMo: internal error (<type>); please run translate_plamo3.ps1 -Diag and report` (traceback only with `PLAMO3_DEBUG=1`) |
| 2 | Usage or invalid request | `error: --num-tokens must be between 1 and 2047`; `input is 2301 tokens; the limit is 2045` |
| 3 | Installation problem or integrity refusal | `PLaMo: the installation is incomplete or modified (<reason>: <file>). Run install.ps1 -Verify.` |
| 4 | Degraded run completed (only with `-Strict`) | warning block plus the result |
| 5 | NPU unavailable or busy | `another PLaMo process (pid 1234, started 10:21:03, mode w4-m2 --service) is using the NPU`; `no NPU/driver` |
| 6 | Environment problem | `Python 3.13 (64-bit) required; found 3.12.4`; `xrt_coreutil.dll not found: install the AMD NPU driver` |
| 130 | Ctrl+C | quiet |

Integrity messages name one of these reasons: file missing, file changed, size changed, extra source file, receipt
stale (not an error: the run is correct but slower, and you are told to run `install.ps1 -Reattest`), or a
configuration variable that you set (the message names it).

## One NPU job at a time

A named lock (`Global\PLaMo3-XDNA1-NPU`) serialises PLaMo processes. A second process waits up to `-Wait` seconds, then
exits with code 5 and tells you who holds the NPU. `--no-lock` is an escape hatch for developers and is not used by the
self-test. Do not run two NPU programs of any kind at once: undefined behaviour on the device is possible.

## Opt-in switches

### Turbo power mode (`-Turbo`, `PLAMO3_NPU_TURBO=1`)

Runs `xrt-smi configure --pmode turbo` for the run and restores the previous mode afterwards (a crash marker restores
it at the next start after a hard kill). It is **system-wide while it is on**, needs AC power to have an effect, and
adds about 0.12 s to startup, so it pays off from about 130 output tokens and in `-Service` sessions. About 6% faster
on long outputs; results are identical. If you lack the privilege the program continues in the current mode and says
so:

```
plamo3-turbo: could not switch the NPU power mode to turbo (xrt-smi exit 1: ...). This usually needs an elevated
(administrator) PowerShell. Continuing in <current> mode; results are identical, only speed differs.
```

It never fails a translation. `--turbo-strict` turns "could not switch" into exit code 5 for benchmarking. Undo by hand
with `& "$env:SystemRoot\System32\AMD\xrt-smi.exe" configure --pmode default`. The installer never enables turbo.

### Keep-warm (`-KeepWarm`, `PLAMO3_KEEP_WARM=1`, `auto` = only on AC power, `-NoKeepWarm` forces off)

An idle-priority helper on the same logical CPU keeps the decode core out of deep idle while generating. About 4.5%
faster on long outputs; costs one logical CPU at 100% while translating (power and heat not measured); nothing when
idle in `-Service`; changes no system setting. With it on the hybrid wait (below) is not installed.

## Environment variables

| Variable | Meaning |
|---|---|
| `PLAMO3_NPU_TURBO=1` | Same as `-Turbo` |
| `PLAMO3_KEEP_WARM=1\|auto\|0` | Same as `-KeepWarm` / `-NoKeepWarm` |
| `PLAMO3_HYBRID_WAIT=0` | Turn off the hybrid command-completion wait (a timer sleep and a poll before the ordinary wait; output never depends on it; it disables itself after 4 consecutive timeouts) |
| `PLAMO3_STATS=1\|0` | Timing summary on standard error |
| `PLAMO3_ALLOW_FALLBACK=1` | Same as `--allow-serial-fallback` |
| `PLAMO3_DEBUG=1` | Show tracebacks |
| `R77_EXIT=off` | Normal process shutdown instead of the fast exit after the output is flushed |
| `R77_TOK=off` | Rebuild the original tokenizer instead of the precomputed table |
| `R77_REPORT_STARTUP=1` | Per-phase startup events in `-Json` output |

Do **not** set any `IRON_PLAMO_*` variable: the runtime's configuration check refuses to start if one is set (the error
names it).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Exit 3 "installation is incomplete or modified" | `.\install.ps1 -Verify`; re-run the installer to repair. Never edit files in the install folder. |
| Message "install receipt missing or stale; hashing all files (about 15 s)" | The folder was moved or copied. `.\install.ps1 -Reattest`. |
| Exit 5 "another PLaMo process is using the NPU" | Close the other process or service; or wait with `-Wait`. |
| A start takes about 1.1 s instead of 0.8 s | Driver stall after a start less than 0.3 s after the previous exit; wait 0.5 s between runs or use `-Service`. |
| First start after a reboot is slow | Large files come from disk; later starts are faster. |
| `2>&1` in PowerShell 5.1 used to abort the script | Fixed: the timing line is only printed on a terminal or with `-Stats`. |
| Antivirus quarantines something | Report it (no executables are shipped; `pyxrt.pyd` is a Python extension, its hash is in `SHA256SUMS`). |
| Garbled Japanese in the console | Use Windows Terminal or set the console to UTF-8; files and pipes are UTF-8. |

Security, trust model and reporting: [SECURITY.md](SECURITY.md). Licences: [NOTICE](NOTICE),
[LICENSE-PLaMo-en.md](LICENSE-PLaMo-en.md).

## `translate_long.ps1` and `install_long.py` (256k Long mode)

`translate_long.ps1 [-Text] [-InputFile FILE] -SourceLang ja|en -TargetLang en|ja [-NumTokens N] [-Long] [-NoRepeatStop] [-Json] [-Stats] [-Quiet] [-Wait S] [-NoLock]`, plus every switch of `translate_plamo3.ps1`. A request that fits the normal path (prompt + output cap <= 2,048 tokens, no `-Long`) is passed unchanged to `translate_plamo3.ps1`. For prompts of 1,793-2,045 tokens without `-NumTokens` the normal path stops at its context limit as before; add `-Long` to translate the full output. `install_long.py [--with-model] [--verify] [--plan] [--yes]` sets up or checks 256k Long mode. Details, limits, speed and exit codes: [LONG.md](LONG.md).
