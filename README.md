# PLaMo 3 Translate on the AMD XDNA1 NPU (up to 256k Context)

**Built with PLaMo.**

[English](README.md) | [日本語 (README_ja.md)](README_ja.md)

Fast local Japanese <-> English translation with [PLaMo 3 Translate 0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b)
on AMD Ryzen "Phoenix" laptops (XDNA1 NPU + Radeon iGPU) under Windows 11 — supporting **up to 256k tokens (262,143
input tokens)** through one unified command (`translate_long.ps1`):

- **Long inputs (> 2,048 tokens, up to 262,143 tokens — Long mode):** **Radeon iGPU prefill + XDNA1 NPU W4 decode**
  (8k tokens: ~8 s prefill / 21–22 ms/token; 32k tokens: ~22 s / 24 ms/token; 262k tokens: ~5 min / ~50 ms/token).
- **Short / standard inputs (≤ 2,048 total tokens — automatic fast path):** handed directly to the pure-NPU
  **exact speculative decoding** engine (`translate_plamo3.ps1`, `w4-m2`) — ready in **0.69 s**, **13.9 ms/token**
  (~72 tokens/s), **1.4 s** one-shot short sentence, with the iGPU left completely idle and outputs token-for-token
  identical to plain greedy decoding of the same quantised model.

See [Performance](#performance) and [LONG.md](LONG.md) for measurement conditions and model-level long-context caveats.

> **Unofficial community project.** Not affiliated with, endorsed by, or supported by Preferred Networks, Inc. (PFN)
> or Advanced Micro Devices, Inc. (AMD). PLaMo is a model family of PFN. AMD, Ryzen, Radeon and XDNA are trademarks of
> AMD. This software runs PLaMo 3 Translate 0.8B under the **PLaMo Community License** (see
> [Licence](#licence-read-this-before-installing)). Provided **"AS IS", without warranty of any kind**.

## Quick start

```powershell
git clone -c core.autocrlf=false https://github.com/nilpe/plamo3-translate-xdna1.git C:\PLaMo3-NPU
cd C:\PLaMo3-NPU
.\install.ps1                        # shows the licence, installs & verifies the NPU runtime (5.4 GB), runs self-test
python install_long.py --with-model  # enables 256k Long mode (verifies long files + downloads official 1.78 GB BF16 weights from HF)

# One command for any input length:
.\translate_long.ps1 -InputFile report.txt -SourceLang ja -TargetLang en
.\translate_long.ps1 "今日はいい天気ですね。" -SourceLang ja -TargetLang en
.\translate_long.ps1 "Thank you for your help." -SourceLang en -TargetLang ja
```

- **NPU-only install (≤ 2,048 tokens only):** if you do not need long inputs (or do not have a Radeon 780M-class iGPU),
  skip `install_long.py --with-model` and use `.\translate_plamo3.ps1` directly (5.4 GB download, 7 GB disk).
- **Long-running service mode** (≤ 2,048 tokens; model stays loaded, one JSON request per stdin line):

```powershell
.\translate_long.ps1 -Service
```

See [INSTALL.md](INSTALL.md) for installation details (options, offline installs, SSH signature checks, updating,
uninstalling, troubleshooting), [LONG.md](LONG.md) for 256k Long mode details, and [USAGE.md](USAGE.md) for all CLI
options, the service protocol, environment variables, and exit codes.

## What you get

| Entry point / Mode | What it is |
|---|---|
| `translate_long.ps1` **(recommended unified CLI)** | Handles any input length up to **262,143 tokens**. Requests within 2,048 total tokens (without `-Long`) run on the fast pure-NPU `w4-m2` path; longer inputs (or `-Long`) run **Long mode** (Radeon iGPU prefill + NPU W4 serial decode, up to 262,144 total context, with repetition stop enabled by default). |
| `w4-m2` (fast short-context path, ≤ 2,048 tokens) | 4-bit-weight body with an 8-bit head, plus **exact** 2-token speculative decoding with a small drafter (`PLaMo-D20-M2`). Every emitted token equals the serial W4 token at the same position. |
| `w4-serial` | The same quantised model on NPU, one token per step (≤ 2,048 tokens). Certified bitwise against an independent order-matched CPU reference (100 standard + 64 long rows). |
| `w8-serial` | 8-bit weights on NPU, one token per step (≤ 2,048 tokens). |
| Opt-in `-Turbo` (`--turbo`) | Switches the NPU power mode to turbo for the run (may require administrator rights; fails soft). About 6% faster on long short-context outputs. |
| Opt-in `-KeepWarm` (`--keep-warm`) | Keeps one CPU core busy while generating on the short-context path. About 4.5% faster without changing any system setting. |

Everything runs locally. **No telemetry**: the translator never makes network connections (details in
[SECURITY.md](SECURITY.md)).

## Requirements

- **PC & OS**: Windows 11 (x64) with an **AMD Ryzen XDNA1 NPU** (Phoenix / Hawk Point class) and the AMD NPU driver
  installed (Device Manager shows "NPU Compute Accelerator Device").
  - For **256k Long mode**: an AMD Radeon 780M / 760M / 740M iGPU with the AMD graphics driver (tested on **Radeon 780M
    only**).
- **RAM & Disk**: 16 GB RAM (about 6 GB free for normal NPU path; about 4 GB free for Long mode) and an **NTFS** volume
  with about **9 GB** free disk space for the full 256k setup (about **7 GB** for NPU-only without the 1.78 GB long-mode
  weights; add ~0.5 GB if using `-WithGateData`).
- **Python**: **64-bit CPython 3.13** (python.org installer, `winget install Python.Python.3.13`, or conda-forge). No
  administrator rights are needed. See [INSTALL.md](INSTALL.md) for why 3.13 is required and how to verify it.
- **Internet (one-time install)**:
  - NPU runtime (`install.ps1`): about **5.4 GB** — 5.35 GB runtime payload from Hugging Face
    ([nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime)),
    3.6 MB of tokenizer/template files from [pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b),
    and two small Python wheels (`numpy`, `ml_dtypes`) from PyPI. Optional gate reference data (`-WithGateData`) adds
    about 359 MB.
  - 256k Long mode (`install_long.py --with-model`): **1.78 GB** (`model.safetensors`, downloaded directly from
    [pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b) at a pinned revision and
    SHA-256).
- **Git**: to clone this repository. To verify the OpenSSH signature of `SHA256SUMS`, the installer uses the
  `ssh-keygen` bundled with Git for Windows (found automatically; Windows' built-in OpenSSH client hangs on signed
  files larger than about 8 KB, see [INSTALL.md](INSTALL.md)).

## Performance

Measured by the author on **one machine**: AMD Ryzen 7 8840U (XDNA1 NPU + Radeon 780M iGPU), 16 GB DDR5-5600, Windows
11 build 26300, NPU driver 32.0.20101.3760, NPU power mode "Default", turbo and keep-warm off, cached build artifacts.

### 256k Long mode (`translate_long.ps1`, Radeon 780M iGPU prefill + XDNA1 NPU decode)

| Input length | Prefill (iGPU) | Decode (NPU, W4 serial) |
|---|---:|---:|
| 8k tokens | ~8 s | 21–22 ms/token |
| 32k tokens | ~22 s | 24 ms/token |
| 262k tokens (max 262,143) | ~5 min (4–5.5 min thermal spread) | ~50 ms/token |

Pipeline correctness is verified end to end (14,336 stencil cases, 10/10 NPU decode positions bitwise up to 262,143,
15 continuous decode compaction steps bitwise across 4k–32k, split-vs-nosplit MD5 identity up to 259k tokens, and
37/37 iGPU prefill vLLM logprob-agreement prompts; see [LONG.md](LONG.md)).

### Short / standard context (≤ 2,048 tokens, automatic NPU `w4-m2` exact speculative path)

Release-build decision run: 24 interleaved fresh-process pairs against the author's development build (medians;
paragraph = 515 tokens in, 595 tokens out):

| Metric | Result |
|---|---|
| Process start -> speculative decoding ready ("spec-ready") | median **0.69 s** |
| Time to first token (27–31 input tokens / 515 input tokens) | **0.11 s** / **1.43 s** |
| Decode speed on a 595-token output (515-token input), default `w4-m2` | **13.9 ms/token** (~72 tokens/s) |
| Whole paragraph request (515 tokens in, 595 out), generation wall time | **9.7 s** |

Measured with the shipped drafter (`PLaMo-D20-M2`) in `-Service` mode, 128–160 requests per set, greedy decoding:

| Request set | Decode (ms/token) | Tokens / cycle |
|---|---:|---:|
| 160 short everyday sentences (authored, ja <-> en) | 14.2 | 1.63 |
| 128 everyday sentences (Tatoeba, ja <-> en) | 14.1 | 1.56 |
| 128 encyclopaedic-style sentences and paragraphs (ja <-> en) | 14.7 | 1.47 |

Additional reference points from the development build with identical kernels:

- Time to first token at 1018 input tokens: about 2.9 s
- One-shot CLI, 15-token sentence (process start to exit): about 1.35 s
- Serial `w4-serial` decode at 27 / 515 / 1018 tokens of context: 16.95 / 17.56 / 18.20 ms/token
- Serial `w8-serial` decode, short context / 1k context: about 23.6 / 26.5 ms/token
- Opt-in `-Turbo` / `-KeepWarm` on long short-context outputs: about −6.4% / −4.5% wall time

What these numbers are **not**:

- Not a general benchmark of the chip; they are measurements on a single PC.
- "Spec-ready" is measured from a fresh process with warm OS file caches. The first start after a reboot reads large
  files from disk and can be slower (not measured).
- Short requests are not always faster than serial decoding: for 3- to 8-token outputs, rejected cycles can cost more
  than speculation saves.
- Two processes started within about 0.3 s of each other can hit a ~0.5 s driver stall (roughly 1 start in 10 in the
  author's measurements); `-Service` avoids this.
- **Memory-bandwidth context**: the 4-bit body reads about 473 MB per token; at the ~40 GB/s practical shim DDR
  bandwidth measured on this machine, that sets a serial bandwidth floor near 12 ms/token (serial W4 reaches
  16.95–18.20 ms/token). Exact speculative decoding brings effective latency down to 13.9 ms/token, closing most of the
  gap toward that serial floor without changing the output.

## Limitations & long-context model note

- **Language & context**: Japanese <-> English only. Long mode runs up to **262,143 prompt tokens** (262,144 total
  context); the pure-NPU speculative path runs up to 2,048 total tokens (prompts > 2,045 tokens route automatically to
  Long mode in `translate_long.ps1`, or are refused by `translate_plamo3.ps1`).
- **Base-model repetition limit on long inputs (experimental quality):** while the 256k pipeline runs accurately up to
  262,143 tokens, translation quality above 2,048 tokens is **unverified** due to a limitation of the base model itself:
  even the official BF16 `pfnet/plamo-3-translate-0.8b` on a GPU frequently loops or stops translating on one-shot
  inputs longer than ~2k tokens (in full-length tests on 16 law documents of 2.6k–6k tokens, official BF16 looped in
  10/16 and Long mode in 8/16; on 40 shorter 300–1,000 token pieces, Long mode matched the normal path at +0.33 chrF).
  A repetition stop is enabled by default (`-NoRepeatStop` disables it) and a warning prints on every long run. See
  [LONG.md](LONG.md).
- **Hardware & OS**: **Tested only** on a Ryzen 7 8840U (XDNA1 NPU + Radeon 780M iGPU), NPU driver 32.0.20101.3760,
  Windows 11 build 26300. Other XDNA1 / RDNA3 PCs (for example Hawk Point, Ryzen 7 8040 series) may work but are
  **untested**. XDNA2 NPUs (Strix Point / Krackan) are not supported. Other driver or Windows versions are recorded and
  produce a warning, nothing more.
- **Single NPU job**: one NPU job at a time; a second PLaMo process waits or exits reporting that the NPU is busy.
  `--service` is available on the ≤ 2,048-token path only (not in Long mode).
- **ASCII-only install path**: spaces are fine, but the folder path must contain only ASCII characters. XRT opens files
  through the Windows ANSI API, so a folder name with non-ASCII characters fails at startup. Both installers and
  runtimes refuse non-ASCII paths with exit code 3 (see [INSTALL.md](INSTALL.md)).
- **Do not move or copy after install**: a moved or copied tree has new NTFS file identities, causing every start to
  re-hash about 5.4 GB (10–20 s). Run `.\install.ps1 -Reattest` if you move the folder.
- **Strict integrity checks**: modified files are refused by design (see [SECURITY.md](SECURITY.md)).
- **Antivirus**: no custom native executables are shipped (only `python.exe` from your venv runs, plus `pyxrt.pyd` as a
  Python extension and precompiled HIP/NPU code objects), though real-time scanners can slow down hashing passes.

## Licence: read this before installing

- **Code in this repository**: Apache License 2.0 ([LICENSE](LICENSE)). Includes files derived from
  [amd/IRON](https://github.com/amd/IRON) (Apache-2.0); see [NOTICE](NOTICE).
- **Model files (runtime payload)**: quantised weight images, embeddings, precomputed tokenizer table, the
  speculative-decoding drafter, and pre-generated NPU command streams downloaded from Hugging Face are **DERIVED
  MODELs** of PLaMo 3 Translate and are licensed under the **PLaMo Community License** of Preferred Networks, Inc.:
  [LICENSE-PLaMo-en.md](LICENSE-PLaMo-en.md) (English) and [LICENSE-PLaMo-ja.md](LICENSE-PLaMo-ja.md) (Japanese
  original). The installer displays a summary and asks you to accept it.
- **Commercial use**: personal, academic, and other non-commercial use is free. **If you use this for a commercial
  purpose** (including internal company use, products or services, hosting, or APIs), **you must register with PFN
  yourself before you start**, and you are subject to the licence's revenue condition. Nothing here registers you or
  grants commercial rights. Read the full licence; this summary is not legal advice.
- **Base model**: `pfnet/plamo-3-translate-0.8b` is **not** redistributed here. `install.ps1` downloads the required
  tokenizer/config files (3.6 MB) and `install_long.py --with-model` downloads `model.safetensors` (1.78 GB) directly
  from Hugging Face at pinned revision `8b4d42855622093e610ba9b90c2d2b59f8d5c15d` and verifies their hashes.
- **Attribution**: "Built with PLaMo", as required by the licence. Model names in this project include "PLaMo".
  Third-party notices (AMD IRON, mlir-aie, XRT/`pyxrt`, pybind11, data credits): [NOTICE](NOTICE).
- **Data credits for the drafter** (`PLaMo-D20-M2`; full details in [NOTICE](NOTICE) section 6). The drafter was
  trained on PLaMo 3 Translate's own translations of source sentences from permissively licensed sources only; no
  source text is distributed:
  - Sentences from [Tatoeba](https://tatoeba.org), contributed by the Tatoeba community under
    [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) (CC0 where Tatoeba lists a sentence as CC0). The
    contributor of every sentence used is recorded in `plamo3-drafter-port-v7-attribution/PLaMo-ATTRIBUTION-D20.jsonl.gz`
    (shipped with the payload, covered by `SHA256SUMS` and verified by the installer).
  - OpenAssistant OASST1/OASST2 (Apache-2.0, the Open-Assistant contributors).
  - e-Gov Law Search statutes (no copyright under Japanese Copyright Act Article 13(i); Digital Agency, Government
    Standard Terms of Use 2.0).
  - Aozora Bunko and Project Gutenberg works (public domain).

## Privacy and security

No telemetry, no analytics, no crash reports, and no update checks. The installers contact `huggingface.co` (runtime
payload and base model files) and `pypi.org` (two pinned, hash-checked wheels); the translator itself makes no network
connections. Your text stays on your machine. Every shipped file is verified against `SHA256SUMS` (signed with the
maintainer's SSH key; see [INSTALL.md](INSTALL.md)), and both runtimes re-check their inputs at every start. See
[SECURITY.md](SECURITY.md) for the trust model and vulnerability reporting.

## Support

Best effort, no promises. Please open an issue and attach `state\selftest-report.json` and `state\env-record.json`
(they contain no personal data; paths show `%USERPROFILE%` in place of your profile directory). Reports on hardware,
driver, or Windows versions other than the tested ones are welcome, though the author cannot reproduce them locally.

## Related repositories

- **NPU runtime payload (Hugging Face)**:
  [nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime)
  — prebuilt quantised weights, command streams, `pyxrt.pyd`, and the `PLaMo-D20-M2` drafter downloaded by `install.ps1`.
- **Separate GPU speculative-decoding drafters (Hugging Face)**:
  - [nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU)
    — DeepSeek-V3-style Multi-Token Prediction (D=3) drafters for NVIDIA GPUs (BF16 target).
  - [nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU)
    — DSpark parallel-block drafter for NVIDIA GPUs (BF16 target).

## Repository layout

| Path | What |
|---|---|
| `README.md`, `README_ja.md` | Project overview (English / Japanese) |
| `INSTALL.md`, `LONG.md`, `USAGE.md`, `SECURITY.md` | Installation, 256k Long mode guide, CLI/service reference, security & trust model |
| `translate_long.ps1` | Primary unified CLI (any length up to 262,143 tokens; delegates ≤ 2,048 tokens to `translate_plamo3.ps1`) |
| `translate_plamo3.ps1` / `.cmd` | Pure-NPU short-context CLI (≤ 2,048 tokens) |
| `install.ps1`, `install.py`, `install_long.py`, `uninstall.ps1`, `selftest.py` | Installers (NPU runtime + 256k Long mode), uninstaller, and self-test |
| `long/`, `plamo3-long/`, `long-manifest.json`, `long-certificate.json` | 256k Long mode runtime (iGPU HIP prefill + NPU copy-and-patch decode, hash-bound) |
| `IRON/`, `lib/` | Pure-NPU runtime source (hash-bound) |
| `hf/` | Model card (`README.md`, `README_ja.md`) for the Hugging Face runtime payload repository |
| `SHA256SUMS`, `SHA256SUMS.sig`, `PAYLOAD-MANIFEST.json`, `release.json`, `known-good-env.json` | Release metadata and signature |
| `tools/` | Release tooling: public-tree exporter, scrub gate, long-mode gates, release scripts |
| `tests/` | Unit tests for the installer, release tools, and Long mode |

The large NPU runtime payload (`plamo3-*`, `xrt/`, optional `gate-data/`) lives on Hugging Face and is placed next to
these files by `install.ps1`.
