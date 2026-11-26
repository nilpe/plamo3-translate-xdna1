---
license: other
license_name: plamo-community-license
license_link: https://plamo.preferredai.jp/info/plamo-community-license-en
base_model: pfnet/plamo-3-translate-0.8b
base_model_relation: quantized
language:
- ja
- en
pipeline_tag: translation
tags:
- plamo
- translation
- npu
- xdna
- amd
- ryzen-ai
- speculative-decoding
- long-context
---

# PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime

**Built with PLaMo.**

[English](README.md) | [日本語 (README_ja.md)](README_ja.md)

> **Unofficial community project.** Not affiliated with, endorsed by, or supported by Preferred Networks, Inc. (PFN) or
> Advanced Micro Devices, Inc. (AMD). PLaMo is a model family of PFN. AMD, Ryzen, Radeon and XDNA are trademarks of AMD.
> Provided **"AS IS", without warranty of any kind**.

## What this repository is

This repository hosts the **NPU runtime payload** for running
[PLaMo 3 Translate 0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b) on AMD Ryzen "Phoenix" PCs (XDNA1 NPU +
Radeon iGPU) under Windows 11, supporting **up to 256k context (262,143 input tokens)** with unified routing
(`translate_long.ps1`) and automatic pure-NPU exact speculative decoding for inputs ≤ 2,048 tokens.

It is **not** a standalone Hugging Face Transformers model: it holds quantised weight images, embedding tables, a
precomputed tokenizer table, a speculative-decoding drafter (`PLaMo-D20-M2`), pre-generated NPU command streams, and a
CPython 3.13 extension (`pyxrt.pyd`).

Use this repository through the code and installers in the GitHub repository:
<https://github.com/nilpe/plamo3-translate-xdna1>

`install.ps1` downloads these files at a pinned revision and verifies every file against an SSH-signed `SHA256SUMS`
shipped in the GitHub repository. Running `python install_long.py --with-model` afterwards enables 256k Long mode
(Radeon iGPU prefill + NPU W4 decode) by verifying the long-mode files shipped in the GitHub repository and downloading
the official 1.78 GB BF16 `model.safetensors` directly from `pfnet/plamo-3-translate-0.8b`. Do not download files from
here manually unless you know what you are doing; start with the GitHub README.

The base model `pfnet/plamo-3-translate-0.8b` is **not** redistributed here (`install.ps1` fetches the 3.6 MB
tokenizer/template files and `install_long.py --with-model` fetches the 1.78 GB `model.safetensors`, both directly from
the upstream repository at revision `8b4d42855622093e610ba9b90c2d2b59f8d5c15d`).

## Performance snapshot

Measured on one machine (AMD Ryzen 7 8840U with XDNA1 NPU + Radeon 780M iGPU, 16 GB DDR5-5600, Windows 11 build 26300,
NPU driver 32.0.20101.3760, power mode "Default", turbo and keep-warm off):

**256k Long mode (`translate_long.ps1`, Radeon 780M iGPU prefill + XDNA1 NPU W4 decode):**

| Input length | Prefill (iGPU) | Decode (NPU) |
|---|---:|---:|
| 8k tokens | ~8 s | 21–22 ms/token |
| 32k tokens | ~22 s | 24 ms/token |
| 262k tokens (max 262,143) | ~5 min (4–5.5 min thermal spread) | ~50 ms/token |

**Short / standard context (≤ 2,048 tokens, automatic pure-NPU `w4-m2` exact speculative path, 24 interleaved fresh-process pairs):**

| Metric | Result |
|---|---|
| Process start -> speculative decoding ready ("spec-ready") | 0.69 s |
| Time to first token (27–31 / 515 input tokens) | 0.11 s / 1.43 s |
| Decode speed (`w4-m2` default, 515 in -> 595 out) | **13.9 ms/token** (~72 tokens/s) |
| Whole paragraph generation wall time (515 in -> 595 out) | 9.7 s |
| Everyday / Tatoeba / encyclopaedic `-Service` sets | 14.2 / 14.1 / 14.7 ms/token (1.63 / 1.56 / 1.47 tok/cycle) |

Full benchmark conditions and caveats are in the
[GitHub README](https://github.com/nilpe/plamo3-translate-xdna1#performance).

## Contents

Individual files, no archives.

| Path | What | Files | Approx. size |
|---|---|---:|---:|
| `plamo3-runtime/` | W4 and W8 weight images, embedding tables, base pre-generated command streams | 64 | 2.19 GB |
| `plamo3-verify-*/`, `plamo3-r*/` | Speculative-decoding verify, drafter, hoisted decode, prefill-head and compact-tokenizer payloads | 29 | 2.54 GB |
| `plamo3-drafter-port-v7/` | Speculative-decoding drafter package (`PLaMo-D20-M2`, derived model; see below) | 173 | 483 MB |
| `plamo3-default/` | Sealed manifests, compact catalogs, and release certificate | 10 | 100 MB |
| `plamo3-drafter-port-v7-attribution/` | Per-record attribution file (`PLaMo-ATTRIBUTION-D20.jsonl.gz`, no source text) | 2 | 41 MB |
| `plamo3-prefill-r46/` | Dual-pipeline prefill graphs | 11 | 7.6 MB |
| `plamo3-npu-cache/` | Compiled NPU kernel/image cache entries | 3 | 0.6 MB |
| `xrt/` | `xrt/python/pyxrt.pyd` (CPython 3.13 binding of AMD XRT 2.21.0, Apache-2.0, with pybind11 BSD-3-Clause) and `xrt/license/LICENSE` | 2 | 0.6 MB |
| **Required runtime payload subtotal** | Downloaded by default by `install.ps1` | **297** | **~5.35 GB** |
| `gate-data/` *(optional)* | Reference inputs for release-gate verification (`install.ps1 -WithGateData`; see `GATE_DATA_LICENSES.md` on GitHub) | 57 | 359 MB |
| **Repository total** | Required payload + optional `gate-data/` | **354** | **~5.71 GB** |

*(Note: `plamo3-verify-*/` covers `r37`, `r39`, `r41`, `r42`, `r44`; `plamo3-r*/` covers `r61`, `r67`, `r73`, `r74`,
`r77`, `r80`, `r83`. The 4.5 MB of 256k Long mode code/kernels ship directly in the GitHub repository.)*

The exact file list, byte sizes, and SHA-256 digests live in `PAYLOAD-MANIFEST.json` and `SHA256SUMS` in the GitHub
repository — intentionally not stored here, so a modification on Hugging Face cannot alter what the installer expects.

## Licence

All files here that derive from PLaMo 3 Translate are **DERIVED MODELs** under the
[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en) of Preferred Networks, Inc.
Copies: [LICENSE-PLaMo-en.md](LICENSE-PLaMo-en.md) and the Japanese original [LICENSE-PLaMo-ja.md](LICENSE-PLaMo-ja.md).

- Personal, academic, and other non-commercial use is free.
- **Commercial use** (including use inside a company, in products or services, hosting, or APIs) requires that **you
  register with PFN yourself** before you start, and is subject to the licence's revenue condition. Nothing here
  registers you or grants commercial rights.
- `xrt/python/pyxrt.pyd` is Apache-2.0 (AMD XRT) with pybind11 (BSD-3-Clause) compiled in; see `NOTICE` in the GitHub
  repository and `xrt/license/LICENSE`.
- The speculative-decoding drafter (`PLaMo-D20-M2`) was trained on PLaMo 3 Translate's own outputs for source sentences
  from permissively licensed sources only (full credits in `NOTICE` of the GitHub repository):
  - Tatoeba sentences, contributed by the Tatoeba community under
    [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) (<https://tatoeba.org>; CC0 where marked CC0 on
    Tatoeba). The contributor of every Tatoeba sentence used is recorded in
    `plamo3-drafter-port-v7-attribution/PLaMo-ATTRIBUTION-D20.jsonl.gz` (41 MB, one JSON line per training request, no
    source text).
  - OpenAssistant OASST1/OASST2 (Apache-2.0).
  - e-Gov Law Search statutes (Japanese Copyright Act Art. 13(i), Digital Agency, Government Standard Terms of Use 2.0).
  - Aozora Bunko and Project Gutenberg works (public domain).

  The drafter is distributed under the PLaMo Community License as a derived model.

## Intended use and limitations

- Japanese <-> English translation, supporting up to **262,143 input tokens** via Long mode (iGPU prefill + NPU decode)
  and up to 2,048 total tokens on the pure-NPU speculative path.
- **Base-model long-context limit:** translation quality above 2,048 tokens is unverified because the base model itself
  (including official BF16 on GPU) often repeats or stops translating on one-shot inputs over ~2k tokens. Long mode
  enables a repetition stop by default (see `LONG.md` on GitHub).
- Tested only on a Ryzen 7 8840U (XDNA1 NPU + Radeon 780M iGPU, NPU driver 32.0.20101.3760, Windows 11 build 26300);
  other XDNA1 PCs are untested and XDNA2 NPUs are not supported.
- The limitations and known risks of PLaMo 3 Translate (mistranslation, omissions, hallucinated content) apply; do not
  rely on the output without review where errors matter.

## Privacy

The runtime makes no network connections and sends no telemetry. This repository collects no statistics beyond what
Hugging Face itself provides.

## Provenance and exactness

All derived files were produced from `pfnet/plamo-3-translate-0.8b` at revision
`8b4d42855622093e610ba9b90c2d2b59f8d5c15d`. W4 output is certified bitwise identical to an independent order-matched
CPU reference on 100 standard and 64 long rows; speculative decoding adds no approximation (every emitted token equals
the serial W4 token at the same position), and the 256k Long mode decode path is bitwise verified up to position
262,143.

## Related repositories

- **Code & installers (GitHub)**: <https://github.com/nilpe/plamo3-translate-xdna1>
- **GPU MTP drafter (Hugging Face)**:
  [nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU)
- **GPU DSpark drafter (Hugging Face)**:
  [nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU)
