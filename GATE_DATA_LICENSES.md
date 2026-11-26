# Licence classification of the optional `gate-data/` package

Copyright (c) 2026 nilpe. The package (`install.ps1 -WithGateData`) holds inputs of the release gates. Rule: it may contain only (1) text written by this project, (2) permissively licensed text with attribution, and (3) PLaMo model outputs and numbers derived from the model, labelled as such. Nothing that contains or derives from CC BY-SA corpora (KFTT, WikiMatrix, Wikimedia, Dolly) is in it.

| Path (below `gate-data/`) | What it is | Class | Terms |
|---|---|---|---|
| `plamo3-w4-r32/order-reference/` (metadata.json, standard.npy, ...) | token ids of our 4 fixed standard prompts and 8 long prompts (sentences written by this project, repeated to length), the 100 teacher tokens and reference rows computed from PLaMo with an independent order-matched CPU model | own text + model outputs/derived numbers | Apache-2.0 for our text; numbers and outputs are derived models/outputs under the PLaMo Community License ("Built with PLaMo") |
| `plamo3-w4-r31/reference/` | reference rows for the same fixed prompts | model-derived numbers | PLaMo Community License |
| `long-translation.json` | the 595-token translation of our paragraph (our text, PLaMo output) | own text + PLaMo output | as above |
| `short66.txt`, `para63.txt` | two texts written by this project (a one-line sentence and a 63-word paragraph) | own text | Apache-2.0 |
| `r36/tap_fixture/` | features of one fixed prompt (ours), computed from PLaMo | model-derived numbers | PLaMo Community License |
| `r55/*-generate-rel-*/generate.json` (files of the 5 fixed prompts) | measurement logs of the 5 fixed prompts (translation-0..3, paragraph-515): tokens, timings, cycle profiles and the drafter's logit vectors, measured with the shipped drafter on this release | own prompts + PLaMo outputs + our measurements | as above |
| `plamo3-r74/tokenizer-gate/corpus.json` | 9,025 Japanese/English sentences from Tatoeba plus 160 multi-sentence texts and one long text built from them (frozen, sha256 pinned in `tokenizer83_gate.py`) | permissively licensed | Credit: "Sentences from Tatoeba (https://tatoeba.org), contributed by the Tatoeba community under CC BY 2.0 FR" (https://creativecommons.org/licenses/by/2.0/fr/; terms of use: https://tatoeba.org/eng/terms_of_use). Contributors of individual sentences are listed on tatoeba.org; for sentences that also served as drafter training inputs they are in `plamo3-drafter-port-v7-attribution/PLaMo-ATTRIBUTION-D20.jsonl.gz`. The corpus file is hash-pinned and carries no per-sentence contributor ids, so the credit is given to the Tatoeba community as a whole; replace the corpus by authored text if per-sentence credit is required. |
| `r39a/`... | no data (code lives under `lib/`) | - | - |

## Removed from the public package (private, author side only)
* `r55/*-calibrate-*/calibrate.json` (4 files): the confidence-gate calibration cases are token-level records of the drafter development subsets d1/d2 (128 cases with dataset ids), which derive from development corpora that include CC BY-SA text. They are not published. The confidence-gate runs author side (`tools/author/run_confidence_author.py`) with the private files plus the public decision data and the installed release code; only a sanitised record (hashes of the private inputs, result, certificate hash) ships as `export-audit/confidence-record.json` and is checked by the release step `relgate-n1c`.
* Any other r55 data and all drafter training/dev corpora: not shipped.

Checks: `tools/scrub_check.py` and the byte scan run over the whole package; the classification above is complete for every file the exporter places under `gate-data/` (`src/gate-prefixes.json` in the maintainer notes lists the sources).
