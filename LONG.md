# Long mode (experimental): inputs longer than 2,048 tokens

**Experimental. Translation quality has not been verified above the normal range (2,048 tokens).** Long mode runs, up to 262,143 tokens, but the model itself, including the official BF16 model on a GPU, often repeats or stops translating on one-shot inputs above about 2k tokens. In our full-length tests on 16 law documents of 2.6k-6k tokens, the official BF16 model looped in 10 of 16 and long mode in 8 of 16. Split long documents into pieces of a few thousand tokens or less (paragraphs, articles, chapters) and translate the pieces.

## What it is
`translate_long.ps1` is the unified entry point for any input length:
- **Short / standard inputs (≤ 2,048 total tokens, no `-Long`):** handed unchanged to `translate_plamo3.ps1` (pure-NPU W4 + exact speculative decoding, 0.69 s startup, 13.9 ms/token).
- **Long inputs (> 2,048 tokens, up to 262,143 prompt tokens):** handled by **Long mode** (its own manifest `long-manifest.json` and certificate `long-certificate.json`): the Radeon iGPU reads the whole input (prefill) and the XDNA1 NPU writes the translation one token at a time (W4 serial, no speculation).

```powershell
.\translate_long.ps1 -InputFile report.txt -SourceLang ja -TargetLang en     # any length
.\translate_long.ps1 "Hello." -SourceLang en -TargetLang ja                   # small request: handed unchanged to translate_plamo3.ps1
.\translate_long.ps1 -InputFile report.txt -SourceLang ja -TargetLang en -Long  # force long mode
```
If the request fits the normal path (prompt + output cap <= 2,048 tokens, no `-Long`), `translate_long.ps1` runs `translate_plamo3.ps1` with the same arguments and returns its exit code. Otherwise it runs long mode. Texts over the 32 KB command-line limit go in with `-InputFile` (UTF-8) or the pipeline.

## Setup (256k Long mode)
Run after `.\install.ps1` in an **ASCII-only folder** (Long mode, like the pure-NPU path, is refused with exit 3 if the install path has non-ASCII characters) on a PC with a Radeon 780M / 760M / 740M iGPU and the AMD graphics driver (tested on a **780M only**).
```powershell
python install_long.py --with-model        # verifies the install, attests the long files, then downloads the 1.78 GB BF16 weights
```
`install_long.py` does not touch `install.py` or any file of the normal path. `--with-model` downloads **one file, 1.78 GB**: `model.safetensors` from the official upstream repository `pfnet/plamo-3-translate-0.8b` at a pinned commit (not from this project). It prints the licence URL (https://huggingface.co/pfnet/plamo-3-translate-0.8b, PLaMo Community License) and the size first, checks free disk space, verifies the pinned sha256 afterwards and makes the file read-only. Without the weights long inputs stop with a message that names this command. `--verify` re-checks everything, `--plan` shows what would happen.

## Speed (Radeon 780M, development laptop)
| input | prefill (iGPU) | decode (NPU) |
|---|---|---|
| 8k tokens | about 8 s | 21-22 ms/token |
| 32k tokens | about 22 s | 24 ms/token |
| 262k tokens (max 262,143) | about 5 minutes (4-5.5 with thermal spread) | about 50 ms/token |
Needs about 4 GB of free RAM. The prompt plus the translation must fit in 262,144 tokens. The default output limit is the smaller of the remaining space and max(1,024, 2 x input tokens); `-NumTokens N` sets it. The command prints an estimate and progress lines; Ctrl+C cancels. `--service` is not available in long mode. While the iGPU works the desktop can feel slow.

## Behaviour you will see
- **Warning every time:** each long-mode run prints one stderr line saying that quality is not verified at this length and that the model itself tends to repeat or stop. The run continues.
- **Repetition stop (on):** if the output starts to repeat (the same 20+ token passage three times in a row) decoding stops, the text ends at the end of the first occurrence of the passage (an exact prefix of what the model would have written), a line on stderr says why, and `-Json` reports `"stop": "repeat"`. `-NoRepeatStop` keeps decoding.
- On 40 pieces of 300-1,000 tokens, long mode and the normal path scored the same against the official human translations (+0.33 chrF, 95% CI -0.17..+0.93). The first token comes from the BF16 head of the iGPU, the rest from the NPU's W4 weights.

## Exit codes
Those of `translate_plamo3.ps1`: 0 ok, 2 usage/invalid request (including an input over 262,143 tokens), 3 installation problem (including a modified long file or a non-ASCII install folder), 5 NPU busy (a long run holds the NPU for its whole duration), 6 environment (no iGPU runtime, not enough memory or disk, weights missing), 130 Ctrl+C.

## How it is protected
`long-manifest.json` and `long-certificate.json` pin every long file, the shared files long mode reads from the normal installation, and the weights (sha256); they are checked only when long mode starts, through handles that deny writes for the whole run, and the long source directory must contain exactly the listed files. `SHA256SUMS` lists all of it. The normal path never reads any long-mode file, and its files are byte-identical to v1.0.0.

## Measured quality data (for the curious)
Official human translations: Japanese Law Translation (Ministry of Justice), 6 laws. A document is "finished" when it ends with the end-of-text token and has no repeat loop.
| input size | documents | finished: long mode / original BF16 model | mean chrF vs human: long / BF16 model |
|---|---|---|---|
| 300-1,000 tokens | 40 | 40 / 40 | 62.7 / 63.3 |
| about 2.6k | 6 | 5 / 3 | 53.1 / 46.9 |
| about 4k | 6 | 2 / 2 | 53.9 / 53.1 |
| about 6k | 4 | 1 / 1 | 34.0 / 39.5 |
No length above 2,048 tokens passed our pre-declared full-length quality criteria, which is why long mode is experimental.
