# Installing

Built with PLaMo. Unofficial; not affiliated with Preferred Networks, Inc. or Advanced Micro Devices, Inc. Read the
[licence summary in the README](README.md#licence-read-this-before-installing) first: the model files are under the
PLaMo Community License, and **commercial users must register with PFN themselves**.

## 1. What you need

| Item | Details |
|---|---|
| PC | Windows 11 x64 with an AMD Ryzen XDNA1 NPU. **Tested only on a Ryzen 7 8840U** (driver 32.0.20101.3760, Windows 11 build 26300). Other XDNA1 PCs may work and are untested. XDNA2 is not supported. |
| NPU driver | The AMD NPU driver (part of AMD's Ryzen AI Software; see the installation guide at ryzenai.docs.amd.com). Device Manager must list "NPU Compute Accelerator Device" without a warning icon. The driver supplies `xrt_coreutil.dll` in `C:\Windows\System32`. We never ship or replace driver or Windows files. |
| RAM / disk | 16 GB RAM (about 6 GB free while translating; ~4 GB free for Long mode). About **9 GB** free on an **NTFS** volume for the full 256k setup (payload 5.35 GB, base tokenizer 3.6 MB, Long mode BF16 weights 1.78 GB, venv 0.2 GB, plus margin; about **7 GB** for NPU-only without Long mode, add ~0.5 GB with `-WithGateData`). |
| Python | **64-bit CPython 3.13**. Details below. |
| Git | To clone the repository. Use `-c core.autocrlf=false` (see below). |
| Internet | One-time: huggingface.co (payload and base model) and pypi.org (two wheels). Nothing else, no telemetry. |
| `ssh-keygen` (optional) | Git for Windows' `ssh-keygen` (`<Git>\usr\bin\ssh-keygen.exe`, found automatically) verifies the signature of `SHA256SUMS`. The built-in Windows OpenSSH client (`C:\Windows\System32\OpenSSH\ssh-keygen.exe`) hangs on signed files larger than about 8 KB, so it is not used for `SHA256SUMS` (about 100 KB); without Git the signature is reported as "cannot be checked". |
| Rights | No administrator rights are needed. Only the opt-in turbo mode may need them (it fails soft). |

### Why Python 3.13, and how to get it

The NPU module `pyxrt.pyd` that we ship is a CPython 3.13 extension for 64-bit Windows, so the interpreter has to match.
The installer only needs *an* interpreter to create the venv; the runtime then uses `venv\Scripts\python.exe`. We do
not install Python for you.

Pick one (none needs administrator rights):

1. **python.org**: download "Windows installer (64-bit)" for Python 3.13.x from
   <https://www.python.org/downloads/windows/>, choose "Install for me only". The tested reference interpreter is
   3.13.13 (conda-forge build); other 3.13 patch levels and builds work but produce a one-line "untested" note.
2. `winget install Python.Python.3.13`
3. conda-forge: `conda create -n plamo313 python=3.13`, then run the installer from that environment.

Check: `py -3.13 -c "import sys; print(sys.version, sys.maxsize > 2**32)"` must print 3.13.x and `True`.
`install.ps1` looks for such an interpreter itself (the `py` launcher, including uv/Astral builds it lists, then
`python` on PATH); pass `-Python C:\path\to\python.exe` to choose one. The Microsoft Store "python.exe" alias stubs are
ignored.

## 2. Install

```powershell
git clone -c core.autocrlf=false https://github.com/nilpe/plamo3-translate-xdna1.git C:\PLaMo3-NPU
cd C:\PLaMo3-NPU
.\install.ps1                        # installs & verifies the NPU runtime (5.4 GB) and runs the self-test
python install_long.py --with-model  # enables 256k Long mode (downloads official 1.78 GB BF16 weights from HF; optional if you only need <= 2,048 tokens)
```

- **The clone is the install root.** The payload, the venv and the base model are placed next to the code, inside that
  folder. Choose a short local path (under 150 characters), not a network share and not a OneDrive-synced folder.
  Spaces in the path are fine. **The path must be ASCII-only** (no accented letters, no Japanese folder names): the NPU library (XRT, inside `pyxrt.pyd`) opens files through the Windows ANSI API, so every start fails below a folder with non-ASCII characters (we measured `pyxrt.xclbin()` failing with an undecodable error from such a path while an ASCII path and a path with spaces work). The installer refuses such a folder with `Choose a local, short, ASCII-only path ... e.g. C:\PLaMo3-NPU`, and the runtime refuses it with exit code 3: `the installation folder contains non-ASCII characters; the NPU library (XRT) cannot open files there. Reinstall into an ASCII-only folder, e.g. C:\PLaMo3-NPU.`
- **Line endings matter.** Every file is hash-checked. If `git` converts line endings (`core.autocrlf=true`, the Git for
  Windows default in some setups) the hashes no longer match. The repository sets `* -text`, and the installer detects
  the case and tells you to re-clone with `git config core.autocrlf false`.
- If PowerShell refuses to run the script ("running scripts is disabled"), start it with
  `powershell -ExecutionPolicy Bypass -File .\install.ps1`, or `Unblock-File .\*.ps1` for a downloaded zip.
  A normal `git clone` is not affected by the default `RemoteSigned` policy.

### What `install.ps1` does (every run resumes; verified files are not downloaded again)

| Step | What happens | Fails with |
|---|---|---|
| 1 Preflight | Windows 11 build, Python 3.13 64-bit, NPU driver and service present, install path rules, free disk for what is still missing, RAM advice, NTFS check. | 5, 6 |
| 2 Licence | Shows a summary and the paths of the full licence texts. Type `I ACCEPT` (or pass `-AcceptLicense`). Recorded in `state\license-accepted.json` with the SHA-256 of the licence text. | 8 |
| 3 Signature | If `SHA256SUMS.sig` is present and `ssh-keygen` is available: verifies it (see below). | 3 |
| 4 Payload | Downloads the files listed in `PAYLOAD-MANIFEST.json` from the pinned Hugging Face repository revision (4 parallel downloads, HTTP range resume, `.part` files, 3 retries). | 7 |
| 5 Verify | Hashes every file against `SHA256SUMS`. Any mismatch aborts before anything else is created. | 3 |
| 6 Python environment | Creates `venv\` and installs exactly two wheels, `numpy 2.4.6` and `ml_dtypes 0.6.0`, with `pip --require-hashes --only-binary=:all: --no-deps --isolated`, then re-hashes the installed files against the wheels' RECORD. | 7, 3 |
| 7 Base model | Downloads the required tokenizer/config files of `pfnet/plamo-3-translate-0.8b` (5 files, 3.6 MB) at revision `8b4d42855622093e610ba9b90c2d2b59f8d5c15d` into `model\`, verifies each file against the shipped hashes **and** the Hugging Face listing (LFS sha256 / git blob id). | 7, 3 |
| 8 Attestation | One more read pass over everything under a write-deny handle: SHA-256 again plus the NTFS file identity, then writes `state\install-receipt.json`. The runtime uses the receipt to skip re-hashing about 5.4 GB at every start. The receipt is written only here, never by the runtime. | 3 |
| 9 Environment record | Writes `state\env-record.json` (Windows build, driver and `xrt_coreutil.dll` versions, hashes, Python and package versions) and warns when something is not in `known-good-env.json`. Nothing is enforced; it is a record. | |
| 10 Self-test | About 2 minutes, uses the NPU: four fixed translations that must equal the shipped tokens, speculative-vs-serial identity on five prompts through `--service`, W8 serial, and a scan for fallback messages. | 10+N |

The installer touches no registry, PATH, scheduled task or service, and writes only inside the install root.

### Options

| `install.ps1` | `install.py` | Meaning |
|---|---|---|
| `-Root DIR` | `--root` | Install root (default: the folder of the script) |
| `-Python PATH` | | Interpreter to run the installer and create the venv |
| `-AcceptLicense` | `--accept-license` | Accept without typing (you still must have read it) |
| `-Plan` | `--plan` | Run only the read-only checks and print what would happen |
| `-Verify` | `--verify` | Re-verify an installation; no download, no self-test |
| `-Reattest` | `--reattest` | Same, and rewrite the receipt (after moving or restoring the folder) |
| `-SelfTestOnly` | `--selftest-only` | Run only the self-test |
| `-SkipSelfTest` | `--skip-selftest` | Skip the self-test (not recommended) |
| `-PayloadSource DIR` | `--payload-source` | Take payload files from a local directory instead of Hugging Face |
| `-ModelSource DIR` | `--model-source` | Copy the base model from an existing snapshot (never hard-linked, never read from your HF cache implicitly) |
| `-NoModel` | `--no-model` | Do not download the base model (verification only) |
| `-WithGateData` | `--with-gate-data` | Also fetch the optional certification reference data (about 359 MB) |
| `-RequireSignature`, `-Fingerprint F`, `-SumsSig FILE` | | Signature options |
| `-NoNpuCheck` | `--no-npu-check` | Install on a PC without an NPU (for preparing files) |
| `-Quiet`, `-LogPath FILE` | `--quiet`, `--log` | Output control. The log defaults to `state\logs\install-<time>.log`. |

### Exit codes of the installer

| Code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Internal error (re-run with `PLAMO3_DEBUG=1` for details and report it) |
| 2 | Usage error |
| 3 | The installation does not verify (hash mismatch, missing file, bad signature, line-ending damage) |
| 5 | No NPU or NPU driver found |
| 6 | Environment problem (Windows, Python, disk, path) |
| 7 | Network problem (download, pip) |
| 8 | Licence not accepted |
| 10 + N | Self-test step N failed (1 preconditions, 2 translations, 3 identity, 4 W8, 5 log scan, 6 turbo); the installation is in place and `state\selftest-report.json` says what failed |
| 130 | Interrupted (run again; it resumes) |

## 3. Check the signature of `SHA256SUMS` (optional but recommended)

`SHA256SUMS` is the trust root for everything we ship. The maintainer signs it with an SSH key (OpenSSH signature,
namespace `file`, principal `nilpe`); the signature `SHA256SUMS.sig` sits next to `SHA256SUMS` in the repository and is also a
release asset. The key is pinned in `release.json` (`signing`) and nothing is downloaded to check it:

* public key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ`
* fingerprint: `SHA256:cdzAIU0HiCtE3KIO2DOTvv66tknD8jUPANG20hSleNU`

Cross-check the fingerprint yourself, on a page the repository cannot change: the key is listed at
https://github.com/nilpe.keys and https://api.github.com/users/nilpe/ssh_signing_keys.

```powershell
.\install.ps1 -RequireSignature -Fingerprint SHA256:cdzAIU0HiCtE3KIO2DOTvv66tknD8jUPANG20hSleNU
```

Manual check with Git for Windows' `ssh-keygen` (the same check the installer makes). In Git Bash:

```bash
echo 'nilpe namespaces="file" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ' > allowed_signers
ssh-keygen -Y verify -f allowed_signers -I nilpe -n file -s SHA256SUMS.sig < SHA256SUMS
```

or in PowerShell, with the full path (`Start-Process` feeds the file to stdin):

```powershell
[IO.File]::WriteAllText("$PWD\allowed_signers", "nilpe namespaces=`"file`" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ`n")
Start-Process -FilePath "C:\Program Files\Git\usr\bin\ssh-keygen.exe" -ArgumentList "-Y","verify","-f","allowed_signers","-I","nilpe","-n","file","-s","SHA256SUMS.sig" -RedirectStandardInput SHA256SUMS -NoNewWindow -Wait
```

Do not use `C:\Windows\System32\OpenSSH\ssh-keygen.exe` for this: that client hangs when the signed file is larger than about 8 KB.

It must print `Good "file" signature for nilpe with ED25519 key SHA256:cdzAIU0HiCtE3KIO2DOTvv66tknD8jUPANG20hSleNU`. The signed git tag of the release is an SSH-signed tag
(`git -c gpg.ssh.allowedSignersFile=allowed_signers tag -v <tag>`).

Without `-RequireSignature` the installer verifies the signature when it can and otherwise prints a note. A **bad**
signature always aborts. `-Fingerprint` must equal the fingerprint of the key pinned in `release.json`; otherwise the
installer refuses.

## 4. Offline or manual installation

On a machine with internet access, clone the repository and fetch the payload with `huggingface-cli download` (or a
browser) into a folder that has the same relative layout; download the base model snapshot; then on the target
machine:

```powershell
.\install.ps1 -PayloadSource D:\plamo-payload -ModelSource D:\plamo-3-translate-0.8b-snapshot
```

The venv step still needs PyPI, or put the two wheels from the release page into `<root>\wheels\` (the hashes are
still checked).

## 5. After installing

- `.\translate_plamo3.ps1 "Hello." -SourceLang en -TargetLang ja`, see [USAGE.md](USAGE.md).
- `.\install.ps1 -Verify` at any time re-verifies everything.
- Do not move or copy the folder. If you do, run `.\install.ps1 -Reattest` (slow starts otherwise: every file is hashed).
- Updating: `git pull` (or clone the new release) and run `.\install.ps1` again: changed files are fetched, the rest is
  kept. Never edit, add or delete files under `IRON\`, `lib\` or the payload folders: the runtime refuses to start with
  a modified source inventory ("the installation is incomplete or modified"), by design.

## 6. Uninstall

```powershell
.\uninstall.ps1 -DryRun     # lists what would go
.\uninstall.ps1             # asks, then removes
.\uninstall.ps1 -KeepModel  # keeps the downloaded base model files in model\
```

It stops Python processes running from this installation's venv (after asking), restores a recorded turbo power mode,
and removes only what `state\install-manifest.json` says the installer created (payload files, `model\`, `venv\`,
`state\`) plus `%LOCALAPPDATA%\plamo3-npu` and `%LOCALAPPDATA%\plamo3-npu-turbo`. It never touches the code of the
clone (delete the folder yourself), the registry, PATH, or your driver. If you ever turned the NPU turbo mode on by
hand: `& "$env:SystemRoot\System32\AMD\xrt-smi.exe" configure --pmode default`.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no 64-bit CPython 3.13 was found` | See section 1; pass `-Python`. |
| Exit 5, "No AMD NPU driver was detected" | Install the AMD NPU driver, reboot. On a PC without an NPU use `-NoNpuCheck` only to stage files. |
| Exit 3, "files differ only by line endings" | Re-clone with `git clone -c core.autocrlf=false`. |
| Exit 3, "differs from SHA256SUMS" for a payload file | A damaged download. Delete that file and run the installer again; it fetches it anew. |
| Exit 7 | Network, proxy or a firewall. Run again: downloads resume. Set `HTTPS_PROXY` if you use a proxy. |
| Exit 6, "disk" | The installer counts only what is still missing; free that space or choose another `-Root`. |
| Warning "xrt_coreutil.dll ... not in the tested list" | Your driver differs from the tested one. It is a warning; if you see crashes or `device_open` errors, report with `state\env-record.json`. |
| Self-test fails at step 3 ("speculative decoding is not active") | Read `state\selftest-report.json`. A fallback means the drafter or its integrity check failed; run `.\install.ps1 -Verify`. |
| Self-test: `IRON_PLAMO_*` variables | Remove them from your environment; the runtime refuses them. |
| Slow first start after moving the folder | `.\install.ps1 -Reattest`. |

More: [USAGE.md](USAGE.md) (runtime errors and exit codes), [SECURITY.md](SECURITY.md) (trust model).
