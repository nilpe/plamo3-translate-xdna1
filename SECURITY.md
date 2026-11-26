# Security, trust model and privacy

Built with PLaMo. Unofficial; not affiliated with Preferred Networks, Inc. or Advanced Micro Devices, Inc.

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting** ("Security" tab of this repository -> "Report a vulnerability").
Do not post exploit details in a public issue. This is a spare-time project: expect a first answer within a couple of
weeks, not days. Supported: the latest release only.

Useful reports: a way to make the runtime execute unverified code or accept a modified file, a way to make the
installer write outside the install root, a download that is accepted without verification, a leak of personal
information in a shipped file or a log. Not in scope: vulnerabilities of the AMD NPU driver, Windows, Python, numpy or
the model itself (report those upstream).

## No telemetry

- The translator makes **no network connections**. A static scan of the runtime source at the time of writing found no
  network-capable imports in the runtime closure (the only file with such imports is a build-time model helper that is
  not part of the runtime). This is a statement about the code we ship, not a third-party audit; you can confirm it with
  a firewall rule that blocks `venv\Scripts\python.exe`.
- The installer contacts exactly: `huggingface.co` (the payload repository and the base model repository, at pinned
  revisions) and `pypi.org` (through pip, two wheels with pinned hashes). It sends no usage data. Requests carry the
  user agent `plamo3-npu-installer/<version>`.
- Nothing is uploaded, ever. Logs and records are local files: `state\logs\`, `state\env-record.json`,
  `state\selftest-report.json`, `state\install-*.json`. They contain no translated text (the self-test uses fixed
  sentences) and show your user profile directory as `%USERPROFILE%`. You decide whether to attach them to a bug report.
- No crash reporting, no update check, no accounts, no analytics.

## What is verified, and how

1. **`SHA256SUMS`** lists every file we ship. It is the trust root. It is published in the repository and the GitHub release, signed with a
   detached OpenSSH signature (`SHA256SUMS.sig`, namespace `file`) by the maintainer, and the git tag is an SSH-signed tag too.
   The signing key is pinned in `release.json` (`signing`): public key `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOuQcDXf30zL1peudH/0yVs74hnipYBI0isf/w6tGsEQ`,
   fingerprint `SHA256:cdzAIU0HiCtE3KIO2DOTvv66tknD8jUPANG20hSleNU`, principal `nilpe`. Cross-check the fingerprint at https://github.com/nilpe.keys and
   https://api.github.com/users/nilpe/ssh_signing_keys; the installer never contacts those
   hosts. It verifies the signature with `ssh-keygen -Y verify` (Git for Windows' copy is found automatically; the Windows built-in OpenSSH client hangs on files larger than about 8 KB and is not used for the 100 KB `SHA256SUMS`) when one is available
   (`-RequireSignature` makes it mandatory, `-Fingerprint` checks the pinned key against your own copy of the fingerprint).
2. **The payload** (5.35 GB on Hugging Face, plus 359 MB optional `gate-data/`) is downloaded at a pinned revision and hashed against `SHA256SUMS`
   before anything else is created. A compromised or changed Hugging Face repository therefore cannot change what you
   install: the expected hashes come from the signed git release.
3. **The base model** (`pfnet/plamo-3-translate-0.8b`, revision `8b4d42855622093e610ba9b90c2d2b59f8d5c15d`; required tokenizer and chat-template files, 3.6 MB) is
   downloaded from Hugging Face and verified against hashes shipped in this repository and against the Hugging Face file
   listing (LFS sha256, or git blob id for small files). We do not redistribute it.
4. **Python packages**: exactly numpy and ml_dtypes, pinned by version and SHA-256 (`pip --require-hashes
   --only-binary=:all: --no-deps --isolated`), then every installed file is re-hashed against the wheel's `RECORD`.
5. **At every start** the runtime re-checks its sealed inputs: the manifests' own digests, the exact set of source files,
   every payload and model file (by NTFS file identity from the install receipt, otherwise by content hash), and the
   drafter package. A mismatch refuses to run before any NPU work (exit code 3).

## What is recorded but not enforced

Components that belong to Windows, the NPU driver or your Python are **recorded and compared with a tested list**
(`known-good-env.json`), never enforced: the Windows build, `xrt_coreutil.dll` (a driver file in `C:\Windows\System32`),
the Python patch level and build, the venv launcher, `numpy`/`ml_dtypes` versions (they are pinned at install, not
re-hashed at every start). An unknown version prints one warning. Only ABI facts that would crash or miscompute are
strict: CPython 3.13, 64-bit, Windows, x64. We ship **no** Windows or driver files (no `DXCore.dll`, no
`xrt_coreutil.dll`); `pyxrt.pyd` is built by us and is hash-bound.

## Limits of this integrity model (read this)

- The digests are integrity checks, not a defence against an administrator or malware that can already write to the
  install folder: someone who can rewrite the sealed manifests can also rewrite the receipt. The model detects accidental
  damage, partial or tampered downloads, and tampering by anyone without write access to the whole tree. It is not a
  sandbox.
- The receipt (`state\install-receipt.json`) is a local cache of verified file identities. It is bound to the install
  root and to the certificate digest, and is written only by the installer after a full successful verification. A stale
  or foreign receipt is ignored (every file is then hashed; slow but equally strict).
- The installer and the runtime run as you, without elevation. The opt-in turbo mode calls the driver vendor's
  `xrt-smi.exe` from `System32\AMD` to change a system-wide power mode; it fails soft when you lack the rights.
- Execution policy: `install.ps1` and `uninstall.ps1` are plain scripts. They do not change your execution policy.

## Everything the installer changes

Inside the install root only: the payload files, `model\`, `venv\`, `state\`, optionally `wheels\`. It does not touch
the registry, PATH, scheduled tasks, services or any folder outside the root. The runtime may create
`%LOCALAPPDATA%\plamo3-npu` (lock and state) and `%LOCALAPPDATA%\plamo3-npu-turbo` (crash marker while turbo is on);
`uninstall.ps1` removes them.

## For maintainers: publishing

`tools/release_tool.py` prepares and verifies; it has no code that uploads, tags or signs. The maintainer signs
`SHA256SUMS` with their own key (`release_tool.py sign-cmd` prints the exact `ssh-keygen -Y sign` command) and publishes
by hand. `tools/scrub_check.py` must pass on the public repository (files and git metadata) and on the payload before
anything is published.
