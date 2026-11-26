#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adds long mode (inputs up to 262,144 tokens: iGPU prefill + NPU decode) to an existing, verified PLaMo 3 Translate v1.x installation.

  python install_long.py [--with-model] [--yes] [--verify] [--plan] [--root DIR] [--source DIR] [--model-source FILE]

* Requires an installation that passes `install.py --verify` (install.py itself is not changed by this tool).
* Installs/verifies long\\ and plamo3-long\\ (hash-checked against SHA256SUMS, then attested against long-manifest.json / long-certificate.json).
* --with-model: downloads the BF16 base-model weights (model.safetensors, 1,780,709,128 bytes) from the OFFICIAL upstream repository (pfnet/plamo-3-translate-0.8b,
  pinned commit, pinned sha256). Nothing is redistributed by this project. It prints the licence URL and the size first, checks free disk space, verifies the hash
  after the download, makes the file read-only and re-attests it against the long manifest. The default translation never reads this file.
Exit codes as install.py: 0 ok, 2 usage, 3 verification failed, 6 environment (disk space ...), 7 network, 8 licence not accepted.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_VERIFY, EXIT_ENV, EXIT_NETWORK, EXIT_LICENSE = 0, 1, 2, 3, 6, 7, 8
MODEL_REPO = 'pfnet/plamo-3-translate-0.8b'
MODEL_REVISION = '8b4d42855622093e610ba9b90c2d2b59f8d5c15d'
MODEL_FILE = 'model.safetensors'
MODEL_SIZE = 1780709128
MODEL_SHA256 = '7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5'
LICENSE_URL = 'https://huggingface.co/pfnet/plamo-3-translate-0.8b'
DISK_MARGIN = 512 * 1024 * 1024
LONG_PREFIXES = ('long/', 'plamo3-long/')
LONG_ROOT_FILES = ('long-manifest.json', 'long-certificate.json', 'translate_long.ps1', 'install_long.py')


class Fail(Exception):
    def __init__(self, code, message, hint=None):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def say(text, quiet=False):
    if not quiet:
        print(text, flush=True)


def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def fmt_gb(n):
    return '%.2f GB' % (n / 1e9)


def parse_sums(path):
    sums = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            h, p = line.split('  ', 1)
            sums[p.strip()] = h
    return sums


# ------------------------------------------------------------------ steps
def verify_base(root, run=subprocess.run):
    """The base v1.x installation must verify (install.py --verify, unchanged). Returns on success."""
    inst = Path(root) / 'install.py'
    if not inst.is_file():
        raise Fail(EXIT_VERIFY, 'no PLaMo 3 Translate installation found in %s (install.py is missing)' % root, 'Install v1.x first (see INSTALL.md), then run this tool.')
    py = Path(root) / 'venv' / 'Scripts' / 'python.exe'
    exe = str(py) if py.is_file() else sys.executable
    r = run([exe, str(inst), '--root', str(root), '--verify', '--quiet', '--no-npu-check'], cwd=str(root), capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail(EXIT_VERIFY, 'the existing installation does not verify (install.py --verify exited %s): %s' % (r.returncode, ' '.join((r.stdout + r.stderr).split())[-200:]),
                   'Repair it first: python install.py --verify, or reinstall v1.x. Long mode is only added to a verified installation.')


def place_long_files(root, source, sums, quiet=False):
    """Copy long/, plamo3-long/ and the two long manifests from `source` into root (skipped when source is root) and check every one against SHA256SUMS."""
    names = sorted(p for p in sums if p.startswith(LONG_PREFIXES) or p in LONG_ROOT_FILES)
    if not names:
        raise Fail(EXIT_VERIFY, 'SHA256SUMS lists no long-mode files: this is not a release that includes long mode', 'Update to v1.1.0 or newer (git pull), then run this tool again.')
    for rel in names:
        dst = Path(root) / rel
        if source and Path(source).resolve() != Path(root).resolve():
            src = Path(source) / rel
            if not src.is_file():
                raise Fail(EXIT_VERIFY, 'file missing in --source: %s' % rel)
            if sha256_file(src) != sums[rel]:               # check the SOURCE before anything is written into the installation
                raise Fail(EXIT_VERIFY, 'long-mode file differs from SHA256SUMS: %s (in --source; nothing was installed from it)' % rel, 'Download the release again.')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, str(dst) + '.part')
            os.replace(str(dst) + '.part', dst)
        if not dst.is_file():
            raise Fail(EXIT_VERIFY, 'long-mode file missing: %s' % rel, 'Update the repository (git pull) or pass --source.')
        if sha256_file(dst) != sums[rel]:
            raise Fail(EXIT_VERIFY, 'long-mode file differs from SHA256SUMS: %s' % rel, 'If git changed line endings, check .gitattributes (* -text). Otherwise download the release again.')
    say('    %d long-mode files verified against SHA256SUMS' % len(names), quiet)
    return names


def check_disk(path, need, free=None):
    free = shutil.disk_usage(path).free if free is None else free
    if free < need + DISK_MARGIN:
        raise Fail(EXIT_ENV, 'not enough free disk space: %s free, %s needed (model weights %s plus %s margin)' % (fmt_gb(free), fmt_gb(need + DISK_MARGIN), fmt_gb(need), fmt_gb(DISK_MARGIN)),
                   'Free some space on the drive that holds the installation and run again.')


def licence_notice(quiet=False):
    say('Base-model weights for long mode', quiet)
    say('  file:     %s  (%s bytes, %s)' % (MODEL_FILE, format(MODEL_SIZE, ','), fmt_gb(MODEL_SIZE)), quiet)
    say('  source:   official upstream repository %s at commit %s (not redistributed by this project)' % (MODEL_REPO, MODEL_REVISION[:8]), quiet)
    say('  licence:  PLaMo Community License; read it at %s (a copy is in LICENSE-PLaMo-en.md)' % LICENSE_URL, quiet)
    say('  integrity: sha256 %s is checked after the download' % MODEL_SHA256, quiet)


def accept(args):
    if args.yes or args.accept_license:
        return
    if not sys.stdin or not sys.stdin.isatty():
        raise Fail(EXIT_LICENSE, 'the model licence must be accepted: re-run with --accept-license (after reading %s)' % LICENSE_URL)
    if input('Download the weights under that licence? [y/N] ').strip().lower() not in ('y', 'yes'):
        raise Fail(EXIT_LICENSE, 'not accepted; nothing was downloaded')


def install_model(root, args, downloader=None, quiet=False):
    dest = Path(root) / 'model' / MODEL_FILE
    if dest.is_file() and dest.stat().st_size == MODEL_SIZE and sha256_file(dest) == MODEL_SHA256:
        say('    model weights already present and verified', quiet)
        return dest
    licence_notice(quiet)
    accept(args)
    (Path(root) / 'model').mkdir(exist_ok=True)
    part = Path(str(dest) + '.part')
    have = part.stat().st_size if part.exists() else 0
    check_disk(root, MODEL_SIZE - have)
    if dest.exists():
        os.chmod(dest, stat.S_IWRITE | stat.S_IREAD)
        dest.unlink()
    if args.model_source:
        src = Path(args.model_source)
        if not src.is_file():
            raise Fail(EXIT_USAGE, '--model-source is not a file: %s' % src)
        shutil.copyfile(src, part)
        os.replace(part, dest)
        say('    copied weights from %s' % src, quiet)
    else:
        say('    downloading (resumes if interrupted) ...', quiet)
        url = 'https://huggingface.co/%s/resolve/%s/%s' % (MODEL_REPO, MODEL_REVISION, MODEL_FILE)
        if downloader is None:
            sys.path.insert(0, str(root))
            import install as base_install
            class _L:
                def info(self, t): say(t, quiet)
            downloader = base_install.Downloader(_L())
        try:
            downloader.fetch(url, str(dest), MODEL_SIZE)
        except Fail:
            raise
        except Exception as error:
            code = getattr(error, 'code', EXIT_NETWORK)
            raise Fail(EXIT_NETWORK if not isinstance(code, int) or code > 100 else code, getattr(error, 'message', str(error)), getattr(error, 'hint', None))
    if dest.stat().st_size != MODEL_SIZE or sha256_file(dest) != MODEL_SHA256:
        try:
            dest.unlink()
        except OSError:
            pass
        raise Fail(EXIT_VERIFY, 'the downloaded weights do not match the pinned size/sha256; the file was deleted', 'Run the command again; if it repeats, report it (do not use the file).')
    os.chmod(dest, stat.S_IREAD)
    say('    weights verified (sha256 %s...) and set read-only' % MODEL_SHA256[:12], quiet)
    return dest


def run(args, *, verify_base_fn=verify_base, downloader=None):
    root = Path(args.root) if args.root else HERE
    quiet = args.quiet
    if not args.skip_base_verify:
        say('[1] verifying the existing installation (install.py --verify)', quiet)
        verify_base_fn(root)
    # The long-mode file hashes come from the SHA256SUMS of the release that carries long mode: the install root's own (after `git pull` of v1.1.0) or, with --source, the source's.
    same = not args.source or Path(args.source).resolve() == root.resolve()
    sums_path = (root if same else Path(args.source)) / 'SHA256SUMS'
    if not sums_path.is_file():
        raise Fail(EXIT_VERIFY, 'SHA256SUMS is missing in %s' % sums_path.parent)
    sums = parse_sums(sums_path)
    say('[2] long-mode files', quiet)
    if args.plan:
        names = sorted(p for p in sums if p.startswith(LONG_PREFIXES) or p in LONG_ROOT_FILES)
        say('    plan: %d long-mode files listed in SHA256SUMS; %s' % (len(names), 'would download %s of weights' % fmt_gb(MODEL_SIZE) if args.with_model else 'no weights download (use --with-model)'), quiet)
        return EXIT_OK
    if not args.verify:
        place_long_files(root, args.source, sums, quiet)
    else:
        place_long_files(root, None, sums, quiet)
    sys.path.insert(0, str(root / 'long'))
    import long_attest
    say('[3] attestation of the long files', quiet)
    ret, manifest = long_attest.attest(root, retain=False, check_weights=False)
    say('    %d files attested (long manifest sha256 %s...)' % (ret.stats['files'], json.loads((root / 'long-certificate.json').read_text())['manifest_sha256'][:12]), quiet)
    if args.with_model and not args.verify:
        say('[4] base-model weights', quiet)
        install_model(root, args, downloader, quiet)
    w = root / 'model' / MODEL_FILE
    if w.is_file():
        say('[5] attesting the weights against the long manifest', quiet)
        long_attest.attest(root, retain=False, check_weights=True)
        say('    ok', quiet)
        state = {'installed': time.strftime('%Y-%m-%dT%H:%M:%S'), 'weights': True, 'manifest_sha256': json.loads((root / 'long-certificate.json').read_text())['manifest_sha256']}
    else:
        say('    note: the model weights are not installed; long inputs will ask for them. Run: python install_long.py --with-model', quiet)
        state = {'installed': time.strftime('%Y-%m-%dT%H:%M:%S'), 'weights': False}
    (root / 'state').mkdir(exist_ok=True)
    (root / 'state' / 'long-install.json').write_text(json.dumps(state, indent=1))
    say('Long mode is installed. Try:  .\\translate_long.ps1 -InputFile long.txt -SourceLang ja -TargetLang en', quiet)
    return EXIT_OK


def build_parser():
    p = argparse.ArgumentParser(prog='install_long.py', description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root')
    p.add_argument('--source', help='directory holding the long-mode files (default: the install root itself, e.g. after git pull)')
    p.add_argument('--with-model', action='store_true', help='download the 1.78 GB BF16 base-model weights from the official upstream repository')
    p.add_argument('--model-source', help='copy the weights from an existing file instead of downloading (hash-checked)')
    p.add_argument('--accept-license', action='store_true')
    p.add_argument('--yes', action='store_true')
    p.add_argument('--verify', action='store_true', help='verify an existing long-mode installation, no changes')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--skip-base-verify', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--quiet', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Fail as e:
        print('install_long: error: %s' % e.message, file=sys.stderr)
        if e.hint:
            print('  hint: %s' % e.hint, file=sys.stderr)
        return e.code
    except Exception as e:
        cls = getattr(e, 'code', None)
        msg = getattr(e, 'message', None) or '%s: %s' % (type(e).__name__, ' '.join(str(e).split())[:200])
        print('install_long: error: %s' % msg, file=sys.stderr)
        return cls if isinstance(cls, int) and cls in (3, 5, 6) else EXIT_INTERNAL


if __name__ == '__main__':
    sys.exit(main())
