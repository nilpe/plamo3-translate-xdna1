# SPDX-License-Identifier: Apache-2.0
"""Attestation of the long-mode files, done only when long mode starts (the default path never reads any of this).

Chain: SHA256SUMS (signed by the maintainer, verified by install_long.py) lists long-certificate.json and long-manifest.json -> the certificate pins the sha256 of the
manifest and of the default release's certificate -> the manifest pins size and sha256 of every long file (code, GPU code objects, decode image, decode package), of the
shared default-release files long mode reads (r34-w4 weights etc.) and of the base-model weights file (hash-pinned, user-downloaded). Every file is read through a
deny-write/deny-delete handle that stays open for the whole run, so what is hashed is what the iGPU child and the NPU step read. The long source directory must contain
exactly the listed files (inventory equality, __pycache__ ignored).
"""
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from long_core import errors, install_root

CERT_NAME = 'long-certificate.json'
MANIFEST_NAME = 'long-manifest.json'
MANIFEST_VERSION = 1
SOURCE_ROOTS = ('long',)


class AttestationError(errors.InstallProblem):
    def __init__(self, reason):
        super().__init__('the long-mode installation is incomplete or modified (%s). Run install_long.py --verify (or reinstall it).' % reason)
        self.reason = reason


def _opener():
    try:
        from runtime_manifest import _open_retained            # sealed helper: CreateFileW with FILE_SHARE_READ only
        return _open_retained
    except Exception:
        return lambda path: Path(path).open('rb')


def sha256_stream(f, size, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    n = 0
    while True:
        b = f.read(block)
        if not b:
            break
        h.update(b)
        n += len(b)
    return h.hexdigest(), n


def source_inventory(root, roots=SOURCE_ROOTS):
    out = set()
    for r in roots:
        base = Path(root) / r
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for fn in filenames:
                out.add(Path(dirpath, fn).relative_to(root).as_posix())
    return out


class Retained:
    """Open deny-write handles kept for the life of the run."""

    def __init__(self):
        self.handles = []
        self.stats = dict(files=0, bytes=0)

    def close(self):
        for h in self.handles:
            try:
                h.close()
            except Exception:
                pass
        self.handles = []


def load(root):
    root = Path(root)
    try:
        cert_bytes = (root / CERT_NAME).read_bytes()
        cert = json.loads(cert_bytes)
        man_bytes = (root / MANIFEST_NAME).read_bytes()
    except (OSError, ValueError):
        raise AttestationError('certificate or manifest missing or unreadable')
    if cert.get('version') != MANIFEST_VERSION or hashlib.sha256(man_bytes).hexdigest() != cert.get('manifest_sha256') or len(man_bytes) != cert.get('manifest_bytes'):
        raise AttestationError('manifest does not match its certificate')
    try:
        manifest = json.loads(man_bytes)
    except ValueError:
        raise AttestationError('manifest unreadable')
    if manifest.get('version') != MANIFEST_VERSION:
        raise AttestationError('unsupported manifest version')
    return cert, manifest


def _safe_rel(rel):
    p = rel.replace('\\', '/')
    if p.startswith('/') or ':' in p or '..' in p.split('/'):
        raise AttestationError('unsafe path in the manifest')
    return p


def attest(root=None, *, workers=2, retain=True, check_weights=True, progress=None):
    """Verifies everything; returns (Retained, manifest). Raises AttestationError (exit 3) with a one-line reason; a missing base-model weights file is a LongModeError raised by the caller."""
    root = Path(root) if root else install_root()
    cert, manifest = load(root)
    base = cert.get('base') or {}
    try:
        base_hash = hashlib.sha256((root / base['path']).read_bytes()).hexdigest()
    except (OSError, KeyError):
        raise AttestationError('the default release certificate is missing')
    if base_hash != base.get('sha256'):
        raise AttestationError('the long files were built for a different default release')
    listed = {}
    for item in manifest['files']:
        listed[_safe_rel(item['path'])] = item
    inv = source_inventory(root)
    extra = sorted(p for p in inv if p not in listed)
    if extra:
        raise AttestationError('unlisted file in the long source directory: %s' % extra[0])
    todo = [(item, 'long') for item in manifest['files']] + [(item, 'shared') for item in manifest.get('shared', [])]
    w = manifest.get('weights')
    if w and check_weights:
        todo.append((w, 'weights'))
    opener = _opener()
    ret = Retained()
    lock = threading.Lock()
    errs = []

    def one(item, kind):
        rel = _safe_rel(item['path'])
        path = root / rel
        try:
            f = opener(path)
        except OSError:
            with lock:
                errs.append('missing %s' % rel if kind != 'weights' else 'weights')
            return
        try:
            digest, n = sha256_stream(f, item['size'])
        except OSError:
            f.close()
            with lock:
                errs.append('unreadable %s' % rel)
            return
        if n != item['size']:
            f.close()
            with lock:
                errs.append('size of %s' % rel)
            return
        if digest != item['sha256']:
            f.close()
            with lock:
                errs.append('content of %s' % rel)
            return
        with lock:
            if retain:
                ret.handles.append(f)
            else:
                f.close()
            ret.stats['files'] += 1
            ret.stats['bytes'] += n
    queue = list(todo)
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not queue or errs:
                    return
                item, kind = queue.pop()
            one(item, kind)
    threads = [threading.Thread(target=worker) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        ret.close()
        if errs[0] == 'weights':
            from long_core import LongModeError
            raise LongModeError('long mode needs the base model weights (model.safetensors, 1.78 GB), which are missing. Run install_long.py --with-model to download them (the normal translation does not need them).')
        raise AttestationError(errs[0])
    return ret, manifest
