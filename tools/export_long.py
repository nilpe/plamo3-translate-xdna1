#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-mode exporter: builds plamo3-long/decode/pd-manifest.json against the files of an INSTALLED default release (no hand-edited sizes), long-manifest.json and
long-certificate.json. Runs on the author's machine (Windows or Linux); the installed release only has to hold the shared files (r34-w4, the npu cache xclbin) and its SHA256SUMS.

  python tools/export_long.py --root <installed v1.x root> --repo <repo/stage dir holding long/ and plamo3-long/> --template <pd-manifest.template.json> --out <dir>

Every shared file is checked against the installed SHA256SUMS (the signed trust root) before it is pinned.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CHUNK = 512 * 1024 * 1024
WEIGHTS = dict(path='model/model.safetensors', size=1780709128, sha256='WEIGHTS_SHA256_PLACEHOLDER',
               upstream=dict(repo='pfnet/plamo-3-translate-0.8b', revision='8b4d42855622093e610ba9b90c2d2b59f8d5c15d', file='model.safetensors',
                             license_url='https://huggingface.co/pfnet/plamo-3-translate-0.8b'))
DROP_KEYS = ('sources', 'origins', 'source_roots', 'certification', 'references', 'audit', 'configuration', 'python_abi', 'tokenizer_manifest')


def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def chunk_hashes(path):
    out = []
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            out.append(hashlib.sha256(b).hexdigest())
    return out


def parse_sums(path):
    sums = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            h, p = line.split('  ', 1)
            sums[p.strip()] = h
    return sums


def rebase_rel(win_path):
    """artifact path of the PoC manifest -> '@ROOT@\\rel'. Only the three known prefixes exist."""
    for prefix, new in (('C:\\Users\\ykogi\\npu\\plamo3-runtime\\', '@ROOT@\\plamo3-runtime\\'), ('C:\\Users\\ykogi\\.npu\\cache\\ce3a15a0bde5955aa9ec901d\\', '@ROOT@\\plamo3-npu-cache\\ce3a15a0bde5955aa9ec901d\\')):
        if win_path.startswith(prefix):
            return new + win_path[len(prefix):]
    return None


def scrub_check(obj):
    text = json.dumps(obj)
    for needle in ('ykogi', 'C:\\\\Users', 'nilpe', '192.168.', 'codex_tasks'):
        if needle in text:
            raise SystemExit('scrub: %r still present in the exported manifest' % needle)


def build(root, repo, template, out, weights_sha256, release='plamo3-translate-xdna1-v1.1.0-long'):
    root, repo, out = Path(root), Path(repo), Path(out)
    sums = parse_sums(root / 'SHA256SUMS')
    pd = json.loads(Path(template).read_text())
    paths_in = json.loads((repo / 'plamo3-long' / 'decode' / 'artifact-paths.json').read_text())
    shared = {}
    for name, item in pd['artifacts'].items():
        rel = paths_in[name]
        assert rel.startswith('@ROOT@'), (name, rel)
        relpath = rel[len('@ROOT@'):].lstrip('\\/').replace('\\', '/')
        if name == 'context0':                      # the long decode image itself: pinned as a long file, not a shared one
            f = repo / relpath
        else:
            f = root / relpath
            want = sums.get(relpath)
            if want is None:
                raise SystemExit('shared file %s is not in the installed SHA256SUMS (refusing to pin an unsigned file)' % relpath)
            digest = sha256_file(f)
            if digest != want:
                raise SystemExit('shared file %s differs from SHA256SUMS' % relpath)
            shared[relpath] = dict(path=relpath, size=f.stat().st_size, sha256=digest)
        item['nbytes'] = f.stat().st_size
        item['chunk_bytes'] = CHUNK
        item['chunks_sha256'] = chunk_hashes(f)
        item['path'] = rel
        item.pop('ntfs_identity', None)
    for k in DROP_KEYS:
        pd.pop(k, None)
    pd['filesystem_identity_version'] = 1
    # the generator also reads the r34-w4 transactions.bin and the stencil comes from this manifest's instruction table
    t = 'plamo3-runtime/r34-w4/transactions.bin'
    f = root / t
    if sums.get(t) != sha256_file(f):
        raise SystemExit('transactions.bin differs from SHA256SUMS')
    shared[t] = dict(path=t, size=f.stat().st_size, sha256=sums[t])
    scrub_check(pd)
    pdpath = repo / 'plamo3-long' / 'decode' / 'pd-manifest.json'
    pdpath.write_text(json.dumps(pd, indent=1, sort_keys=True), encoding='utf-8')
    files = []
    for base in ('long', 'plamo3-long'):
        for dirpath, dirnames, filenames in os.walk(repo / base):
            dirnames[:] = sorted(d for d in dirnames if d not in ('__pycache__', 'tests'))
            for fn in sorted(filenames):
                p = Path(dirpath, fn)
                rel = p.relative_to(repo).as_posix()
                files.append(dict(path=rel, size=p.stat().st_size, sha256=sha256_file(p)))
    files.sort(key=lambda x: x['path'])
    manifest = dict(version=1, release=release, files=files, shared=sorted(shared.values(), key=lambda x: x['path']),
                    weights=dict(WEIGHTS, sha256=weights_sha256), source_roots=['long'])
    mbytes = json.dumps(manifest, indent=1, sort_keys=True).encode('utf-8')
    cert = dict(version=1, release=release, manifest_sha256=hashlib.sha256(mbytes).hexdigest(), manifest_bytes=len(mbytes),
                base=dict(release='plamo3-translate-xdna1-v1', path='plamo3-default/r45-spec/certificate.json', sha256=sums['plamo3-default/r45-spec/certificate.json']))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'long-manifest.json').write_bytes(mbytes)
    (out / 'long-certificate.json').write_text(json.dumps(cert, indent=1, sort_keys=True), encoding='utf-8')
    return manifest, cert


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='installed default release (v1.x) root')
    ap.add_argument('--repo', required=True, help='tree holding long/ and plamo3-long/ (the v1.1 additions)')
    ap.add_argument('--template', required=True, help='author-side pd-manifest template (PoC manifest with pd_layout)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights-sha256', required=True)
    a = ap.parse_args(argv)
    m, c = build(a.root, a.repo, a.template, a.out, a.weights_sha256)
    print(json.dumps(dict(files=len(m['files']), shared=len(m['shared']), manifest_sha256=c['manifest_sha256'], base=c['base']['sha256'][:16])))


if __name__ == '__main__':
    main()
