#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""freeze_long: freezes the long-mode additions of a release (its own epoch, its own gate folder state/gates-long; the default release's epoch and gate index are untouched).

  python tools/freeze_long.py --root <install root> --v100-sums <SHA256SUMS of v1.0.0> [--output FILE]

Records: a random epoch, sha256 of every long-mode file (long/, plamo3-long/, long-manifest.json, long-certificate.json, translate_long.ps1, install_long.py, tools/*_long.py),
the long attestation result (full attest incl. weights when present) and `default_unchanged`: every entry of the v1.0.0 SHA256SUMS (translate_plamo3.ps1, install.py, every
sealed and freeze-bound file, the certificate ...) still hashes to its v1.0.0 value.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def parse_sums(path):
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            h, p = line.split('  ', 1)
            out[p.strip()] = h
    return out


# parallel to freeze_rel: the entry scripts, the exporter, the attestation module and the gate index are part of the frozen source hashes
REQUIRED_SOURCES = ('translate_long.ps1', 'install_long.py', 'tools/export_long.py', 'long/long_attest.py', 'tools/gate_index_long.py', 'tools/freeze_long.py')


def long_files(root):
    root = Path(root)
    names = ['long-manifest.json', 'long-certificate.json', 'translate_long.ps1', 'install_long.py']
    names += ['tools/%s' % f for f in ('export_long.py', 'freeze_long.py', 'gate_index_long.py') if (root / 'tools' / f).is_file()]
    for base in ('long', 'plamo3-long'):
        for dp, dn, fn in os.walk(root / base):
            dn[:] = [d for d in dn if d not in ('__pycache__', 'tests')]
            names += [Path(dp, f).relative_to(root).as_posix() for f in fn]
    return sorted(n for n in names if (root / n).is_file())


def default_unchanged(root, v100_sums):
    """(ok, differing paths): every v1.0.0 entry except SHA256SUMS/.sig themselves."""
    bad = []
    for rel, h in parse_sums(v100_sums).items():
        if rel in ('SHA256SUMS', 'SHA256SUMS.sig'):
            continue
        p = Path(root) / rel
        if not p.is_file() or sha256_file(p) != h:
            bad.append(rel)
    return not bad, bad


def freeze(root, v100_sums, attest=True):
    root = Path(root)
    report = dict(result='PASS', epoch=uuid.uuid4().hex, started_ns=time.time_ns(), release='long')
    report['source_hashes'] = {n: sha256_file(root / n) for n in long_files(root)}
    missing = [n for n in REQUIRED_SOURCES if n not in report['source_hashes']]
    report['required_sources_missing'] = missing
    ok, bad = default_unchanged(root, v100_sums)
    report['default_unchanged'] = ok
    report['default_differing'] = bad[:20]
    if attest:
        sys.path.insert(0, str(root / 'long'))
        import long_attest
        ret, manifest = long_attest.attest(root, retain=False, check_weights=(root / 'model' / 'model.safetensors').is_file())
        report['attested'] = dict(files=ret.stats['files'], bytes=ret.stats['bytes'], weights=(root / 'model' / 'model.safetensors').is_file())
    if not ok or missing:
        report['result'] = 'FAIL'
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--v100-sums', required=True); ap.add_argument('--output'); ap.add_argument('--no-attest', action='store_true')
    a = ap.parse_args(argv)
    rep = freeze(a.root, a.v100_sums, not a.no_attest)
    folder = Path(a.root) / 'state' / 'gates-long'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'freeze.json').write_text(json.dumps(rep, indent=1))
    sd = folder / 'freeze-long'
    sd.mkdir(exist_ok=True)
    (sd / 'summary.json').write_text(json.dumps(dict(step='freeze-long', result=rep['result'], exit_code=0 if rep['result'] == 'PASS' else 1, epoch=rep['epoch'], time=time.time(), package='long')))
    if a.output:
        Path(a.output).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'source_hashes'}))
    return 0 if rep['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
