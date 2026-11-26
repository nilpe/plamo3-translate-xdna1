#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gate_index_long: indexes the long-mode gates (state/gates-long/<step>/summary.json) against the freeze_long epoch. Separate from the default's gate_index_rel (untouched).

record:  python tools/gate_index_long.py --root R --record STEP --result PASS --exit-code 0 [--evidence FILE]
index:   python tools/gate_index_long.py --root R --output FILE
A step counts only with a PASS record of the CURRENT epoch written after the freeze; the index PASSes only when every step of LONG_SUITE has one.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# L1-L12 of DESIGN.md plus the Phase 3 gates; steps marked * are required before anything ships (L4 included).
LONG_SUITE = (
    'freeze-long', 'unit-tests', 'unit-tests-release', 'attest-live', 'attest-mutants', 'install-long', 'install-long-negatives', 'router-default-delegation',
    'router-long', 'repeat-stop-prefix', 'l1-stencil', 'l2-npu-regression', 'l3-npu-steps', 'l4-compaction-bitwise', 'l5-igpu-vllm', 'l6-igpu-float64', 'l7-split-identity',
    'l8-hybrid-report', 'l9-router-equivalence', 'l10-failure-behaviour', 'l11-relocation', 'l12-isolation', 'quality-b1', 'quality-b2',
)


def record(root, step, result, exit_code, evidence=None):
    root = Path(root)
    fr = json.loads((root / 'state' / 'gates-long' / 'freeze.json').read_text())
    d = root / 'state' / 'gates-long' / step
    d.mkdir(parents=True, exist_ok=True)
    row = dict(step=step, result=result, exit_code=exit_code, epoch=fr['epoch'], time=time.time(), package='long')
    if evidence:
        row['evidence_sha256'] = hashlib.sha256(Path(evidence).read_bytes()).hexdigest()
    (d / 'summary.json').write_text(json.dumps(row, indent=1))
    return row


def build_index(root, suite=LONG_SUITE):
    root = Path(root)
    folder = root / 'state' / 'gates-long'
    fr = json.loads((folder / 'freeze.json').read_text())
    assert fr.get('result') == 'PASS' and fr.get('default_unchanged') is True, 'freeze_long is not PASS / the default release is not byte-identical to v1.0.0'
    rows, missing = [], []
    for step in suite:
        p = folder / step / 'summary.json'
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            missing.append(step); continue
        if r.get('epoch') != fr['epoch'] or r.get('result') != 'PASS' or r.get('exit_code') != 0 or p.stat().st_mtime < fr['started_ns'] / 1e9 - 1:
            missing.append(step); continue
        rows.append(dict(step=step, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert not missing, ('steps without a fresh PASS record of the current epoch', missing)
    return dict(result='PASS', epoch=fr['epoch'], steps=len(rows), gates=rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--output'); ap.add_argument('--record'); ap.add_argument('--result', default='PASS')
    ap.add_argument('--exit-code', type=int, default=0); ap.add_argument('--evidence')
    a = ap.parse_args(argv)
    if a.record:
        print(json.dumps(record(a.root, a.record, a.result, a.exit_code, a.evidence))); return 0
    try:
        rep = build_index(a.root)
    except AssertionError as e:
        rep = dict(result='FAIL', error=str(e)[:2000]); print('FAIL', str(e)[:600])
    if a.output:
        Path(a.output).write_text(json.dumps(rep, indent=1))
    else:
        print(json.dumps(rep))
    return 0 if rep['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
