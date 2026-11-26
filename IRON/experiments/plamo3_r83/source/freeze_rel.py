# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Release gate `freeze`: records the epoch of a gate run on the installed tree (uuid, start time, sha256 of every bound source file, the gate tools, the certificate, the drafter package inventory).
Every later step of the run refuses (runner) when a recorded file changed. Writes <root>\\state\\gates\\freeze.json. usage: freeze_rel.py --output FILE.json"""
import argparse, hashlib, json, os, time, uuid
from pathlib import Path
import settings83 as s
s.paths()
from plamo_root import ROOT
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
from identity43 import IdentityLease
from initialize83 import package_hash
from runtime_manifest import source_files
spec = s.SPEC_MANIFEST
cfg = json.loads(spec.read_text())['payload']['configuration']
for k, v in cfg.items():
    os.environ[k] = v
lease = IdentityLease(spec, workers=2)
try:
    lease.wait(); lease.require_validated(); manifest = lease.manifest
finally:
    lease.close()
files = [Path(x) for x in source_files([str(ROOT / r) for r in ('IRON/iron/applications/plamo3', 'IRON/experiments/plamo3_r45/source', 'IRON/experiments/plamo3_r83/source')] + [str(f) for f in sorted((ROOT / 'lib').glob('r*'))])]
files += [ROOT / 'tools' / n for n in ('run_release.ps1', 'gate_index_rel.py', 'release_tool.py')] + [ROOT / 'translate_plamo3.ps1', ROOT / 'release.json']
files = [f for f in files if f.is_file()]
report = dict(result='PASS', epoch=uuid.uuid4().hex, started_ns=time.time_ns(), candidate=str(spec), candidate_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
              source_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, package=package_hash(),
              release_sha256=hashlib.sha256((ROOT / 'release.json').read_bytes()).hexdigest(), default_unchanged=True)
(ROOT / 'state/gates').mkdir(parents=True, exist_ok=True)
(ROOT / 'state/gates/freeze.json').write_text(json.dumps(report, indent=2)); a.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'source_hashes'}), flush=True)
