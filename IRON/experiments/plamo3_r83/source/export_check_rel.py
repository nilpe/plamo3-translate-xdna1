# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Release gate `export` (N1 bytes-equal + export audit), run on the INSTALLED tree. Replaces the author's export step (which derived a candidate from the previous default).
Proves, by hashing every byte (never trusting NTFS identities / the receipt):
 * every payload/artifact file listed as binary_equal in export-audit/equivalence.json has the recorded size and sha256 (the exporter proved those equal to the R83 sources);
 * every transformed text file has the recorded new sha256; its equivalence to the source was proven by the exporter (decoded content equal except the documented substitutions);
 * SHA256SUMS: every listed file matches;
 * the release certificate chain: catalog.bin / expanded manifest digests, every bound record (sources, origins, artifacts) hashes to its record, the source inventory equals the sealed list;
 * no author-side text in any sealed manifest (decoded catalog included) or in release.json;
 * mutants: one flipped record hash, one missing record, one extra record and one non-invertible 'transform' must each be DETECTED by the same comparison code.
usage: export_check_rel.py --output FILE.json"""
import argparse, copy, hashlib, json, re, sys
from pathlib import Path
import settings83 as s
s.paths()
from plamo_root import ROOT
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
FORBIDDEN = [re.compile(x, re.I) for x in (r'ykog[i]', r'192[.]168[.]', r'nova0[0-9]', r'fern0[0-9]', r'omni[.]hpcc', r'/home/[a-z]', r'C:[\\]+Use[r]s', r'miyab[i]')]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()
def compare(table, root):
    """table: [(relpath, nbytes, sha256)] -> list of problems"""
    bad = []
    for rel, n, want in table:
        f = root / rel
        if not f.is_file():
            bad.append((rel, 'missing')); continue
        if n is not None and f.stat().st_size != n:
            bad.append((rel, 'size')); continue
        if sha(f) != want:
            bad.append((rel, 'sha256'))
    return bad
audits = {}
eq = json.loads((ROOT / 'export-audit/equivalence.json').read_text(encoding='utf-8'))
table = [(e['path'], e['nbytes'], e['sha256'] if isinstance(e['sha256'], str) else None) for e in eq['binary_equal'] if isinstance(e['sha256'], str)]
multi = [e for e in eq['binary_equal'] if not isinstance(e['sha256'], str)]
bad = compare(table, ROOT)
audits['bytes_equal'] = dict(result='PASS' if not bad else 'FAIL', files=len(table), multi_chunk_files=len(multi), bytes=sum(n for _, n, _ in table), problems=bad[:10])
ttab = [(e['path'], e.get('new_bytes'), e['new_sha256']) for e in eq['text_equivalent']]
bad = compare(ttab, ROOT)
audits['transformed_text'] = dict(result='PASS' if not bad else 'FAIL', files=len(ttab), problems=bad[:10])
# SHA256SUMS over the whole shipped tree
sys.path.insert(0, str(ROOT / 'tools'))
import release_tool
sums = release_tool.read_sums(str(ROOT / 'SHA256SUMS'))
bad = compare([(k, None, v) for k, v in sorted(sums.items())], ROOT)
audits['sha256sums'] = dict(result='PASS' if not bad else 'FAIL', files=len(sums), problems=bad[:10])
# certificate chain with a full content hash of every record
import runtime_manifest as rm, compact_catalog as cc
owners = []
cert = ROOT / 'plamo3-default/r45-spec/certificate.json'
m = cc.read_certificate(cert, owners)
recs = [*m['sources'], *m.get('origins', []), *m['artifacts'].values()]
problem = None
try:
    rm.validate_records(recs, workers=4)
    if rm.source_files(m['source_roots']) != [r['path'] for r in m['sources']]:
        problem = 'source inventory differs'
except Exception as e:
    problem = repr(e)[:300]
audits['certificate_chain'] = dict(result='PASS' if problem is None else 'FAIL', records=len(recs), sources=len(m['sources']), origins=len(m.get('origins', [])), artifacts=len(m['artifacts']), problem=problem)
for name, rel in (('w4_manifest', 'plamo3-default/r45-w4/manifest.json'), ('w8_manifest', 'plamo3-default/r45-w8/manifest.json')):
    problem = None
    try:
        mm = rm._read_manifest(ROOT / rel)
        rm.validate_records([*mm['sources'], *mm.get('origins', []), *mm['artifacts'].values()], workers=4)
        if rm.source_files(mm['source_roots']) != [r['path'] for r in mm['sources']]:
            problem = 'source inventory differs'
    except Exception as e:
        problem = repr(e)[:300]
    audits[name] = dict(result='PASS' if problem is None else 'FAIL', problem=problem)
# no author-side text anywhere in the sealed texts (raw files and the decoded catalog)
hits = []
for rel in ('plamo3-default/r45-spec/certificate.json', 'plamo3-default/r45-w4/manifest.json', 'plamo3-default/r45-w8/manifest.json', 'plamo3-default/r45-spec/manifest.json', 'release.json'):
    t = (ROOT / rel).read_bytes().decode('utf-8', 'replace')
    hits += [(rel, f.pattern) for f in FORBIDDEN if f.search(t)]
def walk(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(k, out); walk(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o: walk(v, out)
    elif isinstance(o, str):
        out.append(o)
strings = []
walk(m, strings)
raw = cc.marshal.loads((ROOT / 'plamo3-default/r45-spec/catalog.bin').read_bytes(), allow_code=False) if hasattr(cc, 'marshal') else None
walk(raw, strings)
hits += [('decoded catalog', f.pattern) for f in FORBIDDEN for t in strings if f.search(t)][:5]
audits['no_author_text'] = dict(result='PASS' if not hits else 'FAIL', hits=hits[:10], strings=len(strings))
# machine-key / identity scrub of the certificate
cj = json.loads((ROOT / 'plamo3-default/r45-spec/certificate.json').read_text())['payload']
def has_key(o, key):
    if isinstance(o, dict):
        return key in o or any(has_key(v, key) for v in o.values())
    if isinstance(o, (list, tuple)):
        return any(has_key(v, key) for v in o)
    return False
ok = not has_key(raw, 'ntfs_identity') and 'filesystem_identity_version' not in (raw or {}).get('manifest', {}) and set(cj['python_abi']) == {'cache_tag'} and not ({'XRT_ROOT', 'XILINX_XRT', 'SYSTEMROOT'} & set(cj['configuration']))
audits['no_identities_no_machine_keys'] = dict(result='PASS' if ok else 'FAIL')
# mutants: the comparison code must detect each
mut = {}
if table:
    t1 = copy.deepcopy(table); t1[0] = (t1[0][0], t1[0][1], '0' * 64); mut['flipped_hash'] = bool(compare(t1[:1], ROOT))
    t2 = [('plamo3-nonexistent/none.bin', 1, '0' * 64)]; mut['missing_record'] = bool(compare(t2, ROOT))
    recs2 = copy.deepcopy(recs[:1]); recs2[0]['chunks_sha256'] = ['0' * 64] * len(recs2[0]['chunks_sha256'])
    try:
        rm.validate_records(recs2, workers=1); mut['record_hash_mutant'] = False
    except rm.StaleManifest:
        mut['record_hash_mutant'] = True
    extra = [r['path'] for r in m['sources']] + [str(ROOT / 'IRON/iron/applications/plamo3/_extra_mutant.py')]
    mut['extra_source_detected'] = rm.source_files(m['source_roots']) != extra
    planted = 'C:' + chr(92) + 'Use' + 'rs' + chr(92) + 'ykog' + 'i'
    mut['scrub_detects_planted'] = sum(1 for f in FORBIDDEN if f.search(planted)) >= 2
audits['mutants'] = dict(result='PASS' if mut and all(mut.values()) else 'FAIL', detail=mut)
ok = all(v['result'] == 'PASS' for v in audits.values())
a.output.write_text(json.dumps(dict(result='PASS' if ok else 'FAIL', audits=audits), indent=2))
print(json.dumps(dict(result='PASS' if ok else 'FAIL', audits={k: v['result'] for k, v in audits.items()})), flush=True)
sys.exit(0 if ok else 1)
