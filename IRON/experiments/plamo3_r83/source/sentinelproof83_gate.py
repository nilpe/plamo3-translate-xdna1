"""R83 sentinel-proof gate (static), run against the EXPORTED candidate (parts read back from the exported artifacts).
(1) main graphs decode / r37_decode_bulk_taps / verify2, EVERY position: the S2MM (write) BDs cover every byte of the sentinel view; no MM2S (read) BD overlaps it; no non-linear BD of the command lies on its root.
(2) cross-graph scan (d): for EVERY other graph of the manifest (case keys sampled: first, last and about 40 spread keys per instruction, all keys when fewer), no BD READS the main-graph sentinel views; non-linear BDs on those roots are counted and
    any non-linear READ on them fails the gate. (3) drafter views (r40_coarse_row0 / r40_exact_row0): the write BDs are non-linear (strided), so the linear coverage scan does not apply; the gate records the counts and refers to the dynamic
    gate steps sentinel-dyn-drafter (fill / double-fill / mutants). The R41 drafter graphs are also scanned for linear reads overlapping those views (must be 0).
usage: sentinelproof83_gate.py --output f.json [--candidate-manifest path]"""
import argparse, json, sys, os, zlib, collections, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--candidate-manifest'); a = ap.parse_args()
try:
    import settings83 as s; s.paths(); CAND = s.CANDIDATE
except Exception:
    s = None; CAND = None
import hoist83 as H
from sentinel83_proof import accesses, judge
MAIN = {'decode': 'view226', 'r37_decode_bulk_taps': 'view226', 'verify2': 'r39_target_v2_view226'}
DRAFT = {'r40_coarse_row0', 'r40_exact_row0'}
def rd(path):
    try:
        import runtime_manifest as base; m = base._read_manifest(path)
    except Exception:
        m = json.loads(Path(path).read_text())
    return m.get('payload', m)
class Files:
    def __init__(self, m): self.m = m; self.h = {}
    def part(self, part):
        pth = self.m['artifacts'][part['artifact']]['path']; f = self.h.get(pth) or self.h.setdefault(pth, open(pth, 'rb')); f.seek(part['offset']); raw = f.read(part['nbytes'])
        out = zlib.decompress(raw) if part.get('codec') == 'zlib' else raw; assert len(out) == part['decoded_nbytes']; return out
def sample(keys):
    keys = sorted(keys, key=lambda k: (len(k), k))
    if len(keys) <= 48: return keys
    step = max(1, len(keys) // 40); return sorted(set(keys[::step] + [keys[0], keys[-1]]), key=lambda k: (len(k), k))
try:
    cand = rd(a.candidate_manifest or (CAND / 'manifest.json')); V = cand['views']; fc = Files(cand); G = cand['graphs']; I = cand['instructions']
    def slotabs(step):
        args = step['args']
        return lambda sl: (V[args[sl if sl < len(args) else 0]]['root'], V[args[sl if sl < len(args) else 0]]['offset'])
    rep = dict(main={}, cross={}, drafter={})
    # (1) main graphs, every position
    for g, view in MAIN.items():
        v = V[view]; steps = G[g]['steps']; names = [st['instructions'] for st in steps]; keys = sorted(int(k) for k in I[names[0]]['cases']); cov = rd_ = nl = nlr = 0
        for pos in keys:
            acc = []
            for st in steps: acc += accesses(fc.part(I[st['instructions']]['cases'][str(pos)]), slotabs(st))
            j = judge(acc, v['root'], v['offset'], v['offset'] + v['nbytes']); cov += j['covered']; rd_ += j['read_overlap']; nl += j['nonlinear_on_root']; nlr += j['nonlinear_reads']
        rep['main'][g] = dict(view=view, positions=len(keys), covered=cov, read_overlap=rd_, nonlinear_on_root=nl, nonlinear_reads=nlr, ok=(cov == len(keys) and rd_ == 0 and nlr == 0))
    # (2) cross-graph scan
    watch = [(MAIN[g], V[MAIN[g]]) for g in MAIN]; watch = list({w[0]: w for w in watch}.values()); bad = []
    for g, gv in G.items():
        if g in MAIN: continue
        n = rdo = nlr = nl = 0
        for st in gv['steps']:
            iname = st['instructions']; cases = I[iname]['cases']
            for key in sample(cases):
                acc = accesses(fc.part(cases[key]), slotabs(st)); n += 1
                for view, v in watch:
                    j = judge(acc, v['root'], v['offset'], v['offset'] + v['nbytes']); rdo += j['read_overlap']; nl += j['nonlinear_on_root']; nlr += j['nonlinear_reads']
        rep['cross'][g] = dict(parts=n, read_overlap=rdo, nonlinear_on_root=nl, nonlinear_reads=nlr)
        if rdo or nlr: bad.append(g)
    # (3) drafter views
    for g in ('r41_proposal_join', 'r48_proposal_join_en', 'r41_exact'):
        if g not in G: continue
        for st in G[g]['steps']:
            cases = I[st['instructions']]['cases']; cnt = collections.Counter()
            for key in sample(cases):
                acc = accesses(fc.part(cases[key]), slotabs(st))
                for view in DRAFT:
                    v = V[view]; j = judge(acc, v['root'], v['offset'], v['offset'] + v['nbytes'])
                    cnt['parts'] += 1; cnt['linear_read_overlap'] += j['read_overlap']; cnt['nonlinear_on_root'] += j['nonlinear_on_root']; cnt['nonlinear_reads_on_root'] += j['nonlinear_reads']; cnt['covered_linear'] += j['covered']
            rep['drafter'][g] = dict(cnt, note='write BDs are non-linear: the linear coverage scan does not apply; coverage / write-only are proven dynamically by sentinel-dyn-drafter')
        if rep['drafter'].get(g, {}).get('linear_read_overlap'): bad.append('drafter:' + g)
    ok = all(v['ok'] for v in rep['main'].values()) and not bad
    a.output.write_text(json.dumps(dict(result='PASS' if ok else 'FAIL', bad=bad, **rep), indent=1)); print(json.dumps(dict(result='PASS' if ok else 'FAIL', main=rep['main'], bad=bad)), flush=True)
    sys.exit(0 if ok else 1)
except SystemExit: raise
except BaseException:
    a.output.write_text(json.dumps(dict(result='FAIL', error=traceback.format_exc()), indent=2)); traceback.print_exc(); sys.exit(1)
