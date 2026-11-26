"""R83 sentinel-dyn gate (device), run on the EXPORTED candidate (DefaultRuntime(spec_manifest=CANDIDATE_SPEC); PLAMO3_HYBRID_WAIT=0 so the gate drives the ordinary path itself).
For EVERY position / case of the part's graphs, after a short real generate:
 (a) COVERAGE: fill the sentinel view with 0xFF, flush, run through the ordinary rt.execute, require 0 sentinel words left;
 (b) WRITE-ONLY (double fill): run the same position again with the view pre-filled with ZEROS and require every view on the graph's roots (<= 8 MB, the sentinel view included) bitwise identical to the first run;
 (c) MUTANTS that must be detected: coverage on a view the command does not write (an input view: the sentinel is left) and the double fill on a view the command reads (outputs differ).
usage: sentineldyn83_gate.py --part decode|bulk|verify|drafter --output f.json"""
import argparse, json, sys, os, time, hashlib, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True); ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
base, _, shard = a.part.partition('-'); a.part = base; SH = tuple(int(x) for x in shard.split('of')) if shard else None   # part may carry a shard: decode-3of8 = positions i % 8 == 3
def pick(positions): return [p for i, p in enumerate(positions) if SH is None or i % SH[1] == SH[0]]
os.environ['PLAMO3_HYBRID_WAIT'] = '0'
try:
    import settings83 as s; s.paths()
    import hybridwait83 as HW
    from default83 import DefaultRuntime
    app = DefaultRuntime(spec_manifest=s.CANDIDATE_SPEC); assert app.mode == 'w4-m2', app.fallbacks
    rt = app.env.runtime; d = app.drafter; M = rt.manifest; V = M['views']
    meta = json.loads(((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text()); app.generate(meta['prompts'][0], 40, target_language='ja')
    TO, FROM = rt.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, rt.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    class ViewS(HW.ViewSentinel):
        def __init__(self, rt, view, n=None):
            self.rt = rt; self.view = view; self.mv = rt.views[view]; self.n = n or len(self.mv); self.mv = self.mv[:self.n]; self.bo = rt._bo(view); self.to_dev, self.from_dev = TO, FROM
        def zero(self): self.mv[:] = bytes(self.n); self.bo.sync(self.to_dev)
    class ArrS(HW.ArraySentinel):
        def zero(self): getattr(self.d, self.an)[:] = 0; getattr(self.d, self.sn)(self.d._to)
    def views_on(graph):
        roots = {V[x]['root'] for st in M['graphs'][graph]['steps'] for x in st['args']}
        return sorted(v for v, i in V.items() if i['root'] in roots and i['nbytes'] <= 8 * 1024 * 1024 and not v.startswith('r80_root_') and not v.startswith('r83_root_'))
    def snap(views):
        h = hashlib.sha256()
        for v in views: h.update(rt.read(v))
        return h.hexdigest()
    def check(label, spec, run, positions, views):
        bad_cov, bad_dbl = [], []; t0 = time.perf_counter()
        for p in positions:
            spec.prepare(); run(p); ok = spec.done(); sa = snap(views)
            if not ok: bad_cov.append(p)
            spec.zero(); run(p); sb = snap(views)
            if sa != sb: bad_dbl.append(p)
        return dict(label=label, positions=len(positions), not_overwritten=len(bad_cov), double_fill_differs=len(bad_dbl), first_bad=(bad_cov[:3], bad_dbl[:3]), views=len(views), seconds=time.perf_counter() - t0)
    res = {}; ok = True
    def main_graph(g, nm):
        global ok
        lo, hi = M['graphs'][g].get('parameter_ranges', {}).get('position', [0, 2047])
        spec = ViewS(rt, M['graphs'][g]['sync_outputs'][0]); views = views_on(g)
        r = check(g, spec, lambda p: rt.execute(g, {'position': p}), pick(list(range(lo, hi + 1))), views); res[nm] = r; ok = ok and r['not_overwritten'] == 0 and r['double_fill_differs'] == 0
        # mutants on the decode input view (the command READS it): coverage must FAIL (sentinel left) and the double fill must differ
        inp = app.session.model['decode_embedding'] if g == 'decode' else None
        if inp is not None and a.part == 'decode' and (SH is None or SH[0] == 0):
            m = ViewS(rt, inp, 8); mr = check('mutant_input_view', m, lambda p: rt.execute('decode', {'position': p}), [100, 200, 300], views_on('decode'))
            res['mutants'] = dict(coverage_on_unwritten_view_detected=mr['not_overwritten'] > 0, double_fill_on_read_view_detected=mr['double_fill_differs'] > 0, detail=mr); ok = ok and res['mutants']['coverage_on_unwritten_view_detected'] and res['mutants']['double_fill_on_read_view_detected']
    if a.part == 'decode': main_graph('decode', 'decode')
    elif a.part == 'bulk': main_graph('r37_decode_bulk_taps', 'bulk')
    elif a.part == 'verify': main_graph('verify2', 'verify2')
    elif a.part == 'drafter':
        for lang in ('ja', 'en'):
            rt._r48_language = lang; g = 'r41_proposal_join'; views = views_on(g)
            r = check('proposal[%s]' % lang, ArrS(d, '_coarse_host', '_coarse_sync'), lambda p: rt.execute(g, {'position': p}), pick(list(range(0, 2047))), views); res['proposal_' + lang] = r; ok = ok and r['not_overwritten'] == 0 and r['double_fill_differs'] == 0
        if SH is not None and SH[0] != 0: res['exact_skipped_in_this_shard'] = True; app.close(); a.output.write_text(json.dumps(dict(result='PASS' if ok else 'FAIL', part=a.part, shard=SH, **res), indent=1)); print(json.dumps(dict(result='PASS' if ok else 'FAIL', part=a.part, shard=SH)), flush=True); sys.exit(0 if ok else 1)
        r = check('exact', ArrS(d, '_exact_host', '_exact_sync'), lambda p: rt.execute('r41_exact', {}), [0, 1, 2], views_on('r41_exact')); res['exact'] = r; ok = ok and r['not_overwritten'] == 0 and r['double_fill_differs'] == 0
        # mutants: an INPUT view of the proposal command (it reads it): coverage must fail, double fill must differ
        names = [v for v in V if v.startswith('r40_embed') and V[v]['nbytes'] <= 8 * 1024 * 1024]; assert names, 'no r40_embed input view found for the mutant'
        rt._r48_language = 'ja'; m = ViewS(rt, sorted(names)[0], 8); mr = check('mutant_input_view', m, lambda p: rt.execute('r41_proposal_join', {'position': p}), [100, 200, 300], views_on('r41_proposal_join'))
        res['mutants'] = dict(view=sorted(names)[0], coverage_on_unwritten_view_detected=mr['not_overwritten'] > 0, double_fill_on_read_view_detected=mr['double_fill_differs'] > 0, detail=mr); ok = ok and res['mutants']['coverage_on_unwritten_view_detected'] and res['mutants']['double_fill_on_read_view_detected']
    else: raise ValueError(a.part)
    app.close()
    a.output.write_text(json.dumps(dict(result='PASS' if ok else 'FAIL', part=a.part, shard=SH, **res), indent=1)); print(json.dumps(dict(result='PASS' if ok else 'FAIL', part=a.part, shard=SH, **{k: (v if k == 'mutants' else {x: v[x] for x in ('positions', 'not_overwritten', 'double_fill_differs')}) for k, v in res.items()})), flush=True)
    sys.exit(0 if ok else 1)
except SystemExit: raise
except BaseException:
    a.output.write_text(json.dumps(dict(result='FAIL', part=a.part, error=traceback.format_exc()), indent=2)); traceback.print_exc(); sys.exit(1)
