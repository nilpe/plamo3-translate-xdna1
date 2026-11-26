"""R83 hoist-gate. usage: hoistgate83.py --part sdecode|sbulk|sverifya|sverifyb|drafter [-<k>of<n>] | patch | mutants | device --output f.json
single-*: for EVERY position of the EXPORTED candidate (parts read back from the exported artifact files on disk): the candidate part equals the independent re-derivation single(hoist(ORIGINAL R74 blocks)) byte for byte, and proof83/singlecmd83 prove it again
  (BD lifetime, DDR ranges against the ROOT sizes, ctrl values, queue-occupancy trace, per-channel pushes with resolved root/absolute addresses, non-BD instruction order); the candidate graph step args are exactly the root views of the re-derived slot order;
  artifact size and sha256 equal the proven build (merged.json) and the candidate manifest nbytes.
drafter: the same for r41_proposal_join (2047 positions) and r41_refresh_join: candidate part == hoist(original) and the R80 hoist proof.
patch: the candidate patch tables (r36_patch verify2 a/b, r40_patches r37_decode_bulk_taps / decode / r41_proposal_join / r48_proposal_join_en): for EVERY position and 3 base positions patched(base)==part(position) on parts read from the artifacts; every word index lies in a sync span;
  spans inside the stream; drafter word_indices equal the R80 table mapped through the permutation.
mutants: the 23 proof83 mutants (S1-S9 on decode and verify-a, D1-D5) (S1-S9, D1-D5) are caught on streams read from the exported artifacts / the originals.
device: candidate runtime (real state): graphs decode / r37_decode_bulk_taps / verify2 at 27/515/1018 plus drafter joins: candidate payload chain vs the R80-served chain, bitwise equal views (sha256 of every arg view <= 8 MB), timing report-only.
Options for dry runs outside the laptop: --candidate-manifest --old-manifest --bundle --remap "win=local;..." (artifact paths of the originals)."""
import argparse, hashlib, json, random, statistics, struct, sys, time, traceback, zlib, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()))
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--part', required=True)
p.add_argument('--candidate-manifest'); p.add_argument('--old-manifest'); p.add_argument('--bundle'); p.add_argument('--remap', default=''); a = p.parse_args()
try:
    import settings83 as s; s.paths(); CAND = s.CANDIDATE
except Exception:
    s = None; CAND = None
import hoist83 as H, singlecmd83 as S, proof83 as C
BUNDLE = Path(a.bundle) if a.bundle else ROOT / 'plamo3-r83/catalog/bundle'
REMAP = dict(x.split('=', 1) for x in a.remap.split(';') if x)
# R83 delta export: only decode / bulk are new (r83_*); verify2 and the drafter keep the R80 artifacts, instruction names and root views (byte-identical copies live in the r83 bundle under r83_* names for the merged.json hashes).
CANDN = {'r83_single_decode': 'r83_single_decode', 'r83_single_bulk': 'r83_single_bulk', 'r83_single_verify_a': 'r80_single_verify_a', 'r83_single_verify_b': 'r80_single_verify_b', 'r83_drafter_hoist': 'r80_drafter_hoist'}
SINGLE = {'decode': ('decode', ['r83_single_decode']), 'bulk': ('r37_decode_bulk_taps', ['r83_single_bulk']), 'verify-a': ('verify2', ['r83_single_verify_a']), 'verify-b': ('verify2', ['r83_single_verify_b'])}
def load():
    def rd(path):
        try:
            import runtime_manifest as base; m = base._read_manifest(path)
        except Exception:
            m = json.loads(Path(path).read_text())
        return m.get('payload', m)
    cand = rd(a.candidate_manifest or (CAND / 'manifest.json'))
    old = rd(a.old_manifest or (ROOT / 'plamo3-default/r74/manifest.json'))
    return cand, old
class Files:
    def __init__(self, m): self.m = m; self.h = {}
    def path(self, art): pth = self.m['artifacts'][art]['path']; return REMAP.get(pth, pth)
    def part(self, part):
        path = self.path(part['artifact']); f = self.h.get(path) or self.h.setdefault(path, open(path, 'rb')); f.seek(part['offset']); raw = f.read(part['nbytes'])
        out = zlib.decompress(raw) if part.get('codec') == 'zlib' else raw; assert len(out) == part['decoded_nbytes']; return out
    def read(self, part): path = self.path(part['artifact']); f = self.h.get(path) or self.h.setdefault(path, open(path, 'rb')); f.seek(part.get('offset', 0)); return f.read(part['nbytes'])
def segments(old, graph):
    G = old['graphs'][graph]['steps']; V = old['views']; segs = []; s0 = 0
    while s0 < len(G):
        e0 = s0 + 1
        while e0 < len(G) and len(S.root_slots(G[s0:e0 + 1], V)) <= 5: e0 += 1
        segs.append((s0, e0)); s0 = e0
    return segs
def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): h.update(c)
    return h.hexdigest()
def part_single(kind, shard):
    cand, old = load(); fc, fo = Files(cand), Files(old); graph, names = SINGLE[kind]; name = names[0]
    segs = segments(old, graph); seg = segs[0 if kind != 'verify-b' else 1]; steps = old['graphs'][graph]['steps'][seg[0]:seg[1]]
    V = old['views']; RS = {k: v['nbytes'] for k, v in old['roots'].items()}
    merged = json.loads((BUNDLE / 'merged.json').read_text())[name]; cname = CANDN[name]; rec = cand['artifacts'][cname]; path = fc.path(cname)
    assert os.path.getsize(path) == rec['nbytes'] == merged['nbytes'], 'artifact size'
    if shard is None or shard[0] == 0: assert sha_file(path) == merged['sha256'], 'artifact sha256 differs from the proven build'
    cst = cand['graphs'][graph]['steps']; assert [x['instructions'] for x in cst] == [x for x in SINGLE_ALL(graph)], 'candidate graph steps'
    slots = S.root_slots(steps, V); stp = cst[segs.index(seg)]; assert stp['args'] == ['r80_root_' + r for r in slots], 'candidate args are not the root views of the re-derived slot order'
    cases_c = cand['instructions'][cname]['cases']; positions = sorted(int(k) for k in cases_c); lo, hi = old['graphs'][graph]['parameter_ranges']['position']; assert positions == list(range(lo, hi + 1))
    n = dict(parts=0, bd=0, hoisted_bd=0, insts=0, nonlinear=0, total_positions=len(positions))
    for i, pos in enumerate(positions):
        if shard is not None and i % shard[1] != shard[0]: continue
        pls = [fo.part(old['instructions'][st['instructions']]['cases'][str(pos)]) for st in steps]; rc = fc.part(cases_c[str(pos)])
        raw, info = C.prove_single(pls, steps, V, RS, H.header_of(pls[0]), unhoist_head=graph in C.HEAD_UNHOIST); assert raw == rc, (name, pos, 'candidate differs from the independent re-derivation')
        ps = H.parse(rc); A = H.check(ps, [RS[r] for r in slots]); assert A['bd'] == info['bd'] and A['trace'] is not None
        n['parts'] += 1; n['bd'] += info['bd']; n['hoisted_bd'] += info['hoisted_bd']; n['insts'] += info['insts']; n['nonlinear'] += info['nonlinear']
    return dict(kind=kind, shard=shard, artifact=name, artifact_bytes=rec['nbytes'], artifact_sha256=merged['sha256'], expected_parts=merged['parts'], **n)
def SINGLE_ALL(graph): return {'decode': ['r83_single_decode'], 'r37_decode_bulk_taps': ['r83_single_bulk'], 'verify2': ['r80_single_verify_a', 'r80_single_verify_b']}[graph]
def part_drafter(shard):
    cand, old = load(); fc, fo = Files(cand), Files(old); merged = json.loads((BUNDLE / 'merged.json').read_text())['r83_drafter_hoist']; rec = cand['artifacts']['r80_drafter_hoist']; path = fc.path('r80_drafter_hoist')
    assert os.path.getsize(path) == rec['nbytes'] == merged['nbytes']
    if shard is None or shard[0] == 0: assert sha_file(path) == merged['sha256']
    V = cand['views']; n = dict(parts=0, bd=0, hoisted_bd=0, insts=0, nonlinear=0)
    for name in ('r41_proposal_join', 'r41_refresh_join'):
        sizes = {tuple(V[x]['nbytes'] for x in st['args']) for g in cand['graphs'].values() for st in g['steps'] if st['instructions'] == name}
        cc, oc = cand['instructions'][name]['cases'], old['instructions'][name]['cases']; assert set(cc) == set(oc)
        for i, key in enumerate(sorted(oc, key=lambda k: (len(k), k))):
            if shard is not None and i % shard[1] != shard[0]: continue
            assert cc[key]['artifact'] == 'r80_drafter_hoist' and oc[key]['artifact'] == 'r40_instructions'
            ro, rc = fo.part(oc[key]), fc.part(cc[key]); out, info = C.prove_drafter(ro, sizes); assert out == rc, (name, key, 'candidate differs from re-derivation')
            n['parts'] += 1
            for k2 in ('bd', 'hoisted_bd', 'insts', 'nonlinear'): n[k2] += info[k2]
    return dict(kind='drafter', shard=shard, artifact_bytes=rec['nbytes'], artifact_sha256=merged['sha256'], expected_parts=merged['parts'], **n)
def patch_gate():
    cand, old = load(); fc = Files(cand); rnd = random.Random(80); rep = {}
    specs = [('verify2', 'r36', cand['r36_patch']['steps'])] + [(g, 'r40', cand['r40_patches'][g]['steps']) for g in ('r37_decode_bulk_taps', 'decode')]
    jobs = [(g + ':' + nm, d, cand['instructions'][nm]['cases'], 0) for g, kind, st in specs for nm, d in st.items()]
    for spec in ('r41_proposal_join', 'r48_proposal_join_en'):
        d = cand['r40_patches'][spec]['steps']['r41_proposal_join']; jobs.append((spec, d, cand['instructions']['r41_proposal_join']['cases'], 1))
    for label, d, cases, drafter in jobs:
        vb = fc.read(dict(d['values'])); vals = struct.unpack('<%dI' % (len(vb) // 4), vb); width = len(d['word_indices']); positions = sorted(int(x) for x in d['positions'])
        assert len(vals) == len(positions) * width, (label, 'values extent'); spans = d.get('sync_spans') or [d['sync_span']]
        assert all(lo >= 0 and lo + sz <= d['nbytes'] for lo, sz in spans), (label, 'span outside stream')
        assert all(any(lo <= w * 4 and w * 4 + 4 <= lo + sz for lo, sz in spans) for w in d['word_indices']), (label, 'a patched word is not inside any sync span')
        cache = {}
        def hb(q):
            if q not in cache: cache[q] = fc.part(cases[str(q)])
            return cache[q]
        assert len(hb(positions[0])) == d['nbytes'], (label, 'nbytes')
        bases = [positions[0]] + [rnd.choice(positions) for _ in range(2)]; cmp = 0
        for q in positions:
            tgt = hb(q); begin = d['positions'][str(q)] * width
            for b in bases:
                host = bytearray(hb(b)); w = memoryview(host).cast('I')
                for i, off in enumerate(d['word_indices']): w[off] = vals[begin + i]
                assert bytes(host) == tgt, ('patch mismatch', label, b, q); cmp += 1
            if len(cache) > 6: cache = {k: v for k, v in list(cache.items())[-3:]}
        rep[label] = dict(positions=len(positions), width=width, spans=len(spans), compares=cmp)
    return dict(patch=rep, specs=len(rep))
def mutants_gate():
    cand, old = load(); fo = Files(old); V = old['views']; RS = {k: v['nbytes'] for k, v in old['roots'].items()}; res = {}
    for graph, segi, label in (('decode', 0, 'decode'), ('verify2', 0, 'verify2')):
        segs = segments(old, graph); steps = old['graphs'][graph]['steps'][segs[segi][0]:segs[segi][1]]; pls = [fo.part(old['instructions'][st['instructions']]['cases']['515']) for st in steps]
        res[label] = C.mutants_single(pls, steps, V, RS)
    # R83: head-group rule mutants. The candidate bytes are derived with the head group un-hoisted; hoisting it (the R80 form) or un-hoisting a different group must give DIFFERENT bytes (caught as a byte difference by the single-* parts).
    res['head_rule'] = {}
    for graph in C.HEAD_UNHOIST:
        segs = segments(old, graph); steps = old['graphs'][graph]['steps'][segs[0][0]:segs[0][1]]; pls = [fo.part(old['instructions'][st['instructions']]['cases']['515']) for st in steps]
        good, _ = C.prove_single(pls, steps, V, RS, H.header_of(pls[0]), unhoist_head=True); bad, _ = C.prove_single(pls, steps, V, RS, H.header_of(pls[0]), unhoist_head=False)
        slots, slot_of, ps2, osc, orest, tr = S.build(pls, steps, V, False); ng = len(H.groups_of(ps2)); out, _ = H.hoist(ps2, skipk={ng - 3: {'bd', 'ctrl'}}); wrong = H.encode_h(H.header_of(pls[0]), out)
        res['head_rule'][graph + ':S10_head_group_rehoisted'] = 'CAUGHT' if bad != good else 'NOT CAUGHT'; res['head_rule'][graph + ':S11_wrong_group_unhoisted'] = 'CAUGHT' if wrong != good else 'NOT CAUGHT'
    raw = fo.part(old['instructions']['r41_proposal_join']['cases']['515']); asz = [tuple(V[x]['nbytes'] for x in old['graphs']['r41_proposal_join']['steps'][0]['args'])][0]
    res['drafter'] = C.mutants_drafter(raw, list(asz)); caught = sum(len(d) for d in res.values()); assert all(not v.startswith('NOT') for d in res.values() for v in d.values()), res
    return dict(mutants=res, mutants_caught=caught)
def device_gate():
    """candidate runtime (exercised in the pre-freeze dry run on the staged export): candidate runtime, real state, 9 graph/position cells + drafter joins. A = R80-served chain re-derived from the R74 originals (hoist per block, 17 commands);
    B = the candidate's own parts through rt._cases/_instruction_bytes with the candidate step args (root views). Bitwise: sha256 of every arg view <= 8 MB after A, A, B, A; timing report-only."""
    from default83 import DefaultRuntime
    cand, old = load(); fo = Files(old); V0 = old['views']
    app = DefaultRuntime(spec_manifest=s.CANDIDATE_SPEC); assert app.mode == 'w4-m2', app.fallbacks; rt = app.env.runtime; xrt = rt.xrt; sess = app.session
    meta = json.loads(((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text()); prompt = next(x for x in meta['long_cases'] if x['id'] == 'en-512')['prompt']
    got = sess.prefill(prompt); tok = int(got['token'])
    for i in range(6): got = sess.decode(tok); tok = int(got['token'])
    res = {}
    class Win:
        def __init__(self, ctx, entries):
            self.keep = []; _, context, kernel, nargs = rt.contexts[ctx]; self.rl = xrt.runlist(context)
            for pl, bufs in entries:
                bo = xrt.bo(rt.device, len(pl), xrt.bo.cacheable, kernel.group_id(1)); host = bo.map(); host[:len(pl)] = pl; bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                run = xrt.run(kernel); run.set_arg(0, 3); run.set_arg(1, bo); run.set_arg(2, len(pl)); bufs = list(bufs)
                if len(bufs) < nargs - 3: bufs += [bufs[0]] * (nargs - 3 - len(bufs))
                for i, b in enumerate(bufs, start=3): run.set_arg(i, b)
                self.rl.add(run); self.keep.append((bo, run))
        def go(self): t = time.perf_counter(); self.rl.execute(); self.rl.wait(); return time.perf_counter() - t
    for gname in ('decode', 'r37_decode_bulk_taps', 'verify2'):
        if gname == 'verify2':
            from verify_session import VerifySession
            vs = VerifySession(sess); vs.stage([tok, tok]); vs.commit(0)
        Gc = rt.manifest['graphs'][gname]['steps']; Go = old['graphs'][gname]['steps']; V = rt.manifest['views']
        snapviews = sorted({x for e in Go for x in e['args'] if V0[x]['nbytes'] <= 8 * 1024 * 1024})
        def snap():
            h = hashlib.sha256()
            for v in snapviews: h.update(rt.read(v))
            return h.hexdigest()
        for pos in (27, 515, 1018):
            op = [fo.part(old['instructions'][st['instructions']]['cases'][str(pos)]) for st in Go]
            hois = []
            for raw in op:
                ps = H.tag(H.parse(raw)); out, _ = H.hoist(ps); hois.append(H.encode_h(H.header_of(raw), out))
            A = Win(Go[0]['context'], [(pl, [rt._bo(v) for v in Go[i]['args']]) for i, pl in enumerate(hois)])
            cp = [bytes(rt._instruction_bytes(q)) for q in rt._cases(gname, {'position': pos})]
            B = Win(Gc[0]['context'], [(pl, [rt._bo(v) for v in Gc[i]['args']]) for i, pl in enumerate(cp)])
            A.go(); a0 = snap(); A.go(); a1 = snap(); B.go(); b0 = snap(); A.go(); a2 = snap()
            assert a0 == a1 == a2 and b0 == a1, (gname, pos, 'state differs'); rr = {'A': [], 'B': []}; rnd = random.Random(pos)
            for r in range(60):
                for n, w in rnd.sample([('A', A), ('B', B)], 2):
                    d = w.go()
                    if r >= 5: rr[n].append(d * 1000)
            res['%s@%d' % (gname, pos)] = dict(bitwise_equal=True, r80_ms=statistics.median(rr['A']), r83_ms=statistics.median(rr['B']), views=len(snapviews))
    app.close(); return dict(device=res)
try:
    kind, _, sh = a.part.partition('-'); shard = tuple(int(x) for x in sh.split('of')) if sh else None   # runner step hoist-gate-<kind>[-<k>of<n>]
    ALIAS = {'sdecode': 'decode', 'sbulk': 'bulk', 'sverifya': 'verify-a', 'sverifyb': 'verify-b'}
    if kind in ALIAS:
        rep = part_single(ALIAS[kind], shard); ok = rep['parts'] > 0 and (shard is not None or rep['parts'] == rep['expected_parts'])
    elif kind == 'drafter':
        rep = part_drafter(shard); ok = rep['parts'] > 0 and (shard is not None or rep['parts'] == rep['expected_parts'])
    elif a.part == 'patch': rep = patch_gate(); ok = rep['specs'] == 6 and all(v['compares'] > 0 for v in rep['patch'].values())
    elif a.part == 'mutants': rep = mutants_gate(); ok = rep['mutants_caught'] == 27   # 23 proof83 mutants + 4 head-rule mutants (S10 head group re-hoisted, S11 wrong group un-hoisted, for decode and bulk)
    elif a.part == 'device': rep = device_gate(); ok = len(rep['device']) == 9
    else: raise ValueError(a.part)
    a.output.write_text(json.dumps(dict(result='PASS' if ok else 'FAIL', part=a.part, **rep), indent=1)); print(json.dumps(dict(result='PASS' if ok else 'FAIL', part=a.part, **{k: v for k, v in rep.items() if k not in ('device', 'patch', 'mutants')})), flush=True)
    sys.exit(0 if ok else 1)
except SystemExit: raise
except BaseException:
    a.output.write_text(json.dumps(dict(result='FAIL', part=a.part, error=traceback.format_exc()), indent=2)); traceback.print_exc(); sys.exit(1)
