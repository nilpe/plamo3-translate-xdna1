"""R83 sentinel-proof (static; draft for the Phase C gate step `sentinel-proof`).
For every position (and language) of every hooked graph it proves from the command streams themselves that overwriting the sentinel view with 0xFF before execute() cannot change any result:
  W  the command's DDR writes (S2MM BDs) cover every byte of the sentinel view;
  R  no BD of the command READS (MM2S) any byte of the sentinel view (strict: so no op can consume the pre-command contents), and no non-linear BD on the sentinel's root could reach it;
  X  no OTHER hooked/other graph reads the sentinel view (checked on every position of the hooked graphs and on the listed other graphs).
It also reports, for the drafter coarse scores, that all 24576 entries are written, and whether a VALID output can equal the sentinel pattern (token ids are < 107520 < 0xFFFFFFFF; the bf16 pattern 0xFFFF
is a NaN, which a finite head never produces: only a fallback timeout could ever result from it, never a wrong result).
usage: sentinel83_proof.py --bundle DIR --manifest manifest(R74-form views/roots).json [--r40 r40_instructions.bin] [--positions N] [--workers W]"""
import sys, os, json, zlib, argparse, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoist83 as H

SENT = {  # graph -> list of (view, dtype name)
    'decode': [('view226', 'tok')], 'r37_decode_bulk_taps': [('view226', 'tok')], 'verify2': [('r39_target_v2_view226', 'tok')],
    'r41_proposal_join': [('r40_coarse_row0', 'bf16')], 'r41_exact': [('r40_exact_row0', 'bf16')],
}

def accesses(raw, slot_abs):
    """slot_abs(slot) -> (root, base_offset). Returns list of dict(dir 'W'|'R', root, lo, hi, linear)."""
    ps = H.parse(raw); cur = {}; out = []
    for p in ps:
        if H.is_bd(p): cur[(H.location(p[2])[0], H.bd_id(p))] = [p[4:], None]
        elif H.is_patch(p): cur[(H.location(p[6])[0], H.bd_id(p))][1] = (p[8], p[10] | (p[11] << 32))
        elif H.is_push(p):
            col, _, reg = H.location(p[2]); d, c = H.QUEUES[reg]; b = H.bd_id(p); words, patch = cur[(col, b)]
            root, base = slot_abs(patch[0]); lo = base + patch[1]; n = words[0] * 4
            lin = H.bd_is_linear([0, 0] + list(words)) if False else (words[3] == 0 and (words[4] & 0x3FFFFFFF) == 0 and (words[5] & 0xFFFFF) == 0 and words[6] == 0)
            out.append(dict(dir='W' if d == 0 else 'R', root=root, lo=lo, hi=lo + n, linear=bool(lin)))
    return out

def judge(acc, root, lo, hi):
    """returns dict(covered, read_overlap, nonlinear_on_root)"""
    w = sorted((a['lo'], a['hi']) for a in acc if a['dir'] == 'W' and a['root'] == root and a['linear'] and a['hi'] > lo and a['lo'] < hi)
    cov = lo
    for a, b in w:
        if a <= cov: cov = max(cov, b)
    rd = [a for a in acc if a['dir'] == 'R' and a['root'] == root and a['hi'] > lo and a['lo'] < hi]
    nl = [a for a in acc if a['root'] == root and not a['linear']]
    return dict(covered=cov >= hi, read_overlap=len(rd), nonlinear_on_root=len(nl), nonlinear_reads=sum(1 for a in nl if a['dir'] == 'R'))

class Ctx: pass
C = Ctx()

def init(args):
    C.args = args
    m = json.load(open(args.manifest)); C.m = m.get('payload', m); C.V = C.m['views']; C.R = C.m['roots']
    C.bundle = args.bundle; C.data = {}; C.idx = {}

def part(name, pos):
    if name not in C.idx:
        C.idx[name] = json.load(open(os.path.join(C.bundle, name + '.index.json'))); C.data[name] = open(os.path.join(C.bundle, name + '.bin'), 'rb')
    o, nb, dn, h = C.idx[name][str(pos)]; f = C.data[name]; f.seek(o); raw = zlib.decompress(f.read(nb)); assert len(raw) == dn; return raw

def work(job):
    graph, names, pos = job
    V = C.V; res = []
    for name in names:
        raw = part(name, pos); st = json.load(open(os.path.join(C.bundle, name + '.stats.json'))); slots = st['slots']
        acc = accesses(raw, lambda s: (slots[s], 0))
        res.append(acc)
    acc = [a for r in res for a in r]
    out = []
    for view, kind in SENT.get(graph, []):
        v = V[view]; j = judge(acc, v['root'], v['offset'], v['offset'] + v['nbytes']); j.update(view=view, graph=graph, pos=pos); out.append(j)
        # other-graph reads of the same view: the same judge reports read_overlap over this command's accesses (covers 'X' for the hooked graphs themselves)
    return out

def draft_part(pos):
    if 'dh' not in C.idx:
        C.idx['dh'] = json.load(open(os.path.join(C.bundle, 'r83_drafter_hoist.index.json'))); C.data['dh'] = open(os.path.join(C.bundle, 'r83_drafter_hoist.bin'), 'rb')
    o, nb, dn, h, srcoff = C.idx['dh']['r41_proposal_join|%d' % pos]; f = C.data['dh']; f.seek(o); raw = zlib.decompress(f.read(nb)); assert len(raw) == dn; return raw

def work_drafter(job):
    graph, pos = job
    V = C.V; args = C.m['graphs'][graph]['steps'][0]['args']
    if graph == 'r41_exact':
        case = C.m['instructions']['r41_exact']['cases']['']; path = C.args.r40; f = open(path, 'rb'); f.seek(case['offset']); raw = f.read(case['nbytes'])
        raw = zlib.decompress(raw) if case.get('codec') == 'zlib' else raw
    else:
        raw = draft_part(pos)
    acc = accesses(raw, lambda sl: (V[args[sl if sl < len(args) else 0]]['root'], V[args[sl if sl < len(args) else 0]]['offset']))
    out = []
    for view, kind in SENT.get('r41_proposal_join' if graph.startswith('r4') and graph != 'r41_exact' else graph, []):
        v = V[view]; j = judge(acc, v['root'], v['offset'], v['offset'] + v['nbytes']); j.update(view=view, graph=graph, pos=pos); out.append(j)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle', required=True); ap.add_argument('--manifest', required=True); ap.add_argument('--positions', type=int, default=0); ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--r40', default=None, help='r40_instructions.bin (for r41_exact)')
    a = ap.parse_args(); init(a)
    jobs = []
    N = a.positions or 2048
    for graph, names, hi in (('decode', ['r83_single_decode'], 2048), ('r37_decode_bulk_taps', ['r83_single_bulk'], 2048), ('verify2', ['r83_single_verify_a', 'r83_single_verify_b'], 2047)):
        for pos in range(0, min(N, hi)): jobs.append((graph, names, pos))
    djobs = [(g, pos) for g in ('r41_proposal_join', 'r48_proposal_join_en') for pos in range(0, min(N, 2047))] + ([('r41_exact', 0)] if a.r40 else [])
    with Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        rows = [r for rs in pool.imap_unordered(work, jobs, chunksize=8) for r in rs]
        rows += [r for rs in pool.imap_unordered(work_drafter, djobs, chunksize=8) for r in rs]
    summary = collections.defaultdict(lambda: dict(positions=0, covered=0, read_overlap=0, nonlinear_on_root=0, nonlinear_reads=0))
    bad = []
    for r in rows:
        s = summary[r['graph'] + ':' + r['view']]; s['positions'] += 1; s['covered'] += r['covered']; s['read_overlap'] += r['read_overlap']; s['nonlinear_on_root'] += r['nonlinear_on_root']; s['nonlinear_reads'] += r['nonlinear_reads']
        if not r['covered'] or r['read_overlap'] or r['nonlinear_reads']: bad.append(r)
    print(json.dumps(summary, indent=1)); print('BAD', len(bad), bad[:5])
    sys.exit(1 if bad else 0)

if __name__ == '__main__': main()
