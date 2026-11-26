"""R83: hoist transform + static checker for REAL coalesced R80 command-block streams (decode, verify2).
Hoist: for each group g>=1 (group = instructions between wait blocks), the shim BD writes, DDR patches and shim channel-ctrl
maskwrites are moved ahead of wait block g-1, BD ids renumbered per tile (hoisted BDs get ids that are free while group g-1 is in
flight, the rest get the remaining ids); everything else (task pushes, RTP/lock writes, mask polls, core-side writes, waits) keeps its
relative order. Pushes never move across a wait block."""
import sys, os
from array import array
from collections import Counter, deque, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txn_probe_lib import parse, kind, location, QUEUES

BD_BASE, BD_STRIDE, N_BD, QDEPTH = 0x1D000, 0x20, 16, 4
CTRL_REGS = (0x1D200, 0x1D208, 0x1D210, 0x1D218)


class T(list):
    idx = None


def tag(ps):
    out = []
    for i, p in enumerate(ps):
        t = T(p); t.idx = i; out.append(t)
    return out


def header_of(raw):
    w = array("I", raw)
    return list(w[:2])


def encode_h(hdr, parts):
    data = [v for p in parts for v in p]
    return array("I", [hdr[0], hdr[1], len(parts), 16 + 4 * len(data)] + data).tobytes()


def is_bd(p):
    if p[0] == 1:
        c, r, reg = location(p[2]); return r == 0 and BD_BASE <= reg < BD_BASE + N_BD * BD_STRIDE
    return False


def is_patch(p):
    if p[0] == 129:
        c, r, reg = location(p[6]); return r == 0 and BD_BASE <= reg < BD_BASE + N_BD * BD_STRIDE
    return False


def is_ctrl(p):
    if p[0] == 3:
        c, r, reg = location(p[2]); return r == 0 and reg in CTRL_REGS
    return False


def is_push(p):
    return p[0] == 0 and kind(p) == "task_push"


def hoistable(p):
    return is_bd(p) or is_patch(p) or is_ctrl(p)


def bd_id(p):
    if is_bd(p): return (location(p[2])[2] - BD_BASE) // BD_STRIDE
    if is_patch(p): return (location(p[6])[2] - 4 - BD_BASE) // BD_STRIDE
    if is_push(p): return p[4] & 0x1F
    raise ValueError


def col_of(p):
    return location(p[6] if p[0] == 129 else p[2])[0]


def with_id(p, new):
    q = T(p); q.idx = getattr(p, 'idx', None)
    old = bd_id(p)
    if is_bd(p): q[2] += (new - old) * BD_STRIDE
    elif is_patch(p): q[6] += (new - old) * BD_STRIDE
    elif is_push(p): q[4] = (q[4] & ~0x1F) | new
    return q


def groups_of(ps):
    """[(body, waits)] ; body = non-wait instructions before the wait block."""
    out, body, waits = [], [], []
    for p in ps:
        if p[0] == 128:
            waits.append(p)
        else:
            if waits:
                out.append((body, waits)); body, waits = [], []
            body.append(p)
    out.append((body, waits))
    return out


def hoist(ps, cut=None, skip=(), skipk=None):
    gs = groups_of(ps)
    skip = set(skip); skipk = skipk or {}
    if cut is not None and cut < 0: cut = len(gs) + cut  # groups >= cut are kept in place (not hoisted)
    out = []
    pending = []          # hoisted instructions of the next group, to be emitted before this group's waits
    stats = Counter()
    prev_ids = defaultdict(set)   # col -> ids in flight during previous group
    plan = []
    for gi, (body, waits) in enumerate(gs):
        # renumber this group's BDs per tile
        order = defaultdict(list)           # col -> original ids in first-use order
        for p in body:
            if is_bd(p) or is_patch(p) or is_push(p):
                c = col_of(p) if not is_push(p) else location(p[2])[0]
                i = bd_id(p)
                if i not in order[c]: order[c].append(i)
        newid = {}
        hoisted_ids = defaultdict(set)
        for c, ids in order.items():
            assert len(ids) <= N_BD
            free = [i for i in range(N_BD) if i not in prev_ids[c]] if gi > 0 else []
            take = []
            for orig in ids:
                if free: take.append(free.pop(0))
                else: break
            # hoist only the first len(take) BDs (in first-use order); the rest are renumbered out of the hoisted set
            for orig, n in zip(ids, take): newid[(c, orig)] = n; hoisted_ids[c].add(n)
            rest = [i for i in range(N_BD) if i not in hoisted_ids[c]]
            for orig in ids[len(take):]: newid[(c, orig)] = rest.pop(0)
        hoist_set = {(c, o) for c, ids in order.items() for o in ids[:len([1 for (cc, oo), n in newid.items() if cc == c and n in hoisted_ids[c]])]} if (gi > 0 and (cut is None or gi < cut) and gi not in skip and 'bd' not in skipk.get(gi, ())) else set()
        keep, hoisted = [], []
        for p in body:
            q = p
            if is_bd(p) or is_patch(p) or is_push(p):
                c = col_of(p) if not is_push(p) else location(p[2])[0]
                q = with_id(p, newid[(c, bd_id(p))])
            if gi > 0 and (cut is None or gi < cut) and gi not in skip and 'bd' not in skipk.get(gi, ()) and (is_bd(p) or is_patch(p)):
                c = col_of(p)
                if (c, bd_id(p)) in hoist_set: hoisted.append(q); stats["hoisted_" + kind(p)] += 1; continue
            if gi > 0 and (cut is None or gi < cut) and gi not in skip and 'ctrl' not in skipk.get(gi, ()) and is_ctrl(p):
                hoisted.append(q); stats["hoisted_ctrl"] += 1; continue
            keep.append(q)
        plan.append((hoisted, keep, waits))
        # ids in flight during this group = all ids used by it
        prev_ids = defaultdict(set)
        for c, ids in order.items(): prev_ids[c] = {newid[(c, o)] for o in ids}
    # emit
    for gi, (hoisted, keep, waits) in enumerate(plan):
        nxt = plan[gi + 1][0] if gi + 1 < len(plan) else []
        # group 0 body is emitted entire (keep) ; hoisted of group gi were already emitted before waits of gi-1
        out.extend(keep)
        out.extend(nxt)
        out.extend(waits)
    stats["groups"] = len(gs)
    return out, stats


def bd_is_linear(p):
    w = p[4:12]
    return w[3] == 0 and (w[4] & 0x3FFFFFFF) == 0 and (w[5] & 0xFFFFF) == 0 and w[6] == 0


def check(ps, argsize, allow_untokened=True, qdepth=None):
    """Static proof for one stream. argsize: list of buffer sizes for patch slots (slot i -> argsize[i] if i < len else argsize[0])."""
    inflight = set(); q = {}; cur = {}; ctrl_last = {}; stats = Counter(); maxq = 0; trace = []
    for p in ps:
        if is_bd(p):
            col, _, reg = location(p[2]); b = (reg - BD_BASE) // BD_STRIDE
            assert (col, b) not in inflight, f"BD {b} col{col} rewritten in flight"
            assert len(p) - 4 == 8
            cur[(col, b)] = dict(len=p[4], arg=None, lin=bd_is_linear(p)); stats["bd"] += 1
            stats["bd_nonlinear"] += 0 if bd_is_linear(p) else 1
        elif is_patch(p):
            col = location(p[6])[0]; b = bd_id(p)
            assert 0 <= b < N_BD and (col, b) in cur and (col, b) not in inflight, "patch without live BD"
            assert cur[(col, b)]["arg"] is None
            slot, off = p[8], p[10] | (p[11] << 32)
            sz = argsize[slot] if slot < len(argsize) else argsize[0]
            cur[(col, b)]["arg"] = (slot, off)
            assert off % 4 == 0 and off + cur[(col, b)]["len"] * 4 <= sz, f"DDR range slot{slot} off{off} len{cur[(col, b)]['len']*4} > {sz}"
            stats["patch"] += 1
        elif is_ctrl(p):
            col, _, reg = location(p[2])
            ch = (col, reg)
            active = any(len(q.get((col, d, c), ())) for d, c in ((0, 0), (0, 1), (1, 0), (1, 1)) if QUEUES_BY_CTRL[reg] == (d, c))
            val = (p[4], p[5])
            if active:
                assert ctrl_last.get(ch) == val, "ctrl value changes while channel active"
            ctrl_last.setdefault(ch, val)
            assert ctrl_last[ch] == val, "ctrl value differs from earlier write"
            stats["ctrl"] += 1
        elif is_push(p):
            col, _, reg = location(p[2]); d, c = QUEUES[reg]; b = p[4] & 0x1F
            assert p[4] & 0xFFE0 == 0
            tok = bool(p[4] & 0x80000000); assert tok or allow_untokened
            assert (col, b) in cur and cur[(col, b)]["arg"] is not None, "push of unconfigured BD"
            assert (col, b) not in inflight, "BD pushed while in flight"
            dq = q.setdefault((col, d, c), deque()); assert qdepth is None or len(dq) < qdepth, "queue overflow"; trace.append(len(dq))
            dq.append((b, tok)); inflight.add((col, b)); maxq = max(maxq, len(dq)); stats["push"] += 1
        elif p[0] == 128:
            d = p[2] & 255; col = (p[2] >> 16) & 255; c = (p[3] >> 24) & 255
            dq = q.get((col, d, c)); assert dq, "wait without pending task"
            while True:
                b, tok = dq.popleft(); inflight.discard((col, b))
                if tok: break
                assert dq, "wait covers only untokened"
            stats["wait"] += 1
    assert not inflight, "tasks outstanding at end"
    return dict(ok=True, max_queue=maxq, trace=tuple(trace), **stats)


QUEUES_BY_CTRL = {0x1D200: (0, 0), 0x1D208: (0, 1), 0x1D210: (1, 0), 0x1D218: (1, 1)}


def signature(ps):
    """Per (col, dir, chan): ordered list of pushed (BD words, slot, offset); plus ordered non-hoistable, non-BD instruction stream."""
    cur = {}; chan = defaultdict(list); rest = []
    for p in ps:
        if is_bd(p): cur[(location(p[2])[0], bd_id(p))] = [tuple(p[4:]), None]
        elif is_patch(p): cur[(location(p[6])[0], bd_id(p))][1] = (p[8], p[10], p[11])
        elif is_push(p):
            col, _, reg = location(p[2]); b = bd_id(p)
            chan[(col, QUEUES[reg])].append((cur[(col, b)][0], cur[(col, b)][1], p[4] & ~0x1F))
            rest.append(("push", col, QUEUES[reg]))
        elif is_ctrl(p): pass
        else: rest.append(tuple(p))
    return dict(chan), rest


def wordmap(orig_ps, out):
    """old word index -> new word index (instruction-level permutation; header = 4 words). Requires tagged instructions."""
    starts = []; at = 4
    for p in orig_ps:
        starts.append(at); at += len(p)
    m = {}; at = 4
    for p in out:
        for k in range(len(p)): m[starts[p.idx] + k] = at + k
        at += len(p)
    return m


def head_skipk(ps):
    """R83: the 1-row decode head group (second-to-last group of the single-command decode / bulk-taps stream: RMSNorm | [TapCopy] | head | Sample) keeps its BD writes, DDR patches
    and channel-ctrl writes in place (measured -0.4..-0.55 ms per row in situ, bitwise). Every other group is hoisted as before."""
    ng = len(groups_of(ps))
    assert ng >= 3, 'tail needs at least norm/head/sample groups'
    return {ng - 2: {'bd', 'ctrl'}}
