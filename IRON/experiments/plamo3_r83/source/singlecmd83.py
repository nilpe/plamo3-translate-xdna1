"""R79: whole-token single-command transform (static). concat(17 block payloads) with DDR-patch slots re-based onto the ROOT buffers
(slot -> root slot, offset += view offset), optional global hoist on the concatenated stream (G0 of block b hoisted into block b-1's wait block).
Proof (static): resolved (BD words, root, absolute offset) per channel list equal to the original concatenation; non-BD instruction sequence equal;
queue-occupancy trace equal; BD lifetime/range checks on the new stream (H.check with root sizes)."""
import sys, os
from array import array
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoist83 as H
from collections import defaultdict


def view_of(steps_args, V, slot):
    """steps_args: list of view names for the step; slot out of range -> arg 0 (the runtime pads with bufs[0])."""
    a = steps_args[slot if slot < len(steps_args) else 0]
    return V[a]['root'], V[a].get('offset', 0), V[a]['nbytes']


def root_slots(steps, V):
    order = []
    for s in steps:
        for a in s['args']:
            r = V[a]['root']
            if r not in order: order.append(r)
    return order


def rebase(ps, step, V, slot_of):
    out = []
    for p in ps:
        if H.is_patch(p):
            q = H.T(p); q.idx = getattr(p, 'idx', None)
            root, voff, _ = view_of(step['args'], V, p[8])
            off = (p[10] | (p[11] << 32)) + voff
            q[8] = slot_of[root]; q[10] = off & 0xFFFFFFFF; q[11] = off >> 32
            out.append(q)
        else:
            out.append(p)
    return out


def sig_resolved(ps, resolve):
    """per (col,dir,chan): ordered (BD words, root, abs offset, token bits); resolve(slot, off) -> (root, abs)"""
    cur = {}; chan = defaultdict(list); rest = []
    for p in ps:
        if H.is_bd(p): cur[(H.location(p[2])[0], H.bd_id(p))] = [tuple(p[4:]), None]
        elif H.is_patch(p): cur[(H.location(p[6])[0], H.bd_id(p))][1] = resolve(p[8], p[10] | (p[11] << 32))
        elif H.is_push(p):
            col, _, reg = H.location(p[2]); b = H.bd_id(p)
            chan[(col, H.QUEUES[reg])].append((cur[(col, b)][0], cur[(col, b)][1], p[4] & ~0x1F))
            rest.append(("push", col, H.QUEUES[reg]))
        elif H.is_ctrl(p): pass
        else: rest.append(tuple(p))
    return dict(chan), rest


def build(payloads, steps, V, do_hoist):
    """returns dict(raw=bytes of the single command, slots=[root names], proof...). Raises AssertionError on any failed proof."""
    slots = root_slots(steps, V); slot_of = {r: i for i, r in enumerate(slots)}
    rsize = None
    parts_all = []; orig_sig_chan = defaultdict(list); orig_rest = []; trace_orig = []
    for i, (raw, st) in enumerate(zip(payloads, steps)):
        ps = H.parse(raw)
        argsize = [V[a]['nbytes'] for a in st['args']]
        a = H.check(ps, argsize)
        trace_orig.extend(a['trace'])
        ch, rest = sig_resolved(ps, lambda s, o, st=st: (lambda r: (r[0], r[1] + o))(view_of(st['args'], V, s)))
        for k, v in ch.items(): orig_sig_chan[k].extend(v)
        orig_rest.extend(rest)
        parts_all.extend(rebase(H.tag(ps), st, V, slot_of))
    n_orig = len(parts_all)
    ps2 = H.tag(parts_all)
    return slots, slot_of, ps2, orig_sig_chan, orig_rest, trace_orig


def finish(slots, ps2, orig_sig_chan, orig_rest, trace_orig, rootsize, do_hoist, hdr, mut=None, unhoist_head=False):
    argsize = [rootsize[r] for r in slots]
    out = ps2
    if do_hoist:
        out, st = H.hoist(ps2, skipk=H.head_skipk(ps2) if unhoist_head else None)
    else:
        st = {}
    if mut is not None: out = mut(out)
    b = H.check(out, argsize)
    assert b['trace'] == tuple(trace_orig), 'queue occupancy trace differs'
    ch, rest = sig_resolved(out, lambda s, o: (slots[s], o))
    assert dict(ch) == dict(orig_sig_chan), 'resolved data movement differs'
    assert rest == orig_rest, 'non-BD instruction sequence differs'
    assert len(out) == len(ps2)
    return H.encode_h(hdr, out), b, st
