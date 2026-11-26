"""R83 static proof library (check77/check79 lineage): single-command transform proofs, drafter hoist proofs and the mutant batteries.
Functions: prove_single(pls, steps, V, RS, hdr) -> (raw, info); prove_drafter(raw, argsizes_list) -> (hoisted_raw, info); mutants_single(pls, steps, V, RS); mutants_drafter(raw, asz).
Mutants single: S1 slot swapped, S2 view offset dropped, S3 patch beyond root, S4 BD length changed, S5 wait removed at a boundary, S6 push across wait, S7 BD rewritten in flight, S8 ctrl value changed, S9 push pulled over a boundary wait.
Mutants drafter: D1 DDR out of bounds, D2 push before wait, D3 ctrl value changed, D4 BD length changed, D5 wait removed. Every mutant must be rejected."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoist83 as H, singlecmd83 as S

HEAD_UNHOIST = ('decode', 'r37_decode_bulk_taps')   # graphs whose single command keeps the head group un-hoisted (R83)

def prove_single(pls, steps, V, RS, hdr, unhoist_head=False):
    slots, slot_of, ps2, osc, orest, tr = S.build(pls, steps, V, False)
    S.finish(slots, ps2, osc, orest, tr, RS, False, hdr)
    raw, b, st = S.finish(slots, ps2, osc, orest, tr, RS, True, hdr, unhoist_head=unhoist_head)
    return raw, dict(slots=slots, bd=b['bd'], hoisted_bd=st['hoisted_bd_write'], max_queue=b['max_queue'], insts=len(ps2), nonlinear=b['bd_nonlinear'])

def prove_drafter(raw, argsizes):
    ps = H.parse(raw)
    for asz in argsizes: a = H.check(ps, list(asz))
    o, st = H.hoist(ps)
    for asz in argsizes:
        b = H.check(o, list(asz)); assert a['trace'] == b['trace']
    assert H.signature(o) == H.signature(ps) and len(o) == len(ps)
    return H.encode_h(H.header_of(raw), o), dict(bd=a['bd'], hoisted_bd=st['hoisted_bd_write'], insts=len(ps), nonlinear=a['bd_nonlinear'])

def mutants_single(pls, steps, V, RS):
    slots, slot_of, ps2, osc, orest, tr = S.build(pls, steps, V, False); hdr = H.header_of(pls[0]); res = {}
    nblocks = len(steps)
    def run(name, pre=None, post=None, hoist=True):
        p2 = copy.deepcopy(ps2)
        if pre: pre(p2)
        try:
            S.finish(slots, p2, osc, orest, tr, RS, hoist, hdr, mut=post); res[name] = 'NOT CAUGHT'
        except AssertionError as e: res[name] = 'caught(' + str(e)[:50] + ')'
        except Exception as e: res[name] = 'caught(' + type(e).__name__ + ')'
    patches = [i for i, p in enumerate(ps2) if H.is_patch(p)]
    def m_slot(p2):
        for i in patches:
            if len(slots) > 1: p2[i][8] = (p2[i][8] + 1) % len(slots); return
    def m_norebase(p2):
        k = next(i for i in patches if p2[i][10] | (p2[i][11] << 32) > 0); p2[k][10] = 0; p2[k][11] = 0
    def m_oob(p2): k = patches[0]; p2[k][10] = 0xFFFFFFF0; p2[k][11] = 0x7
    def m_bdlen(p2): k = next(i for i, p in enumerate(p2) if H.is_bd(p)); p2[k][4] += 4
    def post_wait(out): out = list(out); k = [i for i, p in enumerate(out) if p[0] == 128][nblocks * 2]; del out[k]; return out
    def post_push(out):
        out = copy.deepcopy(list(out)); w = [i for i, p in enumerate(out) if p[0] == 128][3]
        j = next(i for i in range(w, len(out)) if H.is_push(out[i])); out.insert(w, out.pop(j)); return out
    def post_inflight(out):
        out = copy.deepcopy(list(out)); first = next(i for i, p in enumerate(out) if H.is_push(p)); col = H.location(out[first][2])[0]; bid = H.bd_id(out[first])
        j = next(i for i in range(first + 1, len(out)) if H.is_bd(out[i]) and H.col_of(out[i]) == col); out[j] = H.with_id(out[j], bid); return out
    def post_ctrl(out): out = copy.deepcopy(list(out)); k = next(i for i, p in enumerate(out) if H.is_ctrl(p)); out[k][4] ^= 0x100; return out
    def post_boundary(out):
        out = copy.deepcopy(list(out)); ws = [i for i, p in enumerate(out) if p[0] == 128]
        k = max(i for i in ws[: len(ws) // 2] if out[i + 1][0] != 128); j = next(i for i in range(k + 1, len(out)) if H.is_push(out[i])); out.insert(k - 3, out.pop(j)); return out
    run('S1_slot_swapped', pre=m_slot, hoist=False); run('S2_view_offset_dropped', pre=m_norebase, hoist=False); run('S3_patch_beyond_root', pre=m_oob, hoist=False); run('S4_bd_length_changed', pre=m_bdlen, hoist=False)
    run('S5_wait_removed_at_boundary', post=post_wait); run('S6_push_across_wait', post=post_push); run('S7_bd_rewritten_in_flight', post=post_inflight); run('S8_ctrl_value_changed', post=post_ctrl); run('S9_push_pulled_over_boundary_wait', post=post_boundary)
    return res

def mutants_drafter(raw, asz):
    ps = H.parse(raw); o, st = H.hoist(ps); res = {}
    def t(name, mut):
        m = mut(copy.deepcopy(o))
        try:
            b = H.check(m, asz); ok = b['trace'] != H.check(ps, asz)['trace'] or H.signature(m) != H.signature(ps) or len(m) != len(ps)
            res[name] = 'caught(equivalence)' if ok else 'NOT CAUGHT'
        except AssertionError as e: res[name] = 'caught(' + str(e)[:50] + ')'
        except Exception as e: res[name] = 'caught(' + type(e).__name__ + ')'
    def a(out): k = next(i for i, p in enumerate(out) if H.is_patch(p)); out[k][10] = 0x7FFFFFF0; return out
    def b(out):
        w = next(i for i, p in enumerate(out) if p[0] == 128); j = next(i for i in range(w, len(out)) if H.is_push(out[i])); out.insert(w, out.pop(j)); return out
    def c(out): k = next(i for i, p in enumerate(out) if H.is_ctrl(p)); out[k][4] ^= 0x100; return out
    def d(out): k = next(i for i, p in enumerate(out) if H.is_bd(p)); out[k][4] += 4; return out
    def e(out): k = next(i for i, p in enumerate(out) if p[0] == 128); del out[k]; return out
    t('D1_ddr_out_of_bounds', a); t('D2_push_before_wait', b); t('D3_ctrl_value_changed', c); t('D4_bd_length_changed', d); t('D5_wait_removed', e)
    return res
