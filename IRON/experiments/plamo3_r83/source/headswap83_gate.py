# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Release head-swap-gate (H1 form). R83 compared the candidate with the previous default; in the release the reference is the CERTIFIED R83 itself.
`export-audit/r83-reference.json` is produced once, author side, from the ACTIVE R83 install (tools/author/make_r83_reference.py): every context PDI sha256, EVERY graph (steps: context, instruction family, argument views),
and for the six head families (inst23..inst28) params + the sha256 of every decoded instruction stream. This gate rebuilds the same snapshot from the INSTALLED release (DefaultRuntime on the shipped certificate)
and requires exact equality: all graphs (decode, bulk, verify2, prefill, drafter... not only the unchanged ones), names, contexts, argument views and slot order pinned, every head stream byte-identical, context1 = b5633dde.
Stronger than the R83 gate (which tolerated the three rewritten graphs by name only). usage: headswap83_gate.py --output FILE.json"""
import argparse, hashlib, json, sys, traceback
from pathlib import Path
import settings83 as s
s.paths()
from plamo_root import ROOT
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--mode'); a = p.parse_args()
FAM = ('inst23', 'inst24', 'inst25', 'inst26', 'inst27', 'inst28')
def snapshot(app, inspect_xclbin):
    r = app.session.runtime; m = r.manifest
    out = dict(pdi={k: inspect_xclbin(Path(m['artifacts'][v['artifact']]['path']))['pdis'][0]['sha256'] for k, v in m['contexts'].items()},
               graphs={n: [[x['context'], x['instructions'], list(x['args'])] for x in g['steps']] for n, g in m['graphs'].items()}, fam={})
    for f in FAM:
        d = m['instructions'][f]
        out['fam'][f] = dict(params=list(d['params']), cases={k: hashlib.sha256(bytes(r._instruction_bytes(part))).hexdigest() for k, part in d['cases'].items()})
    return out
try:
    from default83 import DefaultRuntime
    from pdi_inventory import inspect_xclbin
    ref = json.loads((ROOT / 'export-audit/r83-reference.json').read_text(encoding='utf-8'))
    app = DefaultRuntime(); assert app.mode == 'w4-m2', app.fallbacks
    try: new = snapshot(app, inspect_xclbin)
    finally: app.close()
    new = json.loads(json.dumps(new))
    assert new['pdi']['context1'].startswith('b5633dde'), new['pdi']
    assert new['pdi'] == ref['pdi'], 'a context image differs from the certified R83'
    assert set(new['graphs']) == set(ref['graphs']), 'graph set differs'
    bad = [g for g in ref['graphs'] if new['graphs'][g] != ref['graphs'][g]]
    assert not bad, ('graph steps (context / family / argument views / slot order) differ from R83', bad[:5])
    rep = {}
    for f in FAM:
        o, n = ref['fam'][f], new['fam'][f]
        assert o['params'] == n['params'] and set(o['cases']) == set(n['cases']), f
        diff = [k for k in o['cases'] if o['cases'][k] != n['cases'][k]]
        assert not diff, (f, 'head stream bytes differ from R83 in %d cases' % len(diff), diff[:3])
        rep[f] = dict(cases=len(o['cases']))
    # mutants of the comparison itself
    mut = {}
    r2 = json.loads(json.dumps(ref)); g0 = sorted(r2['graphs'])[0]; r2['graphs'][g0][0][2] = r2['graphs'][g0][0][2][::-1] + ['x']
    mut['argument_view_changed'] = r2['graphs'][g0] != new['graphs'][g0]
    r3 = json.loads(json.dumps(ref)); k0 = sorted(r3['fam']['inst26']['cases'])[0]; r3['fam']['inst26']['cases'][k0] = '0' * 64
    mut['head_stream_byte_changed'] = r3['fam']['inst26']['cases'][k0] != new['fam']['inst26']['cases'][k0]
    r4 = json.loads(json.dumps(ref)); r4['pdi']['context1'] = 'afd519d8' + '0' * 56
    mut['context1_pdi_changed'] = r4['pdi'] != new['pdi']
    assert all(mut.values()), mut
    a.output.write_text(json.dumps(dict(result='PASS', pdi=new['pdi'], graphs=len(new['graphs']), families=rep, mutants=mut), indent=1)); print(json.dumps(dict(result='PASS', graphs=len(new['graphs']), families=rep)), flush=True)
except BaseException:
    a.output.write_text(json.dumps(dict(result='FAIL', error=traceback.format_exc()), indent=2)); traceback.print_exc(); sys.exit(1)
