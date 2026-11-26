"""Native C1 vs counterfactual M2 EOS root: logits, taps, committed target/drafter KV."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3'
for p in(APP,ROOT/'lib/r36',ROOT/'lib/r37',ROOT/'lib/r41',ROOT/'lib/r42',Path(__file__).parent):sys.path.insert(0,str(p))
import numpy as np
from draft43 import make_direct
from spec_loop import features,TimedFeatures
from tap_session_v2 import prefill_with_features,decode_with_features
from primer import Primer
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args();records=[];primer=None
try:
 from experimental_runtime import initialize
 primer=Primer(ROOT/__import__('json').loads((ROOT/'release.json').read_text(encoding='utf-8-sig'))['drafter_package']);env=initialize(Path(os.environ['R42_MANIFEST']));s=env['session'];r=s.runtime;r._r42_primer=primer;bundle=Path(r.manifest['r40_cycle']['bundle']);Native,module=make_direct(bundle,r);m=Native(bundle);meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());para=json.loads((ROOT/'gate-data/long-translation.json').read_text());cases=[]
 for i,prompt in enumerate(meta['prompts']):
  got=s.prefill(prompt);out=[got['token']]
  while out[-1]not in s.model['eos']:out.append(s.decode(out[-1])['token'])
  cases += [(f'short{i}-reject',prompt),(f'short{i}-accept',prompt+out[:-2])]
 cases += [('paragraph-tail',next(c for c in meta['long_cases']if c['id']=='en-512')['prompt']+para['generated_tokens'][:-2])]
 kv={v['offset']:n for n,v in r.manifest['views'].items()if n.startswith('view')and v['root']=='root0'and v['nbytes']==524288 and 474494976<=v['offset']<491272192};kv=[kv[k]for k in sorted(kv)];assert len(kv)==32
 def snap():return [r.read(n)for n in kv]
 for name,prompt in cases:
  pf=prefill_with_features(s,prompt);position=s.length;anchor=pf['token'];state=m.prime(features(pf['features']),prompt);base=snap();verifier=TimedFeatures(s)
  for eos in(1,16):
   results=[]
   for mode in('M2-discard-row1','C1'):
    verifier.discard()
    for n,b in zip(kv,base):r.write(n,b)
    s.length=position;m.bind_state(state);before=m.confirmed_device_bytes();m.cycle(state,anchor);assert before==m.confirmed_device_bytes()
    if mode=='C1':got=decode_with_features(s,anchor);token=got['token'];tap=got['features'];logits=s.read_logits()
    else:
     got=verifier.stage([anchor,eos]);token=got['greedy'][0];tap=got['features'][0];logits=verifier.logits()[0];verifier.commit(0)
    updated=m.append_confirmed(state,features([tap]),[anchor]);active=[x[:s.length*256]for x in snap()];dk,dv=updated['layers'][0];assert s.length==position+1 and updated['position']==position;results.append(dict(token=token,tap=tap,logits=logits,target_KV=active,draft_K=dk.copy(),draft_V=dv.copy()))
   left,right=results;assert left['token']==right['token']and left['tap']==right['tap']and left['logits']==right['logits'];assert left['target_KV']==right['target_KV'];assert np.array_equal(left['draft_K'],right['draft_K'])and np.array_equal(left['draft_V'],right['draft_V']);records.append(dict(id=name,EOS_proposal=eos,accepted=left['token']==eos,token=left['token'],position=position,root_logits_bitwise=True,taps_bitwise=True,target_committed_KV_bitwise=True,drafter_refresh_KV_bitwise=True,target_cursor=position+1,drafter_cursor=position));print(json.dumps(records[-1]),flush=True)
 primer.close();report=dict(result='PASS',cases=len(records),rows=records,scope='forced EOS equivalence probe only, not a generation speed claim; actual native C1 logits/taps and all32 committed target cache operands equal M2 root with commit(0); native drafter root reuse and full active refresh KV equal; both EOS IDs, accepting and rejecting prefixes');a.output.write_text(json.dumps(report,indent=2));sys.stdout.flush();os._exit(0)
except BaseException:
 import traceback;err=traceback.format_exc();print(err,flush=True)
 if primer:primer.close()
 a.output.write_text(json.dumps(dict(result='FAIL',rows=records,error=err),indent=2));sys.stdout.flush();os._exit(1)
