"""100 package-native cycles; exact candidates/proposals, all scores and KV."""
import argparse,hashlib,json,os,time,statistics
from pathlib import Path
import settings83
settings83.paths()
import numpy as np
from target_drafter import bits,floats
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[];app=None

def error(x,y):return float(np.mean((np.asarray(x,np.float32)-np.asarray(y,np.float32))**2)/max(float(np.mean(np.asarray(y,np.float32)**2)),1e-20))
try:
 from default83 import DefaultRuntime
 app=DefaultRuntime();assert app.mode=='w4-m2',app.fallbacks;m=app.drafter;runtime=app.env.runtime;bundle=m.root
 from gatefiles83 import retain_verified;retain_verified(bundle)
 cases=json.loads((bundle/'PLaMo-vectors/PLaMo-vectors.json').read_text())['cases'];assert len(cases)==100
 for case in cases:
  v=dict(np.load(bundle/'PLaMo-vectors'/case['file']));lang=case['target_language'];assert str(v['target_language'])==lang;m.select_language(lang)
  s=dict(last_taps=v['last_taps'].copy(),position=int(v['position']),layers=[(v['k0'].copy(),v['v0'].copy())]);m.bind_state(s);before=m.confirmed_device_bytes();start=time.perf_counter();got=m.cycle(s,int(v['anchor']));cycle=(time.perf_counter()-start)*1000
  assert np.array_equal(got['candidate_ids'],v['candidate_ids']),'candidate IDs/order'
  assert got['proposal']==int(v['proposal']);assert before==m.confirmed_device_bytes()
  hidden=error(got['hidden'],v['hidden']);logit=error(got['candidate_logits'],v['candidate_logits']);assert max(hidden,logit)<=1e-4
  # With unchanged target arithmetic, the complete full-cycle output is bitwise.
  assert np.array_equal(bits(got['hidden']),bits(v['hidden']))and np.array_equal(bits(got['candidate_logits']),bits(v['candidate_logits']))
  start=time.perf_counter();updated=m.append_confirmed(s,v['commit_taps'],v['commit_tokens']);refresh=(time.perf_counter()-start)*1000
  kv=max(error(z,v[f'updated_{name}0'])for name,z in zip(('k','v'),updated['layers'][0]));assert kv<=1e-4 and updated['position']==int(v['updated_position'])
  bootstrap=None
  if 'bootstrap_taps'in v:
   primed=m.prime(v['bootstrap_taps'],v['bootstrap_tokens']);bootstrap=max(error(primed['layers'][0][j],s['layers'][0][j])for j in(0,1));assert bootstrap<=1e-4 and primed['position']==s['position']
  m.bind_state(s);assert before==m.confirmed_device_bytes();fast=m.propose(s,int(v['anchor']),audit=True);assert fast['proposal']==int(v['proposal'])and before==m.confirmed_device_bytes()
  order={int(t):i for i,t in enumerate(v['candidate_ids'])};expected=np.array([v['candidate_logits'][order[int(t)]]for t in fast['candidate_ids']]);assert len(order)==512 and set(map(int,fast['candidate_ids']))==set(order);assert np.array_equal(bits(fast['candidate_logits']),bits(expected))
  lean=m.propose(s,int(v['anchor']),audit='logits');assert lean['proposal']==fast['proposal']and np.array_equal(bits(lean['candidate_logits']),bits(fast['candidate_logits'])),'R83 lean audit logits';assert before==m.confirmed_device_bytes()
  again=m.append_confirmed(s,v['commit_taps'],v['commit_tokens']);fastkv=max(error(z,v[f'updated_{name}0'])for name,z in zip(('k','v'),again['layers'][0]));assert fastkv<=1e-4
  row=dict(id=case['id'],target_language=lang,result='PASS',proposal=got['proposal'],hidden_relative_mse=hidden,logit_relative_mse=logit,refresh_relative_mse=kv,fast_refresh_relative_mse=fastkv,bootstrap_relative_mse=bootstrap,confirmed_unchanged=True,bitwise_hidden_logits=True,exact_candidate_IDs=True,cycle_ms=cycle,refresh_ms=refresh);rows.append(row);print(json.dumps(row),flush=True)
 assert all(sum(x['target_language']==lang for x in rows)==50 for lang in ('ja','en'))
 report=dict(result='PASS',cycles=100,per_language={lang:50 for lang in ('ja','en')},limit_relative_mse=1e-4,rows=rows,worker=app.primer.ready_record,package_hash=runtime._r48_package_hash,spec_ready_ms=app.spec_ready_ms,D_mean_ms=statistics.mean(x['cycle_ms']+x['refresh_ms']for x in rows),cycle_profile=m.cycle_profile,refresh_profile=m.refresh_profile)
 a.output.write_text(json.dumps(report,indent=2));app.primer.close();print(json.dumps({k:v for k,v in report.items()if k not in ('rows','cycle_profile','refresh_profile','worker')}),flush=True);os._exit(0)
except BaseException:
 import traceback;err=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=err),indent=2));print(err,flush=True)
 if app is not None and app.primer is not None:app.primer.close()
 os._exit(1)
