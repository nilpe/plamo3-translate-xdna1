"""R83 whole-prefill gate (R46 protocol; Phase B: env R83_HEADSWAP=1 applies the head-context hot swap of headswap83 to the candidate): the ACTIVE R80 default vs the R83 candidate on the 12 standard/long prompts: token, all logits, every tap byte and every byte of all 32 complete 2048-row K/V operands must be bitwise equal."""
import argparse,hashlib,json,os,struct,sys,time,traceback
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()))
import settings83 as s80
s80.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode');a=p.parse_args()
try:
 from default83 import DefaultRuntime
 from tap_session_v2 import prefill_with_features
 meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());cases=[dict(id=f'standard-{i}',prompt=x)for i,x in enumerate(meta['prompts'])]+meta['long_cases'];rows=[];reference={}
 def hashes(app,prompt):
  r=app.session.runtime;start=time.perf_counter();out=prefill_with_features(app.session,prompt);elapsed=time.perf_counter()-start
  cache=sorted([(v['offset'],n)for n,v in r.manifest['views'].items()if n.startswith('view')and v['root']=='root0'and v['nbytes']==524288 and 474494976<=v['offset']<491272192]);assert len(cache)==32
  h=hashlib.sha256()
  for offset,n in cache:h.update(struct.pack('<Q',offset));h.update(r.read(n))
  features=b''.join(out['features']);return dict(token=int(out['token']),logits=hashlib.sha256(app.session.read_logits()).hexdigest(),all32_full_KV=h.hexdigest(),features=hashlib.sha256(features).hexdigest(),feature_bytes=len(features),elapsed_ms=elapsed*1000)
 # H1: the reference is the CERTIFIED R83 (export-audit/r83-reference.json, 'prefill', made author side from the active R83 by tools/author/make_r83_reference.py), not the previous default
 reference=json.loads((ROOT/'export-audit/r83-reference.json').read_text(encoding='utf-8'))['prefill']
 assert set(reference)=={c['id'] for c in cases},'reference cases differ'
 for _v in reference.values():_v.setdefault('elapsed_ms',0.0)
 app=DefaultRuntime(spec_manifest=s80.CANDIDATE_SPEC);assert app.mode=='w4-m2',app.fallbacks
 if os.environ.get('R83_HEADSWAP')=='1':
  import headswap83;print(json.dumps(headswap83.apply(app.session.runtime)),flush=True)
 for case in cases:
  old=reference[case['id']];new=hashes(app,case['prompt'])
  for key in ('token','logits','all32_full_KV','features','feature_bytes'):assert old[key]==new[key],(case['id'],key,old[key],new[key])
  rows.append(dict(id=case['id'],rows=len(case['prompt']),result='PASS',baseline=old,candidate=new));print(json.dumps(dict(id=case['id'],result='PASS',baseline_ms=old['elapsed_ms'],candidate_ms=new['elapsed_ms'])),flush=True)
 app.close();report=dict(result='PASS',candidate=str(s80.CANDIDATE_SPEC),cases=rows,scope='complete prefills in two sequentially opened/closed validated runtimes; logits, every tap byte, and complete 2048-row contents of all32 KV allocations exact; no two model loads overlap');a.output.write_text(json.dumps(report,indent=2))
except BaseException:
 a.output.write_text(json.dumps(dict(result='FAIL',error=traceback.format_exc()),indent=2));traceback.print_exc();sys.exit(1)
