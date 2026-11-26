"""R44 paired five-fixture timing, through the actual R45 default controller."""
import argparse,json,os,sys,time
from pathlib import Path
from settings83 import ROOT,paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--idle',type=int,default=0);a=p.parse_args()
try:
 from default83 import DefaultRuntime
 from readiness import process_age_ms
 app=DefaultRuntime();user_ready_ms=process_age_ms();kw0=app.keepwarm;aff0=None
 if kw0 is not None:
  import keepwarm83;aff0=keepwarm83.thread_mask()  # only when keep-warm is active: the off arm must make no affinity call at all
 assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction';assert app.mode=='w4-m2',app.fallbacks;(os.environ.get('R83_PACKAGE')=='v4C')or __import__('switchwin83').assert_declared_window(app.startup)
 from loop_gate import serial_generate
 from memory45 import snapshot,Sampler
 sampler=Sampler(lambda:app.primer);sampler.start()
 meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());para=json.loads((ROOT/'gate-data/long-translation.json').read_text());cases=[dict(id=f'translation-{i}',prompt=x,cap=256,target_language=('en'if i<2 else'ja'))for i,x in enumerate(meta['prompts'])];cases.append(dict(id='paragraph-515',prompt=next(x for x in meta['long_cases']if x['id']=='en-512')['prompt'],cap=1024,target_language='ja',expected=para['generated_tokens']));rows=[]
 for case in cases:
  baseline=serial_generate(app.session,case['prompt'],case['cap']);tap=serial_generate(app.session,case['prompt'],case['cap'],True);delivered=[];cpu0=time.process_time();got=app.generate(case['prompt'],case['cap'],emit=delivered.append,target_language=case['target_language']);cpu_generate_s=time.process_time()-cpu0;assert app.mode=='w4-m2',app.fallbacks;assert got['tokens']==baseline['tokens']==tap['tokens']==delivered
  if'expected'in case:assert got['tokens']==case['expected']
  row=dict(id=case['id'],result='PASS',cpu_generate_s=cpu_generate_s,prompt_tokens=len(case['prompt']),serial=baseline,serial_taps=tap,speculative=got,cycle_profile=list(app.drafter.cycle_profile),fast_profile=list(app.drafter._fast_profile),refresh_profile=list(app.drafter.refresh_profile),head_profile=list(app.drafter.head_profile),worker_records=list(app.primer.records),memory=snapshot(app.primer));rows.append(row);print(json.dumps(dict(id=case['id'],result='PASS',serial_s=baseline['wall_s'],serial_taps_s=tap['wall_s'],default_s=got['wall_s'],ttft_ms=got['ttft_s']*1000,M2=sum(x['mode']=='M2'for x in got['cycles']),EOS_C1=got['eos_proposal_cycles'])),flush=True)
 from readiness import process_age_ms
 report=dict(sampled_memory=sampler.finish(),package=str(app.drafter.root),user_ready_ms=user_ready_ms,pid=os.getpid(),result='PASS',idle_seconds=a.idle,mode=app.mode,spec_ready_ms=app.spec_ready_ms,process_to_spec_ready_ms=app.spec_ready_ms,startup=app.startup,cases=rows,fallbacks=app.fallbacks)
 report['hybrid']=(dict(app.hybrid.stats) if getattr(app,'hybrid',None) is not None else None);report['hybrid_installed']=getattr(app,'hybrid',None) is not None;report['process_cpu_s']=time.process_time()
 kw=app.keepwarm;report['keep_warm_mode']=getattr(app,'keep_warm_mode',None);report['keep_warm']=None
 if kw is not None:assert keepwarm83.thread_mask()==aff0,'decode thread affinity not restored after generate';report['keep_warm']=kw.stats();report['main_cpu_s']=keepwarm83._process_cpu_s(keepwarm83.k32())
 app.close()
 if kw is not None:report['keep_warm']=kw.stats()
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result='PASS',spec_ready_ms=report['spec_ready_ms'])),flush=True)
except BaseException:
 import traceback;error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));print(error,flush=True);sys.exit(1)
