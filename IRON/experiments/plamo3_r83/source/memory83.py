"""Attribute bounded resident warm-up with two identical 100-prefix passes."""
import argparse,json,os,sys
from pathlib import Path
from settings83 import ROOT,paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resident',type=Path,required=True);a=p.parse_args()
try:
 from default83 import DefaultRuntime
 from memory45 import snapshot
 oracle=json.loads(a.resident.read_text());assert oracle['result']=='PASS' and oracle['prefix_requests']==100
 expected={r['id']:r['tokens']for r in oracle['rows']};meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());paragraph=next(x for x in meta['long_cases']if x['id']=='en-512')['prompt']
 app=DefaultRuntime();assert app.mode=='w4-m2';worker=app.primer.child.pid;rows=[];requests=0
 def measure(label):
  runtime=app.env.runtime;caches=[d['cases'].cache for d in runtime.manifest['instructions'].values()if hasattr(d['cases'],'cache')]
  sizes=sum(sys.getsizeof(c)+sum(sys.getsizeof(k)+sys.getsizeof(v)for k,v in c.items())for c in caches)
  rows.append(dict(label=label,**snapshot(app.primer),graphs=len(runtime._graphs),instruction_BOs=sum(len(g['instructions'])for g in runtime._graphs.values()),instruction_host_bytes=sum(s['host'].nbytes for g in runtime._graphs.values()for s in g['instructions'].values()if s['host']is not None),sub_BOs=len(runtime._view_bos),typed_case_cache_entries=sum(len(c)for c in caches),typed_case_cache_shallow_bytes=sizes,runtime_events=len(app.env.events),profiles={n:len(getattr(app.drafter,n))for n in('cycle_profile','refresh_profile','head_profile','_fast_profile')}));print(json.dumps(rows[-1]),flush=True)
 def check(prompt,cap,wanted,language="ja"):
  global requests
  got=app.generate(prompt,cap,target_language=language);assert got['tokens']==wanted and app.mode=='w4-m2'and app.primer.child.pid==worker;requests+=1
 def fixtures():
  for i,prompt in enumerate(meta['prompts']):check(prompt,256,expected[f'fixtures-0-translation-{i}'],('en'if i<2 else'ja'))
  check(paragraph,1024,expected['fixtures-0-paragraph'])
 measure('initialized');fixtures();measure('warm-five')
 for repeat in(0,1):
  for pi,prompt in enumerate(meta['prompts']):
   for j in range(25):check(prompt+meta['teacher_tokens'][pi*25:pi*25+j],25,expected[f'prefix-{pi}-{j}'],('en'if pi<2 else'ja'))
  measure(f'prefix-pass-{repeat}')
 fixtures();measure('repeat-five')
 report=dict(result='PASS',pid=os.getpid(),worker_pid=worker,requests=requests,snapshots=rows,all_tokens_equal_primary_serial_oracles=True,scope='additional cache/RSS attribution; same default process, no per-request gc or reload, no growing result history; two identical100-prefix passes; no timing claims substituted for the three-process protocol')
 app.close();a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result='PASS',requests=requests)),flush=True)
except BaseException:
 import traceback;error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));print(error,flush=True);sys.exit(1)
