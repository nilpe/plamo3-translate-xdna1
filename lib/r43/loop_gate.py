"""End-to-end identity, callback latency and paired serial-tap cost."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3'
for x in(APP,ROOT/'lib/r36',ROOT/'lib/r37',ROOT/'lib/r41',ROOT/'lib/r42',Path(__file__).parent):sys.path.insert(0,str(x))

def serial_generate(s,prompt,limit,taps=False):
 from tap_session_v2 import prefill_with_features,decode_with_features
 records=[];start=time.perf_counter();got=prefill_with_features(s,prompt)if taps else s.prefill(prompt);ttft=time.perf_counter()-start;tokens=[got['token']];at=time.perf_counter()
 while len(tokens)<limit and tokens[-1]not in s.model['eos']and s.length<s.max_seq:
  row=decode_with_features(s,tokens[-1])if taps else s.decode(tokens[-1]);tokens.append(row['token']);records.append(row.get('stats',{}))
 elapsed=time.perf_counter()-at;return dict(prefill_stats=got.get('stats'),stats=records,tokens=tokens,ttft_s=ttft,decode_s=elapsed,wall_s=time.perf_counter()-start,decode_ms_per_token=elapsed*1000/(len(tokens)-1)if len(tokens)>1 else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='PLaMo-D7-M2');p.add_argument('--mode',choices=['smoke','translations','teacher-prefixes'],default='translations');p.add_argument('--idle',type=int,default=0);a=p.parse_args()
 from readiness import process_age_ms
 from primer import Primer
 at=time.perf_counter();primer=Primer(ROOT/__import__('json').loads((ROOT/'release.json').read_text(encoding='utf-8-sig'))['drafter_package']);worker_launch_ms=(time.perf_counter()-at)*1000;spawn_age=process_age_ms()
 try:
  at=time.perf_counter();import numpy as np;np_ms=(time.perf_counter()-at)*1000
  from experimental_runtime import initialize
  manifest=Path(os.environ.get('R42_MANIFEST',str(ROOT/'plamo3-verify-r42/final/PLaMo-D7-M2/manifest.json')));at=time.perf_counter();env=initialize(manifest);init_ms=(time.perf_counter()-at)*1000;s=env['session'];s.runtime._r42_primer=primer;bundle=Path(s.runtime.manifest['r40_cycle']['bundle']);at=time.perf_counter();from draft43 import make_direct;Native,module=make_direct(bundle,s.runtime);factory_ms=(time.perf_counter()-at)*1000;at=time.perf_counter();drafter=Native(bundle);constructor_ms=(time.perf_counter()-at)*1000;at=time.perf_counter();primer.ready();ready_wait_ms=(time.perf_counter()-at)*1000;ready_ms=process_age_ms()
  from loop43 import generate
  meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());cases=[]
  if a.mode=='teacher-prefixes':
   for pi,prompt in enumerate(meta['prompts']):
    for row in range(25):cases.append(dict(id=f'standard-{pi}-row{row}',prompt=prompt+meta['teacher_tokens'][pi*25:pi*25+row],cap=25))
  else:
   cases=[dict(id=f'translation-{i}',prompt=x,cap=256)for i,x in enumerate(meta['prompts'])]
   if a.mode=='translations':
    para=json.loads((ROOT/'gate-data/long-translation.json').read_text());assert len(para['generated_tokens'])==595;prompt=next(x for x in meta['long_cases']if x['id']=='en-512')['prompt'];cases.append(dict(id='paragraph-515',prompt=prompt,cap=1024,expected=para['generated_tokens']))
   if a.mode=='smoke':cases=cases[:1]
  rows=[]
  for c in cases:
   assert primer.pending is None;baseline=serial_generate(s,c['prompt'],c['cap']);tap=serial_generate(s,c['prompt'],c['cap'],True)if a.mode!='teacher-prefixes'else None
   if tap is not None:assert tap['tokens']==baseline['tokens']
   if'expected'in c:assert baseline['tokens']==c['expected']
   delivered=[];got=generate(s,drafter,c['prompt'],c['cap'],emit=lambda token:delivered.append(token));equal=got['tokens']==baseline['tokens']==delivered;got['worker_quiesce_ms']=primer.quiesce();row=dict(id=c['id'],prompt_tokens=len(c['prompt']),result='PASS'if equal else'FAIL',serial=baseline,serial_taps=tap,speculative=got);rows.append(row);print(json.dumps(dict(id=c['id'],result=row['result'],serial_wall=baseline['wall_s'],tap_wall=tap['wall_s']if tap else None,spec_wall=got['wall_s'],serial_phase=got['serial_phase_rows'],M2_cycles=sum(x['mode']=='M2'for x in got['cycles']),EOS_C1_cycles=got['eos_proposal_cycles'],ttft_ms=got['ttft_s']*1000,catchup_ms=got['switch']['elapsed_ms']if got['switch']else None)),flush=True);assert equal,row['id']
  report=dict(result='PASS',model=a.model,mode=a.mode,bootstrap_policy=primer.ready_record['norm_policy'],idle_seconds=a.idle,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),process_to_spec_ready_ms=ready_ms,runtime_setup_s=env['setup_s'],runtime_startup_events=env.get('startup_events'),startup=dict(worker_launch_ms=worker_launch_ms,worker_spawn_age_ms=spawn_age,numpy_import_ms=np_ms,initialize_ms=init_ms,factory_ms=factory_ms,constructor_ms=constructor_ms,ready_prepare_ms=drafter.ready_prepare_ms,worker_ready_wait_ms=ready_wait_ms,worker=primer.ready_record,worker_start_to_ready_ms=(primer.ready_record['ready_ns']-primer.started_ns)/1e6),cases=rows,cycle_profile=drafter.cycle_profile,refresh_profile=drafter.refresh_profile,head_profile=drafter.head_profile,fast_profile=drafter._fast_profile,worker_records=primer.records,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),loop_sha256=hashlib.sha256(Path(__file__).with_name('loop43.py').read_bytes()).hexdigest(),scope='fixed first-ready token boundary; EOS cancels unused prime, no M2 then; wall/TTFT use real token callbacks; cancellation quiesce separately reported; all serial/tap/spec IDs compared')
  a.output.write_text(json.dumps(report,indent=2));primer.close();return 0
 except BaseException:
  import traceback;error=traceback.format_exc();print(error,flush=True);a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));primer.close();return 1
if __name__=='__main__':
 code=main();sys.stdout.flush();sys.stderr.flush();os._exit(code)
