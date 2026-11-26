"""Instrument the actual public entry with the unchanged 5-token/10% judge."""
import argparse,contextlib,io,json,os,runpy,sys,time
from pathlib import Path
from settings83 import ROOT,APP,paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
try:
 from readiness import process_age_ms
 from default83 import DefaultRuntime
 import startup_r20,startup_probe
 original=DefaultRuntime.generate;measured={}
 def instrumented(self,prompt,limit=256,emit=None,**kwargs):
  assert self.mode=='w4-m2',self.fallbacks;assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction';import switchwin83;switchwin83.assert_declared_window(self.startup)
  assert prompt==self.session.model['prompts'][3]
  emissions=[];_now_ms=time.time()*1000;user_ready_ms=process_age_ms();_proc_start_epoch_ms=_now_ms-user_ready_ms
  def publish(token):
   emissions.append(dict(token=token,end_ms=process_age_ms()))
   if emit is not None:emit(token)
  got=original(self,prompt,limit,emit=publish,**kwargs)
  assert self.mode=='w4-m2',self.fallbacks
  samples=[dict(token=i,start_ms=x['end_ms'],end_ms=y['end_ms'],wall_ms=y['end_ms']-x['end_ms'])for i,(x,y)in enumerate(zip(emissions,emissions[1:]),1)]
  steady=startup_probe.steady_state(samples)
  from loop_gate import serial_generate
  baseline=serial_generate(self.session,prompt,limit)
  assert got['tokens']==baseline['tokens']
  measured.update(pid=os.getpid(),proc_start_epoch_ms=_proc_start_epoch_ms,result='PASS',mode=self.mode,prompt_tokens=len(prompt),spec_ready_ms=self.spec_ready_ms,user_ready_ms=user_ready_ms,startup=self.startup,steady_state=steady,steady_reached_within_1s=steady['reached_ms']is not None and steady['reached_ms']<=1000,steady_confirmed_within_1s=steady['confirmed_ms']is not None and steady['confirmed_ms']<=1000,decode_samples=samples,default=got,token_identity=True)
  return got
 DefaultRuntime.generate=instrumented
 text,source,target=startup_r20.PROMPTS[3]
 entry=Path(__file__).with_name('cli83.py');sys.argv=[str(entry),text,'--source-lang',source,'--target-lang',target,'--mode','w4-m2','--num-tokens','25','--json'];capture=io.StringIO()
 with contextlib.redirect_stdout(capture):
  _prev_exit=os.environ.get('R77_EXIT');os.environ['R77_EXIT']='off'   # R83: this harness runs the CLI IN-PROCESS; the fast exit must never end it (also excluded by cli83.real_process_entry: runpy frame)
  try:runpy.run_path(str(entry),run_name='__main__')
  except SystemExit as exit:
   assert exit.code in(None,0),exit.code
  finally:
   if _prev_exit is None:os.environ.pop('R77_EXIT',None)
   else:os.environ['R77_EXIT']=_prev_exit
 public=json.loads(capture.getvalue());assert public['tokens']==measured['default']['tokens']and public['mode']=='w4-m2'
 measured['public_entry']=str(entry);measured['public_json']=public;measured['scope']='actual app dispatcher, cached CLI, validated tokenizer and default controller; explicit candidate mode while dormant; instrumentation only injects a token timestamp callback, original generation unchanged; serial comparison after frozen timing; original five-token10% criterion unchanged, M2 adjacent emits may yield no onset'
 a.output.write_text(json.dumps(measured,indent=2));print(json.dumps(dict(result='PASS',pid=measured['pid'],spec_ready_ms=measured['spec_ready_ms'],user_ready_ms=measured['user_ready_ms'],steady_state=measured['steady_state'])),flush=True)
except BaseException:
 import traceback;error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));print(error,flush=True);sys.exit(1)
