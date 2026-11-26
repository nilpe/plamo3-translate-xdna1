"""R83 device end-to-end check of the keep-warm bracket on the real W4 M2 runtime (staged bundle): usage via the runner step `keepwarm-e2e` (the runtime is constructed with keep_warm='on' explicitly, independent of env/default).
Checks (mode on): helper spawned only after spec-ready and never delaying it; one generate that raises (limit=0) restores the decode thread affinity and leaves the spinner off;
a real short generate pins during the call, restores afterwards and returns tokens identical to the serial path; helper CPU ~0 when idle between requests and ~1 s/s inside a request;
the helper exits on close; the decode thread's affinity is the process mask outside generate.
Writes --output json."""
import argparse,json,os,struct,sys,time
from pathlib import Path
from settings83 import ROOT,paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--idle',type=int,default=0);a=p.parse_args()
try:
 import keepwarm83 as K
 from default83 import DefaultRuntime
 from readiness import process_age_ms
 from loop_gate import serial_generate
 rows=[];fails=[]
 def check(ok,msg):
  rows.append(dict(ok=bool(ok),msg=msg))
  if not ok:fails.append(msg)
 base=K.thread_mask();k32=K.k32()
 app=DefaultRuntime(keep_warm='on');ready_ms=process_age_ms();kw=app.keepwarm
 assert app.mode=='w4-m2',app.fallbacks
 check(kw is not None and kw.mode in('on','ac'),'keep-warm active in this arm (mode %r)'%getattr(app,'keep_warm_mode',None))
 check(K.thread_mask()==base,'decode thread affinity untouched after construction')
 t=time.time()+10
 while time.time()<t and kw.mm[K.F_READY]!=1:time.sleep(0.01)
 check(kw.mm[K.F_READY]==1,'helper ready (after construction)')
 check(kw.spawn_ms is not None and kw.spawn_ms<100,'spawn took %s ms'%kw.spawn_ms)
 pid=struct.unpack('<I',bytes(kw.mm[K.F_PID:K.F_PID+4]))[0];check(pid==kw.proc.pid,'one helper process (no launcher layer): %d vs %d'%(pid,kw.proc.pid))
 import ctypes
 k2=ctypes.WinDLL('kernel32');k2.OpenProcess.restype=ctypes.c_void_p;k2.GetPriorityClass.argtypes=[ctypes.c_void_p];h=k2.OpenProcess(0x1000|0x100000,False,pid)
 check(k2.GetPriorityClass(h)==0x40,'helper priority class idle')
 f=[(ctypes.c_uint64*1)() for _ in range(4)];k2.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.c_void_p]*4
 def hcpu():
  k2.GetProcessTimes(h,*[ctypes.byref(x) for x in f]);return (f[2][0]+f[3][0])/1e7
 time.sleep(0.5);a0=hcpu();time.sleep(1.0);a1=hcpu();check(a1-a0<0.06,'idle helper CPU %.3f s in 1 s'%(a1-a0))
 # exception path on the real runtime
 raised=False
 try:app.generate([1,2,3],0)
 except ValueError:raised=True
 check(raised,'generate(limit=0) raised ValueError');check(K.thread_mask()==base,'affinity restored after the raising generate');check(kw.mm[K.F_FLAG]==0,'spinner flag off after the raising generate')
 meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());prompt=meta['prompts'][0]
 serial=serial_generate(app.session,prompt,64)
 seen={};delivered=[]
 def emit(tok):
  if 'mask' not in seen:seen['mask']=K.thread_mask();seen['flag']=kw.mm[K.F_FLAG];seen['hmask']=kw.helper_cpu_mask()
  delivered.append(tok)
 c0=hcpu();t0=time.perf_counter();got=app.generate(prompt,64,emit=emit,target_language='en');dt=time.perf_counter()-t0;c1=hcpu()
 check(got['tokens']==serial['tokens']==delivered,'tokens identical to the serial path (%d tokens)'%len(got['tokens']))
 check(seen.get('mask')==1<<kw.cpu and seen.get('flag')==1 and seen.get('hmask')==seen.get('mask'),'pinned to cpu %d with the spinner on during generate (mask %s flag %s helper %s)'%(kw.cpu,seen.get('mask'),seen.get('flag'),seen.get('hmask')))
 check(K.thread_mask()==base and kw.mm[K.F_FLAG]==0,'affinity restored and spinner off after generate')
 check(c1-c0>0.5*dt,'helper spun during the request (%.3f CPU s over %.3f s)'%(c1-c0,dt))
 time.sleep(0.3);b0=hcpu();time.sleep(1.0);b1=hcpu();check(b1-b0<0.06,'helper idle again between requests (%.3f s in 1 s)'%(b1-b0))
 st=kw.stats();check(st['windows']>=2 and not st['disabled'] and not st['notes'],'controller stats clean: %r'%({k:v for k,v in st.items() if k!='helper'},))
 app.close();ok=False
 t=time.time()+5
 while time.time()<t:
  if k2.WaitForSingleObject(h,0)==0:ok=True;break
  time.sleep(0.05)
 check(ok,'helper exited after runtime close')
 rep=dict(result='PASS' if not fails else 'FAIL',spec_ready_ms=app.spec_ready_ms,user_ready_ms=ready_ms,keepwarm=kw.stats(),checks=rows,failures=fails)
 a.output.write_text(json.dumps(rep,indent=2,default=str));print(json.dumps(dict(result=rep['result'],failures=fails)),flush=True);os._exit(0 if not fails else 1)
except BaseException:
 import traceback;error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));print(error,flush=True);os._exit(1)
