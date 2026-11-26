"""Early-spawned CPU worker with persistent shared spans and framed control pipe."""
import ctypes,json,mmap,os,pickle,struct,subprocess,sys,time
from pathlib import Path
from prime_worker import CAP,RAW,OUT,KV,CONTROL,SUBMIT,COMPLETE,REQUEST,RESPONSE,SIZE,put,get,send,receive
class Primer:
 def __init__(self,bundle):
  self.bundle=str(bundle);self.tag='Local\\plamo3-r43-'+str(os.getpid());self.arena=mmap.mmap(-1,SIZE,tagname=self.tag);self.pending=None;self.job=0;self.toggle=0;self._health_ns=0;self.ready_record=None;self.last_taps=None;self.records=[];self.started_ns=time.perf_counter_ns();env=os.environ.copy()
  for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='2'
  self.child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('prime_worker.py')),self.bundle,self.tag],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,env=env,bufsize=0)
 def available(self):
  import msvcrt
  count=ctypes.c_ulong();api=ctypes.windll.kernel32;api.PeekNamedPipe.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p];ok=api.PeekNamedPipe(ctypes.c_void_p(msvcrt.get_osfhandle(self.child.stdout.fileno())),None,0,None,ctypes.byref(count),None)
  if not ok:raise RuntimeError('priming worker pipe closed, exit='+str(self.child.poll()))
  return count.value>=4
 def ready(self,wait=True):
  if self.ready_record is None and(wait or self.available()):
   self.ready_record=receive(self.child.stdout);assert self.ready_record['kind']=='ready',self.ready_record;self.ready_record['parent_ready_ns']=time.perf_counter_ns()
  return self.ready_record is not None
 def start(self,rows,tokens):
  assert self.pending is None;self.ready();assert len(rows)==len(tokens);at=time.perf_counter_ns();self.job+=1;self.pending=self.job;self.arena[CONTROL]=0
  import numpy as np
  ready_ns=time.perf_counter_ns()
  for i,row in enumerate(rows):
   if not isinstance(row,(bytes,bytearray,memoryview)):
    u=np.ascontiguousarray(row,np.float32).view(np.uint32);row=((u+0x7fff+((u>>16)&1))>>16).astype('<u2').tobytes()
   assert len(row)==12288;self.arena[i*12288:(i+1)*12288]=row
  self.last_taps=bytes(self.arena[(len(tokens)-1)*12288:len(tokens)*12288]);copied_ns=time.perf_counter_ns();command=dict(kind='prime',job=self.job,tokens=list(map(int,tokens)));prepared_ns=time.perf_counter_ns();put(self.arena,REQUEST,command);self.toggle^=1;self.arena[SUBMIT]=self.toggle;self.sent_ns=time.perf_counter_ns();return dict(job=self.job,start_ns=at,end_ns=self.sent_ns,submit_ms=(self.sent_ns-at)/1e6,ready_ms=(ready_ns-at)/1e6,copy_ms=(copied_ns-ready_ns)/1e6,prepare_message_ms=(prepared_ns-copied_ns)/1e6,send_ms=(self.sent_ns-prepared_ns)/1e6,threads=[t.name for t in __import__('threading').enumerate()])
 def done(self):
  if self.pending is None:return False
  return self.arena[COMPLETE]==self.toggle
 def result(self):
  assert self.pending is not None;waited=0
  while self.arena[COMPLETE]!=self.toggle:
   time.sleep(.001);waited+=1
   if self.child.poll()is not None or waited>=90000:raise RuntimeError('CPU priming worker unavailable/timed out')
  message=get(self.arena,RESPONSE);assert message['job']==self.pending,message;self.pending=None;message['received_ns']=time.perf_counter_ns();self.records.append({k:v for k,v in message.items()if k!='profile'})
  if message['kind']=='error':raise RuntimeError(message['error'])
  if message['kind']=='cancelled':return None,message
  assert message['kind']=='result';import numpy as np
  p=message['position'];k=np.frombuffer(self.arena,'<f4',count=p*128,offset=OUT).copy().reshape(1,p,128);v=np.frombuffer(self.arena,'<f4',count=p*128,offset=OUT+KV).copy().reshape(1,p,128);last=(np.frombuffer(self.last_taps,'<u2').astype(np.uint32)<<16).view(np.float32).reshape(4,1536);return dict(position=p,last_taps=last,layers=[(k,v)]),message
 def cancel(self):
  if self.pending is None:return None
  self.arena[CONTROL]=1;return self.pending
 def quiesce(self):
  at=time.perf_counter()
  if self.pending is not None:self.cancel();self.result()
  return(time.perf_counter()-at)*1000
 def close(self):
  if self.child.poll()is None:
   self.quiesce();self.job+=1;put(self.arena,REQUEST,dict(kind='close'));self.toggle^=1;self.arena[SUBMIT]=self.toggle;self.child.stdin.close();self.child.wait(timeout=10)
  self.child.stdout.close();self.arena.close()
