"""R83 persistent CPU-only bootstrap service (no XRT/target-model process).
Changes vs R45 (numerics unchanged; BLAS results are bitwise independent of thread count, checked 1..8 threads x 80 sizes):
 adaptive BLAS threads (1 for short prompts: no multi-thread scheduling stalls), parallel post-authorization setup,
 warm-up jobs after ready (idle until the first request), timings recorded in the ready record."""
import ctypes,glob,json,mmap,os,pickle,runpy,struct,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode=True   # H1: the package reference scripts run through runpy; never write __pycache__ into the hash-bound package directory
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'plamo3_r45'/'source'))
from settings45 import ROOT,paths
# the worker may run under the base interpreter of the venv; the venv's site-packages (numpy) are added explicitly. Release layout: <root>\venv; a development tree keeps IRON\.venv.
for _venv in (ROOT/'venv',ROOT/'IRON/.venv'):
 if (_venv/'Lib/site-packages').is_dir():sys.path.insert(0,str(_venv/'Lib/site-packages'));break
paths()
CAP=2048;RAW=CAP*6144*2;OUT=RAW;KV=CAP*128*4;CONTROL=OUT+2*KV;SUBMIT=CONTROL+8;COMPLETE=CONTROL+16;REQUEST=CONTROL+64;MAILBOX=16384;RESPONSE=REQUEST+MAILBOX;SIZE=RESPONSE+MAILBOX

def send(stream,value):
 payload=pickle.dumps(value,protocol=5);frame=memoryview(struct.pack('<I',len(payload))+payload)
 while frame:
  n=stream.write(frame)
  if not n:raise EOFError('control pipe stopped accepting bytes')
  frame=frame[n:]
 stream.flush()
def receive(stream):
 head=stream.read(4)
 if not head:return None
 size=struct.unpack('<I',head)[0];parts=bytearray()
 while len(parts)<size:
  b=stream.read(size-len(parts))
  if not b:raise EOFError('incomplete control message')
  parts.extend(b)
 return pickle.loads(parts)

def put(arena,offset,value):
 payload=pickle.dumps(value,protocol=5)
 if len(payload)>MAILBOX-4:raise ValueError('control record exceeds fixed mailbox')
 struct.pack_into('<I',arena,offset,len(payload));arena[offset+4:offset+4+len(payload)]=payload
def get(arena,offset):
 size=struct.unpack_from('<I',arena,offset)[0]
 if not 0<size<=MAILBOX-4:raise ValueError('invalid fixed mailbox extent')
 return pickle.loads(bytes(arena[offset+4:offset+4+size]))

SINGLE_THREAD_ROWS=32;WARM_ROWS=(24,64)
def parent_watch():
 # H1 (BUG 6): the worker must not outlive its launcher. A handle on the parent process is opened right at start (SYNCHRONIZE) and polled without blocking; a dead
 # parent ends the worker. (Windows does not reparent children, so os.getppid() alone never changes.) Returns a callable: True while the parent is alive. Never raises.
 try:
  ppid=os.getppid()
  if os.name!='nt':return lambda:os.getppid()==ppid
  from ctypes import wintypes
  k=ctypes.WinDLL('kernel32',use_last_error=True)
  k.OpenProcess.restype=wintypes.HANDLE;k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
  k.WaitForSingleObject.restype=wintypes.DWORD;k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
  h=k.OpenProcess(0x00100000,False,ppid)  # SYNCHRONIZE
  if not h:return lambda:False  # the parent is already gone (or cannot be inspected: be conservative and stop)
  return lambda:k.WaitForSingleObject(h,0)==0x102  # WAIT_TIMEOUT: still running
 except Exception:return lambda:True
def blas_setter():
 # numpy's bundled OpenBLAS; numerics are thread-count independent, so this only changes scheduling.
 try:
  import numpy as np
  dll=glob.glob(os.path.join(os.path.dirname(np.__file__),'..','numpy.libs','libscipy_openblas64_*.dll'))[0];lib=ctypes.CDLL(dll);fn=lib.scipy_openblas_set_num_threads64_;fn.argtypes=[ctypes.c_int];fn.restype=None
  return fn
 except Exception:return None

def main():
 parent_alive=parent_watch()
 bundle=Path(sys.argv[1]);tag=sys.argv[2];start=time.perf_counter();arena=mmap.mmap(-1,SIZE,tagname=tag)
 if struct.calcsize('P')!=8 or __import__('platform').machine().upper()not in('AMD64','X86_64'):raise RuntimeError('x64 TSO IPC required')
 last_job=0;prefault_at=time.perf_counter()
 for off in range(0,SIZE,4096):arena[off]=0
 prefault_ms=(time.perf_counter()-prefault_at)*1000
 import numpy as np
 import argparse,importlib.util,hashlib,zipfile,concurrent.futures  # stdlib, imported by the package reference code; no package file is read here
 from cpu_bootstrap import Bootstrap
 set_threads=blas_setter();threads=int(os.environ.get('OPENBLAS_NUM_THREADS','2'))
 authorization=receive(sys.stdin.buffer)
 if not authorization or authorization.get('kind')!='validated-package':raise RuntimeError('package authorization missing')
 authorized_at=time.perf_counter();phases={}
 def mark(name,at):phases[name]=round((time.perf_counter()-at)*1000,3)
 expected=authorization['manifest_sha256'];manifest_bytes=(bundle/'PLaMo-manifest.json').read_bytes();digest=__import__('hashlib').sha256(manifest_bytes).hexdigest()
 if digest!=expected:raise RuntimeError('authorized package manifest changed')
 # Package reference code and weights are read only after the parent's retained-validation barrier (authorization above).
 pool=ThreadPoolExecutor(max_workers=6,thread_name_prefix='r83-setup')
 names=('feature.down.weight','feature.up.weight','mix.down.weight','mix.up.weight','blocks.0.mixer.qkv_proj.down.weight','blocks.0.mixer.qkv_proj.up.weight')
 meta=json.loads(manifest_bytes);files=[bundle/meta['parameters'][n]['packed_file']for n in names]
 def timed(fn,*a):
  t=time.perf_counter();r=fn(*a);return r,(time.perf_counter()-t)*1000
 def prefetch(path):
  with open(path,'rb')as f:
   while f.read(1<<22):pass
  return path.name
 at=time.perf_counter()
 class Model:pass
 m=Model();m.meta=meta;m.w={};m.quant={}
 def load_body():
  m.embedding=np.load(bundle/m.meta['embedding'],mmap_mode='r');body=np.load(bundle/'PLaMo-body.npz')
  return {name:body[entry['array']]for name,entry in m.meta['parameters'].items()if entry['type']!='int8-row64'}
 body_job=pool.submit(timed,load_body);reads=[pool.submit(timed,prefetch,f)for f in files]
 module,ref_ms=timed(runpy.run_path,str(bundle/'PLaMo-reference.py'));phases['reference_run_path']=round(ref_ms,3);target=module['target'];unpack=module['unpack_row64'];mark('reference_ready',at)
 raw,body_ms=body_job.result();phases['body_embedding']=round(body_ms,3)
 for name,value in raw.items():m.w[name]=value if name.startswith('rope.')else target.bf(value)
 phases['prefetch_max']=round(max(r.result()[1] for r in reads),3);mark('prefetch_done',at)
 at=time.perf_counter();jobs=[pool.submit(lambda n=n:(n,unpack(bundle/m.meta['parameters'][n]['packed_file'],*m.meta['parameters'][n]['shape'])))for n in names]
 for j in jobs:n,v=j.result();m.quant[n]=v
 mark('unpack',at)
 def fastnorm(x,gamma):
  x=np.asarray(x,np.float32);inv=np.float32(1)/np.sqrt(np.mean(x*x,axis=1,dtype=np.float32)+np.float32(1e-6));return target.bf(target.bf(x*inv[:,None])*np.asarray(gamma,np.float32))
 class ParallelBootstrap(Bootstrap):
  def __init__(self,model,target,norm=None):
   self.m=model;self.t=target;self.norm=norm;self.weights={};self.profile=[]
   def conv(n):
    q,s=model.quant[n];return n,(np.ascontiguousarray(q,np.float32),np.ascontiguousarray(s,np.float32))
   def conv_kv():
    q,s=model.quant['blocks.0.mixer.qkv_proj.up.weight'];return 'kv',(np.ascontiguousarray(q[-256:],np.float32),np.ascontiguousarray(s[-256:],np.float32))
   futures=[pool.submit(conv,n)for n in('feature.down.weight','feature.up.weight','mix.down.weight','mix.up.weight','blocks.0.mixer.qkv_proj.down.weight')]+[pool.submit(conv_kv)]
   for f in futures:n,v=f.result();self.weights[n]=v
 at=time.perf_counter()
 if os.environ.get('R42_BOOTSTRAP')!='fastnorm':raise RuntimeError('R83 worker requires the fastnorm policy')
 prime=ParallelBootstrap(m,target,norm=fastnorm);mark('bootstrap_init',at);pool.shutdown(wait=True)
 mark_total=round((time.perf_counter()-authorized_at)*1000,3)
 send(sys.stdout.buffer,dict(kind='ready',ipc='shared-mailbox-x64-sequence',prefault_ms=prefault_ms,ready_ns=time.perf_counter_ns(),norm_policy=os.environ.get('R42_BOOTSTRAP','bitwise'),setup_ms=(time.perf_counter()-start)*1000,pid=os.getpid(),package_manifest_sha256=digest,post_authorization_ms=mark_total,phases_ms=phases,blas_threads=threads,blas_setter=set_threads is not None))
 # Warm-up (after readiness; the first request cannot arrive before a prefill completes): BLAS buffers, thread pool, pages.
 def run_threads(rows):
  want=1 if rows<=SINGLE_THREAD_ROWS else threads
  if set_threads is not None:set_threads(want)
 for n in WARM_ROWS:
  run_threads(n);rng=np.random.default_rng(7);prime.profile.clear();prime((rng.standard_normal((n,4,1536)).astype(np.float32)*.7),list(range(100,100+n)));prime.profile.clear()
 next_check=time.monotonic()+.25
 while True:
  while arena[SUBMIT]==last_job:
   time.sleep(.001)
   if time.monotonic()>=next_check:
    next_check=time.monotonic()+.25
    if not parent_alive():arena.close();return  # H1: the launcher died (killed without a tree kill)
  last_job=arena[SUBMIT];command=get(arena,REQUEST)
  if command is None or command['kind']=='close':break
  prime.profile.clear();job=command['job'];tokens=command['tokens'];n=len(tokens);assert 1<=n<=CAP;run_threads(n);at=time.perf_counter();prime.cancelled=lambda:arena[CONTROL]!=0
  try:
   raw=(np.frombuffer(arena,'<u2',count=n*6144).astype(np.uint32)<<16).view(np.float32).reshape(n,4,1536);state=prime(raw,tokens);k,v=state['layers'][0];np.frombuffer(arena,'<f4',count=k.size,offset=OUT)[:]=k.reshape(-1);np.frombuffer(arena,'<f4',count=v.size,offset=OUT+KV)[:]=v.reshape(-1);put(arena,RESPONSE,dict(kind='result',job=job,position=state['position'],done_ns=time.perf_counter_ns(),compute_ms=(time.perf_counter()-at)*1000,profile=prime.profile[-1:]));del state,k,v,raw
  except InterruptedError:put(arena,RESPONSE,dict(kind='cancelled',job=job,done_ns=time.perf_counter_ns(),compute_ms=(time.perf_counter()-at)*1000))
  except BaseException:put(arena,RESPONSE,dict(kind='error',job=job,error=traceback.format_exc()))
  arena[COMPLETE]=last_job
 arena.close()
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc(file=sys.stderr);sys.stderr.flush();sys.exit(1)
