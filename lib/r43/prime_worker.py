"""Persistent CPU-only bootstrap service. No XRT/target-model process is created."""
import json,mmap,os,pickle,runpy,struct,sys,time,traceback
from pathlib import Path
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

def main():
 bundle=Path(sys.argv[1]);tag=sys.argv[2];start=time.perf_counter();arena=mmap.mmap(-1,SIZE,tagname=tag)
 if struct.calcsize('P')!=8 or __import__('platform').machine().upper()not in('AMD64','X86_64'):raise RuntimeError('x64 TSO IPC required')
 last_job=0;prefault_at=time.perf_counter()
 for off in range(0,SIZE,4096):arena[off]=0
 prefault_ms=(time.perf_counter()-prefault_at)*1000
 import numpy as np
 from cpu_bootstrap import Bootstrap
 module=runpy.run_path(str(bundle/'PLaMo-reference.py'));target=module['target'];unpack=module['unpack_row64']
 class Model:pass
 m=Model();m.meta=json.loads((bundle/'PLaMo-manifest.json').read_text());m.embedding=np.load(bundle/m.meta['embedding'],mmap_mode='r');m.w={};m.quant={};body=np.load(bundle/'PLaMo-body.npz')
 for name,entry in m.meta['parameters'].items():
  if entry['type']!='int8-row64':m.w[name]=body[entry['array']]if name.startswith('rope.')else target.bf(body[entry['array']])
 for name in('feature.down.weight','feature.up.weight','mix.down.weight','mix.up.weight','blocks.0.mixer.qkv_proj.down.weight','blocks.0.mixer.qkv_proj.up.weight'):
  e=m.meta['parameters'][name];m.quant[name]=unpack(bundle/e['packed_file'],*e['shape'])
 def fastnorm(x,gamma):
  x=np.asarray(x,np.float32);inv=np.float32(1)/np.sqrt(np.mean(x*x,axis=1,dtype=np.float32)+np.float32(1e-6));return target.bf(target.bf(x*inv[:,None])*np.asarray(gamma,np.float32))
 prime=Bootstrap(m,target,norm=fastnorm if os.environ.get('R42_BOOTSTRAP')=='fastnorm'else None);send(sys.stdout.buffer,dict(kind='ready',ipc='shared-mailbox-x64-sequence',prefault_ms=prefault_ms,ready_ns=time.perf_counter_ns(),norm_policy=os.environ.get('R42_BOOTSTRAP','bitwise'),setup_ms=(time.perf_counter()-start)*1000,pid=os.getpid(),package_manifest_sha256=__import__('hashlib').sha256((bundle/'PLaMo-manifest.json').read_bytes()).hexdigest()))
 while True:
  while arena[SUBMIT]==last_job:time.sleep(.001)
  last_job=arena[SUBMIT];command=get(arena,REQUEST)
  if command is None or command['kind']=='close':break
  job=command['job'];tokens=command['tokens'];n=len(tokens);assert 1<=n<=CAP;at=time.perf_counter();prime.cancelled=lambda:arena[CONTROL]!=0
  try:
   raw=(np.frombuffer(arena,'<u2',count=n*6144).astype(np.uint32)<<16).view(np.float32).reshape(n,4,1536);state=prime(raw,tokens);k,v=state['layers'][0];np.frombuffer(arena,'<f4',count=k.size,offset=OUT)[:]=k.reshape(-1);np.frombuffer(arena,'<f4',count=v.size,offset=OUT+KV)[:]=v.reshape(-1);put(arena,RESPONSE,dict(kind='result',job=job,position=state['position'],done_ns=time.perf_counter_ns(),compute_ms=(time.perf_counter()-at)*1000,profile=prime.profile[-1:]));del state,k,v,raw
  except InterruptedError:put(arena,RESPONSE,dict(kind='cancelled',job=job,done_ns=time.perf_counter_ns(),compute_ms=(time.perf_counter()-at)*1000))
  except BaseException:put(arena,RESPONSE,dict(kind='error',job=job,error=traceback.format_exc()))
  arena[COMPLETE]=last_job
 arena.close()
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc(file=sys.stderr);sys.stderr.flush();sys.exit(1)
