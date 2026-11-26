# SPDX-License-Identifier: Apache-2.0
"""Exact direct-runtime tests. No tolerances, no target-informed draft proposals."""
import argparse,array,ast,hashlib,json,os,struct,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
APP=((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'IRON/iron/applications/plamo3')
sys.path.insert(0,str(APP));sys.path.insert(0,str(Path(__file__).parent))
from verify_session import VerifySession,toy_generate

def header(stream,count):
 if stream.read(6)!=b"\x93NUMPY":raise ValueError("not a NumPy reference")
 version=stream.read(2)
 if version not in(b"\x01\x00",b"\x02\x00"):raise ValueError("unsupported NumPy version")
 n=struct.unpack("<H"if version==b"\x01\x00"else"<I",stream.read(2 if version==b"\x01\x00"else 4))[0]
 h=ast.literal_eval(stream.read(n).decode("ascii"))
 if h!={"descr":"<f4","fortran_order":False,"shape":(count,107520)}:raise ValueError("reference layout differs")

def same(actual,expected,label):
 if actual!=expected:
  aa=array.array('H');aa.frombytes(actual);bb=array.array('H');bb.frombytes(expected)
  d=[(i,a,b)for i,(a,b)in enumerate(zip(aa,bb))if a!=b]
  raise AssertionError(f'{label}: {len(d)} differing BF16 words; first={d[:8]}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['smoke','teacher','rollback','generation','cost'],required=True);a=p.parse_args()
 import startup_r20 as startup
 e=startup.initialize(SimpleNamespace(manifest=a.manifest,timing_helper=APP/'startup_probe.py',validation='identity',hash_workers=2,copy_mode='readinto',overlap=True,prompt=3))
 s=e['session'];r=s.runtime;v=VerifySession(s);M=v.rows
 root=((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'gate-data/plamo3-w4-r32/order-reference');meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
 # Cache aliases are unique read/write 2048x128 BF16 views in the weight/state root.
 kv=[name for name,view in r.manifest['views'].items()if view['root']=='root0' and view['nbytes']==524288 and view['offset']>=474494976 and view['offset']<491272192 and not name.startswith('v')]
 # All normal view names begin with v; select exact base view prefix instead.
 kv=[name for name,view in r.manifest['views'].items()if name.startswith('view') and view['root']=='root0' and view['nbytes']==524288 and 474494976<=view['offset']<491272192]
 by_offset={r.manifest['views'][n]['offset']:n for n in kv};kv=[by_offset[k]for k in sorted(by_offset)]
 if len(kv)!=32:raise ValueError(f'expected exactly 32 cache operands, got {len(kv)}')
 def snap():return [r.read(n)for n in kv]
 def restore(data,length):
  v.discard()
  for n,b in zip(kv,data):r.write(n,b)
  s.length=length;s.last_graph=None
 def prefix_equal(a,b,length,label):
  for j,(x,y)in enumerate(zip(a,b)):same(x[:length*256],y[:length*256],f'{label}/KV{j}')
 report=dict(result='RUNNING',mode=a.mode,M=M,manifest=str(a.manifest),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),criterion='byte-for-byte BF16 logits and exact tokens; no tolerance',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),session_sha256=hashlib.sha256(Path(__file__).with_name('verify_session.py').read_bytes()).hexdigest(),records=[])
 if a.mode=='smoke':
  prompt=meta['prompts'][0];s.prefill(prompt);before=snap();p0=s.length
  inputs=meta['teacher_tokens'][:M];actual=v.stage(inputs);rows=v.logits();staged=snap();v.commit(M-1)
  restore(before,p0)
  for i,t in enumerate(inputs):
   got=s.decode(t);same(rows[i],s.read_logits(),f'smoke row{i}');assert actual['greedy'][i]==got['token']
  prefix_equal(staged,snap(),p0+M,'smoke')
  report['records'].append(dict(rows=M,bitwise=True,KV_bitwise=True))
 elif a.mode=='teacher':
  serial_counts={};verify_counts={}
  for group,count in [('standard',100),('long',64)]:
   cases=([dict(id=f'standard-{i}',prompt=p,teacher_outputs=meta['teacher_tokens'][i*25:(i+1)*25])for i,p in enumerate(meta['prompts'])]if group=='standard'else meta['long_cases'])
   passed=0;verified=0
   with (root/(group+'.npy')).open('rb')as f:
    header(f,count)
    for case in cases:
     expected=[]
     for _ in case['teacher_outputs']:
      words=array.array('I');words.frombytes(f.read(107520*4))
      if len(words)!=107520 or any(w&65535 for w in words):raise ValueError('reference must be exact BF16')
      expected.append(array.array('H',(w>>16 for w in words)).tobytes())
     s.prefill(case['prompt']);same(s.read_logits(),expected[0],case['id']+'/prefill');passed+=1
     before=snap();p0=s.length;inputs=[];serial_rows=[];serial_tokens=[]
     total=((len(expected)+M-1)//M)*M
     for i in range(total):
      token=(case['teacher_outputs'][i]if i<len(case['teacher_outputs'])else serial_tokens[-1]);inputs.append(token)
      got=s.decode(token);row=s.read_logits();serial_rows.append(row);serial_tokens.append(got['token'])
      if i+1<len(expected):same(row,expected[i+1],case['id']+f'/serial{i+1}');passed+=1
     serial_kv=snap();restore(before,p0)
     for first in range(0,total,M):
      got=v.stage(inputs[first:first+M]);rows=v.logits()
      for i,row in enumerate(rows):same(row,serial_rows[first+i],case['id']+f'/verify{first+i}');verified+=1
      assert got['greedy']==serial_tokens[first:first+M]
      v.commit(M-1)
     prefix_equal(snap(),serial_kv,p0+total,case['id'])
     row=dict(case=case['id'],certified_serial_rows=len(expected),verify_decode_rows=total,bitwise=True,KV_bitwise=True)
     report['records'].append(row);print(json.dumps(row),flush=True)
    if f.read(1):raise ValueError('extra reference rows')
   assert passed==count;serial_counts[group]=passed;verify_counts[group]=verified
  report.update(certified_serial_rows=serial_counts,verify_decode_rows=verify_counts,initial_prefill_rows_retained=12)
 elif a.mode=='rollback':
  cases=[dict(id='standard-0',prompt=meta['prompts'][0],teacher_outputs=meta['teacher_tokens'][:25]),next(c for c in meta['long_cases']if c['id']=='ja-1024')]
  for case in cases:
   s.prefill(case['prompt']);base=snap();p0=s.length
   for wrong in range(M):
    inputs=list(case['teacher_outputs'][:M]);inputs[wrong]=(inputs[wrong]+1)%s.vocab_size
    for k in range(M):
     restore(base,p0);v.stage(inputs);rows=v.logits();staged=snap()
     for j,(old,new)in enumerate(zip(base,staged)):
      same(new[:p0*256],old[:p0*256],f'prefix cache{j}');same(new[(p0+M)*256:],old[(p0+M)*256:],f'padding cache{j}')
     decision=v.commit(k);assert decision['length']==p0+k+1
     replacement=(inputs[(k+1)%M]+17)%s.vocab_size
     got=s.decode(replacement);after=s.read_logits();after_kv=snap()
     restore(base,p0)
     for i,t in enumerate(inputs):
      s.decode(t);same(rows[i],s.read_logits(),f'perturbed {case["id"]} depth{wrong} row{i}')
      if i==k:committed=snap()
     prefix_equal(staged,committed,p0+k+1,'committed prefix')
     restore(committed,p0+k+1);fresh=s.decode(replacement);same(after,s.read_logits(),'rollback continuation');assert got['token']==fresh['token']
     prefix_equal(after_kv,snap(),p0+k+2,'resumed prefix')
     rec=dict(case=case['id'],wrong_depth=wrong,accepted_drafts=k,committed_length=p0+k+1,logits_bitwise=True,KV_bitwise=True)
     report['records'].append(rec);print(json.dumps(rec),flush=True)
 elif a.mode=='generation':
  paragraph=json.loads(((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'gate-data/long-translation.json').read_text(encoding='utf-8'))
  cases=[(f'standard-{i}',p,None)for i,p in enumerate(meta['prompts'])]+[('paragraph-515',paragraph['case']['prompt'],paragraph['generated_tokens'])]
  for name,prompt,known in cases:
   reference=[s.prefill(prompt)['token']]
   while reference[-1]not in s.model['eos'] and len(reference)<s.max_seq-len(prompt):reference.append(s.decode(reference[-1])['token'])
   assert reference[-1]in s.model['eos']
   if known is not None:assert reference==known,'serial paragraph history changed'
   for fixed in (None,0):
    got=toy_generate(s,v,prompt,fixed=fixed)
    if got['tokens']!=reference:raise AssertionError(f'toy generation differs {name} fixed={fixed}')
    rec=dict(case=name,drafter='previous-token'if fixed is None else 'fixed-zero',token_identical=True,tokens=len(reference),passes=got['passes'],accepted_drafts=sum(got['accepted_drafts']),decode_ms=got['decode_ms'])
    report['records'].append(rec);print(json.dumps(rec),flush=True)
 else:raise NotImplementedError('cost comes after the exact gates')
 report['result']='PASS';a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result='PASS',mode=a.mode,M=M,records=len(report['records']))),flush=True)
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:traceback.print_exc();code=1
 sys.stdout.flush();sys.stderr.flush();os._exit(code)
