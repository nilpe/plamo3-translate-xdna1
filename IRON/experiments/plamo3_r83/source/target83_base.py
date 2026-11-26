"""Opt-in v4C native cycle on the verifier's exact PDI and hw_context."""
import hashlib,json,runpy,sys,time
from pathlib import Path
from types import SimpleNamespace,MethodType
import numpy as np
from cycle_runtime import install as patch_install

def bits(x):
 u=np.ascontiguousarray(x,np.float32).view(np.uint32);return((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
def floats(x):return(np.frombuffer(x,'<u2').astype(np.uint32)<<16).view(np.float32)
def screen(scores,ids,k):
 subset=np.argpartition(scores,len(scores)-k)[-k:];threshold=scores[subset].min();above=subset[scores[subset]>threshold];tied=np.flatnonzero(scores==threshold);n=k-len(above)
 if len(tied)>n:tied=tied[np.argpartition(ids[tied],n-1)[:n]]
 chosen=np.concatenate((above,tied));return chosen[np.lexsort((ids[chosen],-scores[chosen]))]

# --- R83: package-dependent construction work that needs no device state is prepared in a background thread that
# starts at the retained-validation/authorization barrier (never before), overlapping the remaining runtime setup.
# Same reference code, same checks, same arithmetic; the inline path (no preload) is the previous behaviour. ---
import threading
from concurrent.futures import ThreadPoolExecutor
_PRELOAD={}
def _load_package(bundle,authorized_sha256=None,background=False):
 from concurrent.futures import Future
 bundle=Path(bundle);manifest_bytes=(bundle/'PLaMo-manifest.json').read_bytes();digest=hashlib.sha256(manifest_bytes).hexdigest()
 if authorized_sha256 is not None and digest!=authorized_sha256:raise ValueError('package manifest differs from the validated manifest')
 module=runpy.run_path(str(bundle/'PLaMo-reference.py'));meta=json.loads(manifest_bytes);r=module['bf16']
 def body():
  embedding=np.load(bundle/meta['embedding'],mmap_mode='r');raw=np.load(bundle/'PLaMo-body.npz');w={}
  for name,entry in meta['parameters'].items():
   if entry['type']!='int8-row64':w[name]=raw[entry['array']]if name.startswith('rope.')else r(raw[entry['array']])
  return embedding,w
 def angles_for(w):
  positions=np.arange(2048,dtype=np.float32)[:,None];angles=positions*w['rope.0.inv_freq'];con=np.float32(meta['rotary'][0]['concentration']);co=r(np.cos(angles).astype(np.float32)*con);si=r(np.sin(angles).astype(np.float32)*con)
  return bits(np.stack((co,si),-1).reshape(2048,128))
 with ThreadPoolExecutor(max_workers=3,thread_name_prefix='r83-pkg')as pool:
  languages={lang:pool.submit(module['language_config'],bundle,lang)for lang in('ja','en')};loaded=pool.submit(body)
  embedding,w=loaded.result();languages={lang:f.result()for lang,f in languages.items()}  # both screens validated before return (fail-fast, as before)
 angles=Future()
 def rest():
  # RoPE tables and the controller's lazy imports: not needed before the first request's first cycle.
  try:angles.set_result(angles_for(w))
  except BaseException as error:angles.set_exception(error)
  try:
   import loop83
  except BaseException:pass
 if background:threading.Thread(target=rest,daemon=True,name='r83-pkg-rest').start()
 else:rest()
 return dict(module=module,meta=meta,digest=digest,embedding=embedding,w=w,angles=angles,languages=languages)
def start_preload(runtime):
 spec=runtime.manifest['r40_cycle'];key=str(Path(spec['bundle']))
 if key in _PRELOAD:return
 box=dict(done=threading.Event(),value=None,error=None,started=time.perf_counter(),finished=None);_PRELOAD[key]=box
 def run():
  try:box['value']=_load_package(key,spec['package_manifest_sha256'],background=True)
  except BaseException as error:box['error']=error
  finally:
   box['finished']=time.perf_counter();box['done'].set()
 threading.Thread(target=run,daemon=True,name='r83-preload').start()
def _take(bundle,authorized_sha256):
 box=_PRELOAD.pop(str(Path(bundle)),None)
 if box is None:return _load_package(bundle,authorized_sha256)
 box['done'].wait()
 if box['error']is not None:raise box['error']
 return box['value']

class MappedBank:
 def __init__(self,bank,mapping):self.bank=bank;self.mapping=mapping
 def __array_function__(self,func,types,args,kwargs):
  if func is not np.take:return NotImplemented
  assert args[0]is self and kwargs.get('axis')==0
  indices=self.mapping[np.asarray(args[1])]
  return np.take(self.bank,indices,**kwargs)

def make_direct(bundle,runtime):
 # Route the unchanged command to a prebound resident weight BO. All
 # proposal instructions and arithmetic remain identical across the banks.
 if not hasattr(runtime,'_r48_original_execute'):
  runtime._r48_original_execute=runtime.execute;runtime._r48_language='ja'
  def execute(rt,graph,values):
   if graph=='r41_proposal_join'and rt._r48_language=='en':graph='r48_proposal_join_en'
   return rt._r48_original_execute(graph,values)
  runtime.execute=MethodType(execute,runtime)
 bundle=Path(bundle);spec=runtime.manifest['r40_cycle'];assert not spec['UNGATED'],'proxy is forbidden for native package/generation gates';pre=_take(bundle,spec['package_manifest_sha256']);module=pre['module'];Base=module['PLaMoCReference'];assert pre['digest']==spec['package_manifest_sha256'];patch_install(runtime)
 class Native(Base):
  def __init__(self,bundle):
   # Generation needs packed native weights, not unpacked CPU model copies.
   self.root=Path(bundle);self.meta=pre['meta'];self.c=self.meta['config'];self.variant='B';self.native=True;self.r=module['bf16'];self.quant={};self.w=pre['w'];self.embedding=pre['embedding']
   self.target_language=None;self._coarse_language='ja';self._languages=pre['languages'];self.vocab=np.zeros(24576,np.uint32);self.row_map=np.zeros(24576,np.uint32);self.K=512;self.bank=self.meta['head']['exact_policy'];self._gold_bank=None
   assert self.bank=='int8';self.phase='target-C-one-context';self._state=None;self._position=None;self.cycle_profile=[];self.refresh_profile=[];self.head_profile=[];self.front_profile=[];self.back_profile=[];self.attention_profile=[];self.native_matrices=SimpleNamespace(times=[],physical_bytes=runtime.manifest['roots']['r40_weights']['nbytes'],pdi=spec['pdi']);self._union=np.memmap(spec['exact_bank_file'],mode='r',dtype=np.uint8,shape=(self.meta['head']['union_physical_rows'],1600));self._bank=MappedBank(self._union,self.row_map);self._screens={lang:np.memmap(self.root/self.meta['target_language'][lang]['coarse_file'],mode='r',dtype=np.uint8)for lang in('ja','en')};self._exact=np.frombuffer(runtime.views['r40_exact_weights'],np.uint8).reshape(512,1600)
   assert len({runtime.manifest['graphs'][g]['steps'][0]['context']for g in('verify2','r41_proposal_join','r40_refresh_join','r41_exact')})==1,'whole cycle must share the verifier hw_context'
   self._zero=bytes(2048*128*2)
   runtime.prepare('r41_proposal_join',{'position':0});runtime.prepare('r48_proposal_join_en',{'position':0});runtime.prepare('r41_exact',{});runtime.prepare('r40_refresh_join',{});runtime.prepare('r41_refresh_join',{})
  @property
  def _angles(self):
   return pre['angles'].result()
  def prefault(self):
   # Map the union exact bank's pages into this process once, in the background after readiness, so the first
   # proposal gathers do not take soft page faults. Read-only touches; no data is changed.
   def touch():
    try:self._union.reshape(-1)[::4096].sum()
    except BaseException:pass
   threading.Thread(target=touch,daemon=True,name='r83-union-prefault').start()
  def select_language(self,language):
   if language not in ('ja','en'):raise ValueError('target_language must explicitly be ja or en')
   cfg=self._languages[language]
   runtime._r48_language=language;self._coarse_language=language
   self.target_language=language;self.vocab=cfg['vocab_ids'];self.row_map=cfg['id_map'];self._bank=MappedBank(self._union,self.row_map)
  @property
  def bank_q(self):
   if self._gold_bank is None:self._gold_bank=module['unpack_row64'](self.root/self.meta['head']['exact_file'],self.meta['head']['union_physical_rows'],1536)
   return self._gold_bank[0][self.row_map]
  @property
  def bank_s(self):
   if self._gold_bank is None:self._gold_bank=module['unpack_row64'](self.root/self.meta['head']['exact_file'],self.meta['head']['union_physical_rows'],1536)
   return self._gold_bank[1][self.row_map]
  def write_row(self,name,data,row=0):
   alias=name+'_row'+str(row)
   if alias in runtime.views:runtime.write(alias,data)
   else:runtime.write(name,data,offset=row*524288)
  def read_row(self,name,n,row=0):
   alias=name+'_row'+str(row)
   return floats(runtime.read(alias,nbytes=n*2))if alias in runtime.views else floats(runtime.read(name,nbytes=n*2,offset=row*524288))
  def bind_state(self,state):
   if self._state is state and self._position==state['position']:return
   k,v=state['layers'][0];assert k.shape==v.shape==(1,state['position'],128);assert 0<=state['position']<=2048
   for part,data in(('K',k),('V',v)):
    for kind in('confirmed','spec'):
     name='r40_'+kind+part;runtime.write(name,self._zero)
     if state['position']:runtime.write(name,bits(data[0]))
     runtime.sync_to(name)
   self._state=state;self._position=state['position']
  def confirmed_device_bytes(self):return runtime.read('r40_confirmedK')+runtime.read('r40_confirmedV')
  def cycle(self,state,anchor):
   if self.target_language is None:raise ValueError('target_language missing')
   if state['position']+3>2048:raise ValueError('use exact serial fallback at domain boundary')
   t=time.perf_counter();self.bind_state(state);position=int(state['position']);self.write_row('r40_taps',bits(state['last_taps']));self.write_row('r40_embed',bits(self.embedding[int(anchor)]))
   self.write_row('r40_rope',self._angles[position])
   prepared=time.perf_counter();body=runtime.execute('r41_proposal_join',{'position':position});at=time.perf_counter();hidden=self.read_row('r40_hidden',1536);coarse=self.read_row('r40_coarse',len(self.vocab));returned=time.perf_counter();chosen=screen(coarse,self.vocab,512);ids=self.vocab[chosen];selected=time.perf_counter();np.take(self._bank,chosen,axis=0,out=self._exact,mode='clip');runtime.sync_to('r40_exact_weights');gathered=time.perf_counter();exact=runtime.execute('r41_exact',{});logits=self.read_row('r40_exact',512);order=np.lexsort((ids,-logits));ended=time.perf_counter()
   self.head_profile.append(dict(selection_ms=(selected-returned)*1000,gather_ms=(gathered-selected)*1000,exact_ms=(ended-gathered)*1000,runtime=exact));self.cycle_profile.append(dict(prepare_ms=(prepared-t)*1000,native_body_ms=(at-prepared)*1000,return_ms=(returned-at)*1000,selection_ms=(selected-returned)*1000,gather_ms=(gathered-selected)*1000,exact_ms=(ended-gathered)*1000,total_ms=(ended-t)*1000,body_runtime=body,exact_runtime=exact));return dict(proposal=int(ids[order[0]]),top3=ids[order[:3]],candidate_ids=ids,candidate_logits=logits,coarse_logits=coarse,hidden=hidden)
  def append_confirmed(self,state,raw,tokens):
   started=time.perf_counter();raw=self.r(raw);tokens=np.asarray(tokens,np.int64);rows=len(tokens)
   if not rows:return state
   assert rows in(1,2)and int(state['position'])+rows<=2048;self.bind_state(state);position=int(state['position']);features=np.concatenate((state['last_taps'][None],raw[:-1]),0)
   for row in range(rows):self.write_row('r40_taps',bits(features[row]),row);self.write_row('r40_embed',bits(self.embedding[tokens[row]]),row)
   for row in range(2):self.write_row('r40_rope',self._angles[min(position+row,2047)],row)
   prepared=time.perf_counter();stats=runtime.execute('r40_refresh_join',{});at=time.perf_counter();ks=[];vs=[]
   for row in range(rows):
    k=self.read_row('r40_refreshk',128,row);v=self.read_row('r40_refreshkv',256,row)[128:];ks.append(k);vs.append(v)
    for part,data in(('K',k),('V',v)):
     for kind in('confirmed','spec'):
      name='r40_'+kind+part;alias=name+'_p'+str(position+row)
      if alias in runtime.views:runtime.write(alias,bits(data));runtime.sync_to(alias)
      else:runtime.write(name,bits(data),offset=(position+row)*256);runtime.sync_to(name)
   old=state['layers'][0];k=np.concatenate((old[0],np.asarray(ks,np.float32)[None]),1);v=np.concatenate((old[1],np.asarray(vs,np.float32)[None]),1);updated=dict(last_taps=raw[-1],position=position+rows,layers=[(k,v)]);self._state=updated;self._position=updated['position'];self.refresh_profile.append(dict(rows=rows,prepare_ms=(prepared-started)*1000,native_ms=(at-prepared)*1000,return_ms=(time.perf_counter()-at)*1000,total_ms=(time.perf_counter()-started)*1000,runtime=stats));return updated
  def prime(self,raw,tokens):
   raw=self.r(raw);tokens=np.asarray(tokens,np.int64);assert len(tokens)==len(raw)>=1;state=dict(last_taps=raw[0],position=0,layers=[(np.empty((1,0,128),np.float32),np.empty((1,0,128),np.float32))]);self.bind_state(state)
   for i in range(1,len(tokens),2):state=self.append_confirmed(state,raw[i:i+2],tokens[i:i+2])
   assert state['position']==len(tokens)-1;return state
 return Native,module
