"""Opt-in v4C native cycle on the verifier's exact PDI and hw_context."""
import hashlib,json,runpy,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cycle_runtime import install as patch_install

def bits(x):
 u=np.ascontiguousarray(x,np.float32).view(np.uint32);return((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
def floats(x):return(np.frombuffer(x,'<u2').astype(np.uint32)<<16).view(np.float32)
def screen(scores,ids,k):
 subset=np.argpartition(scores,len(scores)-k)[-k:];threshold=scores[subset].min();above=subset[scores[subset]>threshold];tied=np.flatnonzero(scores==threshold);n=k-len(above)
 if len(tied)>n:tied=tied[np.argpartition(ids[tied],n-1)[:n]]
 chosen=np.concatenate((above,tied));return chosen[np.lexsort((ids[chosen],-scores[chosen]))]

def make_direct(bundle,runtime):
 bundle=Path(bundle);module=runpy.run_path(str(bundle/'PLaMo-reference.py'));Base=module['PLaMoCReference'];spec=runtime.manifest['r40_cycle'];assert not spec['UNGATED'],'proxy is forbidden for native package/generation gates';assert hashlib.sha256((bundle/'PLaMo-manifest.json').read_bytes()).hexdigest()==spec['package_manifest_sha256'];patch_install(runtime)
 class Native(Base):
  def __init__(self,bundle):
   # Generation needs packed native weights, not unpacked CPU model copies.
   self.root=Path(bundle);self.meta=json.loads((self.root/'PLaMo-manifest.json').read_text());self.c=self.meta['config'];self.variant='B';self.native=True;self.r=module['bf16'];self.quant={};self.w={};self.embedding=np.load(self.root/self.meta['embedding'],mmap_mode='r');raw=np.load(self.root/'PLaMo-body.npz')
   for name,entry in self.meta['parameters'].items():
    if entry['type']!='int8-row64':self.w[name]=raw[entry['array']]if name.startswith('rope.')else self.r(raw[entry['array']])
   self.vocab=np.load(self.root/'PLaMo-head.npz')['vocab_ids'];self.K=512;self.bank=self.meta['head']['exact_policy'];self._gold_bank=None
   assert self.bank=='int8';self.phase='target-C-one-context';self._state=None;self._position=None;self.cycle_profile=[];self.refresh_profile=[];self.head_profile=[];self.front_profile=[];self.back_profile=[];self.attention_profile=[];self.native_matrices=SimpleNamespace(times=[],physical_bytes=runtime.manifest['roots']['r40_weights']['nbytes'],pdi=spec['pdi']);self._bank=np.memmap(spec['exact_bank_file'],mode='r',dtype=np.uint8,shape=(len(self.vocab),1600));self._exact=np.frombuffer(runtime.views['r40_exact_weights'],np.uint8).reshape(512,1600)
   assert len({runtime.manifest['graphs'][g]['steps'][0]['context']for g in('verify2','r41_proposal_join','r40_refresh_join','r41_exact')})==1,'whole cycle must share the verifier hw_context'
   positions=np.arange(2048,dtype=np.float32)[:,None];angles=positions*self.w['rope.0.inv_freq'];con=np.float32(self.meta['rotary'][0]['concentration']);co=self.r(np.cos(angles).astype(np.float32)*con);si=self.r(np.sin(angles).astype(np.float32)*con);self._angles=bits(np.stack((co,si),-1).reshape(2048,128));self._zero=bytes(2048*128*2)
   runtime.prepare('r41_proposal_join',{'position':0});runtime.prepare('r41_exact',{});runtime.prepare('r40_refresh_join',{});runtime.prepare('r41_refresh_join',{})
  @property
  def bank_q(self):
   if self._gold_bank is None:self._gold_bank=module['unpack_row64'](self.root/self.meta['head']['exact_file'],len(self.vocab),1536)
   return self._gold_bank[0]
  @property
  def bank_s(self):
   if self._gold_bank is None:self._gold_bank=module['unpack_row64'](self.root/self.meta['head']['exact_file'],len(self.vocab),1536)
   return self._gold_bank[1]
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
