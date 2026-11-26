"""R41 exact native cycle, with independent CPU worker bootstrap."""
from target43_io import make_direct as old_factory

def make_direct(bundle,runtime):
 Base,module=old_factory(bundle,runtime)
 class Native(Base):
  def __init__(self,bundle):super().__init__(bundle);self.primer=runtime._r42_primer;self.primer.ready();assert self.primer.ready_record['package_manifest_sha256']==runtime.manifest['r40_cycle']['package_manifest_sha256'];assert self.primer.ready_record['norm_policy']==runtime.manifest.get('r42_bootstrap',self.primer.ready_record['norm_policy']);self.phase='R42-fixed-worker-switch'
  def prime(self,raw,tokens):
   self.primer.start(raw,tokens);state,record=self.primer.result();assert state is not None;self.bind_state(state);return state
  def catchup(self,state,raw,tokens):
   self._root_pending=None;self.bind_state(state)
   for start in range(0,len(tokens),2):state=self.append_confirmed(state,raw[start:start+2],tokens[start:start+2])
   return state
 return Native,module

# R42 IO revision: persistent BOs, bounded row syncs; no arithmetic changes.
from target_drafter import bits,floats
import time
import numpy as np

def make_direct(bundle,runtime,_factory=make_direct):
 Base,module=_factory(bundle,runtime)
 class Partial(Base):
  def bind_state(self,state):
   if getattr(self,'_cpu_owner',None)is not state:
    self._cpu_kv=[np.empty((1,2048,128),np.float32),np.empty((1,2048,128),np.float32)]
    for dst,src in zip(self._cpu_kv,state['layers'][0]):dst[:,:state['position']]=src
    self._cpu_owner=state
   return super().bind_state(state)
  def _sync(self,name,offset,size,to=False):
   with runtime._lock:
    runtime._ready();assert not runtime._dirty,'bounded sync requires no pending host writes'
    view=runtime.manifest['views'][name];assert 0<=offset and 0<size and offset+size<=view['nbytes']
    direction=runtime.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE if to else runtime.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    runtime._bo(name).sync(direction,size,offset)
  def _region(self,span,name,offset,size,to=False):
   a=runtime.manifest['views'][span];b=runtime.manifest['views'][name];assert a['root']==b['root'];self._sync(span,b['offset']-a['offset']+offset,size,to)
  def read_row(self,name,n,row=0):
   if row==0 and name in('r40_hidden','r40_coarse'):
    self._region('r40_head_return_sync',name,0,n*2);return floats(runtime.read(name,nbytes=n*2,sync=False))
   return super().read_row(name,n,row)
  def append_confirmed(self,state,raw,tokens):
   start=time.perf_counter();raw=self.r(raw);tokens=np.asarray(tokens,np.int64);rows=len(tokens)
   if not rows:return state
   assert rows in(1,2)and state['position']+rows<=2048;self.bind_state(state);position=int(state['position']);pending=self._root_pending;reuse=bool(pending is not None and pending[0]is state and pending[1]==position and pending[2]==int(tokens[0])and pending[3]==bits(state['last_taps']).tobytes());self._root_pending=None;ks=[];vs=[];projected=rows-int(reuse);stats=None
   if reuse:
    for part,values in(('K',ks),('V',vs)):
     name='r40_spec'+part;self._region('r40_spec_sync',name,position*256,256);values.append(np.frombuffer(runtime.read(name,offset=position*256,nbytes=256,sync=False),'<u2').copy())
   if projected:
    for row in range(projected):
     index=row+int(reuse);source=state['last_taps']if index==0 else raw[index-1];self.write_row('r40_taps',bits(source),row);self.write_row('r40_embed',bits(self.embedding[tokens[index]]),row)
    for row in range(projected):self.write_row('r40_rope',self._angles[position+int(reuse)+row],row)
    prepared=time.perf_counter();stats=runtime.execute('r41_refresh_join'if projected==1 else'r40_refresh_join',{});at=time.perf_counter()
    for row in range(projected):
     for name,off,values in(('r40_refreshk',row*524288,ks),('r40_refreshkv',row*524288+256,vs)):
      self._region('r40_refresh_pair_sync',name,off,256);values.append(np.frombuffer(runtime.read(name,offset=off,nbytes=256,sync=False),'<u2').copy())
   else:prepared=at=time.perf_counter()
   with runtime._lock:
    runtime._ready();assert not runtime._dirty
    for part,values in(('K',ks),('V',vs)):
     for kind in('spec','confirmed'):
      name=kind+part;skip=int(reuse)if kind=='spec'else 0
      if rows>skip:self._cache_host[name][position+skip:position+rows]=np.asarray(values[skip:],np.uint16);self._region('r40_cache_sync','r40_'+name,(position+skip)*256,(rows-skip)*256,True)
   # Appends write beyond every previous state view; old visible prefixes stay unchanged.
   for dst,values in zip(self._cpu_kv,(ks,vs)):dst[:,position:position+rows]=floats(np.asarray(values,np.uint16)).reshape(1,rows,128)
   k,v=(x[:,:position+rows]for x in self._cpu_kv);updated=dict(last_taps=raw[-1],position=position+rows,layers=[(k,v)]);self._state=updated;self._cpu_owner=updated;self._position=updated['position'];self.refresh_profile.append(dict(rows=rows,reused_root_rows=int(reuse),projected_rows=projected,prepare_ms=(prepared-start)*1000,native_ms=(at-prepared)*1000,return_ms=(time.perf_counter()-at)*1000,total_ms=(time.perf_counter()-start)*1000,runtime=stats));return updated
 return Partial,module

# Proposal-only API keeps the full cycle/gold API unchanged.
def make_direct(bundle,runtime,_full=make_direct):
 Base,module=_full(bundle,runtime)
 class FastProposal(Base):
  def __init__(self,bundle):
   super().__init__(bundle);self._fast_profile=[];self._coarse_host=np.frombuffer(runtime.views['r40_coarse_row0'],'<u2');self._exact_host=np.frombuffer(runtime.views['r40_exact_row0'],'<u2');self._coarse_sync=runtime._bo('r40_coarse_row0').sync;self._exact_sync=runtime._bo('r40_exact_row0').sync;self._bank_sync=runtime._bo('r40_exact_weights').sync;self._coarse_copy=np.empty(24576,np.uint16);self._from=runtime.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE;self._to=runtime.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
   assert self._coarse_host.shape==(24576,)and self._exact_host.shape==(512,)
   from fast_prepare import install
   install(runtime);at=time.perf_counter();runtime.prepare(runtime.manifest['experimental_verifier']['graph'],{'position':0});runtime.prepare(runtime.manifest['r37_features']['serial_bulk']['graph'],{'position':0});self.ready_prepare_ms=(time.perf_counter()-at)*1000
  def propose(self,state,anchor,audit=False):
   if state['position']+3>2048:raise ValueError('serial fallback required')
   start=time.perf_counter();self.bind_state(state);position=int(state['position']);packed=bits(state['last_taps']);self.write_row('r40_taps',packed);self.write_row('r40_embed',bits(self.embedding[int(anchor)]));self.write_row('r40_rope',self._angles[position]);prepared=time.perf_counter();body=runtime.execute('r41_proposal_join',{'position':position});at=time.perf_counter()
   with runtime._lock:
    runtime._ready();assert not runtime._dirty;self._coarse_sync(self._from);synced=time.perf_counter();np.copyto(self._coarse_copy,self._coarse_host)
   returned=time.perf_counter();scores=(self._coarse_copy.astype(np.uint32)<<16).view(np.float32);subset=np.argpartition(scores,len(scores)-512)[-512:];threshold=scores[subset].min();above=subset[scores[subset]>threshold];tied=np.flatnonzero(scores==threshold);n=512-len(above)
   if len(tied)>n:tied=tied[np.argpartition(self.vocab[tied],n-1)[:n]]
   chosen=np.concatenate((above,tied));chosen=chosen[np.argsort(self.vocab[chosen],kind='stable')];ids=self.vocab[chosen];selected=time.perf_counter();np.take(self._bank,chosen,axis=0,out=self._exact,mode='clip');copied=time.perf_counter();self._bank_sync(self._to);gathered=time.perf_counter();exact=runtime.execute('r41_exact',{});executed=time.perf_counter()
   with runtime._lock:
    runtime._ready();assert not runtime._dirty;self._exact_sync(self._from);read=time.perf_counter();logits=(self._exact_host.astype(np.uint32)<<16).view(np.float32)
   # IDs are ascending, so first argmax implements the global-ID tie rule.
   proposal=int(ids[int(np.argmax(logits))]);self._root_pending=(state,position,int(anchor),packed.tobytes());end=time.perf_counter();profile=dict(mode='proposal-only',prepare_ms=(prepared-start)*1000,native_body_ms=(at-prepared)*1000,return_ms=(returned-at)*1000,coarse_sync_ms=(synced-at)*1000,coarse_convert_ms=(returned-synced)*1000,selection_ms=(selected-returned)*1000,gather_cpu_ms=(copied-selected)*1000,gather_sync_ms=(gathered-copied)*1000,gather_ms=(gathered-selected)*1000,exact_ms=(end-gathered)*1000,exact_execute_ms=(executed-gathered)*1000,exact_sync_ms=(read-executed)*1000,exact_convert_pick_ms=(end-read)*1000,total_ms=(end-start)*1000,body_runtime=body,exact_runtime=exact);self._fast_profile.append(profile);self.cycle_profile.append(profile);self.head_profile.append(dict(selection_ms=profile['selection_ms'],gather_ms=profile['gather_ms'],exact_ms=profile['exact_ms'],runtime=exact));result=dict(proposal=proposal)
   if audit:result.update(candidate_ids=ids,candidate_logits=logits,coarse_logits=(self._coarse_copy.astype(np.uint32)<<16).view(np.float32))
   return result
 return FastProposal,module

# Exact native COPY128 appends keep confirmed KV resident. CPU layers are snapshots
# materialized on inspection; generation reads only position and last_taps.
from collections.abc import Mapping
import weakref
class DeviceState(Mapping):
 def __init__(self,owner,position,last_taps):
  self.owner=owner;self.position=int(position);self.last_taps=last_taps;self.layers=None
 def __getitem__(self,key):
  if key=='position':return self.position
  if key=='last_taps':return self.last_taps
  if key=='layers':
   if self.layers is None:self.layers=self.owner._materialize(self.position)
   return self.layers
  raise KeyError(key)
 def __iter__(self):return iter(('last_taps','position','layers'))
 def __len__(self):return 3

proposal_factory=make_direct
def make_resident(bundle,runtime,_old=proposal_factory):
 Base,module=_old(bundle,runtime)
 if 'r43_cache_publish'not in runtime.manifest:return Base,module
 class Resident(Base):
  def __init__(self,bundle):
   super().__init__(bundle);self._states=[]
   for rows in(1,2):
    for reuse in(0,1):runtime.prepare(f'r43_commit_{rows}_{reuse}',{'position':0})
  def _materialize(self,position):
   if not position:return [(np.empty((1,0,128),np.float32),np.empty((1,0,128),np.float32))]
   return [(floats(runtime.read('r40_confirmedK',nbytes=position*256)).reshape(1,position,128),floats(runtime.read('r40_confirmedV',nbytes=position*256)).reshape(1,position,128))]
  def bind_state(self,state):
   if self._state is state and self._position==state['position']:return
   # Preserve any externally retained old state's CPU snapshot before rebinding
   # overwrites the device prefix. Normal generation retains only its current one.
   live=[]
   for ref in self._states:
    old=ref()
    if old is not None:
     old['layers'];live.append(ref)
   self._states=live
   return super().bind_state(state)
  def append_confirmed(self,state,raw,tokens):
   started=time.perf_counter();raw=self.r(raw);tokens=np.asarray(tokens,np.int64);rows=len(tokens)
   if not rows:return state
   assert rows in(1,2)and int(state['position'])+rows<=2048
   self.bind_state(state);position=int(state['position']);pending=self._root_pending;reuse=bool(pending is not None and pending[0]is state and pending[1]==position and pending[2]==int(tokens[0])and pending[3]==bits(state['last_taps']).tobytes());self._root_pending=None;projected=rows-int(reuse)
   for row in range(projected):
    index=row+int(reuse);source=state['last_taps']if index==0 else raw[index-1];self.write_row('r40_taps',bits(source),row);self.write_row('r40_embed',bits(self.embedding[tokens[index]]),row);self.write_row('r40_rope',self._angles[position+index],row)
   prepared=time.perf_counter();stats=runtime.execute(f'r43_commit_{rows}_{int(reuse)}',{'position':position});at=time.perf_counter();updated=DeviceState(self,position+rows,raw[-1]);self._states=[ref for ref in self._states if ref()is not None];self._states.append(weakref.ref(updated));self._state=updated;self._position=updated['position'];self.refresh_profile.append(dict(mode='resident-cache',rows=rows,reused_root_rows=int(reuse),projected_rows=projected,prepare_ms=(prepared-started)*1000,native_ms=(at-prepared)*1000,return_ms=(time.perf_counter()-at)*1000,total_ms=(time.perf_counter()-started)*1000,runtime=stats));return updated
 return Resident,module

# Selected refresh keeps the existing projection graph and replaces Python
# read/convert/repack/upload calls with bounded copies between persistent spans.
def make_direct(bundle,runtime,_base=proposal_factory):
 Base,module=_base(bundle,runtime)
 class FastRefresh(Base):
  def __init__(self,bundle):
   super().__init__(bundle);self._states=[];self._cache_sync={n:runtime._bo('r40_'+n).sync for n in self._cache_host};self._refresh_host={n:np.frombuffer(runtime.views['r40_'+n],'<u2')for n in('refreshk','refreshkv')};self._refresh_sync={n:runtime._bo('r40_'+n).sync for n in self._refresh_host};self._rows_kv=np.empty((2,2,128),np.uint16)
   assert all(x.nbytes>=524288+512 for x in self._refresh_host.values())
  def _materialize(self,position):
   if not position:return [(np.empty((1,0,128),np.float32),np.empty((1,0,128),np.float32))]
   return [(floats(runtime.read('r40_confirmedK',nbytes=position*256)).reshape(1,position,128),floats(runtime.read('r40_confirmedV',nbytes=position*256)).reshape(1,position,128))]
  def bind_state(self,state):
   if self._state is state and self._position==state['position']:return
   live=[]
   for ref in self._states:
    old=ref()
    if old is not None:old['layers'];live.append(ref)
   self._states=live
   return super().bind_state(state)
  def append_confirmed(self,state,raw,tokens):
   started=time.perf_counter();raw=self.r(raw);tokens=np.asarray(tokens,np.int64);rows=len(tokens)
   if not rows:return state
   assert rows in(1,2)and int(state['position'])+rows<=2048;self.bind_state(state);position=int(state['position']);pending=self._root_pending;reuse=bool(pending is not None and pending[0]is state and pending[1]==position and pending[2]==int(tokens[0])and pending[3]==bits(state['last_taps']).tobytes());self._root_pending=None;projected=rows-int(reuse)
   for row in range(projected):
    index=row+int(reuse);source=state['last_taps']if index==0 else raw[index-1];self.write_row('r40_taps',bits(source),row);self.write_row('r40_embed',bits(self.embedding[tokens[index]]),row);self.write_row('r40_rope',self._angles[position+index],row)
   prepared=time.perf_counter();stats=runtime.execute('r41_refresh_join'if projected==1 else'r40_refresh_join',{})if projected else None;at=time.perf_counter()
   with runtime._lock:
    runtime._ready();assert not runtime._dirty
    if reuse:
     for j,part in enumerate(('K','V')):
      self._cache_sync['spec'+part](self._from,256,position*256);np.copyto(self._rows_kv[j,0],self._cache_host['spec'+part][position])
    root_returned=time.perf_counter()
    for row in range(projected):
     for j,(name,offset)in enumerate((('refreshk',row*524288),('refreshkv',row*524288+256))):
      self._refresh_sync[name](self._from,256,offset);np.copyto(self._rows_kv[j,row+int(reuse)],self._refresh_host[name][offset//2:offset//2+128])
    projected_returned=time.perf_counter()
    for j,part in enumerate(('K','V')):
     for kind in('spec','confirmed'):
      skip=int(reuse)if kind=='spec'else 0
      if rows>skip:
       np.copyto(self._cache_host[kind+part][position+skip:position+rows],self._rows_kv[j,skip:rows]);self._cache_sync[kind+part](self._to,(rows-skip)*256,(position+skip)*256)
   published=time.perf_counter();updated=DeviceState(self,position+rows,raw[-1]);self._states=[ref for ref in self._states if ref()is not None];self._states.append(weakref.ref(updated));self._state=updated;self._position=updated['position'];self.refresh_profile.append(dict(mode='persistent-BF16-copy',rows=rows,reused_root_rows=int(reuse),projected_rows=projected,prepare_ms=(prepared-started)*1000,native_ms=(at-prepared)*1000,root_return_ms=(root_returned-at)*1000,projected_return_ms=(projected_returned-root_returned)*1000,publish_ms=(published-projected_returned)*1000,state_ms=(time.perf_counter()-published)*1000,return_ms=(time.perf_counter()-at)*1000,total_ms=(time.perf_counter()-started)*1000,runtime=stats));return updated
 return FastRefresh,module
