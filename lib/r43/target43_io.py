"""Exact confirmed-root KV reuse and persistent coalesced host buffer views."""
import time
import numpy as np
from target43_base import make_direct as base_factory,bits,floats

def make_direct(bundle,runtime):
 Base,module=base_factory(bundle,runtime);spec=runtime.manifest['r40_cycle'];assert spec.get('io')=='coalesced-root-KV','IO path requires its separately audited manifest'
 class NativeIO(Base):
  def __init__(self,bundle):
   super().__init__(bundle);self.phase='target-C-one-context-root-KV';self._root_pending=None;self._head_return_ready=False;self._cache_host={name:np.frombuffer(runtime.views['r40_'+name],'<u2').reshape(2048,128)for name in('specK','specV','confirmedK','confirmedV')}
   for name in('r40_cache_sync','r40_spec_sync','r40_refresh_pair_sync','r40_head_return_sync'):runtime._bo(name)
  def prime(self,raw,tokens):
   if runtime.manifest.get('r41_bootstrap')!='hybrid':return super().prime(raw,tokens)
   from cpu_bootstrap import Bootstrap
   if not hasattr(self,'_fast_boot'):
    def norm(x,gamma):
     width=x.shape[1];out=np.empty_like(x);pad=np.zeros((32,width),np.float32)
     expected=floats(runtime.read(runtime.manifest['graphs']['r41_boot_norm'+str(width)]['steps'][0]['args'][1],nbytes=width*2,sync=False));assert np.array_equal(bits(gamma),bits(expected))
     for start in range(0,len(x),32):
      size=min(32,len(x)-start);pad.fill(0);pad[:size]=x[start:start+size];runtime.write('r41_boot_in',bits(pad));runtime.execute('r41_boot_norm'+str(width),{});out[start:start+size]=floats(runtime.read('r41_boot_out',nbytes=32*width*2)).reshape(32,width)[:size]
     return out
    self._fast_boot=Bootstrap(self,module['target'],norm=norm)
   self._root_pending=None;state=self._fast_boot(raw,tokens);self.bind_state(state);return state
  def bind_state(self,state):
   if self._state is not state or self._position!=state['position']:self._root_pending=None
   return super().bind_state(state)
  def read_row(self,name,n,row=0):
   if row==0 and name=='r40_hidden':runtime.sync_from('r40_head_return_sync');self._head_return_ready=True;return floats(runtime.read('r40_hidden_row0',nbytes=n*2,sync=False))
   if row==0 and name=='r40_coarse'and self._head_return_ready:self._head_return_ready=False;return floats(runtime.read('r40_coarse_row0',nbytes=n*2,sync=False))
   return super().read_row(name,n,row)
  def cycle(self,state,anchor):
   out=super().cycle(state,anchor);self._root_pending=(state,int(state['position']),int(anchor),bits(state['last_taps']).tobytes());return out
  def append_confirmed(self,state,raw,tokens):
   start=time.perf_counter();raw=self.r(raw);tokens=np.asarray(tokens,np.int64);rows=len(tokens)
   if not rows:return state
   assert rows in(1,2)and state['position']+rows<=2048;self.bind_state(state);position=int(state['position']);pending=self._root_pending;reuse=bool(pending is not None and pending[0]is state and pending[1]==position and pending[2]==int(tokens[0])and pending[3]==bits(state['last_taps']).tobytes());self._root_pending=None;ks=[];vs=[];projected=rows-int(reuse);stats=None
   if reuse:
    # These device rows were computed from the same confirmed h[j] and token j+1.
    # Only append_confirmed publishes them; cycle never modifies confirmed KV.
    runtime.sync_from('r40_spec_sync');ks.append(floats(runtime.read('r40_specK',offset=position*256,nbytes=256,sync=False)));vs.append(floats(runtime.read('r40_specV',offset=position*256,nbytes=256,sync=False)))
   if projected:
    source=np.concatenate((state['last_taps'][None],raw[:-1]),0)[int(reuse):]
    for row in range(projected):self.write_row('r40_taps',bits(source[row]),row);self.write_row('r40_embed',bits(self.embedding[tokens[row+int(reuse)]]),row)
    for row in range(2):self.write_row('r40_rope',self._angles[min(position+int(reuse)+row,2047)],row)
    prepared=time.perf_counter();stats=runtime.execute('r41_refresh_join'if projected==1 else'r40_refresh_join',{});at=time.perf_counter();runtime.sync_from('r40_refresh_pair_sync')
    for row in range(projected):ks.append(floats(runtime.read('r40_refreshk',offset=row*524288,nbytes=256,sync=False)));vs.append(floats(runtime.read('r40_refreshkv',offset=row*524288+256,nbytes=256,sync=False)))
   else:prepared=at=time.perf_counter()
   # One persistent bounded BO replaces per-position BO construction and uploads.
   # Host writes follow completed commands, like the persistent gathered exact BO.
   with runtime._lock:
    runtime._ready();assert not runtime._dirty,'cannot publish KV over pending host writes'
    for part,values in(('K',ks),('V',vs)):
     packed=bits(np.asarray(values,np.float32))
     for kind in('spec','confirmed'):self._cache_host[kind+part][position:position+rows]=packed
    runtime.sync_to('r40_cache_sync')
   old=state['layers'][0];k=np.concatenate((old[0],np.asarray(ks,np.float32)[None]),1);v=np.concatenate((old[1],np.asarray(vs,np.float32)[None]),1);updated=dict(last_taps=raw[-1],position=position+rows,layers=[(k,v)]);self._state=updated;self._position=updated['position'];self.refresh_profile.append(dict(rows=rows,reused_root_rows=int(reuse),projected_rows=projected,prepare_ms=(prepared-start)*1000,native_ms=(at-prepared)*1000,return_ms=(time.perf_counter()-at)*1000,total_ms=(time.perf_counter()-start)*1000,runtime=stats));return updated
 return NativeIO,module
