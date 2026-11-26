"""Exact value partition and bank-order gather; global-ID tie is explicit."""
import time
import numpy as np

def select(scores,ids,k=512):
 threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
 above=np.flatnonzero(scores>threshold);ties=np.flatnonzero(scores==threshold);left=k-len(above)
 if len(ties)>left:ties=ties[np.argpartition(ids[ties],left-1)[:left]]
 # Local bank order enables forward gathers; no change to the candidate set.
 chosen=np.sort(np.concatenate((above,ties)));assert len(chosen)==k
 return chosen

def self_test():
 rng=np.random.default_rng(44);ids=rng.permutation(24576).astype(np.int64);cases=0
 for i in range(256):
  raw=rng.integers(0,65536,len(ids),dtype=np.uint16);raw[(raw&32767)>32640]=0;scores=(raw.astype(np.uint32)<<16).view(np.float32)
  if i%4==0:scores[:]=rng.choice(np.array([0.,-0.,1.,-1.,np.inf,-np.inf],np.float32),len(ids))
  if i%4==1:scores[:]=1.
  expected=np.sort(ids[np.lexsort((ids,-scores))[:512]]);chosen=select(scores,ids);assert np.array_equal(np.sort(ids[chosen]),expected);assert np.all(chosen[1:]>chosen[:-1]);logits=np.array([2.,2.,-1.],np.float32);tokens=np.array([9,3,1]);assert int(tokens[logits==logits.max()].min())==3;cases+=1
 return cases

def make_direct(bundle,runtime,factory):
 assert runtime.manifest['r44_selected']['host']=='order','host policy must be sealed before first use'
 Base,module=factory(bundle,runtime)
 from target_drafter import bits
 class Ordered(Base):
  def propose(self,state,anchor,audit=False):
   if state['position']+3>2048:raise ValueError('serial fallback required')
   start=time.perf_counter();self.bind_state(state);position=int(state['position']);packed=bits(state['last_taps']);self.write_row('r40_taps',packed);self.write_row('r40_embed',bits(self.embedding[int(anchor)]));self.write_row('r40_rope',self._angles[position]);prepared=time.perf_counter();body=runtime.execute('r41_proposal_join',{'position':position});at=time.perf_counter()
   with runtime._lock:
    runtime._ready();assert not runtime._dirty;self._coarse_sync(self._from);synced=time.perf_counter();np.copyto(self._coarse_copy,self._coarse_host)
   returned=time.perf_counter();scores=(self._coarse_copy.astype(np.uint32)<<16).view(np.float32);chosen=select(scores,self.vocab);ids=self.vocab[chosen];selected=time.perf_counter();np.take(self._bank,chosen,axis=0,out=self._exact,mode='clip');copied=time.perf_counter();self._bank_sync(self._to);gathered=time.perf_counter();exact=runtime.execute('r41_exact',{});executed=time.perf_counter()
   with runtime._lock:
    runtime._ready();assert not runtime._dirty;self._exact_sync(self._from);read=time.perf_counter();logits=(self._exact_host.astype(np.uint32)<<16).view(np.float32)
   proposal=int(ids[logits==logits.max()].min());self._root_pending=(state,position,int(anchor),packed.tobytes());end=time.perf_counter();profile=dict(mode='proposal-only-value-partition-bank-order',prepare_ms=(prepared-start)*1000,native_body_ms=(at-prepared)*1000,return_ms=(returned-at)*1000,coarse_sync_ms=(synced-at)*1000,coarse_convert_ms=(returned-synced)*1000,selection_ms=(selected-returned)*1000,gather_cpu_ms=(copied-selected)*1000,gather_sync_ms=(gathered-copied)*1000,gather_ms=(gathered-selected)*1000,exact_ms=(end-gathered)*1000,exact_execute_ms=(executed-gathered)*1000,exact_sync_ms=(read-executed)*1000,exact_convert_pick_ms=(end-read)*1000,total_ms=(end-start)*1000,body_runtime=body,exact_runtime=exact);self._fast_profile.append(profile);self.cycle_profile.append(profile);self.head_profile.append(dict(selection_ms=profile['selection_ms'],gather_ms=profile['gather_ms'],exact_ms=profile['exact_ms'],runtime=exact));result=dict(proposal=proposal)
   if audit:result.update(candidate_ids=ids,candidate_logits=logits,coarse_logits=scores)
   return result
 return Ordered,module
