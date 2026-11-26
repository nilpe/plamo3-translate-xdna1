"""Experimental exact M2 chain, no oracle and no target-assisted proposal."""
import time
import numpy as np
from tap_session_v2 import prefill_with_features,decode_with_features
from feature_session import FeatureVerifySession,read_features
from verify_session import VerifySession

def features(rows):
 b=b''.join(rows);return(np.frombuffer(b,'<u2').astype(np.uint32)<<16).view(np.float32).reshape(len(rows),4,1536)

class TimedFeatures(FeatureVerifySession):
 def stage(self,tokens):
  t=time.perf_counter();got=VerifySession.stage(self,tokens);at=time.perf_counter();self.pending_features=read_features(self.serial,self.rows);got['features']=self.pending_features;got['tap_s']=time.perf_counter()-at;got['elapsed_s']=time.perf_counter()-t;return got

def generate(serial,drafter,prompt,limit):
 if limit<1:raise ValueError('positive output cap required')
 begin=time.perf_counter();prefill=prefill_with_features(serial,prompt);at=time.perf_counter();state=drafter.prime(features(prefill['features']),prompt);prime_s=time.perf_counter()-at;ttft=time.perf_counter()-begin;known=prefill['token'];output=[known];verifier=TimedFeatures(serial);cycles=[];fallback=[];eos=set(serial.model['eos']);decode_start=time.perf_counter()
 while len(output)<limit and known not in eos and serial.length<serial.max_seq:
  if state['position']!=serial.length-1:raise RuntimeError('target/drafter absolute positions diverged')
  if state['position']+3>2048:
   t=time.perf_counter();got=decode_with_features(serial,known);at=time.perf_counter();state=drafter.append_confirmed(state,features([got['features']]),[known]);refresh_s=time.perf_counter()-at;known=got['token'];output.append(known);fallback.append(dict(seconds=time.perf_counter()-t,refresh_s=refresh_s));continue
  cycle_start=time.perf_counter();at=time.perf_counter();draft=drafter.cycle(state,known);draft_s=time.perf_counter()-at;proposal=int(draft['proposal']);at=time.perf_counter();verified=verifier.stage([known,proposal]);verify_s=time.perf_counter()-at
  accepted=int(verified['greedy'][0]==proposal);committed=verifier.commit(accepted);true_tokens=[known,proposal][:accepted+1];at=time.perf_counter();state=drafter.append_confirmed(state,features(committed['features']),true_tokens);refresh_s=time.perf_counter()-at
  emitted=[]
  if accepted:
   emitted.append(proposal)
   if proposal not in eos and len(output)+len(emitted)<limit:emitted.append(int(verified['greedy'][1]))
  else:emitted.append(int(verified['greedy'][0]))
  output.extend(emitted);known=emitted[-1];elapsed=time.perf_counter()-cycle_start;cycles.append(dict(position=state['position'],accepted_drafts=accepted,emitted=len(emitted),draft_s=draft_s,verify_s=verify_s,taps_s=verified['tap_s'],refresh_s=refresh_s,host_s=elapsed-draft_s-verify_s-refresh_s,seconds=elapsed))
 decode_s=time.perf_counter()-decode_start
 return dict(tokens=output,ttft_s=ttft,prefill_s=prefill['elapsed_s'],prime_s=prime_s,decode_s=decode_s,wall_s=time.perf_counter()-begin,cycles=cycles,fallback=fallback,E2=sum(c['emitted']for c in cycles)/len(cycles)if cycles else None,decode_ms_per_token=1000*decode_s/(len(output)-1)if len(output)>1 else None,stopped_eos=known in eos,logical_length=serial.length,drafter_position=state['position'])
