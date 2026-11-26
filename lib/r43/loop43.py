"""Fixed rule: switch at the first token boundary observing CPU prime completion."""
import time
import numpy as np
from spec_loop import features,TimedFeatures
from tap_session_v2 import prefill_with_features,decode_with_features

def generate(serial,drafter,prompt,limit,emit=None):
 if limit<1:raise ValueError('positive output cap required')
 begin=time.perf_counter();events=[]
 def publish(token):
  events.append(time.perf_counter())
  if emit is not None:emit(int(token))
 prefill=prefill_with_features(serial,prompt);known=prefill['token'];output=[known];publish(known);ttft=events[0]-begin;decode_start=events[0];state=None;cycles=[];fallback=[];serial_rows=[];catchup=None;prime=None;eos=set(serial.model['eos']);verifier=None;observations=[];worker=drafter.primer;drafter._root_pending=None
 if known not in eos and len(output)<limit and serial.length<serial.max_seq:prime=worker.start(prefill['features'],prompt)
 while len(output)<limit and known not in eos and serial.length<serial.max_seq:
  boundary_ns=time.perf_counter_ns()
  if state is None:
   entered_ns=boundary_ns;done=worker.done();boundary_ns=time.perf_counter_ns();observations.append(dict(boundary=len(output)-1,at_ns=boundary_ns,boundary_entry_ns=entered_ns,done=done))
   if done:
    at=time.perf_counter();state,record=worker.result();assert state is not None;raw=features([x['features']for x in serial_rows])if serial_rows else np.empty((0,4,1536),np.float32);state=drafter.catchup(state,raw,[x['input']for x in serial_rows]);assert state['position']==serial.length-1;catchup=dict(rows=len(serial_rows),elapsed_ms=(time.perf_counter()-at)*1000,worker=record,observed_boundary=len(output)-1,observed_ns=boundary_ns);verifier=TimedFeatures(serial)
   else:
    at=time.perf_counter();inp=known;pos=serial.length;got=decode_with_features(serial,known);known=got['token'];output.append(known);publish(known);row=dict(input=inp,position=pos,features=got['features'],seconds=time.perf_counter()-at);serial_rows.append(row)
    if pos==2047:fallback.append(dict(seconds=row['seconds'],refresh_s=0,priming_serial=True))
    continue
  if state['position']!=serial.length-1:raise RuntimeError('target/drafter positions diverged')
  if state['position']+3>2048:
   at=time.perf_counter();inp=known;got=decode_with_features(serial,known);t=time.perf_counter();state=drafter.append_confirmed(state,features([got['features']]),[inp]);refresh=time.perf_counter()-t;known=got['token'];output.append(known);publish(known);fallback.append(dict(seconds=time.perf_counter()-at,refresh_s=refresh,priming_serial=False));continue
  start=time.perf_counter();at=start;draft=drafter.propose(state,known)if hasattr(drafter,'propose')else drafter.cycle(state,known);draft_s=time.perf_counter()-at;proposal=int(draft['proposal']);at=time.perf_counter()
  if proposal in(1,16):
   # Only the root row is logically committed, whether EOS is accepted or rejected.
   # The counterfactual M2 row1 is speculative and is never part of the visible prefix.
   inp=known;before=serial.length;got=decode_with_features(serial,inp);verify_s=time.perf_counter()-at;accepted=int(got['token']==proposal);at=time.perf_counter();state=drafter.append_confirmed(state,features([got['features']]),[inp]);refresh_s=time.perf_counter()-at;known=int(got['token']);output.append(known);publish(known);assert serial.length==before+1 and state['position']==serial.length-1;elapsed=time.perf_counter()-start;cycles.append(dict(position=state['position'],accepted_drafts=accepted,emitted=1,mode='EOS-C1',proposal=proposal,refresh_rows=1,draft_s=draft_s,verify_s=verify_s,taps_s=0,refresh_s=refresh_s,host_s=elapsed-draft_s-verify_s-refresh_s,seconds=elapsed));continue
  verified=verifier.stage([known,proposal]);verify_s=time.perf_counter()-at;accepted=int(verified['greedy'][0]==proposal);committed=verifier.commit(accepted);at=time.perf_counter();state=drafter.append_confirmed(state,features(committed['features']),[known,proposal][:accepted+1]);refresh_s=time.perf_counter()-at;emitted=[]
  if accepted:
   emitted.append(proposal)
   if proposal not in eos and len(output)+1<limit:emitted.append(int(verified['greedy'][1]))
  else:emitted.append(int(verified['greedy'][0]))
  for token in emitted:output.append(token);publish(token)
  known=emitted[-1];elapsed=time.perf_counter()-start;cycles.append(dict(position=state['position'],accepted_drafts=accepted,emitted=len(emitted),mode='M2',proposal=proposal,refresh_rows=accepted+1,draft_s=draft_s,verify_s=verify_s,taps_s=verified['tap_s'],refresh_s=refresh_s,host_s=elapsed-draft_s-verify_s-refresh_s,seconds=elapsed))
 end=events[-1];cancelled=worker.cancel()if state is None else None;logical=serial.length-1;result=dict(prefill_stats=prefill.get('stats'),tokens=output,ttft_s=ttft,prefill_s=prefill['elapsed_s'],prime_s=(catchup['worker']['compute_ms']/1000 if catchup else None),decode_s=end-decode_start,wall_s=end-begin,call_return_s=time.perf_counter()-begin,cycles=cycles,fallback=fallback,serial_phase_rows=len(serial_rows),serial_phase_s=sum(x['seconds']for x in serial_rows),switch=catchup,prime_submit=prime,boundary_observations=observations,prime_cancelled_at_eos=bool(cancelled is not None and known in eos),never_entered_M2=not any(x['mode']=='M2'for x in cycles),never_entered_proposal=not cycles,E2=sum(x['emitted']for x in cycles)/len(cycles)if cycles else None,decode_ms_per_token=(end-decode_start)*1000/(len(output)-1)if len(output)>1 else None,stopped_eos=known in eos,logical_length=serial.length,drafter_position=logical,KV_materialized_position=state['position']if state is not None else None,eos_proposal_cycles=sum(x['mode']=='EOS-C1'for x in cycles),scope='wall/TTFT are actual emit callback timestamps; EOS cancels unused priming; drafter_position is logical confirmed-pair cursor, KV materialization is separate')
 if catchup:
  assert catchup['worker']['done_ns']<=catchup['observed_ns'];assert not any(x['done']for x in observations[:-1]);result['fixed_switch_verified']=True
 return result
