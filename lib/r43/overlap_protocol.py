"""CPU protocol gates for fixed switch, serial EOS, caps and boundary fallback."""
import argparse,ast,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args();rows=[]
def packed(token):return np.full(6144,token,np.float32).tobytes()
def features(items):return np.stack([np.frombuffer(x,np.float32).reshape(4,1536)for x in items])
source=Path(__file__).with_name('loop43.py');node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='generate')
for ready in(0,1,2,10000):
 for eos_index in(0,1,4,9):
  for boundary in(False,True):
   count=eos_index+1;outputs=list(range(21,21+count));outputs[-1]=16;prompt=[1]*(2047 if boundary else 23);cap=2 if boundary else count+2;delivered=[]
   class Serial:
    max_seq=2048;model={'eos':[1,16]}
    def __init__(self):self.length=0;self.decoded=0
   s=Serial()
   class Worker:
    pending=None
    def start(self,raw,tokens):assert len(delivered)==1;self.pending=1;self.complete_ns=time.perf_counter_ns()if ready==0 else None;return {'job':1}
    def done(self):return self.complete_ns is not None
    def result(self):assert self.done();self.pending=None;return dict(position=len(prompt)-1,last_taps=np.zeros((4,1536),np.float32),layers=[]),dict(done_ns=self.complete_ns,compute_ms=0)
    def cancel(self):return self.pending
   worker=Worker()
   def prefill(serial,tokens):serial.length=len(tokens);return dict(token=outputs[0],features=[packed(t)for t in tokens],elapsed_s=0)
   def decode(serial,token):
    assert token==outputs[serial.decoded];pos=serial.length;serial.length+=1;serial.decoded+=1
    if serial.decoded>=ready and worker.complete_ns is None:worker.complete_ns=time.perf_counter_ns()
    return dict(token=outputs[serial.decoded],features=packed(token),position=pos)
   class Verifier:
    def __init__(self,serial):self.serial=serial
    def stage(self,tokens):self.inputs=tokens;self.greedy=[outputs[s.decoded+1],outputs[min(s.decoded+2,count-1)]];return dict(greedy=self.greedy,tap_s=0)
    def commit(self,accepted):
     n=1+accepted;s.length+=n;s.decoded+=n;return dict(features=[packed(t)for t in self.inputs[:n]])
   class Drafter:
    primer=worker;_root_pending=None
    def catchup(self,state,raw,tokens):assert tokens==outputs[:s.decoded];assert all(np.all(x==t)for x,t in zip(raw,tokens));state['position']+=len(tokens);return state
    def cycle(self,state,anchor):assert state['position']==s.length-1;return dict(proposal=outputs[s.decoded+1]if s.decoded%2==0 else 999)
    def append_confirmed(self,state,raw,tokens):assert all(np.all(x==t)for x,t in zip(raw,tokens));state['position']+=len(tokens);return state
   scope=dict(time=time,np=np,features=features,TimedFeatures=Verifier,prefill_with_features=prefill,decode_with_features=decode);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);got=scope['generate'](s,Drafter(),prompt,cap,emit=delivered.append);expected=outputs[:min(cap,2 if boundary else len(outputs))];assert got['tokens']==delivered==expected,(ready,eos_index,boundary,got,expected)
   if eos_index==0 or(eos_index<=ready and not boundary):assert not got['cycles']
   if got['switch']is not None:assert got['switch']['observed_boundary']==min(ready,len(outputs)-1)
   if boundary and eos_index>0:assert len(got['fallback'])==1 and got['logical_length']==2048 and got['drafter_position']==2047 and not got['cycles']
   rows.append(dict(ready_boundary=ready,EOS_index=eos_index,boundary=boundary,result='PASS',M2_cycles=len(got['cycles']),serial_rows=got['serial_phase_rows'],KV_materialized_position=got['KV_materialized_position'],cancelled_at_EOS=got['prime_cancelled_at_eos']))
r=dict(result='PASS',cases=len(rows),rows=rows,scope='actual generate function compiled unchanged from source AST; only NPU/worker dependencies mocked; first-ready boundary, accepted/rejected paths, EOS before prime, capped output, primed/unprimed2047 fallback, exact callback stream');a.output.write_text(json.dumps(r,indent=2));print(json.dumps(dict(result='PASS',cases=len(rows))),flush=True)
