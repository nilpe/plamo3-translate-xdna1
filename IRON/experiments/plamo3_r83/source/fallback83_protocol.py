"""Actual loop protocol: two EOS IDs, accepted/rejected EOS, true pairs and boundaries."""
import argparse,ast,base64,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--mutant',choices=['none','never','always','inverted'],default='none');a=p.parse_args();records=[];sys.path.insert(0,str(Path(__file__).parent))
import gate83
from gate83 import confidence,gate_fires
gate83_ref=gate83.reference_confidence
MUT={'none':gate_fires,'never':lambda c,t:False,'always':lambda c,t:True,'inverted':lambda c,t:t is not None and c>t}[a.mutant]
# R83: the gate is exercised through injected candidate logits. eos id 1 -> statistic top1_prob (tau 0.5), eos id 16 -> statistic margin (tau 2.0).
STAT_TAU={1:('top1_prob',0.5),16:('margin',2.0)}
def mklogits(stat,kind):
 x=np.zeros(512,np.float32)
 if stat=='top1_prob':
  if kind=='low':x[:]=10.0
  elif kind=='high':x[0]=10.0
  else:x[:]=-200.0;x[:2]=0.0
 else:
  if kind=='low':x[0]=3.0;x[1]=2.0
  elif kind=='high':x[0]=5.0
  else:x[0]=4.0;x[1]=2.0
 return x
PATTERNS=('off','all','first','alt','equal')
KC=[0]
def pats():
 # every base case runs once with the gate off (the unchanged R54 matrix) and once with one injected gated pattern, cycling through all four
 KC[0]+=1;return('off',PATTERNS[1+KC[0]%4])
def want_low(pattern,n):return pattern=='all'or(pattern=='first'and n==0)or(pattern=='alt'and n%2==1)

def packed(t):return np.full(6144,t,np.float32).tobytes()
def features(rows):return np.stack([np.frombuffer(x,np.float32).reshape(4,1536)for x in rows])
source=Path(__file__).with_name('loop83.py');node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='generate');WorkerUnavailable=type("WorkerUnavailable",(RuntimeError,),{});scope=dict(time=time,np=np,base64=base64,features=features,WorkerUnavailable=WorkerUnavailable,confidence=confidence,gate_fires=MUT)
const=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)and n.targets[0].id=='SENTENCE_FINAL_IDS');REAL=frozenset(ast.literal_eval(const.value));assert len(REAL)==1170 and 51353 in REAL and 16 not in REAL and 1 not in REAL,'declared R83 sentence-final table';FIN={'real':REAL,'inject':frozenset({22,24,26})}
for fin in('real','inject'):
 for fault in('start','done','result','prime-submitted','prime-caught-up','target-committed','serial-output'):
  for eos_id in(1,16):
   for policy in('normal','force16'):
    for ready in(0,1,3,10000):
     for N in(23,2045,2046,2047):
      for end in(0,2,4,9):
       for pattern in pats():
        output=list(range(21,22+end));output[-1]=eos_id;prompt=[2]*N;delivered=[];eos_equivalence=[];refreshes=[]
        class Serial:
         max_seq=2048;model={'eos':[1,16]}
         def __init__(self):self.length=0;self.decoded=0;self.inputs=[]
        serial=Serial();serial.decode=lambda token:decode(serial,token)
        class Worker:
         pending=None;complete_ns=None;alive=True
         def healthy(self):return self.alive
         def close(self):self.alive=False
         def start(self,raw,tokens):
          if fault=='start':raise WorkerUnavailable('start injection')
          assert delivered==[output[0]];self.pending=1;self.complete_ns=time.perf_counter_ns()if ready==0 else None;return {'job':1}
         def done(self):
          if fault=='done':raise WorkerUnavailable('done injection')
          return self.complete_ns is not None
         def result(self):
          if fault=='result':raise WorkerUnavailable('result injection')
          self.pending=None;return dict(position=N-1,last_taps=np.full((4,1536),2,np.float32),pairs=[(2,2)]*(N-1)),dict(done_ns=self.complete_ns,compute_ms=0)
         def cancel(self):return self.pending
        worker=Worker()
        def prefill(s,tokens):s.length=len(tokens);s.inputs=list(tokens);return dict(token=output[0],features=[packed(t)for t in tokens],elapsed_s=0)
        def decode(s,token):
         assert token==output[s.decoded];position=s.length;s.inputs.append(token);s.length+=1;s.decoded+=1
         if s.decoded>=ready and worker.complete_ns is None:worker.complete_ns=time.perf_counter_ns()
         return dict(token=output[s.decoded],features=packed(token),position=position)
        class Verifier:
         def __init__(self,s):self.s=s
         def stage(self,tokens):assert tokens[1]not in(1,16),'EOS proposal must use C1';assert not drafter.last['gated'],'a gated cycle must use C1';self.tokens=tokens;return dict(greedy=[output[serial.decoded+1],output[min(serial.decoded+2,len(output)-1)]],tap_s=0)
         def commit(self,accepted):serial.inputs.extend(self.tokens[:1+accepted]);serial.length+=1+accepted;serial.decoded+=1+accepted;return dict(features=[packed(t)for t in self.tokens[:1+accepted]])
        class Drafter:
         primer=worker;_root_pending=None
         def __init__(self):self.last=None;self.calls=0;self.log=[]
         def catchup(self,s,raw,tokens):return self.append_confirmed(s,raw,tokens)
         def cycle(self,s,anchor):
          n=self.calls;kind='low'if pattern=='off'else'equal'if pattern=='equal'else'low'if want_low(pattern,n)else'high';stat_name=STAT_TAU[eos_id][0];gate_expected=want_low(pattern,n)
          assert s['position']==serial.length-1 and s['position']+3<=2048;self.calls+=1;proposal=int(policy[-1])if policy=='force1'else 16 if policy=='force16'else output[serial.decoded+1]if serial.decoded%2==0 else 999;self.last=dict(proposal=proposal,position=s['position'],length=serial.length,pairs=s['pairs'].copy(),inputs=serial.inputs.copy(),anchor=anchor,gated=gate_expected,n=n);self.log.append((n,proposal,gate_expected));return dict(proposal=proposal,candidate_logits=mklogits(stat_name,kind))
         def append_confirmed(self,s,raw,tokens):
          pairs=s['pairs'].copy();prev=int(s['last_taps'][0,0])
          for row,t in zip(raw,tokens):assert np.all(row==t);pairs.append((prev,int(t)));prev=int(t)
          updated=dict(position=s['position']+len(tokens),pairs=pairs,last_taps=np.full((4,1536),prev,np.float32));assert pairs==list(zip(serial.inputs[:-1],serial.inputs[1:]));assert len(pairs)==serial.length-1;refreshes.append(len(tokens))
          if self.last is not None and(self.last['proposal']in(1,16)or self.last['gated']):
           before=self.last;assert len(tokens)==1 and updated['position']==before['position']+1 and serial.length==before['length']+1;counterfactual_inputs=before['inputs']+[before['anchor']];assert serial.inputs==counterfactual_inputs;assert pairs==list(zip(counterfactual_inputs[:-1],counterfactual_inputs[1:]));assert s['pairs']==before['pairs'];eos_equivalence.append(dict(proposal=before['proposal'],rows=1,KV_pairs_equal=True,cursor_equal=True));self.last=None
          return updated
        drafter=Drafter();fallbacks=[]
        def hook(event,data):
         if event==fault:worker.alive=False
        def emit(token):
         delivered.append(token)
         if fault=='serial-output' and len(delivered)==2:worker.alive=False
        GSTAT,GTAU=STAT_TAU[eos_id];LOGON=not(pattern=='alt'and eos_id==16);scope.update(LOG_LOGITS=LOGON,GATE_STAT=GSTAT,GATE_TAU=None if pattern=='off'else GTAU,SENTENCE_FINAL=FIN[fin],TimedFeatures=Verifier,prefill_with_features=prefill,decode_with_features=decode);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);got=scope['generate'](serial,drafter,prompt,len(output)+2,emit=emit,on_fallback=lambda e:fallbacks.append(str(e)),hook=hook);expected=output[:2048-N+1];assert got['tokens']==delivered==expected,(eos_id,policy,ready,N,end);assert len(fallbacks)<=1;assert bool(got['worker_fallback'])==bool(fallbacks);assert serial.inputs==prompt+expected[:-1];assert got['eos_proposal_cycles']+got['gated_cycles']==len(eos_equivalence);assert all(c['refresh_rows']==1 and c['emitted']==1 for c in got['cycles']if c['mode']in('EOS-C1','GATE-C1'))
       # R83 exact gated-path assertions: expected mode of every cycle from the drafter's own call log and the injected confidence (independent of the controller's comparison)
       want=['EOS-C1'if prop in(1,16)else'GATE-C1'if low else'M2'for _,prop,low in drafter.log];assert[c['mode']for c in got['cycles']]==want,(pattern,want,[c['mode']for c in got['cycles']]);assert[c['position']for c in got['cycles']if c['mode']=='GATE-C1']==[c['position']for c in got['cycles']if c['mode']=='GATE-C1'];assert got['gated_cycles']==want.count('GATE-C1')
       assert all(c['gated']==low and(c['gate_tau']is None)==(pattern=='off')and c['accepted_drafts']in(0,1)for c,(_,_,low)in zip(got['cycles'],drafter.log));assert all(c['accepted_drafts']==0 for c in got['cycles']if c['mode']=='GATE-C1')
       if pattern=='all':assert not any(c['mode']=='M2'for c in got['cycles'])
       if pattern in('off','equal'):assert not any(c['mode']=='GATE-C1'for c in got['cycles'])
       if pattern=='first'and got['cycles']:assert got['cycles'][0]['mode']in('GATE-C1','EOS-C1')and sum(c['mode']=='GATE-C1'for c in got['cycles'])<=1
       if pattern=='alt':assert all(i%2==1 for i,c in enumerate(got['cycles'])if c['mode']=='GATE-C1')
       for c in got['cycles']:
        if pattern!='off':
         assert c['gate_stat']==GSTAT and c['conf']is not None
         if LOGON:x=np.frombuffer(base64.b64decode(c['logits_b64']),np.float32);assert c['conf']==confidence(x,GSTAT,GTAU),(pattern,c['conf']);assert c['conf']==gate83_ref(x,GSTAT)or(c['conf']<GTAU)==(gate83_ref(x,GSTAT)<GTAU)
         else:assert c['logits_b64']is None,'production records must not carry logits'
        b=ready
        while b<=end and output[b]in FIN[fin]:b+=1
        bstar=b;lim=len(output)+2;reach=bstar<end and bstar+1<lim and N+bstar<2048
        if not fallbacks:
         if reach:assert got['switch']is not None and got['switch']['observed_boundary']==bstar and got['serial_phase_rows']==bstar and got['boundary_observations'][-1]['eligible'],(fault,fin,ready,N,end,bstar)
         else:assert got['switch']is None and got['never_entered_proposal'] and not got['cycles'],(fault,fin,ready,N,end,bstar)
        expected_fault={'start':end>0,'done':end>0,'prime-submitted':end>0,'result':reach,'prime-caught-up':reach,'serial-output':((end>1 and N<2047)if bstar>=1 else None)}.get(fault)
        if expected_fault is not None:assert bool(fallbacks)==expected_fault,(fault,fin,ready,N,end,bstar,fallbacks)
        if N==2047:assert not drafter.calls
        if end==0 or(end<=ready and N==23):assert not got['cycles']
        records.append(dict(pattern=pattern,gated=got['gated_cycles'],final_table=fin,fault=fault,fallback=bool(fallbacks),EOS=eos_id,policy=policy,ready=ready,prompt_length=N,EOS_index=end,result='PASS',EOS_C1=len(eos_equivalence),M2=sum(c['mode']=='M2'for c in got['cycles']),boundary_fallback=len(got['fallback']),cursor=got['logical_length'],materialized=got['KV_materialized_position']))
r=dict(result='PASS',cases=len(records),fallback_cases=sum(x['fallback']for x in records),rows=records,scope='actual R83 loop AST, worker start/done/result failures and deaths at submit/catch-up/target commit/serial emission; target active KV inputs and visible tokens equal independent serial prefix, one log transition, EOS and2045–2047');a.output.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k!='rows'}),flush=True)
