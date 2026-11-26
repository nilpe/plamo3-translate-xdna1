# SPDX-License-Identifier: Apache-2.0
"""Logical commit protocol for a fixed-size, exact chain verifier."""
import operator,struct,time
class VerifySession:
 def __init__(self,serial):
  from fast_prepare import install
  install(serial.runtime)
  self.serial=serial;self.runtime=serial.runtime;self.spec=self.runtime.manifest['experimental_verifier'];self.rows=self.spec['M'];self.pending=None
 def stage(self,tokens):
  if self.pending is not None:raise ValueError('previous verify pass has not been committed or discarded')
  tokens=[self.serial._token(t)for t in tokens];p=self.serial.length
  if len(tokens)!=self.rows or p+self.rows>self.serial.max_seq:raise ValueError('verify extent exceeds declared rows/context')
  begin=time.perf_counter();self.runtime.write(self.spec['input'],b''.join(self.serial.embedding_row(t)for t in tokens))
  stats=self.runtime.execute(self.spec['graph'],{'position':p})
  greedy=list(struct.unpack('<'+'i'*self.rows,self.runtime.read(self.spec['tokens'],sync=False)))
  for t in greedy:self.serial._token(t)
  self.pending=dict(start=p,tokens=tokens,greedy=greedy)
  return dict(greedy=greedy,stats=stats,elapsed_s=time.perf_counter()-begin)
 def logits(self):
  if self.pending is None:raise ValueError('no pending pass')
  data=self.runtime.read(self.spec['logits']);n=self.serial.vocab_size*2
  if len(data)!=self.rows*n:raise ValueError('wrong verifier logits extent')
  return [data[i*n:(i+1)*n]for i in range(self.rows)]
 def commit(self,accepted_drafts):
  if self.pending is None:raise ValueError('no pending pass')
  if isinstance(accepted_drafts,bool):raise ValueError('accepted count must be an integer')
  k=operator.index(accepted_drafts)
  if not 0<=k<self.rows:raise ValueError('accepted draft count outside 0..M-1')
  p=self.pending
  self.serial.length=p['start']+k+1
  self.serial.last_token=p['greedy'][k];self.serial.last_graph=None
  self.pending=None
  return dict(length=self.serial.length,token=self.serial.last_token)
 def discard(self):self.pending=None

def toy_generate(serial,verify,prompt,*,fixed=None):
 eos=serial.model['eos'];tokens=[serial.prefill(prompt)['token']];passes=0;accepted=[];started=time.perf_counter()
 while tokens[-1]not in eos and len(tokens)<serial.max_seq-len(prompt):
  if serial.length+verify.rows>serial.max_seq:
   tokens.append(serial.decode(tokens[-1])['token']);continue
  # Proposals depend only on emitted tokens or a fixed vocabulary ID.
  drafts=[tokens[-1]]+[tokens[-1]if fixed is None else fixed]*(verify.rows-1)
  got=verify.stage(drafts)['greedy'];k=0
  while k<verify.rows-1 and drafts[k+1]==got[k]:k+=1
  verify.commit(k);accepted.append(k);passes+=1
  for t in drafts[1:k+1]+[got[k]]:
   tokens.append(t)
   if t in eos or len(tokens)>=serial.max_seq-len(prompt):break
 if tokens[-1]not in eos:raise AssertionError('toy-drafter translation failed to reach EOS')
 return dict(tokens=tokens,passes=passes,accepted_drafts=accepted,decode_ms=(time.perf_counter()-started)*1000)
