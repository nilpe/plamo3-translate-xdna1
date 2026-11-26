"""Opt-in tap return; default RuntimeSession graphs and numerical kernels unchanged."""
import time

def decode_with_features(s,token):
 token=s._token(token)
 if not 0<=s.length<s.max_seq:raise ValueError('decode beyond target context')
 t=time.perf_counter();r=s.runtime;r.write(s.model['decode_embedding'],s.embedding_row(token));features=[];stats=[]
 for part in r.manifest['r37_features']['serial']:
  stats.append(r.execute(part['graph'],{'position':s.length}));features.append(r.read(part['view'],nbytes=3072,sync=False))
 s.length+=1;result=s._carry('decode')
 return dict(token=result,position=s.length-1,features=b''.join(features),stats=stats,elapsed_s=time.perf_counter()-t)

def prefill_with_features(s,tokens):
 tokens=[s._token(t)for t in tokens]
 if not tokens or len(tokens)>s.max_seq:raise ValueError('prompt outside target context')
 s.reset();r=s.runtime;cfg=r.manifest['r37_features'];batch=s.model['prefill_rows'];size=batch*s.row_bytes;t=time.perf_counter();stats=[];result=[]
 for start in range(0,len(tokens),batch):
  chunk=tokens[start:start+batch];dense=bytearray(size)
  for i,tok in enumerate(chunk):dense[i*s.row_bytes:(i+1)*s.row_bytes]=s.embedding_row(tok)
  r.write(s.model['prefill_input'],dense);values=dict(start=start,rows=len(chunk),position=start+len(chunk)-1);features=[]
  for part in cfg['prefill']:
   stats.append(r.execute(part['graph'],values));features.append(r.read(part['view'],nbytes=len(chunk)*s.row_bytes,sync=False))
  result.extend(b''.join(f[i*s.row_bytes:(i+1)*s.row_bytes]for f in features)for i in range(len(chunk)))
  s.length=start+len(chunk)
  if s.length==len(tokens):stats.append(r.execute(cfg['prefill_head'],values))
 carry=s._carry('prefill')
 return dict(token=carry,position=s.length-1,features=result,stats=stats,elapsed_s=time.perf_counter()-t)
