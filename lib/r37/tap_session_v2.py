"""Bulk serial feature copies, with the existing certified serial PDI."""
import time
from tap_session import prefill_with_features

def decode_with_features(s,token):
 token=s._token(token)
 if not 0<=s.length<s.max_seq:raise ValueError('decode beyond context')
 r=s.runtime;cfg=r.manifest['r37_features']['serial_bulk'];t=time.perf_counter();r.write(s.model['decode_embedding'],s.embedding_row(token));stats=r.execute(cfg['graph'],{'position':s.length});features=r.read(cfg['view'],nbytes=12288);s.length+=1;carry=s._carry('decode')
 return dict(token=carry,position=s.length-1,features=features,stats=stats,elapsed_s=time.perf_counter()-t)
