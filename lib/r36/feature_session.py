# SPDX-License-Identifier: Apache-2.0
"""Optional BF16 input-aligned features; k is accepted drafts, never row count."""
import time
from verify_session import VerifySession

def read_features(serial,rows):
 spec=serial.runtime.manifest['feature_taps']
 if spec['order']!=['layer2','layer8','layer14','final_norm']or spec['width']!=1536:raise ValueError('feature contract differs')
 return [serial.runtime.read(n)for n in spec['views'][:rows]]

def decode_with_features(serial,token):
 begin=time.perf_counter();got=serial.decode(token);got['features']=read_features(serial,1)[0];got['elapsed_s']=time.perf_counter()-begin
 return got

class FeatureVerifySession(VerifySession):
 def stage(self,tokens):
  begin=time.perf_counter();got=super().stage(tokens);self.pending_features=read_features(self.serial,self.rows);got['features']=self.pending_features;got['elapsed_s']=time.perf_counter()-begin
  return got
 def commit(self,accepted_drafts):
  got=super().commit(accepted_drafts);got['features']=self.pending_features[:accepted_drafts+1];self.pending_features=None
  return got
 def discard(self):
  super().discard();self.pending_features=None
