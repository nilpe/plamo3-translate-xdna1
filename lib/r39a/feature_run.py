# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
APP=((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'IRON/iron/applications/plamo3');sys.path.insert(0,str(APP));sys.path.insert(0,str(Path(__file__).parent))
# The merged R37 bundle preserves the serial default graph. Its explicit bulk
# tap driver must supply serial features; R36's inline-graph driver reads zeros.
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));sys.path.insert(0,str(ROOT/'lib/r36'));sys.path.insert(0,str(ROOT/'lib/r37'))
from feature_session import FeatureVerifySession
from tap_session_v2 import decode_with_features
from verify_gate_v3 import same

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=a.parse_args()
 from experimental_runtime import initialize
 env=initialize(a.manifest);s=env['session'];v=FeatureVerifySession(s)
 root=((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'gate-data/r36/tap_fixture');meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'));blob=(root/'features.bin').read_bytes();assert hashlib.sha256(blob).hexdigest()==meta['feature_sha256']
 ref=[blob[i*12288:(i+1)*12288]for i in range(meta['rows'])];assert len(ref)==8
 s.prefill(meta['prompt'])
 for i,t in enumerate(meta['inputs']):same(decode_with_features(s,t)['features'],ref[i],f'serial features row{i}')
 s.prefill(meta['prompt'])
 for i in range(0,8,2):
  got=v.stage(meta['inputs'][i:i+2])
  for j in range(2):same(got['features'][j],ref[i+j],f'M2 features row{i+j}')
  committed=v.commit(1);assert committed['features']==ref[i:i+2]
 s.prefill(meta['prompt']);got=v.stage(meta['inputs'][:2]);committed=v.commit(0);assert committed['features']==ref[:1]and s.length==len(meta['prompt'])+1
 same(decode_with_features(s,meta['inputs'][1])['features'],ref[1],'rejected row overwritten by serial fallback')
 # Same advancing-position pairs as R35; includes BF16 return sync and CPU copy.
 original=json.loads(((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text(encoding='utf-8'));cases=[dict(id='short-27',prompt=original['prompts'][3],teacher_outputs=original['teacher_tokens'][75:100]),next(c for c in original['long_cases']if c['id']=='en-1024')];times=[]
 for c in cases:
  s.prefill(c['prompt']);position=s.length;values={1:[],2:[]};read_samples=[]
  for sweep in range(3):
   for M in((1,2)if sweep%2==0 else(2,1)):
    for it in range(26):
     s.length=position+(it&1);v.discard()
     got=decode_with_features(s,c['teacher_outputs'][0])if M==1 else v.stage(c['teacher_outputs'][:2])
     if it>=5:values[M].append(got['elapsed_s']*1000)
  times.append(dict(case=c['id'],C_with_features_ms={m:statistics.median(x)for m,x in values.items()},samples_ms=values))
 report=dict(result='PASS',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_gate_sha256=hashlib.sha256((ROOT/'lib/r36/feature_gate.py').read_bytes()).hexdigest(),adapter='R37 explicit serial bulk-tap driver; same independent reference bytes, rows, commit/fallback assertions and timing loops',criterion='byte-exact BF16 vs independent checkpoint-derived reference',serial_rows=8,verify_rows=8,features_per_row=4,bf16_values=8*4*1536,commit_cases=[0,1],identity=meta,manifest=str(a.manifest),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),timing=times)
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k not in('identity','timing')}),flush=True)
if __name__=='__main__':main()
