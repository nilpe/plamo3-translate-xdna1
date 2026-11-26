"""Real CPU-worker IPC stress, including more than256 jobs and cancellation."""
import argparse,json,time
from pathlib import Path
import numpy as np
from settings83 import SPEC_MANIFEST,paths
paths()
from primer83 import Primer
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args();m=Primer(__import__('settings83').DRAFTER_PACKAGE);rows=[]
try:
 import os
 from identity43 import IdentityLease
 for k,v in json.loads(SPEC_MANIFEST.read_text())['payload']['configuration'].items():os.environ[k]=v
 from initialize83 import package_hash
 package_hash()
 lease=IdentityLease(SPEC_MANIFEST,workers=2);lease.wait()
 from types import SimpleNamespace
 m.authorize(SimpleNamespace(_artifact_lease=lease,manifest=lease.manifest))
 m.ready()
 for i in range(600):
  value=i%63+1;raw=np.full((4,1536),value,np.float32);row=m.start([raw],[16]);state,answer=m.result();assert answer['job']==i+1 and state['position']==0 and np.array_equal(state['last_taps'],raw);rows.append(row)
 z=bytes(12288);m.start([z]*515,[16]*515);job=m.cancel();m.quiesce();assert m.pending is None;row=m.start([z],[16]);state,answer=m.result();assert state['position']==0 and answer['job']==602;m.close();assert m.child.poll()is not None;lease.close();v=[x['send_ms']for x in rows];report=dict(result='PASS',jobs=602,sequence_wraps_crossed=2,cancellation_drained=True,worker=m.ready_record,send_mean_ms=float(np.mean(v)),send_p90_ms=float(np.percentile(v,90)),send_max_ms=max(v),rows=rows,scope='one outstanding request, alternating single-byte flags on x64 TSO; full job ID checked in immutable framed response;600 real CPU zero-length bootstraps, cancel and clean restart; no NPU');a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)
except BaseException:
 import traceback;error=traceback.format_exc();print(error,flush=True);m.close();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));raise
