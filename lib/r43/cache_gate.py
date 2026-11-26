"""Selected persistent BF16 publication vs original CPU path at capacity boundaries."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3'
for p in(APP,ROOT/'lib/r36',ROOT/'lib/r37',ROOT/'lib/r41',ROOT/'lib/r42',Path(__file__).parent):sys.path.insert(0,str(p))
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args();rows=[];primer=None
try:
 from primer import Primer
 from experimental_runtime import initialize
 from draft43 import make_direct
 primer=Primer(ROOT/__import__('json').loads((ROOT/'release.json').read_text(encoding='utf-8-sig'))['drafter_package']);e=initialize(Path(os.environ['R42_MANIFEST']));r=e['runtime'];r._r42_primer=primer;bundle=Path(r.manifest['r40_cycle']['bundle']);Native,module=make_direct(bundle,r);m=Native(bundle);legacy=next(c for c in type(m).__mro__ if c.__name__=='Partial');rng=np.random.default_rng(43);last=m.r(rng.normal(size=(4,1536)).astype(np.float32));raw=m.r(rng.normal(size=(2,4,1536)).astype(np.float32));tokens=np.array([123,456],np.int64)
 for position,count,reuse in[(0,1,False),(0,2,False),(2047,1,False),(2046,2,False),(2045,1,True),(2045,2,True)]:
  state=dict(position=position,last_taps=last.copy(),layers=[(m.r(rng.normal(size=(1,position,128)).astype(np.float32)),m.r(rng.normal(size=(1,position,128)).astype(np.float32)))]);results=[]
  for native in(False,True):
   # Explicit external reset must invalidate any pending proposal root.
   copy=dict(position=position,last_taps=state['last_taps'].copy(),layers=[tuple(x.copy()for x in state['layers'][0])]);m.bind_state(copy)
   if reuse:m.propose(copy,int(tokens[0]))
   updated=m.append_confirmed(copy,raw[:count],tokens[:count])if native else legacy.append_confirmed(m,copy,raw[:count],tokens[:count]);k,v=updated['layers'][0];device=m.confirmed_device_bytes();results.append((updated['position'],k.copy(),v.copy(),device));assert all(np.array_equal(x,y)for x,y in zip(copy['layers'][0],state['layers'][0]))
  lhs,rhs=results;assert lhs[0]==rhs[0]==position+count and np.array_equal(lhs[1],rhs[1])and np.array_equal(lhs[2],rhs[2])and lhs[3]==rhs[3];rows.append(dict(position=position,rows=count,reuse=reuse,result='PASS',CPU_KV_bitwise=True,entire_confirmed_device_KV_bitwise=True));print(json.dumps(rows[-1]),flush=True)
 primer.close();a.output.write_text(json.dumps(dict(result='PASS',cases=len(rows),rows=rows,program_change=False),indent=2));sys.stdout.flush();os._exit(0)
except BaseException:
 import traceback;err=traceback.format_exc();print(err,flush=True)
 if primer:primer.close()
 a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=err),indent=2));sys.stdout.flush();os._exit(1)
