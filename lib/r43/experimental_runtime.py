"""Opt-in early context setup with the unchanged strict execution barrier."""
import importlib.abc,json,os,sys,time
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3';sys.path.insert(0,str(APP));_OWNERS=[]
class NoCompiler(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in {'aie','mlir','torch','transformers'}:raise ImportError('experimental direct runtime forbids '+fullname)
def initialize(manifest):
 for name in list(sys.modules):
  if name.split('.')[0]in {'aie','mlir','torch','transformers'}:raise RuntimeError('preexisting compiler/model framework '+name)
 sys.meta_path.insert(0,NoCompiler());manifest=Path(manifest)
 for k,v in json.loads(Path(__file__).with_name('configuration.json').read_text()).items():os.environ[k]=v
 import startup_r20
 startup_r20.bootstrap()
 if manifest.name=='certificate.json':
  from identity43 import IdentityLease
  from direct43 import DirectRuntime
 else:
  from runtime_identity import IdentityLease
  from direct_runtime import DirectRuntime
 from runtime_session import RuntimeSession
 from concurrent.futures import ThreadPoolExecutor
 started=time.perf_counter();events=[];owners={};_OWNERS.append(owners)
 def event(name,seconds):
  if owners.get('startup_done'):return
  events.append(dict(phase=name,ms=seconds*1000,end_ms=(time.perf_counter()-started)*1000))
 with ThreadPoolExecutor(max_workers=1,thread_name_prefix='r43-early-xrt')as pool:
  def executable_ready(lease):
   rt=DirectRuntime(lease.manifest,lease.paths,event=event,artifact_lease=lease,validation_barrier=lease.require_validated,copy_mode='readinto');owners['runtime']=rt;return rt.begin_contexts(pool).result
  lease=IdentityLease(manifest,workers=2,event=event,executable_ready=executable_ready);owners['lease']=lease;runtime=owners['runtime'];committed=pool.submit(lease.wait);runtime.open(concurrent_setup=True);committed.result();lease.require_validated()
 owners['startup_done']=True
 return dict(session=RuntimeSession(runtime),runtime=runtime,validation=lease.stats,setup_s=time.perf_counter()-started,startup_events=events)
