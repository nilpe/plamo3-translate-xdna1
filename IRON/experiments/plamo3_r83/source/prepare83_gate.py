"""R83 prepare-gate (host only, no NPU): prepare83._prepare on a fake runtime with fake pyxrt objects.
Scenarios: first creation, growth (4 KiB -> 8 KiB -> 64 KiB), size changes in both directions on every family, repeated identical case (no rewrite), a second graph, set_arg(2) on every run of a family,
close() releasing grown BOs. After EVERY prepare each run of each step must carry arg1 = its family's current BO, arg2 = the current size, the BO content[:size] must equal the case stream, the BO must have been flushed
over at least `size` bytes after the last write, and capacity >= size. Mutants of prepare83.py (stale size, skipped growth, wrong BO after growth, missing flush, no set_arg on all runs, partial BO copy, ignored same-part skip) must ALL be caught."""
import argparse,json,re,sys,traceback,types
from pathlib import Path
import settings83 as s
s.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode');a=p.parse_args()
SRC=(Path(__file__).with_name('prepare83.py')).read_text(encoding='utf-8')
class BO:
 n=0
 def __init__(self,size):BO.n+=1;self.id=BO.n;self.size=size;self.buf=bytearray(size);self.synced=0;self.dirty=True
 def map(self):return memoryview(self.buf)
 def sync(self,direction,size=None,offset=0):
  self.synced_ranges=getattr(self,'synced_ranges',[])+[(offset,self.size if size is None else size)]
class Run:
 def __init__(self,kernel):self.args={}
 def set_arg(self,i,v):self.args[i]=v
class RunList:
 def __init__(self,ctx):self.runs=[]
 def add(self,r):self.runs.append(r)
class Kernel:
 def group_id(self,i):return 1
class XRT:
 class bo:
  cacheable=1
  def __new__(cls,device,size,flags,gid):return BO(size)
 class xclBOSyncDirection:XCL_BO_SYNC_BO_TO_DEVICE=0
 runlist=RunList;run=Run
class Part(dict):pass
def make_runtime(mod,families):
 rt=types.SimpleNamespace();rt.xrt=XRT;rt.device=object();rt._graphs={};rt._lock=__import__('threading').RLock()
 k=Kernel();rt.contexts={'c':(None,object(),k,8)}
 rt.manifest=dict(graphs={'g':dict(steps=[dict(context='c',instructions=f,args=['v']) for f in families for _ in range(3)])},instructions={f:dict(cases={}) for f in families},r37_features=dict(prefill=[dict(graph='g')],prefill_head='g'))
 rt._ready=lambda:None;rt._span=lambda n,t:0.0;rt._bo=lambda v:BO(16)
 cases={}
 def _cases(graph,values):return [cases[(f,values['rows'])] for f in families for _ in range(3)]
 rt._cases=_cases
 rt._instruction_size=lambda part:len(part['bytes']);rt._instruction_bytes=lambda part:part['bytes']
 return rt,cases
def run_scenario(mod):
 fams=['fa','fb'];rt,cases=make_runtime(mod,fams)
 import random;rng=random.Random(5)
 sizes=[100,3000,100,5000,5000,9000,200,64000,64,64000,12]    # growth, shrink, equal repeat, growth to 64 KiB
 for rows,sz in enumerate(sizes,start=1):
  for fi,f in enumerate(fams):
   n=max(4,(sz+fi*40)//4*4);cases[(f,rows)]=Part(bytes=bytes(rng.randrange(256) for _ in range(n)),nbytes=n)
  st=mod._prepare(rt,'g',dict(rows=rows));cs=[cases[(f,rows)] for f in fams]
  for current,part in zip(st['steps'],[cases[(f,rows)] for f in fams for _ in range(3)]):
   storage=current['storage'];run=current['run'];size=len(part['bytes'])
   assert run.args[1] is storage['bo'],('stale/wrong BO on run',rows)
   assert run.args[2]==size,('stale size',rows,run.args[2],size)
   assert bytes(storage['bo'].buf[:size])==part['bytes'],('BO content',rows)
   assert storage['bo'].size>=size and storage['capacity']==storage['bo'].size,('capacity',rows)
   rng_=getattr(storage['bo'],'synced_ranges',[]);assert any(o==0 and l>=size for o,l in rng_),('missing flush',rows)
  for storage in st['instructions'].values():storage['bo'].synced_ranges=[]
 # same-part skip: second prepare with unchanged case must not rewrite or resize
 before={k:v['bo'].id for k,v in st['instructions'].items()};mod._prepare(rt,'g',dict(rows=len(sizes)));assert before=={k:v['bo'].id for k,v in st['instructions'].items()}
 # close(): the DirectRuntime.close pattern releases host views and clears storage dicts
 for state in rt._graphs.values():
  for storage in state['instructions'].values():storage['host'].release();storage.clear()
 return True
def load(src):
 mod=types.ModuleType('prepare83_under_test');exec(compile(src,'prepare83_mut','exec'),mod.__dict__);return mod
MUT={
 'stale_size':("for run in storage['runs']:\n                        run.set_arg(2, size)","pass"),
 'set_arg_first_run_only':("for run in storage['runs']:\n                        run.set_arg(2, size)","for run in storage['runs'][:1]:\n                        run.set_arg(2, size)"),
 'skip_growth':("if size > storage['capacity']:","if False:"),
 'wrong_bo_after_growth':("                    for run in storage['runs']:\n                        run.set_arg(1, bo)\n","                    pass\n"),
 'missing_flush':("storage['bo'].sync(to_device, min(storage['capacity'], (size + 63) // 64 * 64), 0)","pass"),
 'short_flush':("(size + 63) // 64 * 64), 0)","(size // 2)), 0)"),
 'partial_copy':("storage['host'][:size] = payload","storage['host'][:size - 4] = payload[:size - 4]"),
 'ignore_part_change':("if storage['part'] != part:","if storage['part'] is None:"),
 'old_bo_content_not_copied_after_growth':("storage['bo'], storage['host'], storage['capacity'] = bo, host, capacity","storage['bo'], storage['host'], storage['capacity'] = bo, host, capacity - 1"),
}
try:
 assert run_scenario(load(SRC))
 caught={};
 for name,(old,new) in MUT.items():
  assert SRC.count(old)==1,(name,'pattern not unique/present')
  try:
   run_scenario(load(SRC.replace(old,new)));caught[name]=False
  except (AssertionError,Exception) as e:caught[name]=True
 assert all(caught.values()),caught
 a.output.write_text(json.dumps(dict(result='PASS',scenario_sizes=11,mutants=caught),indent=1));print(json.dumps(dict(result='PASS',mutants=caught)),flush=True)
except BaseException:
 a.output.write_text(json.dumps(dict(result='FAIL',error=traceback.format_exc()),indent=2));traceback.print_exc();sys.exit(1)
