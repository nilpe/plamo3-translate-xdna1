"""R83 gate: (1) the production gate decision (fast NumPy path with exact fallback, gate83.py) equals the R57 reference decision
(1/math.fsum(exp(l-lmax)) < tau, float64, strict) on every recorded logit vector (R57 calibration + R57 decision data + fresh R83 runs), on random vectors,
and on an adversarial boundary set that places S = sum exp within a few ulp of 1/tau; (2) the R57 checks (independent summation order, NaN/inf fail closed,
strict '<', loop constants, free names of generate); (3) the error bound is respected by every observed |S_np - S_ref|/S_ref.
--mutant nofallback|tinyband runs the same assertions against a fast path whose fallback band is removed / shrunk: they must FAIL (see gate_mutants83.py)."""
import argparse,ast,base64,builtins,json,math,os,struct,subprocess,sys,symtable
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--mutant',choices=['none','nofallback','tinyband'],default='none')
p.add_argument('--logs',nargs='*',default=[]);p.add_argument('--calibration',nargs='*',default=[]);p.add_argument('--decision',nargs='*',default=[]);p.add_argument('--fresh',nargs='*',default=[]);p.add_argument('--min-calibration-cycles',type=int,default=0);p.add_argument('--min-decision-cycles',type=int,default=0);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).parent))
import gate83
if a.mutant=='nofallback':gate83.REL_BOUND=0.0
if a.mutant=='tinyband':gate83.REL_BOUND=1e-16
def ref_top1(xs):
 m=max(xs);return 1.0/math.fsum(math.exp(x-m)for x in sorted(xs))   # different summation order; fsum is exactly rounded
def ref_margin(xs):
 s=sorted(xs,reverse=True);return s[0]-s[1]
def bits(x):return struct.pack('<d',x).hex()
TAUS=(0.4,0.05,0.1,0.2,0.3,0.35,0.45,0.5,0.6,0.75,0.9)
stats=dict(decisions=0,fallback=0,fast=0,max_rel_err=0.0,min_rel_distance=float('inf'),bound=gate83.error_bound(512),rel_bound=gate83.REL_BOUND)
def decide(x,tau,xs=None):
 xs=xs if xs is not None else[float(v)for v in x];want=ref_top1(xs);fires=want<tau   # the R57 decision, independent order
 got,fell=gate83.fast_top1(x,tau);stats['decisions']+=1;stats['fallback']+=bool(fell);stats['fast']+=not fell
 assert gate83.gate_fires(got,tau)==fires,('fast decision differs from the R57 reference',tau,bits(got),bits(want))
 assert gate83.gate_fires(gate83.confidence(x,'top1_prob',tau),tau)==fires,'confidence() decision'
 if fell:assert bits(got)==bits(want),'fallback must return the exact reference bits'
 else:
  rel=abs(got-want)/want;stats['max_rel_err']=max(stats['max_rel_err'],rel);assert rel<=gate83.error_bound(len(xs)),('error bound violated',rel)
 s_ref=1.0/want;stats['min_rel_distance']=min(stats['min_rel_distance'],abs(s_ref-1/tau)*tau)
 return fires
def check(x,taus=TAUS):
 xs=[float(v)for v in x]
 assert bits(gate83.reference_confidence(x,'top1_prob'))==bits(ref_top1(xs)),'top1_prob reference';assert bits(gate83.reference_confidence(x,'margin'))==bits(ref_margin(xs)),'margin'
 assert bits(gate83.confidence(x,'margin',2.0))==bits(ref_margin(xs)),'margin via confidence()'
 for t in taus:decide(x,t,xs)
const={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse((Path(__file__).parent/'loop83.py').read_text()).body if isinstance(n,ast.Assign)and n.targets[0].id in('GATE_STAT','GATE_TAU')}
assert const=={'GATE_STAT':'top1_prob','GATE_TAU':0.4},const   # the R57 calibration selection (unchanged)
src=(Path(__file__).parent/'loop83.py').read_text();tree=ast.parse(src)
log_node=next(n for n in tree.body if isinstance(n,ast.Assign)and n.targets[0].id=='LOG_LOGITS')
for env,want in((None,False),('1',True),('0',False)):
 old=os.environ.pop('R83_LOG_LOGITS',None)
 if env is not None:os.environ['R83_LOG_LOGITS']=env
 try:assert eval(compile(ast.Expression(log_node.value),'LOG_LOGITS','eval'),{'os':os})is want,('LOG_LOGITS default',env)
 finally:
  os.environ.pop('R83_LOG_LOGITS',None)
  if old is not None:os.environ['R83_LOG_LOGITS']=old
tab=symtable.symtable(src,'loop83.py','exec')
top=[sym.get_name()for sym in tab.get_symbols()if sym.is_assigned()or sym.is_imported()or sym.is_namespace()]
fn=next(c for c in tab.get_children()if c.get_name()=='generate')
free=[sym.get_name()for sym in fn.get_symbols()if sym.is_global()and sym.is_referenced()]
unresolved=[n for n in free if n not in top and not hasattr(builtins,n)];assert not unresolved,('names used by generate but not defined/imported in loop83',unresolved)
assert{'confidence','gate_fires','GATE_STAT','GATE_TAU','base64','LOG_LOGITS'}<=set(free)
call=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and getattr(n.func,'id','')=='confidence'];assert call and all(len(c.args)==3 for c in call),'generate must pass tau to confidence()'
n=0;rng=np.random.default_rng(56)
for i in range(300):
 x=(rng.standard_normal(512)*[0.5,2,8][i%3]).astype(np.float32)
 if i%5==0:x=((x.view(np.uint32)>>16)<<16).view(np.float32)   # bf16-valued like the device logits
 if i%7==0:x[:]=x[0]
 if i%11==0:x[1]=x[0]
 check(x);n+=1
for bad in(np.array([np.nan]+[0.]*511,np.float32),np.array([np.inf]+[0.]*511,np.float32),np.array([-np.inf]+[0.]*511,np.float32),np.array([0.]*511+[-np.inf],np.float32),np.array([np.nan,np.inf]+[0.]*510,np.float32),np.array([],np.float32)):
 for stat in gate83.STATS:assert math.isnan(gate83.reference_confidence(bad,stat))
 for t in TAUS:
  c=gate83.confidence(bad,'top1_prob',t);assert math.isnan(c)and not gate83.gate_fires(c,t),('non-finite must fail closed',t)
assert gate83.gate_fires(0.4,0.5)and not gate83.gate_fires(0.5,0.5)and not gate83.gate_fires(0.6,0.5)and not gate83.gate_fires(float('nan'),0.5)and not gate83.gate_fires(0.1,None)
big=np.zeros(5000,np.float32);assert gate83.fast_top1(big,0.4)[1]and gate83.fast_top1(big,0.4)[0]==ref_top1([0.0]*5000)   # > MAX_FAST_N: exact path
# ---- adversarial boundary set: S = sum exp(l - l_max) within +-40 ulp steps of 1/tau (float64 logits; production logits are float32 but the function accepts both) ----
def boundary_vectors(tau,seed,count,span=40):
 r=np.random.default_rng(seed);target=1.0/tau
 for c in range(count):
  m=512;d=-np.abs(r.standard_normal(m-2))*r.choice([0.7,2.0,5.0])-r.uniform(0.0,1.0);part=float(np.exp(d).sum())
  if part>0.8*(target-1.0):d=d-math.log(part/(0.5*(target-1.0)));part=float(np.exp(d).sum())
  last=target-1.0-part;assert last>0;base=np.concatenate(([0.0],d,[math.log(last)]));perm=r.permutation(m);v=base[perm];idx=int(np.argmax(perm==m-1))
  for k in range(-span,span+1):
   w=v.copy();val=w[idx]
   for _ in range(abs(k)):val=np.nextafter(val,np.inf if k>0 else -np.inf)
   w[idx]=val;yield w
bset=0;mut_dis=0;fires_seen=set()
for tau,seed in((0.4,1),(0.4,2),(0.25,3),(0.55,4)):
 for w in boundary_vectors(tau,seed,80 if tau==0.4 else 20):
  xs=[float(v)for v in w];fires=decide(w,tau,xs);fires_seen.add(fires);bset+=1
  s_np=float(np.exp(w-w.max()).sum());mut_dis+=((1.0/s_np<tau)!=fires)   # a fast path without the fallback would have flipped this decision
assert fires_seen=={True,False},'boundary set must straddle the threshold'
assert bset>=10000,bset
if a.mutant=='none':assert mut_dis>0,('boundary set must contain cases where S_np alone gives a different decision (else it cannot catch a missing fallback)',mut_dis)
stats['boundary_cases']=bset;stats['boundary_disagreements_without_fallback']=int(mut_dis)
# ---- recorded logits ----
logged=dict(calibration=0,decision=0,fresh=0);fired=0;files=dict(calibration=[],decision=[],fresh=[])
def visit(cycles,kind,name):
 global fired
 for c in cycles:
  if not c.get('logits_b64'):continue
  x=np.frombuffer(base64.b64decode(c['logits_b64']),np.float32);assert x.shape==(512,);check(x);logged[kind]+=1
  if c.get('conf')is not None:
   stat=c['gate_stat'];tau=c['gate_tau'];got=c['conf'];xs=[float(v)for v in x];want=ref_top1(xs)if stat=='top1_prob'else ref_margin(xs)
   if kind=='fresh'or str(c.get('conf_def','')).startswith('R83:'):assert bits(got)==bits(want)or(abs(got-want)<=gate83.error_bound(512)*want and(got<tau)==(want<tau)),(name,'conf','logged conf must be the reference value or its fast estimate on the same side of tau')   # records of the R83 production path (they say so in conf_def: fast estimate + exact fallback band) are held to the production rule, whichever list they are given in; records of the earlier exact path keep the bitwise rule
   else:assert bits(got)==bits(want),(name,stat)
   assert c['gated']==(want<tau),(name,'gated flag');fired+=bool(c['gated']);assert c['mode']in('EOS-C1','GATE-C1')if c['gated']else c['mode']in('EOS-C1','M2'),(name,c['mode'])
   assert(c['mode']=='GATE-C1')==(c['gated']and c['proposal']not in(1,16)),(name,'mode')
  else:assert c.get('gate_tau')is None or c['mode']!='GATE-C1',(name,'unlogged conf')
for kind,paths in(('calibration',a.calibration),('decision',a.decision),('fresh',a.fresh+a.logs)):
 for f in paths:
  d=json.loads(Path(f).read_text(encoding='utf-8-sig'));assert d['result']=='PASS',f
  for case in d['cases']:
   spec=case.get('speculative')or case.get('spec')
   if spec:visit(spec['cycles'],kind,case.get('id'))
  files[kind].append(str(f))
assert logged['calibration']>=a.min_calibration_cycles and logged['decision']>=a.min_decision_cycles,(logged,a.min_calibration_cycles,a.min_decision_cycles)
r=dict(result='PASS',mutant=a.mutant,synthetic=n,logged_cycles=logged,fired=fired,files={k:len(v)for k,v in files.items()},stats=stats,decision_equivalence='fast decision == R57 reference decision for every vector at taus '+str(TAUS));a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
