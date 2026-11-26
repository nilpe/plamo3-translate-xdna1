"""Additional gold gates on actual one-row controls, alongside all26 full package golds."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3'
for x in(APP,ROOT/'lib/r36',ROOT/'lib/r37',ROOT/'lib/r41',Path(__file__).parent):sys.path.insert(0,str(x))
import settings83
settings83.paths()
import numpy as np
from target_drafter import bits,floats
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
try:
 from experimental_runtime import initialize
 r=initialize(Path(os.environ['R42_MANIFEST']))['runtime'];c=r.manifest['r40_cycle'];bundle=Path(c['bundle']);from gatefiles83 import retain_verified;retain_verified(bundle);vectors=bundle/'PLaMo-service-vectors';meta=json.loads((vectors/'PLaMo-service-vectors.json').read_text());rows=[];outside=[]
 def equal(label,got,expected):
  ok=np.array_equal(bits(got),bits(expected));rows.append(dict(id=label,result='PASS'if ok else'FAIL',words=np.size(expected)));assert ok,label
 for case in meta['cases']:
  f=vectors/case['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256'];v=dict(np.load(f));op=case['op']
  if op in('PCA','coarse','exact','union-exact'):
   graph={'PCA':'r41_pca','coarse':'r41_mat_13','exact':'r41_exact','union-exact':'r41_exact'}[op];args=r.manifest['graphs'][graph]['steps'][0]['args'];n=v['output'].shape[1]
   if op in('exact','union-exact'):
    meta_package=json.loads((bundle/'PLaMo-manifest.json').read_text());bank=np.memmap(c['exact_bank_file'],mode='r',dtype=np.uint8,shape=(meta_package['head']['union_physical_rows'],1600));ids=v['union_rows']if op=='union-exact'else v['union_map'];r.write('r40_exact_weights',np.asarray(bank[ids]));r.sync_to('r40_exact_weights')
   elif op=='coarse':
    meta_package=json.loads((bundle/'PLaMo-manifest.json').read_text());r.write(args[0],np.fromfile(bundle/meta_package['target_language'][case['target_language']]['coarse_file'],np.uint8));r.sync_to(args[0])
   actual=np.frombuffer(r.read(args[0]),np.uint8).reshape(n,v['codes'].shape[1]+64);assert np.array_equal(actual[:,64:].view(np.int8),v['codes']);assert np.array_equal(actual[:,:4].copy().view('<f4').reshape(-1),v['scales'])
   for row in range(2):
    r.write(args[1],bits(v['x'][row]));r.execute(graph,{});equal(case['file']+'-single-input'+str(row),floats(r.read(args[2],nbytes=n*2)),v['output'][row])
  elif op in('attention','attention-M2'):
   for row in range(1 if op=='attention'else 2):
    prefix=len(v['k'])if op=='attention'else int(v['valid_prefix'][row]);q=v['q']if op=='attention'else v['q'][row];expected=v['output']if op=='attention'else v['output'][row]
    if prefix==2048:outside.append(dict(id=case['file'],prefix=prefix,reason='actual one-row proposal graph ends at valid2047; unchanged two-row service gate checks the supplied2048 gold'));continue
    k=np.full((2048,128),np.nan,np.float32);value=k.copy();k[:prefix]=v['k'][:prefix];value[:prefix]=v['v'][:prefix];r.write('r40_specK',bits(k));r.write('r40_specV',bits(value));r.write('r40_q',bits(q.reshape(2048)));r.sync_to('r40_specK');r.sync_to('r40_specV');r.sync_to('r40_q');before=r.read('r40_specK')+r.read('r40_specV');r.execute('r41_attention',{'position':prefix-1});got=floats(r.read('r40_attn',nbytes=4096)).reshape(16,128);equal(case['file']+'-single-prefix'+str(prefix),got,expected);assert before==r.read('r40_specK')+r.read('r40_specV')
 report=dict(result='PASS',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=len(rows),words=sum(x['words']for x in rows),rows=rows,outside_one_row_domain=outside,scope='additional bitwise checks of actual one-row PCA/coarse/exact controls and every applicable supplied K64 gold, including coupled block boundaries and poisoned future KV; all26 original service golds stay mandatory');assert len(rows)==29;a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True);os._exit(0)
except BaseException:
 import traceback;error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));print(error,flush=True);os._exit(1)
