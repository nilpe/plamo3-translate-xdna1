"""Binding bitwise v4C per-service gold vectors on the unchanged target kernels."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import settings83
settings83.paths()
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3'
for x in(APP,ROOT/'lib/r36',ROOT/'lib/r37',ROOT/'lib/r41',Path(__file__).parent):sys.path.insert(0,str(x))
from target_drafter import bits,floats
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='PLaMo-D7-M2');a=p.parse_args();rows=[]
try:
 from initialize83 import initialize
 import settings83
 env=initialize(settings83.SPEC_MANIFEST);r=env.runtime;c=r.manifest['r40_cycle'];assert not c['UNGATED'];bundle=Path(c['bundle']);from gatefiles83 import retain_verified;retain_verified(bundle);vectors=bundle/'PLaMo-service-vectors';meta=json.loads((vectors/'PLaMo-service-vectors.json').read_text());assert meta['PASS']and len(meta['cases'])==31
 import runpy
 module=runpy.run_path(str(bundle/'PLaMo-reference.py'));union=np.load(bundle/json.loads((bundle/'PLaMo-manifest.json').read_text())['head']['union_ids_file']);bank_rows=json.loads((bundle/'PLaMo-manifest.json').read_text())['head']['union_physical_rows'];assert Path(c['exact_bank_file']).stat().st_size==bank_rows*1600,'exact bank size differs from the manifest row count';bank=np.memmap(c['exact_bank_file'],mode='r',dtype=np.uint8,shape=(bank_rows,1600))
 assert not bank[len(union):,64:].any()and not bank[:,4:64].any()and np.all(bank[len(union):,:4].copy().view('<f4')==1)
 def write(name,x,row=0):
  alias=name+'_row'+str(row)
  if alias in r.views:r.write(alias,bits(x))
  else:r.write(name,bits(x),offset=row*524288)
 def read(name,n,row=0):
  alias=name+'_row'+str(row)
  return floats(r.read(alias,nbytes=n*2))if alias in r.views else floats(r.read(name,nbytes=n*2,offset=row*524288))
 def equal(label,x,y):
  xb=bits(x).reshape(-1);yb=bits(y).reshape(-1);ok=np.array_equal(xb,yb);item=dict(id=label,result='PASS'if ok else'FAIL',words=len(yb),different=int(np.count_nonzero(xb!=yb)));rows.append(item);print(json.dumps(item),flush=True);assert ok,item
 def packed_check(view,q,s):
  raw=np.frombuffer(r.read(view),np.uint8).reshape(len(q),q.shape[1]+64);assert np.array_equal(raw[:,64:].view(np.int8),q);assert np.array_equal(raw[:,:4].copy().view('<f4').reshape(-1).view(np.uint32),np.asarray(s,np.float32).view(np.uint32));assert not np.any(raw[:,4:64])
 for case in meta['cases']:
  file=vectors/case['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==case['sha256'];v=dict(np.load(file));op=case['op'];label=case['file']
  if op=='norm':
   width=case['width'];key=case['name'].removesuffix('.weight');actual=floats(r.read(c['gammas'][key],nbytes=width*2));assert np.array_equal(bits(actual),bits(v['gamma'])),'exported gamma differs from gold';r.write('r40_service_gamma',bits(actual))
   for row in range(2):write('r40_service_in',v['x'][row],row)
   r.execute('r40_gold_norm'+str(width),{});got=np.stack([read('r40_service_out',width,row)for row in range(2)]);equal(label,got,v['output'])
  elif op=='rope':
   got=[]
   for start in range(0,len(v['x']),2):
    for row in range(2):
     idx=min(start+row,len(v['x'])-1);write('r40_service_in',v['x'][idx],row);write('r40_rope',v['table'][idx],row)
    r.execute('r40_gold_rope',{})
    for row in range(min(2,len(v['x'])-start)):got.append(read('r40_service_out',128,row))
   equal(label,np.asarray(got),v['output'])
  elif op in('attention','attention-M2'):
   if op=='attention':prefix=len(v['k']);pos=min(prefix-1,2046);selected=0 if prefix<2048 else 1;q=np.stack((v['q'],v['q']));length=min(2048,prefix+1)
   else:prefix=int(v['valid_prefix'][0]);pos=prefix-1;selected=None;q=v['q'];length=len(v['k']);assert np.array_equal(v['valid_prefix'],[pos+1,pos+2])
   k=np.full((2048,128),np.nan,np.float32);value=k.copy();k[:len(v['k'])]=v['k'];value[:len(v['v'])]=v['v']
   if len(v['k'])<length:k[len(v['k']):length]=.125;value[len(v['v']):length]=-.25
   r.write('r40_specK',bits(k));r.write('r40_specV',bits(value))
   for row in range(2):write('r40_q',q[row].reshape(2048),row)
   r.execute('r40_attention',{'position':pos});got=np.stack([read('r40_attn',2048,row).reshape(16,128)for row in range(2)]);equal(label,got if selected is None else got[selected],v['output'])
  elif op=='silu':
   gate=np.zeros(6144,np.float32);up=gate.copy();gate[:len(v['gate'])]=v['gate'];up[:len(v['up'])]=v['up']
   for row in range(2):write('r40_gateup',gate,row);write('r40_up',up,row)
   r.execute('r40_silu',{});equal(label,read('r40_silu',6144)[:len(v['output'])],v['output'])
  elif op=='ID-map':
   cfg=module['language_config'](bundle,case['target_language']);assert np.array_equal(cfg['id_map'],v['union_map'])and np.array_equal(cfg['union_ids'],v['union_ids']);assert np.array_equal(cfg['vocab_ids'],v['expected_global_ids']);rows.append(dict(id=label,result='PASS',words=len(cfg['vocab_ids']),different=0))
  elif op in('PCA','coarse','exact','union-exact'):
   graph={'PCA':'r40_pca','coarse':('r48_coarse_en'if op=='coarse'and case['target_language']=='en'else'r40_mat_13'),'exact':'r40_exact','union-exact':'r40_exact'}[op];args=r.manifest['graphs'][graph]['steps'][0]['args'];n=v['output'].shape[1]
   if op in('exact','union-exact'):
    ids=v['union_rows']if op=='union-exact'else v['union_map'];packed=np.asarray(bank[ids]);assert np.array_equal(packed[:,64:].view(np.int8),v['codes']);assert np.array_equal(packed[:,:4].copy().view('<f4').reshape(-1),v['scales']);assert np.array_equal(union[ids],v['global_ids']if op=='union-exact'else v['ids'])
    if op=='union-exact':assert np.array_equal(packed,v['packed_rows']),'raw union headers differ'
    r.write('r40_exact_weights',packed)
   else:
    if op=='coarse':
     cfg=module['language_config'](bundle,case['target_language']);assert r.read(args[0])==Path(cfg['coarse_file']).read_bytes(),'resident screen differs from package'
    packed_check(args[0],v['codes'],v['scales'])
   for row in range(2):write(args[1],v['x'][row],row)
   r.execute(graph,{});equal(label,np.stack([read(args[2],n,row)for row in range(2)]),v['output'])
  else:raise ValueError('unhandled package service '+op)
 report=dict(result='PASS',cases=len(rows),words=sum(row['words']for row in rows),rows=rows,package_vectors_sha256=hashlib.sha256((vectors/'PLaMo-service-vectors.json').read_bytes()).hexdigest(),scope='all31 supplied services, each final target-format output bitwise; actual frozen gamma/encoder/coarse/bank weights verified; K64 single and coupled M2 boundary/poison fixtures; no tolerance substitution')
 assert len(rows)==31;a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True);os._exit(0)
except BaseException:
 import traceback;err=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=err),indent=2));print(err,flush=True);os._exit(1)
