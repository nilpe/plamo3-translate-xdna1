"""Full compact-schema and content-tamper gates; no device is opened."""
import argparse,copy,hashlib,json,marshal,os,struct,sys,tempfile,time
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));APP=ROOT/'IRON/iron/applications/plamo3'
for p in(APP,Path(__file__).parent):sys.path.insert(0,str(p))
import numpy as np
import runtime_manifest as base
from compact_catalog import Cases,pack,decode,expand,read_certificate
from direct43 import DirectRuntime
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args();rows=[]
try:
 path=Path(os.environ.get('R42_MANIFEST',str(ROOT/'plamo3-verify-r42/final/PLaMo-D7-M2/manifest.json'))).with_name('manifest.json');m=base._read_manifest(path);blob,count=pack(m);decoded=decode(blob)
 class Lease:
  def __init__(self,m):self.manifest=m;self.paths={n:Path(r['path'])for n,r in m['artifacts'].items()}
  def read_into(self,*args):raise AssertionError('no BO/model load allowed')
  def require_validated(self):raise AssertionError('no execution allowed')
 def structure(m):
  lease=Lease(m);return DirectRuntime(m,lease.paths,artifact_lease=lease,validation_barrier=lease.require_validated,copy_mode='readinto')
 at=time.perf_counter();structure(decoded);schema_ms=(time.perf_counter()-at)*1000;assert expand(decoded)==m
 for n,d in decoded['instructions'].items():
  for key,part in m['instructions'][n]['cases'].items():assert d['cases'][key]==part
  for key in('01','+1','1,','NaN','18446744073709551616'):
   if key in m['instructions'][n]['cases']:continue
   try:d['cases'][key]
   except KeyError:pass
   else:raise AssertionError('unknown/noncanonical key accepted')
 rows.append(dict(id='all-exact-cases-and-missing-keys',result='PASS',cases=count,schema_ms=schema_ms))
 raw=marshal.loads(blob,allow_code=False);chosen=next(n for n,d in raw['manifest']['instructions'].items()if'r43_cases'in d['cases']and d['cases']['r43_cases']['count']>1);fields=[('offset-oob',lambda r:r.update(slices=struct.pack('<QII',2**63,4,4)+r['slices'][16:])),('empty-size',lambda r:r.update(slices=struct.pack('<QII',0,0,4)+r['slices'][16:])),('bad-decoded-alignment',lambda r:r.update(slices=struct.pack('<QII',0,4,3)+r['slices'][16:])),('oversized-decoded',lambda r:r.update(slices=struct.pack('<QII',0,4,64*1024*1024+4)+r['slices'][16:])),('unknown-artifact',lambda r:r.update(artifact='absent')),('wrong-codec',lambda r:r.update(codec='pickle')),('wrong-key-arity',lambda r:r.update(arity=r['arity']+1)),('duplicate-key',lambda r:r.update(keys=r['keys'][:r['arity']*8]*2+r['keys'][2*r['arity']*8:])),('truncated-slices',lambda r:r.update(slices=r['slices'][:-1])),('bool-count',lambda r:r.update(count=True))]
 for name,change in fields:
  bad=copy.deepcopy(raw);change(bad['manifest']['instructions'][chosen]['cases']['r43_cases'])
  try:structure(decode(marshal.dumps(bad,4)))
  except(ValueError,TypeError,KeyError,OverflowError):rows.append(dict(id=name,result='PASS',rejected=True))
  else:raise AssertionError(name+' was accepted')
 for name,change in [('unknown-graph-context',lambda z:z['graphs']['decode']['steps'][0].update(context='absent')),('view-overflow',lambda z:next(iter(z['views'].values())).update(offset=2**63)),('unknown-root',lambda z:next(iter(z['views'].values())).update(root='absent'))]:
  z=decode(blob);change(z)
  try:structure(z)
  except(ValueError,TypeError,KeyError):rows.append(dict(id=name,result='PASS',rejected=True))
  else:raise AssertionError(name+' was accepted')
 with tempfile.TemporaryDirectory(prefix='r43-integrity-',dir=ROOT/'tmp')as temp:
  folder=Path(temp);full=folder/'full.json';full.write_bytes(path.read_bytes());binary=folder/'catalog.bin';binary.write_bytes(blob)
  def rec(f):return dict(**base.digest_file(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  cert=dict(version=1,kind='PLaMo-R43-catalog-v1',catalog=rec(binary),expanded_manifest=rec(full),expanded_payload_sha256=m['manifest_sha256']);cf=folder/'certificate.json';base.write_manifest(cf,cert);owners=[]
  try:structure(read_certificate(cf,owners))
  finally:
   for f in owners:f.close()
  for label,f in [('binary-content',binary),('expanded-sidecar-content',full),('certificate-content',cf)]:
   original=f.read_bytes();stat=f.stat();bad=bytearray(original);bad[len(bad)//2]^=1;f.write_bytes(bad);os.utime(f,ns=(stat.st_atime_ns,stat.st_mtime_ns));owners=[]
   try:
    try:read_certificate(cf,owners)
    except(BaseException)as e:
     if isinstance(e,(AssertionError,KeyboardInterrupt,SystemExit)):raise
     rows.append(dict(id=label,result='PASS',rejected=True,mtime_restored=True,error=type(e).__name__))
    else:raise AssertionError(label+' content change was accepted')
   finally:
    for o in owners:o.close()
    f.write_bytes(original)
 report=dict(result='PASS',checks=len(rows),rows=rows,exact_cases=count,typed_schema_ms=schema_ms,no_device_or_model_access=True,scope='every packed case key/part equal independent expanded catalog; original root/view/context checks; malformed keys/slices/codecs rejected; binary/full-sidecar/certificate corruption detected despite restored mtime; all data hashes checked before decoding');a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)
except BaseException:
 import traceback;error=traceback.format_exc();print(error,flush=True);a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=error),indent=2));raise
