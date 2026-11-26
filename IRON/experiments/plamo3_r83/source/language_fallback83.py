"""Real NPU: missing/unsupported tags and corrupted map/union fail exactly."""
import argparse,copy,hashlib,json,os,sys,tempfile
from pathlib import Path
import settings83
settings83.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
from default83 import DefaultRuntime
from compact_catalog import pack,MAGIC
import runtime_manifest as identity

def kv(app):
 r=app.env.runtime;views={v['offset']:n for n,v in r.manifest['views'].items()if n.startswith('view')and v['root']=='root0'and v['nbytes']==524288 and 474494976<=v['offset']<491272192};assert len(views)==32
 return [r.read(views[k],nbytes=app.session.length*256)for k in sorted(views)]
try:
 meta=json.loads((settings83.ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());prompt=next(x for x in meta['long_cases']if x['id']=='en-512')['prompt'];cap=32
 app=DefaultRuntime('w4-serial');gold=app.generate(prompt,cap);gold_kv=kv(app);app.close()
 def run(label,kwargs,language):
  logs=[];app=DefaultRuntime(log=logs.append,**kwargs);assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction';out=[];got=app.generate(prompt,cap,emit=out.append,target_language=language)
  assert app.mode=='w4-serial'and len(logs)==1,(label,app.mode,logs)
  assert got['tokens']==gold['tokens']==out and kv(app)==gold_kv and app.session.length==gold['logical_length']
  app.env.lease.require_validated();app.close();rows.append(dict(id=label,result='PASS',tokens=out,active32_KV_bitwise=True,validated_fallback=True,logs=logs));print(json.dumps(rows[-1]),flush=True)
 for label,language in [('missing-language',None),('unsupported-language','fr')]:run(label,{},language)
 original=__import__('plamo_root').rebase(json.loads(settings83.SPEC_MANIFEST.with_name('manifest.json').read_text())['payload'])
 package=json.loads((settings83.DRAFTER_PACKAGE/'PLaMo-manifest.json').read_text())
 with tempfile.TemporaryDirectory(prefix='r48-invalid-',dir=settings83.ROOT/'tmp')as tmp:
  for label,file in [('corrupt-map',package['target_language']['ja']['id_map_file']),('corrupt-union',package['head']['exact_file'])]:
   folder=Path(tmp)/label;folder.mkdir();source=settings83.DRAFTER_PACKAGE/file;bad=folder/source.name;data=bytearray(source.read_bytes());data[len(data)//2]^=1;bad.write_bytes(data);st=source.stat();os.utime(bad,ns=(st.st_atime_ns,st.st_mtime_ns))
   broken=copy.deepcopy(original);record=next(x for x in broken['origins']if x['path']==str(source));record['path']=str(bad)
   full=folder/'manifest.json';identity.write_manifest(full,broken);binary=folder/'catalog.bin';binary.write_bytes(pack(broken)[0])
   def rec(path):return dict(**identity.digest_file(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   cert=folder/'certificate.json';identity.write_manifest(cert,dict(version=1,kind=MAGIC,configuration=broken['configuration'],python_abi=broken['python_abi'],catalog=rec(binary),expanded_manifest=rec(full),expanded_payload_sha256=broken['manifest_sha256']))
   run(label,dict(spec_manifest=cert),'ja')
 a.output.write_text(json.dumps(dict(result='PASS',cases=4,rows=rows,original_package_untouched=True),indent=2))
except BaseException:
 import traceback;err=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=err),indent=2));print(err,flush=True);raise
