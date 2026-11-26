"""Real-NPU fallbacks: independent exact W4 output and retained validation."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import sys
from settings83 import ROOT, SPEC_MANIFEST, W4_MANIFEST, paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
rows=[]
try:
    from default83 import DefaultRuntime
    from initialize83 import initialize
    from primer83 import Primer
    from loop_gate import serial_generate
    meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text())
    prompt=next(x for x in meta['long_cases']if x['id']=='en-512')['prompt'];cap=80
    def active_KV_hash(session):
        import hashlib,struct
        runtime=session.runtime
        views={v['offset']:name for name,v in runtime.manifest['views'].items()if name.startswith('view')and v['root']=='root0'and v['nbytes']==524288 and 474494976<=v['offset']<491272192}
        assert len(views)==32,'all32 target cache operands required'
        digest=hashlib.sha256()
        for offset,name in sorted(views.items()):digest.update(struct.pack('<Q',offset));digest.update(runtime.read(name,nbytes=session.length*256))
        return digest.hexdigest()
    app=DefaultRuntime('w4-serial');baseline=app.generate(prompt,cap);baseline_KV=active_KV_hash(app.session);app.close()
    class DeadAtInit(Primer):
        def authorize(self,runtime):
            super().authorize(runtime);self.child.terminate();self.child.wait(timeout=5)
    class WrongWorkerIdentity(Primer):
        def ready(self,wait=True):
            ready=super().ready(wait)
            if ready:self.ready_record['package_manifest_sha256']='0'*64
            return ready
    def unavailable(bundle):raise OSError('fault gate: worker creation unavailable')
    import direct43
    open_original=direct43.DirectRuntime.open
    def partial_open(self,*args,**kwargs):
        open_original(self,*args,**kwargs)
        raise OSError('fault gate: runtime initialization after BO/context setup')
    def init_with_partial_failure(path,**kwargs):
        if Path(path)==SPEC_MANIFEST:
            direct43.DirectRuntime.open=partial_open
            try:return initialize(path,**kwargs)
            finally:direct43.DirectRuntime.open=open_original
        return initialize(path,**kwargs)
    cases=[('missing-spec-runtime',dict(spec_manifest=ROOT/'tmp/r45-deliberately-missing/certificate.json')),
           ('wrong-package-path',dict(bundle=ROOT/'tmp/r45-deliberately-missing-package')),
           ('worker-create-fails',dict(primer_factory=unavailable)),
           ('worker-init-dies',dict(primer_factory=DeadAtInit)),
           ('worker-ready-identity-invalid',dict(primer_factory=WrongWorkerIdentity)),
           ('runtime-partial-init-fails',dict(initialize=init_with_partial_failure))]
    with tempfile.TemporaryDirectory(prefix='r45-fallback-',dir=ROOT/'tmp')as folder:
        bad=Path(folder)/'certificate.json';raw=bytearray(SPEC_MANIFEST.read_bytes());raw[len(raw)//2]^=1;bad.write_bytes(raw)
        cases.append(('corrupt-spec-certificate',dict(spec_manifest=bad)))
        # Invalid package-origin fixtures keep the original package untouched.
        # Their catalogs contain a deliberately stale expected record and must
        # fail before any command. They are fault inputs, never deployable exports.
        import runtime_manifest as identity
        from compact_catalog import pack, MAGIC
        import hashlib
        original=__import__('plamo_root').rebase(json.loads(SPEC_MANIFEST.with_name('manifest.json').read_text())['payload'])
        package_origin=next(x for x in original['origins']if x['path']==str(__import__('settings83').DRAFTER_PACKAGE/'PLaMo-manifest.json'))
        for label,missing in [('package-content-invalid',False),('package-file-missing',True)]:
            temp=Path(folder)/label;temp.mkdir()
            package_file=temp/'PLaMo-manifest.json'
            if not missing:
                data=bytearray(Path(package_origin['path']).read_bytes());data[len(data)//2]^=1;package_file.write_bytes(data)
            broken=copy.deepcopy(original)
            next(x for x in broken['origins']if x['path']==package_origin['path'])['path']=str(package_file)
            full=temp/'manifest.json';identity.write_manifest(full,broken)
            binary=temp/'catalog.bin';binary.write_bytes(pack(broken)[0])
            def record(path):return dict(**identity.digest_file(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            certificate=temp/'certificate.json'
            identity.write_manifest(certificate,dict(version=1,kind=MAGIC,configuration=broken['configuration'],python_abi=broken['python_abi'],catalog=record(binary),expanded_manifest=record(full),expanded_payload_sha256=broken['manifest_sha256']))
            cases.append((label,dict(spec_manifest=certificate)))
        for name,options in cases:
            logs=[];app=DefaultRuntime(log=logs.append,**options);assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction'
            assert app.mode=='w4-serial' and len(logs)==1,(name,app.mode,logs)
            app.env.lease.require_validated();delivered=[];got=app.generate(prompt,cap,emit=delivered.append,target_language='ja')
            assert got['tokens']==delivered==baseline['tokens'],name
            assert app.session.length==baseline['logical_length'],(name,app.session.length,baseline['logical_length'])
            assert active_KV_hash(app.session)==baseline_KV,name
            rows.append(dict(active32_KV_bitwise_sha256=baseline_KV,id=name,result='PASS',tokens=len(delivered),logs=logs,validated_serial=True,cursor_equal=True))
            app.close();print(json.dumps(rows[-1]),flush=True)
    for event,wanted in [('before-request',None),('prime-submitted',None),('prime-caught-up',None),('target-committed',0),('target-committed',1)]:
        logs=[];app=DefaultRuntime(log=logs.append);assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction';fired=[]
        def kill(name,data):
            if name==event and not fired and (wanted is None or data['accepted']==wanted):
                app.primer.child.terminate();app.primer.child.wait(timeout=5);fired.append(dict(event=name,data=data))
        if event=='before-request':app.primer.child.terminate();app.primer.child.wait(timeout=5);fired.append(dict(event=event))
        app._hook=kill;delivered=[];got=app.generate(prompt,cap,emit=delivered.append,target_language='ja')
        assert fired,(event,wanted)
        assert got['tokens']==delivered==baseline['tokens'],(event,wanted)
        assert app.mode=='w4-serial' and len(logs)==1,(event,wanted,logs)
        assert app.session.length==baseline['logical_length'],(event,wanted,app.session.length,baseline['logical_length'])
        assert active_KV_hash(app.session)==baseline_KV,(event,wanted)
        # The same process remains usable after a worker has died.
        again=app.generate(meta['prompts'][0],25)
        target=serial_generate(app.session,meta['prompts'][0],25)
        assert again['tokens']==target['tokens'] and len(logs)==1
        rows.append(dict(active32_KV_bitwise_sha256=baseline_KV,id='worker-death-'+event+'-'+str(wanted),result='PASS',fired=fired,logs=logs,tokens=len(delivered),cursor_equal=True,reused_after_death=True,fallback=got.get('worker_fallback')))
        app.close();print(json.dumps(rows[-1]),flush=True)
    # If the independent fallback target is invalid too, fail closed. The
    # intentionally invalid fixture is never a deployable export or NPU input.
    import settings83
    import direct_runtime
    original_w4=settings83.W4_MANIFEST
    execute43=direct43.DirectRuntime.execute;execute_serial=direct_runtime.DirectRuntime.execute
    execution_attempts=[]
    def forbidden_execution(*args,**kwargs):
        execution_attempts.append(True);raise AssertionError('unvalidated target attempted execution')
    with tempfile.TemporaryDirectory(prefix='r45-fail-closed-',dir=ROOT/'tmp')as directory:
        folder=Path(directory);broken=__import__('plamo_root').rebase(json.loads(original_w4.read_text())['payload'])
        artifact=broken['model']['embedding']['artifact'];bad_target=folder/'bad-target.bin';bad_target.write_bytes(b'not a validated target')
        broken['artifacts'][artifact]['path']=str(bad_target)
        bad_manifest=folder/'manifest.json';identity.write_manifest(bad_manifest,broken)
        settings83.W4_MANIFEST=bad_manifest;logs=[]
        direct43.DirectRuntime.execute=forbidden_execution;direct_runtime.DirectRuntime.execute=forbidden_execution
        try:
            try:DefaultRuntime(spec_manifest=folder/'missing-certificate.json',log=logs.append)
            except Exception as error:
                assert not isinstance(error,AssertionError),error
                assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction'
                assert len(logs)==1 and not execution_attempts
                rows.append(dict(id='invalid-fallback-target-fails-closed',result='PASS',exception=type(error).__name__,logs=logs,NPU_commands=0))
            else:raise AssertionError('invalid serial fallback target was accepted')
        finally:
            settings83.W4_MANIFEST=original_w4;direct43.DirectRuntime.execute=execute43;direct_runtime.DirectRuntime.execute=execute_serial
    report=dict(result='PASS',cases=len(rows),rows=rows,scope='all fallback outputs, callbacks, and all32 active target KV operands identical to independent serial W4; failed initialization releases partial contexts, no concurrent model loads; permanent serial after worker death; one clear log per transition')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result='PASS',cases=len(rows))),flush=True)
except BaseException:
    import traceback
    error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=error),indent=2));print(error,flush=True);sys.exit(1)
