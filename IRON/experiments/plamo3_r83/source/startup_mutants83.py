"""R83 startup-attestation mutant gate (no device, no model).

Startup must still REFUSE (before the identity commit / any validated use) each of: a modified bound file, a replaced file whose content differs,
a missing file, an extra source on disk that the manifest does not bind, a bound source that is gone, a malformed record, a retained input whose
identity changes during setup, a changed commit-time inventory/configuration. It must still ACCEPT (by full content SHA-256, as before) a replaced
file with identical content. Every case is run against the real helper process AND the in-process fallback; helper death / hang / garbage must fall
back to in-process attestation (still refusing a mutated file), never skip it.
Part 1: real temp files + the real helper (no manifest involved). Part 2: the real candidate certificate with the manifest mutated after load."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import settings83 as s
s.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--spec',type=Path);a=p.parse_args()
spec=a.spec or s.SPEC_MANIFEST
import identity43,lease83,srccheck83
from identity43 import base
import runtime_identity as ri
rows=[]
def row(id,**k):rows.append(dict(id=id,result='PASS',**k));print(json.dumps(rows[-1]),flush=True)
def record(path):
    with open(path,'rb') as f:
        data=f.read();f.seek(0);ident=ri.maybe_identity(f)
    assert ident is not None
    return dict(path=str(path),nbytes=len(data),chunk_bytes=base.CHUNK,chunks_sha256=[hashlib.sha256(data).hexdigest()],ntfs_identity=ident)
def helper(root,recs,*,expected=None,spawn=None):
    request=dict(items=[(r['path'],[r]) for r in recs],roots=[str(root)],expected_inventory=expected if expected is not None else base.source_files([str(root)]),version_ok=True,force_audit=False)
    ch=(spawn or srccheck83.SourceChecker.spawn)(lease83.APP_DIR);assert ch is not None and ch.send(request)
    try:return ch.result()
    finally:ch.close()
# ---------------- part 1: real files + the real helper process ----------------
tmp=Path(tempfile.mkdtemp(prefix='r83gate-',dir=str(s.ROOT/'state')))
try:
    for i in range(6):(tmp/f'm{i}.py').write_bytes((f'x={i}\n'*50).encode())
    recs=[record(tmp/f'm{i}.py') for i in range(6)]
    r=helper(tmp,recs);assert r and r['ok'] and r['matches']==6 and r['fallback_paths']==[],r;row('helper-control-unmodified',helper_result=r)
    # modified bound file: same size, one byte flipped in place (identity changes -> full content hash -> mismatch)
    data=bytearray((tmp/'m1.py').read_bytes());data[3]^=1;(tmp/'m1.py').write_bytes(bytes(data))
    r=helper(tmp,recs);assert r and not r['ok'] and 'content changed' in r['message'],r;row('helper-modified-same-size-refused',message=r['message'])
    (tmp/'m1.py').write_bytes(b'x=1\n'*50)  # restore content (new identity, same bytes) = "replaced with identical content"
    recs2=[record(tmp/f'm{i}.py') for i in range(6)];(tmp/'m1.py').write_bytes(b'x=1\n'*50)
    r=helper(tmp,recs);assert r and r['ok'] and r['fallback_paths']==[str(tmp/'m1.py')] and r['hashed_bytes']==recs[1]['nbytes'],r;row('helper-replaced-identical-content-accepted-by-content-hash',fallback_paths=r['fallback_paths'])
    with open(tmp/'m2.py','ab') as f:f.write(b'#')
    r=helper(tmp,recs);assert r and not r['ok'] and 'size changed' in r['message'],r;row('helper-modified-appended-refused',message=r['message'])
    (tmp/'m2.py').write_bytes(b'x=2\n'*50)
    os.remove(tmp/'m3.py')
    r=helper(tmp,recs,expected=[str(tmp/f'm{i}.py') for i in range(6)]);assert r and not r['ok'] and r['kind']=='os',r;row('helper-missing-file-refused',message=r['message'])
    recs_wo=[x for x in recs if not x['path'].endswith('m3.py')]
    r=helper(tmp,recs_wo,expected=[str(tmp/f'm{i}.py') for i in range(6)]);assert r and not r['ok'] and 'inventory' in r['message'],r;row('helper-bound-source-removed-inventory-refused',message=r['message'])
    (tmp/'m3.py').write_bytes(b'x=3\n'*50)
    recs_all=[record(tmp/f'm{i}.py') for i in range(6)]
    (tmp/'extra.py').write_text('y=1\n')
    r=helper(tmp,recs_all,expected=[str(tmp/f'm{i}.py') for i in range(6)]);assert r and not r['ok'] and 'inventory' in r['message'],r;row('helper-extra-unbound-source-inventory-refused',message=r['message'])
    r=helper(tmp,recs_all,expected=base.source_files([str(tmp)]),);assert r and r['ok'],r  # extra now bound only if records exist
    bad=copy.deepcopy(recs_all);bad[0].pop('chunks_sha256')
    r=helper(tmp,bad);assert r and not r['ok'],r;row('helper-malformed-record-refused',message=r['message'])
    # helper identity query equals the parent's (same bytes of the identity dict)
    recs_all=[record(tmp/f'm{i}.py') for i in range(6)]
    with open(tmp/'m0.py','rb') as f:mine=identity43.maybe_identity(f);theirs=ri.maybe_identity(f)
    assert mine==theirs==recs_all[0]['ntfs_identity'];row('identity-query-equal-between-identity43-and-runtime_identity',identity=mine)
finally:shutil.rmtree(tmp,ignore_errors=True)
# ---------------- part 2: the real certificate, manifest mutated after load ----------------
for k,v in json.loads(spec.read_text())['payload']['configuration'].items():os.environ[k]=v
real_load=lease83.load_deferred
def mutated_load(mutate):
    def load(path,**kw):
        m=real_load(path,**kw);m=dict(m);m['sources']=copy.deepcopy(m['sources']);mutate(m);return m
    return load
def try_lease(mutate=None,*,use_checker=True,spawn=None,timeout=None,after_ctor=None):
    lease83.USE_CHECKER=use_checker
    if mutate:lease83.load_deferred=mutated_load(mutate)
    old_spawn=srccheck83.SourceChecker.spawn;old_timeout=srccheck83.SourceChecker.TIMEOUT_S
    if spawn:srccheck83.SourceChecker.spawn=classmethod(lambda cls,app,spawn=spawn:spawn(app))
    if timeout:srccheck83.SourceChecker.TIMEOUT_S=timeout
    lease=None;reason=None
    try:
        lease=lease83.IdentityLease(spec,workers=2)
        if after_ctor:after_ctor(lease)
        lease.wait();lease.require_validated()
    except (base.StaleManifest,OSError) as e:reason=f'{type(e).__name__}: {e}'
    finally:
        lease83.load_deferred=real_load;lease83.USE_CHECKER=True;srccheck83.SourceChecker.spawn=old_spawn;srccheck83.SourceChecker.TIMEOUT_S=old_timeout
        stats=dict(lease.stats) if lease is not None else None;validated=lease._validated if lease is not None else False
        if lease is not None:lease.close()
    return reason,stats,validated
def usable(m):
    # a plain source record that exists on disk
    runtime={r['path']for r in[*m['origins'],*m['artifacts'].values()]}
    return next(i for i,r in enumerate(m['sources'])if r['path'].endswith('.py')and r['path']not in runtime)
def mod_modified(m):
    r=m['sources'][usable(m)];r['ntfs_identity']=dict(r['ntfs_identity'],usn=r['ntfs_identity']['usn']+1);h=r['chunks_sha256'][0];r['chunks_sha256']=[('0'if h[0]!='0'else'1')+h[1:]]+r['chunks_sha256'][1:]
def mod_replaced_identical(m):
    r=m['sources'][usable(m)];r['ntfs_identity']=dict(r['ntfs_identity'],usn=r['ntfs_identity']['usn']+1)
def mod_missing(m):
    r=m['sources'][usable(m)];r['path']=r['path']+'.missing'
def mod_extra_on_disk(m):  # a bound source that exists on disk is not bound by the manifest -> disk has an extra
    del m['sources'][usable(m)]
def mod_extra_bound(m):  # manifest binds a source the disk does not have
    r=copy.deepcopy(m['sources'][usable(m)]);r['path']=r['path']+'.gone';m['sources'].append(r)
def mod_malformed(m):
    m['sources'][usable(m)].pop('chunks_sha256')
class Dead:  # helper that exits immediately with no output
    @staticmethod
    def spawn(app):
        proc=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.buffer.read();sys.exit(3)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0);return srccheck83.SourceChecker(proc)
class Garbage:
    @staticmethod
    def spawn(app):
        proc=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.buffer.read();sys.stdout.write("{not json")'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0);return srccheck83.SourceChecker(proc)
class Hang:
    @staticmethod
    def spawn(app):
        proc=subprocess.Popen([sys.executable,'-c','import sys,time;sys.stdin.buffer.read();time.sleep(60)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0);return srccheck83.SourceChecker(proc)
class LiarOk:  # helper claiming success for a different number of files -> must not be trusted
    @staticmethod
    def spawn(app):
        proc=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.buffer.read();sys.stdout.write(\'{"ok":true,"files":1,"matches":1,"hashed_bytes":0,"fallback_paths":[],"inventory_files":1,"child_ms":1}\')'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0);return srccheck83.SourceChecker(proc)
reason,stats,validated=try_lease();assert reason is None and validated and stats['source_attestation']=='helper-process' and stats['unique_files']==stats['identity_matches']and stats['fallback_files']==0,(reason,stats);row('lease-control-helper',stats={k:v for k,v in stats.items()if k!='fallback_paths'})
reason,stats,validated=try_lease(use_checker=False);assert reason is None and validated and stats['source_attestation'].startswith('in-process')and stats['unique_files']==stats['identity_matches'],(reason,stats);row('lease-control-inprocess',stats={k:v for k,v in stats.items()if k!='fallback_paths'})
for name,mut,expect in(('modified-file',mod_modified,'content changed'),('missing-file',mod_missing,''),('extra-source-on-disk-unbound',mod_extra_on_disk,'inventory'),('bound-source-removed-from-disk',mod_extra_bound,''),('malformed-record',mod_malformed,'')):
    for mode,kw in(('helper',{}),('inprocess',dict(use_checker=False)),('helper-dead-fallback',dict(spawn=Dead.spawn)),('helper-garbage-fallback',dict(spawn=Garbage.spawn)),('helper-liar-fallback',dict(spawn=LiarOk.spawn))):
        reason,stats,validated=try_lease(mut,**kw)
        assert reason is not None and not validated,(name,mode,'ACCEPTED',stats)
        assert expect in reason,(name,mode,reason)
        row(f'lease-{name}-{mode}-refused',reason=reason[:200])
reason,stats,validated=try_lease(mod_replaced_identical);assert reason is None and validated and stats['fallback_files']==1 and stats['hashed_bytes']>0 and len(stats['fallback_paths'])==1,(reason,stats);row('lease-replaced-identical-content-accepted-by-content-hash',fallback_paths=stats['fallback_paths'],hashed_bytes=stats['hashed_bytes'])
reason,stats,validated=try_lease(mod_replaced_identical,use_checker=False);assert reason is None and validated and stats['fallback_files']==1,(reason,stats);row('lease-replaced-identical-content-inprocess',fallback_paths=stats['fallback_paths'])
for name,spawn in(('helper-dead',Dead.spawn),('helper-garbage',Garbage.spawn),('helper-liar',LiarOk.spawn)):
    reason,stats,validated=try_lease(spawn=spawn);assert reason is None and validated and stats['source_attestation']=='in-process (fallback)' and stats['unique_files']==stats['identity_matches'],(name,reason,stats);row(f'lease-{name}-falls-back-and-validates',source_attestation=stats['source_attestation'])
at=time.time();reason,stats,validated=try_lease(spawn=Hang.spawn,timeout=2.0);assert reason is None and validated and stats['source_attestation']=='in-process (fallback)',(reason,stats);row('lease-helper-hang-times-out-and-falls-back',seconds=round(time.time()-at,1))
# replaced file during setup (a retained runtime input changes identity after the ctor): commit refuses
def retained_identity_changes(lease):
    target=next(n for n,(st,mp,ident) in lease._owners.items()if n.endswith('PLaMo-manifest.json'))
    real=identity43.maybe_identity;stream=lease._owners[target][0]
    identity43.maybe_identity=lambda st:dict(R83_FAULT=True)if st is stream else real(st)
    retained_identity_changes.restore=lambda:setattr(identity43,'maybe_identity',real)
try:
    reason,stats,validated=try_lease(after_ctor=retained_identity_changes)
finally:retained_identity_changes.restore()
assert reason and 'identity changed during setup' in reason and not validated,(reason,);row('lease-retained-input-replaced-during-setup-refused',reason=reason[:160])
# commit-time inventory / configuration changes (existing R48-R61 cases, now with the concurrent commit)
old_inv,old_cfg=base.source_files,base.configuration
try:
    base.source_files=lambda roots:old_inv(roots)+['R83_INVALID_SOURCE']
    reason,stats,validated=try_lease();assert reason and not validated,reason
finally:base.source_files=old_inv
row('lease-commit-inventory-changed-refused',reason=reason[:160])
try:
    base.configuration=lambda:dict(old_cfg(),R83_INVALID_CONFIGURATION='fault')
    reason,stats,validated=try_lease();assert reason and not validated,reason
finally:base.configuration=old_cfg
row('lease-configuration-changed-refused',reason=reason[:160])
report=dict(result='PASS',checks=len(rows),rows=rows,spec=str(spec),no_device_model_or_package_execution=True)
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result='PASS',checks=len(rows))),flush=True)
