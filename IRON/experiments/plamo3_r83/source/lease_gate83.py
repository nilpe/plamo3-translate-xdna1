"""Real retained files: commit failures and unavailable-journal content fallback."""
import argparse,json,os
from pathlib import Path
import settings83 as s
s.paths()
import identity43
from lease83 import IdentityLease
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
for k,v in json.loads(s.SPEC_MANIFEST.read_text())['payload']['configuration'].items():os.environ[k]=v
base=identity43.base
old_identity,old_config,old_inventory=identity43.maybe_identity,base.configuration,base.source_files
old_open=base._open_retained
try:
 for case in ('identity-changed','configuration-changed','inventory-changed','journal-unavailable'):
  target=str(s.DRAFTER_PACKAGE/('PLaMo-manifest.json'if case!='journal-unavailable'else json.loads((s.DRAFTER_PACKAGE/'PLaMo-manifest.json').read_text())['head']['exact_file']))
  lease=None
  try:
   if case=='journal-unavailable':
    selected=[]
    def tracked(path):
     stream=old_open(path)
     if str(path)==target:selected.append(stream)
     return stream
    base._open_retained=tracked
    def no_journal(stream):return None if any(stream is x for x in selected)else old_identity(stream)
    identity43.maybe_identity=no_journal
   lease=IdentityLease(s.SPEC_MANIFEST,workers=2)
   if case=='identity-changed':
    def changed(stream):
     return dict(R48_FAULT=True)if stream is lease._owners[target][0]else old_identity(stream)
    identity43.maybe_identity=changed
   elif case=='configuration-changed':base.configuration=lambda:dict(old_config(),R48_INVALID_CONFIGURATION='fault')
   elif case=='inventory-changed':base.source_files=lambda roots:old_inventory(roots)+['R48_INVALID_SOURCE']
   if case=='journal-unavailable':
    lease.wait();lease.require_validated();assert target in lease.stats['fallback_paths'];assert lease.stats['hashed_bytes']>=Path(target).stat().st_size
    row=dict(id=case,result='PASS',full_content_fallback=True,stats=dict(lease.stats))
   else:
    try:lease.wait()
    except base.StaleManifest as error:reason=str(error)
    else:raise AssertionError('invalid lease committed: '+case)
    try:lease.require_validated()
    except base.StaleManifest:pass
    else:raise AssertionError('failed lease authorized use')
    assert not lease._validated
    row=dict(id=case,result='PASS',failed_before_authorization=True,reason=reason)
   rows.append(row)
  finally:
   identity43.maybe_identity,base.configuration,base.source_files=old_identity,old_config,old_inventory
   base._open_retained=old_open
   if lease is not None:lease.close()
 # ---- R83: tampering with ANY retained runtime-read file is rejected (package files and every other bound file) ----
 import ctypes
 from ctypes import wintypes
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);create=kernel.CreateFileW;create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE;closeh=kernel.CloseHandle;closeh.argtypes=[wintypes.HANDLE]
 INVALID=ctypes.c_void_p(-1).value
 def access_allowed(path,access):
  # Probe only: open with write/delete access and full sharing, never modify. A valid handle means the retention failed.
  h=create(str(path),access,7,None,3,0x80,None)
  if h==INVALID:return False
  closeh(h);return True
 lease=IdentityLease(s.SPEC_MANIFEST,workers=2)
 try:
  lease.wait();lease.require_validated();owners=dict(lease._owners);policy=lease.manifest['r48_attestation'];package={str(Path(x)).casefold()for x in policy['package_paths']}
  assert package<={str(Path(x)).casefold()for x in owners}
  gate_only=[str(s.DRAFTER_PACKAGE.parent/n)for n in policy['r56_scope']['gate_only_relative']];assert len(gate_only)==133
  retained_cf={str(Path(x)).casefold()for x in owners};assert not any(str(Path(x)).casefold()in retained_cf for x in gate_only)
  real=identity43.maybe_identity;changed_rejected=0
  for name,(stream,_,before)in owners.items():
   lease._failure=None;lease._validated=False
   identity43.maybe_identity=lambda st,target=stream:dict(R83_FAULT=True)if st is target else real(st)
   try:lease.wait()
   except base.StaleManifest:changed_rejected+=1
   else:raise AssertionError('changed identity accepted: '+name)
   finally:identity43.maybe_identity=real
   assert not lease._validated
  lease._failure=None;lease._validated=False;lease.wait();lease.require_validated()
  write_denied=delete_denied=0
  for name in owners:
   assert not access_allowed(name,0x40000000),'write access obtained: '+name
   write_denied+=1
   assert not access_allowed(name,0x00010000),'delete/rename access obtained: '+name
   delete_denied+=1
  rows.append(dict(id='every-retained-file-identity-changed',result='PASS',retained_files=len(owners),rejected=changed_rejected,package_runtime_files=len(package),all_rejected_before_authorization=True))
  rows.append(dict(id='every-retained-file-write-and-delete-denied',result='PASS',write_denied=write_denied,delete_denied=delete_denied))
  rows.append(dict(id='gate-only-files-not-retained',result='PASS',gate_only_files=len(gate_only),note='excluded by user decision 2026-11-17; verified+retained by gate processes (gatefiles83)'))
 finally:lease.close()
 class Flipped:
  def __init__(self,stream):self._s=stream;self._flipped=False
  def read(self,*args):
   data=self._s.read(*args)
   if data and not self._flipped:self._flipped=True;data=bytes([data[0]^1])+data[1:]
   return data
  def __getattr__(self,name):return getattr(self._s,name)
  def __enter__(self):return self
  def __exit__(self,*a):self._s.close()
 tampered=0
 for target in sorted(package):
  actual=next(str(x)for x in policy['package_paths']if str(Path(x)).casefold()==target)
  def tracked(path,actual=actual):
   stream=old_open(path);return Flipped(stream)if str(path)==actual else stream
  base._open_retained=tracked
  identity43.maybe_identity=lambda st,real=real:None if isinstance(st,Flipped)else real(st)
  lease=None
  try:
   lease=IdentityLease(s.SPEC_MANIFEST,workers=2);lease.wait()
  except base.StaleManifest as error:
   assert 'content changed' in str(error)or 'short read' in str(error),str(error);tampered+=1
  else:raise AssertionError('content-tampered package file accepted: '+actual)
  finally:
   identity43.maybe_identity=real;base._open_retained=old_open
   if lease is not None:lease.close()
 rows.append(dict(id='every-runtime-package-file-content-tampered',result='PASS',files=tampered,note='identity unavailable -> full chunk SHA-256 of the retained handle; one flipped bit rejected for each runtime-read package file'))
 report=dict(result='PASS',checks=len(rows),rows=rows,no_device_model_or_package_execution=True)
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
except BaseException:
 import traceback
 error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,error=error),indent=2));raise
