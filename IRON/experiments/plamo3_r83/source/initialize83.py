"""R46 validation policy, with explicit complete v5 package coverage."""
import settings83
settings83.paths()
from pathlib import Path
import hashlib,json
from initialize45 import initialize as old_initialize

TOK_EVENT_HOOK=None   # R83: optional callable(phase) set by default83 (tokenizer prefetch start point 'root0')

def package_hash():
 # Export/inspection only. This is not called from runtime initialization.
 root=settings83.DRAFTER_PACKAGE.parent;inventory=root/'PLaMo-PACKAGE-SHA256.json';rows=json.loads(inventory.read_text());total=0
 from runtime_manifest import _open_retained
 for name,row in rows.items():
  with _open_retained(root/name)as f:
   h=hashlib.sha256();count=0
   while b:=f.read(1024*1024):h.update(b);count+=len(b)
   if count!=row['bytes']or h.hexdigest()!=row['sha256']:raise ValueError('package content mismatch: '+name)
   total+=count
 return dict(mode='export-full-SHA256',files=len(rows),bytes=total,inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest())

def initialize(manifest,*,on_validated=None):
 # Every startup event is also reported to the declared switch-interval window (switchwin83); it opens only on xclbin_hw_contexts.
 import initialize45,switchwin83,turbo83
 # R83: the opt-in turbo switch (when a session is active) holds the startup at the event recorded right before device_open until its
 # `configure` has returned; barrier_hook() is None when the switch is off, so the default path does nothing here.
 barrier=turbo83.barrier_hook()
 class Events(list):
  def append(self,event):
   list.append(self,event);switchwin83.notify(event['phase'])
   if TOK_EVENT_HOOK is not None:
    try:TOK_EVENT_HOOK(event['phase'])
    except Exception:pass
   if barrier is not None:barrier(event['phase'])
 class Environment(initialize45.Environment):
  def __init__(self):
   super().__init__();self.events=Events()
 original_environment=initialize45.Environment
 try:
  initialize45.Environment=Environment
  return _initialize(manifest,on_validated=on_validated)
 finally:
  initialize45.Environment=original_environment

def _initialize(manifest,*,on_validated=None):
 # H1: the v4C compatibility branch (old initializer for the v4C package) is gone with the v4C package; the R46 retained-attestation path below is the only one.
 def validated(runtime):
  if 'r40_cycle'in runtime.manifest:
   lease=runtime._artifact_lease;lease.require_validated();policy=runtime.manifest.get('r48_attestation')
   if not policy or policy['scheme']!='R46-retained-NTFS-attestation-v1':raise ValueError('missing package attestation policy')
   expected={str(Path(p)).casefold()for p in policy['package_paths']};retained={str(Path(p)).casefold()for p in lease._owners}
   if len(expected)!=policy['files']or not expected<=retained:raise ValueError('package coverage/retention incomplete')
   scope=policy.get('r56_scope')
   if scope:
    # Retained runtime files + declared gate-only files must account for the whole (retained, bound) package inventory.
    inventory=Path(scope['inventory_file_retained']);root=inventory.parent;names=set(json.loads(inventory.read_bytes()))
    runtime_names={Path(p).relative_to(root).as_posix()for p in policy['package_paths']}-{inventory.name}
    if runtime_names|set(scope['gate_only_relative'])!=names or runtime_names&set(scope['gate_only_relative']):raise ValueError('package scope does not cover the package inventory')
   runtime._r48_package_hash=dict(mode='R46-retained-NTFS-attestation-v1',files=len(expected),validation=dict(lease.stats),export_inventory_sha256=policy['inventory_sha256'],all_package_files_retained=True)
  if on_validated:on_validated(runtime)
 # Candidate-only selection; existing default and serial sources are unchanged.
 import identity43
 from lease83 import IdentityLease
 original=identity43.IdentityLease
 try:
  identity43.IdentityLease=IdentityLease
  return old_initialize(manifest,on_validated=validated)
 finally:
  identity43.IdentityLease=original
