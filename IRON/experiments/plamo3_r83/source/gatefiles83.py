"""R83 gate-only package files (cycle/service gold vectors): the runtime never reads them and they are no longer attested at startup.
Gate processes retain them with deny-write/delete handles and verify every byte against the package inventory first."""
import hashlib,json
from pathlib import Path
_HELD=[]
def retain_verified(bundle):
 bundle=Path(bundle);root=bundle.parent;entries=json.loads((root/'PLaMo-PACKAGE-SHA256.json').read_bytes());from runtime_manifest import _open_retained
 count=0
 for name,row in entries.items():
  if not name.startswith(('PLaMo-D20-M2/PLaMo-vectors/','PLaMo-D20-M2/PLaMo-service-vectors/')):continue
  f=_open_retained(root/name);h=hashlib.sha256();n=0
  while b:=f.read(1<<20):h.update(b);n+=len(b)
  if n!=row['bytes']or h.hexdigest()!=row['sha256']:raise ValueError('gate vector differs from the package inventory: '+name)
  _HELD.append(f);count+=1
 if count!=133:raise ValueError('gate vector count '+str(count))
 return count
