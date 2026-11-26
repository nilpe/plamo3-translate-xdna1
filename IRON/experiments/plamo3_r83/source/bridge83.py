"""Unmodified target/protocol gates against candidate; no public dispatch change."""
import os,runpy,sys
from pathlib import Path
import settings83 as settings
settings.paths();os.environ['R42_MANIFEST']=str(settings.SPEC_MANIFEST)
kind,script,*args=sys.argv[1:];sys.argv=[script]+args
if kind=='legacy':
 from initialize83 import initialize
 from primer83 import Primer
 import primer
 workers=[];owners=[]
 class GatePrimer(Primer):
  def __init__(self,bundle):super().__init__(settings.DRAFTER_PACKAGE);workers.append(self)
 primer.Primer=GatePrimer
 import experimental_runtime
 def initialized(path):
  env=initialize(settings.SPEC_MANIFEST);owners.append(env)
  for worker in workers:
   if not worker.authorized and not worker.closed:worker.authorize(env.runtime)
  return dict(session=env.session,runtime=env.runtime,validation=env.lease.stats,setup_s=env.setup_s,startup_events=env.events)
 experimental_runtime.initialize=initialized
 import target43_io,target83_base,draft43
 target43_io.base_factory=target83_base.make_direct
 old=draft43.make_direct
 from host44 import make_direct
 def factory(bundle,runtime):
  Base,module=make_direct(bundle,runtime,old)
  class DeclaredJapaneseProbe(Base):
   def __init__(self,bundle):super().__init__(bundle);self.select_language('ja')
  return DeclaredJapaneseProbe,module
 draft43.make_direct=factory
runpy.run_path(script,run_name='__main__')
