"""Fresh processes: literal R46 default v4C versus isolated R48 v5."""
import os,runpy,sys,types
from pathlib import Path
import settings83
if os.environ.get('R83_PACKAGE')=='v4C':
 from default45 import DefaultRuntime as Original
 class Baseline(Original):
  def generate(self,*args,target_language=None,**kwargs):return super().generate(*args,**kwargs)
 sys.modules['default83']=types.SimpleNamespace(DefaultRuntime=Baseline)
runpy.run_path(str(Path(__file__).with_name('timing83.py')),run_name='__main__')
