"""R83 gate (no NPU): the shorter GIL switch interval is used ONLY in the declared window of speculative startup and is ALWAYS restored.
Declared window: default interval during device_open / xclbin_load / xclbin_register / hw_context_create / kernel_binding (pyxrt holds the GIL
there); short interval only after the startup event 'xclbin_hw_contexts' and until speculative construction ends; restored on success, on
speculative failure BEFORE the serial fallback initialization, on any exception (also BaseException) and at the end of construction.
The real DefaultRuntime constructor and the real initialize83 event wiring run; only the initializer / primer are fakes. Window mutants must FAIL
this gate (early open at construction start, open on device_open, open at the first hw_context, no restore, wrong restore value)."""
import argparse,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
import settings83 as settings
settings.paths()
import switchwin83
from default83 import DefaultRuntime
DEFAULT=sys.getswitchinterval();assert DEFAULT==0.005,DEFAULT
sys.setswitchinterval(settings.STARTUP_SWITCH_INTERVAL_S);STARTUP=sys.getswitchinterval();sys.setswitchinterval(DEFAULT);assert 0<STARTUP<0.005 and abs(STARTUP-settings.STARTUP_SWITCH_INTERVAL_S)<1e-9
HW=['xclbin_load:context0','xclbin_register:context0','hw_context_create:context0','kernel_binding:context0','xclbin_load:context1','xclbin_register:context1','hw_context_create:context1','kernel_binding:context1','xclbin_load:r39_generalized_context','xclbin_register:r39_generalized_context','hw_context_create:r39_generalized_context','kernel_binding:r39_generalized_context','xclbin_load:r46_prefill_body','xclbin_register:r46_prefill_body','hw_context_create:r46_prefill_body','kernel_binding:r46_prefill_body']
PHASES=['validation_manifest_json_schema_config','validation_source_inventory','pyxrt_import','device_open']+HW+['xclbin_hw_contexts','validation_ntfs_identity_and_fallback','bo_allocate_map_initialize_sync']
TRIG=PHASES.index('xclbin_hw_contexts')
class Survived(Exception):pass
def env():return types.SimpleNamespace(lease=types.SimpleNamespace(require_validated=lambda:None),session=types.SimpleNamespace(),events=[],close=lambda:None,runtime=None)
class FakePrimer:
 closed=False
 def __init__(self,bundle):pass
 def close(self):self.closed=True
def run(name,mode,prior,behaviours,primer_factory=None,*,expect_error=None,expect_opened=None,expect_serial_second_call=True):
 traces=[];calls=[]
 def initialize(manifest,*,on_validated=None):
  index=len(calls);calls.append(str(manifest));behaviour=behaviours[min(index,len(behaviours)-1)];trace=[];traces.append(trace)
  entry=sys.getswitchinterval()
  limit={'ok':len(PHASES),'raise-before':PHASES.index('device_open')+1,'raise-in-ctx':PHASES.index('hw_context_create:context1')+1,'raise-after':len(PHASES),'interrupt-after':len(PHASES)}[behaviour]
  for phase in PHASES[:limit]:
   before=sys.getswitchinterval();switchwin83.notify(phase);trace.append((phase,before,sys.getswitchinterval()))
  if behaviour in('raise-before','raise-in-ctx','raise-after'):raise ValueError('injected initialization failure')
  if behaviour=='interrupt-after':raise KeyboardInterrupt('injected')
  return env()
 sys.setswitchinterval(prior);logs=[];error=None;app=None
 try:app=DefaultRuntime(mode,log=logs.append,initialize=initialize,primer_factory=primer_factory or FakePrimer)
 except BaseException as exc:error=exc
 after=sys.getswitchinterval();sys.setswitchinterval(DEFAULT)
 assert after==prior,(name,'interval not restored',after,prior)
 assert switchwin83._active is None,(name,'window still installed')
 speculative=mode in(None,'w4-m2')
 for index,trace in enumerate(traces):
  spec_call=speculative and index==0 and primer_factory is None
  for position,(phase,before,end) in enumerate(trace):
   if not spec_call:
    assert before==prior and end==prior,(name,index,phase,'interval changed outside the speculative initialize call',before,end)
   elif position<TRIG:
    assert before==prior and end==prior,(name,phase,'short interval before the declared window (during device/xclbin/hw_context)',before,end)
   elif position==TRIG:
    assert before==prior and end==STARTUP,(name,phase,'window must open exactly at xclbin_hw_contexts',before,end)
   else:
    assert before==STARTUP and end==STARTUP,(name,phase,'window must stay open after xclbin_hw_contexts',before,end)
 record=app.startup.get('switch_window') if app is not None else None
 if app is not None:
  assert app.startup['switch_interval_restored_s']==prior and app.startup['switch_interval_default_s']==prior
  assert ('switch_window'in app.startup)==speculative
  if speculative and primer_factory is None and expect_opened is not None:assert record['opened']==expect_opened and record['closed'],(name,record)
  if record and record.get('opened'):
   assert record['trigger']=='xclbin_hw_contexts' and set(HW)<=set(record['seen_before_open']),(name,record)
   assert record['seen_before_open'][:len(HW)+4]==PHASES[:len(HW)+4],(name,'hw phases must all precede the window',record['seen_before_open'])
 if expect_error is not None:assert isinstance(error,expect_error),(name,error)
 else:assert error is None,(name,error)
 return dict(case=name,mode=mode,prior=prior,fell_back=bool(app and app.fallbacks),raised=type(error).__name__ if error else None,initialize_calls=len(calls),window=record)
def cases():
 rows=[]
 # speculative: the fake runtime is incomplete, so after the trigger the speculative branch fails and the validated serial fallback runs at the default interval
 rows.append(run('speculative-reaches-trigger-then-fails-then-serial','w4-m2',DEFAULT,['ok','ok'],expect_opened=True))
 rows.append(run('speculative-raises-after-trigger-then-serial','w4-m2',DEFAULT,['raise-after','ok'],expect_opened=True))
 rows.append(run('speculative-raises-inside-hw-contexts-never-opens','w4-m2',DEFAULT,['raise-in-ctx','ok'],expect_opened=False))
 rows.append(run('speculative-raises-before-device','w4-m2',DEFAULT,['raise-before','ok'],expect_opened=False))
 rows.append(run('both-fail-closed','w4-m2',DEFAULT,['raise-after','raise-after'],expect_error=ValueError,expect_opened=True))
 rows.append(run('interrupt-after-open','w4-m2',DEFAULT,['interrupt-after','ok'],expect_error=KeyboardInterrupt,expect_opened=True))
 rows.append(run('serial-w4','w4-serial',DEFAULT,['ok']));rows.append(run('serial-w8','w8-serial',DEFAULT,['ok']))
 rows.append(run('serial-fails','w4-serial',DEFAULT,['raise-after'],expect_error=ValueError))
 def bad_primer(bundle):raise OSError('injected primer failure')
 rows.append(run('primer-factory-fails','w4-m2',DEFAULT,['ok','ok'],primer_factory=bad_primer))
 rows.append(run('non-default-prior-interval','w4-m2',0.02,['raise-after','ok'],expect_opened=True))
 rows.append(run('default-mode-none',None,DEFAULT,['raise-after','ok'],expect_opened=True))
 return rows
def wiring():
 # Real initialize83 wiring: Environment.events notifications must reach the active window, and the Environment class must be restored.
 import initialize45,initialize83
 original=initialize45.Environment;saved=initialize83.old_initialize;seen=[]
 def fake_old(manifest,*,on_validated=None):
  e=initialize45.Environment();assert type(e) is not original
  for phase in PHASES:
   before=sys.getswitchinterval();e.events.append(dict(phase=phase,ms=1.0,end_ms=1.0));seen.append((phase,before,sys.getswitchinterval()))
  return e
 window=switchwin83.Window(settings.STARTUP_SWITCH_INTERVAL_S);window.arm();switchwin83.install(window)
 try:
  initialize83.old_initialize=fake_old
  e=initialize83.initialize(Path('nonexistent-r83-wiring-manifest.json'))
  assert initialize45.Environment is original,'Environment class not restored'
  assert [x['phase'] for x in e.events]==PHASES
  for position,(phase,before,end) in enumerate(seen):
   if position<TRIG:assert before==DEFAULT and end==DEFAULT,('wiring',phase,before,end)
   elif position==TRIG:assert before==DEFAULT and end==STARTUP,('wiring',phase,before,end)
   else:assert before==STARTUP and end==STARTUP,('wiring',phase,before,end)
  record=window.close();assert sys.getswitchinterval()==DEFAULT and record['opened']
 finally:
  switchwin83.uninstall(window);window.close();initialize83.old_initialize=saved;sys.setswitchinterval(DEFAULT)
  assert initialize45.Environment is original
 def boom(manifest,*,on_validated=None):raise ValueError('boom')
 try:
  initialize83.old_initialize=boom
  try:initialize83.initialize(Path('nonexistent-r83-wiring-manifest.json'));raise Survived('no error')
  except ValueError:pass
  assert initialize45.Environment is original,'Environment class not restored after failure'
 finally:initialize83.old_initialize=saved
 return dict(case='initialize83-wiring',events=len(PHASES),opened_exactly_at='xclbin_hw_contexts')
class Early(switchwin83.Window):
 def __init__(self,short_s):super().__init__(short_s);sys.setswitchinterval(short_s);self.opened=True
class OnDevice(switchwin83.Window):trigger='device_open'
class OnFirstContext(switchwin83.Window):trigger='hw_context_create:context0'
class OnLastContextCreate(switchwin83.Window):trigger='hw_context_create:r46_prefill_body'
class NoRestore(switchwin83.Window):
 def close(self):
  self.armed=False;self.closed=True;self.record.update(opened=self.opened,closed=True,restored_s=sys.getswitchinterval());return self.record
class WrongRestore(switchwin83.Window):
 def close(self):
  was=self.opened;self.armed=False
  if was and not self.closed:sys.setswitchinterval(0.005)
  self.closed=True;self.record.update(opened=was,closed=True,restored_s=sys.getswitchinterval());return self.record
real=switchwin83.Window
rows=cases();wire=wiring()
mutants=[]
for mutant in(Early,OnDevice,OnFirstContext,OnLastContextCreate,NoRestore,WrongRestore):
 switchwin83.Window=mutant
 try:
  try:cases();wiring();raise Survived(mutant.__name__)
  except AssertionError as exc:mutants.append(dict(mutant=mutant.__name__,failed_as_required=True,first_failure=str(exc)[:160]))
 finally:
  switchwin83.Window=real;sys.setswitchinterval(DEFAULT);switchwin83._active=None
assert len(mutants)==6
a.output.write_text(json.dumps(dict(result='PASS',startup_interval_s=STARTUP,default_interval_s=DEFAULT,window='opens at startup event xclbin_hw_contexts, closes at end of speculative construction / before serial fallback / on any exit',cases=rows,wiring=wire,mutants=mutants),indent=2));print(json.dumps(dict(result='PASS',cases=len(rows),mutants=len(mutants))),flush=True)
