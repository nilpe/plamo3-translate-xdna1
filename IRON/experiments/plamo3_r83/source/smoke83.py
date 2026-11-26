import argparse,json,sys,time
from pathlib import Path
from settings83 import ROOT,paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--mode',default='smoke');a=p.parse_args()
try:
 from default83 import DefaultRuntime
 from loop_gate import serial_generate
 meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text())
 app=DefaultRuntime();assert sys.getswitchinterval()==0.005,'GIL switch interval not restored after construction';assert app.startup['switch_interval_startup_s']<app.startup['switch_interval_default_s'];import switchwin83;switchwin83.assert_declared_window(app.startup)
 assert app.mode=='w4-m2',app.fallbacks
 s=app.session;prompt=meta['prompts'][0];base=serial_generate(s,prompt,256);tokens=[];got=app.generate(prompt,256,emit=tokens.append,target_language='en');assert tokens==got['tokens']==base['tokens']
 report=dict(result='PASS',mode=app.mode,spec_ready_ms=app.spec_ready_ms,startup=app.startup,serial=base,default=got,fallbacks=app.fallbacks)
 print(json.dumps(dict(result='TOKENS_PASS',spec_ready_ms=app.spec_ready_ms,startup=app.startup)),flush=True)
 # Test actual quiescent teardown, not os._exit-only ownership.
 app.close();app.close();report['idempotent_close']=True
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result='PASS',spec_ready_ms=report['spec_ready_ms'],tokens=len(tokens),M2=len(got['cycles']),close=True)),flush=True)
except BaseException:
 import traceback;error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',error=error),indent=2));print(error,flush=True);sys.exit(1)
