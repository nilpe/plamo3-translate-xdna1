"""R83 gate: the gated-path protocol gates must FAIL against three mutated gate comparisons (never fires, always fires, inverted comparison direction);
R83 addition: the confidence gate must FAIL against a fast path with the near-threshold exact fallback dropped (nofallback) or shrunk to 1e-16 (tinyband)."""
import argparse,json,subprocess,sys,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args();here=Path(__file__).parent;rows=[]
for script,mutants in(('eos83_protocol.py',('never','always','inverted')),('cap83_protocol.py',('never','always','inverted')),('fallback83_protocol.py',('never','always','inverted')),('confidence83_gate.py',('nofallback','tinyband'))):
 for mutant in mutants:
  with tempfile.TemporaryDirectory()as d:
   at=time.time();run=subprocess.run([sys.executable,str(here/script),'--mutant',mutant,'--output',str(Path(d)/'x.json')],capture_output=True,text=True,timeout=480)
  failed=run.returncode!=0 and'AssertionError'in run.stderr+run.stdout
  rows.append(dict(script=script,mutant=mutant,returncode=run.returncode,failed_by_assertion=failed,seconds=time.time()-at,tail=(run.stderr+run.stdout)[-300:]))
  print(json.dumps({k:v for k,v in rows[-1].items()if k!='tail'}),flush=True)
ok=all(r['failed_by_assertion']for r in rows);report=dict(result='PASS'if ok else'FAIL',mutants=rows);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(result=report['result'],cases=len(rows))),flush=True);sys.exit(0 if ok else 1)
