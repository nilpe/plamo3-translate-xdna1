"""Run the unchanged R36/R35 verifier gates against the R40 merged bundle."""
import argparse,json,os,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='PLaMo-D7-M2');p.add_argument('--mode',choices=['teacher','rollback','generation','features'],required=True);a=p.parse_args();root=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));manifest=Path(os.environ.get('R42_MANIFEST',str(root/'plamo3-verify-r43/final'/a.model/'manifest.json')))
for k,v in json.loads(manifest.read_text())['payload']['configuration'].items():os.environ[k]=v
sys.path.insert(0,str(root/'IRON/iron/applications/plamo3'));sys.path.insert(0,str(root/'lib/r36'));sys.path.insert(0,str(root/'lib/r37'));sys.path.insert(0,str(root/'lib/r41'));sys.path.insert(0,str(root/'lib/r42'));sys.path.insert(0,str(Path(__file__).parent))
from experimental_runtime import initialize
import startup_r20
startup_r20.initialize=lambda args:initialize(args.manifest)
script=root/('lib/r39a/feature_run.py'if a.mode=='features'else'lib/r36/verify_gate_v3.py');sys.path.insert(0,str(root/'lib/r39a'));sys.argv=[str(script),'--manifest',str(manifest),'--output',str(a.output)]+([]if a.mode=='features'else['--mode',a.mode]);runpy.run_path(str(script),run_name='__main__')
