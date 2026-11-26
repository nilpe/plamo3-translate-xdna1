"""Four real candidate CLI translations plus mixed-language service, serial IDs."""
import json,os,subprocess,sys,time
from pathlib import Path
SRC=((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'IRON/experiments/plamo3_r83/source');sys.path.insert(0,str(SRC));import settings83 as s;s.paths()
import startup_r20
m=json.loads(s.W4_MANIFEST.read_text())['payload'];rows=[]
wanted=[]
for i in range(4):
 tokens=[]
 for token in m['references']['prompts'][i]['actual_tokens']:
  tokens.append(token)
  if token in m['model']['eos']:break
 wanted.append(tokens)
# A cold import must survive unavailable configured manifest reads. Exercise
# both explicit serial and speculative-init fallback in isolated processes.
for mode in ('w4-serial','w4-m2'):
 cold="""import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
original=Path.read_bytes
def fault(path):
 if path.name=='PLaMo-manifest.json' and 'PLaMo-D20-M2'in path.parts:raise FileNotFoundError('fault gate: configured manifest unavailable')
 return original(path)
Path.read_bytes=fault
import cli83
from default83 import DefaultRuntime
logs=[]
with DefaultRuntime(sys.argv[2],log=logs.append)as app:
 assert app.mode=='w4-serial'
 prompt=app.session.model['prompts'][0]
 got=app.generate(prompt,256,target_language='en')
 assert len(logs)==(1 if sys.argv[2]=='w4-m2'else 0)
 print(json.dumps(dict(tokens=got['tokens'],mode=app.mode,logs=logs)))
"""
 proc=subprocess.run([sys.executable,'-c',cold,str(SRC),mode],capture_output=True,text=True,encoding='utf-8',timeout=60);assert proc.returncode==0,(mode,proc.stdout,proc.stderr);got=json.loads(proc.stdout);assert got['tokens']==wanted[0];rows.append(dict(id='cold-configured-manifest-'+mode,result='PASS',response=got))
for i,(text,source,target)in enumerate(startup_r20.PROMPTS):
 command=[sys.executable,str(SRC/'cli83.py'),text,'--source-lang',source,'--target-lang',target,'--json']
 proc=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=60);assert proc.returncode==0,(proc.stdout,proc.stderr);got=json.loads(proc.stdout);assert got['mode']=='w4-m2'and got['tokens']==wanted[i];rows.append(dict(id=i,result='PASS',target_language=target,command=command,spec_ready_ms=got['spec_ready_ms'],stderr=proc.stderr))
requests='\n'.join(json.dumps(dict(id=str(i),text=text,source_lang=source,target_lang=target),ensure_ascii=False)for i,(text,source,target)in enumerate(startup_r20.PROMPTS))+'\n'
proc=subprocess.run([sys.executable,str(SRC/'cli83.py'),'--service'],input=requests,capture_output=True,text=True,encoding='utf-8',timeout=60);assert proc.returncode==0,(proc.stdout,proc.stderr);got=[json.loads(l)for l in proc.stdout.splitlines()];assert len(got)==5 and got[0]['service_ready'] and all(got[i+1]['tokens']==wanted[i]and got[i+1]['mode']=='w4-m2'for i in range(4));rows.append(dict(id='mixed-service',result='PASS',target_languages=['en','en','ja','ja'],response=got,stderr=proc.stderr))
((Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir())))/'state/gates/cli.json').write_text(json.dumps(dict(result='PASS',cases=len(rows),rows=rows,source_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),scope='isolated frozen R83 CLI and real tokenizer; active public entry remains the R49 default'),indent=2));print(json.dumps(dict(result='PASS',cases=len(rows))))
