"""Self-test of gate_index_rel.distinct_processes (function extracted by ast from gate_index_rel.py; the script itself needs a frozen epoch folder)."""
import ast,sys
from pathlib import Path
src=Path(__file__).with_name('gate_index_rel.py').read_text(encoding='utf-8')
fn=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='distinct_processes'][0]
ns={};exec(compile(ast.Module([fn],[]),'gate_index_rel','exec'),ns);f=ns['distinct_processes']
T0=1_700_000_000_000.0
cases=[
 ('all distinct pids, with times',[(i,T0+i*5000) for i in range(12)],True),
 ('all distinct pids, legacy (no times)',[(i,None) for i in range(12)],True),
 ('R83 incident: same pid 2.7 s apart, creation times known',[(385376,T0)]+[(i,T0+i*5000) for i in range(1,11)]+[(385376,T0+2700+60000)],True),
 ('R83 incident without creation times (frozen old records)',[(385376,None),(385376,None)]+[(i,None) for i in range(10)],False),
 ('same process counted twice (identical creation time)',[(7,T0),(7,T0)]+[(i,T0+i*5000) for i in range(10,20)],False),
 ('same process twice, tiny jitter 3 ms',[(7,T0),(7,T0+3.0)],False),
 ('pid reuse but one record lacks the time',[(7,T0),(7,None)],False),
 ('three records of one pid, all far apart',[(7,T0),(7,T0+4000),(7,T0+9000)],True),
 ('three records, two of them 50 ms apart',[(7,T0),(7,T0+4000),(7,T0+4050)],False),
]
bad=0
for name,rec,want in cases:
    got=f(rec);ok=got==want;bad+=not ok;print('PASS' if ok else 'FAIL',name,'->',got)
print('RESULT','PASS' if bad==0 else 'FAIL',len(cases),'cases');sys.exit(1 if bad else 0)
