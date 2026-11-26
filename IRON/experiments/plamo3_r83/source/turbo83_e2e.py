"""R83 turbo end-to-end check (switch v2: background acquire, startup barrier, detached release) through the real cli83 entry (NPU job; run under the runner's NPU lock).
A. switch OFF + fake xrt-smi: zero xrt-smi calls, tokens T0.
B. --turbo + fake xrt-smi: examine / configure turbo / examine ... configure <previous> / examine, marker removed, tokens == T0.
C. PLAMO3_NPU_TURBO=1 + fake xrt-smi: same as B (env route).
D. REAL xrt-smi: examine before, cli83 --turbo, examine after: the mode must be the one before (Default expected); raw examine output logged; tokens == T0.
usage: turbo83_e2e.py --output out.json"""
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
import turbo83 as T
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
TEXT='今日はとても良い天気ですね。散歩に行きましょう。'
def cli(extra,env):
    t=time.time();r=subprocess.run([sys.executable,str(HERE/'cli83.py'),'--json']+extra+['--',TEXT],env=env,capture_output=True,text=True,encoding='utf-8',timeout=240)
    out=None
    for line in r.stdout.splitlines():
        if line.startswith('{'):out=json.loads(line)
    return dict(rc=r.returncode,seconds=round(time.time()-t,2),turbo=out and out.get('turbo'),tokens=out and out['tokens'],mode=out and out['mode'],spec_ready_ms=out and out['spec_ready_ms'],stderr_tail=r.stderr.strip().splitlines()[-6:])
def settle(state_dir=None,timeout=40):
    """the release runs in a detached helper after cli83 exits: wait until the marker is gone"""
    path=os.path.join(state_dir,'turbo-state.json') if state_dir else T.state_path()
    t=time.time()+timeout
    while os.path.exists(path) and time.time()<t:time.sleep(0.1)
    time.sleep(0.3)
def raw_examine():
    r=subprocess.run(T.xrt_smi_command()+['examine','-r','platform'],capture_output=True,text=True,encoding='utf-8',errors='replace');return r.stdout
res=dict(result='FAIL');fails=[]
def check(ok,msg):
    if not ok:fails.append(msg)
base=dict(os.environ)
for k in('PLAMO3_NPU_TURBO','PLAMO3_TURBO_TEST','PLAMO3_XRT_SMI','PLAMO3_TURBO_STATE_DIR','FAKE_XRT_DIR','FAKE_XRT_MARKER'):base.pop(k,None)
tmp=tempfile.mkdtemp(prefix='turbo83e2e-')
def fake_env(name,mode='default'):
    d=Path(tmp)/name;(d/'fake').mkdir(parents=True);(d/'mode.txt').parent.mkdir(exist_ok=True);(d/'fake'/'mode.txt').write_text(mode)
    e=dict(base);e.update(PLAMO3_TURBO_TEST='1',PLAMO3_XRT_SMI=str(HERE/'fake_xrt_smi83.py'),PLAMO3_TURBO_STATE_DIR=str(d/'state'),FAKE_XRT_DIR=str(d/'fake'),FAKE_XRT_MARKER=str(d/'state'/'turbo-state.json'));return e,d
def calls(d):
    try:return [l.split(' ',2)[2].rsplit(' marker=',1)[0] for l in (d/'fake'/'calls.log').read_text().splitlines()]
    except OSError:return []
e,d=fake_env('off');A=cli([],e);settle(str(d/'state'));res['A_off_fake']=dict(A,calls=calls(d));check(A['rc']==0 and A['tokens'],'A: cli failed');check(calls(d)==[],'A: xrt-smi called with the switch off: %r'%calls(d))
T0=A['tokens']
for label,extra,envs in(('B_turbo_flag_fake',['--turbo'],{}),('C_turbo_env_fake',[],{'PLAMO3_NPU_TURBO':'1'})):
    e,d=fake_env(label);e.update(envs);B=cli(extra,e);settle(str(d/'state'));c=calls(d);res[label]=dict(B,calls=c,mode_after=(d/'fake'/'mode.txt').read_text(),marker_left=(d/'state'/'turbo-state.json').exists())
    check(B['rc']==0 and B['tokens']==T0,label+': rc/tokens');check(c==['examine -r platform','configure --pmode turbo','examine -r platform','examine -r platform','configure --pmode default','examine -r platform'],label+': calls %r'%c)
    check(res[label]['mode_after']=='default' and not res[label]['marker_left'],label+': not restored')
# E. barrier engaged in the real runtime: slow fake xrt-smi (400 ms per call), the runtime must wait before device_open; tokens identical
e,d=fake_env('slow');e['FAKE_XRT_DELAY_MS']='400';E=cli(['--turbo'],e);settle(str(d/'state'),60);c=calls(d)
res['E_barrier_fake']=dict(E,calls=c,mode_after=(d/'fake'/'mode.txt').read_text(),marker_left=(d/'state'/'turbo-state.json').exists())
check(E['rc']==0 and E['tokens']==T0,'E: rc/tokens');check(res['E_barrier_fake']['mode_after']=='default' and not res['E_barrier_fake']['marker_left'],'E: not restored')
check(E.get('turbo') is not None and E['turbo'].get('barrier_wait_ms',0)>=100 and E['turbo'].get('switched_ms',0)>=700,'E: the startup barrier did not wait for the slow acquire: %r'%E.get('turbo'))
# D real device
e=dict(base);before_raw=raw_examine();before=T.read_mode();D=cli(['--turbo'],e);settle();after_raw=raw_examine();after=T.read_mode()
res['D_real']=dict(D,examine_before=before_raw,examine_after=after_raw,mode_before=before,mode_after=after,marker_left=os.path.exists(T.state_path()))
check(before is not None and after==before,'D: real pmode not restored (%r -> %r)'%(before,after));check(D['rc']==0 and D['tokens']==T0,'D: rc/tokens');check(not os.path.exists(T.state_path()),'D: marker left')
res['failures']=fails;res['result']='PASS' if not fails else 'FAIL'
a.output.write_text(json.dumps(res,indent=1,ensure_ascii=False));print(json.dumps(dict(result=res['result'],failures=fails,before=before,after=after)),flush=True);sys.exit(0 if not fails else 1)
