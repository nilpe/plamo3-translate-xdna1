"""R83 fast-exit-gate (CPU only, no device): behaviour of DefaultRuntime.fast_exit / fast_exit_allowed on a fake idle runtime, and MUTANTS of default83.py that must all be caught.
Contract checked on the real code: (1) both streams are flushed BEFORE os._exit is reached; (2) the Environment.close guard: a runtime with an active request or uncertain completion is NOT exited (returns False, nothing stopped);
(3) the keep-warm helper and the priming worker are stopped (close called once each) before the exit; (4) the exit code passed to os._exit is the requested code (0); (5) fast_exit_allowed is False with --service and with turbo on and
with R77_EXIT=off, True for the plain one-shot; (6) nothing is exited when the runtime is already closed / has no env. Mutants (each must fail at least one check): exit before flush, no active/uncertain guard, exit in --service,
exit with turbo on, worker not stopped, keep-warm helper not stopped, non-zero exit code, exit even when R77_EXIT=off."""
import argparse,json,os,sys,threading,traceback,types
from pathlib import Path
import settings83 as s
s.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
rep=dict(result='FAIL')
class Exit(BaseException):pass
class Stream:
    def __init__(self,log,name):self.log,self.name=log,name
    def flush(self):self.log.append('flush:'+self.name)
    def write(self,x):pass
def scenario(mod,*,active=False,uncertain=False,keepwarm=True,closed=False,noenv=False,rc=0,envvar=None):
    log=[];D=mod.DefaultRuntime;app=D.__new__(D)
    app._closed=closed
    class RT:pass
    rt=RT();rt._lock=threading.Lock();rt._active=active;rt._completion_uncertain=uncertain
    app.env=None if noenv else types.SimpleNamespace(runtime=rt)
    class P:
        def close(self):log.append('primer.close')
    class K:
        def close(self):log.append('keepwarm.close')
    app.primer=P();app.keepwarm=K() if keepwarm else None
    old_out,old_err,old_exit,old_env=sys.stdout,sys.stderr,os._exit,os.environ.get('R77_EXIT')
    def fake_exit(code):log.append('exit:%d'%code);raise Exit()
    sys.stdout,sys.stderr,os._exit=Stream(log,'stdout'),Stream(log,'stderr'),fake_exit
    if envvar is None:os.environ.pop('R77_EXIT',None)
    else:os.environ['R77_EXIT']=envvar
    try:
        try:ret=app.fast_exit(rc);exited=False
        except Exit:ret=None;exited=True
    finally:
        sys.stdout,sys.stderr,os._exit=old_out,old_err,old_exit
        if old_env is None:os.environ.pop('R77_EXIT',None)
        else:os.environ['R77_EXIT']=old_env
    return dict(log=log,exited=exited,ret=ret)
def checks(mod):
    """returns the list of failed check names (empty = production behaviour)"""
    bad=[]
    r=scenario(mod)
    ex=[i for i,x in enumerate(r['log']) if x.startswith('exit:')]
    if not r['exited'] or not ex:bad.append('idle_runtime_must_exit')
    else:
        if 'flush:stdout' not in r['log'][:ex[0]] or 'flush:stderr' not in r['log'][:ex[0]]:bad.append('flush_before_exit')
        if 'primer.close' not in r['log'][:ex[0]]:bad.append('worker_stopped')
        if 'keepwarm.close' not in r['log'][:ex[0]]:bad.append('keepwarm_stopped')
        if r['log'][ex[0]]!='exit:0':bad.append('exit_code_zero')
        first_flush=min((i for i,x in enumerate(r['log']) if x.startswith('flush:')),default=999)
        if first_flush>ex[0]:bad.append('flush_before_exit')
    r=scenario(mod,rc=0,keepwarm=False)
    if not r['exited'] or 'keepwarm.close' in r['log']:bad.append('no_keepwarm_ok')
    for kw in(dict(active=True),dict(uncertain=True)):
        r=scenario(mod,**kw)
        if r['exited'] or r['ret'] is not False or any(x.startswith('exit:') or x.endswith('.close') for x in r['log']):bad.append('guard_active_or_uncertain')
    for kw in(dict(closed=True),dict(noenv=True)):
        r=scenario(mod,**kw)
        if r['exited']:bad.append('closed_or_no_env_must_not_exit')
    r=scenario(mod,envvar='off')
    if r['exited']:bad.append('env_off_must_not_exit')
    A=mod.DefaultRuntime.fast_exit_allowed
    old=os.environ.pop('R77_EXIT',None)
    try:
        if A(turbo_on=False,service=False) is not True:bad.append('plain_one_shot_allowed')
        if A(turbo_on=False,service=True):bad.append('service_must_not_exit')
        if A(turbo_on=True,service=False):bad.append('turbo_must_not_exit')
        os.environ['R77_EXIT']='off'
        if A(turbo_on=False,service=False):bad.append('env_off_not_allowed')
    finally:
        os.environ.pop('R77_EXIT',None)
        if old is not None:os.environ['R77_EXIT']=old
    return sorted(set(bad))
def load(text):
    mod=types.ModuleType('default83_mut');mod.__file__=str(Path(s.__file__).with_name('default83.py'))
    exec(compile(text,'default83_mut','exec'),mod.__dict__);return mod
try:
    import default83
    src=Path(default83.__file__).read_text(encoding='utf-8')
    base_bad=checks(default83);assert base_bad==[],('production code fails its own contract',base_bad)
    def rep_(old,new):
        assert src.count(old)==1,old;return src.replace(old,new,1)
    mutants={
     'exit_before_flush':rep_("            for stream in (sys.stdout, sys.stderr):\n                stream.flush()\n        except Exception:\n            return False\n        try:\n            if getattr(self, 'keepwarm'","        except Exception:\n            return False\n        try:\n            if getattr(self, 'keepwarm'").replace("            for stream in (sys.stdout, sys.stderr):\n                stream.flush()\n        except Exception:\n            return False\n        os._exit(rc)","        except Exception:\n            return False\n        os._exit(rc)"),
     'no_active_or_uncertain_guard':rep_("                if runtime._active or runtime._completion_uncertain:\n                    return False","                pass"),
     'exit_in_service':rep_("return (not turbo_on) and (not service) and","return (not turbo_on) and"),
     'exit_with_turbo_on':rep_("return (not turbo_on) and (not service) and","return (not service) and"),
     'worker_not_stopped':rep_("            if self.primer is not None:\n                self.primer.close()\n            for stream","            for stream"),
     'keepwarm_not_stopped':rep_("                self.keepwarm.close()\n            if self.primer","                pass\n            if self.primer"),
     'nonzero_exit_code':rep_("        os._exit(rc)","        os._exit(rc or 1)"),
     'ignores_R83_EXIT_off':rep_("os.environ.get('R77_EXIT', 'fast') != 'fast' or self._closed","self._closed"),
    }
    caught={}
    for name,text in mutants.items():
        assert text!=src,name
        caught[name]=checks(load(text))
        assert caught[name],('mutant NOT caught',name)
    # ---------------- in-process safety of the CLI entry (cli83.main / real_process_entry) ----------------
    import importlib,io,tempfile,time as _time
    cli_path=Path(default83.__file__).with_name('cli83.py');cli_src=cli_path.read_text(encoding='utf-8')
    def load_named(text,name,path):
        mod=types.ModuleType(name);mod.__file__=str(path);exec(compile(text,name,'exec'),mod.__dict__);return mod
    def cli_checks(text):
        """returns failed check names of a cli83 variant: main() without process_entry must never reach the fast exit; main(process_entry=True) must; real_process_entry() is False under runpy."""
        bad=[];calls=[]
        class Tok:
            def chat(self,m,**k):return [1,2,3]
            def decode(self,t):return 'x'
        class App:
            mode='w4-m2';requested_mode='w4-m2';session=types.SimpleNamespace(max_seq=2048);spec_ready_ms=1.0;keepwarm=None;startup={};fallbacks=[]
            def __init__(self,mode=None,keep_warm=None,allow_fallback=True):calls.append('init')   # H1: the CLI now passes allow_fallback
            def __enter__(self):return self
            def __exit__(self,*a):calls.append('close')
            def make_tokenizer(self):return Tok()
            def generate(self,prompt,n,target_language=None):return dict(tokens=[1,2],ttft_s=.1,wall_s=.2,stopped_eos=True,logical_length=5)
            @staticmethod
            def fast_exit_allowed(*,turbo_on,service):return True
            def fast_exit(self,rc=0):calls.append('fast_exit');raise Exit()
        fake=types.ModuleType('default83');fake.DefaultRuntime=App
        saved={k:sys.modules.get(k) for k in('default83','cli83_t')};argv=sys.argv;out=sys.stdout
        try:
            sys.modules['default83']=fake;mod=load_named(text,'cli83_t',cli_path)
            for pe,want in((None,False),(False,False),(True,True)):
                calls.clear();sys.argv=['cli83.py','--no-lock','--json','--','hello']   # H1: --no-lock keeps this static/behavioural contract independent of the NPU mutex (own gate: lock);sys.stdout=io.StringIO()
                try:
                    r=mod.main() if pe is None else mod.main(process_entry=pe)
                except Exit:r='exit'
                finally:sys.stdout=out
                if ('fast_exit' in calls)!=want:bad.append('main_process_entry=%s_fast_exit_must_be_%s'%(pe,want))
            # real_process_entry: True from a plain call, False when a runpy frame is on the stack
            _f=sys._getframe(0);own_runpy=False
            while _f is not None:
                own_runpy|=_f.f_globals.get('__name__')=='runpy';_f=_f.f_back
            if mod.real_process_entry() is not (not own_runpy):bad.append('real_entry_true_when_plain')
            tmp=Path(tempfile.mkdtemp(prefix='fxgate74_'));script=tmp/'probe.py';script.write_text('import cli83_t\nRESULT=cli83_t.real_process_entry()\n')
            sys.modules['cli83_t']=mod
            import runpy
            res=runpy.run_path(str(script),run_name='__main__')
            if res.get('RESULT') is not False:bad.append('real_entry_false_under_runpy')
        finally:
            sys.argv=argv;sys.stdout=out
            for k,v in saved.items():
                if v is None:sys.modules.pop(k,None)
                else:sys.modules[k]=v
        return sorted(set(bad))
    cb=cli_checks(cli_src);assert cb==[],('production cli83 fails its own contract',cb)
    def crep(old,new):
        assert cli_src.count(old)==1,old;return cli_src.replace(old,new,1)
    cli_mutants={
     'main_in_process_exits_the_caller':crep("if process_entry and app.fast_exit_allowed(","if app.fast_exit_allowed("),
     'process_entry_default_true':crep("def main(process_entry=False, argv=None):","def main(process_entry=True, argv=None):"),
     'real_entry_ignores_runpy_frames':crep("if f.f_globals.get('__name__')=='runpy' or f.f_code.co_filename.endswith('runpy.py'):return False","pass"),
     'real_entry_always_true':crep("    import threading\n    if threading.current_thread() is not threading.main_thread():return False\n    f=sys._getframe(1)","    return True\n    f=sys._getframe(1)"),
    }
    cli_caught={}
    for name,text in cli_mutants.items():
        cli_caught[name]=cli_checks(text);assert cli_caught[name],('cli mutant NOT caught',name)
    # ---------------- soak-check must read REAL post-soak records ----------------
    soak_path=Path(default83.__file__).with_name('fastexit83_soak.py');soak_src=soak_path.read_text(encoding='utf-8')
    def soak_checks(text):
        mod=load_named(text,'fxsoak_t',soak_path);bad=[]
        def case(name,mk,expect_ok):
            d=Path(tempfile.mkdtemp(prefix='fxsoak74_'));end=_time.time()-100
            f=d/'20260101-000000-000-startup-rel-0';f.mkdir();(f/'summary.json').write_text(json.dumps(dict(step='startup',package='rel',result='PASS',epoch='E')))
            os.utime(f/'summary.json',(end+50,end+50));mk(f,end)
            g=d/'20260101-000001-000-generate-rel-0';g.mkdir();(g/'summary.json').write_text(json.dumps(dict(step='generate',package='rel',result='PASS',epoch='E')));(g/'generate.json').write_text(json.dumps(dict(result='PASS')));os.utime(g/'summary.json',(end+50,end+50));os.utime(g/'generate.json',(end+60,end+60))
            got=mod.verify_followups(d,'E',end)
            if bool(got['startup'])!=expect_ok:bad.append(name)
        def good(f,end):(f/'startup.json').write_text(json.dumps(dict(result='PASS')));os.utime(f/'startup.json',(end+60,end+60))
        def missing(f,end):pass
        def stale(f,end):(f/'startup.json').write_text(json.dumps(dict(result='PASS')));os.utime(f/'startup.json',(end-60,end-60))
        def garbage(f,end):(f/'startup.json').write_text('{not json');os.utime(f/'startup.json',(end+60,end+60))
        def failed(f,end):(f/'startup.json').write_text(json.dumps(dict(result='FAIL')));os.utime(f/'startup.json',(end+60,end+60))
        case('good_record_accepted',good,True);case('missing_startup_json_accepted',missing,False);case('stale_startup_json_accepted',stale,False)
        case('unparseable_startup_json_accepted',garbage,False);case('failed_startup_json_accepted',failed,False)
        return sorted(set(bad))
    sb=soak_checks(soak_src);assert sb==[],('production soak-check fails its own contract',sb)
    def srep(old,new):
        assert soak_src.count(old)==1,old;return soak_src.replace(old,new,1)
    soak_mutants={
     'accepts_missing_or_stale_record':srep("            if not rec.is_file() or rec.stat().st_mtime<=end:continue\n","            pass\n"),
     'accepts_unparseable_or_failed_record':srep("            if d.get('result')!='PASS':continue\n","            pass\n").replace("            try:d=json.loads(rec.read_text(encoding='utf-8-sig'))\n            except Exception:continue\n","            d={}\n"),
     'accepts_stale_only':srep("if not rec.is_file() or rec.stat().st_mtime<=end:continue","if not rec.is_file():continue"),
    }
    soak_caught={}
    for name,text in soak_mutants.items():
        soak_caught[name]=soak_checks(text);assert soak_caught[name],('soak mutant NOT caught',name)
    rep=dict(result='PASS',production_failed_checks=base_bad,mutants_caught={k:v for k,v in caught.items()},cli_mutants_caught=cli_caught,soak_check_mutants_caught=soak_caught,scope='fake idle runtime (no device): contract of DefaultRuntime.fast_exit/fast_exit_allowed; end-to-end exit codes, leftovers and drift are in fast-exit-soak')
    a.output.write_text(json.dumps(rep,indent=1));print(json.dumps(rep),flush=True)
except BaseException:
    rep['error']=traceback.format_exc();a.output.write_text(json.dumps(rep,indent=1));traceback.print_exc();sys.exit(1)
