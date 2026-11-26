"""R83 keepwarm-gate: keepwarm83.py (helper + controller) and the default83.py bracket, with mutants that must ALL be caught.
usage: keepwarm83_gate.py --output result.json [--only-base] [--mutant NAME]
No NPU is used (the runtime bracket is exercised on a stub). Groups:
 mode       mode parsing (0|1|auto + aliases, invalid -> off + one stderr line), precedence CLI > env > default, DEFAULT_KEEP_WARM constant, CLI flags present
 power      ac mode with a fake power status (ac / dc / unknown), on ignores power, off never spawns
 affinity   decode-thread affinity pinned only during generate, restored after a normal generate and after an exception; flag set/cleared; helper mask == thread mask
 idle       spinner off when idle (helper CPU ~0 with the flag off, also right after a generate) and really spinning when on
 death      no helper left after: normal exit without close, uncaught exception, KeyboardInterrupt, hard kill of the parent (also while spinning), close(); hard cap; spin cap
 failure    helper missing / dies / spawn fails / affinity call fails / helper killed mid-run: no pin, no spin, one status note, feature disabled, generate unaffected
 cpu        CPU-choice rule table (core 0 and its SMT sibling avoided, process mask respected, efficiency class, first SMT thread of the core)
 default    runtime bracket: start only after spec-ready, never in off / serial modes, power check in ac mode, generate bracket restores in finally
 interplay  other processes (drafter-worker stand-in, a second keep-warm instance, a turbo-release-like waiter) are not disturbed and do not disturb it"""
import argparse,contextlib,ctypes,importlib.util,io,json,os,re,shutil,struct,subprocess,sys,tempfile,threading,time
HERE=os.path.dirname(os.path.abspath(__file__))
KW_SRC=open(os.path.join(HERE,'keepwarm83.py'),encoding='utf-8').read()
DEF_SRC=open(os.path.join(HERE,'default83.py'),encoding='utf-8').read()
SET_SRC=open(os.path.join(HERE,'settings83.py'),encoding='utf-8').read()
CLI_SRC=open(os.path.join(HERE,'cli83.py'),encoding='utf-8').read()
sys.path.insert(0,HERE)

class Ctx:
    def __init__(self,kw_src,def_src,groups,set_src=None):
        self.set_src=set_src or SET_SRC
        self.root=tempfile.mkdtemp(prefix='kw80gate_');self.moddir=os.path.join(self.root,'mod');os.makedirs(self.moddir)
        open(os.path.join(self.moddir,'keepwarm83.py'),'w',encoding='utf-8').write(kw_src)
        self.M=self._load('keepwarm83',os.path.join(self.moddir,'keepwarm83.py'))
        sys.modules['keepwarm83']=self.M
        self.D=None
        if True:
            p=os.path.join(self.moddir,'default83_ut.py');open(p,'w',encoding='utf-8').write(def_src);self.D=self._load('default83_ut',p)
        self.fails=[];self.k=self.M.k32();self.k2=ctypes.WinDLL('kernel32');self.k2.GetPriorityClass.argtypes=[ctypes.c_void_p];self.k2.GetCurrentProcess.restype=ctypes.c_void_p;self.groups=groups;self.live=[]
    def _load(self,name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
    def check(self,ok,msg):
        if not ok:self.fails.append(msg)
    def kw(self,mode='on',**kw):
        kw.setdefault('log',lambda line:self.logs.append(line));k=self.M.KeepWarm(mode,stats_dir=self.root,**kw);self.live.append(k);return k
    logs=[]
    def cleanup(self):
        for k in self.live:
            try:k.close()
            except Exception:pass
            try:
                if k.proc is not None and k.proc.poll() is None:k.proc.kill()
            except Exception:pass
        shutil.rmtree(self.root,ignore_errors=True)
    # -- process helpers
    def open_proc(self,pid):return self.k.OpenProcess(0x00100000|0x1000,False,pid)
    def running(self,h):return self.k.WaitForSingleObject(h,0)==0x102
    def wait_exit(self,h,timeout):
        t=time.time()+timeout
        while time.time()<t:
            if not self.running(h):return True
            time.sleep(0.05)
        return not self.running(h)
    def cpu_s(self,h):
        f=[(ctypes.c_uint64*1)() for _ in range(4)];self.k.GetProcessTimes(h,*[ctypes.byref(x) for x in f]);return (f[2][0]+f[3][0])/1e7
    def helper_pid(self,kw):return struct.unpack('<I',bytes(kw.mm[self.M.F_PID:self.M.F_PID+4]))[0]
    def started(self,mode='on',**kw):
        k=self.kw(mode,**kw);k.start(block=True);assert k.wait_ready(),'helper did not become ready';return k

def run_py(c,code,args=(),**kw):
    p=os.path.join(c.root,'host_%d.py'%len(os.listdir(c.root)));open(p,'w',encoding='utf-8').write(code)
    env=dict(os.environ);env['PYTHONPATH']=c.moddir
    return subprocess.Popen([sys.executable,p]+list(args),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,**kw)

# ------------------------------------------------------------------ groups
def g_mode(c):
    M=c.M;P=M.parse_mode
    for v,want in(('0','off'),('1','on'),('auto','ac'),('AUTO','ac'),(' 1 ','on'),('off','off'),('on','on'),('ac','ac'),('Off','off')):
        c.check(P(v)==(want,None),'parse_mode(%r) -> %r, want %r'%(v,P(v),want))
    for v in(None,'','  '):c.check(P(v)==(None,None),'parse_mode(%r) must be unset'%(v,))
    for v in('yes','2','true','-1','on!','no','enable','10'):
        m,n=P(v);c.check(m=='off' and n and '\n' not in n,'invalid value %r must give off + a one-line note, got %r %r'%(v,m,n))
    logs=[];log=logs.append;R=M.resolve_mode
    c.check(R(True,{'PLAMO3_KEEP_WARM':'0'},'off',log)=='on','CLI --keep-warm must win over env 0')
    c.check(R(False,{'PLAMO3_KEEP_WARM':'1'},'on',log)=='off','CLI --no-keep-warm must win over env 1 and default on')
    c.check(R(None,{'PLAMO3_KEEP_WARM':'1'},'off',log)=='on','env 1 must win over default off')
    c.check(R(None,{'PLAMO3_KEEP_WARM':'0'},'ac',log)=='off','env 0 must win over default ac')
    c.check(R(None,{'PLAMO3_KEEP_WARM':'auto'},'off',log)=='ac','env auto -> ac')
    c.check(R(None,{},'ac',log)=='ac' and R(None,{},'on',log)=='on' and R(None,{},'off',log)=='off','unset env -> the default constant')
    c.check(R(None,{'PLAMO3_KEEP_WARM':''},'ac',log)=='ac','empty env -> default')
    c.check(not logs,'no log line for valid values: %r'%logs)
    logs.clear();got=R(None,{'PLAMO3_KEEP_WARM':'banana'},'on',log)
    c.check(got=='off' and len(logs)==1 and 'banana' in logs[0],'invalid env must give off with exactly one note, got %r %r'%(got,logs))
    c.check(R(None,{},'bogus',log)=='off','invalid default -> off')
    m=re.search(r"^DEFAULT_KEEP_WARM\s*=\s*'(off|on|ac)'",c.set_src,re.M);c.check(bool(m),'settings83.DEFAULT_KEEP_WARM constant missing or not off|on|ac')
    c.check(bool(m) and m.group(1)=='off','DEFAULT_KEEP_WARM must be off (user decision: opt-in), got %r'%(m.group(1) if m else None))
    c.check(len(re.findall(r'DEFAULT_KEEP_WARM\s*=',c.set_src))==1,'DEFAULT_KEEP_WARM must be defined exactly once')
    c.check("'--keep-warm'" in CLI_SRC and "'--no-keep-warm'" in CLI_SRC and 'resolve_mode(a.keep_warm' in CLI_SRC and 'keep_warm=keep_warm' in CLI_SRC,'cli83 must expose --keep-warm/--no-keep-warm and pass the resolved mode to DefaultRuntime')

def g_power(c):
    M=c.M;state={'v':'ac'};M.power_status=lambda:state['v']
    k=c.started('ac');base=M.thread_mask()
    tok=k.enter();c.check(tok is not None and M.thread_mask()==1<<k.cpu,'ac mode on AC power must pin+spin');k.leave(tok);c.check(M.thread_mask()==base,'affinity restored (ac)')
    state['v']='dc';tok=k.enter();c.check(tok is None and M.thread_mask()==base and k.mm[M.F_FLAG]==0,'ac mode on battery must not pin or spin');k.leave(tok)
    c.check(k.skipped_dc==1,'skipped_dc counter')
    state['v']='unknown';tok=k.enter();c.check(tok is None and k.mm[M.F_FLAG]==0,'ac mode with unknown power status must not spin');k.leave(tok)
    state['v']='ac';tok=k.enter();c.check(tok is not None,'ac mode back on AC must spin again (checked at each generate)');k.leave(tok);k.close()
    c.logs.clear();k2=c.started('on');state['v']='dc';tok=k2.enter();c.check(tok is not None,'mode on ignores power status');k2.leave(tok);k2.close()
    k3=c.kw('off');k3.start();tok=k3.enter();c.check(tok is None and k3.proc is None and k3._spawn_thread is None,'mode off must never spawn or pin')
    # ac mode started on battery: no helper until the first generate on AC (spawned without blocking; that generate runs unpinned)
    state['v']='dc';k4=c.kw('ac');tok=k4.enter();c.check(tok is None and k4.proc is None,'ac on battery: no helper spawned');state['v']='ac'
    tok=k4.enter();c.check(tok is None,'first AC generate after a battery start only spawns the helper (non-blocking)');
    t=time.time()+5
    while time.time()<t and not (k4.mm is not None and k4.mm[M.F_READY]==1):time.sleep(0.02)
    tok=k4.enter();c.check(tok is not None,'next AC generate spins');k4.leave(tok);k4.close()

def stub_generate(c,kw,body,mode='w4-m2'):
    """DefaultRuntime.generate (unbound) on a stub that only has the attributes the bracket needs."""
    D=c.D;stub=type('S',(),{})();stub.keepwarm=kw;stub.mode=mode;stub._generate=lambda *a,**k:body();return D.DefaultRuntime.generate(stub,[1],1)

def g_affinity(c):
    M=c.M;k=c.started('on');base=M.thread_mask();seen={};pc0=c.k2.GetPriorityClass(c.k2.GetCurrentProcess())
    def body():
        seen['mask']=M.thread_mask();seen['flag']=k.mm[M.F_FLAG];seen['helper']=k.helper_cpu_mask();return 'ok'
    r=stub_generate(c,k,body)
    c.check(r=='ok','generate result passed through')
    c.check(seen['mask']==1<<k.cpu,'decode thread must be pinned to the keep-warm CPU during generate (got %x)'%seen['mask'])
    c.check(seen['flag']==1,'spinner flag must be on during generate')
    c.check(seen['helper']==seen['mask'],'helper pinned to the same logical CPU as the decode thread (helper %x vs thread %x)'%(seen['helper'],seen['mask']))
    c.check(M.thread_mask()==base,'thread affinity restored after a normal generate (%x != %x)'%(M.thread_mask(),base))
    c.check(k.mm[M.F_FLAG]==0,'spinner flag must be off after generate')
    c.check(c.k2.GetPriorityClass(c.k2.GetCurrentProcess())==pc0,'the decode process priority class must not change')
    class Boom(Exception):pass
    def bad():seen['mask2']=M.thread_mask();raise Boom()
    try:stub_generate(c,k,bad);c.check(False,'exception must propagate')
    except Boom:pass
    c.check(seen['mask2']==1<<k.cpu,'pinned during the failing generate')
    c.check(M.thread_mask()==base,'thread affinity restored after an exception (%x != %x)'%(M.thread_mask(),base))
    c.check(k.mm[M.F_FLAG]==0,'spinner flag must be off after an exception')
    def kb():raise KeyboardInterrupt()
    try:stub_generate(c,k,kb)
    except KeyboardInterrupt:pass
    c.check(M.thread_mask()==base and k.mm[M.F_FLAG]==0,'KeyboardInterrupt in generate restores affinity and flag')
    # serial mode and off: no pin at all
    seen.clear();stub_generate(c,k,lambda:seen.setdefault('m',M.thread_mask()),mode='w4-serial');c.check(seen['m']==base,'serial modes are not pinned')
    stub_generate(c,None,lambda:seen.setdefault('n',M.thread_mask()));c.check(seen['n']==base,'no controller -> no pin')
    k.close()

def g_idle(c):
    M=c.M;k=c.started('on');h=c.open_proc(c.helper_pid(k));time.sleep(0.3)
    c.check(c.k2.GetPriorityClass(h)==0x40,'helper must run in IDLE_PRIORITY_CLASS (got %#x)'%c.k2.GetPriorityClass(h))
    a=c.cpu_s(h);time.sleep(1.0);b=c.cpu_s(h);c.check(b-a<0.06,'idle helper (flag off) used %.3f s CPU in 1.0 s'%(b-a))
    tok=k.enter();a=c.cpu_s(h);time.sleep(1.0);b=c.cpu_s(h);c.check(b-a>0.35,'spinning helper used only %.3f s CPU in 1.0 s'%(b-a));c.check(M.k32().WaitForSingleObject(k.ev,0)==0,'go-event must be signalled while spinning');k.leave(tok);c.check(M.k32().WaitForSingleObject(k.ev,0)==0x102,'go-event must be reset after leave()')
    time.sleep(0.25);a=c.cpu_s(h);time.sleep(1.0);b=c.cpu_s(h);c.check(b-a<0.06,'helper must stop spinning after leave(): %.3f s CPU in 1.0 s'%(b-a))
    tok=k.enter();time.sleep(0.2);k.leave(tok);time.sleep(0.2);tok=k.enter();time.sleep(0.2);k.leave(tok);time.sleep(0.25)
    a=c.cpu_s(h);time.sleep(0.8);b=c.cpu_s(h);c.check(b-a<0.05,'repeated enter/leave windows leave the helper idle (%.3f s)'%(b-a))
    c.k.CloseHandle(h);k.close()

HOST_DEATH='''
import sys,os,time,json
import keepwarm83 as K
how,out=sys.argv[1],sys.argv[2]
kw=K.make('on',hard_cap=float(os.environ.get('GATE_HARD','600')),spin_cap=float(os.environ.get('GATE_SPIN','600')));kw.start(block=True)
assert kw.wait_ready()
tok=kw.enter() if os.environ.get('GATE_SPINNING','1')=='1' else None
import struct
real=struct.unpack('<I',bytes(kw.mm[K.F_PID:K.F_PID+4]))[0]
open(out,'w').write(json.dumps(dict(helper=real,popen=kw.proc.pid,pid=os.getpid())))
t=time.time()+30
while not os.path.exists(out+'.go') and time.time()<t:time.sleep(0.01)
if how=='kill':time.sleep(120)
elif how=='exit':sys.exit(0)
elif how=='raise':raise RuntimeError('boom')
elif how=='ctrlc':raise KeyboardInterrupt
elif how=='hardexit':os._exit(9)
elif how=='close':
    kw.leave(tok);kw.close();time.sleep(0.2);time.sleep(float(os.environ.get('GATE_HOLD','4')))
elif how=='hold':time.sleep(float(os.environ.get('GATE_HOLD','8')))
'''
def host(c,how,env=None,spinning=True):
    out=os.path.join(c.root,'h_%s_%d.json'%(how,len(os.listdir(c.root))));e=dict(os.environ)
    e.update(env or {});e['PYTHONPATH']=c.moddir;e['GATE_SPINNING']='1' if spinning else '0'
    hp=os.path.join(c.root,'host_death.py');open(hp,'w',encoding='utf-8').write(HOST_DEATH)
    p=subprocess.Popen([sys.executable,hp,how,out],env=e,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    t=time.time()+20
    while not os.path.exists(out) and time.time()<t and p.poll() is None:time.sleep(0.05)
    assert os.path.exists(out),'host did not start: '+(p.stderr.read() if p.poll() is not None else '')
    time.sleep(0.1);info=json.load(open(out));h=c.open_proc(info['helper']);assert h,'helper pid not openable';open(out+'.go','w').write('go');return p,info,h

def g_death(c):
    for how in('exit','raise','ctrlc','hardexit'):
        p,info,h=host(c,how);p.wait(30)
        c.check(c.wait_exit(h,4),'helper still alive 4 s after the parent ended via %r'%how);c.k.CloseHandle(h)
    p,info,h=host(c,'kill');c.check(c.running(h),'helper alive while the parent lives (spinning)');p.kill();p.wait(10)
    c.check(c.wait_exit(h,4),'helper still alive 4 s after the parent was killed hard while spinning');c.k.CloseHandle(h)
    p,info,h=host(c,'hold',spinning=False);time.sleep(0.3);p.kill();p.wait(10)
    c.check(c.wait_exit(h,4),'helper still alive 4 s after the parent was killed hard while idle');c.k.CloseHandle(h)
    p,info,h=host(c,'close',spinning=True,env=dict(GATE_HOLD='6'));c.check(c.wait_exit(h,2.5),'close() must stop the helper within 2.5 s while the parent lives');p.kill();p.wait(10);c.k.CloseHandle(h)
    p,info,h=host(c,'hold',env=dict(GATE_HARD='2',GATE_HOLD='8'),spinning=False);c.check(c.wait_exit(h,5.5),'hard cap (2 s) must end the helper while the parent still lives');p.kill();p.wait(10);c.k.CloseHandle(h)
    p,info,h=host(c,'hold',env=dict(GATE_SPIN='1',GATE_HOLD='7'),spinning=True);time.sleep(1.6);a=c.cpu_s(h);time.sleep(1.0);b=c.cpu_s(h)
    c.check(b-a<0.2,'spin cap (1 s) must stop a stuck flag from spinning (%.2f s CPU in the next 1.0 s)'%(b-a));p.kill();p.wait(10);c.k.CloseHandle(h)

def g_failure(c):
    M=c.M;base=M.thread_mask()
    def gen(k):
        seen={};r=stub_generate(c,k,lambda:seen.setdefault('m',M.thread_mask()) and 'ok')
        return r,seen['m']
    # helper script missing -> python exits at once
    c.logs.clear();k=c.kw('on',helper_path=os.path.join(c.root,'nope.py'),cpu=6);k.start(block=True);time.sleep(0.8)
    r,m=gen(k);c.check(r=='ok' and m==base,'missing helper: generate must proceed unpinned (mask %x)'%m);c.check(k.disabled and len(c.logs)==1 and 'not running' in c.logs[0],'missing helper: one status note and disabled (%r)'%c.logs)
    r,m=gen(k);c.check(len(c.logs)==1 and m==base,'a disabled feature must not note again or pin');c.check(k.mm[M.F_FLAG]==0,'no spin')
    # helper that exits at once
    c.logs.clear();p=os.path.join(c.root,'quit.py');open(p,'w').write('import sys;sys.exit(5)');k=c.kw('on',helper_path=p,cpu=6);k.start(block=True);time.sleep(0.6)
    r,m=gen(k);c.check(r=='ok' and m==base and k.disabled,'dying helper: unpinned + disabled')
    # python missing -> spawn fails
    c.logs.clear();k=c.kw('on',python=os.path.join(c.root,'no-python.exe'),cpu=6);k.start(block=True);r,m=gen(k);c.check(r=='ok' and m==base and k.disabled and len(c.logs)>=1,'spawn failure: unpinned + disabled + note (%r)'%c.logs)
    # no eligible CPU (not in the process mask)
    c.logs.clear();k=c.kw('on',cpu=63);k.start(block=True);r,m=gen(k);c.check(r=='ok' and m==base and k.disabled,'CPU outside the process mask: feature disabled')
    # SetThreadAffinityMask fails in enter
    c.logs.clear();k=c.started('on');orig=M.k32().SetThreadAffinityMask
    class K2:
        def __init__(s,k):s.k=k
        def __getattr__(s,n):return getattr(s.k,n)
        class _F:
            def __init__(s,real):s.real=real
            def __call__(s,h,m):return 0
        @property
        def SetThreadAffinityMask(s):return K2._F(None)
    real=M._K;M._K=K2(real)
    try:tok=k.enter()
    finally:M._K=real
    c.check(tok is None and k.disabled and M.thread_mask()==base and k.mm[M.F_FLAG]==0,'SetThreadAffinityMask failure: no pin, no spin, disabled')
    # helper killed mid-run
    c.logs.clear();k=c.started('on');tok=k.enter();c.check(tok is not None,'precondition');h=c.open_proc(c.helper_pid(k));ht=c.k.OpenProcess(0x1|0x00100000,False,c.helper_pid(k));c.k.TerminateProcess(ht,1);c.wait_exit(h,3);time.sleep(0.2)
    k.leave(tok);c.check(M.thread_mask()==base,'leave after the helper died still restores affinity')
    r,m=gen(k);c.check(r=='ok' and m==base and k.disabled,'helper killed mid-run: next generate runs unpinned and the feature is disabled (mask %x)'%m)
    n=len(c.logs);gen(k);c.check(len(c.logs)==n and n==1,'exactly one status note for a dead helper (got %d)'%len(c.logs));c.k.CloseHandle(h)
    # enter() internal error
    c.logs.clear();k=c.started('on');k.mm.close()
    try:tok=k.enter()
    except Exception:tok='RAISED'
    c.check(tok is None and M.thread_mask()==base,'internal error in enter(): returns None, no pin, never raises (%r)'%(tok,));k.mm=None

def g_cpu(c):
    M=c.M;C=M.choose_cpu;pairs=[(3<<(2*i),0) for i in range(8)]
    full=0xFFFF
    c.check(C(full,pairs,0)==2,'8C/16T, pick 0 -> first thread of core 1 (cpu 2), got %r'%C(full,pairs,0))
    c.check(C(full,pairs,2)==6,'8C/16T, pick 2 -> cpu 6, got %r'%C(full,pairs,2))
    c.check(C(full,pairs,-1)==14,'8C/16T, last eligible core -> cpu 14, got %r'%C(full,pairs,-1))
    c.check(C(full,pairs,99)==14,'pick clamps to the last core')
    c.check(C(0x3,pairs,0) is None and C(0x1,pairs,0) is None,'only core 0 available -> None')
    c.check(C(0xF,pairs,0)==2 and C(0xF,pairs,5)==2,'mask 0xF: core 0 (and its sibling) avoided -> cpu 2')
    c.check(C(0xFF00,pairs,0)==8,'process mask limited to cores 4..7 -> cpu 8, got %r'%C(0xFF00,pairs,0))
    c.check(C(0x20,pairs,0)==5,'single-CPU mask (5) -> cpu 5 (only thread of its core in the mask)')
    c.check(C(0x2,pairs,0) is None,'cpu 1 is the SMT sibling of cpu 0 -> None')
    c.check(C(0xFFFC,pairs,0)==2,'cpu 0/1 removed from the mask: cpu 2')
    hyb=[(3,0),(0xC,0),(0x30,1),(0xC0,1),(0x300,0),(0x400,1)]   # efficiency class 1 = performance
    c.check(C(0x7FF,hyb,0)==4,'hybrid: only the highest efficiency class is used -> cpu 4, got %r'%C(0x7FF,hyb,0))
    nosmt=[(1<<i,0) for i in range(6)]
    c.check(C(0x3F,nosmt,0)==1 and C(0x3F,nosmt,2)==3,'no SMT: cores of one thread; core 0 (cpu 0) avoided')
    cores=M.cpu_cores();pm=M.process_mask();cpu=C(pm,cores)
    c.check(cpu is not None and (pm>>cpu)&1 and cpu not in(0,1) and sum(1 for m,_ in cores if m&1 and m&(1<<cpu))==0,'live topology: chosen cpu %r is in the mask and not on core 0'%cpu)
    c.check(all(bin(m).count('1')<=2 for m,_ in cores) and len(cores)>=1,'live topology parsed (%d cores)'%len(cores))
    mine=[m for m,_ in cores if m&(1<<cpu)];c.check(mine and (mine[0]&-mine[0])==(1<<cpu),'chosen cpu is the first SMT thread of its core')

def g_default(c):
    D=c.D;M=c.M;events=[];orig_start=M.KeepWarm.start
    def fake_start(self,block=False):events.append('kw-start')
    M.KeepWarm.start=fake_start
    D.settings.STARTUP_SWITCH_INTERVAL_S=0
    def construct(self,mode,spec_manifest,bundle,log,primer_factory,initialize):
        events.append('construct-begin');self.mode='w4-m2' if mode in(None,'w4-m2') else mode;self.spec_ready_ms=1.0;self.log=lambda l:None;self.env=None;events.append('spec-ready')
    D.DefaultRuntime._construct=construct;D.DefaultRuntime.mode='w4-m2';D.DefaultRuntime.log=staticmethod(lambda l:None)
    state={'v':'ac'};M.power_status=lambda:state['v']
    for km,power,expect in(('on','ac',1),('on','dc',1),('ac','ac',1),('ac','dc',0),('off','ac',0)):
        events.clear();state['v']=power;app=D.DefaultRuntime('w4-m2',keep_warm=km)
        c.check(events.count('kw-start')==expect,'keep_warm=%s power=%s: start called %d times, want %d'%(km,power,events.count('kw-start'),expect))
        if expect:c.check(events.index('spec-ready')<events.index('kw-start'),'helper start must come after spec-ready (%r)'%events)
        c.check(getattr(app,'keepwarm',None) is None or app.keepwarm.mode==km,'controller mode')
    events.clear();app=D.DefaultRuntime('w4-serial',keep_warm='on');c.check('kw-start' not in events and app.keepwarm is None,'serial modes never start the helper')
    # env resolution inside the runtime when not given
    os.environ['PLAMO3_KEEP_WARM']='0';events.clear();D.DefaultRuntime('w4-m2');c.check('kw-start' not in events,'env 0 -> no helper');os.environ['PLAMO3_KEEP_WARM']='1';events.clear();D.DefaultRuntime('w4-m2');c.check('kw-start' in events,'env 1 -> helper');os.environ.pop('PLAMO3_KEEP_WARM',None)
    # the bracket is a try/finally around the whole request
    src=DEF_SRC if D.__file__.endswith('default83_ut.py') else ''
    c.check(re.search(r'token = kw\.enter\(\) if kw is not None else None\s+try:\s+return self\._generate\(.*?\)\s+finally:\s+if kw is not None:\s+kw\.leave\(token\)',open(D.__file__,encoding='utf-8').read(),re.S) is not None,'generate must bracket _generate with enter() before and leave() in finally')
    # ---- default policy (off): the timed path is R80's: no helper, no KeepWarm object, no power query, no affinity call, no extra thread
    import threading
    os.environ.pop('PLAMO3_KEEP_WARM',None);events.clear();cnt={'power':0,'aff':0,'inst':0}
    def pw():cnt['power']+=1;return 'ac'
    M.power_status=pw
    real_init=M.KeepWarm.__init__
    def counting_init(self,*a,**k):cnt['inst']+=1;real_init(self,*a,**k)
    M.KeepWarm.__init__=counting_init
    real_k=M.k32()
    class Prox:
        def __getattr__(s,n):return getattr(real_k,n)
        @property
        def SetThreadAffinityMask(s):
            def f(h,m):cnt['aff']+=1;return real_k.SetThreadAffinityMask(h,m)
            return f
    M._K=Prox();t0=threading.active_count()
    try:
        app=D.DefaultRuntime('w4-m2');app._generate=lambda *a,**k:'ok';r=app.generate([1],1)
        c.check(D.settings.DEFAULT_KEEP_WARM=='off','the real settings83 default must be off')
        c.check(r=='ok' and app.keepwarm is None,'default policy: no keep-warm controller (got %r)'%(app.keepwarm,))
        c.check('kw-start' not in events,'default policy: the helper must not be spawned')
        c.check(cnt['power']==0,'default policy: GetSystemPowerStatus must not be queried (%d calls)'%cnt['power'])
        c.check(cnt['aff']==0,'default policy: no thread-affinity call (%d calls)'%cnt['aff'])
        c.check(cnt['inst']==0,'default policy: no KeepWarm object (%d)'%cnt['inst'])
        c.check(threading.active_count()==t0,'default policy: no extra thread (%d -> %d)'%(t0,threading.active_count()))
        # explicit --no-keep-warm / env 0 are equally inert
        for kwarg in('off',):
            events.clear();cnt.update(power=0,aff=0,inst=0);app=D.DefaultRuntime('w4-m2',keep_warm=kwarg);app._generate=lambda *a,**k:'ok';app.generate([1],1)
            c.check(cnt==dict(power=0,aff=0,inst=0) and 'kw-start' not in events,'explicit off is inert %r'%cnt)
    finally:
        M._K=real_k;M.KeepWarm.__init__=real_init
    M.KeepWarm.start=orig_start

def g_interplay(c):
    M=c.M
    # drafter-worker stand-in: an unrelated child with its own handles; our helper must neither change its affinity nor die with/for it
    wk=subprocess.Popen([sys.executable,'-c','import sys,time;sys.stdin.read()'],stdin=subprocess.PIPE);wh=c.open_proc(wk.pid)
    k=c.started('on');tok=k.enter();time.sleep(0.2);pm=M.process_mask()
    wcpu0=c.k.GetProcessAffinityMask
    c.check(c.running(wh),'worker alive while keep-warm spins');k.leave(tok)
    # affinity of the worker process is untouched (process affinity mask == ours)
    pmw=ctypes.c_size_t();sm=ctypes.c_size_t();ok=c.k.GetProcessAffinityMask(wh,ctypes.byref(pmw),ctypes.byref(sm));c.check(ok and pmw.value==pm,'drafter worker process affinity changed (%x vs %x)'%(pmw.value,pm))
    hh=c.open_proc(c.helper_pid(k));wk.stdin.close();wk.wait(10);time.sleep(0.2);c.check(c.running(hh),'helper must survive the worker exiting')
    # worker kill does not disturb a spin window
    tok=k.enter();wk2=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(0.2);wk2.kill();wk2.wait(10);time.sleep(0.2);c.check(c.running(hh) and k.mm[M.F_FLAG]==1,'worker death must not stop an active spin window');k.leave(tok)
    # a turbo-release-like waiter (detached python waiting for our pid) is not disturbed by keep-warm and vice versa
    waiter=subprocess.Popen([sys.executable,'-c','import ctypes,os,sys,time;k=ctypes.WinDLL("kernel32");h=k.OpenProcess(0x00100000,0,int(sys.argv[1]));k.WaitForSingleObject(h,60000)',str(os.getpid())],creationflags=0x08000000)
    tok=k.enter();time.sleep(0.2);c.check(waiter.poll() is None,'release-like waiter alive during spin');k.leave(tok);k.close();c.check(c.wait_exit(hh,3),'helper exits on close with the waiter alive');c.check(waiter.poll() is None,'closing keep-warm must not end the turbo-release-like waiter');waiter.kill()
    # two instances (two processes) with different tags: killing one parent leaves the other helper spinning
    pa,ia,ha=host(c,'kill');pb,ib,hb=host(c,'kill');c.check(ia['helper']!=ib['helper'],'two instances need two helpers')
    pa.kill();pa.wait(10);c.check(c.wait_exit(ha,4),'helper A must die with parent A');c.check(c.running(hb),'helper B must survive the death of parent A')
    a=c.cpu_s(hb);time.sleep(0.8);b=c.cpu_s(hb);c.check(b-a>0.3,'helper B must keep spinning (%.2f s CPU)'%(b-a));pb.kill();pb.wait(10);c.check(c.wait_exit(hb,4),'helper B dies with parent B')
    c.k.CloseHandle(wh);c.k.CloseHandle(hh)

GROUPS=dict(mode=g_mode,power=g_power,affinity=g_affinity,idle=g_idle,death=g_death,failure=g_failure,cpu=g_cpu,default=g_default,interplay=g_interplay)

def run_suite(kw_src,def_src,groups,set_src=None):
    c=Ctx(kw_src,def_src,groups,set_src);saved=dict(os.environ);_k=ctypes.WinDLL('kernel32');_k.GetCurrentThread.restype=ctypes.c_void_p;_k.SetThreadAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t];_pm=c.M.process_mask()
    def reset():_k.SetThreadAffinityMask(_k.GetCurrentThread(),_pm)
    reset()
    try:
        for g in groups:
            try:GROUPS[g](c)
            except BaseException as e:
                import traceback;c.fails.append('group %s raised %s: %s'%(g,type(e).__name__,str(e)[:200]))
        return c.fails
    finally:
        reset();c.cleanup();os.environ.clear();os.environ.update(saved);sys.modules.pop('keepwarm83',None);sys.modules.pop('default83_ut',None)
        if 'keepwarm83' in sys.modules:pass

def sub(src,old,new,count=1):
    if isinstance(old,tuple):
        for o,n in zip(old,new):src=sub(src,o,n)
        return src
    assert src.count(old)>=1,'mutant anchor not found: '+old[:60]
    return src.replace(old,new,count)
# (name, target 'kw'|'def', old, new, groups that must catch it)
MUTANTS=[
 ('invalid_value_enables','kw',"if mode is None:return 'off','PLaMo: invalid keep-warm value %r (use 0|1|auto); keep-warm off'%(str(value),)","if mode is None:return 'on','PLaMo: invalid keep-warm value %r (use 0|1|auto); keep-warm off'%(str(value),)",('mode',)),
 ('zero_means_on','kw',"_ALIASES={'0':'off'","_ALIASES={'0':'on'",('mode',)),
 ('cli_off_ignored','kw',"    if cli is False:return 'off'\n","",('mode',)),
 ('env_beats_cli','kw',"    if cli is True:return 'on'\n    if cli is False:return 'off'\n    mode,note=parse_mode(environ.get(ENV_NAME))","    mode,note=parse_mode(environ.get(ENV_NAME))\n    if cli is True and mode is None:return 'on'",('mode',)),
 ('default_ignored','kw',"    d,_=parse_mode(default)\n    return d or 'off'","    return 'off'",('mode',)),
 ('ac_ignores_dc','kw',"if self.mode!='ac':return True","return True",('power',)),
 ('unknown_power_is_ac','kw',"if s=='ac':return True","if s in('ac','unknown'):return True",('power',)),
 ('on_consults_power','kw',"if self.mode!='ac':return True","if self.mode=='off':return True",('power',)),
 ('affinity_not_restored','kw',"k=k32();k.SetThreadAffinityMask(k.GetCurrentThread(),prev)","k=k32()",('affinity','power')),
 ('restore_wrong_mask','kw',"k=k32();k.SetThreadAffinityMask(k.GetCurrentThread(),prev)","k=k32();k.SetThreadAffinityMask(k.GetCurrentThread(),1<<self.cpu)",('affinity','power')),
 ('flag_left_on','kw',"        try:self.mm_flag_off()\n        finally:","        try:pass\n        finally:",('affinity','idle')),
 ('event_not_reset','kw',"            if self.ev:k32().ResetEvent(self.ev)\n","",('idle',)),
 ('leave_not_in_finally','def',"        try:\n            return self._generate(prompt, limit, emit, target_language=target_language)\n        finally:\n            if kw is not None:\n                kw.leave(token)","        r = self._generate(prompt, limit, emit, target_language=target_language)\n        if kw is not None:\n            kw.leave(token)\n        return r",('affinity',)),
 ('spinner_runs_on_other_core','kw',"if not k.SetThreadAffinityMask(k.GetCurrentThread(),1<<cpu):return 4","if not k.SetThreadAffinityMask(k.GetCurrentThread(),1<<(cpu^2)):return 4",('affinity',)),
 ('main_pinned_to_other_core','kw',"prev=k.SetThreadAffinityMask(k.GetCurrentThread(),1<<self.cpu)\n            if not prev","prev=k.SetThreadAffinityMask(k.GetCurrentThread(),1<<(self.cpu^2))\n            if not prev",('affinity','power')),
 ('helper_spins_when_idle','kw',"while mm[F_FLAG]==1:\n                    for _ in range(3000)","while mm[F_FLAG]!=2:\n                    for _ in range(3000)",('idle',)),
 ('helper_ignores_parent_death','kw',"if r==0:reason='parent-exit';break","if r==0:time.sleep(0.01);continue",('death',)),
 ('helper_not_idle_priority','kw',"k.SetPriorityClass(k.GetCurrentProcess(),0x40)","pass",('idle',)),
 ('hard_cap_ignored','kw',"left=hard_cap-(time.perf_counter()-start)","left=1e9",('death',)),
 ('spin_cap_ignored','kw',"if time.perf_counter()-t0>spin_cap:break\n","pass\n",('death',)),
 ('close_does_not_stop_helper','kw',"                self.mm[F_FLAG]=2\n                if self.ev:k32().SetEvent(self.ev)\n","                pass\n",('death',)),
 ('dead_helper_not_detected','kw',"            if not self.helper_alive():","            if False:",('failure',)),
 ('failure_not_disabling','kw',"        if disable:self.disabled=True;self.status='disabled'","        if disable:self.status='disabled'",('failure',)),
 ('enter_exceptions_propagate','kw',"        except Exception as e:\n            self.mm_flag_off();self.note('enter failed (%s: %s); running without it'%(type(e).__name__,e),True);return None","        except ZeroDivisionError as e:\n            return None",('failure',)),
 ('cpu_rule_allows_core0','kw',"if not m or (mask&1):continue","if not m:continue",('cpu',)),
 ('cpu_rule_picks_sibling','kw',"return (m&-m).bit_length()-1","return m.bit_length()-1",('cpu',)),
 ('cpu_rule_ignores_process_mask','kw',"m=mask&proc_mask\n        if not m","m=mask\n        if not m",('cpu',)),
 ('cpu_rule_ignores_efficiency','kw',"top=max(e for e,_ in elig);elig=sorted(m for e,m in elig if e==top)","elig=sorted(m for e,m in elig)",('cpu',)),
 ('helper_starts_before_spec_ready','def',("        self._init_keepwarm(keep_warm)\n        self._init_hybrid()\n\n    def _init_hybrid","        try:\n            self._construct(mode, spec_manifest, bundle, log, primer_factory, initialize)"),("        pass\n        self._init_hybrid()\n\n    def _init_hybrid","        self._init_keepwarm(keep_warm)\n        try:\n            self._construct(mode, spec_manifest, bundle, log, primer_factory, initialize)"),('default',)),
 ('serial_mode_starts_helper','def',"if mode == 'off' or self.mode != 'w4-m2':","if mode == 'off':",('default',)),
 ('default_off_queries_power','def',"                return  # off: nothing is imported, started, queried or pinned","                __import__('keepwarm83').power_status();return  # off",('default',)),
 ('default_off_starts_helper','def',"                return  # off: nothing is imported, started, queried or pinned","                __import__('keepwarm83').make('on',log=self.log).start();return  # off",('default',)),
 ('default_off_creates_controller','def',"                return  # off: nothing is imported, started, queried or pinned","                self.keepwarm=__import__('keepwarm83').make('on',log=self.log);return  # off",('default',)),
 ('default_off_spawns_helper','def',("if mode == 'off' or self.mode != 'w4-m2':","if mode == 'on' or keepwarm83.power_status() == 'ac':"),("if self.mode != 'w4-m2':","if True:"),('default',)),
 ('default_off_makes_affinity_call','def',"token = kw.enter() if kw is not None else None","token = kw.enter() if kw is not None else (__import__('keepwarm83').thread_mask() and None)",('default',)),
 ('default_off_extra_thread','def',"                return  # off: nothing is imported, started, queried or pinned","                __import__('threading').Thread(target=lambda: __import__('time').sleep(5), daemon=True).start();return  # off",('default',)),
 ('default_constant_not_off','set',"DEFAULT_KEEP_WARM='off'","DEFAULT_KEEP_WARM='ac'",('mode',)),
 ('ac_mode_skips_power_check','def',"if mode == 'on' or keepwarm83.power_status() == 'ac':","if True:",('default',)),
 ('parent_priority_changed','kw',"            k=k32();prev=k.SetThreadAffinityMask(k.GetCurrentThread(),1<<self.cpu)","            k=k32();prev=k.SetThreadAffinityMask(k.GetCurrentThread(),1<<self.cpu);k.SetPriorityClass(k.GetCurrentProcess(),0x40)",('affinity',)),
]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--only-base',action='store_true');ap.add_argument('--mutant');ap.add_argument('--output-dir');a=ap.parse_args()
    rep=dict(result='FAIL');t0=time.time()
    try:
        if a.mutant:
            m=next(x for x in MUTANTS if x[0]==a.mutant);ks,ds,ss=KW_SRC,DEF_SRC,SET_SRC
            if m[1]=='kw':ks=sub(KW_SRC,m[2],m[3])
            elif m[1]=='set':ss=sub(SET_SRC,m[2],m[3])
            else:ds=sub(DEF_SRC,m[2],m[3])
            f=run_suite(ks,ds,m[4],ss);print(json.dumps(dict(mutant=m[0],caught=bool(f),failures=f[:5])));return 0 if f else 1
        base=run_suite(KW_SRC,DEF_SRC,list(GROUPS));print('base suite: %d failures (%.0f s)'%(len(base),time.time()-t0),flush=True)
        for f in base:print('  FAIL',f,flush=True)
        rep=dict(groups=list(GROUPS),base_failures=base,mutants=[])
        if not a.only_base:
            for name,target,old,new,groups in MUTANTS:
                ks,ds,ss=KW_SRC,DEF_SRC,SET_SRC
                if target=='kw':ks=sub(KW_SRC,old,new)
                elif target=='set':ss=sub(SET_SRC,old,new)
                else:ds=sub(DEF_SRC,old,new)
                at=time.time();f=run_suite(ks,ds,list(groups),ss);caught=bool(f);rep['mutants'].append(dict(name=name,caught=caught,groups=groups,first_failure=f[0] if f else None,seconds=round(time.time()-at,1)))
                print('mutant %-34s %s  (%s) %.0fs'%(name,'CAUGHT' if caught else 'SURVIVED',(f[0][:90] if f else ''),time.time()-at),flush=True)
        rep['all_mutants_caught']=all(m['caught'] for m in rep['mutants']) if rep['mutants'] else None
        ok=not base and (a.only_base or rep['all_mutants_caught'])
        rep['result']='PASS' if ok else 'FAIL';rep['mutant_count']=len(rep['mutants']);rep['seconds']=round(time.time()-t0,1)
    except BaseException:
        import traceback;rep['error']=traceback.format_exc();print(rep['error'],flush=True)
    open(a.output,'w').write(json.dumps(rep,indent=2));print(rep['result'],flush=True);return 0 if rep['result']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
