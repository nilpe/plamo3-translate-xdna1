"""R83 keep-warm switch: an IDLE-priority spinner helper process pinned to the SAME logical CPU as the decode thread (R70).

Why: in the M2 cycle the host steps (selection, draft, refresh) run right after a ~20 ms blocking verify wait; during that wait the decode thread's core drops
into a C-state and exits it slowly (R70: -1.5 ms per M2 cycle, tokens identical). An always-runnable idle-priority thread on that core keeps it out of deep idle and
is pre-empted at once when the decode thread wakes. Another core, an SMT sibling or pinning alone does not help (R70).

Scope: process-scoped only. ctypes into kernel32; no system-wide setting (no powercfg, no power plan, no timer resolution, no pmode); no new executable.
No arithmetic is touched. Any failure of the helper disables the feature for the rest of the process and generation proceeds exactly as R80 does (no pin, no spin).

Modes: off | on | ac (on only while GetSystemPowerStatus reports AC power, checked at each generate).
Selection: CLI --keep-warm / --no-keep-warm, env PLAMO3_KEEP_WARM=0|1|auto (aliases off|on|ac), default = settings83.DEFAULT_KEEP_WARM. Invalid value -> off (+ one stderr line).

Helper usage (internal): python -I -S keepwarm83.py --helper TAG PARENT_PID CPU HARD_CAP_S SPIN_CAP_S STATS_PATH
"""
import os,sys,time,struct,threading

MODES=('off','on','ac')
ENV_NAME='PLAMO3_KEEP_WARM'
_ALIASES={'0':'off','off':'off','no':None,'1':'on','on':'on','auto':'ac','ac':'ac'}
HARD_CAP_S=6*3600.0          # helper lifetime cap (it normally dies with its parent long before)
SPIN_CAP_S=600.0             # one continuous spin window cap (a flag left on cannot spin forever)
READY_WAIT_S=2.0

def _log_default(line):
    print(line,file=sys.stderr,flush=True)

def parse_mode(value):
    """-> (mode, note). value None/'' = unset (caller applies the default). Invalid -> ('off', note)."""
    if value is None:return None,None
    v=str(value).strip().lower()
    if v=='':return None,None
    mode=_ALIASES.get(v)
    if mode is None:return 'off','PLaMo: invalid keep-warm value %r (use 0|1|auto); keep-warm off'%(str(value),)
    return mode,None

def resolve_mode(cli=None,environ=None,default='off',log=None):
    """Precedence: CLI (True = on, False = off, None = not given) > env PLAMO3_KEEP_WARM > default."""
    log=log or _log_default;environ=os.environ if environ is None else environ
    if cli is True:return 'on'
    if cli is False:return 'off'
    mode,note=parse_mode(environ.get(ENV_NAME))
    if note:log(note)
    if mode is not None:return mode
    d,_=parse_mode(default)
    return d or 'off'

# ---------------------------------------------------------------- kernel32 (ctypes, lazily)
_K=None
def k32():
    global _K
    if _K is None:
        import ctypes
        from ctypes import wintypes as w
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=w.HANDLE;k.GetCurrentThread.restype=w.HANDLE
        k.SetThreadAffinityMask.argtypes=[w.HANDLE,ctypes.c_size_t];k.SetThreadAffinityMask.restype=ctypes.c_size_t
        k.GetProcessAffinityMask.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];k.GetProcessAffinityMask.restype=w.BOOL
        k.SetPriorityClass.argtypes=[w.HANDLE,w.DWORD];k.SetPriorityClass.restype=w.BOOL
        k.SetThreadPriority.argtypes=[w.HANDLE,ctypes.c_int];k.SetThreadPriority.restype=w.BOOL
        k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
        k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
        k.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];k.WaitForSingleObject.restype=w.DWORD
        k.WaitForMultipleObjects.argtypes=[w.DWORD,ctypes.POINTER(w.HANDLE),w.BOOL,w.DWORD];k.WaitForMultipleObjects.restype=w.DWORD
        k.CreateEventW.argtypes=[ctypes.c_void_p,w.BOOL,w.BOOL,w.LPCWSTR];k.CreateEventW.restype=w.HANDLE
        k.SetEvent.argtypes=[w.HANDLE];k.SetEvent.restype=w.BOOL;k.ResetEvent.argtypes=[w.HANDLE];k.ResetEvent.restype=w.BOOL
        k.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.c_void_p]*4;k.GetProcessTimes.restype=w.BOOL
        k.GetLogicalProcessorInformationEx.argtypes=[w.DWORD,ctypes.c_void_p,ctypes.POINTER(w.DWORD)];k.GetLogicalProcessorInformationEx.restype=w.BOOL
        k.GetSystemPowerStatus.argtypes=[ctypes.c_void_p];k.GetSystemPowerStatus.restype=w.BOOL
        k.TerminateProcess.argtypes=[w.HANDLE,w.UINT];k.TerminateProcess.restype=w.BOOL
        _K=k
    return _K

def thread_mask():
    """The calling thread's affinity mask (read by a set-and-restore with the process mask)."""
    k=k32();pm=process_mask();prev=k.SetThreadAffinityMask(k.GetCurrentThread(),pm)
    if prev:k.SetThreadAffinityMask(k.GetCurrentThread(),prev)
    return prev

def process_mask():
    import ctypes
    k=k32();p=ctypes.c_size_t();s=ctypes.c_size_t()
    if not k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(p),ctypes.byref(s)):raise OSError('GetProcessAffinityMask failed')
    return p.value

def cpu_cores():
    """Physical cores of processor group 0 from GetLogicalProcessorInformationEx(RelationProcessorCore): [(mask, efficiency_class), ...] in OS order."""
    import ctypes
    from ctypes import wintypes as w
    k=k32();n=w.DWORD(0);k.GetLogicalProcessorInformationEx(0,None,ctypes.byref(n))
    if n.value==0:raise OSError('GetLogicalProcessorInformationEx size query failed')
    buf=(ctypes.c_ubyte*n.value)()
    if not k.GetLogicalProcessorInformationEx(0,buf,ctypes.byref(n)):raise OSError('GetLogicalProcessorInformationEx failed')
    raw=bytes(buf)[:n.value];off=0;cores=[]
    while off+8<=len(raw):
        rel,size=struct.unpack_from('<II',raw,off)
        if size<8:break
        if rel==0:  # RelationProcessorCore: Flags(1) EfficiencyClass(1) Reserved(20) GroupCount(2) GROUP_AFFINITY[]{Mask(8) Group(2) Reserved(6)}
            eff=raw[off+9];groups=struct.unpack_from('<H',raw,off+30)[0]
            for g in range(groups):
                mask,group=struct.unpack_from('<QH',raw,off+32+16*g)
                if group==0:cores.append((mask,eff))
        off+=size
    return cores

def choose_cpu(proc_mask,cores,pick=None):
    """CPU-choice rule (pure function; measured in ROUND80.md).
    1. only physical cores that intersect the process affinity mask and belong to processor group 0;
    2. drop the core that holds logical CPU 0 (interrupts/DPCs) and therefore its SMT sibling;
    3. keep only the highest EfficiencyClass present (performance cores on hybrid parts; all equal on this Zen4);
    4. order the remaining cores by their lowest logical CPU and take the core at index PICK (default RULE_PICK) of that list, clamped to the list;
    5. within that core use its lowest logical CPU that is in the process mask (the other SMT thread of the core is left alone).
    Returns the logical CPU index or None (no eligible core -> the feature stays off)."""
    pick=RULE_PICK if pick is None else pick
    elig=[]
    for mask,eff in cores:
        m=mask&proc_mask
        if not m or (mask&1):continue
        elig.append((eff,m))
    if not elig:return None
    top=max(e for e,_ in elig);elig=sorted(m for e,m in elig if e==top)
    idx=min(max(pick,0),len(elig)-1) if pick>=0 else max(len(elig)+pick,0)
    m=elig[idx]
    return (m&-m).bit_length()-1
RULE_PICK=2   # eligible core index (0-based, after dropping core 0); set from the R83 measurement (ROUND80.md)

# ---------------------------------------------------------------- power
def power_status():
    """'ac' | 'dc' | 'unknown' (GetSYSTEM_POWER_STATUS.ACLineStatus 1 / 0 / 255). The gate replaces this function with a fake."""
    import ctypes
    b=(ctypes.c_ubyte*12)()
    if not k32().GetSystemPowerStatus(b):return 'unknown'
    return {1:'ac',0:'dc'}.get(b[0],'unknown')

# ---------------------------------------------------------------- shared state layout (named mapping, 32 bytes)
# [0] flag: 0 idle, 1 spin, 2 stop   [1] ready (helper set priority+affinity)   [4:8] helper pid   [8:16] helper thread mask (read back)   [16:24] spin windows
F_FLAG,F_READY,F_PID,F_MASK,F_WIN=0,1,4,8,16

def _process_cpu_s(k):
    import ctypes
    f=[(ctypes.c_uint64*1)() for _ in range(4)]
    k.GetProcessTimes(k.GetCurrentProcess(),*[ctypes.byref(x) for x in f])
    return dict(kernel_s=f[2][0]/1e7,user_s=f[3][0]/1e7)

def helper_main(tag,parent_pid,cpu,hard_cap,spin_cap,stats_path):
    import mmap,json
    k=k32();start=time.perf_counter()
    ph=k.OpenProcess(0x00100000,False,parent_pid)      # SYNCHRONIZE: a dead/unopenable parent means exit at once
    if not ph or k.WaitForSingleObject(ph,0)==0:return 3
    k.SetPriorityClass(k.GetCurrentProcess(),0x40)     # IDLE_PRIORITY_CLASS
    k.SetThreadPriority(k.GetCurrentThread(),-15)      # THREAD_PRIORITY_IDLE
    mm=mmap.mmap(-1,32,tagname=tag);ev=k.CreateEventW(None,True,False,tag+'_go')
    if cpu>=0:
        if not k.SetThreadAffinityMask(k.GetCurrentThread(),1<<cpu):return 4
        mm[F_MASK:F_MASK+8]=struct.pack('<Q',k.SetThreadAffinityMask(k.GetCurrentThread(),1<<cpu))
    mm[F_PID:F_PID+4]=struct.pack('<I',os.getpid());mm[F_READY]=1
    import ctypes
    handles=(ctypes.c_void_p*2)(ph,ev);windows=0;active=0.0;chunks=0;reason='?'
    while True:
        left=hard_cap-(time.perf_counter()-start)
        if left<=0:reason='hard-cap';break
        r=k.WaitForMultipleObjects(2,handles,False,int(min(left,3600)*1000))   # zero CPU while idle: parent death or "go"
        if r==0:reason='parent-exit';break
        if r==0x102:continue
        if r==1:
            if mm[F_FLAG]==2:reason='stop';break
            if mm[F_FLAG]==1:
                windows+=1;t0=time.perf_counter();n=0
                while mm[F_FLAG]==1:
                    for _ in range(3000):pass
                    n+=1
                    if n%100==0:
                        if k.WaitForSingleObject(ph,0)==0:reason='parent-exit';break
                        if time.perf_counter()-t0>spin_cap:break
                        if time.perf_counter()-start>hard_cap:reason='hard-cap';break
                active+=time.perf_counter()-t0;chunks+=n;mm[F_WIN:F_WIN+8]=struct.pack('<Q',windows)
                if reason in('parent-exit','hard-cap'):break
                if mm[F_FLAG]==2:reason='stop';break
                if time.perf_counter()-t0>spin_cap:        # a stuck flag: stop spinning until the flag is cycled
                    while mm[F_FLAG]==1 and k.WaitForSingleObject(ph,0)!=0:time.sleep(0.05)
            else:time.sleep(0.0005)   # event still set while the flag is already 0: parent is resetting it
            continue
        reason='wait-error-%d'%r;break
    try:
        d=_process_cpu_s(k);d.update(active_wall_s=active,chunks=chunks,windows=windows,cpu=cpu,reason=reason,lifetime_s=time.perf_counter()-start)
        open(stats_path,'w').write(json.dumps(d))
    except Exception:pass
    return 0

# ---------------------------------------------------------------- controller (decode process)
class KeepWarm:
    """One per runtime. enter()/leave() bracket each generate on the decode thread; both never raise."""
    def __init__(self,mode,log=None,stats_dir=None,cpu=None,hard_cap=HARD_CAP_S,spin_cap=SPIN_CAP_S,spawn_policy='ready',helper_path=None,python=None,collect_stats=False):
        self.mode=mode;self.log=log or _log_default;self.stats_dir=stats_dir;self.cpu=cpu;self.hard_cap=hard_cap;self.spin_cap=spin_cap
        self.spawn_policy=spawn_policy;self.helper_path=helper_path or os.path.abspath(__file__);self.python=python or getattr(sys,'_base_executable',None) or sys.executable;self.collect_stats=collect_stats
        self.proc=None;self.mm=None;self.ev=None;self.tag=None;self.disabled=False;self.closed=False;self.status='off' if mode=='off' else 'idle'
        self.notes=[];self.windows=0;self.skipped_dc=0;self.skipped_notready=0;self.active_s=0.0;self._tok=None;self._thread=None;self._spawn_thread=None;self._lock=threading.Lock()
        self.spawn_ms=None;self.helper_pid=None;self.stats_path=None;self.helper_stats=None
    # -- notes
    def note(self,text,disable=False):
        self.notes.append(text);self.log('PLaMo: keep-warm '+text)
        if disable:self.disabled=True;self.status='disabled'
    # -- helper life cycle
    def _setup(self):
        import mmap
        k=k32()
        if self.cpu is None:self.cpu=choose_cpu(process_mask(),cpu_cores())
        if self.cpu is None or not (process_mask()>>self.cpu)&1:raise RuntimeError('no eligible logical CPU')
        self.tag='plamo3kw71_%d_%d_%s'%(os.getpid(),int(time.time()*1000)%10000000,os.urandom(3).hex())
        self.mm=mmap.mmap(-1,32,tagname=self.tag);self.mm[F_FLAG]=0;self.mm[F_READY]=0
        self.ev=k.CreateEventW(None,True,False,self.tag+'_go')
        if not self.ev:raise OSError('CreateEventW failed')
        d=self.stats_dir or os.environ.get('TEMP') or '.';self.stats_path=os.path.join(d,self.tag+'.json')
    def _spawn(self):
        import subprocess
        at=time.perf_counter()
        try:
            self.proc=subprocess.Popen([self.python,'-I','-S',self.helper_path,'--helper',self.tag,str(os.getpid()),str(self.cpu),str(self.hard_cap),str(self.spin_cap),self.stats_path],
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,creationflags=0x08000000|0x00000200)   # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        except Exception as e:self.note('helper spawn failed (%s: %s); running without it'%(type(e).__name__,e),True)
        self.spawn_ms=(time.perf_counter()-at)*1000
    def start(self,block=False):
        """Create the shared state and spawn the helper in a daemon thread (never blocks the caller unless block=True). Safe to call repeatedly."""
        if self.mode=='off' or self.disabled or self.closed:return
        with self._lock:
            if self._spawn_thread is not None or self.proc is not None:return
            try:self._setup()
            except Exception as e:self.note('setup failed (%s: %s); running without it'%(type(e).__name__,e),True);return
            self.status='starting';self._spawn_thread=threading.Thread(target=self._spawn,name='keepwarm-spawn',daemon=True);self._spawn_thread.start()
        if block:self._spawn_thread.join()
    def wait_ready(self,timeout=READY_WAIT_S):
        t=time.time()+timeout
        while time.time()<t:
            if self.disabled or self.mm is None:return False
            if self.mm[F_READY]==1:return True
            if self.proc is not None and self.proc.poll() is not None:return False
            time.sleep(0.002)
        return False
    def helper_alive(self):
        return self.proc is not None and self.proc.poll() is None
    def helper_cpu_mask(self):
        return struct.unpack('<Q',bytes(self.mm[F_MASK:F_MASK+8]))[0] if self.mm is not None else 0
    # -- per generate
    def _power_ok(self):
        if self.mode!='ac':return True
        try:s=power_status()
        except Exception:s='unknown'
        if s=='ac':return True
        self.skipped_dc+=1;self.status='dc-skip' if s=='dc' else 'power-unknown-skip';return False
    def enter(self):
        """Pin the calling (decode) thread to the keep-warm CPU and turn the spinner on. Returns a token for leave() or None (feature not active for this generate)."""
        try:
            if self.mode=='off' or self.disabled or self.closed:return None
            if not self._power_ok():return None
            if self.spawn_policy=='lazy' and self.proc is None and self._spawn_thread is None:
                self.start(block=True);self.wait_ready()
            elif self.proc is None and self._spawn_thread is None:
                self.start()                      # 'ac' mode that was on battery at start, or a late start: spawn without waiting; this generate runs unpinned
                self.skipped_notready+=1;return None
            if self.disabled:return None
            if self._spawn_thread is not None and self._spawn_thread.is_alive():self.skipped_notready+=1;return None
            if not self.helper_alive():
                self.note('helper is not running (exit code %s); running without it'%(None if self.proc is None else self.proc.poll()),True);return None
            k=k32();prev=k.SetThreadAffinityMask(k.GetCurrentThread(),1<<self.cpu)
            if not prev:self.note('SetThreadAffinityMask failed; running without it',True);return None
            try:
                self._thread=threading.get_ident();self.mm[F_FLAG]=1
                if not k.SetEvent(self.ev):raise OSError('SetEvent failed')
            except BaseException:
                self.mm_flag_off();k.SetThreadAffinityMask(k.GetCurrentThread(),prev);raise
            self.status='spinning';self._tok=(prev,time.perf_counter());return self._tok
        except Exception as e:
            self.mm_flag_off();self.note('enter failed (%s: %s); running without it'%(type(e).__name__,e),True);return None
    def mm_flag_off(self):
        try:
            if self.mm is not None:self.mm[F_FLAG]=0
            if self.ev:k32().ResetEvent(self.ev)
        except Exception:pass
    def leave(self,token):
        """Spinner off, event reset, previous thread affinity restored. Never raises. Must run on the thread that called enter()."""
        if token is None:return
        prev,t0=token;self._tok=None
        try:self.mm_flag_off()
        finally:
            try:
                if self._thread==threading.get_ident():k=k32();k.SetThreadAffinityMask(k.GetCurrentThread(),prev)
                else:self.note('leave() on a different thread than enter(); affinity not restored',True)
            except Exception as e:self.note('affinity restore failed (%s)'%e)
            self.windows+=1;self.active_s+=time.perf_counter()-t0;self.status='idle'
    def close(self,wait=False):
        """Stop the helper. Does not block process exit unless wait=True (harness: reads the helper's CPU stats)."""
        if self.closed:return
        self.closed=True
        try:
            if self.mm is not None:
                self.mm[F_FLAG]=2
                if self.ev:k32().SetEvent(self.ev)
            if self.proc is not None:
                if wait or self.collect_stats:
                    try:self.proc.wait(5)
                    except Exception:self.proc.kill()
                    try:
                        import json;self.helper_stats=json.loads(open(self.stats_path).read())
                    except Exception:self.helper_stats=None
        except Exception:pass
    def stats(self):
        return dict(mode=self.mode,status=self.status,cpu=self.cpu,windows=self.windows,active_s=self.active_s,skipped_dc=self.skipped_dc,skipped_notready=self.skipped_notready,
            spawn_ms=self.spawn_ms,spawn_policy=self.spawn_policy,disabled=self.disabled,notes=list(self.notes),helper=self.helper_stats)

def make(mode,log=None,**kw):
    """Factory used by the runtime. spawn policy from env PLAMO3_KEEP_WARM_SPAWN=ready|lazy (measurement hook), CPU override PLAMO3_KEEP_WARM_CPU (test hook)."""
    cpu=os.environ.get('PLAMO3_KEEP_WARM_CPU');kw.setdefault('cpu',int(cpu) if cpu and cpu.isdigit() else None)
    kw.setdefault('spawn_policy',os.environ.get('PLAMO3_KEEP_WARM_SPAWN','ready'))
    return KeepWarm(mode,log=log,**kw)

if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='--helper':
    a=sys.argv[2:]
    try:sys.exit(helper_main(a[0],int(a[1]),int(a[2]),float(a[3]),float(a[4]),a[5]))
    except KeyboardInterrupt:sys.exit(0)
