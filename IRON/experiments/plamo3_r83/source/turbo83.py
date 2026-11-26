"""R83 opt-in NPU turbo switch (v2: acquire in a background thread started at the very start of the process; release in a detached helper) (system-wide driver power mode, `xrt-smi configure --pmode`). OFF by default.

Contract (see TURBO.md):
* The switch only ever sets `turbo` (acquire) or the mode that was recorded before it switched (restore). Anything else is refused.
* Every start of the app (switch on or off) runs `stale_check()`: if a state/marker file exists whose holders are all dead
  (pid gone or a different process start time) the recorded mode is restored, verified by re-reading, and the marker removed.
  With no marker file, the only cost is one os.path.exists call: no xrt-smi call at all.
* Concurrency: one state file under an OS file lock holds the previous mode plus the list of live holders (refcount).
  The first holder switches and records the previous mode; later holders just join; the LAST holder to leave restores it.
* Never fails a translation: every error becomes a warning and the app runs in whatever mode is current.
* The result of `configure` is never trusted: the mode is always re-read with `examine -r platform` (R63: an error exit can accompany
  a successful change).
"""
import contextlib,json,os,re,sys,threading,time

TURBO='turbo'
KNOWN_MODES=('default','performance','balanced','powersaver','turbo')
SLOW_MODES=('balanced','powersaver')
DEFAULT_XRT_SMI=os.path.join(os.environ.get('SystemRoot') or os.environ.get('SYSTEMROOT') or 'C:\\Windows','System32','AMD','xrt-smi.exe')
ENV_SWITCH='PLAMO3_NPU_TURBO'
CALL_TIMEOUT_S=45  # upper bound kept for the (test) knob below; real calls use the per-command limits
EXAMINE_TIMEOUT_S=6  # H1: a hung xrt-smi is given up quickly (it used to delay the exit by 45 s)
CONFIGURE_TIMEOUT_S=10
EXIT_JOIN_S=3  # H1: at exit the acquire thread is waited for this long only; a still-running switch is left to the watchdog (the marker is kept)
LOCK_TIMEOUT_S=120
BARRIER_PHASE='pyxrt_import'  # startup event recorded right before device_open: turbo must be in effect before the hw_contexts are created (R83 probe 3b: a mode change after context creation has no effect on open contexts)
BARRIER_TIMEOUT_S=3.0
JOIN_TIMEOUT_S=EXIT_JOIN_S
ENV_RELEASE='PLAMO3_TURBO_RELEASE'  # 'detached' (default) or 'inline'
HELPER_WAIT_PARENT_S=20

class Refused(Exception):
    """a pmode request that is neither `turbo` nor the recorded previous mode"""

def _log(message):
    print('plamo3-turbo: '+message,file=sys.stderr,flush=True)

def enabled_by_env(environ=None):
    value=(os.environ if environ is None else environ).get(ENV_SWITCH,'').strip().lower()
    return value in('1','true','yes','on','turbo')

def state_dir():
    d=os.environ.get('PLAMO3_TURBO_STATE_DIR')
    if d:return d
    base=os.environ.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'),'.local','state')
    return os.path.join(base,'plamo3-npu-turbo')
def state_path():return os.path.join(state_dir(),'turbo-state.json')
def lock_path():return os.path.join(state_dir(),'turbo.lock')

# ---- process identity (pid + creation time) -------------------------------------------------------------------------
def process_start(pid):
    """creation time of a live process as a string, None if the pid does not exist."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.restype=wintypes.HANDLE;k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.CloseHandle.argtypes=[wintypes.HANDLE]
        h=k.OpenProcess(0x1000,False,int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:return None
        try:
            c,e,kt,ut=(wintypes.FILETIME() for _ in range(4))
            if not k.GetProcessTimes(h,c,e,kt,ut):return None
            code=wintypes.DWORD()
            if k.GetExitCodeProcess(h,ctypes.byref(code)) and code.value!=259:return None  # STILL_ACTIVE
            return str((c.dwHighDateTime<<32)|c.dwLowDateTime)
        finally:k.CloseHandle(h)
    try:  # POSIX (workstation self-tests only)
        with open('/proc/%d/stat'%int(pid)) as f:data=f.read()
        return data.rsplit(')',1)[1].split()[19]
    except (OSError,IndexError,ValueError):return None

def holder_alive(holder):
    # H1: a malformed holder entry (hand-edited or corrupt marker) counts as dead instead of raising
    if not isinstance(holder,dict) or type(holder.get('pid')) is not int:return False
    now=process_start(holder['pid'])
    return now is not None and now==holder.get('start')

# ---- file lock --------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def file_lock(path,timeout=LOCK_TIMEOUT_S):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    f=open(path,'a+b');deadline=time.monotonic()+timeout;got=False
    try:
        while True:
            try:
                if os.name=='nt':
                    import msvcrt;f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                got=True;break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('lock timeout: '+path)
                time.sleep(0.05)
        yield
    finally:
        if got:
            try:
                if os.name=='nt':
                    import msvcrt;f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_UN)
            except OSError:pass
        f.close()

# ---- xrt-smi ------------------------------------------------------------------------------------------------------------
def xrt_smi_command():
    if os.environ.get('PLAMO3_TURBO_TEST')=='1' and os.environ.get('PLAMO3_XRT_SMI'):
        exe=os.environ['PLAMO3_XRT_SMI']
        return [sys.executable,exe] if exe.lower().endswith('.py') else [exe]
    return [DEFAULT_XRT_SMI]

def _run(args):
    import subprocess
    limit=EXAMINE_TIMEOUT_S if list(args)[:1]==['examine'] else CONFIGURE_TIMEOUT_S
    if os.environ.get('PLAMO3_TURBO_TEST')=='1' and os.environ.get('PLAMO3_TURBO_TEST_TIMEOUT_S'):limit=float(os.environ['PLAMO3_TURBO_TEST_TIMEOUT_S'])
    return subprocess.run(xrt_smi_command()+list(args),capture_output=True,text=True,timeout=min(limit,CALL_TIMEOUT_S),encoding='utf-8',errors='replace',stdin=subprocess.DEVNULL)

def read_mode():
    """current pmode as a lower-case word, None if it cannot be read"""
    try:r=_run(['examine','-r','platform'])
    except FileNotFoundError:
        _log('xrt-smi.exe not found at %s; the AMD NPU driver is missing or too old.'%(xrt_smi_command()[-1],));return None
    except Exception as error:  # OSError, subprocess.SubprocessError (timeout)
        _log('cannot run xrt-smi examine: %s'%error);return None
    m=re.search(r'Power Mode\s*:\s*([A-Za-z]+)',(r.stdout or '')+'\n'+(r.stderr or ''))
    if not m:return None
    mode=m.group(1).lower()
    return mode if mode in KNOWN_MODES else None

def set_mode(mode,reason,previous=None):
    """the ONLY place that issues `configure --pmode`. Allowed: turbo (reason acquire) and exactly the recorded previous mode (reason restore)."""
    if reason=='acquire' and mode==TURBO:pass
    elif reason=='restore' and mode in KNOWN_MODES and mode==previous:pass
    else:raise Refused('refusing to set pmode %r (reason %r, recorded previous %r)'%(mode,reason,previous))
    try:r=_run(['configure','--pmode',mode])
    except Exception as error:
        return False,'xrt-smi configure failed to run: %s'%error
    return r.returncode==0,'exit %s %s'%(r.returncode,((r.stderr or '')+(r.stdout or '')).strip().replace('\n',' | ')[:200])

def _change_and_verify(mode,reason,previous=None):
    """configure, then re-read; success is decided ONLY by the re-read"""
    ok,detail=set_mode(mode,reason,previous)
    got=read_mode()
    return got==mode,got,detail

# ---- state file -----------------------------------------------------------------------------------------------------------
def _read_state():
    try:
        with open(state_path(),encoding='utf-8') as f:s=json.load(f)
        if isinstance(s,dict) and isinstance(s.get('holders'),list):
            s['holders']=[h for h in s['holders'] if isinstance(h,dict) and type(h.get('pid')) is int]  # H1: drop malformed holder entries (they would crash holder_alive forever)
            return s
    except (OSError,ValueError):pass
    return None
def _write_state(s):
    os.makedirs(state_dir(),exist_ok=True);tmp=state_path()+'.%d.tmp'%os.getpid()
    with open(tmp,'w',encoding='utf-8') as f:json.dump(s,f,indent=1);f.flush();os.fsync(f.fileno())
    os.replace(tmp,state_path())
def _remove_state():
    try:os.remove(state_path())
    except OSError:pass
def _me():
    pid=os.getpid();return dict(pid=pid,start=process_start(pid),ts=time.time())

def _restore_recorded(state,why):
    """restore state['previous_mode'] if the NPU is still in turbo; verify; True when the marker may be removed."""
    previous=state.get('previous_mode')
    if previous not in KNOWN_MODES:
        _log('marker has an unusable previous mode %r: removing it without touching the NPU power mode'%(previous,));return True
    current=read_mode()
    if current is None:
        _log('cannot read the NPU power mode to restore %s (%s); keeping the marker for the next start. Manual fix: xrt-smi configure --pmode default (admin shell)'%(previous,why));return False
    if current==previous:return True
    if current!=TURBO:
        _log('NPU power mode is %s (not the turbo this tool set); leaving it alone'%current);return True
    ok,got,detail=_change_and_verify(previous,'restore',previous)
    if ok:
        _log('restored NPU power mode %s (%s)'%(previous,why))
        if previous in SLOW_MODES:_warn_slow(previous)
        return True
    _log('WARNING: could not restore NPU power mode %s (now %s; %s). Manual fix: xrt-smi configure --pmode %s (admin shell)'%(previous,got,detail,'default' if previous not in KNOWN_MODES else previous))
    return False

def _warn_slow(mode):
    _log('WARNING: the NPU power mode is %s, which makes this app much slower (balanced is about 3x slower). Fix: xrt-smi configure --pmode default (admin shell)'%mode)

def stale_check():
    """run at every start. No marker file -> no xrt-smi call. Marker with only dead holders -> restore + verify + remove."""
    try:
        if not os.path.exists(state_path()):return False
        with file_lock(lock_path()):
            s=_read_state()
            if s is None:
                _remove_state();return False
            if any(holder_alive(h) for h in s['holders']):return False
            if _restore_recorded(s,'stale marker from a dead process'):_remove_state();return True
            return False
    except Exception as error:
        _log('stale-marker check failed (%s); continuing'%error);return False

def _install_signal_handlers():
    """SIGTERM / SIGBREAK become SystemExit so the `finally` restore runs (SIGINT already raises KeyboardInterrupt). A hard kill is covered by the marker."""
    import signal,threading
    saved={}
    if threading.current_thread() is not threading.main_thread():return saved
    def handler(signum,frame):raise SystemExit(128+signum)
    for name in('SIGTERM','SIGBREAK'):
        sig=getattr(signal,name,None)
        if sig is None:continue
        try:saved[sig]=signal.signal(sig,handler)
        except (ValueError,OSError):pass
    return saved
def _restore_signal_handlers(saved):
    import signal
    for sig,old in (saved or {}).items():
        try:signal.signal(sig,old)
        except (ValueError,OSError,TypeError):pass

_ACTIVE=None  # the running Session (set by start(), cleared at exit); read by barrier_hook()

def _test_knob(name,default=None):
    """fault-injection knobs, honoured ONLY under PLAMO3_TURBO_TEST=1 (gates)"""
    return os.environ.get(name,default) if os.environ.get('PLAMO3_TURBO_TEST')=='1' else default

class Session:
    """context manager. start()/__enter__ launches the acquire sequence in a background thread and returns at once (the xrt-smi calls overlap
    imports, manifest and runtime init: subprocess waits release the GIL). `switched` is set as soon as `configure` has RETURNED (or the
    thread decided there is nothing to switch / it failed); the runtime calls wait_switched() through the startup-event barrier before the
    device is opened. __exit__ always joins the thread first, then releases (finally semantics). Never raises."""
    def __init__(self,enabled):
        self.enabled=enabled;self.joined=False;self.mode_before=None;self.thread=None
        self.switched=threading.Event();self.done=threading.Event();self.barrier_done=False;self.stats={};self._t0=None;self._signals=None
        self.cancel=threading.Event();self.too_late=False  # H1: the barrier gave up waiting -> the acquire thread must not switch any more
    # ---- lifecycle ----
    def start(self):
        global _ACTIVE
        if self.thread is not None or not self.enabled:
            if not self.enabled:self.switched.set();self.done.set()
            return self
        self._t0=time.perf_counter();self._signals=_install_signal_handlers();_ACTIVE=self
        self.thread=threading.Thread(target=self._run,name='plamo3-turbo-acquire',daemon=True);self.thread.start()
        return self
    def __enter__(self):return self.start()
    def _run(self):
        try:self._acquire()
        except BaseException as error:  # never let the thread die silently with the marker unknown: the exit path still releases when joined
            _log('WARNING: could not apply turbo (%s); running in the current power mode'%(error,))
        finally:
            self.stats['acquire_done_ms']=(time.perf_counter()-self._t0)*1000
            self.switched.set();self.done.set()
    def _signal_switched(self):
        if not self.switched.is_set():
            self.stats['switched_ms']=(time.perf_counter()-self._t0)*1000;self.switched.set()
    def wait_switched(self,timeout=None):
        """block until the switch attempt has finished (configure returned / nothing to do / failed). False on timeout (a warning is logged, the run continues)."""
        if not self.enabled or self.thread is None:return True
        timeout=BARRIER_TIMEOUT_S if timeout is None else timeout
        at=time.perf_counter();ok=self.switched.wait(timeout);self.stats['barrier_wait_ms']=(time.perf_counter()-at)*1000
        if not ok:
            self.too_late=True;self.cancel.set();self.stats['too_late']=1
            _log('WARNING: the NPU power mode switch was still running after %.1f s when the NPU was opened (xrt-smi is slow); this run uses the current power mode'%timeout)
        return ok
    def __exit__(self,*exc):
        global _ACTIVE
        try:
            self._join()
            if self.joined:self._finish()
        finally:
            if _ACTIVE is self:_ACTIVE=None
            _restore_signal_handlers(self._signals)
        return False
    def _join(self):
        if self.thread is None:return
        deadline=time.monotonic()+JOIN_TIMEOUT_S
        while self.thread.is_alive() and time.monotonic()<deadline:
            try:self.thread.join(0.05)
            except KeyboardInterrupt:continue  # the restore matters more than the first Ctrl+C
        if self.thread.is_alive():
            self.joined=False  # cannot touch the state under a still-running acquire; the marker (holder = this process) lets the next start restore
            _log('WARNING: the turbo acquire is still running at exit; the marker is kept and the next start of the app restores the previous mode')
    def _finish(self):
        try:
            if os.environ.get(ENV_RELEASE,'detached')!='inline' and self._release_detached():return
            self._release()
        except Exception as error:_log('WARNING: turbo release failed (%s). Manual fix: xrt-smi configure --pmode default (admin shell)'%error)
    def _release_detached(self):
        """hand the restore (examine, configure, examine: ~0.5 s) to a helper process that waits for this one to exit, so the user does not wait for it.
        The state file keeps this process as a holder; the helper runs stale_check() once this process is dead (all holders dead -> restore + verify + remove).
        True when the helper was started."""
        try:
            import subprocess
            flags=0
            if os.name=='nt':flags=getattr(subprocess,'DETACHED_PROCESS',8)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0x200)|getattr(subprocess,'CREATE_NO_WINDOW',0x08000000)
            subprocess.Popen([sys.executable,os.path.abspath(__file__),'--release-after',str(os.getpid()),str(process_start(os.getpid()))],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,creationflags=flags)
            return True
        except Exception as error:
            _log('turbo release helper could not be started (%s); restoring inline'%error);return False
    # ---- acquire (runs in the background thread) ----
    def _acquire(self):
        with file_lock(lock_path()):
            s=_read_state()
            if s is not None:
                live=[h for h in s['holders'] if holder_alive(h)]
                if live:
                    # another turbo instance owns the switch: join (refcount); re-assert turbo only if somebody changed it
                    s['holders']=live+[_me()];_write_state(s);self.joined=True;self.mode_before=s.get('previous_mode');self._signal_switched()
                    if read_mode()!=TURBO:_log('WARNING: turbo is not active although another instance holds it; running in the current mode')
                    return
                if not _restore_recorded(s,'stale marker from a dead process'):
                    _log('WARNING: a previous turbo session could not be restored; not switching. Running in the current mode');return
                _remove_state()
            current=read_mode()
            if current is None:
                _log('WARNING: cannot read the NPU power mode (xrt-smi examine failed); not switching, running in the current mode');return
            self.mode_before=current
            if current==TURBO:
                _log('NPU power mode is already turbo (set outside this app); leaving it as is');self._signal_switched();return
            if current in SLOW_MODES:_warn_slow(current)
            if _test_knob('PLAMO3_TURBO_TEST_RAISE')=='before-marker':raise RuntimeError('injected: before marker')
            if self.cancel.is_set():
                _log('turbo not applied: the NPU had already been opened without it (no effect on this run); not switching');return
            # marker BEFORE the switch: previous mode, pid, process start time, timestamp
            _write_state(dict(version=1,previous_mode=current,created=time.time(),holders=[_me()]));self.joined=True
            ok,detail=set_mode(TURBO,'acquire')
            if self.cancel.is_set():self.too_late=True;self.stats['too_late']=1;_log('WARNING: turbo was applied too late for this run (the NPU was already open); the power mode is restored at exit')
            self._signal_switched()  # configure has returned: the runtime may open the device now (success is verified below by the re-read)
            if _test_knob('PLAMO3_TURBO_TEST_RAISE')=='after-switch':raise RuntimeError('injected: after switch')
            got=read_mode();ok=(got==TURBO)
            self.stats['verified_ms']=(time.perf_counter()-self._t0)*1000
            if ok:
                _log('NPU power mode %s -> turbo'%current);_spawn_watchdog();return
            if got is not None and got!=current and got!=TURBO:
                _log('WARNING: unexpected NPU power mode %s after the switch attempt'%got)
            self.joined=False;_remove_state()
            _log('WARNING: could not set NPU power mode turbo (%s); running in %s. This usually needs an elevated (administrator) PowerShell; results are identical, only speed differs.'%(detail,got))
    def _release(self):
        with file_lock(lock_path()):
            s=_read_state()
            if s is None:return
            me=os.getpid()
            holders=[h for h in s['holders'] if h.get('pid')!=me and holder_alive(h)]
            if holders:
                s['holders']=holders;_write_state(s);return
            if _restore_recorded(s,'last turbo instance exiting'):_remove_state()
            else:
                s['holders']=[];_write_state(s)  # no live holder: the next start of the app sees a stale marker and retries
                _log('WARNING: marker kept; the next start of the app retries the restore')

def session(enabled):return Session(enabled)

def barrier_hook():
    """startup-event callback for the runtime (initialize83.Events): None when no turbo session is active (the OFF path does nothing at all)"""
    s=_ACTIVE
    if s is None or not s.enabled or s.thread is None:return None
    phase=_test_knob('PLAMO3_TURBO_BARRIER_PHASE',BARRIER_PHASE)
    def hook(event_phase):
        if s.barrier_done or event_phase!=phase:return
        s.barrier_done=True;s.wait_switched()
    return hook

def _spawn_watchdog():
    """H1: a restore watchdog that survives a kill of this process, including `taskkill /T` and a job-object kill. It waits for this process to
    disappear and then restores through the normal stale-marker path; it exits by itself when the marker is gone (normal restore). To escape the
    process tree it is started through an intermediate interpreter that exits at once (the watchdog's parent is then dead, so a tree kill from
    the root does not reach it) and, where the job allows it, with CREATE_BREAKAWAY_FROM_JOB. Never raises; False when it could not be started.
    PLAMO3_TURBO_WATCHDOG=off disables it."""
    if os.environ.get('PLAMO3_TURBO_WATCHDOG','on')=='off':return False
    try:
        import subprocess
        me=os.getpid();start=process_start(me)
        script=os.path.abspath(__file__)
        launcher=("import subprocess,sys\n"
                  "f=int(sys.argv[1]);a=[sys.argv[2],sys.argv[3],'--watch',sys.argv[4],sys.argv[5]]\n"
                  "kw=dict(stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)\n"
                  "try:subprocess.Popen(a,creationflags=f|0x01000000,**kw)\n"
                  "except OSError:subprocess.Popen(a,creationflags=f,**kw)\n")
        base=0
        if os.name=='nt':base=getattr(subprocess,'DETACHED_PROCESS',8)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0x200)|getattr(subprocess,'CREATE_NO_WINDOW',0x08000000)
        cmd=[sys.executable,'-c',launcher,str(base),sys.executable,script,str(me),str(start)]
        kw=dict(stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)
        if os.name=='nt':
            try:subprocess.Popen(cmd,creationflags=base|0x01000000,**kw)
            except OSError:subprocess.Popen(cmd,creationflags=base,**kw)
        else:subprocess.Popen(cmd,**kw)
        return True
    except Exception as error:
        _log('turbo restore watchdog could not be started (%s); a kill of this process is repaired by the next start of the app'%error);return False

def _watch_main(pid,start):
    """watchdog body: poll until the app process is gone, then restore through stale_check(); end when the marker disappears (the app restored itself)."""
    while True:
        if not os.path.exists(state_path()):return 0
        if process_start(pid)!=start:break
        time.sleep(0.5)
    time.sleep(0.2)
    return _helper_main(pid,start,wait=0)

def _helper_main(pid,start,wait=None):
    """detached release helper: wait until the app process is gone, then restore through the normal stale-marker path (no live holder -> restore, verify, remove)."""
    deadline=time.monotonic()+(HELPER_WAIT_PARENT_S if wait is None else wait)
    while time.monotonic()<deadline and process_start(pid)==start:time.sleep(0.02)
    if process_start(pid)==start:return 0  # the app is still alive: leave the marker for the next start
    import io,contextlib
    buf=io.StringIO()
    with contextlib.redirect_stderr(buf):stale_check()
    text=buf.getvalue()
    if text.strip():
        try:
            os.makedirs(state_dir(),exist_ok=True)
            with open(os.path.join(state_dir(),'turbo-release.log'),'a',encoding='utf-8') as f:f.write('%s pid=%s\n%s'%(time.strftime('%Y-%m-%d %H:%M:%S'),pid,text))
        except OSError:pass
    return 0

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--release-after':sys.exit(_helper_main(int(sys.argv[2]),sys.argv[3]))
    if len(sys.argv)==4 and sys.argv[1]=='--watch':sys.exit(_watch_main(int(sys.argv[2]),sys.argv[3]))
    sys.exit('usage: turbo83.py --release-after <pid> <start> | --watch <pid> <start>')
