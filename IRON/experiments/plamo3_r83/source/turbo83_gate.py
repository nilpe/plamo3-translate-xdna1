"""R83 gate for the opt-in NPU turbo switch v2 (turbo83.py: background acquire + startup barrier + detached release) with a fake xrt-smi (a Python script), plus mutants that must FAIL.
usage: turbo83_gate.py --output result.json [--work dir]
Checks: restore after a normal run / after an exception / stale-marker restore on the next start (hard-killed holder, pid reuse, live holder untouched) /
concurrent instances (two real processes, both exit orders) / refusal to set balanced or powersaver (or anything but turbo and the recorded previous mode) /
warning-and-continue when configure or examine fails or the state dir is unusable / no xrt-smi call at all when the switch is off (no marker) /
balanced and powersaver restore with a slowness warning / marker is written BEFORE the switch /
R83 concurrency: quick exit while the acquire thread is still running (join), barrier waits for `configure` to return and times out, thread exception (before the marker, after the switch),
hard exit while the acquire thread is running (stale marker), detached release (restore done by a helper process after the app exits, last-holder rule kept)."""
import argparse,contextlib,importlib.util,io,json,os,shutil,subprocess,sys,tempfile,time
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=open(os.path.join(HERE,'turbo83.py'),encoding='utf-8').read()
FAKE=os.path.join(HERE,'fake_xrt_smi83.py')

class Ctx:
    def __init__(self,root,source):
        self.root=root;self.moddir=os.path.join(root,'mod');self.fake=os.path.join(root,'fake');self.state=os.path.join(root,'state')
        os.makedirs(self.moddir);os.makedirs(self.fake);open(os.path.join(self.moddir,'turbo83.py'),'w',encoding='utf-8').write(source)
        os.environ.update(PLAMO3_TURBO_TEST='1',PLAMO3_TURBO_RELEASE='inline',PLAMO3_XRT_SMI=FAKE,PLAMO3_TURBO_STATE_DIR=self.state,FAKE_XRT_DIR=self.fake,FAKE_XRT_MARKER=os.path.join(self.state,'turbo-state.json'))
        for k in('FAKE_XRT_FAIL_EXAMINE','FAKE_XRT_FAIL_CONFIGURE','FAKE_XRT_ERR_APPLY','FAKE_XRT_SILENT_NOOP','FAKE_XRT_FAIL_TO','FAKE_XRT_DELAY_MS','PLAMO3_TURBO_TEST_RAISE','PLAMO3_TURBO_BARRIER_PHASE','PLAMO3_NPU_TURBO'):os.environ.pop(k,None)
        spec=importlib.util.spec_from_file_location('turbo83_under_test',os.path.join(self.moddir,'turbo83.py'));self.M=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.M)
        self.fails=[]
    def check(self,ok,msg):
        if not ok:self.fails.append(msg)
    def reset(self,mode='default'):
        shutil.rmtree(self.state,ignore_errors=True)
        # H1: the restore watchdog / detached release helper of the previous scenario polls the marker every 0.5 s and may write turbo-release.log (re-creating the state dir) just after the removal: give every helper time to see the marker gone, then remove what it left, so no scenario starts on a leftover or races a helper (found by the A2 gate run: FileExistsError in os.makedirs(c.state))
        time.sleep(0.8);shutil.rmtree(self.state,ignore_errors=True)
        for f in('calls.log',):
            try:os.remove(os.path.join(self.fake,f))
            except OSError:pass
        open(os.path.join(self.fake,'mode.txt'),'w').write(mode)
        for k in('FAKE_XRT_FAIL_EXAMINE','FAKE_XRT_FAIL_CONFIGURE','FAKE_XRT_ERR_APPLY','FAKE_XRT_SILENT_NOOP','FAKE_XRT_FAIL_TO','FAKE_XRT_DELAY_MS','PLAMO3_TURBO_TEST_RAISE','PLAMO3_TURBO_BARRIER_PHASE'):os.environ.pop(k,None)
        os.environ['PLAMO3_TURBO_RELEASE']='inline'
    def mode(self):return open(os.path.join(self.fake,'mode.txt')).read().strip()
    def calls(self):
        try:return [l.split(' ',2)[2].rsplit(' marker=',1) for l in open(os.path.join(self.fake,'calls.log')).read().splitlines()]
        except OSError:return []
    def configures(self):return [c[0] for c in self.calls() if c[0].startswith('configure')]
    def has_state(self):return os.path.exists(os.path.join(self.state,'turbo-state.json'))
    def state_json(self):return json.load(open(os.path.join(self.state,'turbo-state.json')))
    @contextlib.contextmanager
    def err(self):
        buf=io.StringIO()
        with contextlib.redirect_stderr(buf):yield buf
    def spawn(self,args,**kw):
        env=dict(os.environ);env['PLAMO3_TURBO_RELEASE']=getattr(self,'release','detached');env['PYTHONPATH']=self.moddir+os.pathsep+env.get('PYTHONPATH','')
        return subprocess.Popen([sys.executable,os.path.abspath(__file__),'--hold']+args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,**kw)

def hold_main(args):
    """child process: acquire a session, signal ready, wait for the go file (or die hard), release."""
    sys.path.insert(0,os.environ['PYTHONPATH'].split(os.pathsep)[0]);import turbo83 as M
    name,wd,how=args[0],args[1],args[2]
    s=M.session(True);s.__enter__()
    if how=='killmid':  # hard exit while the acquire thread is running: wait until the marker exists (configure is then in flight), then die
        t=time.time()+30
        while not os.path.exists(os.path.join(os.environ['PLAMO3_TURBO_STATE_DIR'],'turbo-state.json')) and time.time()<t:time.sleep(0.005)
        os._exit(9)
    if how=='exit':  # normal exit through the detached release, no waiting for it
        s.thread.join();s.__exit__(None,None,None);open(os.path.join(wd,name+'.ready'),'w').write('%d %.3f'%(os.getpid(),time.time()));return 0  # written right after the release returned, while the app is still alive
    s.thread.join();open(os.path.join(wd,name+'.ready'),'w').write(str(os.getpid()))
    if how=='kill':os._exit(9)
    t=time.time()+60
    while not os.path.exists(os.path.join(wd,name+'.go')) and time.time()<t:time.sleep(0.05)
    s.__exit__(None,None,None);return 0
def wait_for(path,timeout=60):
    t=time.time()+timeout
    while not os.path.exists(path):
        if time.time()>t:return False
        time.sleep(0.05)
    return True

def settle(c,timeout=20):
    """wait for detached release helpers: marker gone (restored) or timeout"""
    t=time.time()+timeout
    while c.has_state() and time.time()<t:time.sleep(0.1)
    time.sleep(0.3)

def suite(c):
    M=c.M
    # 1 restore after a normal run; marker written BEFORE the switch; marker content
    c.reset('default');seen={}
    with c.err():
        with M.session(True) as s:
            s.wait_switched(30);seen['mode']=c.mode();seen['state']=c.state_json() if c.has_state() else None
    c.check(seen['mode']=='turbo','normal: not turbo inside the session')
    c.check(c.mode()=='default','normal: not restored to default')
    c.check(not c.has_state(),'normal: marker left behind')
    cf=c.configures();c.check(cf==['configure --pmode turbo','configure --pmode default'],'normal: configure calls %r'%cf)
    c.check(c.calls()[0][0].startswith('examine'),'normal: first call is not examine')
    c.check(c.calls()[-1][0].startswith('examine'),'normal: restore not verified by a final examine')
    turbo_call=[x for x in c.calls() if x[0]=='configure --pmode turbo']
    c.check(bool(turbo_call) and turbo_call[0][1]=='1','normal: marker not present when configure turbo ran')
    st=seen['state'] or {}
    c.check(st.get('previous_mode')=='default' and st.get('holders') and st['holders'][0].get('pid')==os.getpid() and st['holders'][0].get('start') and st['holders'][0].get('ts') and st.get('created'),'normal: marker content %r'%st)
    # 2 restore after an exception (and the exception still propagates)
    c.reset('default');raised=False
    with c.err():
        try:
            with M.session(True):
                raise RuntimeError('boom')
        except RuntimeError:raised=True
    c.check(raised,'exception: not propagated');c.check(c.mode()=='default','exception: not restored');c.check(not c.has_state(),'exception: marker left')
    # 3 stale-marker restore: hard-killed holder, found by a switch-OFF start
    c.reset('default');p=c.spawn(['k1',c.root,'kill']);p.communicate(timeout=120)
    c.check(c.mode()=='turbo','stale: the killed holder did not leave turbo set (test setup)');c.check(c.has_state(),'stale: no marker after a hard kill (test setup)')
    with c.err():r=c.M.stale_check()
    c.check(r is True and c.mode()=='default' and not c.has_state(),'stale: switch-off start did not restore (ret %r mode %s state %s)'%(r,c.mode(),c.has_state()))
    # 3b pid reuse: live pid, different start time -> stale
    c.reset('turbo');os.makedirs(c.state)
    json.dump(dict(version=1,previous_mode='default',created=time.time(),holders=[dict(pid=os.getpid(),start='1',ts=time.time())]),open(os.path.join(c.state,'turbo-state.json'),'w'))
    with c.err():r=c.M.stale_check()
    c.check(r is True and c.mode()=='default','stale: pid with a different start time not treated as dead')
    # 3c live holder is never restored under
    c.reset('turbo');os.makedirs(c.state)
    json.dump(dict(version=1,previous_mode='default',created=time.time(),holders=[dict(pid=os.getpid(),start=c.M.process_start(os.getpid()),ts=time.time())]),open(os.path.join(c.state,'turbo-state.json'),'w'))
    with c.err():r=c.M.stale_check()
    c.check(r is False and c.mode()=='turbo' and c.has_state() and not c.configures(),'stale: a live holder was restored under')
    # 3d next start WITH the switch on: restore the stale marker, then switch again, then restore
    c.reset('default');p=c.spawn(['k2',c.root,'kill']);p.communicate(timeout=120)
    with c.err():
        c.M.stale_check()
        with c.M.session(True) as s:s.wait_switched(30);inside=c.mode()
    c.check(inside=='turbo' and c.mode()=='default' and not c.has_state(),'stale+on: sequence wrong')
    # 4 concurrent instances, both exit orders
    for release,order in(('detached',('A','B')),('detached',('B','A')),('inline',('A','B')),('inline',('B','A'))):
        c.release=release
        c.reset('default');wd=c.root;procs={}
        for n in('A','B'):
            for f in(n+'.ready',n+'.go'):
                try:os.remove(os.path.join(wd,'%s%s'%(n,f[1:])))
                except OSError:pass
        procs['A']=c.spawn(['cA',wd,'wait']);ok=wait_for(os.path.join(wd,'cA.ready'))
        procs['B']=c.spawn(['cB',wd,'wait']);ok=ok and wait_for(os.path.join(wd,'cB.ready'))
        c.check(ok,'concurrent: children did not start')
        holders=len(c.state_json()['holders']) if c.has_state() else 0
        c.check(holders==2 and c.mode()=='turbo','concurrent: holders %d mode %s'%(holders,c.mode()))
        first,second=order
        open(os.path.join(wd,'c%s.go'%first),'w').write('1');procs[first].communicate(timeout=120)
        time.sleep(1.5)  # the detached helper of the first exit must find the other holder alive and do nothing
        h=len([x for x in c.state_json()['holders'] if c.M.holder_alive(x)]) if c.has_state() else 0
        c.check(c.mode()=='turbo' and c.has_state() and h==1,'concurrent(%s/%s first): first exit dropped turbo under the other (mode %s holders %d)'%(release,first,c.mode(),h))
        open(os.path.join(wd,'c%s.go'%second),'w').write('1');procs[second].communicate(timeout=120)
        settle(c)
        c.check(c.mode()=='default' and not c.has_state(),'concurrent(%s first): last exit did not restore'%first)
        cf=c.configures();c.check(cf.count('configure --pmode turbo')==1 and cf.count('configure --pmode default')==1,'concurrent: configure calls %r'%cf)
        for n in('cA','cB'):
            for suffix in('.ready','.go'):
                try:os.remove(os.path.join(wd,n+suffix))
                except OSError:pass
    # R83 concurrency
    # 10 quick exit while the acquire thread is still running (slow xrt-smi): the exit must join the thread and then restore
    c.reset('default');os.environ['FAKE_XRT_DELAY_MS']='200'
    with c.err():
        s=c.M.session(True);s.__enter__();s.__exit__(None,None,None)
        if s.thread is not None:s.thread.join(30)
    c.check(c.mode()=='default' and not c.has_state(),'quick-exit: not restored after an immediate exit (mode %s state %s)'%(c.mode(),c.has_state()))
    c.check(c.configures()==['configure --pmode turbo','configure --pmode default'],'quick-exit: configure calls %r'%c.configures())
    # 11 barrier: returns only after configure has returned (mode is turbo); other phases do not wait; OFF: no hook
    c.reset('default');os.environ['FAKE_XRT_DELAY_MS']='200'
    c.check(c.M.barrier_hook() is None,'barrier: a hook exists without an active session')
    with c.err():
        s=c.M.session(True);s.__enter__();hook=c.M.barrier_hook();c.check(hook is not None,'barrier: no hook with an active session')
        t=time.perf_counter();hook('bo_allocate:x');quick=time.perf_counter()-t
        c.check(quick<0.1,'barrier: a non-barrier phase waited %.3f s'%quick)
        hook(c.M.BARRIER_PHASE);inside=c.mode()
        c.check(inside=='turbo','barrier: returned before configure had returned (mode %s)'%inside)
        s.__exit__(None,None,None);s.thread.join(30)
    c.check(c.mode()=='default' and not c.has_state(),'barrier: not restored')
    with c.err():
        s=c.M.session(False);s.__enter__();c.check(c.M.barrier_hook() is None,'barrier: switch off but a hook exists');s.__exit__(None,None,None)
    # 12 barrier timeout: returns in time, warns, the run continues, the exit still joins + restores
    c.reset('default');os.environ['FAKE_XRT_DELAY_MS']='500';saved_to=c.M.BARRIER_TIMEOUT_S;c.M.BARRIER_TIMEOUT_S=0.2
    with c.err() as buf:
        s=c.M.session(True);s.__enter__();hook=c.M.barrier_hook();t=time.perf_counter();hook(c.M.BARRIER_PHASE);took=time.perf_counter()-t
        c.check(took<0.45,'barrier-timeout: waited %.3f s with a 0.2 s timeout'%took)
        s.__exit__(None,None,None);s.thread.join(30)
    c.M.BARRIER_TIMEOUT_S=saved_to
    c.check('WARNING' in buf.getvalue() and 'still running' in buf.getvalue(),'barrier-timeout: no warning')
    c.check(c.mode()=='default' and not c.has_state(),'barrier-timeout: not restored')
    # 13 acquire-thread exceptions: the app never fails; before the marker nothing is touched; after the switch the exit still restores; the barrier never hangs
    for where in('before-marker','after-switch'):
        c.reset('default');os.environ['PLAMO3_TURBO_TEST_RAISE']=where;ran=False
        with c.err() as buf:
            with c.M.session(True) as s:
                hook=c.M.barrier_hook();t=time.perf_counter();hook(c.M.BARRIER_PHASE);took=time.perf_counter()-t;ran=True
            c.check(took<1.0,'thread-exception(%s): barrier hung %.2f s'%(where,took))
        c.check(ran and 'WARNING' in buf.getvalue(),'thread-exception(%s): ran %s / no warning'%(where,ran))
        c.check(c.mode()=='default' and not c.has_state(),'thread-exception(%s): mode %s state %s'%(where,c.mode(),c.has_state()))
        if where=='before-marker':c.check(not c.configures(),'thread-exception(before-marker): configure was called %r'%c.configures())
    os.environ.pop('PLAMO3_TURBO_TEST_RAISE',None)
    # 14 hard exit while the acquire thread is running (marker written, configure in flight): the next start restores
    c.reset('default');os.environ['FAKE_XRT_DELAY_MS']='300';p=c.spawn(['km',c.root,'killmid']);p.communicate(timeout=120)
    time.sleep(1.5)  # the in-flight fake configure lands after its parent died (a real xrt-smi child survives its parent too)
    c.check(c.has_state(),'kill-mid: no marker after a hard exit during the acquire')
    os.environ.pop('FAKE_XRT_DELAY_MS')
    with c.err():r=c.M.stale_check()
    c.check(c.mode()=='default' and not c.has_state(),'kill-mid: next start did not restore (ret %r mode %s state %s)'%(r,c.mode(),c.has_state()))
    # 15 detached release: the restore is done by a helper process AFTER the app has exited
    c.release='detached';c.reset('default');p=c.spawn(['dr',c.root,'exit']);p.communicate(timeout=120);app_pid,exit_ts=open(os.path.join(c.root,'dr.ready')).read().split();exit_ts=float(exit_ts)
    c.check(c.mode() in('turbo','default'),'detached: setup')
    settle(c)
    c.check(c.mode()=='default' and not c.has_state(),'detached: helper did not restore (mode %s state %s)'%(c.mode(),c.has_state()))
    restore=[l for l in open(os.path.join(c.fake,'calls.log')).read().splitlines() if 'configure --pmode default' in l]
    # the restore must come AFTER the app finished its release (the helper waits for the app to exit); an inline restore is logged BEFORE that point
    c.check(len(restore)==1 and float(restore[0].split(' ')[0])>exit_ts,'detached: the restore was not made after the app released/exited %r (app released at %.3f)'%(restore,exit_ts))
    c.check([x[0] for x in c.calls()]==['examine -r platform','configure --pmode turbo','examine -r platform','examine -r platform','configure --pmode default','examine -r platform'],'detached: call sequence %r'%[x[0] for x in c.calls()])
    # 5 refusal
    c.reset('default')
    for mode,reason,prev in(('balanced','acquire',None),('powersaver','acquire',None),('default','acquire',None),('performance','acquire',None),('balanced','restore','default'),('powersaver','restore','default'),('turbo','restore','default'),('default','restore',None),('junk','restore','junk')):
        try:c.M.set_mode(mode,reason,prev);c.check(False,'refusal: set_mode(%s,%s,%s) was allowed'%(mode,reason,prev))
        except c.M.Refused:pass
    c.check(not c.configures(),'refusal: configure was called')
    # forged marker with an unusable previous mode: never configured
    c.reset('turbo');os.makedirs(c.state)
    json.dump(dict(version=1,previous_mode='junk',created=0,holders=[dict(pid=99999999,start='1',ts=0)]),open(os.path.join(c.state,'turbo-state.json'),'w'))
    with c.err():c.M.stale_check()
    c.check(not c.configures() and c.mode()=='turbo' and not c.has_state(),'refusal: forged previous mode was acted on')
    # 6 balanced / powersaver as the remembered mode: restored, with a slowness warning
    for slow in('balanced','powersaver'):
        c.reset(slow)
        with c.err() as buf:
            with c.M.session(True):pass
        text=buf.getvalue()
        c.check(c.mode()==slow and not c.has_state(),'%s: not restored (mode %s)'%(slow,c.mode()))
        c.check('slower' in text and slow in text,'%s: no slowness warning'%slow)
        c.check(c.configures()==['configure --pmode turbo','configure --pmode '+slow],'%s: configure calls %r'%(slow,c.configures()))
    # 7 failures: warn and continue
    for label,envs in(('configure-fails',{'FAKE_XRT_FAIL_CONFIGURE':'1'}),('silent-noop',{'FAKE_XRT_SILENT_NOOP':'1'})):
        c.reset('default');os.environ.update(envs);ran=False
        with c.err() as buf:
            with c.M.session(True):ran=True
        c.check(ran and 'WARNING' in buf.getvalue() and c.mode()=='default' and not c.has_state(),'%s: ran %s mode %s state %s'%(label,ran,c.mode(),c.has_state()))
        for k in envs:os.environ.pop(k)
    c.reset('default');os.environ['FAKE_XRT_FAIL_EXAMINE']='1';ran=False
    with c.err() as buf:
        with c.M.session(True):ran=True
    c.check(ran and 'WARNING' in buf.getvalue() and not c.configures() and not c.has_state(),'examine-fails: ran %s configures %r'%(ran,c.configures()))
    # error exit but the mode changed (R63 restricted token): success decided by re-reading, restore verified
    c.reset('default');os.environ['FAKE_XRT_ERR_APPLY']='1'
    with c.err():
        with c.M.session(True) as s:s.wait_switched(30);inside=c.mode()
    c.check(inside=='turbo' and c.mode()=='default' and not c.has_state(),'err-but-applied: inside %s after %s'%(inside,c.mode()))
    # restore fails: warning, marker kept, next start retries
    c.reset('default');os.environ['FAKE_XRT_FAIL_TO']='default'
    with c.err() as buf:
        with c.M.session(True):pass
    c.check('WARNING' in buf.getvalue() and c.mode()=='turbo' and c.has_state(),'restore-fails: warning/marker (mode %s state %s)'%(c.mode(),c.has_state()))
    os.environ.pop('FAKE_XRT_FAIL_TO')
    with c.err():r=c.M.stale_check()
    c.check(r is True and c.mode()=='default' and not c.has_state(),'restore-fails: next start did not retry (ret %r mode %s)'%(r,c.mode()))
    # unusable state dir: warn and continue
    c.reset('default');bad=os.path.join(c.root,'afile');open(bad,'w').write('x');os.environ['PLAMO3_TURBO_STATE_DIR']=os.path.join(bad,'sub');ran=False
    with c.err() as buf:
        with c.M.session(True):ran=True
        c.M.stale_check()
    c.check(ran and 'WARNING' in buf.getvalue() and not c.configures(),'bad-state-dir: ran %s'%ran)
    os.environ['PLAMO3_TURBO_STATE_DIR']=c.state
    # 8 switch OFF: no xrt-smi call at all without a marker
    c.reset('default')
    with c.err():
        r=c.M.stale_check()
        with c.M.session(c.M.enabled_by_env({})):pass
        with c.M.session(c.M.enabled_by_env({'PLAMO3_NPU_TURBO':'0'})):pass
    c.check(r is False and c.calls()==[],'off: xrt-smi was called %r'%c.calls())
    c.check(c.M.enabled_by_env({'PLAMO3_NPU_TURBO':'1'}) and not c.M.enabled_by_env({'PLAMO3_NPU_TURBO':''}),'off: env parsing')
    # 9 global: nothing but turbo / the recorded previous mode was ever configured is covered above (refusal + exact configure lists)
    return c.fails

# mutants: (name, old, new) applied to turbo83.py; every one must make the suite FAIL
MUTANTS=[
 ('no-restore-on-exit',"            if self.joined:self._finish()","            if False:self._finish()"),
 ('no-restore-after-exception',"            if self.joined:self._finish()","            if self.joined and exc[0] is None:self._finish()"),
 ('no-stale-check',"        if not os.path.exists(state_path()):return False\n        with file_lock(lock_path()):\n            s=_read_state()\n            if s is None:\n                _remove_state();return False","        return False\n        with file_lock(lock_path()):\n            s=_read_state()\n            if s is None:\n                _remove_state();return False"),
 ('pid-only-liveness',"return now is not None and now==holder.get('start')","return now is not None"),
 ('last-holder-rule-removed',"            if holders:\n                s['holders']=holders;_write_state(s);return","            if False:\n                s['holders']=holders;_write_state(s);return"),
 ('refusal-removed',"    else:raise Refused(","    elif True:pass\n    else:raise Refused("),
 ('trust-exit-code-on-restore',"    return got==mode,got,detail","    return ok,got,detail"),
 ('trust-exit-code-on-acquire',"got=read_mode();ok=(got==TURBO)","got=read_mode()"),
 ('thread-exception-skips-release',"            _log('WARNING: could not apply turbo (%s); running in the current power mode'%(error,))","            self.joined=False;_log('WARNING: could not apply turbo (%s); running in the current power mode'%(error,))"),
 ('off-path-calls-examine',"    try:\n        if not os.path.exists(state_path()):return False","    try:\n        read_mode()\n        if not os.path.exists(state_path()):return False"),
 ('marker-after-switch',"            _write_state(dict(version=1,previous_mode=current,created=time.time(),holders=[_me()]));self.joined=True\n            ok,detail=set_mode(TURBO,'acquire')","            ok,detail=set_mode(TURBO,'acquire')\n            _write_state(dict(version=1,previous_mode=current,created=time.time(),holders=[_me()]));self.joined=True"),
 ('restore-always-default',"    previous=state.get('previous_mode')\n    if previous not in KNOWN_MODES:","    previous='default'\n    if previous not in KNOWN_MODES:"),
 ('forged-marker-accepted',"    if previous not in KNOWN_MODES:\n        _log('marker has an unusable previous mode","    if False:\n        _log('marker has an unusable previous mode"),
 ('no-slow-warning',"    _log('WARNING: the NPU power mode is %s, which makes this app much slower","    return\n    _log('WARNING: the NPU power mode is %s, which makes this app much slower"),
 ('join-skipped',"            self._join()\n            if self.joined:self._finish()","            if self.joined:self._finish()"),
 ('restore-before-acquire-finished',"        if self.thread is None:return\n        deadline=","        return\n        deadline="),
 ('barrier-skipped',"ok=self.switched.wait(timeout)","ok=True"),
 ('barrier-never-times-out',"timeout=BARRIER_TIMEOUT_S if timeout is None else timeout","timeout=None"),
 ('barrier-event-not-set-on-error',"            self.switched.set();self.done.set()\n    def _signal_switched","            self.done.set()\n    def _signal_switched"),
 ('detached-release-removed',"            if os.environ.get(ENV_RELEASE,'detached')!='inline' and self._release_detached():return","            if False:return"),
 ('detached-helper-does-not-restore',"    with contextlib.redirect_stderr(buf):stale_check()","    with contextlib.redirect_stderr(buf):pass"),
]
def run_suite(source,root):
    c=Ctx(root,source)
    try:return suite(c)
    except Exception as error:
        return c.fails+['suite crashed: %r'%(error,)]

def run_one(idx,base):
    """one suite in its own process (isolated environment, no leftover threads): idx -1 = unmodified control, else MUTANTS[idx]"""
    r=os.path.join(base,'c' if idx<0 else 'm%d'%idx);os.makedirs(r,exist_ok=True)
    if idx<0:return dict(failures=run_suite(SRC,r))
    name,old,new=MUTANTS[idx]
    if SRC.count(old)!=1:return dict(name=name,result='FAIL',reason='mutation target not found exactly once')
    f=run_suite(SRC.replace(old,new),r);return dict(name=name,caught=bool(f),first_failure=f[0] if f else None)

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--hold':sys.exit(hold_main(sys.argv[2:]))
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--work');p.add_argument('--run-one',type=int);p.add_argument('--jobs',type=int,default=8);a=p.parse_args()
    base=a.work or tempfile.mkdtemp(prefix='turbo83gate-');os.makedirs(base,exist_ok=True)
    if a.run_one is not None:
        res=run_one(a.run_one,base);open(a.output,'w').write(json.dumps(res));return 0
    # control + every mutant, each in its own process (up to --jobs at once; they touch no NPU)
    todo=[-1]+list(range(len(MUTANTS)));running={};results={};outs={}
    while todo or running:
        while todo and len(running)<a.jobs:
            i=todo.pop(0);outs[i]=os.path.join(base,'res%d.json'%i)
            running[i]=subprocess.Popen([sys.executable,os.path.abspath(__file__),'--run-one',str(i),'--output',outs[i],'--work',base],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for i,pr in list(running.items()):
            if pr.poll() is not None:
                del running[i]
                try:results[i]=json.loads(open(outs[i]).read())
                except Exception as error:results[i]=dict(failures=['suite process failed: %r'%(error,)],name='?')
        time.sleep(0.2)
    out=dict(result='FAIL',control=dict(failures=results[-1].get('failures')),mutants=[])
    ok=not results[-1].get('failures')
    for i in range(len(MUTANTS)):
        r=results[i]
        if 'result' in r:ok=False
        else:ok=ok and bool(r.get('caught'))
        out['mutants'].append(r)
    out['result']='PASS' if ok else 'FAIL';open(a.output,'w').write(json.dumps(out,indent=1));print(json.dumps(out,indent=1));return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
