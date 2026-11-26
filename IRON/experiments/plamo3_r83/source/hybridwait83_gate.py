"""R83 hybrid-wait-gate (draft for Phase C; keepwarm83_gate style: behaviour tests on the real module against a fake runtime/xrt, then a battery of SOURCE MUTANTS, each of which must be caught).
usage: hybridwait83_gate.py [--out result.json]   (runs on any OS: the high-resolution timer is replaced by a fake sleeper; one test builds the real kernel32 timer when running on Windows)
Behaviour tests (unmodified module): T1 flush-before-execute order; T2 ordinary wait() exactly once after the poll; T3 timeout fallback (device never writes) with a counted fallback; T4 exception in the poll path is caught
and falls back; T5 sentinel only on each hooked graph's single sync_outputs view, nothing for other graphs; T6 off switch PLAMO3_HYBRID_WAIT=0: install() returns None and rt.execute / rt.xrt / runlists are the SAME objects;
T7 uninstall restores everything; T8 default83._init_hybrid conditions (w4-m2 only, drafter present, keep-warm off); T9 forbidden calls absent; T10 pyxrt never patched; T11 polling reads with FROM_DEVICE.
Mutants: no_flush, wait_not_called, no_timeout, sentinel_on_input_view, install_with_keepwarm, install_on_serial_option, forbidden_timer_resolution, forbidden_affinity, forbidden_thread_priority, forbidden_helper_process,
off_switch_ignored, exception_uncaught, no_sync_from_device, patch_pyxrt, uninstall_does_not_restore."""
import sys, os, re, ast, json, types, time, importlib, collections, tempfile, shutil, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'hybridwait83.py'), encoding='utf-8').read()
DEF = open(os.path.join(HERE, 'default83.py'), encoding='utf-8').read()
FORBIDDEN = ['timeBeginPeriod', 'NtSetTimerResolution', 'powercfg', 'SetThreadAffinityMask', 'SetProcessAffinityMask', 'SetThreadPriority', 'SetPriorityClass', 'SetProcessInformation',
             'subprocess', 'Popen', 'os.system', 'threading', 'multiprocessing', 'sched_setaffinity', 'SetProcessWorkingSetSize', 'xrt-smi', 'SetThreadIdealProcessor', 'SetThreadInformation']

# ---------- fakes ----------
class FakeDir: XCL_BO_SYNC_BO_TO_DEVICE = 'TO'; XCL_BO_SYNC_BO_FROM_DEVICE = 'FROM'
class Dev:
    """device side of the fake: results appear in the 'device copy' DEV_MS after execute()"""
    def __init__(self, log, dev_ms=3.0, never=False): self.log = log; self.dev_ms = dev_ms; self.never = never; self.done_at = None; self.dev = {}; self.host = {}
class FakeBO:
    def __init__(self, dev, name, n): self.dev = dev; self.name = name; self.n = n
    def sync(self, d):
        L = self.dev.log; L.append(('sync', d, self.name))
        if d == 'TO': self.dev.dev[self.name] = bytes(self.dev.host[self.name])
        else:
            if self.dev.done_at is not None and time.perf_counter() >= self.dev.done_at and not self.dev.never and self.name in self.dev.result: self.dev.dev[self.name] = self.dev.result[self.name]
            self.dev.host[self.name][:] = self.dev.dev.get(self.name, bytes(self.n))
class FakeRunlist:
    def __init__(self, dev, rt): self.dev = dev; self.rt = rt; self.runs = []
    def add(self, r): self.runs.append(r)
    def execute(self):
        self.dev.log.append(('execute',)); self.dev.done_at = time.perf_counter() + self.dev.dev_ms / 1000
        self.dev.result = {n: (b'\x07\x00\x00\x00' * (len(self.dev.host[n]) // 4)) for n in self.dev.host}
    def wait(self, *a):
        self.dev.log.append(('wait',) + a)
        if self.dev.never: return
        d = self.dev.done_at; now = time.perf_counter()
        if d is not None and d > now: time.sleep(d - now)
class FakeXrt:
    xclBOSyncDirection = FakeDir
    def __init__(self, dev, rt): self.dev = dev; self.rt = rt
    def runlist(self, ctx): return FakeRunlist(self.dev, self.rt)
class FakeRT:
    def __init__(self, log, dev_ms=3.0, never=False):
        self.log = log; self.dev = Dev(log, dev_ms, never)
        names = {'decode': 'view226', 'r37_decode_bulk_taps': 'view226', 'verify2': 'v2view'}
        self.manifest = dict(graphs={g: dict(sync_outputs=[v], steps=[dict(args=['argA', 'argB'])]) for g, v in names.items()})
        self.manifest['graphs']['r37_prefill_tap0'] = dict(sync_outputs=['pview'], steps=[])
        self.views = {}; self.bos = {}
        for v, n in (('view226', 4), ('v2view', 8), ('pview', 16), ('argA', 64), ('argB', 64)):
            self.dev.host[v] = bytearray(n); self.dev.dev[v] = bytes(n); self.views[v] = memoryview(self.dev.host[v]); self.bos[v] = FakeBO(self.dev, v, n)
        self.xrt = FakeXrt(self.dev, self); self.graph_runlists = {}; self._graphs = {}
        for g in self.manifest['graphs']: self._graphs[g] = dict(segments=[dict(runlist=self.xrt.runlist('c'))])
        self._base_execute = self._exec; self.execute = self._base_execute
    def _bo(self, v): return self.bos[v]
    def _exec(self, graph, values):
        seg = self._graphs[graph]['segments'][0]; seg['runlist'].execute(); seg['runlist'].wait(); return {}
class FakeDrafter:
    def __init__(self, log, rt):
        self._coarse_host = __import__('numpy').zeros(8, dtype='<u2') if False else bytearray(16); self._to = 'TO'; self._from = 'FROM'
    # the gate only needs the ViewSentinel path; the array sentinel is exercised by the on-device dynamic gate

# ---------- loading a (possibly mutated) module ----------
class FakeTimer:
    def __init__(self): pass
    def sleep_ns(self, n):
        if n > 0: time.sleep(n / 1e9)
    def close(self): pass
def load(src, tag):
    mod = types.ModuleType('hybridwait83_' + tag); mod.__file__ = 'hybridwait83_%s.py' % tag
    exec(compile(src, mod.__file__, 'exec'), mod.__dict__); mod._Timer = FakeTimer; return mod
def mkrun(mod, dev_ms=3.0, never=False, prime=6):
    log = []; rt = FakeRT(log, dev_ms, never); hw = mod.HybridWait(rt, None, 1.0, None); hw.install(); return rt, hw, log
def call(rt, g): rt.execute(g, {'position': 5})

# ---------- tests: each returns None (pass) or raises AssertionError ----------
def T1(mod):
    rt, hw, log = mkrun(mod)
    for _ in range(8): call(rt, 'verify2')
    log.clear(); call(rt, 'verify2')
    i_exec = [i for i, e in enumerate(log) if e[0] == 'execute'][0]
    assert any(e == ('sync', 'TO', 'v2view') for e in log[:i_exec]), 'sentinel not flushed BO_TO_DEVICE before execute'
def T2(mod):
    rt, hw, log = mkrun(mod)
    for _ in range(8): call(rt, 'decode')
    log.clear(); call(rt, 'decode')
    waits = [e for e in log if e[0] == 'wait']; assert len(waits) == 1, ('ordinary wait() count', len(waits))
    assert log.index(waits[0]) > max(i for i, e in enumerate(log) if e[0] == 'sync' and e[1] == 'FROM'), 'wait() not after the poll'
    assert hw.stats['hybrid_waits'] >= 1, 'no hybrid wait happened'
def T3(mod):
    rt, hw, log = mkrun(mod, dev_ms=2.0)
    for _ in range(6): call(rt, 'verify2')
    rt.dev.never = True; rt.dev.dev_ms = 2.0
    t0 = time.perf_counter(); n0 = len(log)
    # poll limit guard so a missing timeout cannot hang the gate
    orig = mod.ViewSentinel.done; cnt = [0]
    def guarded(self):
        cnt[0] += 1; assert cnt[0] < 200000, 'poll loop ran without a timeout'; return orig(self)
    mod.ViewSentinel.done = guarded
    try: call(rt, 'verify2')
    finally: mod.ViewSentinel.done = orig
    assert hw.stats['fallbacks'] >= 1 and hw.stats['timeouts'] >= 1, 'timeout fallback not counted'
    assert any(e[0] == 'wait' for e in log[n0:]), 'ordinary wait not called after the timeout'
def T4(mod):
    rt, hw, log = mkrun(mod)
    for _ in range(8): call(rt, 'decode')
    orig = mod.ViewSentinel.done
    def boom(self): raise RuntimeError('injected')
    mod.ViewSentinel.done = boom
    try: call(rt, 'decode')
    finally: mod.ViewSentinel.done = orig
    assert hw.stats['fallbacks'] >= 1 and hw.stats['errors'] >= 1, 'exception not counted'
def T5(mod):
    rt, hw, log = mkrun(mod)
    assert set(hw.specs) == set(mod.MAIN_GRAPHS), ('hooked graphs', sorted(hw.specs))
    for g, sp in hw.specs.items(): assert sp.view == rt.manifest['graphs'][g]['sync_outputs'][0], ('sentinel view is not the sync_outputs view', g, sp.view)
    log.clear(); call(rt, 'r37_prefill_tap0'); assert not any(e[0] == 'sync' for e in log), 'unhooked graph touched'
def T6(mod):
    os.environ['PLAMO3_HYBRID_WAIT'] = '0'
    try:
        log = []; rt = FakeRT(log); ex, xr = rt.execute, rt.xrt; rls = [s['runlist'] for st in rt._graphs.values() for s in st['segments']]
        out = mod.install(rt, None)
        assert out is None and rt.execute is ex and rt.xrt is xr and [s['runlist'] for st in rt._graphs.values() for s in st['segments']] == rls, 'off switch did not leave the ordinary path'
        log.clear(); call(rt, 'decode'); assert not any(e[0] == 'sync' for e in log), 'sentinel traffic with the switch off'
    finally: os.environ.pop('PLAMO3_HYBRID_WAIT', None)
def T7(mod):
    log = []; rt = FakeRT(log); ex, xr = rt.execute, rt.xrt; rls = [s['runlist'] for st in rt._graphs.values() for s in st['segments']]
    hw = mod.HybridWait(rt, None, 1.0, None); hw.install(); hw.uninstall()
    assert rt.execute is ex and rt.xrt is xr and [s['runlist'] for st in rt._graphs.values() for s in st['segments']] == rls, 'uninstall did not restore'
def T8(mod):
    tree = ast.parse(DEF); fn = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef) and n.name == '_init_hybrid'][0]
    ns = {}; exec(compile(ast.Module([fn], []), 'x', 'exec'), ns); f = ns['_init_hybrid']
    class Fake:
        def __init__(self, mode, drafter, keepwarm, kwmode): self.mode = mode; self.drafter = drafter; self.keepwarm = keepwarm; self.keep_warm_mode = kwmode; self.env = types.SimpleNamespace(runtime=object()); self.log = lambda s: None; self.hybrid = 'unset'
    calls = []
    fake_mod = types.SimpleNamespace(enabled=lambda: True, install=lambda rt, d, log=None: calls.append(1) or 'HW')
    sys.modules['hybridwait83'] = fake_mod
    try:
        for mode, dr, kw, kwm, expect in (('w4-m2', object(), None, 'off', True), ('w4-serial', object(), None, 'off', False), ('w8-serial', object(), None, 'off', False), ('w4-m2', None, None, 'off', False),
                                          ('w4-m2', object(), object(), 'off', False), ('w4-m2', object(), None, 'on', False), ('w4-m2', object(), None, 'auto', False)):
            calls.clear(); o = Fake(mode, dr, kw, kwm); f(o); assert bool(calls) == expect, ('install condition', mode, dr is not None, kw is not None, kwm, 'installed' if calls else 'not installed')
    finally: sys.modules.pop('hybridwait83', None)
def _poll_guard(mod, limit=200000):
    """a mutant without a timeout would poll forever: every breaker test runs under a poll budget that raises AssertionError"""
    orig = mod.ViewSentinel.done; cnt = [0]
    def guarded(self):
        cnt[0] += 1; assert cnt[0] < limit, 'poll loop ran without a timeout'; return orig(self)
    mod.ViewSentinel.done = guarded
    return lambda: setattr(mod.ViewSentinel, 'done', orig)
def T12(mod):
    """H1 circuit breaker: BREAKER_THRESHOLD (4) consecutive timeouts of ONE graph disable the hint for that graph; per graph; a disabled graph only uses the ordinary wait."""
    assert mod.BREAKER_THRESHOLD == 4, ('documented threshold', mod.BREAKER_THRESHOLD)
    rt, hw, log = mkrun(mod)
    for _ in range(8): call(rt, 'decode'); call(rt, 'verify2')
    undo = _poll_guard(mod)
    try:
        rt.dev.never = True
        for _ in range(mod.BREAKER_THRESHOLD - 1): call(rt, 'decode')
        assert 'decode' not in hw.disabled, 'tripped before the threshold'
        call(rt, 'decode')
        assert 'decode' in hw.disabled and hw.stats['breaker_trips'] == 1, 'did not trip at the threshold'
        assert 'verify2' not in hw.disabled, 'the breaker must be per graph'
        log.clear(); call(rt, 'decode')
        assert not any(e[0] == 'sync' and e[1] == 'FROM' for e in log) and any(e[0] == 'wait' for e in log), 'a disabled graph still polls or lost the ordinary wait'
        assert hw.report()['disabled_graphs'] == ['decode'], hw.report()
    finally: undo()
def T13(mod):
    """a hybrid success resets the consecutive count: timeouts that are not consecutive never trip the breaker."""
    assert mod.BREAKER_THRESHOLD == 4, ('documented threshold', mod.BREAKER_THRESHOLD)
    rt, hw, log = mkrun(mod)
    for _ in range(8): call(rt, 'decode')
    undo = _poll_guard(mod)
    try:
        for _ in range(3):
            rt.dev.never = True
            for _ in range(mod.BREAKER_THRESHOLD - 1): call(rt, 'decode')
            rt.dev.never = False; call(rt, 'decode')
        assert 'decode' not in hw.disabled, 'non-consecutive timeouts tripped the breaker'
    finally: undo()
def code_names(src):
    """NAME tokens and non-docstring STRING tokens of the code (comments and docstrings ignored)."""
    import tokenize, io
    names = []; strs = []; prev = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME: names.append(tok.string)
        elif tok.type == tokenize.STRING and prev not in (tokenize.INDENT, tokenize.NEWLINE, None, tokenize.NL): strs.append(tok.string)
        prev = tok.type
    return names, strs

def T9(mod_src):
    names, strs = code_names(mod_src)
    for w in FORBIDDEN:
        if w == 'os.system':
            assert not any(a == 'os' and b == 'system' for a, b in zip(names, names[1:])), 'forbidden os.system'
        else:
            assert w not in names and not any(w in x for x in strs), ('forbidden call present', w)
def T10(mod_src):
    assert not re.search(r'pyxrt\s*\.\s*\w+\s*=|setattr\s*\(\s*(x|self\.rt\.xrt|pyxrt)', mod_src) and 'import pyxrt' not in mod_src, 'pyxrt patched or imported'
    assert not re.search(r'\b(RL|runlist)\s*\.\s*(execute|wait|add)\s*=', mod_src), 'class attribute patching of runlist'
def T11(mod):
    rt, hw, log = mkrun(mod)
    for _ in range(8): call(rt, 'decode')
    log.clear(); call(rt, 'decode'); assert any(e == ('sync', 'FROM', 'view226') for e in log), 'poll does not read with BO_FROM_DEVICE'

BEHAVIOUR = [('T1_flush_before_execute', T1), ('T2_ordinary_wait_after_poll', T2), ('T3_timeout_fallback', T3), ('T4_exception_falls_back', T4), ('T5_sentinel_views_only', T5),
             ('T6_off_switch_ordinary_path', T6), ('T7_uninstall_restores', T7), ('T8_install_conditions', T8), ('T11_poll_reads_from_device', T11), ('T12_breaker_trips_at_threshold', T12), ('T13_breaker_needs_consecutive', T13)]
STATIC = [('T9_forbidden_calls', T9), ('T10_pyxrt_not_patched', T10)]

def run_all(src, tag, static_only=False):
    res = {}
    mod = None if static_only else load(src, tag)
    for name, fn in ([] if static_only else BEHAVIOUR):
        try: fn(mod); res[name] = 'PASS'
        except AssertionError as e: res[name] = 'FAIL: ' + str(e)[:120]
        except Exception as e: res[name] = 'FAIL(exception): %s %s' % (type(e).__name__, str(e)[:100])
    for name, fn in STATIC:
        try: fn(src); res[name] = 'PASS'
        except AssertionError as e: res[name] = 'FAIL: ' + str(e)[:120]
    return res

def mutants():
    def rep(old, new, count=1):
        assert old in SRC, old; return SRC.replace(old, new, count)
    M = {}
    M['no_flush'] = rep("        self.bo.sync(self.to_dev)                       # flush: no dirty CPU line may survive the device write", "        pass")
    M['wait_not_called'] = rep("        out = self._real.wait()                          # ALWAYS: the ordinary synchronisation point", "        out = None")
    M['no_timeout'] = rep("                if time.perf_counter_ns() > deadline:", "                if False:")
    M['sentinel_on_input_view'] = rep("            self.specs[g] = ViewSentinel(rt, so[0])", "            self.specs[g] = ViewSentinel(rt, 'argA')")
    M['no_sync_from_device'] = rep("        self.bo.sync(self.from_dev)\n        b = bytes(self.mv)", "        b = bytes(self.mv)")
    M['off_switch_ignored'] = rep("    return low not in _OFF", "    return True")
    M['breaker_threshold_0'] = rep('BREAKER_THRESHOLD = 4', 'BREAKER_THRESHOLD = 0')
    M['breaker_threshold_1'] = rep('BREAKER_THRESHOLD = 4', 'BREAKER_THRESHOLD = 1')
    M['breaker_threshold_1000'] = rep('BREAKER_THRESHOLD = 4', 'BREAKER_THRESHOLD = 1000')
    M['breaker_never_resets'] = rep('            self.consec[g] = 0\n', '')
    M['breaker_disabled_graph_still_polls'] = rep('if spec is None or self.current in self.disabled:', 'if spec is None:')
    M['hint_always_done'] = rep('                if spec.done():', '                if True:')
    M['exception_uncaught'] = rep("        except Exception:\n            self.stats['errors'] += 1\n            self.stats['fallbacks'] += 1", "        except ZeroDivisionError:\n            self.stats['errors'] += 1\n            self.stats['fallbacks'] += 1")
    M['forbidden_timer_resolution'] = SRC + "\n_x = ctypes.WinDLL('winmm').timeBeginPeriod(1)\n"
    M['forbidden_affinity'] = SRC + "\n_x = ctypes.WinDLL('kernel32').SetThreadAffinityMask(0, 1)\n"
    M['forbidden_thread_priority'] = SRC + "\n_x = ctypes.WinDLL('kernel32').SetThreadPriority(0, 1)\n"
    M['forbidden_helper_process'] = SRC + "\nimport subprocess\n_x = subprocess.Popen(['x'])\n"
    M['patch_pyxrt'] = SRC + "\nimport pyxrt\npyxrt.runlist.wait = None\n"
    M['uninstall_does_not_restore'] = rep("        rt.xrt = self._orig_xrt\n", "")
    return M

def mutant_dispatch(name, src):
    """installation-condition mutants mutate default83.py; the others mutate hybridwait83.py"""
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); ap.add_argument('--output'); a = ap.parse_args(); a.out = a.out or a.output
    out = dict(base=run_all(SRC, 'base'), mutants={})
    assert all(v == 'PASS' for v in out['base'].values()), out['base']
    for name, src in mutants().items():
        r = run_all(src, name, static_only=name.startswith('forbidden') or name == 'patch_pyxrt'); out['mutants'][name] = 'CAUGHT by ' + ','.join(k for k, v in r.items() if v != 'PASS') if any(v != 'PASS' for v in r.values()) else 'NOT CAUGHT'
    # default83 condition mutants
    global DEF
    base_def = DEF
    for name, (old, new) in {'install_with_keepwarm': ("or getattr(self, 'keepwarm', None) is not None or self.keep_warm_mode not in ('off',)", "or False"), 'install_on_serial_option': ("if self.mode != 'w4-m2' or self.drafter is None", "if self.drafter is None")}.items():
        assert old in base_def, old; DEF = base_def.replace(old, new); r = run_all(SRC, name); out['mutants'][name] = 'CAUGHT by ' + ','.join(k for k, v in r.items() if v != 'PASS') if any(v != 'PASS' for v in r.values()) else 'NOT CAUGHT'
    DEF = base_def
    if sys.platform == 'win32':
        mod = importlib.import_module('hybridwait83'); t = mod._Timer(); a0 = time.perf_counter_ns(); t.sleep_ns(2_000_000); el = (time.perf_counter_ns() - a0) / 1e6; t.close(); out['real_timer_2ms_sleep_ms'] = el; assert 1.0 < el < 20, el
    out['all_mutants_caught'] = all(v.startswith('CAUGHT') for v in out['mutants'].values()); print(json.dumps(out, indent=1))
    out['result'] = 'PASS' if out['all_mutants_caught'] else 'FAIL'
    if a.out: open(a.out, 'w').write(json.dumps(out))
    sys.exit(0 if out['all_mutants_caught'] else 1)

if __name__ == '__main__': main()
