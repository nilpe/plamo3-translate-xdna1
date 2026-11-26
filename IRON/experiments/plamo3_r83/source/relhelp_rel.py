# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Release end-to-end helpers (A1 staged helpers, retargeted to the installed tree) Every process runs the installed release tree. Kills only its own PIDs."""
import json, os, re, subprocess, sys, time
from pathlib import Path as _P
STAGE = str(next(a for a in _P(__file__).resolve().parents if (a / 'IRON' / 'iron' / 'applications' / 'plamo3').is_dir()))
REAL = STAGE
PY = STAGE + r'\venv\Scripts\python.exe'
PY_ACTIVE = PY
TRANSLATE = STAGE + r'\IRON\iron\applications\plamo3\translate.py'
ACTIVE_TRANSLATE = TRANSLATE
WRAPPER = STAGE + r'\translate_plamo3.ps1'
XRT = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', 'AMD', 'xrt-smi.exe')
OUT = REAL + '\\state\\regress'

def env(extra=None, staged=True):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(('R77_', 'PLAMO3_', 'R83_', 'R74_', 'H1_')):
            e.pop(k)
    e.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for k, v in (extra or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    return e

def pmode():
    try:
        o = subprocess.run([XRT, 'examine', '-r', 'platform'], capture_output=True, text=True, timeout=60).stdout
        m = re.search(r'Power Mode\s*:\s*(\S+)', o)
        return m.group(1) if m else 'UNKNOWN'
    except Exception as e:
        return 'ERR:' + repr(e)

def procs(pattern=r'python|xrt-smi'):
    o = subprocess.run(['powershell', '-NoProfile', '-Command',
                        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '%s' } | ForEach-Object { '{0}|{1}|{2}|{3}' -f $_.ProcessId,$_.ParentProcessId,$_.Name,$_.CommandLine }" % pattern],
                       capture_output=True, text=True).stdout
    out = []
    for l in o.splitlines():
        p = l.split('|', 3)
        if len(p) == 4:
            out.append(dict(pid=int(p[0]), ppid=int(p[1]), name=p[2], cmd=p[3]))
    return out

def run(cmd, *, input=None, e=None, timeout=120):
    t = time.perf_counter()
    if isinstance(input, str):
        input = input.encode('utf-8')
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e or env())
    to = False
    try:
        out, err = p.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        to = True
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], capture_output=True)
        out, err = p.communicate()
    return dict(rc=p.returncode, timeout=to, wall=round(time.perf_counter() - t, 3), stdout=out.decode('utf-8', 'replace'), stderr=err.decode('utf-8', 'replace'), cmd=[str(c) for c in cmd][-6:])

def cli(args, **kw):
    return run([PY, TRANSLATE] + list(args), **kw)

def ps(command, **kw):
    return run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command], **kw)

class Report:
    def __init__(self, name):
        self.name, self.rows, self.fail = name, [], 0
        os.makedirs(OUT, exist_ok=True)
        self.pm0, self.pr0 = pmode(), [p['pid'] for p in procs()]
        print('== %s pmode_before=%s' % (name, self.pm0), flush=True)
    def check(self, test, ok, detail=''):
        self.rows.append(dict(test=test, result='PASS' if ok else 'FAIL', detail=str(detail)[:600]))
        self.fail += (not ok)
        print('%s %s %s' % ('PASS' if ok else 'FAIL', test, str(detail)[:300].replace('\n', ' | ')), flush=True)
    def close(self):
        pm1 = pmode()
        left = [p for p in procs(r'python') if p['pid'] not in self.pr0 and 'h1a' not in p['cmd'] and p['pid'] != os.getpid()]
        self.check('pmode_default_after', pm1 == 'Default', pm1)
        self.check('no_leftover_python', not left, [(p['pid'], p['cmd'][:120]) for p in left])
        json.dump(dict(name=self.name, result='FAIL' if self.fail else 'PASS', pmode_before=self.pm0, pmode_after=pm1, rows=self.rows), open(OUT + '\\' + self.name + '.json', 'w'), indent=1)
        print('RESULT %s %s (%d checks)' % (self.name, 'FAIL' if self.fail else 'PASS', len(self.rows)), flush=True)
        return 1 if self.fail else 0

def no_trace(r):
    return 'Traceback' not in r['stderr'] + r['stdout'] and 'File "' not in r['stderr']

def one_line(r):
    return len([l for l in r['stderr'].splitlines() if l.strip()]) == 1
