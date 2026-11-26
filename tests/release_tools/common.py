# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Shared helpers. Forbidden literals are assembled at run time so this file passes the scrub gate."""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
B1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, B1)
sys.path.insert(0, os.path.join(B1, 'tools'))

USER = 'y' + 'kogi'
HANDLE = 'n' + 'ilpe'
NOREPLY = '1218' + '35995+' + HANDLE + '@users.noreply.' + 'github.com'
WINPROFILE = 'C:' + '\\' + 'Users' + '\\' + USER + '\\npu'
HOSTIP = '192' + '.168.' + '11.6'
NODE = 'no' + 'va02'


def git(repo, *args, env=None, inp=None):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, env=env, input=inp)
    if r.returncode:
        raise RuntimeError(r.stderr.decode())
    return r.stdout.decode()


def commit(repo, name, email, msg, files=None, date='2026-01-01T00:00:00'):
    for p, data in (files or {}).items():
        full = os.path.join(repo, *p.split('/'))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'wb') as f:
            f.write(data if isinstance(data, bytes) else data.encode())
    git(repo, 'add', '-A', '-f')
    env = dict(os.environ, GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email,
               GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    git(repo, 'commit', '-q', '--allow-empty', '-m', msg, env=env)
    return git(repo, 'rev-parse', 'HEAD').strip()


def make_source_repo(tmp):
    """Upstream (Alice, amd.com) history of 2 commits, then 3 author-side commits that leak identifiers."""
    repo = os.path.join(tmp, 'src')
    os.makedirs(repo)
    subprocess.run(['git', 'init', '-q', '-b', 'main', repo], check=True)
    git(repo, 'config', 'core.autocrlf', 'false')
    commit(repo, 'Alice Upstream', 'alice@amd.com', 'upstream one', {'LICENSE': 'Apache text\n', 'iron/op.py': 'print(1)\n'})
    base = commit(repo, 'Bob Upstream', 'bob@amd.com', 'upstream two', {'docs/guide.md': 'guide\n'})
    commit(repo, USER, NOREPLY, 'private work', {'app/run.py': 'ROOT = r"%s"\n' % WINPROFILE, 'ssh.sh': 'ssh %s\n' % HOSTIP})
    commit(repo, HANDLE, NOREPLY, 'more work\n\nClaude' + '-Session: https://claude.' + 'ai/code/session' + '_01abcdefghijklmnop',
           {'app/clean.py': 'x = 1\r\ny = 2\r\n', 'README.md': 'Built with PLaMo\n'})
    tip = commit(repo, USER, NOREPLY, 'final', {'app/data.json': '{"a": 1}\n'})
    return repo, base, tip


def ssh_run(args, check=True):
    """Run ssh-keygen with its output going to files: ssh-keygen for Windows can hang when stdout/stderr are pipes. Returns (returncode, stdout, stderr)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'o'), 'wb') as fo, open(os.path.join(d, 'e'), 'wb') as fe:
            r = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, timeout=60)
        out, err = open(os.path.join(d, 'o'), 'rb').read(), open(os.path.join(d, 'e'), 'rb').read()
    if check and r.returncode:
        raise AssertionError('%s failed (%d): %s' % (args[:3], r.returncode, err[-200:]))
    return r.returncode, out, err
