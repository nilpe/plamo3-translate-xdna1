#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Scrub gate for the public release: fails on personal or internal identifiers.

It scans (a) every file of a directory tree (names, raw bytes, UTF-16 text, JSON
escaped text), (b) git metadata of a repository (authors, committers, messages,
refs, tags, config, hooks, every reachable blob) and (c) optionally large binary
payloads (--deep). Exit status: 0 clean, 1 hits, 2 usage error.

The needles are written as character-class regexes on purpose, so that this file
itself passes the gate it implements.

    scrub_check.py --files DIR [--deep] [--json OUT]
    scrub_check.py --git REPO --upstream-base SHA --author nilpe --author-email ADDR
    scrub_check.py --files DIR --git REPO ...        (both)
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

SEV_ERROR, SEV_WARN = 'error', 'warn'
_SEP = r'(?:\\\\|\\|/)+'           # path separator, also JSON-escaped
_ALLOWED_MAIL_DOMAINS = (
    'amd.com', 'xilinx.com', 'example.com', 'example.org', 'example.net', 'pfn.jp',
    'preferred.jp', 'github.com', 'huggingface.co', 'python.org', 'llvm.org',
    'apache.org', 'numpy.org', 'pypi.org', 'readthedocs.io', 'gnu.org',
)
_TLDS = r'(?:com|org|net|jp|io|dev|edu|gov|info|ai|me|cc|us|uk|de|fr|cn|co\.jp|ac\.jp)'

# (rule id, severity, regex source (bytes, case-insensitive), description)
RULES = [
    ('user-name', SEV_ERROR, r'y[k]ogi', 'author machine user name'),
    ('author-mail', SEV_ERROR, r'n[i]lpe@', 'author e-mail (only the git author line may carry it)'),
    ('noreply-mail', SEV_ERROR, r'(?<![0-9])[0-9]+\+[A-Za-z0-9-]+@users\.noreply\.github\.com|users\.noreply\.github\.com',
     'GitHub noreply address'),
    ('alt-handle', SEV_WARN, r'n[i]lpe0555', 'second account handle (HF); a user decision'),
    ('private-ip', SEV_ERROR,
     r'(?<![0-9.])(?:192\.168|10\.(?:[0-9]{1,3})|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9]|\.[0-9])',
     'private IPv4 address'),
    ('cluster-host', SEV_ERROR,
     r'\b(?:n[o]va|f[e]rn)0[0-9]\b|m[i]yabi|omni\.hp[c]c|hp[c]c\.jp|jca[h]pc|xg24[i]002|x10[3]57',
     'internal compute host / account'),
    ('linux-home', SEV_ERROR, r'(?<![A-Za-z0-9_.\-])/home/[A-Za-z0-9_.\-]+', 'Linux home path'),
    ('mac-home', SEV_ERROR, r'(?<![A-Za-z0-9_.\-])/Users/[A-Za-z0-9_.\-]+/', 'macOS home path'),
    ('win-profile', SEV_ERROR, r'[A-Za-z]:' + _SEP + r'Users' + _SEP, 'Windows user profile path'),
    ('session-url', SEV_ERROR,
     r'claude\.ai' + _SEP + r'(?:code|chat|share|artifact)|session_01[0-9A-Za-z]{8,}|Claude[-]Session',
     'agent session URL / trailer'),
    ('agent-trailer', SEV_ERROR, r'Co-Authored-By:\s*Claude|Generated with \[?Claude', 'agent attribution trailer'),
    ('token-hf', SEV_ERROR, r'\bhf_[A-Za-z0-9]{30,}', 'Hugging Face token'),
    ('token-github', SEV_ERROR, r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})', 'GitHub token'),
    ('token-other', SEV_ERROR,
     r'\bsk-[A-Za-z0-9]{20,}|\bAKIA[0-9A-Z]{16}\b|\bxox[bpa]-[0-9A-Za-z-]{10,}|\bAIza[0-9A-Za-z_\-]{30,}',
     'API key pattern'),
    ('private-key', SEV_ERROR, r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----', 'private key block'),
    ('bearer', SEV_ERROR, r'Authorization:\s*Bearer\s+[A-Za-z0-9._\-]{16,}|HF_TOKEN\s*=\s*[A-Za-z0-9_\-]{8,}',
     'credential in text'),
    ('ssh-config', SEV_ERROR, r'\.ssh' + _SEP + r'config|known[_]hosts', 'ssh client configuration reference'),
    ('internal-tool', SEV_ERROR, r'c[o]dex_tasks|\bwin\.sh\b|runjob[0-9]*\.sh|pairs[0-9]+\.sh', 'internal helper name'),
    ('other-user', SEV_ERROR, r"another user'?s? (?:CPU|GPU)", 'remark about another person'),
    ('agent-name', SEV_WARN, r'\bCod[e]x\b', 'internal agent name'),
    ('machine-env', SEV_WARN, r'\b(?:COMPUTERNAME|USERDOMAIN|LOGONSERVER)\s*[=:]', 'environment dump'),
    ('email', SEV_ERROR,
     r'(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.' + _TLDS + r'\b', 'e-mail address'),
]
def _compile():
    return [(rid, sev, re.compile(src.encode('ascii'), re.I), desc) for rid, sev, src, desc in RULES]


_COMPILED = _compile()


def _mail_ok(match_text):
    dom = match_text.rsplit(b'@', 1)[-1].decode('latin-1').lower()
    return any(dom == d or dom.endswith('.' + d) for d in _ALLOWED_MAIL_DOMAINS)


def _unescape_json_bytes(data):
    data = data.replace(b'\\\\', b'\\').replace(b'\\/', b'/')
    return re.sub(rb'\\u00([0-7][0-9a-fA-F])', lambda m: bytes([int(m.group(1), 16)]), data)


def _scan_view(view, stripped, windows=None):
    """Run the rule set on one view. windows=None: the whole view (the reference behaviour). windows=[(start, end), ...] (disjoint, sorted):
    the same rules and the same counting on those regions only (see windowed_regions: every match of every rule lies inside one region)."""
    out = {}
    parts = [view] if windows is None else [view[s:e] for s, e in windows]
    for rid, sev, rx, desc in _COMPILED:
        if stripped and rid in ('email', 'linux-home', 'mac-home'):
            continue
        n, snips = 0, []
        for part in parts:
            for m in rx.finditer(part):
                if rid == 'email' and _mail_ok(m.group(0)):
                    continue
                n += 1
                if len(snips) < 3:
                    snips.append(m.group(0)[:80].decode('latin-1'))
        if n:
            out[rid] = (sev, n, snips)
    return out


def scan_bytes(data, name_hint='', windowed=False):
    """Return {rule: (severity, count, [snippets])}. The count is the number of occurrences in the most revealing
    view (raw bytes, JSON-unescaped text, UTF-16 text transcoded to UTF-8, NUL-stripped binary).
    windowed=True (the --fast-binary path for binary files): every view is only searched around the literal anchors of the rules."""
    if isinstance(data, str):
        data = data.encode('utf-8', 'replace')
    views = [(data, False)]
    ext = os.path.splitext(name_hint)[1].lower()
    if b'\\' in data and ext in ('.json', '.jsonl', '.js', '.md', '.txt', ''):
        views.append((_unescape_json_bytes(data), False))
    if data[:2] in (b'\xff\xfe', b'\xfe\xff'):
        try:
            views.append((data.decode('utf-16', errors='replace').encode('utf-8'), False))
        except Exception:
            pass
    elif data.count(b'\x00', 0, 4096) > 64 and ext in ('', '.txt', '.json', '.log', '.dll', '.pyd', '.exe', '.bin', '.ini'):
        views.append((data.replace(b'\x00', b''), True))     # UTF-16LE strings inside binaries
    out = {}
    for view, stripped in views:
        for rid, (sev, n, snips) in _scan_view(view, stripped, windowed_regions(view) if windowed else None).items():
            if n > out.get(rid, (sev, 0, []))[1]:
                out[rid] = (sev, n, snips)
    return out


def scan_name(rel):
    return {rid: (sev, 1, [m.group(0).decode('latin-1')[:80]]) for rid, sev, rx, desc in _COMPILED
            if rid != 'email' for m in [rx.search(rel.encode('utf-8', 'replace'))] if m}


BINARY_EXT = {'.bin', '.npy', '.npz', '.xclbin', '.pyd', '.dll', '.elf', '.pdi', '.o', '.exe', '.compact', '.marshal', '.safetensors', '.pt', '.pth', '.so', '.whl', '.zip', '.gz', '.png', '.jpg'}
# Literal cores of the rules (case-insensitive). EVERY possible match of a rule contains at least one of its cores (checked by tools/author/scrub_fast_test.py with a regex sampler);
# the cores are written as concatenations so that this file passes its own gate. The windowed fast path (--fast-binary, binary files only) runs the complete rule set on regions
# around the cores; text-like files always get the full pass.
RULE_ANCHORS = {
    'user-name': [b'y' + b'kogi'],
    'author-mail': [b'n' + b'ilpe@'],
    'noreply-mail': [b'users.n' + b'oreply.github.com'],
    'alt-handle': [b'n' + b'ilpe0555'],
    'private-ip': [b'192.16' + b'8.', b'10' + b'.', b'172' + b'.'],
    'cluster-host': [b'n' + b'ova0', b'f' + b'ern0', b'm' + b'iyabi', b'omni.hp' + b'cc', b'hp' + b'cc.jp', b'jca' + b'hpc', b'xg24' + b'i002', b'x10' + b'357'],
    'linux-home': [b'/ho' + b'me/'],
    'mac-home': [b'/us' + b'ers/'],
    'win-profile': [b'us' + b'ers'],
    'session-url': [b'claud' + b'e.ai', b'sessi' + b'on_01', b'claude-s' + b'ession'],
    'agent-trailer': [b'co-authored' + b'-by', b'generated w' + b'ith'],
    'token-hf': [b'hf' + b'_'],
    'token-github': [b'gh' + b'p_', b'gh' + b'o_', b'gh' + b'u_', b'gh' + b's_', b'gh' + b'r_', b'github' + b'_pat_'],
    'token-other': [b'sk' + b'-', b'ak' + b'ia', b'xo' + b'x', b'aiz' + b'a'],
    'private-key': [b'private k' + b'ey-----'],
    'bearer': [b'author' + b'ization:', b'hf_' + b'token'],
    'ssh-config': [b'.s' + b'sh', b'known_' + b'hosts'],
    'internal-tool': [b'c' + b'odex_tasks', b'win' + b'.sh', b'run' + b'job', b'pa' + b'irs'],
    'other-user': [b'another u' + b'ser'],
    'agent-name': [b'co' + b'dex'],
    'machine-env': [b'computer' + b'name', b'user' + b'domain', b'logon' + b'server'],
    'email': [b'.co' + b'm', b'.o' + b'rg', b'.n' + b'et', b'.j' + b'p', b'.' + b'io', b'.d' + b'ev', b'.e' + b'du', b'.g' + b'ov', b'.in' + b'fo', b'.' + b'ai', b'.' + b'me', b'.' + b'cc',
              b'.' + b'us', b'.' + b'uk', b'.' + b'de', b'.' + b'fr', b'.' + b'cn'],
}
ANCHORS = sorted({a for v in RULE_ANCHORS.values() for a in v})
_ANCHOR_VARIANTS = ANCHORS + [a.decode('latin-1').encode('utf-16le') for a in ANCHORS]
_ANCHOR_RX = re.compile(b'|'.join(re.escape(a) for a in sorted(_ANCHOR_VARIANTS, key=len, reverse=True)))
WINDOW = 4096                      # bytes around an anchor
RUN_CAP = 1 << 20                  # a window edge is extended through a run of match-continuation bytes, at most this far
_RUN = bytes(range(48, 58)) + bytes(range(65, 91)) + bytes(range(97, 123)) + b"._%+-@/\\=:' \t\r\n"


def windowed_regions(view):
    """Sorted, disjoint [start, end) regions of `view` that contain every match of every rule: +-WINDOW bytes around each anchor occurrence (case-insensitive, ASCII and
    UTF-16LE spellings, found across any block boundary because the whole view is searched at once), merged, and each edge extended through a run of bytes that a match
    could continue with (an e-mail address, a token, a path), so that look-behind / look-ahead and unbounded repeats see the same context as in the full view."""
    low = view.lower()
    # one alternation over all spellings (~100 MB/s); matches are non-overlapping, an anchor that starts inside another one lies within its window anyway (WINDOW >> anchor length)
    spots = [(m.start(), m.end()) for m in _ANCHOR_RX.finditer(low)]
    if not spots:
        return []
    n = len(view)
    regions = []
    for s, e in spots:
        s, e = max(0, s - WINDOW), min(n, e + WINDOW)
        if regions and s <= regions[-1][1]:
            if e > regions[-1][1]:
                regions[-1][1] = e
        else:
            regions.append([s, e])
    out = []
    for s, e in regions:
        left = view[max(0, s - RUN_CAP):s]
        s -= len(left) - len(left.rstrip(_RUN))
        right = view[e:e + RUN_CAP]
        e += len(right) - len(right.lstrip(_RUN))
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def looks_binary(path, head):
    return os.path.splitext(path)[1].lower() in BINARY_EXT or head.count(b'\x00') > 64 and os.path.splitext(path)[1].lower() not in ('.json', '.jsonl', '.md', '.txt', '.py', '.ps1', '.cmd')


def anchor_prefilter(path):
    """True when any rule anchor occurs in the file (16 MB blocks, case-insensitive)."""
    ov = max(len(a) for a in _ANCHOR_VARIANTS)
    with open(path, 'rb') as f:
        tail = b''
        while True:
            b = f.read(16 << 20)
            if not b:
                return False
            buf = (tail + b).lower()
            if any(a in buf for a in _ANCHOR_VARIANTS):
                return True
            tail = b[-ov:]


SCAN_BLOCK, SCAN_OVERLAP = 32 * 1024 * 1024, 1024   # a file is scanned in blocks of SCAN_BLOCK bytes with SCAN_OVERLAP bytes of carry (module level so that the tests can shrink them)


def scan_file(args):
    path, rel, deep, max_bytes = args[:4]
    fast = len(args) > 4 and args[4]
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return rel, size_err(e), [], 0
    hits = [(r, s, n, ['FILENAME: ' + t for t in sn]) for r, (s, n, sn) in scan_name(rel).items()]
    windowed = False
    if fast:
        with open(path, 'rb') as fh:
            head = fh.read(4096)
        windowed = looks_binary(path, head)   # binary files: every view is searched around the rule anchors only; text-like files keep the full pass
    if size > max_bytes and not deep:
        return rel, 'skipped-large', hits, size
    merged = {}
    block, overlap = SCAN_BLOCK, SCAN_OVERLAP
    with open(path, 'rb') as f:
        carry, first = b'', True
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            data = carry + chunk
            for rid, (sev, n, sn) in scan_bytes(data, rel if first else '', windowed).items():
                old = merged.get(rid, (sev, 0, []))
                merged[rid] = (sev, old[1] + n, (old[2] + sn)[:3])
            first = False
            carry = data[-overlap:]
            if size <= block:
                break
    hits += [(r, s, n, sn) for r, (s, n, sn) in merged.items()]
    return rel, 'ok', hits, size


def size_err(e):
    return 'error:' + str(e)


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != '.git']
        for f in fn:
            p = os.path.join(dp, f)
            yield p, os.path.relpath(p, root).replace(os.sep, '/')


def load_allow(path):
    if not path:
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def allowed(entries, rel, rule, snippet):
    for e in entries:
        if e.get('rule') not in (rule, '*'):
            continue
        if not fnmatch.fnmatch(rel, e.get('path', '*')):
            continue
        if e.get('contains') and e['contains'].lower() not in snippet.lower():
            continue
        return True
    return False


def scan_tree(root, deep=False, max_bytes=64 * 1024 * 1024, allow=(), workers=None, fast=False):
    jobs = [(p, rel, deep, max_bytes, fast) for p, rel in walk(root)]
    findings, skipped, errors = [], [], []
    if len(jobs) > 64:
        with ProcessPoolExecutor(max_workers=workers or min(2, os.cpu_count() or 2)) as pool:
            results = list(pool.map(scan_file, jobs, chunksize=16))
    else:
        results = [scan_file(j) for j in jobs]
    for rel, status, hits, size in results:
        if status == 'skipped-large':
            skipped.append((rel, size))
        elif status.startswith('error'):
            errors.append((rel, status))
        for rid, sev, n, snips in hits:
            snippet = snips[0] if snips else ''
            if allowed(allow, rel, rid, ' '.join(snips)):
                continue
            findings.append(dict(where='file', path=rel, rule=rid, severity=sev, snippet=snippet, count=n))
    return dict(files=len(jobs), findings=findings, skipped_large=skipped, errors=errors)


# ----------------------------------------------------------------------------- git

def git(repo, *args, check=True, binary=False, inp=None):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, input=inp)
    if check and r.returncode:
        raise RuntimeError('git %s failed: %s' % (' '.join(args), r.stderr.decode('utf-8', 'replace')))
    return r.stdout if binary else r.stdout.decode('utf-8', 'replace')


def scan_git(repo, upstream_base=None, author=None, author_email=None, allow=()):
    """Authors, committers, messages, refs, config, hooks and every reachable blob of a repository.

    Commits reachable from upstream_base (the public AMD history) keep their authors and their e-mail addresses; all
    other rules still apply to them. Every other commit must carry exactly the configured identity, and that identity's
    address is the only e-mail the gate tolerates in commit metadata."""
    findings = []

    def add(where, path, rid, sev, snippet, count=1):
        if not allowed(allow, path, rid, snippet):
            findings.append(dict(where=where, path=path, rule=rid, severity=sev, snippet=snippet, count=count))

    def meta_scan(text, ours):
        """Scan commit metadata. Upstream authors keep their addresses; ours may only carry the configured one."""
        if ours and author_email:
            text = text.replace(author_email, '')
        res = scan_bytes(text, 'x.txt')
        if not ours:     # the public upstream history is not ours to scrub: only identity rules apply to it
            for rid in ('email', 'noreply-mail', 'author-mail', 'agent-trailer', 'session-url', 'agent-name'):
                res.pop(rid, None)
        return res

    upstream = set(git(repo, 'rev-list', upstream_base).split()) if upstream_base else set()
    log = git(repo, 'log', '--all', '--format=%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x01')
    commits = 0
    for rec in log.split('\x01'):
        rec = rec.strip('\n')
        if not rec:
            continue
        h, an, ae, cn, ce, body = (rec.split('\x00', 5) + [''] * 6)[:6]
        commits += 1
        ours = h not in upstream
        if ours and author is not None:
            for who, name, mail in (('author', an, ae), ('committer', cn, ce)):
                if name != author or (author_email is not None and mail != author_email):
                    add('git-commit', h[:12], 'git-identity', SEV_ERROR, '%s %s <%s>' % (who, name, mail))
        text = '%s\n%s\n%s\n%s\n%s\n' % (an, ae, cn, ce, body)
        for rid, (sev, n, sn) in meta_scan(text, ours).items():
            add('git-commit', h[:12], rid, sev, sn[0], n)
    refs = git(repo, 'for-each-ref', '--format=%(refname) %(objecttype) %(taggername) %(taggeremail) %(subject)')
    for rid, (sev, n, sn) in meta_scan(refs, True).items():
        add('git-ref', 'refs', rid, sev, sn[0], n)
    gitdir = git(repo, 'rev-parse', '--absolute-git-dir').strip()
    for name in ('config', 'HEAD', 'description', 'packed-refs', 'info/exclude', 'FETCH_HEAD', 'ORIG_HEAD'):
        p = os.path.join(gitdir, name)
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                for rid, (sev, n, sn) in scan_bytes(f.read(), 'x.txt').items():
                    add('git-meta', '.git/' + name, rid, sev, sn[0], n)
    hooks = os.path.join(gitdir, 'hooks')
    if os.path.isdir(hooks):
        for n in os.listdir(hooks):
            if not n.endswith('.sample'):
                add('git-meta', '.git/hooks/' + n, 'git-hook', SEV_ERROR, 'active hook present')
    for r in git(repo, 'remote').split():
        add('git-meta', 'remote', 'git-remote', SEV_ERROR, 'remote %s configured' % r)
    fsck = subprocess.run(['git', '-C', repo, 'fsck', '--unreachable', '--no-reflogs', '--no-progress'],
                          capture_output=True).stdout.decode('utf-8', 'replace')
    unreachable = [l for l in fsck.splitlines() if l.startswith('unreachable')]
    if unreachable:
        add('git-meta', 'objects', 'unreachable-objects', SEV_ERROR, '%d unreachable objects' % len(unreachable),
            len(unreachable))
    # reachable file names and blobs; old public blobs may hold upstream e-mail addresses
    # blobs already public through the upstream history only answer to the identity rules (an old AMD docs page may
    # mention a home directory or an e-mail address; that is not ours to scrub)
    upstream_objs = set()
    if upstream_base:
        upstream_objs = {l.split(' ', 1)[0] for l in git(repo, 'rev-list', '--objects', upstream_base).splitlines()}
    upstream_rules = {'user-name', 'author-mail', 'cluster-host', 'session-url', 'agent-trailer', 'token-hf',
                      'token-github', 'token-other', 'private-key', 'bearer', 'internal-tool', 'other-user'}
    objs = [o.split(' ', 1) for o in git(repo, 'rev-list', '--objects', '--all').splitlines()]
    names = {o[0]: (o[1] if len(o) > 1 else '') for o in objs}
    check = subprocess.run(['git', '-C', repo, 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                           input=('\n'.join(o[0] for o in objs) + '\n').encode(), capture_output=True)
    blobs = []
    for line in check.stdout.decode().splitlines():
        sha, typ, size = line.split()
        if typ == 'blob':
            blobs.append((sha, int(size)))
    proc = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    seen_names = set()
    for sha, size in blobs:
        proc.stdin.write((sha + '\n').encode())
        proc.stdin.flush()
        proc.stdout.readline()
        data = proc.stdout.read(size + 1)[:size]
        nm = names.get(sha, '')
        for rid, (sev, n, sn) in scan_bytes(data, nm).items():
            if sha in upstream_objs and rid not in upstream_rules:
                continue
            add('git-blob', '%s (%s)' % (nm, sha[:10]), rid, sev, sn[0], n)
        if nm and nm not in seen_names:
            seen_names.add(nm)
            for rid, (sev, n, sn) in scan_name(nm).items():
                if sha in upstream_objs and rid not in upstream_rules:
                    continue
                add('git-blob', nm, rid, sev, 'FILENAME: ' + sn[0], 1)
    proc.stdin.close()
    proc.wait()
    return dict(commits=commits, blobs=len(blobs), findings=findings, upstream_commits=len(upstream))


def summarize(findings):
    by_rule = Counter()
    files = defaultdict(set)
    for f in findings:
        by_rule[(f['rule'], f['severity'])] += f.get('count', 1)
        files[(f['rule'], f['severity'])].add(f['path'])
    return [dict(rule=r, severity=s, hits=n, files=len(files[(r, s)])) for (r, s), n in sorted(by_rule.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fast-binary', action='store_true', help='binary files are first checked for the literal anchors of all rules; only files with an anchor get the full regex pass (text-like files always do)')
    ap.add_argument('--files', help='directory tree to scan')
    ap.add_argument('--git', help='git repository to scan (metadata, refs, config, blobs)')
    ap.add_argument('--upstream-base', help='commit whose ancestors are the public upstream history')
    ap.add_argument('--author', help='the only identity allowed on non-upstream commits (name)')
    ap.add_argument('--author-email', help='e-mail of that identity')
    ap.add_argument('--deep', action='store_true', help='also scan files larger than --max-mb (payload binaries)')
    ap.add_argument('--max-mb', type=int, default=64)
    ap.add_argument('--allow', help='JSON allow list: [{"rule","path","contains","reason"}]')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--warn-fails', action='store_true', help='treat warnings as failures')
    ap.add_argument('--top', type=int, default=15, help='files to list per rule in the text summary')
    a = ap.parse_args(argv)
    if not a.files and not a.git:
        ap.error('give --files and/or --git')
    allow = load_allow(a.allow)
    report = dict(version=1)
    findings = []
    if a.files:
        r = scan_tree(a.files, a.deep, a.max_mb * 1024 * 1024, allow, fast=a.fast_binary)
        report['files'] = r
        findings += r['findings']
        print('files scanned: %d, skipped (> %d MB, use --deep): %d' % (r['files'], a.max_mb, len(r['skipped_large'])))
    if a.git:
        r = scan_git(a.git, a.upstream_base, a.author, a.author_email, allow)
        report['git'] = r
        findings += r['findings']
        print('git: %d commits (%d upstream), %d blobs scanned' % (r['commits'], r['upstream_commits'], r['blobs']))
    report['summary'] = summarize(findings)
    errors = [f for f in findings if f['severity'] == SEV_ERROR]
    warns = [f for f in findings if f['severity'] == SEV_WARN]
    for row in report['summary']:
        print('  %-16s %-5s hits=%-8d files=%d' % (row['rule'], row['severity'], row['hits'], row['files']))
        shown = {}
        for f in findings:
            if f['rule'] == row['rule'] and f['severity'] == row['severity'] and f['path'] not in shown and len(shown) < a.top:
                shown[f['path']] = f['snippet']
        for p, s in shown.items():
            print('      %s  [%s]' % (p, s.replace('\n', ' ')[:60]))
    print('RESULT: %s (errors=%d, warnings=%d)' % ('FAIL' if errors or (warns and a.warn_fails) else 'PASS',
                                                  len(errors), len(warns)))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    return 1 if errors or (warns and a.warn_fails) else 0


if __name__ == '__main__':
    sys.exit(main())
