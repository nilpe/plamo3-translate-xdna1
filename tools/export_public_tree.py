#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Build the public GitHub tree from the release worktree.

Result: a fresh local git repository whose history is the AMD upstream history (up to the fork point) plus exactly one
commit that holds the public tree. None of the author-side commits, objects or refs is carried over. The script never
pushes, never adds a remote and never tags; it prints the commands for the user.

    export_public_tree.py --source REPO --rev REV --allowlist public-tree.txt --out DIR [--overlay DIR]
                          [--rewrites rewrites.json] [--sums SHA256SUMS] [--author nilpe --author-email ADDR]
                          [--signature SHA256SUMS.sig] [--bundle OUT.bundle] [--build]

--signature FILE: the detached OpenSSH signature of the tree's SHA256SUMS; it is verified with the key pinned in the tree's
release.json ('signing') and then added to the single squashed commit as SHA256SUMS.sig. --bundle OUT: also write a git bundle
of the built repository (branch main), so it can be moved to the machine that tags and pushes.

Without --build nothing is written: the plan (fork point, file counts, rewrite hits, bound-file check) is printed.

Allow list (public-tree.txt): sections [repo] and [overlay], one glob per line, '+ glob' include / '- glob' exclude,
the last matching line wins, '**' matches across directories, '#' starts a comment.
Rewrites (rewrites.json): {"rules":[{"id","glob","find","replace","regex":false}], "rename":[{"from","to"}]}.
Files listed in --sums (the bound payload: their hashes are sealed in manifests) are NEVER rewritten: a rule that
would change one is an error, and every sums entry present in the tree must hash identically afterwards.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUTHOR = 'nilpe'
DEFAULT_MESSAGE = ('PLaMo 3 Translate on the AMD XDNA1 NPU: public release\n\n'
                   'Squashed release of the runtime, installer and tools on top of the amd/IRON history.\n'
                   'Built with PLaMo. Unofficial; not affiliated with Preferred Networks, Inc. or AMD.\n')


class ExportError(Exception):
    pass


def run(cmd, cwd=None, inp=None, env=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, env=env)
    if check and r.returncode:
        raise ExportError('%s failed (%d): %s' % (' '.join(cmd[:4]), r.returncode, r.stderr.decode('utf-8', 'replace')))
    return r.stdout


def glob_to_regex(pat):
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if pat.startswith('**/', i):
            out.append('(?:.*/)?'); i += 3
        elif pat.startswith('**', i):
            out.append('.*'); i += 2
        elif c == '*':
            out.append('[^/]*'); i += 1
        elif c == '?':
            out.append('[^/]'); i += 1
        else:
            out.append(re.escape(c)); i += 1
    return re.compile('^' + ''.join(out) + '$')


def parse_allowlist(text):
    sections, cur = {'repo': [], 'overlay': []}, None
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            cur = line[1:-1].strip()
            if cur not in sections:
                raise ExportError('allowlist line %d: unknown section %s' % (n, cur))
            continue
        if cur is None:
            raise ExportError('allowlist line %d: entry outside a section' % n)
        sign = '+'
        if line[0] in '+-' and line[1:2] == ' ':
            sign, line = line[0], line[2:].strip()
        sections[cur].append((sign == '+', glob_to_regex(line), line))
    return sections


def select(paths, rules):
    chosen = []
    for p in paths:
        verdict = False
        for inc, rx, _ in rules:
            if rx.match(p):
                verdict = inc
        if verdict:
            chosen.append(p)
    return chosen


# ---------------------------------------------------------------- git helpers

def is_ours(name, email, ours_names):
    return name in ours_names or any(x in email.lower() for x in ours_names)


def find_fork_point(repo, rev, ours_names):
    """Newest ancestor of rev that is not ours, with no author-side commit anywhere in its history."""
    log = run(['git', '-C', repo, 'log', '--first-parent', '--format=%H%x00%an%x00%ae%x00%cn%x00%ce', rev]).decode()
    ours_count = 0
    for line in log.splitlines():
        h, an, ae, cn, ce = line.split('\x00')
        if is_ours(an, ae, ours_names) or is_ours(cn, ce, ours_names):
            ours_count += 1
            continue
        base = h
        # prove it: nothing in base's history is ours, and every commit above it is ours
        hist = run(['git', '-C', repo, 'log', '--format=%an%x00%ae%x00%cn%x00%ce', base]).decode().splitlines()
        for x in hist:
            an2, ae2, cn2, ce2 = x.split('\x00')
            if is_ours(an2, ae2, ours_names) or is_ours(cn2, ce2, ours_names):
                raise ExportError('author-side commit inside the candidate upstream history of %s' % base[:10])
        above = run(['git', '-C', repo, 'rev-list', '--count', '%s..%s' % (base, rev)]).decode().strip()
        return base, int(above)
    raise ExportError('no upstream (non-author) commit found on the first-parent chain of %s' % rev)


def list_tree(repo, rev):
    out = run(['git', '-C', repo, 'ls-tree', '-r', '-z', '--full-tree', rev])
    entries = []
    for rec in out.split(b'\0'):
        if not rec:
            continue
        meta, path = rec.split(b'\t', 1)
        mode, typ, sha = meta.decode().split()
        entries.append((path.decode('utf-8'), mode, typ, sha))
    return entries


def read_blobs(repo, shas):
    proc = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for sha in shas:
        proc.stdin.write((sha + '\n').encode()); proc.stdin.flush()
        hdr = proc.stdout.readline().split()
        size = int(hdr[2])
        out[sha] = proc.stdout.read(size + 1)[:size]
    proc.stdin.close(); proc.wait()
    return out


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def parse_sums(path):
    sums = {}
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line:
                continue
            m = re.match(r'^([0-9a-f]{64}) [ *](.+)$', line)
            if not m:
                raise ExportError('%s line %d is not a sha256sum line' % (path, n))
            sums[m.group(2).replace('\\', '/')] = m.group(1)
    return sums


# ---------------------------------------------------------------- rewrites

def apply_rewrites(files, rules, renames, bound):
    """files: {path: bytes}. Returns (new files, report). Rules only touch text (valid UTF-8) files."""
    report = {}
    # renames first (pure prefix moves)
    moved = {}
    for path, data in files.items():
        new = path
        for r in renames:
            if path.startswith(r['from']):
                new = r['to'] + path[len(r['from']):]
                break
        if new in moved:
            raise ExportError('rename collision on %s' % new)
        moved[new] = data
        if new != path:
            report.setdefault('rename', []).append((path, new))
    files = moved
    for rule in rules:
        rx = glob_to_regex(rule['glob'])
        pat = re.compile(rule['find']) if rule.get('regex') else None
        hits = 0
        for path in sorted(files):
            if not rx.match(path):
                continue
            try:
                text = files[path].decode('utf-8')
            except UnicodeDecodeError:
                continue
            if pat:
                new, n = pat.subn(rule['replace'], text)
            else:
                n = text.count(rule['find'])
                new = text.replace(rule['find'], rule['replace'])
            if n:
                if path in bound:
                    raise ExportError('rule %s would change the sealed file %s' % (rule['id'], path))
                files[path] = new.encode('utf-8')
                hits += n
        report[rule['id']] = hits
    return files, report


# ---------------------------------------------------------------- main build

def collect(repo, rev, allow_rules, overlay, overlay_rules):
    entries = list_tree(repo, rev)
    skipped = [e[0] for e in entries if e[1] == '120000' or e[2] != 'blob']
    entries = [e for e in entries if e[1] != '120000' and e[2] == 'blob']
    chosen = set(select([e[0] for e in entries], allow_rules))
    blobs = read_blobs(repo, [e[3] for e in entries if e[0] in chosen])
    files = {e[0]: blobs[e[3]] for e in entries if e[0] in chosen}
    modes = {e[0]: e[1] for e in entries if e[0] in chosen}
    over = {}
    if overlay:
        allp = []
        for dp, dn, fn in os.walk(overlay):
            dn[:] = [d for d in dn if d not in ('.git', '__pycache__', 'tests_out', 'out')]
            for f in fn:
                allp.append(os.path.relpath(os.path.join(dp, f), overlay).replace(os.sep, '/'))
        for p in select(allp, overlay_rules):
            with open(os.path.join(overlay, p), 'rb') as fh:
                over[p] = fh.read()
    files.update(over)
    return files, modes, sorted(set(over)), skipped


def write_tree(out, files, modes, overlay_paths):
    for p, data in files.items():
        dst = os.path.join(out, *p.split('/'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(data)
        if modes.get(p) == '100755' and os.name != 'nt':
            os.chmod(dst, 0o755)


def build_repo(src, base, out, files, modes, overlay_paths, author, email, message, date):
    if os.path.exists(out) and os.listdir(out):
        raise ExportError('output directory %s is not empty' % out)
    os.makedirs(out, exist_ok=True)
    tmpl = tempfile.mkdtemp(prefix='emptytmpl')          # no sample hooks, no templates
    try:
        run(['git', 'init', '-q', '--template=' + tmpl, '-b', 'main', out])
    finally:
        shutil.rmtree(tmpl, ignore_errors=True)
    # copy only the objects reachable from the fork point (the author-side history stays behind)
    pack = run(['git', '-C', src, 'pack-objects', '--revs', '--stdout', '--quiet'], inp=(base + '\n').encode())
    run(['git', '-C', out, 'index-pack', '--stdin', '--fix-thin'], inp=pack)
    run(['git', '-C', out, 'update-ref', 'refs/heads/main', base])
    for k, v in (('core.autocrlf', 'false'), ('core.safecrlf', 'false'), ('core.filemode', 'false'),
                 ('user.name', author), ('user.email', email), ('commit.gpgsign', 'false')):
        run(['git', '-C', out, 'config', k, v])
    run(['git', '-C', out, 'read-tree', '--empty'])
    write_tree(out, files, modes, overlay_paths)
    run(['git', '-C', out, 'add', '-A', '-f'])
    env = dict(os.environ, GIT_AUTHOR_NAME=author, GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_NAME=author,
               GIT_COMMITTER_EMAIL=email)
    if date:
        env['GIT_AUTHOR_DATE'] = env['GIT_COMMITTER_DATE'] = date
    run(['git', '-C', out, 'commit', '-q', '--no-verify', '-F', '-', '--cleanup=verbatim'], inp=message.encode('utf-8'),
        env=env)
    # the local config carried the identity only for the commit; keep the public repo's config free of it
    for k in ('user.name', 'user.email'):
        run(['git', '-C', out, 'config', '--unset', k], check=False)
    return run(['git', '-C', out, 'rev-parse', 'HEAD']).decode().strip()


def verify_blobs(out, files):
    """Committed blobs must equal the files byte for byte (catches autocrlf, filters, hooks)."""
    entries = list_tree(out, 'HEAD')
    got = {e[0] for e in entries}
    want = set(files)
    if got != want:
        raise ExportError('committed file set differs from the plan: +%s -%s' % (sorted(got - want)[:5], sorted(want - got)[:5]))
    blobs = read_blobs(out, [e[3] for e in entries])
    bad = [e[0] for e in entries if blobs[e[3]] != files[e[0]]]
    if bad:
        raise ExportError('committed bytes differ from the tree for %d files, e.g. %s' % (len(bad), bad[:3]))
    return len(entries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='private repository / release worktree')
    ap.add_argument('--rev', default='HEAD')
    ap.add_argument('--base', help='fork point (default: detected from commit authors)')
    ap.add_argument('--ours', action='append', default=None,
                    help='author names / e-mail fragments that identify author-side commits (repeatable)')
    ap.add_argument('--allowlist', required=True)
    ap.add_argument('--overlay', help='directory with release files (docs, installer) added on top')
    ap.add_argument('--rewrites')
    ap.add_argument('--sums', help='SHA256SUMS of the release: sealed files must come out byte-identical')
    ap.add_argument('--out', help='output repository directory (required with --build)')
    ap.add_argument('--author', default=DEFAULT_AUTHOR)
    ap.add_argument('--author-email', help='commit e-mail (required with --build; use the address GitHub shows for your account)')
    ap.add_argument('--message-file')
    ap.add_argument('--date', help='commit date (ISO 8601) for reproducible output')
    ap.add_argument('--signature', help='detached SSH signature of the tree\'s SHA256SUMS (verified, then committed as SHA256SUMS.sig)')
    ap.add_argument('--bundle', help='write a git bundle (branch main) of the built repository to this file (needs --build)')
    ap.add_argument('--build', action='store_true', help='actually write the repository (default: dry run)')
    ap.add_argument('--max-mb', type=float, default=50.0, help='refuse a tree larger than this (plan: < 50 MB)')
    a = ap.parse_args(argv)
    try:
        return _main(a)
    except ExportError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1


def _main(a):
    ours = a.ours or ['y' + 'kogi', 'nilpe']
    with open(a.allowlist, encoding='utf-8') as f:
        sections = parse_allowlist(f.read())
    rules, renames = [], []
    if a.rewrites:
        with open(a.rewrites, encoding='utf-8') as f:
            cfg = json.load(f)
        rules, renames = cfg.get('rules', []), cfg.get('rename', [])
    src = os.path.abspath(a.source)
    rev = run(['git', '-C', src, 'rev-parse', a.rev + '^{commit}']).decode().strip()
    if a.base:
        base = run(['git', '-C', src, 'rev-parse', a.base + '^{commit}']).decode().strip()
        above = int(run(['git', '-C', src, 'rev-list', '--count', '%s..%s' % (base, rev)]).decode())
    else:
        base, above = find_fork_point(src, rev, ours)
    print('source revision : %s' % rev[:12])
    print('upstream base   : %s  (%d author-side commits above it are squashed away)' % (base[:12], above))
    files, modes, overlay_paths, skipped = collect(src, rev, sections['repo'], a.overlay, sections['overlay'])
    if skipped:
        print('skipped (symlink/submodule): %d' % len(skipped))
    bound = {}
    if a.sums:
        bound = parse_sums(a.sums)
    files, report = apply_rewrites(files, rules, renames, set(bound))
    for k, v in report.items():
        print('rewrite %-24s %s' % (k, v if not isinstance(v, list) else '%d renames' % len(v)))
    if a.signature:
        import release_tool
        for need in ('SHA256SUMS', 'release.json'):
            if need not in files:
                raise ExportError('--signature needs %s in the tree' % need)
        tmp = tempfile.mkdtemp(prefix='plamo-sigcheck-')
        try:
            sums_path = os.path.join(tmp, 'SHA256SUMS')
            with open(sums_path, 'wb') as f:
                f.write(files['SHA256SUMS'])
            ok, msg = release_tool.ssh_verify(sums_path, os.path.abspath(a.signature), json.loads(files['release.json'].decode('utf-8-sig')).get('signing'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not ok:
            raise ExportError('the signature does not verify against the tree\'s SHA256SUMS: ' + msg)
        print('signature       : %s' % msg)
        with open(a.signature, 'rb') as f:
            files['SHA256SUMS.sig'] = f.read()
        modes.pop('SHA256SUMS.sig', None)
    mismatch = [p for p, h in bound.items() if p in files and sha256(files[p]) != h]
    if mismatch:
        raise ExportError('%d sealed files differ from SHA256SUMS (line endings? rewrites?): %s' % (len(mismatch), mismatch[:3]))
    in_sums = sum(1 for p in files if p in bound)
    if a.sums:
        print('sealed files in tree: %d identical to SHA256SUMS (of %d entries; the rest is payload/model)' % (in_sums, len(bound)))
    total = sum(len(v) for v in files.values())
    print('public tree     : %d files, %.2f MB (%d from the overlay)' % (len(files), total / 1e6, len(overlay_paths)))
    if total > a.max_mb * 1e6:
        raise ExportError('tree is %.1f MB, limit %.0f MB' % (total / 1e6, a.max_mb))
    if not a.build:
        print('DRY RUN: nothing written. Add --build --out DIR to create the repository.')
        return 0
    if not a.out:
        raise ExportError('--out is required with --build')
    if a.bundle and os.path.exists(a.bundle):
        raise ExportError('--bundle target %s exists' % a.bundle)
    if not a.author_email:
        raise ExportError('--author-email is required with --build')
    msg = open(a.message_file, encoding='utf-8').read() if a.message_file else DEFAULT_MESSAGE
    head = build_repo(src, base, os.path.abspath(a.out), files, modes, overlay_paths, a.author, a.author_email, msg, a.date)
    n = verify_blobs(os.path.abspath(a.out), files)
    print('built           : %s  HEAD %s, %d blobs byte-identical to the plan' % (a.out, head[:12], n))
    if a.bundle:
        run(['git', '-C', os.path.abspath(a.out), 'bundle', 'create', os.path.abspath(a.bundle), 'main'])
        print('bundle          : %s' % a.bundle)
    print('next            : python tools/scrub_check.py --files %s --git %s --upstream-base %s --author %s --author-email %s'
          % (a.out, a.out, base[:12], a.author, a.author_email))
    print('publishing is manual: this tool never adds a remote, tags or pushes. For an SSH-signed tag run:')
    print('    git -C %s -c gpg.format=ssh -c user.signingkey=<your .pub file> tag -s vX.Y.Z -m "release"' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
