#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Release preparation: stage the payload, compute SHA256SUMS, prepare signing, verify signatures, plan uploads.

This tool PREPARES and VERIFIES only. It has no code path that uploads, pushes, tags, creates repositories or signs:
the signing key belongs to the maintainer, who runs the printed ssh-keygen command, and the upload commands are printed for
the maintainer to run. Every sub-command is therefore a dry run by construction.

    release_tool.py stage      --tree DIR --out DIR [--repo-tree DIR] [--model-files JSON] [--release NAME]
                               [--hf-repo ID --hf-revision REV]
    release_tool.py check      --tree DIR --sums FILE            (re-hash a tree against SHA256SUMS)
    release_tool.py sign-cmd   --sums FILE [--pubkey FILE]       (prints the ssh-keygen -Y sign command)
    release_tool.py verify-sig --sums FILE --sig FILE --release-json FILE [--fingerprint SHA256:...]
    release_tool.py plan       --stage DIR --hf-repo ID --github-repo OWNER/NAME --tag vX.Y.Z
"""
import argparse
import base64
import datetime
import hashlib
import json
import ntpath
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

EXCLUDE_DIRS = {'.git', '__pycache__', 'venv', 'model', 'state', '.venv', 'npu-home'}
SELF_FILES = {'SHA256SUMS', 'SHA256SUMS.sig'}
PAYLOAD_PREFIXES = ('plamo3-', 'data/', 'xrt/')
# optional certification reference data (gate-data/, fetched only with install.ps1 -WithGateData)
GATE_DATA_PREFIXES = ('gate-data/',)
DEFAULT_REVISION = 'main'


def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def walk_tree(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in EXCLUDE_DIRS or os.path.relpath(os.path.join(dp, d), root) != d and False)
        rel_dir = os.path.relpath(dp, root).replace(os.sep, '/')
        for f in sorted(fn):
            rel = f if rel_dir == '.' else rel_dir + '/' + f
            if rel in SELF_FILES or rel.endswith('.pyc'):
                continue
            out.append(rel)
    return sorted(out)


def classify(rel, in_git):
    if in_git:
        return 'git', ('doc' if rel.lower().endswith(('.md', '.txt')) or rel in ('LICENSE', 'NOTICE') else 'code')
    if rel.startswith(GATE_DATA_PREFIXES):
        return 'hf', 'gate-data'
    if rel.startswith(PAYLOAD_PREFIXES):
        return 'hf', 'payload'
    return 'hf', 'other'


def cmd_stage(a):
    tree = os.path.abspath(a.tree)
    if not os.path.isdir(tree):
        print('ERROR: --tree is not a directory', file=sys.stderr); return 2
    git_files = set(walk_tree(a.repo_tree)) if a.repo_tree else set()
    files = walk_tree(tree)
    entries, lines = [], []
    for rel in files:
        p = os.path.join(tree, *rel.split('/'))
        digest = sha256_file(p)
        source, klass = classify(rel, rel in git_files)
        entries.append(dict(path=rel, size=os.path.getsize(p), sha256=digest, source=source, **{'class': klass}))
        lines.append('%s  %s\n' % (digest, rel))
    model = []
    if a.model_files:
        with open(a.model_files, encoding='utf-8') as f:
            for m in json.load(f):
                model.append(dict(path='model/' + m['name'], size=m['size'], sha256=m['sha256'], source='external',
                                  **{'class': 'model'}, url=m['url'], optional=bool(m.get('optional', False))))
    os.makedirs(a.out, exist_ok=True)
    sums = os.path.join(a.out, 'SHA256SUMS')
    with open(sums, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(sorted(lines, key=lambda s: s.split('  ', 1)[1]))
    manifest = dict(version=1, release=a.release, generated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    hf=dict(repo=a.hf_repo, revision=a.hf_revision), files=entries + model)
    with open(os.path.join(a.out, 'PAYLOAD-MANIFEST.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write('\n')
    upload = [e for e in entries if e['source'] == 'hf']
    with open(os.path.join(a.out, 'UPLOAD-LIST.txt'), 'w', encoding='utf-8', newline='\n') as f:
        for e in upload:
            f.write('%s\t%d\t%s\n' % (e['path'], e['size'], e['sha256']))
    tot = sum(e['size'] for e in upload)
    print('files: %d (git %d, hf %d, external %d); hf bytes %.3f GB' % (
        len(entries) + len(model), sum(e['source'] == 'git' for e in entries), len(upload), len(model), tot / 1e9))
    print('wrote %s, PAYLOAD-MANIFEST.json, UPLOAD-LIST.txt' % sums)
    print('sha256(SHA256SUMS) = %s' % sha256_file(sums))
    print('next: python tools/release_tool.py sign-cmd --sums %s' % sums)
    return 0


def read_sums(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line:
                continue
            m = re.match(r'^([0-9a-f]{64}) [ *](.+)$', line)
            if not m:
                raise ValueError('%s: line %d is not a sha256sum line' % (path, n))
            out[m.group(2)] = m.group(1)
    return out


def cmd_check(a):
    sums = read_sums(a.sums)
    bad, missing = [], []
    for rel, want in sorted(sums.items()):
        p = os.path.join(a.tree, *rel.split('/'))
        if not os.path.isfile(p):
            missing.append(rel); continue
        if sha256_file(p) != want:
            bad.append(rel)
    extra = sorted(set(walk_tree(a.tree)) - set(sums)) if a.strict else []
    for x in bad: print('MISMATCH', x)
    for x in missing: print('MISSING ', x)
    for x in extra: print('EXTRA   ', x)
    print('checked %d: %d mismatch, %d missing, %d extra' % (len(sums), len(bad), len(missing), len(extra)))
    return 0 if not (bad or missing or extra) else 1


def sign_command(sums, pubkey='~/.ssh/id_ed25519.pub', namespace='file'):
    """The command the maintainer runs: writes <sums>.sig next to the file (OpenSSH signature, namespace 'file')."""
    return ['ssh-keygen', '-Y', 'sign', '-n', namespace, '-f', pubkey, os.path.abspath(sums)]


def cmd_sign_cmd(a):
    sums = os.path.abspath(a.sums)
    if not os.path.isfile(sums):
        print('ERROR: %s not found' % sums, file=sys.stderr); return 2
    cmd = sign_command(sums, a.pubkey)
    print('File to sign : %s' % sums)
    print('sha256       : %s' % sha256_file(sums))
    print('\nRun this yourself, on the machine that holds your key (this tool never signs; the private key is found through ssh-agent or the key file next to the .pub):\n')
    print('    ' + ' '.join(shlex.quote(c) for c in cmd))
    print('\nThen verify before publishing:\n')
    print('    python tools/release_tool.py verify-sig --sums %s --sig %s.sig --release-json release.json --fingerprint <YOUR-KEY-FINGERPRINT>' % (shlex.quote(sums), shlex.quote(sums)))
    print('\nPublish SHA256SUMS and SHA256SUMS.sig as assets of the GitHub release and next to SHA256SUMS in the public repository.')
    return 0


def ssh_key_fingerprint(public_key):
    """'SHA256:<base64 without padding>' of the key blob of an OpenSSH public key line (the comment field is ignored)."""
    parts = public_key.split()
    if len(parts) < 2 or not parts[0].startswith(('ssh-', 'ecdsa-', 'sk-')):
        raise ValueError('not an OpenSSH public key')
    return 'SHA256:' + base64.b64encode(hashlib.sha256(base64.b64decode(parts[1], validate=True)).digest()).decode('ascii').rstrip('=')


SSH_BUILTIN_MAX_BYTES = 8000


def builtin_ssh_keygen_path(environ=os.environ):
    return ntpath.join(environ.get('SystemRoot', r'C:\\Windows'), 'System32', 'OpenSSH', 'ssh-keygen.exe')


def is_builtin_ssh_keygen(path, environ=os.environ):
    return ntpath.normcase(ntpath.abspath(path)) == ntpath.normcase(ntpath.abspath(builtin_ssh_keygen_path(environ)))


def find_ssh_keygen(which=shutil.which, isfile=os.path.isfile, environ=os.environ):
    """ssh-keygen to use. Order: Git for Windows' usr\\bin\\ssh-keygen.exe, any other ssh-keygen on PATH, the built-in Windows OpenSSH client last
    (that one hangs on -Y verify when the signed file is larger than ~8 KB, see SSH_BUILTIN_MAX_BYTES)."""
    cands = []
    git = which('git')
    if git:
        d = ntpath.dirname(git)
        root = ntpath.dirname(d) if ntpath.basename(d).lower() in ('cmd', 'bin') else d
        if ntpath.basename(root).lower() == 'mingw64' or ntpath.basename(d).lower() == 'mingw64':
            root = ntpath.dirname(root)
        cands.append(ntpath.join(root, 'usr', 'bin', 'ssh-keygen.exe'))
    pf = environ.get('ProgramFiles')
    if pf:
        cands.append(ntpath.join(pf, 'Git', 'usr', 'bin', 'ssh-keygen.exe'))
    other = which('ssh-keygen')
    if other and not is_builtin_ssh_keygen(other, environ):
        cands.append(other)
    cands.append(builtin_ssh_keygen_path(environ))
    for c in cands:
        if c and isfile(c):
            return c
    return None


def ssh_verify(sums, sig, signing, fingerprint=None, ssh_keygen=None):
    """Return (ok, message): the detached OpenSSH signature `sig` of `sums` must verify against the key pinned in `signing`
    (release.json 'signing': public_key, principal, namespace [, fingerprint]); an external `fingerprint` pin must equal that key's SHA256 fingerprint."""
    signing = signing or {}
    key, principal, namespace = signing.get('public_key'), signing.get('principal'), signing.get('namespace')
    if not (key and principal and namespace):
        return False, 'release.json pins no signing key'
    if not (re.fullmatch(r'[A-Za-z0-9._-]+', principal) and re.fullmatch(r'[A-Za-z0-9._-]+', namespace)):
        return False, 'invalid principal or namespace'
    try:
        got = ssh_key_fingerprint(key)
    except ValueError as e:
        return False, 'the pinned signing key is not valid (%s)' % e
    if signing.get('fingerprint') and signing['fingerprint'] != got:
        return False, 'the pinned fingerprint %s does not match the pinned key (%s)' % (signing['fingerprint'], got)
    if fingerprint and re.sub(r'\s+', '', fingerprint) != got:
        return False, 'the key pinned in release.json (%s) is not the pinned %s' % (got, re.sub(r'\s+', '', fingerprint))
    ssh_keygen = ssh_keygen or find_ssh_keygen()
    if not ssh_keygen:
        return False, 'ssh-keygen not found'
    if is_builtin_ssh_keygen(ssh_keygen) and os.path.getsize(sums) > SSH_BUILTIN_MAX_BYTES:
        return False, ('the Windows built-in ssh-keygen hangs on files larger than about 8 KB, so SHA256SUMS cannot be checked with it; '
                               'install Git for Windows (its ssh-keygen is used automatically) to check the signature')
    tmp = tempfile.mkdtemp(prefix='plamo-sig-')
    try:
        allowed = os.path.join(tmp, 'allowed_signers')
        with open(allowed, 'w', encoding='ascii', newline='\n') as f:
            f.write('%s namespaces="%s" %s\n' % (principal, namespace, ' '.join(key.split()[:2])))
        # files instead of pipes: ssh-keygen for Windows can hang on pipes
        with open(sums, 'rb') as fin, open(os.path.join(tmp, 'out'), 'wb') as fo, open(os.path.join(tmp, 'err'), 'wb') as fe:
            r = subprocess.run([ssh_keygen, '-Y', 'verify', '-f', allowed, '-I', principal, '-n', namespace, '-s', sig], stdin=fin, stdout=fo, stderr=fe, timeout=60)
        out = open(os.path.join(tmp, 'out'), 'rb').read().decode('utf-8', 'replace').strip()
        err = open(os.path.join(tmp, 'err'), 'rb').read().decode('utf-8', 'replace').strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode == 0 and out.startswith('Good "%s" signature for %s ' % (namespace, principal)):
        return True, 'good signature by key %s (%s)' % (got, principal)
    return False, 'signature not valid (%s)' % ' '.join((err or out or 'ssh-keygen exit %d' % r.returncode).split())[:200]


def cmd_verify_sig(a):
    with open(a.release_json, encoding='utf-8-sig') as f:
        signing = json.load(f).get('signing')
    ok, msg = ssh_verify(a.sums, a.sig, signing, a.fingerprint)
    print(('OK: ' if ok else 'FAIL: ') + msg)
    return 0 if ok else 1


def cmd_plan(a):
    with open(os.path.join(a.stage, 'PAYLOAD-MANIFEST.json'), encoding='utf-8') as f:
        man = json.load(f)
    hf_files = [e for e in man['files'] if e['source'] == 'hf']
    tot = sum(e['size'] for e in hf_files)
    lines = [
        '# Upload plan (nothing has been uploaded; run the steps yourself)', '',
        'Release %s: %d payload files, %.2f GB for Hugging Face; %d files live in the git repository.' % (
            man.get('release'), len(hf_files), tot / 1e9, sum(e['source'] == 'git' for e in man['files'])), '',
        '## 0. Before anything', '',
        '- `python tools/scrub_check.py --files <public repo> --git <public repo> --upstream-base <fork point> --author nilpe --author-email <addr>` must print PASS.',
        '- `python tools/scrub_check.py --files <payload stage> --deep` must print PASS (payload text and binaries).',
        '- `python tools/release_tool.py check --tree <stage> --sums SHA256SUMS --strict` must print 0 problems.',
        '- Re-read the PLaMo Community License page; the pinned copy is dated in LICENSE-PLaMo-en.md.', '',
        '## 1. Hugging Face (payload, new PUBLIC repo; the name must contain "PLaMo")', '',
        '    hf repo create %s --repo-type model          # public; set the model card from hf/README.md' % a.hf_repo,
        '    hf upload-large-folder %s <stage dir> --repo-type model --include "plamo3-*" --include "data/*" --include "xrt/*"' % a.hf_repo,
        '', 'Then put the resulting commit id into PAYLOAD-MANIFEST.json (`hf.revision`), re-run `release_tool.py stage`, and',
        'sign the new SHA256SUMS. Users download by that revision, so a later change on the HF side cannot alter what they get.', '',
        '## 2. GitHub (code)', '',
        '    git -C <public repo> remote add origin https://github.com/%s.git' % a.github_repo,
        '    git -C <public repo> push origin main',
        '    git -C <public repo> -c gpg.format=ssh -c user.signingkey=<your .pub file> tag -s %s -m "%s"        # SSH-signed tag, signed with YOUR key' % (a.tag, a.tag),
        '    git -C <public repo> push origin %s' % a.tag, '',
        '## 3. GitHub release assets', '',
        '    python tools/release_tool.py sign-cmd --sums <stage>/SHA256SUMS         # prints the ssh-keygen command',
        '    gh release create %s --repo %s --title "%s" --notes-file RELEASE-NOTES.md \\' % (a.tag, a.github_repo, a.tag),
        '        <stage>/SHA256SUMS <stage>/SHA256SUMS.sig', '',
        '## 4. After publishing', '',
        '- Clone the repo fresh, run install.ps1 against the published payload on a clean machine; run `install.ps1 -Verify`.',
        '- `python tools/release_tool.py verify-sig --sums SHA256SUMS --sig SHA256SUMS.sig --release-json release.json --fingerprint <SHA256:...>`.',
    ]
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.write:
        with open(os.path.join(a.stage, 'UPLOAD-PLAN.md'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage'); s.set_defaults(fn=cmd_stage)
    s.add_argument('--tree', required=True); s.add_argument('--out', required=True)
    s.add_argument('--repo-tree'); s.add_argument('--model-files'); s.add_argument('--release', default='unreleased')
    s.add_argument('--hf-repo', default='<HF-OWNER>/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime')
    s.add_argument('--hf-revision', default=DEFAULT_REVISION)
    s = sub.add_parser('check'); s.set_defaults(fn=cmd_check)
    s.add_argument('--tree', required=True); s.add_argument('--sums', required=True); s.add_argument('--strict', action='store_true')
    s = sub.add_parser('sign-cmd'); s.set_defaults(fn=cmd_sign_cmd)
    s.add_argument('--sums', required=True); s.add_argument('--pubkey', default='~/.ssh/id_ed25519.pub')
    s = sub.add_parser('verify-sig'); s.set_defaults(fn=cmd_verify_sig)
    s.add_argument('--sums', required=True); s.add_argument('--sig', required=True)
    s.add_argument('--release-json', required=True); s.add_argument('--fingerprint')
    s = sub.add_parser('plan'); s.set_defaults(fn=cmd_plan)
    s.add_argument('--stage', required=True); s.add_argument('--hf-repo', required=True)
    s.add_argument('--github-repo', required=True); s.add_argument('--tag', required=True)
    s.add_argument('--write', action='store_true', help='also write UPLOAD-PLAN.md into the stage directory')
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
