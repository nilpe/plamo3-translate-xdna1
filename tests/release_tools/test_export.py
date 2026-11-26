# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from common import *   # noqa: F401,F403
import export_public_tree as ex
import scrub_check as sc

ME = HANDLE
MAIL = 'release-bot@users.noreply.example.org'     # any address: the tool takes it as an argument


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo, self.base, self.tip = make_source_repo(self.tmp)
        self.allow = os.path.join(self.tmp, 'allow.txt')
        with open(self.allow, 'w') as f:
            f.write('[repo]\n- **\n+ LICENSE\n+ app/clean.py\n+ app/data.json\n+ iron/**\n+ README.md\n- **/*.sh\n[overlay]\n- **\n+ docs/*.md\n+ tools/x.py\n')
        self.overlay = os.path.join(self.tmp, 'overlay')
        os.makedirs(os.path.join(self.overlay, 'docs'))
        os.makedirs(os.path.join(self.overlay, 'tools'))
        open(os.path.join(self.overlay, 'docs', 'INSTALL.md'), 'w').write('install\n')
        open(os.path.join(self.overlay, 'docs', 'secret.txt'), 'w').write('not in the allow list\n')
        open(os.path.join(self.overlay, 'tools', 'x.py'), 'w').write('pass\n')

    def args(self, *extra):
        return ['--source', self.repo, '--rev', self.tip, '--allowlist', self.allow, '--overlay', self.overlay,
                '--author', ME, '--author-email', MAIL, '--date', '2026-02-02T00:00:00+00:00', *extra]

    def test_fork_point_detection(self):
        base, above = ex.find_fork_point(self.repo, self.tip, [USER, HANDLE])
        self.assertEqual(base, self.base)
        self.assertEqual(above, 3)

    def test_dry_run_writes_nothing(self):
        out = os.path.join(self.tmp, 'out')
        self.assertEqual(ex.main(self.args()), 0)
        self.assertFalse(os.path.exists(out))

    def test_build_history_and_content(self):
        out = os.path.join(self.tmp, 'out')
        self.assertEqual(ex.main(self.args('--build', '--out', out)), 0)
        log = git(out, 'log', '--format=%H %an <%ae> %s').splitlines()
        self.assertEqual(len(log), 3)                       # alice, bob, one squashed commit
        self.assertIn(self.base, log[1])
        self.assertIn('%s <%s>' % (ME, MAIL), log[0])
        self.assertEqual(git(out, 'rev-parse', 'HEAD~1').strip(), self.base)
        files = set(git(out, 'ls-files').split())
        self.assertEqual(files, {'LICENSE', 'app/clean.py', 'app/data.json', 'iron/op.py', 'README.md', 'docs/INSTALL.md', 'tools/x.py'})
        # CRLF preserved byte for byte (no autocrlf)
        with open(os.path.join(out, 'app', 'clean.py'), 'rb') as f:
            self.assertEqual(f.read(), b'x = 1\r\ny = 2\r\n')
        # the author-side history is not in the object database
        for sha in git(self.repo, 'rev-list', '%s..%s' % (self.base, self.tip)).split():
            r = subprocess.run(['git', '-C', out, 'cat-file', '-e', sha], capture_output=True)
            self.assertNotEqual(r.returncode, 0, 'private commit %s leaked' % sha[:8])
        self.assertEqual(git(out, 'remote').strip(), '')
        self.assertEqual(git(out, 'tag').strip(), '')
        # and the scrub gate agrees
        self.assertEqual(sc.main(['--files', out, '--git', out, '--upstream-base', self.base, '--author', ME, '--author-email', MAIL]), 0)

    def test_rewrite_rule_applies_to_docs_only_and_is_counted(self):
        open(os.path.join(self.overlay, 'docs', 'INSTALL.md'), 'w').write('see /srv/old\n')
        rules = os.path.join(self.tmp, 'rw.json')
        json.dump({'rules': [{'id': 'p', 'glob': 'docs/*.md', 'find': '/srv/old', 'replace': '<root>'}]}, open(rules, 'w'))
        out = os.path.join(self.tmp, 'out')
        self.assertEqual(ex.main(self.args('--rewrites', rules, '--build', '--out', out)), 0)
        self.assertEqual(open(os.path.join(out, 'docs', 'INSTALL.md')).read(), 'see <root>\n')

    def test_rewrite_of_sealed_file_is_refused(self):
        sums = os.path.join(self.tmp, 'SHA256SUMS')
        data = b'{"a": 1}\n'
        open(sums, 'w').write('%s  app/data.json\n' % hashlib.sha256(data).hexdigest())
        rules = os.path.join(self.tmp, 'rw.json')
        json.dump({'rules': [{'id': 'p', 'glob': 'app/*.json', 'find': '"a"', 'replace': '"b"'}]}, open(rules, 'w'))
        self.assertEqual(ex.main(self.args('--rewrites', rules, '--sums', sums)), 1)
        # without the rule the sealed file matches
        self.assertEqual(ex.main(self.args('--sums', sums)), 0)

    def test_sums_mismatch_is_refused(self):
        sums = os.path.join(self.tmp, 'SHA256SUMS')
        open(sums, 'w').write('%s  app/data.json\n' % ('0' * 64))
        self.assertEqual(ex.main(self.args('--sums', sums)), 1)

    def test_output_dir_must_be_empty_and_email_required(self):
        out = os.path.join(self.tmp, 'out')
        os.makedirs(out)
        open(os.path.join(out, 'x'), 'w').write('x')
        self.assertEqual(ex.main(self.args('--build', '--out', out)), 1)
        a = [x for x in self.args('--build', '--out', os.path.join(self.tmp, 'o2'))]
        i = a.index('--author-email')
        del a[i:i + 2]
        self.assertEqual(ex.main(a), 1)

    def test_size_limit(self):
        self.assertEqual(ex.main(self.args('--max-mb', '0.00001')), 1)

    def test_allowlist_parser(self):
        sec = ex.parse_allowlist('[repo]\n- **\n+ a/**\n- a/b/**\n+ a/b/keep.txt\n')
        got = ex.select(['a/x', 'a/b/y', 'a/b/keep.txt', 'c'], sec['repo'])
        self.assertEqual(got, ['a/x', 'a/b/keep.txt'])
        with self.assertRaises(ex.ExportError):
            ex.parse_allowlist('+ a\n')

    def test_draft_allowlist_parses(self):
        with open(os.path.join(B1, 'tools', 'public-tree.txt')) as f:
            sec = ex.parse_allowlist(f.read())
        self.assertTrue(sec['repo'] and sec['overlay'])


@unittest.skipUnless(shutil.which('ssh-keygen'), 'ssh-keygen not installed')
class SignedExportTests(unittest.TestCase):
    setUp_base = ExportTests.setUp
    args = ExportTests.args
    """--signature verifies SHA256SUMS.sig against the key pinned in the tree's release.json, then commits it; --bundle writes a git bundle."""

    def setUp(self):
        self.setUp_base()
        with open(self.allow, 'a') as f:
            f.write('+ SHA256SUMS\n+ release.json\n')
        key = os.path.join(self.tmp, 'k')
        ssh_run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', 'throwaway', '-f', key])
        pub = open(key + '.pub').read().split()
        import release_tool as rt
        signing = dict(public_key=' '.join(pub[:2]), fingerprint=rt.ssh_key_fingerprint(pub[0] + ' ' + pub[1]), principal=HANDLE, namespace='file')
        json.dump(dict(signing=signing), open(os.path.join(self.overlay, 'release.json'), 'w'))
        self.sums = os.path.join(self.overlay, 'SHA256SUMS')
        open(self.sums, 'w').write('%s  README.md\n' % ('a' * 64))
        self.key = key
        ssh_run(['ssh-keygen', '-q', '-Y', 'sign', '-n', 'file', '-f', key + '.pub', self.sums])

    def test_signed_export_commits_the_signature_and_writes_a_bundle(self):
        out, bundle = os.path.join(self.tmp, 'out'), os.path.join(self.tmp, 'out.bundle')
        self.assertEqual(ex.main(self.args('--signature', self.sums + '.sig', '--bundle', bundle, '--build', '--out', out)), 0)
        self.assertIn('SHA256SUMS.sig', git(out, 'ls-files').split())
        self.assertEqual(open(os.path.join(out, 'SHA256SUMS.sig'), 'rb').read(), open(self.sums + '.sig', 'rb').read())
        self.assertEqual(subprocess.run(['git', '-C', out, 'bundle', 'verify', bundle], capture_output=True).returncode, 0)
        self.assertEqual(len(git(out, 'log', '--format=%H').split()), 3)          # still one squashed commit on the upstream history

    def test_bad_signature_is_refused_before_anything_is_written(self):
        out = os.path.join(self.tmp, 'out')
        open(self.sums, 'a').write('%s  other\n' % ('b' * 64))                   # signature no longer matches the file
        self.assertEqual(ex.main(self.args('--signature', self.sums + '.sig', '--build', '--out', out)), 1)
        self.assertFalse(os.path.exists(out))


if __name__ == '__main__':
    unittest.main()
