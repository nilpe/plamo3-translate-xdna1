# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
import json
import os
import subprocess
import tempfile
import unittest

from common import *   # noqa: F401,F403
import scrub_check as sc


def rules(data, name='x.txt'):
    return set(sc.scan_bytes(data if isinstance(data, bytes) else data.encode('utf-8'), name))


class RuleTests(unittest.TestCase):
    def test_positive_samples(self):
        cases = {
            'user-name': USER, 'author-mail': HANDLE + '@example.org x', 'noreply-mail': NOREPLY,
            'private-ip': 'host ' + HOSTIP + ' up', 'cluster-host': 'job on ' + NODE, 'linux-home': '/ho' + 'me/someone/data',
            'mac-home': '/Us' + 'ers/someone/x', 'win-profile': WINPROFILE, 'session-url': 'https://claude.' + 'ai/code/abc',
            'agent-trailer': 'Co-Authored-By' + ': Claude <x>', 'token-hf': 'hf_' + 'A' * 34, 'token-github': 'ghp_' + 'b' * 36,
            'private-key': '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----', 'ssh-config': '~/.ssh/' + 'config', 'internal-tool': 'co' + 'dex' + '_tasks',
            'email': 'someone' + '@corp.example.com.jp', 'other-user': 'another ' + "user's CPU use was 22%",
        }
        for rule, text in cases.items():
            self.assertIn(rule, rules(text), rule)

    def test_negative_samples(self):
        ok = ['Windows 10.0.26300.0 build', 'driver 32.0.20101.3760', 'a = w@x.T', 'PLaMo 3 Translate Built with PLaMo',
              'contact: support' + '@amd.com', 'see https://huggingface.co/pfnet/plamo-3-translate-0.8b', '%USERPROFILE%\\npu',
              '172.32' + '.0.1', '10.0.26300', 'C:\\PLaMo3-NPU\\venv', 'ip 8.8.8.8', 'numpy@2.4.6']
        for t in ok:
            self.assertEqual(rules(t), set(), t)

    def test_private_ip_ranges(self):
        for ip in ('10.' + '1.2.3', '172.' + '16.0.9', '172.' + '31.255.1', '192.' + '168.0.1'):
            self.assertIn('private-ip', rules('x ' + ip + ' y'), ip)

    def test_json_escaped_and_utf16(self):
        esc = json.dumps({'p': WINPROFILE + '\\x'})
        self.assertIn('win-profile', rules(esc, 'a.json'))
        self.assertIn('user-name', rules(json.dumps({'p': '\\u0079' + USER[1:]}), 'a.json'))
        self.assertIn('user-name', rules(('hello ' + USER).encode('utf-16'), 'a.txt'))
        blob = b'\x00' * 100 + ('/' + 'home').encode('utf-16-le') + b'\x00' + USER.encode('utf-16-le') + b'\x01\x02' * 50
        self.assertIn('user-name', rules(blob, 'x.bin'))

    def test_counts_are_occurrences(self):
        res = sc.scan_bytes((USER + ' ') * 7 + 'x', 'a.txt')
        self.assertEqual(res['user-name'][1], 7)

    def test_unicode_text_with_japanese_is_fine(self):
        self.assertEqual(rules('今日はいい天気ですね。 Thank you.'), set())


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_fast_binary_prefilter_keeps_every_text_file_and_flags_binaries_with_anchors(self):
        import random
        rnd = random.Random(1)
        noise = bytes(rnd.randrange(256) for _ in range(5000))
        self.put('clean.bin', noise)
        self.put('dirty.bin', noise + ('C:' + chr(92) + 'Use' + 'rs' + chr(92) + 'x').encode() + noise)
        self.put('mail.txt', 'write to foo' + '@' + 'mailhost.com\n')
        r = sc.scan_tree(self.tmp, fast=True)
        got = sorted((f['path'], f['rule']) for f in r['findings'])
        self.assertIn(('dirty.bin', 'win-profile'), got)
        self.assertIn(('mail.txt', 'email'), got)
        self.assertFalse([g for g in got if g[0] == 'clean.bin'])

    def test_windowed_fast_path_reports_what_the_full_pass_reports(self):
        import random
        rnd = random.Random(2)
        old_block = sc.SCAN_BLOCK
        sc.SCAN_BLOCK = 64 * 1024                      # exercise the block boundaries and the carry with small files
        try:
            needles = [('C:' + chr(92) + 'Use' + 'rs' + chr(92) + 'x').encode(), ('foo' + '@' + 'mailhost.com').encode(), ('y' + 'kogi').encode(), ('y' + 'kogi').encode().decode().upper().encode(),
                       ('hf' + '_' + 'a' * 34).encode(), ('y' + 'kogi').encode().decode('latin-1').encode('utf-16le')]
            for i, nd in enumerate(needles):
                for j, off in enumerate((0, 1, 64 * 1024 - 3, 64 * 1024, 64 * 1024 + 3, 64 * 1024 + 1024 - 2, 150000 - len(nd))):
                    buf = bytearray(rnd.randrange(1, 256) for _ in range(150000))
                    buf[off:off + len(nd)] = nd
                    self.put('n%d_%d.bin' % (i, j), bytes(buf))
            self.put('clean.bin', bytes(rnd.randrange(1, 256) for _ in range(150000)))
            full = sc.scan_tree(self.tmp, deep=True)
            fast = sc.scan_tree(self.tmp, deep=True, fast=True)
        finally:
            sc.SCAN_BLOCK = old_block
        key = lambda r: sorted((f['path'], f['rule'], f['count']) for f in r['findings'])
        self.assertEqual(key(full), key(fast))
        self.assertGreater(len(key(fast)), 30)

    def put(self, rel, data):
        p = os.path.join(self.tmp, *rel.split('/'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(data if isinstance(data, bytes) else data.encode())

    def test_clean_tree_passes_and_dirty_fails(self):
        self.put('a/b.py', 'print("hi")\n')
        self.assertEqual(sc.main(['--files', self.tmp]), 0)
        self.put('a/c.json', json.dumps({'root': WINPROFILE}))
        self.assertEqual(sc.main(['--files', self.tmp]), 1)

    def test_filename_is_scanned(self):
        self.put('notes_' + USER + '.txt', 'ok')
        r = sc.scan_tree(self.tmp)
        self.assertTrue(any(f['rule'] == 'user-name' and 'FILENAME' in f['snippet'] for f in r['findings']))

    def test_large_skipped_unless_deep(self):
        self.put('big.bin', b'\x00' * 3000 + USER.encode() + b'\x00' * 3000)
        r = sc.scan_tree(self.tmp, deep=False, max_bytes=1000)
        self.assertEqual(len(r['skipped_large']), 1)
        self.assertEqual(r['findings'], [])
        r = sc.scan_tree(self.tmp, deep=True, max_bytes=1000)
        self.assertEqual([f['rule'] for f in r['findings']], ['user-name'])

    def test_match_across_chunk_boundary(self):
        old = sc.scan_file
        data = b'a' * (32 * 1024 * 1024 - 2) + USER.encode() + b'b' * 100
        self.put('big.txt', data)
        r = sc.scan_tree(self.tmp, deep=True, max_bytes=10)
        self.assertEqual([f['rule'] for f in r['findings']], ['user-name'])

    def test_allow_list(self):
        self.put('docs/x.md', 'see /ho' + 'me/runner/work for CI\n')
        self.assertEqual(sc.main(['--files', self.tmp]), 1)
        allow = os.path.join(self.tmp, '..', 'allow.json')
        with open(allow, 'w') as f:
            json.dump([{'rule': 'linux-home', 'path': 'docs/*.md', 'contains': '/ho' + 'me/runner', 'reason': 'CI doc'}], f)
        self.assertEqual(sc.main(['--files', self.tmp, '--allow', allow]), 0)

    def test_warning_does_not_fail_unless_asked(self):
        self.put('a.md', 'my account ' + HANDLE + '0555\n')
        self.assertEqual(sc.main(['--files', self.tmp]), 0)
        self.assertEqual(sc.main(['--files', self.tmp, '--warn-fails']), 1)

    def test_this_tool_and_tests_are_clean(self):
        import shutil
        for sub in ('tools', 'tests'):
            dst = os.path.join(self.tmp, 'copy_' + sub)
            # tools/author holds the maintainer-side exporter (it names the maintainer's tree on purpose) and is not part of the public tree
            shutil.copytree(os.path.join(B1, sub), dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'author'))
            self.assertEqual(sc.main(['--files', dst]), 0, sub)


class GitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, 'r')
        os.makedirs(self.repo)
        subprocess.run(['git', 'init', '-q', '-b', 'main', self.repo], check=True)
        self.base = commit(self.repo, 'Alice', 'alice@amd.com', 'upstream', {'a.txt': 'x\n'})
        self.mine = HANDLE + '@users.noreply.' + 'github.com'

    def scan(self):
        return sc.scan_git(self.repo, self.base, HANDLE, self.mine)['findings']

    def test_clean_release_commit(self):
        commit(self.repo, HANDLE, self.mine, 'release', {'b.txt': 'y\n'})
        self.assertEqual(self.scan(), [])

    def test_wrong_identity(self):
        commit(self.repo, USER, NOREPLY, 'release', {'b.txt': 'y\n'})
        rules_ = {f['rule'] for f in self.scan()}
        self.assertIn('git-identity', rules_)
        self.assertIn('user-name', rules_)

    def test_trailer_and_remote_and_hook(self):
        commit(self.repo, HANDLE, self.mine, 'release\n\nClaude' + '-Session: https://claude.' + 'ai/code/x', {'b.txt': 'y\n'})
        git(self.repo, 'remote', 'add', 'origin', 'https://example.org/x.git')
        hook = os.path.join(self.repo, '.git', 'hooks', 'pre-commit')
        os.makedirs(os.path.dirname(hook), exist_ok=True)
        open(hook, 'w').write('#!/bin/sh\n')
        rules_ = {f['rule'] for f in self.scan()}
        self.assertTrue({'session-url', 'git-remote', 'git-hook'} <= rules_)

    def test_secret_in_blob(self):
        commit(self.repo, HANDLE, self.mine, 'release', {'b.txt': 'token = ' + 'hf_' + 'Z' * 36 + '\n'})
        self.assertIn('token-hf', {f['rule'] for f in self.scan()})

    def test_unreachable_objects_are_flagged(self):
        commit(self.repo, HANDLE, self.mine, 'release', {'b.txt': 'y\n'})
        git(self.repo, 'reset', '-q', '--hard', self.base)
        git(self.repo, 'reflog', 'expire', '--expire=now', '--all')
        self.assertIn('unreachable-objects', {f['rule'] for f in self.scan()})

    def test_upstream_email_is_allowed_but_ours_is_not(self):
        self.assertEqual([f for f in self.scan() if f['rule'] == 'email'], [])
        commit(self.repo, HANDLE, self.mine, 'release by someone' + '@mycorp.net', {})
        self.assertIn('email', {f['rule'] for f in self.scan()})


if __name__ == '__main__':
    unittest.main()
