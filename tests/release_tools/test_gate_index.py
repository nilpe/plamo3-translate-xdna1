# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""gate_index_rel.build_index on a synthetic records folder: the clean folder PASSES, every mutant FAILS."""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

os.environ['PLAMO3_INSTALL_TESTING'] = '1'
from common import *   # noqa: F401,F403
import gate_index_rel as gi

EPOCH = 'a' * 32


def find(folder, step):
    return [d for d in sorted(folder.iterdir()) if d.is_dir() and d.name.split('-', 2)[2].startswith(step + '-rel-')]


def put(folder, step, index=0, epoch=EPOCH, result='PASS', exit_code=0, extra=None, stamp=None):
    d = folder / ('%s-%s-rel-%d-%d' % (stamp or time.strftime('%Y%m%d-%H%M%S'), step, index, len(list(folder.iterdir()))))
    d.mkdir()
    (d / 'summary.json').write_text(json.dumps(dict(step=step, package='rel', index=index, epoch=epoch, result=result, exit_code=exit_code)))
    for name, content in (extra or {}).items():
        (d / name).write_text(content if isinstance(content, str) else json.dumps(content))
    return d


class Index(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.f = self.tmp / 'gates'
        self.f.mkdir()
        (self.f / 'freeze.json').write_text(json.dumps(dict(epoch=EPOCH, started_ns=int((time.time() - 60) * 1e9))))
        self.pid = 1000
        for s in gi.SUITE:
            name, _, idx = s.partition('@')
            self.add(name, int(idx or 0))

    def add(self, name, idx=0, **kw):
        extra = kw.pop('extra', None) or {}
        if name == 'startup':
            self.pid += 1
            extra['startup.json'] = dict(result='PASS', token_identity=True, pid=self.pid, proc_start_epoch_ms=1.7e12 + self.pid * 5000)
        elif name == 'export':
            extra['export.json'] = dict(result='PASS', audits=dict(a=dict(result='PASS')))
        elif name == 'fallback':
            extra['fallback.json'] = dict(result='PASS', cases=15)
        elif name not in ('freeze',):
            extra.setdefault(name + '.json', dict(result='PASS'))
        return put(self.f, name, idx, extra=extra, **kw)

    def fails(self):
        with self.assertRaises(AssertionError):
            gi.build_index(self.f)

    def test_clean_folder_passes(self):
        rep = gi.build_index(self.f)
        self.assertEqual((rep['result'], rep['epoch'], rep['listed_runs']), ('PASS', EPOCH, len(gi.SUITE)))

    def test_missing_step(self):
        shutil.rmtree(find(self.f, 'relgate-n7')[0])
        self.fails()

    def test_missing_duplicate_listed_step(self):                 # 'generate' and 'startup' are listed more than once
        shutil.rmtree(find(self.f, 'generate')[0])
        self.fails()

    def test_failed_newest_record(self):
        self.add('relgate-n3', result='FAIL', exit_code=1)
        self.fails()

    def test_foreign_epoch_record_does_not_count(self):
        d = find(self.f, 'relgate-n4')[0]
        row = json.loads((d / 'summary.json').read_text()); row['epoch'] = 'b' * 32; (d / 'summary.json').write_text(json.dumps(row))
        self.fails()

    def test_stale_record(self):
        d = find(self.f, 'relgate-n5')[0]
        old = time.time() - 3600
        os.utime(d / 'summary.json', (old, old))
        self.fails()

    def test_unparseable_record(self):
        (find(self.f, 'cli-check')[0] / 'summary.json').write_text('{not json')
        self.fails()

    def test_unparseable_record_of_a_step_not_needed_still_fails(self):
        put(self.f, 'extra-step')
        (find(self.f, 'extra-step')[0] / 'summary.json').write_text('')
        self.fails()

    def test_unparseable_step_output(self):
        (find(self.f, 'relgate-n7')[0] / 'relgate-n7.json').write_text('{')
        self.fails()

    def test_step_output_not_pass(self):
        (find(self.f, 'relgate-n7')[0] / 'relgate-n7.json').write_text(json.dumps(dict(result='FAIL')))
        self.fails()

    def test_duplicate_pid_and_creation_time(self):
        ds = find(self.f, 'startup')
        a, b = (json.loads((d / 'startup.json').read_text()) for d in ds[:2])
        b['pid'], b['proc_start_epoch_ms'] = a['pid'], a['proc_start_epoch_ms']
        (ds[1] / 'startup.json').write_text(json.dumps(b))
        self.fails()

    def test_fallback_case_count_and_export_audit(self):
        d = find(self.f, 'fallback')[0]
        (d / 'fallback.json').write_text(json.dumps(dict(result='PASS', cases=14)))
        self.fails()

    def test_suite_has_the_release_step_list(self):
        for step in ('freeze', 'relgate-n1b', 'relgate-n15', 'sentinel-proof', 'fast-exit-soak-check', 'confidence-run', 'hybrid-wait-gate'):
            self.assertIn(step, gi.SUITE)
        self.assertFalse([s for s in gi.SUITE if s.startswith('hoist-gate')])


if __name__ == '__main__':
    unittest.main()
