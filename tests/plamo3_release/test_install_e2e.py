"""End-to-end through the REAL runtime_manifest / runtime_identity of a synthetic install tree (no NPU): placeholders sealed in the manifest,
resolved after the digest check, receipt overlay vs full hashing, tamper and moved-tree behaviour. Runs in a child interpreter per tree so that
plamo_root.ROOT is derived from that tree's own module location."""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import unittest

from _common import TmpDirMixin, APP, IS_WINDOWS

FILES = ("plamo_root.py", "plamo_errors.py", "runtime_manifest.py", "runtime_identity.py")

BUILD = textwrap.dedent(r'''
    import json, os, sys, hashlib
    sys.path.insert(0, sys.argv[1])
    import runtime_manifest as rm, runtime_identity as ri, plamo_root as pr
    root = pr.ROOT
    def rec(rel, token='@ROOT@'):
        r = rm.digest_file(root / rel)
        r['path'] = token + '\\' + rel.replace('/', '\\')
        return r
    manifest = dict(version=1, configuration={}, python_abi={'cache_tag': sys.implementation.cache_tag, 'version': [3, 9, 9]},
                    source_roots=['@ROOT@\\IRON\\src'], sources=[rec('IRON/src/a.py'), rec('IRON/src/b.py')],
                    origins=[rec('model/config.json', '@MODEL@'.replace('@MODEL@', '@ROOT@'))], artifacts={'x': rec('data/x.bin')})
    manifest['origins'][0]['path'] = '@MODEL@\\config.json'
    (root / 'plamo3-default').mkdir(exist_ok=True)
    rm.write_manifest(root / 'plamo3-default' / 'manifest.json', manifest)
    print('BUILT')
''')

VERIFY = textwrap.dedent(r'''
    import json, os, sys
    sys.path.insert(0, sys.argv[1])
    import runtime_manifest as rm, runtime_identity as ri, plamo_root as pr
    root = pr.ROOT
    out = {}
    m = rm.load(root / 'plamo3-default' / 'manifest.json')
    out['roots'] = m['source_roots']
    out['paths'] = [r['path'] for r in m['sources']] + [m['origins'][0]['path'], m['artifacts']['x']['path']]
    out['has_identity'] = [bool(r.get('ntfs_identity')) for r in m['sources']]
    try:
        lease = ri.IdentityLease(root / 'plamo3-default' / 'manifest.json', workers=1)
        lease.wait()
        out['stats'] = {k: lease.stats[k] for k in ('identity_matches', 'fallback_files', 'hashed_bytes', 'validated')}
        lease.close()
    except Exception as e:
        out['lease_error'] = type(e).__name__ + ': ' + str(e)
    print(json.dumps(out))
''')

RECEIPT = textwrap.dedent(r'''
    import json, os, sys, hashlib
    sys.path.insert(0, sys.argv[1])
    import runtime_manifest as rm, runtime_identity as ri, plamo_root as pr
    root = pr.ROOT
    rows = []
    for rel in ('IRON/src/a.py', 'IRON/src/b.py', 'model/config.json', 'data/x.bin'):
        with rm._open_retained(root / rel) as f:
            i = ri.maybe_identity(f)
        if i is None:
            print('NO-IDENTITY'); sys.exit(0)
        rows.append([rel, i['volume_serial'], i['file_id_128'], i['size'], i['last_write_time'], i['usn'], i['journal_id']])
    cert = root / 'plamo3-default' / 'r45-spec' / 'certificate.json'
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_bytes(b'{"kind":"x"}')
    (root / 'state').mkdir(exist_ok=True)
    (root / 'state' / 'install-receipt.json').write_text(json.dumps(dict(version=1, install_root=str(root), certificate_path=pr.CERTIFICATE_RELPATH,
        certificate_sha256=hashlib.sha256(cert.read_bytes()).hexdigest(), ids=rows)))
    print('RECEIPT', len(rows))
''')


def run(script, tree, *extra):
    app = tree / "IRON" / "iron" / "applications" / "plamo3"
    r = subprocess.run([sys.executable, "-c", script, str(app), *extra], capture_output=True, text=True, timeout=120)
    return r


class InstallE2E(TmpDirMixin, unittest.TestCase):
    def make_tree(self, name="inst"):
        root = (self.tmp / name).resolve()
        app = root / "IRON" / "iron" / "applications" / "plamo3"
        app.mkdir(parents=True)
        for f in FILES:
            shutil.copy(APP / f, app / f)
        (root / "IRON" / "src").mkdir()
        (root / "IRON" / "src" / "a.py").write_text("A = 1\n")
        (root / "IRON" / "src" / "b.py").write_text("B = 2\n")
        (root / "model").mkdir()
        (root / "model" / "config.json").write_text('{"x": 1}')
        (root / "data").mkdir()
        (root / "data" / "x.bin").write_bytes(b"\x01\x02" * 1000)
        return root

    def build(self, root):
        r = run(BUILD, root)
        self.assertIn("BUILT", r.stdout, r.stderr)

    def verify(self, root):
        r = run(VERIFY, root)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_placeholders_in_the_sealed_text_and_resolution_after_digest(self):
        root = self.make_tree()
        self.build(root)
        raw = (root / "plamo3-default" / "manifest.json").read_text()
        self.assertIn("@ROOT@", raw)
        self.assertIn("@MODEL@", raw)
        self.assertNotIn(str(root).replace("\\", "\\\\"), raw)  # no absolute path is sealed
        out = self.verify(root)
        self.assertEqual(out["roots"], [str(root / "IRON" / "src")])
        self.assertEqual(out["paths"][0], str(root / "IRON" / "src" / "a.py"))
        self.assertEqual(out["paths"][2], str(root / "model" / "config.json"))
        self.assertEqual(out["paths"][3], str(root / "data" / "x.bin"))
        self.assertNotIn("lease_error", out)
        self.assertTrue(out["stats"]["validated"])
        self.assertEqual(out["stats"]["fallback_files"], 4)  # no receipt: every file hashed
        self.assertEqual(out["stats"]["identity_matches"], 0)

    def test_the_same_sealed_manifest_works_in_a_copied_tree(self):
        root = self.make_tree("one")
        self.build(root)
        other = self.tmp / "two"
        shutil.copytree(root, other)
        out = self.verify(other.resolve())
        self.assertEqual(out["paths"][0], str((other.resolve()) / "IRON" / "src" / "a.py"))
        self.assertTrue(out["stats"]["validated"])

    def test_tampered_file_is_refused(self):
        root = self.make_tree()
        self.build(root)
        (root / "data" / "x.bin").write_bytes(b"\x01\x03" * 1000)
        out = self.verify(root)
        self.assertIn("StaleManifest", out.get("lease_error", ""))
        self.assertIn("content changed", out["lease_error"])

    def test_extra_source_file_is_refused(self):
        root = self.make_tree()
        self.build(root)
        (root / "IRON" / "src" / "c.py").write_text("C = 3\n")
        r = run(VERIFY, root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("source inventory changed", r.stderr)

    def test_manifest_text_change_is_refused_before_resolution(self):
        root = self.make_tree()
        self.build(root)
        p = root / "plamo3-default" / "manifest.json"
        p.write_bytes(p.read_bytes().replace(b"@ROOT@", b"@ROOT@\\\\..\\\\x", 1))  # path escape injected into the sealed text (bytes: no newline translation)
        r = run(VERIFY, root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("manifest content changed", r.stderr)

    def test_legacy_python_patch_level_in_manifest_is_not_compared(self):
        root = self.make_tree()
        self.build(root)  # BUILD writes version [3, 9, 9]
        out = self.verify(root)
        self.assertTrue(out["stats"]["validated"])

    @unittest.skipUnless(IS_WINDOWS, "NTFS identities")
    def test_receipt_skips_hashing_and_tamper_still_refused(self):
        root = self.make_tree()
        self.build(root)
        r = run(RECEIPT, root)
        if "NO-IDENTITY" in r.stdout:
            self.skipTest("no NTFS identity available (USN journal) on this volume")
        self.assertIn("RECEIPT 4", r.stdout, r.stderr)
        out = self.verify(root)
        self.assertEqual(out["has_identity"], [True, True])
        self.assertEqual(out["stats"]["identity_matches"], 4)
        self.assertEqual(out["stats"]["hashed_bytes"], 0)
        # a rewrite of a bound file changes its identity: hashing resumes for it and the change is caught
        (root / "data" / "x.bin").write_bytes(b"\x01\x03" * 1000)
        out = self.verify(root)
        self.assertIn("content changed", out["lease_error"])

    @unittest.skipUnless(IS_WINDOWS, "NTFS identities")
    def test_receipt_of_a_moved_tree_is_ignored_and_everything_is_hashed(self):
        root = self.make_tree("one")
        self.build(root)
        r = run(RECEIPT, root)
        if "NO-IDENTITY" in r.stdout:
            self.skipTest("no NTFS identity available")
        other = self.tmp / "two"
        shutil.copytree(root, other)
        out = self.verify(other.resolve())
        self.assertEqual(out["has_identity"], [False, False])
        self.assertEqual(out["stats"]["identity_matches"], 0)
        self.assertEqual(out["stats"]["fallback_files"], 4)
        self.assertTrue(out["stats"]["validated"])


if __name__ == "__main__":
    unittest.main()
