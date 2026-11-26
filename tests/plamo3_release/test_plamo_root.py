import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

from _common import TmpDirMixin, make_install, APP


class PathModel(TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.pr, self.root = make_install(self.tmp / "inst")

    def test_root_from_module_location(self):
        self.assertEqual(self.pr.ROOT, self.root)
        self.assertEqual(self.pr.MODEL, self.root / "model")

    def test_root_ignores_environment(self):
        saved = {k: os.environ.get(k) for k in ("PLAMO_ROOT", "PLAMO3_ROOT", "XRT_ROOT")}
        os.environ["PLAMO_ROOT"] = "/decoy"
        os.environ["PLAMO3_ROOT"] = "/decoy"
        os.environ["XRT_ROOT"] = "/decoy"
        try:
            pr, root = make_install(self.tmp / "inst2")
            self.assertEqual(pr.ROOT, root)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_resolve_root_and_model(self):
        r = self.pr.resolve_token("@ROOT@\\plamo3-runtime\\a.bin")
        self.assertEqual(Path(r), self.root / "plamo3-runtime" / "a.bin")
        r = self.pr.resolve_token("@ROOT@/plamo3-runtime/a.bin")
        self.assertEqual(Path(r), self.root / "plamo3-runtime" / "a.bin")
        r = self.pr.resolve_token("@MODEL@\\config.json")
        self.assertEqual(Path(r), self.root / "model" / "config.json")
        self.assertEqual(Path(self.pr.resolve_token("@ROOT@")), self.root)

    def test_backward_compat_absolute_unchanged(self):
        for s in ("C:\\Us" "ers\\someone\\npu\\x.bin", "/abs/path/x", "relative/x", "", "user@host.example", "a@b"):
            self.assertEqual(self.pr.resolve_token(s), s)

    def test_refusals(self):
        bad = [
            "C:\\x\\@ROOT@\\y",          # not at the start
            "x@MODEL@",                   # not at the start
            "@ROOT@\\..\\escape",         # parent component
            "@ROOT@/a/../b",
            "@ROOT@x",                    # no separator
            "@FOO@\\x",                   # unknown token
            "@ROOT@@ROOT@\\x",
        ]
        for s in bad:
            with self.assertRaises(self.pr.PathModelError, msg=s):
                self.pr.resolve_token(s)

    def test_rebase_tree_in_place(self):
        m = {"a": ["@ROOT@\\x", {"p": "@MODEL@\\y", "n": 3, "q": "plain"}], "b": "C:\\abs", "c": {"d": "@ROOT@/z/w"}}
        out = self.pr.rebase(m)
        self.assertIs(out, m)
        self.assertEqual(Path(m["a"][0]), self.root / "x")
        self.assertEqual(Path(m["a"][1]["p"]), self.root / "model" / "y")
        self.assertEqual(m["b"], "C:\\abs")
        self.assertEqual(Path(m["c"]["d"]), self.root / "z" / "w")

    def test_rebase_key_with_token_refused(self):
        with self.assertRaises(self.pr.PathModelError):
            self.pr.rebase({"@ROOT@\\x": 1})

    def test_rebase_error_class(self):
        class Boom(RuntimeError):
            pass

        with self.assertRaises(Boom):
            self.pr.rebase(["@ROOT@\\..\\x"], Boom)

    def test_rebase_noop_without_tokens(self):
        m = {"sources": [{"path": "C:\\a"}], "x": [1, 2.5, None, True]}
        before = json.dumps(m)
        self.pr.rebase(m)
        self.assertEqual(json.dumps(m), before)


class ConfigAbi(unittest.TestCase):
    def setUp(self):
        import plamo_root

        self.pr = plamo_root

    def test_legacy_keys_ignored_both_sides(self):
        live = {"NPU2": "0", "PLAMO3_RUNLIST": "1"}
        old = dict(live, XRT_ROOT="C:\\Xilinx\\XRT", SYSTEMROOT="C:\\Windows", XILINX_XRT="x")
        self.assertTrue(self.pr.config_equal(old, live))
        self.assertTrue(self.pr.config_equal(live, old))
        self.assertTrue(self.pr.config_equal(live, live))

    def test_other_keys_still_compared(self):
        self.assertFalse(self.pr.config_equal({"NPU2": "0"}, {"NPU2": "1"}))
        self.assertFalse(self.pr.config_equal({"IRON_PLAMO_X": "1"}, {}))
        self.assertFalse(self.pr.config_equal({}, {"IRON_PLAMO_X": "1"}))

    def test_abi_cache_tag_only(self):
        live = {"cache_tag": "cpython-313", "version": [3, 13, 99]}
        self.assertTrue(self.pr.abi_equal({"cache_tag": "cpython-313", "version": [3, 13, 13]}, live))
        self.assertTrue(self.pr.abi_equal({"cache_tag": "cpython-313"}, live))
        self.assertFalse(self.pr.abi_equal({"cache_tag": "cpython-312", "version": [3, 13, 13]}, live))
        self.assertFalse(self.pr.abi_equal(None, live))

    def test_check_platform(self):
        if sys.implementation.cache_tag == "cpython-313" and sys.maxsize > 2**32:
            self.pr.check_platform()
        old = sys.implementation
        import types

        class Fake:
            cache_tag = "cpython-312"

        sys.implementation = Fake()
        try:
            with self.assertRaises(self.pr.PathModelError) as cm:
                self.pr.check_platform()
            self.assertIn("Python 3.13 (64-bit) required", str(cm.exception))
        finally:
            sys.implementation = old

    def test_runtime_manifest_configuration_has_no_machine_keys(self):
        sys.path.insert(0, str(APP))
        import runtime_manifest

        saved = {k: os.environ.get(k) for k in ("XRT_ROOT", "SYSTEMROOT", "XILINX_XRT")}  # SYSTEMROOT is the real Windows variable: restore it exactly
        os.environ["XRT_ROOT"] = "x"
        os.environ["SYSTEMROOT"] = "y"
        os.environ["XILINX_XRT"] = "z"
        try:
            cfg = runtime_manifest.configuration()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertFalse({"XRT_ROOT", "SYSTEMROOT", "XILINX_XRT"} & set(cfg))


class Receipt(TmpDirMixin, unittest.TestCase):
    CERT_REL = "plamo3-default/r45-spec/certificate.json"

    def setUp(self):
        super().setUp()
        self.pr, self.root = make_install(self.tmp / "inst")
        cert = self.root / self.CERT_REL
        cert.parent.mkdir(parents=True)
        cert.write_bytes(b'{"kind":"cert"}')
        self.cert_sha = hashlib.sha256(cert.read_bytes()).hexdigest()
        (self.root / "state").mkdir()
        self.manifest = lambda: {
            "sources": [{"path": str(self.root / "IRON" / "a.py")}],
            "origins": [{"path": str(self.root / "plamo3-runtime" / "b.bin")}, {"path": "C:\\elsewhere\\c.bin"}],
            "artifacts": {"x": {"path": str(self.root / "plamo3-runtime" / "x.xclbin")}},
        }

    def write_receipt(self, **over):
        doc = dict(
            version=1,
            install_root=str(self.root),
            certificate_path=self.CERT_REL,
            certificate_sha256=self.cert_sha,
            ids=[
                ["IRON/a.py", 1, "aa", 10, 11, 12, 13],
                ["plamo3-runtime/b.bin", 1, "bb", 20, 21, 22, 23],
                ["plamo3-runtime/x.xclbin", 1, "cc", 30, 31, 32, 33],
            ],
        )
        doc.update(over)
        (self.root / "state" / "install-receipt.json").write_text(json.dumps(doc))
        self.pr.reset_receipt_cache()

    def test_overlay_adds_identity(self):
        self.write_receipt()
        m = self.manifest()
        n = self.pr.apply_receipt(m)
        self.assertEqual(n, 3)
        self.assertEqual(m["filesystem_identity_version"], 1)
        self.assertEqual(
            m["sources"][0]["ntfs_identity"],
            dict(volume_serial=1, file_id_128="aa", size=10, last_write_time=11, usn=12, journal_id=13),
        )
        self.assertNotIn("ntfs_identity", m["origins"][1])  # outside the root: untouched

    def test_missing_receipt_notifies_and_does_nothing(self):
        notes = []
        m = self.manifest()
        self.assertEqual(self.pr.apply_receipt(m, lambda msg, key=None: notes.append(msg)), 0)
        self.assertNotIn("filesystem_identity_version", m)
        self.assertTrue(notes and "receipt is missing" in notes[0])

    def test_legacy_identity_manifest_no_notice(self):
        notes = []
        m = self.manifest()
        m["filesystem_identity_version"] = 1
        self.pr.apply_receipt(m, lambda msg, key=None: notes.append(msg))
        self.assertEqual(notes, [])

    def test_stale_receipts(self):
        cases = [
            ("moved tree", dict(install_root=str(self.root / "elsewhere"))),
            ("other certificate", dict(certificate_sha256="0" * 64)),
            ("version", dict(version=2)),
            ("garbage ids", dict(ids=[["short"]])),
        ]
        for name, over in cases:
            self.write_receipt(**over)
            m = self.manifest()
            notes = []
            n = self.pr.apply_receipt(m, lambda msg, key=None: notes.append(msg))
            # a row with too few fields raises IndexError only when matched; it must never leak
            self.assertEqual(n, 0, name)
            self.assertNotIn("filesystem_identity_version", m, name)

    def test_unparsable_receipt(self):
        (self.root / "state" / "install-receipt.json").write_text("{not json")
        self.pr.reset_receipt_cache()
        m = self.manifest()
        self.assertEqual(self.pr.apply_receipt(m), 0)

    def test_never_overrides_existing_identity(self):
        self.write_receipt()
        m = self.manifest()
        m["sources"][0]["ntfs_identity"] = {"keep": True}
        self.pr.apply_receipt(m)
        self.assertEqual(m["sources"][0]["ntfs_identity"], {"keep": True})

    def test_prepare_rebases_then_overlays(self):
        self.write_receipt()
        m = {
            "sources": [{"path": "@ROOT@\\IRON\\a.py"}],
            "origins": [],
            "artifacts": {"x": {"path": "@ROOT@/plamo3-runtime/x.xclbin"}},
        }
        self.pr.prepare(m)
        self.assertEqual(Path(m["sources"][0]["path"]), self.root / "IRON" / "a.py")
        self.assertEqual(m["sources"][0]["ntfs_identity"]["file_id_128"], "aa")
        self.assertEqual(m["artifacts"]["x"]["ntfs_identity"]["file_id_128"], "cc")

    def test_receipt_is_never_written_by_runtime(self):
        before = set(os.listdir(self.root / "state"))
        self.pr.apply_receipt(self.manifest())
        self.assertEqual(set(os.listdir(self.root / "state")), before)
        import inspect

        src = inspect.getsource(self.pr)
        self.assertNotIn("write_text", src)
        self.assertNotIn("write_bytes", src)


if __name__ == "__main__":
    unittest.main()
