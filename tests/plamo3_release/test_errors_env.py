import io
import json
import os
import sys
import unittest

from _common import TmpDirMixin, APP, load_isolated, IS_WINDOWS
import plamo_errors as E


class ErrorModel(unittest.TestCase):
    def test_exit_code_table(self):
        self.assertEqual([E.OK, E.INTERNAL, E.USAGE, E.INSTALL, E.DEGRADED, E.NPU, E.ENVIRONMENT, E.INTERRUPTED], [0, 1, 2, 3, 4, 5, 6, 130])

    def test_classify(self):
        class StaleManifest(RuntimeError):
            pass

        c = E.classify(StaleManifest("runtime source inventory changed; rebuild the bundle"))
        self.assertEqual(c.code, E.INSTALL)
        self.assertIn("extra or missing source file", c.message)
        self.assertIn("install.ps1 -Verify", c.message)
        c = E.classify(RuntimeError("Failed to create context virtual (0xc01e0009)"))
        self.assertEqual(c.code, E.NPU)
        e = FileNotFoundError(2, "No such file", "C:\\x\\y.bin")
        self.assertEqual(E.classify(e).code, E.INSTALL)
        self.assertEqual(E.classify(ImportError("DLL load failed while importing pyxrt")).code, E.ENVIRONMENT)
        self.assertEqual(E.classify(KeyError("zzz")).code, E.INTERNAL)
        self.assertEqual(E.classify(E.PromptTooLong(2301, 2048)).code, E.USAGE)

    def test_report_is_one_line_no_traceback(self):
        buf = io.StringIO()
        try:
            raise ValueError("secret internal detail")
        except ValueError as exc:
            rc = E.report(exc, stream=buf)
        out = buf.getvalue()
        self.assertEqual(rc, E.INTERNAL)
        self.assertEqual(out.count("\n"), 1)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("secret internal detail", out)

    def test_report_traceback_only_with_debug(self):
        os.environ["PLAMO3_DEBUG"] = "1"
        try:
            buf = io.StringIO()
            try:
                raise ValueError("x")
            except ValueError as exc:
                E.report(exc, stream=buf)
            self.assertIn("Traceback", buf.getvalue())
        finally:
            os.environ.pop("PLAMO3_DEBUG")

    def test_prompt_too_long_message(self):
        self.assertIn("2301 tokens", E.PromptTooLong(2301, 2048).message)
        self.assertIn("2048", E.PromptTooLong(2301, 2048).message)

    def test_note_once(self):
        buf = io.StringIO()
        E.note("PLaMo: note: a", key="t-once", stream=buf)
        E.note("PLaMo: note: a", key="t-once", stream=buf)
        self.assertEqual(buf.getvalue().count("note: a"), 1)


class Environment(TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        import environment

        self.env = environment

    def test_collect_never_raises_and_has_python(self):
        rec = self.env.collect("cheap")
        self.assertEqual(rec["python"]["cache_tag"], sys.implementation.cache_tag)
        self.assertIn("files", rec)

    def test_compare_unknown_versions_warn(self):
        rec = {
            "files": {"xrt_coreutil": {"file_version": "32.00.20101.9999"}},
            "windows": {"build": "10.0.99999"},
            "python": {"version": "3.13.99"},
            "packages": {"numpy": "9.9.9", "ml_dtypes": "0.6.0"},
        }
        known = {"xrt_coreutil": ["32.00.20101.3761"], "windows_build": ["10.0.26300"], "python": ["3.13.13"], "numpy": ["2.4.6"], "ml_dtypes": ["0.6.0"]}
        w = self.env.compare(rec, known)
        self.assertEqual(len(w), 4)
        self.assertTrue(all(x.startswith("PLaMo: note:") for x in w))
        self.assertIn("32.00.20101.9999", w[0])

    def test_compare_known_is_silent(self):
        rec = {"files": {"xrt_coreutil": {"file_version": "32.00.20101.3761"}}, "windows": {"build": "10.0.26300"}, "python": {"version": "3.13.13"}, "packages": {"numpy": "2.4.6"}}
        known = {"xrt_coreutil": ["32.00.20101.3761"], "windows_build": ["10.0.26300"], "python": ["3.13.13"], "numpy": ["2.4.6"]}
        self.assertEqual(self.env.compare(rec, known), [])

    def test_compare_without_list_is_silent(self):
        self.assertEqual(self.env.compare({"files": {"xrt_coreutil": {"file_version": "1"}}}, None), [])

    def test_check_warns_once_and_logs_changes(self):
        known = self.tmp / "known.json"
        known.write_text(json.dumps({"python": ["0.0.0"]}))
        state = self.tmp / "state"
        state.mkdir()
        self.env.STATE, self.env.RECORD, self.env.CHANGES = state, state / "env-record.json", state / "logs" / "env-changes.jsonl"
        buf = io.StringIO()
        self.env.check(known, stream=buf)
        self.env.check(known, stream=buf)
        self.assertEqual(buf.getvalue().count("Python"), 1)  # once per process
        self.assertTrue(self.env.RECORD.exists())
        n = len(self.env.CHANGES.read_text().splitlines())
        self.env.check(known, stream=io.StringIO())
        self.assertEqual(len(self.env.CHANGES.read_text().splitlines()), n)  # unchanged record: nothing appended

    def test_check_survives_garbage(self):
        k = self.tmp / "bad.json"
        k.write_text("{nope")
        self.assertIsNotNone(self.env.check(k, stream=io.StringIO(), write_state=False))

    def test_report_text_hides_profile_dir(self):
        text = self.env.report_text()
        home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
        if os.environ.get("USERPROFILE"):
            self.assertNotIn(home, text)

    @unittest.skipUnless(IS_WINDOWS, "Windows only")
    def test_windows_facts(self):
        rec = self.env.collect("full")
        self.assertTrue(rec["windows"]["build"])
        self.assertTrue(rec["files"]["xrt_coreutil"]["file_version"] or rec["files"]["xrt_coreutil"].get("missing"))


if __name__ == "__main__":
    unittest.main()
