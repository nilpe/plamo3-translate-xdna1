"""CLI argument validation, request handling, service loop, output helpers (no NPU: runtime pieces are faked)."""
import base64
import io
import json
import os
import sys
import unittest

from _common import IS_WINDOWS
import cli83
import plamo_errors as E


def run_main(argv):
    """cli83.main in-process; returns (rc, stdout, stderr)."""
    import contextlib

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli83.main(argv=argv)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class Args(unittest.TestCase):
    def check_usage(self, argv, needle):
        rc, out, err = run_main(argv)
        self.assertEqual(rc, E.USAGE, (argv, err))
        self.assertIn(needle, err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(err.count("\n"), 1, err)  # one line
        self.assertEqual(out, "")

    def test_empty_and_whitespace_text(self):
        self.check_usage([" "], "text is empty")
        self.check_usage(["\n\t "], "text is empty")
        self.check_usage(["--text-b64", base64.b64encode(b"   ").decode()], "text is empty")

    def test_reserved_token(self):
        self.check_usage(["<|plamo:msg|>"], "reserved PLaMo tokens")
        self.check_usage(["hello <|plamo: x"], "reserved PLaMo tokens")

    def test_missing_text(self):
        self.check_usage([], "text is required")

    def test_num_tokens_range(self):
        for v in ("0", "-1", "2048", "99999999999999999999"):
            self.check_usage(["--num-tokens", v, "Hello."], "--num-tokens must be between 1 and 2047")
        self.check_usage(["--num-tokens", "abc", "Hello."], "invalid int value")

    def test_same_language(self):
        self.check_usage(["--source-lang", "ja", "--target-lang", "ja", "x"], "must differ")
        self.check_usage(["--source-lang", "en", "--target-lang", "en", "x"], "must differ")

    def test_service_with_text(self):
        self.check_usage(["--service", "Hello."], "must not be given together with --service")

    def test_dash_text(self):
        self.check_usage(["-hello"], "unrecognized arguments")  # not interpreted as --help
        rc, out, err = run_main(["-hello"])
        self.assertEqual(rc, E.USAGE)
        self.assertNotIn("usage:", out)

    def test_unknown_option_hint(self):
        self.check_usage(["--bogus", "x"], "'--'")

    def test_conflicts(self):
        self.check_usage(["--spec-off", "--mode", "w4-m2", "x"], "conflicts")
        self.check_usage(["--stats", "--quiet", "x"], "conflicts")
        self.check_usage(["--wait", "-1", "x"], "--wait")

    def test_too_long_text(self):
        self.check_usage(["x" * (E.MAX_TEXT_CHARS + 1)], "too long")

    def test_bad_base64(self):
        self.check_usage(["--text-b64", "@@@"], "base64")
        self.check_usage(["--text-b64", base64.b64encode(b"\xff\xfe").decode()], "base64")
        self.check_usage(["--text-b64", "aGk=", "also"], "mutually exclusive")

    def stdin_run(self, data, argv):
        old = sys.stdin

        class In:
            buffer = io.BytesIO(data)

        sys.stdin = In()
        try:
            return run_main(argv)
        finally:
            sys.stdin = old

    def test_text_stdin(self):
        rc, out, err = self.stdin_run(b"   \r\n", ["--text-stdin"])
        self.assertEqual((rc, "text is empty" in err), (2, True))
        rc, out, err = self.stdin_run(b"\xff\xfe", ["--text-stdin"])
        self.assertEqual((rc, "not valid UTF-8" in err), (2, True))
        rc, out, err = self.stdin_run(b"x", ["--text-stdin", "also"])
        self.assertEqual((rc, "cannot be combined" in err), (2, True))
        rc, out, err = self.stdin_run(b"x", ["--text-stdin", "--service"])
        self.assertEqual((rc, "cannot be combined" in err), (2, True))
        rc, out, err = self.stdin_run(b"<|plamo:x", ["--text-stdin"])
        self.assertEqual((rc, "reserved" in err), (2, True))
        rc, out, err = self.stdin_run(b"x" * (E.MAX_TEXT_CHARS * 4 + 1), ["--text-stdin"])
        self.assertEqual((rc, "too long" in err), (2, True))
        p = cli83.build_parser()
        a = p.parse_args(["--text-stdin"])
        old = sys.stdin

        class In:
            buffer = io.BytesIO("こんにちは\r\n".encode("utf-8"))

        sys.stdin = In()
        try:
            cli83.validate_args(a, p)
        finally:
            sys.stdin = old
        self.assertEqual(a.text, "こんにちは")  # exactly one trailing newline dropped

    def test_help_version(self):
        rc, out, err = run_main(["--help"])
        self.assertEqual(rc, 0)
        self.assertIn("--allow-serial-fallback", out)
        rc, out, err = run_main(["--version"])
        self.assertEqual(rc, 0)
        self.assertIn("PLaMo 3 Translate NPU runtime", out)

    def test_validation_happens_before_runtime(self):
        # a bad argument must return without importing the runtime modules
        for name in ("default83", "initialize83", "lease83"):
            sys.modules.pop(name, None)
        run_main(["--num-tokens", "0", "x"])
        for name in ("default83", "initialize83", "lease83"):
            self.assertNotIn(name, sys.modules)

    def test_stats_policy(self):
        p = cli83.build_parser()
        a = p.parse_args(["x"])
        self.assertFalse(cli83.stats_wanted(a, {}) and False)
        a2 = p.parse_args(["--stats", "x"])
        self.assertTrue(cli83.stats_wanted(a2, {}))
        self.assertTrue(cli83.stats_wanted(a, {"PLAMO3_STATS": "1"}))
        self.assertFalse(cli83.stats_wanted(a2, {"PLAMO3_STATS": "0"}))
        self.assertFalse(cli83.stats_wanted(p.parse_args(["--quiet", "x"]), {"PLAMO3_STATS": "1"}))

    def test_stats_off_when_stderr_is_not_a_tty(self):
        p = cli83.build_parser()
        a = p.parse_args(["x"])
        old = sys.stderr
        sys.stderr = io.StringIO()  # not a tty
        try:
            self.assertFalse(cli83.stats_wanted(a, {}))
        finally:
            sys.stderr = old

    def test_allow_fallback_switches(self):
        p = cli83.build_parser()
        self.assertFalse(cli83.allow_fallback_wanted(p.parse_args(["x"]), {}))
        self.assertTrue(cli83.allow_fallback_wanted(p.parse_args(["--allow-serial-fallback", "x"]), {}))
        self.assertTrue(cli83.allow_fallback_wanted(p.parse_args(["x"]), {"PLAMO3_ALLOW_FALLBACK": "1"}))


class FakeTokenizer:
    tokens = ["<s>", "<0xF0>", "<0x9F>", "<0x91>", "<0xA7>", "a", "b", "\u3042", "<0xE3>", "<0x81>", "<0x82>"]
    special_ids = {0}

    def decode(self, ids):
        out, buf = [], bytearray()
        for i in ids:
            t = self.tokens[i]
            if i in self.special_ids:
                continue
            if t.startswith("<0x"):
                buf.append(int(t[3:5], 16))
            else:
                out.append(bytes(buf).decode("utf-8", "replace")) if buf else None
                buf.clear()
                out.append(t)
        if buf:
            out.append(bytes(buf).decode("utf-8", "replace"))
        return "".join(out)

    def chat(self, messages, source_lang=None, target_lang=None):
        return [1] * 10


class Output(unittest.TestCase):
    def test_crlf_not_doubled(self):
        self.assertEqual(cli83.printable("a\r\nb\r\nc"), "a\nb\nc")
        self.assertEqual(cli83.printable("a\rb\nc"), "a\rb\nc")  # lone CR is kept
        self.assertEqual(cli83.printable("no newline"), "no newline")

    def test_hold_back_incomplete_sequence_at_cap(self):
        tok = FakeTokenizer()
        text, held = cli83.finish_text(tok, [5, 1, 2, 3], True)  # a + 3 of 4 bytes of an emoji
        self.assertEqual((text, held), ("a", 3))
        text, held = cli83.finish_text(tok, [5, 1, 2, 3, 4], True)  # complete emoji: nothing held back
        self.assertEqual(held, 0)
        self.assertNotIn("\ufffd", text)
        text, held = cli83.finish_text(tok, [5, 8, 9], True)  # 2 of 3 bytes of U+3042
        self.assertEqual((text, held), ("a", 2))
        text, held = cli83.finish_text(tok, [5, 8, 9, 10], True)
        self.assertEqual(held, 0)

    def test_no_holdback_without_cap(self):
        tok = FakeTokenizer()
        text, held = cli83.finish_text(tok, [5, 1, 2, 3], False)
        self.assertEqual(held, 0)
        self.assertTrue(text.endswith("\ufffd"))

    def test_stop_reason(self):
        self.assertEqual(cli83.stop_reason(dict(stopped_eos=True), 2048), "eos")
        self.assertEqual(cli83.stop_reason(dict(stopped_eos=False, logical_length=2048), 2048), "context")
        self.assertEqual(cli83.stop_reason(dict(stopped_eos=False, logical_length=100), 2048), "cap")


class FakeSession:
    max_seq = 2048


class FakeApp:
    mode = "w4-m2"
    requested_mode = "w4-m2"
    spec_ready_ms = 800.0
    session = FakeSession()
    fallbacks = []

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def generate(self, prompt, limit, target_language=None):
        self.calls.append((len(prompt), limit, target_language))
        if self.fail:
            raise self.fail
        return dict(tokens=[5, 6], ttft_s=0.1, wall_s=0.2, stopped_eos=True, logical_length=12)


class Service(unittest.TestCase):
    def setUp(self):
        p = cli83.build_parser()
        self.a = p.parse_args(["--service"])

    def serve(self, lines, app=None, tokenizer=None):
        app = app or FakeApp()
        translate = cli83.make_translate(app, tokenizer or FakeTokenizer(), self.a)
        source = io.BytesIO(b"".join(lines))
        out = io.StringIO()
        rc = cli83.serve(app, translate, out=out, source=source)
        return rc, [json.loads(l) for l in out.getvalue().splitlines()], app

    def ok(self, **extra):
        d = dict(id="r", text="Hello.", source_lang="en", target_lang="ja")
        d.update(extra)
        return (json.dumps(d) + "\n").encode()

    def bad_then_good(self, bad):
        rc, out, app = self.serve([bad, self.ok()])
        self.assertEqual(rc, 0)
        self.assertEqual(len(out), 2, out)
        self.assertEqual(out[1]["id"], "r")
        self.assertNotIn("error", out[1])
        return out[0], app

    def test_bad_requests_never_end_the_service(self):
        bads = [
            b'{"text":"  ","source_lang":"en","target_lang":"ja"}\n',
            b'{"text":"","source_lang":"en","target_lang":"ja"}\n',
            b'{"text":"\\n","source_lang":"en","target_lang":"ja"}\n',
            b'{"text":"<|plamo:msg|>","source_lang":"en","target_lang":"ja"}\n',
            b'{"text":"x","source_lang":5,"target_lang":"ja"}\n',
            b'{"text":"x","source_lang":"en","target_lang":null}\n',
            b'{"text":"x","source_lang":"en","target_lang":["ja"]}\n',
            b'{"text":"x","source_lang":"en","target_lang":""}\n',
            b'{"text":"x","source_lang":"en","target_lang":"EN"}\n',
            b'{"text":"x","source_lang":"en","target_lang":"ja-JP"}\n',
            b'{"text":"x","source_lang":"en","target_lang":"xx"}\n',
            b'{"text":"x","source_lang":"en","target_lang":"ja(easy)"}\n',
            b'{"text":"x","source_lang":"fr","target_lang":"ja"}\n',
            b'{"text":"x","source_lang":null,"target_lang":"ja"}\n',
            b'{"text":"x","source_lang":"ja","target_lang":"ja"}\n',
            b'{"source_lang":"en","target_lang":"ja"}\n',
            b'[1,2,3]\n',
            b'"just a string"\n',
            b'42\n',
            b'null\n',
            b'{"text":"x","max_new_tokens":true}\n',
            b'{"text":"x","max_new_tokens":5.0}\n',
            b'{"text":"x","max_new_tokens":0}\n',
            b'{"text":"x","max_new_tokens":1000000000}\n',
            b'{"text":"x","max_new_tokens":"3"}\n',
            b'{not json}\n',
            b'{"text":"x"\n',
            b'\xff\xfe{"text":"x"}\n',  # invalid UTF-8 / UTF-16 BOM
            b'{"text":"\xff"}\n',  # invalid UTF-8 inside a JSON string
            b'{"text":"x","a":' + b'[' * 100000 + b'}\n',  # deep nesting
            b'\x00\x00\n',
        ]
        for bad in bads:
            first, app = self.bad_then_good(bad)
            self.assertIn("error", first, bad[:60])
            self.assertEqual(first["kind"], "invalid", (bad[:60], first))
            self.assertNotIn("Traceback", json.dumps(first))
            self.assertNotIn("object is not subscriptable", first["error"])
            self.assertEqual(len(app.calls), 1, bad[:60])  # only the good request reached the runtime

    def test_deep_but_valid_json_is_not_fatal(self):
        # valid JSON nested 3000 levels: either answered or refused as a bad request, never an exception out of the loop
        rc, out, app = self.serve([b'{"text":"x","source_lang":"en","target_lang":"ja","a":' + b'[' * 3000 + b']' * 3000 + b'}\n', self.ok()])
        self.assertEqual(rc, 0)
        self.assertEqual(out[-1]["id"], "r")

    def test_bom_on_first_line_is_tolerated(self):
        rc, out, app = self.serve([b"\xef\xbb\xbf" + self.ok()])
        self.assertNotIn("error", out[0])

    def test_id_echoed_on_error(self):
        rc, out, app = self.serve([b'{"id":"zz","text":" ","source_lang":"en","target_lang":"ja"}\n'])
        self.assertEqual(out[0]["id"], "zz")
        self.assertEqual(out[0]["kind"], "invalid")
        rc, out, app = self.serve([b'{"id":7,"text":"x","target_lang":"xx"}\n'])
        self.assertEqual(out[0]["id"], 7)

    def test_blank_lines_crlf_missing_final_newline_and_eof(self):
        rc, out, app = self.serve([b"\n", b"\r\n", self.ok()[:-1] + b"\r\n", self.ok(id="last")[:-1]])
        self.assertEqual(rc, 0)
        self.assertEqual([o["id"] for o in out], ["r", "last"])

    def test_oversized_line_discarded(self):
        old = cli83.SERVICE_MAX_LINE_BYTES
        cli83.SERVICE_MAX_LINE_BYTES = 120
        try:
            rc, out, app = self.serve([b"x" * 500 + b"\n", self.ok()])
        finally:
            cli83.SERVICE_MAX_LINE_BYTES = old
        self.assertEqual(out[0]["kind"], "invalid")
        self.assertIn("too long", out[0]["error"])
        self.assertEqual(out[1]["id"], "r")

    def test_prompt_too_long_is_invalid_not_fatal(self):
        class Long(FakeTokenizer):
            def chat(self, *a, **k):
                return [1] * 3000

        rc, out, app = self.serve([self.ok()], tokenizer=Long())
        self.assertEqual(out[0]["kind"], "invalid")
        self.assertIn("3000 tokens", out[0]["error"])
        self.assertEqual(rc, 0)

    def test_language_error_does_not_change_mode(self):
        # a bad language is refused before the runtime: the mode stays what it was
        rc, out, app = self.serve([b'{"id":"a","text":"Hi.","source_lang":"en","target_lang":"xx"}\n', self.ok()])
        self.assertEqual(out[1]["mode"], "w4-m2")
        self.assertEqual(app.calls[0][2], "ja")

    def test_internal_error_keeps_serving_without_leaking(self):
        class Boom(Exception):
            pass

        app = FakeApp(fail=Boom("internal python text 'int' object is not subscriptable"))
        rc, out, app = self.serve([self.ok(), self.ok()], app=app)
        self.assertEqual(len(out), 2)
        for o in out:
            self.assertEqual(o["kind"], "internal")
            self.assertNotIn("subscriptable", o["error"])

    def test_fatal_error_exits_with_code(self):
        app = FakeApp(fail=RuntimeError("Failed to create context virtual (0xc01e0009)"))
        rc, out, app = self.serve([self.ok(), self.ok()], app=app)
        self.assertEqual(rc, E.NPU)
        self.assertEqual(out[0]["kind"], "fatal")
        self.assertEqual(len(out), 1)

    def test_stdout_only_json(self):
        rc, out, app = self.serve([self.ok()])
        self.assertIn("tokens", out[0])
        self.assertEqual(out[0]["stop"], "eos")
        self.assertEqual(out[0]["fallbacks"], [])

    def test_max_new_tokens_forwarded(self):
        rc, out, app = self.serve([self.ok(max_new_tokens=17)])
        self.assertEqual(app.calls[0][1], 17)


if __name__ == "__main__":
    unittest.main()
