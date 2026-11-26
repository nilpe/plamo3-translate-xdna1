# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-token/template equivalence; no target weights or NPU imports."""

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import textwrap
import time


def cases():
    rng = random.Random(20201112)
    natural = [
        "今日はいい天気ですね。",
        "駅までの道を教えてください。",
        "Thank you for your help.",
        "The meeting starts at nine tomorrow morning.",
        "空港へ行くバスは何時に出ますか。",
        "Please send the updated report before Friday.",
        "This is a test of the translation system.",
        "翻訳する前に、専門用語を確認してください。",
        "日本語とEnglishが混ざった文章です。",
        "１，２，３：全角文字と半角 ABC 123",
        "café naïve résumé Straße",
        "你好，世界！",
    ]
    texts = ["", " ", "\n", "\r\n", "\t", "  leading and trailing  "]
    for i in range(1200):
        base = natural[i % len(natural)]
        texts.append(
            (" " * (i % 5))
            + base
            + ("\n" if i % 3 == 0 else " ")
            + str(i)
            + ("!" * (i % 9))
        )
    texts.extend([c * n for c in " a!日\n\t" for n in (1, 2, 3, 4, 5, 8, 16, 33, 128)])
    texts.extend(
        ["e\u0301", "é", "😀\u200d🚀", "𠮷野家", "\U0010ffff", "\uee00", "a\uee00b"]
    )
    texts.extend(
        [
            f"before<|plamo:{name}|>after"
            for name in (
                "bos",
                "eos",
                "pad",
                "unk",
                "tag",
                "msg",
                "key",
                "val",
                "index",
                "item",
                "unknown",
            )
        ]
    )
    for _ in range(256):
        alphabet = "abc ABC123 あいうえお漢字éαβγ\n\t😀🦊\u0301\u200d"
        texts.append(
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 100)))
        )
    chats = []
    languages = [
        ("ja", "en"),
        ("en", "ja"),
        (None, "ja"),
        ("zh-hans", "en"),
        ("ja(easy)", "ja(easy)"),
        ("en", "ja-easy"),
    ]
    for i in range(1024):
        src, dst = languages[i % len(languages)]
        messages = [
            {
                "role": "user",
                "content": natural[i % len(natural)]
                + ("\n " if i % 2 else " ")
                + str(i),
            }
        ]
        if i % 4 == 0:
            messages = [
                {"role": "user", "content": "  Previous source.\n"},
                {"role": "assistant", "content": " 前の翻訳。 "},
                *messages,
            ]
        if i % 13 == 0:
            messages = [
                {"role": "user", "content": "Earlier source"},
                {"role": "assistant", "content": "以前の翻訳"},
                *messages,
            ]
        options = dict(source_lang=src, target_lang=dst)
        if i % 3 == 0:
            options["context"] = "\n reference text\t" if i % 2 else ""
        if i % 7 == 0:
            options["glossary"] = [
                {"source": "NPU", "target": "ニューラルプロセッサ"},
                {"source": "model", "target": "モデル"},
            ]
        if dst == "ja" and i % 2:
            options["form"] = "polite" if i % 3 else "plain"
        chats.append((messages, options))
    valid = [{"role": "user", "content": "test"}]
    invalid = [
        ([], {}),
        ([None], {}),
        ([{}], {}),
        ([{"role": "system", "content": "x"}], {}),
        ([{"role": "assistant", "content": "x"}], {}),
        (valid * 2, {}),
        ([{"role": "user", "content": ""}], {}),
        ([{"role": "user", "content": " \n"}], {}),
        ([{"role": "user", "content": None}], {}),
        ([{"role": "user", "content": ["x"]}], {}),
        ([{"role": "user", "content": "<|plamo:tag|>"}], {}),
        ([{"role": "user", "content": "x", "tool_call_id": "a"}], {}),
        ([{"role": "user", "content": "x", "tool_calls": [{}]}], {}),
        ([{"role": "user", "content": "x", "function_call": {"name": "x"}}], {}),
    ]
    for value in (
        None,
        1,
        [],
        {},
        "",
        "EN",
        "english",
        "en-US",
        "en--",
        "ja-easy-extra",
        "<|plamo:bos|>",
    ):
        invalid.append((valid, {"target_lang": value}))
    for name, values in {
        "source_lang": [1, "", "EN", "en-US", "<|plamo:bos|>"],
        "context": [1, [], {}, "<|plamo:tag|>"],
        "form": ["formal", False, "polite"],
        "glossary": [
            "x",
            {},
            [{}],
            [{"source": "a"}],
            [{"source": "a", "target": "b", "extra": "c"}],
            [{"source": "", "target": "b"}],
            [{"source": "a", "target": None}],
            [{"source": "<|plamo:tag|>", "target": "b"}],
        ],
        "tools": [[{"type": "function", "function": {"name": "f", "parameters": {}}}]],
        "documents": [[{"title": "x", "text": "y"}]],
        "add_generation_prompt": [False],
    }.items():
        invalid.extend((valid, {name: v}) for v in values)
    invalid.append(
        (
            valid,
            {
                "target_lang": "ja",
                "form": "polite",
                "glossary": [{"source": "a", "target": "b"}],
            },
        )
    )
    return texts, chats, invalid


def corrected_encoder(hf):
    """Only fix the pinned NumPy bool addition in fallback allocation accounting."""
    cls = type(hf.aho_corasick)
    function = cls._encode
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    node.decorator_list = []
    changed = 0

    class Fix(ast.NodeTransformer):
        def visit_AugAssign(self, n):
            nonlocal changed
            if "PATH_NUM_TOKENS" in ast.unparse(n.target):

                class Count(ast.NodeTransformer):
                    def visit_Compare(self, c):
                        return ast.Call(
                            func=ast.Name(id="int", ctx=ast.Load()),
                            args=[c],
                            keywords=[],
                        )

                n.value = Count().visit(n.value)
                changed += 1
            return n

    tree = ast.fix_missing_locations(Fix().visit(tree))
    if changed != 1:
        raise RuntimeError("pinned byte-count expression changed")
    namespace = dict(function.__globals__)
    exec(compile(tree, "<pinned-int-bool-count>", "exec"), namespace)
    return cls, function, namespace[node.name]


def outcome(call):
    try:
        return ("value", call())
    except Exception as exc:
        return ("error", type(exc).__name__, str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--deadline", type=int, default=480)
    args = parser.parse_args()
    import threading

    watchdog = threading.Timer(args.deadline, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    os.environ.update(
        USE_TORCH="0", USE_TF="0", NUMBA_DISABLE_JIT="1", HF_HUB_OFFLINE="1"
    )
    import fast_tokenizer as fast

    if args.prepare:
        fast.export(args.checkpoint, args.artifact)
    began = time.perf_counter()
    actual = fast.FastPlamo3Tokenizer(args.artifact)
    fast_ms = (time.perf_counter() - began) * 1000
    forbidden = ["transformers", "numba", "numpy", "iron", "aie"]
    fast_imports = {
        name: any(m == name or m.startswith(name + ".") for m in sys.modules)
        for name in forbidden
    }
    from transformers import AutoTokenizer

    began = time.perf_counter()
    expected = AutoTokenizer.from_pretrained(
        str(args.checkpoint), trust_remote_code=True
    )
    hf_ms = (time.perf_counter() - began) * 1000
    cls, original, corrected = corrected_encoder(expected)
    texts, chats, invalid = cases()
    counts = dict(
        text_cases=0,
        original_token_matches=0,
        corrected_byte_fallback_matches=0,
        baseline_errors=0,
        baseline_byte_differences=0,
        decode_checks=0,
        template_strings=0,
        template_token_checks=0,
        error_cases=0,
        unicode_roundtrips=0,
    )
    failures = []
    baseline_errors = []
    checksum = hashlib.sha256()

    def token_check(text, add_special_tokens, label):
        got = actual.encode(text, add_special_tokens=add_special_tokens)
        old = outcome(
            lambda: expected.encode(text, add_special_tokens=add_special_tokens)
        )
        if old[0] == "value" and old[1] == got:
            counts["original_token_matches"] += 1
        else:
            cls._encode = staticmethod(corrected)
            try:
                fixed = outcome(
                    lambda: expected.encode(text, add_special_tokens=add_special_tokens)
                )
            finally:
                cls._encode = staticmethod(original)
            if fixed[0] != "value" or fixed[1] != got:
                failures.append(
                    dict(
                        kind="tokens",
                        label=label,
                        text=text,
                        original=old,
                        corrected=fixed,
                        actual=got,
                    )
                )
            else:
                counts["corrected_byte_fallback_matches"] += 1
                counts[
                    (
                        "baseline_errors"
                        if old[0] == "error"
                        else "baseline_byte_differences"
                    )
                ] += 1
                if len(baseline_errors) < 8:
                    baseline_errors.append(dict(text=text, result=old))
        checksum.update(json.dumps(got).encode())
        for skip in (False, True):
            want = expected.decode(got, skip_special_tokens=skip)
            have = actual.decode(got, skip_special_tokens=skip)
            counts["decode_checks"] += 1
            if want != have:
                failures.append(
                    dict(
                        kind="decode",
                        label=label,
                        skip=skip,
                        expected=want,
                        actual=have,
                    )
                )

    for i, text in enumerate(texts):
        counts["text_cases"] += 1
        for special in (False, True):
            token_check(text, special, "text-" + str(i))
        if "<|plamo:" not in text and "\uee00" not in text:
            ids = actual.encode(text, add_special_tokens=False)
            if actual.decode(ids, skip_special_tokens=False) != text:
                failures.append(dict(kind="byte-roundtrip", text=text))
            counts["unicode_roundtrips"] += 1
    for i, (messages, options) in enumerate(chats):
        want = expected.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **options
        )
        have = actual.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **options
        )
        counts["template_strings"] += 1
        if want != have:
            failures.append(
                dict(kind="template-string", index=i, expected=want, actual=have)
            )
        token_check(want, False, "chat-" + str(i))
        counts["template_token_checks"] += 1
    for i, (messages, options) in enumerate(invalid):
        options = (
            dict(target_lang="en", add_generation_prompt=True, **options)
            if not {"target_lang", "add_generation_prompt"} & options.keys()
            else {"target_lang": "en", "add_generation_prompt": True, **options}
        )
        a = outcome(
            lambda: expected.apply_chat_template(messages, tokenize=False, **options)
        )
        b = outcome(
            lambda: actual.apply_chat_template(messages, tokenize=False, **options)
        )
        counts["error_cases"] += 1
        if a != b:
            failures.append(
                dict(
                    kind="error-contract",
                    index=i,
                    messages=messages,
                    options=options,
                    expected=a,
                    actual=b,
                )
            )
    for ids in (
        [0, 1, 2, 3, 16],
        [100, 2, 101],
        [100, 16, 101],
        [100, 1, 101],
        list(range(256)),
    ):
        for skip in (False, True):
            a = expected.decode(ids, skip_special_tokens=skip)
            b = actual.decode(ids, skip_special_tokens=skip)
            counts["decode_checks"] += 1
            if a != b:
                failures.append(
                    dict(
                        kind="special-decode", ids=ids, skip=skip, expected=a, actual=b
                    )
                )
    result = dict(
        result="PASS" if not failures else "FAIL",
        counts=counts,
        fast_load_ms=fast_ms,
        hf_construct_after_import_ms=hf_ms,
        fast_imports=fast_imports,
        baseline_error_examples=baseline_errors,
        failures=failures[:30],
        failure_count=len(failures),
        tokens_sha256=checksum.hexdigest(),
        baseline_byte_note="Reference NUMBA_DISABLE_JIT uses NumPy booleans: allocation counts collapse for >2-byte fallback. Those cases compare against identical pinned algorithm with only explicit int(bool) counts; original successful ordinary cases must match unchanged.",
        manifest=str(args.artifact / "tokenizer-manifest.json"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("failures", "baseline_error_examples")
            },
            ensure_ascii=False,
        )
    )
    watchdog.cancel()
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
