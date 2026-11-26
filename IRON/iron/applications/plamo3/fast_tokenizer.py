# SPDX-License-Identifier: Apache-2.0
"""Pinned PLaMo tokenizer, using a serialized exact integer-score decoder.

The checkpoint has tokenizer.jsonl, not a Hugging Face tokenizer.json. Its
right-to-left integer-scored Unigram tie rule is preserved here; converting to
a generic floating-point Unigram model would change that contract.

No Transformers, Numba, NumPy, IRON, or compiler imports on the warm path.
"""

import hashlib
import json
import marshal
from pathlib import Path
import re
from collections.abc import Mapping

FORMAT = "plamo3-exact-unigram-v1"
BOUNDARY_CHAR = "\uee00"
BOUNDARY_TOKEN_ID = 10000000
UNKNOWN_SCORE = -10000000
SOURCE_FILES = (
    "tokenizer.jsonl",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenization_plamo.py",
)
# This implementation of render_chat is the pinned translation template.
TEMPLATE_SHA256 = "633ae24b5166d2c958d9b0a39523f2cc5bd6832752ec2ab0285793cab6415323"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(checkpoint, destination):
    """Prepare once, outside warm startup. Only trusted plain data is serialized."""
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (checkpoint / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    template_hash = digest(checkpoint / "chat_template.jinja")
    if template_hash != TEMPLATE_SHA256:
        raise ValueError("PLaMo chat template changed; port it and rerun equivalence")
    with (checkpoint / "tokenizer.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    tokens = [row[0] for row in rows]
    transitions, terminals, scores, byte_ids = {}, [-1], [0], [None] * 256
    for token_id, row in [
        *enumerate(rows),
        (BOUNDARY_TOKEN_ID, [BOUNDARY_CHAR, 0, "CONTROL"]),
    ]:
        token, score = row[:2]
        if len(row) > 2 and row[2] == "BYTE":
            byte_ids[int(token[3:5], 16)] = token_id
            continue
        node = 0
        for char in token:
            key = (node << 21) | ord(char)
            child = transitions.get(key)
            if child is None:
                child = transitions[key] = len(terminals)
                terminals.append(-1)
                scores.append(0)
            node = child
        terminals[node], scores[node] = token_id, round(score * 10000)
    if any(v is None for v in byte_ids):
        raise ValueError("incomplete byte fallback vocabulary")
    names = {token: i for i, token in enumerate(tokens)}
    special = {int(i): data for i, data in config["added_tokens_decoder"].items()}
    # Named special tokens, not every added token whose metadata says special:
    # this checkpoint names BOS as EOS, so added token id 2 is not skipped.
    named = [
        config.get(k)
        for k in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
        )
    ]
    named.extend(config.get("additional_special_tokens", []))
    special_ids = sorted({names[name] for name in named if name is not None})
    payload = (tokens, transitions, terminals, scores, byte_ids)
    blob = destination / "tokenizer.marshal"
    with blob.open("wb") as stream:
        marshal.dump(payload, stream, 4)
    manifest = dict(
        format=FORMAT,
        checkpoint=str(checkpoint),
        implementation_sha256=digest(__file__),
        sources={name: digest(checkpoint / name) for name in SOURCE_FILES},
        payload=blob.name,
        payload_sha256=digest(blob),
        payload_bytes=blob.stat().st_size,
        vocab_size=len(tokens),
        trie_nodes=len(terminals),
        bos_token=config["bos_token"],
        bos_id=names[config["bos_token"]],
        eos_id=names[config["eos_token"]],
        add_bos_token=config.get("add_bos_token", False),
        add_eos_token=config.get("add_eos_token", False),
        special_ids=special_ids,
        added_tokens={str(i): data["content"] for i, data in special.items()},
        break_spaces=config.get("break_around_consecutive_spaces_threshold"),
        break_repeats=config.get("break_around_repeated_chars_threshold"),
    )
    target = destination / "tokenizer-manifest.json"
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def _template_error(message):
    # Preserve the existing error class while avoiding Jinja on valid warm calls.
    from jinja2.exceptions import TemplateError

    raise TemplateError(message)


def _check_text(value, name, allow_empty=False):
    if not isinstance(value, str):
        _template_error(name + " must be a string.")
    if not allow_empty and not value.strip():
        _template_error(name + " must not be empty.")
    if "<|plamo:" in value:
        _template_error(name + " must not contain reserved PFML tokens (<|plamo:...).")


def _check_lang(value, name):
    _check_text(value, name)
    parts = value.split("-")
    if (
        len(parts) > 2
        or len(parts[0]) not in (2, 3)
        or (len(parts) == 2 and len(parts[1]) != 4)
    ):
        _template_error(
            name
            + " must be a lowercase language code, e.g. en, ja, zh-hans, or ja-easy."
        )
    if any(char not in "abcdefghijklmnopqrstuvwxyz-" for char in value):
        _template_error(
            name + " must use lowercase language codes without region subtags."
        )


_MISSING = object()


def render_chat(
    messages,
    *,
    bos_token="<|plamo:bos|>",
    target_lang=_MISSING,
    source_lang=None,
    context=None,
    form=None,
    glossary=None,
    tools=None,
    documents=None,
    add_generation_prompt=True,
    **kwargs,
):
    """Literal translation of the pinned PFML3 Jinja template, including errors."""
    if len(messages) == 0 or len(messages) % 2 != 1:
        _template_error(
            "Provide source/translation pairs followed by one user message."
        )
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            _template_error("Each message must contain a role and text content.")
        if message.get("role") == "system":
            _template_error(
                "System messages are not supported; use context for reference text."
            )
        if message.get("role") != ("user" if index % 2 == 0 else "assistant"):
            _template_error(
                "Alternate user source text and assistant translation; end with user."
            )
        if (
            message.get("tool_calls")
            or message.get("tool_call_id")
            or message.get("function_call")
        ):
            _template_error("Tool calls are not supported.")
        _check_text(message.get("content"), "message content")
    if tools or documents:
        _template_error(
            "Tool calls and documents are not supported; use context for reference text."
        )
    if not add_generation_prompt:
        _template_error("Use add_generation_prompt=True for translation inference.")
    if target_lang is _MISSING:
        _template_error('target_lang is required, e.g. target_lang="ja".')
    if target_lang == "ja(easy)":
        target_lang = "ja-easy"
    if source_lang == "ja(easy)":
        source_lang = "ja-easy"
    _check_lang(target_lang, "target_lang")
    if source_lang is not None:
        _check_lang(source_lang, "source_lang")
    if context is not None:
        _check_text(context, "context", allow_empty=True)
    if form is not None and (form not in ("polite", "plain") or target_lang != "ja"):
        _template_error('form must be polite or plain and requires target_lang="ja".')
    if glossary is not None:
        if (
            not hasattr(glossary, "__len__")
            or not hasattr(glossary, "__getitem__")
            or isinstance(glossary, (str, Mapping))
        ):
            _template_error(
                "glossary must be a list of objects with source and target strings."
            )
        for term in glossary:
            if (
                not isinstance(term, Mapping)
                or len(term) != 2
                or "source" not in term
                or "target" not in term
            ):
                _template_error(
                    "Each glossary entry must contain exactly source and target."
                )
            _check_text(term["source"], "glossary source")
            _check_text(term["target"], "glossary target")
    tag, msg, key, val = (
        "<|plamo:tag|>",
        "<|plamo:msg|>",
        "<|plamo:key|>",
        "<|plamo:val|>",
    )
    out = [bos_token, tag, "dataset", msg, "translation"]
    if context:
        out.extend((tag, "context", msg, context))
    for message in messages[:-1]:
        out.extend((tag, "input" if message["role"] == "user" else "output"))
        lang = source_lang if message["role"] == "user" else target_lang
        if lang is not None:
            out.extend((key, "lang", val, lang))
        out.extend((msg, message["content"]))
    if glossary:
        out.extend((tag, "glossary"))
        for term in glossary:
            out.extend(
                ("<|plamo:index|>", term["source"], "<|plamo:item|>", term["target"])
            )
    out.extend((tag, "input"))
    if source_lang is not None:
        out.extend((key, "lang", val, source_lang))
    out.extend(
        (msg, messages[-1]["content"], tag, "output", key, "lang", val, target_lang)
    )
    if form is not None:
        out.extend((key, "form", val, form))
    out.append(msg)
    return "".join(out)


class FastPlamo3Tokenizer:
    """IRON tokenizer duck type, loaded only from validated prebuilt data."""

    def __init__(self, manifest, *, checkpoint=None, validate_sources=True):
        manifest = Path(manifest)
        if manifest.is_dir():
            manifest = manifest / "tokenizer-manifest.json"
        self.metadata = meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta["format"] != FORMAT or meta["implementation_sha256"] != digest(
            __file__
        ):
            raise ValueError(
                "fast tokenizer implementation changed; re-export and revalidate"
            )
        if meta["sources"]["chat_template.jinja"] != TEMPLATE_SHA256:
            raise ValueError("unsupported PFML3 template")
        if validate_sources:
            directory = Path(checkpoint or meta["checkpoint"])
            for name, expected in meta["sources"].items():
                if digest(directory / name) != expected:
                    raise ValueError("tokenizer source changed: " + name)
        blob = manifest.parent / meta["payload"]
        if (
            blob.parent != manifest.parent
            or blob.stat().st_size != meta["payload_bytes"]
        ):
            raise ValueError("fast tokenizer artifact differs from its manifest")
        # Read once: marshal.load(file) makes many tiny reads; loads(bytes) is
        # substantially faster and the same bytes are hashed before decoding.
        data = blob.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["payload_sha256"]:
            raise ValueError("fast tokenizer artifact differs from its manifest")
        self.tokens, self.transitions, self.terminals, self.scores, self.byte_ids = (
            marshal.loads(data)
        )
        if (
            len(self.tokens) != meta["vocab_size"]
            or len(self.terminals) != meta["trie_nodes"]
        ):
            raise ValueError("fast tokenizer artifact shape mismatch")
        self.special_ids = set(meta["special_ids"])
        self.legacy_added_ids = set(map(int, meta["added_tokens"])) - self.special_ids
        self._special = re.compile(r"(<\|plamo:[^|\s]{,64}\|>)")
        patterns = []
        if meta["break_repeats"] is not None:
            patterns.append(f"(.)\\2{{{meta['break_repeats'] - 1},}}")
        if meta["break_spaces"] is not None:
            patterns.append(f" {{{meta['break_spaces']},}}")
        self._breaks = re.compile(f"({'|'.join(patterns)})") if patterns else None

    def encode(self, text, *, add_special_tokens=True):
        text = self._special.sub(BOUNDARY_CHAR + r"\1" + BOUNDARY_CHAR, text)
        if self._breaks is not None:
            text = self._breaks.sub(BOUNDARY_CHAR + r"\1" + BOUNDARY_CHAR, text)
        n = len(text)
        costs, lengths, chosen = [0] * (n + 1), [0] * n, [-1] * n
        transition, terminal, scores = self.transitions, self.terminals, self.scores
        for start in range(n - 1, -1, -1):
            node, best, best_length, token = 0, 1 << 60, 0, -1
            for end in range(start, n):
                node = transition.get((node << 21) | ord(text[end]))
                if node is None:
                    break
                candidate = terminal[node]
                if candidate >= 0:
                    cost, length = costs[end + 1] - scores[node], end - start + 1
                    if cost < best or (cost == best and length > best_length):
                        best, best_length, token = cost, length, candidate
            unknown = costs[start + 1] - UNKNOWN_SCORE
            if unknown < best:
                best, best_length, token = unknown, 1, -1
            costs[start], lengths[start], chosen[start] = best, best_length, token
        result, position = [], 0
        while position < n:
            token = chosen[position]
            if token < 0:
                result.extend(
                    self.byte_ids[byte] for byte in text[position].encode("utf-8")
                )
            elif token != BOUNDARY_TOKEN_ID:
                result.append(token)
            position += lengths[position]
        if add_special_tokens:
            if self.metadata["add_bos_token"]:
                result.insert(0, self.metadata["bos_id"])
            if self.metadata["add_eos_token"]:
                result.append(self.metadata["eos_id"])
        return result

    def __call__(self, text):
        return self.encode(text)

    def decode(
        self, token_ids, *, skip_special_tokens=True, spaces_between_special_tokens=True
    ):
        pieces, segments = [], []

        def flush():
            if pieces:
                segments.append(b"".join(pieces).decode("utf-8", errors="replace"))
                pieces.clear()

        for index in token_ids:
            if skip_special_tokens and index in self.special_ids:
                continue
            token = self.tokens[index]
            if index in self.legacy_added_ids:
                flush()
                segments.append(token)
            else:
                pieces.append(
                    bytes([int(token[3:5], 16)])
                    if token.startswith("<0x")
                    else token.encode("utf-8")
                )
        flush()
        return (" " if spaces_between_special_tokens else "").join(segments)

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=True, **kwargs
    ):
        # AutoTokenizer inspects the first item before rendering; preserve its
        # empty-sequence error as well as the template's own validation errors.
        _ = messages[0]
        options = dict(
            bos_token=self.metadata["bos_token"],
            add_generation_prompt=add_generation_prompt,
        )
        options.update(kwargs)
        text = render_chat(messages, **options)
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def chat(self, messages, *, target_lang="en", source_lang=None, **kwargs):
        return self.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            target_lang=target_lang,
            source_lang=source_lang,
            **kwargs,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(export(args.checkpoint, args.destination))
