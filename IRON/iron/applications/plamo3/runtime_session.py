# SPDX-License-Identifier: Apache-2.0
"""Bounded embedding staging and greedy tokens over a direct cached runtime."""

import operator
import struct
import time


class RuntimeSession:
    """Own the logical KV cursor; reset leaves masked device tail rows untouched."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.model = runtime.manifest["model"]
        self.hidden = int(self.model["hidden"])
        self.vocab_size = int(self.model["vocab_size"])
        self.max_seq = int(self.model["max_seq"])
        self.row_bytes = self.hidden * 2
        embedding = self.model["embedding"]
        self.embedding = runtime._artifact(embedding["artifact"])
        self.embedding_offset = int(embedding.get("offset", 0))
        if (
            self.embedding_offset < 0
            or self.embedding_offset + self.vocab_size * self.row_bytes
            > len(self.embedding)
        ):
            raise ValueError("embedding table exceeds exported artifact")
        self.length = 0
        self.last_graph = None
        self.last_token = None

    def reset(self):
        self.length = 0
        self.last_graph = None
        self.last_token = None

    def _token(self, token):
        if isinstance(token, bool):
            raise ValueError("token must be an integer")
        token = operator.index(token)
        if not 0 <= token < self.vocab_size:
            raise ValueError("token outside exported vocabulary")
        return token

    def embedding_row(self, token):
        token = self._token(token)
        start = self.embedding_offset + token * self.row_bytes
        return memoryview(self.embedding)[start : start + self.row_bytes]

    def _carry(self, graph):
        # execute has synchronized this four-byte graph output already.
        raw = self.runtime.read(self.model["tokens"][graph], nbytes=4, sync=False)
        (token,) = struct.unpack("<i", raw)
        token = self._token(token)
        self.last_token = token
        self.last_graph = graph
        return token

    def prefill(self, tokens):
        tokens = [self._token(token) for token in tokens]
        if not tokens or len(tokens) > self.max_seq:
            raise ValueError("prompt must contain 1..max_seq tokens")
        self.reset()
        target = self.model["prefill_input"]
        size = len(self.runtime.views[target])
        batch = int(self.model.get("prefill_rows", 32))
        if batch not in (32, 64, 128) or size != batch * self.row_bytes:
            raise ValueError("direct prefill buffer differs from exported capacity")
        start_time = time.perf_counter()
        records = []
        for start in range(0, len(tokens), batch):
            chunk = tokens[start : start + batch]
            dense = bytearray(size)
            for row, token in enumerate(chunk):
                dense[row * self.row_bytes : (row + 1) * self.row_bytes] = (
                    self.embedding_row(token)
                )
            self.runtime.write(target, dense)
            values = {
                "start": start,
                "rows": len(chunk),
                "position": start + len(chunk) - 1,
            }
            graph = (
                "prefill_body"
                if start + len(chunk) < len(tokens)
                and "prefill_body" in self.runtime.manifest.get("graphs", {})
                else "prefill"
            )
            records.append(dict(self.runtime.execute(graph, values), graph=graph))
            self.length = start + len(chunk)
        token = self._carry("prefill")
        return {
            "token": token,
            "position": self.length - 1,
            "stats": records,
            "elapsed_s": time.perf_counter() - start_time,
        }

    def decode(self, token, position=None):
        token = self._token(token)
        if isinstance(position, bool):
            raise ValueError("decode position must be an integer")
        position = self.length if position is None else operator.index(position)
        if position != self.length or not 0 <= position < self.max_seq:
            raise ValueError(
                "decode position must equal the logical KV length below max_seq"
            )
        start_time = time.perf_counter()
        self.runtime.write(self.model["decode_embedding"], self.embedding_row(token))
        stats = self.runtime.execute("decode", {"position": position})
        self.length = position + 1
        result = self._carry("decode")
        return {
            "token": result,
            "position": position,
            "stats": stats,
            "elapsed_s": time.perf_counter() - start_time,
        }

    def read_logits(self, graph=None):
        graph = self.last_graph if graph is None else graph
        if graph not in ("prefill", "decode"):
            raise ValueError("no completed graph has produced logits")
        return self.runtime.read(
            self.model["outputs"][graph], nbytes=self.vocab_size * 2
        )


def check_references(session, *, generated=None, generated_prompt=3):
    """After timing, compare every exported row and carry token exactly.

    Reference ``tokens`` are teacher-forced next inputs. ``actual_tokens`` are
    reference NPU greedy outputs and may differ after EOS. Live prefix checks
    use only rows through the first EOS, inclusive; bit equality checks all rows.
    """
    manifest = session.runtime.manifest
    references = manifest.get("references")
    if not references or not references.get("prompts"):
        raise ValueError(
            "exact serial reference rows are missing from the runtime bundle"
        )
    blob = session.runtime._artifact(references["artifact"])
    row_bytes = session.vocab_size * 2
    records = []
    greedy_ok = None
    for reference in references["prompts"]:
        prompt_id = int(reference["prompt"])
        count = int(reference["rows"])
        start = int(reference["offset"])
        if count <= 0 or start < 0 or start + count * row_bytes > len(blob):
            raise ValueError("reference row slice exceeds artifact")
        teacher = reference["tokens"]
        actual = reference["actual_tokens"]
        if len(teacher) != count or len(actual) != count:
            raise ValueError("reference token counts differ from row count")
        live = int(reference["live_rows"])
        if not 1 <= live <= count:
            raise ValueError("invalid reference live row count")
        prompt = manifest["model"]["prompts"][prompt_id]
        result = session.prefill(prompt)
        passed = 0
        for row in range(count):
            if row:
                result = session.decode(teacher[row - 1])
            logits = session.read_logits()
            expected = memoryview(blob)[
                start + row * row_bytes : start + (row + 1) * row_bytes
            ]
            bitwise = logits == expected
            token_equal = result["token"] == actual[row]
            if not bitwise or not token_equal:
                first_byte = next(
                    (
                        i
                        for i, pair in enumerate(zip(logits, expected))
                        if pair[0] != pair[1]
                    ),
                    None,
                )
                raise AssertionError(
                    f"direct reference differs: prompt={prompt_id}, row={row}, bitwise={bitwise}, token={result['token']}/{actual[row]}, first_byte={first_byte}"
                )
            passed += 1
        records.append(
            {
                "prompt": prompt_id,
                "rows": passed,
                "live_rows": live,
                "logits_bitwise": True,
                "carry_tokens_equal": True,
            }
        )
        if generated is not None and prompt_id == generated_prompt:
            if len(generated) < live:
                raise AssertionError(
                    "timing trace is too short to compare the full live greedy prefix"
                )
            greedy_ok = list(generated[:live]) == list(actual[:live])
            if not greedy_ok:
                raise AssertionError(
                    "timing greedy live prefix differs from serial reference"
                )
    if generated is not None and greedy_ok is None:
        raise ValueError("timed prompt has no exported serial reference")
    return {
        "passed": True,
        "rows": sum(record["rows"] for record in records),
        "live_rows": sum(record["live_rows"] for record in records),
        "timing_greedy_live_prefix_equal": greedy_ok,
        "prompts": records,
        "export_reference_gate": references.get("gate", manifest.get("reference_gate")),
    }
