#!/usr/bin/env python3
"""Extract the position-dependent words ("holes") of the r34-w4 serial decode streams.

Read-only. Reads only the workstation copy of the runtime package:
  manifest.json (payload), transactions.bin (zlib parts).

For each instruction with params == ["position"] it decodes every exported case,
finds the 32-bit words that differ between cases, fits each to one of the three
affine forms in pos, verifies the fit on every case, parses the enclosing
transaction op (WRITE / BLOCKWRITE / MASKWRITE / DDR_PATCH), and attaches the
layer(s) and purpose established by reading the source (see REPORT.md).

Usage:
  python3 holes.py [--package DIR] [--json OUT] [--quiet]
"""

import argparse
import json
import sys
import zlib
from array import array
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_PACKAGE = Path(
    "~/workspace/plamo3-fast-inference-by-xdna/runs/upload/hf/npu/plamo3-runtime/r34-w4"
).expanduser()

OP_NAMES = {0: "WRITE", 1: "BLOCKWRITE", 3: "MASKWRITE", 4: "MASKPOLL",
            128: "CUSTOM128", 129: "DDR_PATCH"}
# (index of the size word, minimum op words) per opcode, as in
# runs/r24/testtree/transaction_blocks.py (pinned mlir_aie TxnEncoding.h).
SIZE_FIELDS = {0: (5, 6), 1: (3, 4), 3: (6, 7), 4: (6, 7), 128: (1, 4), 129: (1, 12)}
POS_FORMS = {
    "a*pos+b": lambda q: q,
    "a*(pos//64+1)+b": lambda q: q // 64 + 1,
    "a*ceil((pos+1)/16)+b": lambda q: -(-(q + 1) // 16),
}
POSITION_INSTRUCTIONS = ["block0", "block1", "block2", "block3", "block4", "block5", "inst28"]
MAX_POS = 262143
HEADER = (0x06030100, 0x104)

# Decode step -> instruction (from graphs.decode.steps; index = layer, 16 = final head).
# Layers 7 and 15 are the full-attention layers (block3, block4); all others sliding.


def load_package(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest["payload"] if "payload" in manifest else manifest
    blob = (root / "transactions.bin").read_bytes()
    return payload, blob


def case_bytes(payload, blob, name, pos):
    part = payload["instructions"][name]["cases"][str(pos)]
    raw = zlib.decompress(blob[part["offset"]: part["offset"] + part["nbytes"]])
    if len(raw) != part["decoded_nbytes"]:
        raise ValueError(f"{name} case {pos}: decoded size mismatch")
    return raw


def parse_ops(raw):
    words = array("I", raw)
    if (words[0], words[1]) != HEADER or words[3] != len(raw):
        raise ValueError("unexpected transaction header")
    at, ops = 4, []
    while at < len(words):
        opcode = words[at]
        field, minimum = SIZE_FIELDS[opcode]
        size = words[at + field]
        if size % 4 or size // 4 < minimum or at + size // 4 > len(words):
            raise ValueError(f"malformed op at word {at}")
        ops.append({"at": at, "opcode": opcode, "n": size // 4,
                    "words": list(words[at: at + size // 4])})
        at += size // 4
    if len(ops) != words[2]:
        raise ValueError("op count mismatch")
    return words, ops


def decode_addr(addr):
    """AIE tile address -> (col, row, offset) for NPU1 (col<<25 | row<<20 | off)."""
    return (addr >> 25) & 0x7, (addr >> 20) & 0x1F, addr & 0xFFFFF


def shim_bd(off):
    if 0x1D000 <= off < 0x1D100:
        return (off - 0x1D000) // 0x20, ((off - 0x1D000) % 0x20) // 4
    return None


def fit(points):
    """points: [(pos, value)]. Return all (form, a, b) that reproduce every point."""
    out = []
    for form, f in POS_FORMS.items():
        seen = {}
        ok = True
        for q, y in points:
            x = f(q)
            if x in seen and seen[x] != y:
                ok = False
                break
            seen[x] = y
        if not ok or len(seen) < 2:
            continue
        xs = sorted(seen)
        x0, x1 = xs[0], xs[1]
        da = seen[x1] - seen[x0]
        if da % (x1 - x0):
            continue
        a = da // (x1 - x0)
        b = seen[x0] - a * x0
        if all(a * f(q) + b == y for q, y in points):
            out.append((form, a, b))
    return out


def formula_text(form, a, b):
    core = {"a*pos+b": "pos", "a*(pos//64+1)+b": "(pos//64+1)",
            "a*ceil((pos+1)/16)+b": "ceil((pos+1)/16)"}[form]
    coef = "" if a == 1 else f"{a}*"
    if b == 0:
        return f"{coef}{core}"
    return f"{coef}{core}{'+' if b > 0 else '-'}{abs(b)}"


def value_at(form, a, b, q):
    return a * POS_FORMS[form](q) + b


# ---------------------------------------------------------------- classification
# RTP words written by the MHA recipe (resident_streams.py:329-355, descriptor() at
# :175, Rtp([kind,a,b,c]) at resident_array.py:204). Keyed by (col, row, DM offset).
RTP_WORDS = {
    (0, 3, 0x1A84): ("MHA role1 (K stream lane)", "a",
                     "K packet count = 8*blocks (descriptor(1,1,blocks*(bk//8),0))"),
    (1, 2, 0x1A84): ("MHA role2 (V stream lane)", "a",
                     "V packet count = 8*blocks (descriptor(2,1,blocks*(bk//8),0))"),
    (0, 4, 0x9004): ("MHA role10 qk_core", "a",
                     "blocks = pos//64+1 (qk_core: blocks=w[op.a])"),
    (0, 4, 0x9008): ("MHA role10 qk_core", "b",
                     "start = ceil((pos+1)/16) (qk_core: start=w[op.b]; kernel ix[1])"),
    (0, 5, 0xC000): ("MHA role11 soft_core", "kind",
                     "blocks = pos//64+1 (soft_core: blocks=w[kind])"),
    (0, 5, 0xC004): ("MHA role11 soft_core", "a",
                     "sq = (ceil((pos+1)/16)+1)*16 (score row length; cf. attention_stages.cc:19)"),
    (0, 5, 0xC008): ("MHA role11 soft_core", "b",
                     "sk = pos+1 valid keys (tail mask)"),
    (0, 5, 0xC00C): ("MHA role11 soft_core", "c",
                     "start = ceil((pos+1)/16)"),
    (1, 5, 0x1804): ("MHA role12 pv_core", "a",
                     "blocks = pos//64+1 (pv_core: blocks=w[op.a])"),
    (1, 5, 0x1808): ("MHA role12 pv_core", "b",
                     "sk = pos+1 valid keys (pv_core: sk=w[op.b])"),
    (1, 5, 0x180C): ("MHA role12 pv_core", "c",
                     "start = ceil((pos+1)/16) (pv_core: start=w[op.c])"),
}

LIMIT_NOTES = {
    "blocks": "pos//64+1 <= 4096 at 262143; int32 RTP word (resident_attention.py:314 idx=int32)",
    "packets": "8*blocks <= 32768 at 262143; int32 RTP word, fits 16 bits unsigned only",
    "start": "ceil((pos+1)/16) <= 16384; int32",
    "sq": "16*(ceil+1) <= 262160 at 262143; int32 (resident_attention.py:342 int32 args). "
          "Whether the kernel uses sq as a stride is UNVERIFIED (C++ partial_softmax not in snapshot)",
    "sk": "pos+1 <= 262144 at 262143; > 16 bits; int32 everywhere in the kernel ABI",
    "bd_len": "BD word0 length in 32-bit words = 4096*blocks <= 16,777,216 (2^24) at 262143. "
              "Largest value in today's stream is 0x20000 (2^17, 18 bits). The width of the shim "
              "BD0 length field is UNVERIFIED here; a 25-bit field is needed",
    "dma_plus": "64-bit DDR_PATCH plus (words 10-11); 256*pos <= 67,108,608 at 262143 fits",
    "record": "per-position Sample record offset grows past view348 (40960 B); must be extended "
              "or redirected",
}


def classify_ddr(name, slot, base, layers_by_slot):
    """Return (purpose, field, layer_text, limit_key) for a DDR_PATCH hole."""
    if name == "block5":
        if base == 0x8000:
            return ("Sample per-position 4-byte record (view348+0x8000+4*pos); "
                    "consumer not found in snapshot", "record", "decode L16 (final head)", "record")
        return ("Sample per-position 16-byte record (view348+16*pos); "
                "consumer not found in snapshot", "record", "decode L16 (final head)", "record")
    if name == "inst28":
        return ("prefill-only per-position record (not in decode graph)", "record",
                "prefill only", "record")
    if name == "block0":
        if slot == 2 and base == 0:
            return ("RoPE angle row read, sliding table (view312+0 = view2, 2048x256B), "
                    "copied to view311+14592 (sliding angles)", "dma_plus", "L0 (block0)", "dma_plus")
        if slot == 2 and base == 0x80000:
            return ("RoPE angle row read, full/YaRN table (view312+512K = view4), "
                    "copied to view5 (full angles)", "dma_plus", "L0 (block0)", "dma_plus")
        if slot == 0 and base == 0xC00:
            return ("layer-0 K-cache row write (view310+3072+256*pos), read by MHA K base",
                    "dma_plus", "L0 (block0)", "dma_plus")
        if slot == 0 and base == 0x80C00:
            return ("layer-0 V-cache row write (view310+0x80C00+256*pos), read by MHA V base",
                    "dma_plus", "L0 (block0)", "dma_plus")
    if name in ("block1", "block2", "block3", "block4"):
        if slot == 2 and base == 0:
            return ("K-cache row write, row=pos (QKNormCache drain at position*128)",
                    "dma_plus", layers_by_slot, "dma_plus")
        if slot == 2 and base == 0x80000:
            return ("V-cache row write, row=pos (DecodeQKV drain at position*128)",
                    "dma_plus", layers_by_slot, "dma_plus")
    raise ValueError(f"unclassified DDR hole {name} slot={slot} base={base:#x}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    payload, blob = load_package(args.package)
    steps = payload["graphs"]["decode"]["steps"]
    layer_instr = {}
    for layer, step in enumerate(steps[:16]):
        layer_instr[layer] = step["instructions"]
    final_instr = steps[16]["instructions"]
    instr_layers = defaultdict(list)
    for layer, name in layer_instr.items():
        instr_layers[name].append(layer)
    # views per (instruction, layer) for slot-indexed holes
    layer_args = {layer: steps[layer]["args"] for layer in range(16)}
    decode_used = set(layer_instr.values()) | {final_instr}

    holes = []
    summary = {}
    for name in POSITION_INSTRUCTIONS:
        inst = payload["instructions"][name]
        cases = inst["cases"]
        keys = sorted(cases, key=int)
        positions = [int(k) for k in keys]
        raws = {p: array("I", case_bytes(payload, blob, name, p)) for p in positions}
        lengths = {len(v) for v in raws.values()}
        if len(lengths) != 1:
            raise SystemExit(f"{name}: case lengths differ: {sorted(lengths)}")
        nwords = lengths.pop()
        base = raws[positions[0]]
        varying = [i for i in range(nwords)
                   if len({raws[p][i] for p in positions}) > 1]
        _, ops0 = parse_ops(case_bytes(payload, blob, name, positions[0]))
        op_of_word = {}
        for op in ops0:
            for j in range(op["n"]):
                op_of_word[op["at"] + j] = op
        ops_by_at = {op["at"]: op for op in ops0}
        # BD base patches keyed by shim register, for pairing BLOCKWRITE lengths
        ddr_patch_at = {}
        for op in ops0:
            if op["opcode"] == 129:
                w = op["words"]
                plus = w[10] | (w[11] << 32)
                ddr_patch_at[(w[6], w[8])] = plus
        dynamic = {}
        for i in varying:
            op = op_of_word[i]
            points = [(p, int(raws[p][i])) for p in positions]
            fits = fit(points)
            if len(fits) != 1:
                raise SystemExit(f"{name} word {i}: formula fit count {len(fits)} {fits}")
            form, a, b = fits[0]
            # verify exactly on every case
            if not all(a * POS_FORMS[form](p) + b == y for p, y in points):
                raise SystemExit(f"{name} word {i}: formula does not reproduce all cases")
            dynamic[i] = (op, form, a, b)

        for i in varying:
            op, form, a, b = dynamic[i]
            w = op["words"]
            row = {
                "instruction": name,
                "word_index": i,
                "op_index": op["at"],
                "op_opcode": op["opcode"],
                "op": OP_NAMES[op["opcode"]],
                "formula": formula_text(form, a, b),
                "form": form,
                "a": a,
                "b": b,
                "value_pos0": value_at(form, a, b, 0),
                "value_pos2047": value_at(form, a, b, 2047),
                "value_pos262143": value_at(form, a, b, MAX_POS),
                "decode_layers": sorted(instr_layers.get(name, []))
                if name != "block5" else [16],
                "decode_used": name in decode_used,
            }
            if op["opcode"] == 129:
                slot = w[8]
                addr = w[6]
                col, rowi, off = decode_addr(addr)
                bd = shim_bd(off)
                plus_const = b
                row.update(
                    register=f"{addr:#010x}", field=f"BD{bd[0]} word{bd[1]} base (DDR_PATCH "
                    f"arg slot {slot}, plus = {formula_text(form, a, b)})" if bd else "DDR_PATCH",
                    slot=slot,
                    shim_col=col,
                )
                layers_text = ""
                if name in ("block1", "block2", "block3", "block4") or name == "block0":
                    if name == "block0":
                        layers_text = "L0"
                    else:
                        layers_text = ",".join(f"L{l}" for l in sorted(instr_layers[name]))
                    kv = {}
                    for l in sorted(instr_layers[name]):
                        kv[l] = layer_args[l][slot]
                    row["views_by_layer"] = kv
                elif name == "block5":
                    row["views_by_layer"] = {16: steps[16]["args"][slot]}
                    layers_text = "L16"
                purpose, _field_kind, _layers, limit = classify_ddr(
                    name, slot, b, layers_text)
                if name == "block0" and slot == 0 and b == 0xC00:
                    row["views_by_layer"] = {0: layer_args[0][0]}
                row.update(purpose=purpose, limit_key=limit)
            elif op["opcode"] == 0:
                col, rowi, off = decode_addr(w[2])
                key = (col, rowi, off)
                if key not in RTP_WORDS:
                    raise SystemExit(f"unmapped WRITE register {w[2]:#x} in {name}")
                role, field, meaning = RTP_WORDS[key]
                # word index within the op: op words are [0,0,reg,0,value,size]
                limit = ("blocks" if "blocks" in meaning else
                         "packets" if "packet" in meaning else
                         "start" if "start" in meaning else
                         "sq" if "sq" in meaning else "sk")
                row.update(
                    register=f"{w[2]:#010x}",
                    field=f"tile({col},{rowi}) DM {off:#x} = {role} word '{field}'",
                    slot=None, shim_col=None,
                    purpose=meaning, limit_key=limit,
                )
            elif op["opcode"] == 1:
                col, rowi, off = decode_addr(op["words"][2])
                bd = shim_bd(off)
                if not bd:
                    raise SystemExit(f"unexpected BLOCKWRITE target {op['words'][2]:#x}")
                assert i == op["at"] + 4 and bd[1] == 0, "expected BD word0 length"
                # Base patch for the same BD (word1): the first DDR_PATCH on that
                # register after this BLOCKWRITE. The register is patched several times
                # for different buffers, so "first after" is the pairing rule.
                base_addr = op["words"][2] + 4
                pl = next((o["words"][10] | (o["words"][11] << 32)
                           for o in ops0
                           if o["opcode"] == 129 and o["at"] > op["at"]
                           and o["words"][6] == base_addr), None)
                kind = "K" if (pl is not None and pl < 0x80000) else "V"
                row["paired_base_plus"] = pl
                purpose = (f"BD{bd[0]} word0 Buffer_Length (32-bit words) = 4096*blocks "
                           f"= 64 keys x 256 B; {kind}-cache read (base from its DDR_PATCH)")
                row.update(
                    register=f"{op['words'][2]:#010x}",
                    field=f"BD{bd[0]} word0 Buffer_Length (words), tile({col},{rowi})",
                    slot=None, shim_col=None,
                    purpose=purpose, limit_key="bd_len",
                )
            else:
                raise SystemExit(f"unexpected opcode for hole in {name}: {op['opcode']}")
            row["op_words"] = [hex(x) for x in op["words"][:12]]
            holes.append(row)
        summary[name] = {"cases": len(positions), "words": nwords, "varying": len(varying)}

    # Global numbering, decode-used holes first in instruction order.
    for n, row in enumerate(holes, 1):
        row["index"] = n
    totals = Counter(h["form"] for h in holes)
    summary["total_holes"] = len(holes)
    summary["by_form"] = dict(totals)
    summary["decode_used_holes"] = sum(1 for h in holes if h["decode_used"])
    summary["sanity"] = {
        "all_cases_same_length": True,
        "all_formulas_reproduce_every_case": True,
        "instructions_used_by_decode": sorted(decode_used),
        "layer_instruction": {str(k): v for k, v in layer_instr.items()},
        "final_instruction": final_instr,
    }

    if not args.quiet:
        hdr = f"{'#':>2} {'instr':7} {'word':>5} {'op':10} {'register/addr':22} {'field':60} {'formula':30} {'layers':40} purpose"
        print(hdr)
        print("-" * len(hdr))
        for h in holes:
            layers = (",".join(f"L{l}" for l in h["decode_layers"])
                      if h["decode_used"] else "prefill only")
            print(f"{h['index']:>2} {h['instruction']:7} {h['word_index']:>5} {h['op']:10} "
                  f"{h['register']:22} {h['field'][:60]:60} {h['formula']:30} "
                  f"{layers[:40]:40} {h['purpose'][:90]}")
        print()
        print(json.dumps({k: v for k, v in summary.items() if k != "sanity"}, indent=2))

    out = {"package": str(args.package), "summary": summary, "holes": holes,
           "limit_notes": LIMIT_NOTES}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
