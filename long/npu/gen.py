#!/usr/bin/env python3
"""PD-256k stream generator: copy-and-patch of the r34-w4 serial decode streams.

Pure Python, workstation-side. Reads the r34-w4 package (read-only) and the
study's holes.json. For each of the 7 position-dependent instructions it takes
the exported case at position 0 as the stencil and writes only the position-
dependent ("hole") words, from their formulas, for any pos in 0..262143.

Modes
  compat  Original buffer layout. Every hole takes its exported formula, so the
          output must be byte-identical to the exported case for pos 0..2047
          (test T1, see `python3 gen.py --t1`).
  pd      PD-256k layout (see layout.py). Sliding layers (block0/1/2) use a
          4096-row compacting window: RTP-type words are evaluated at
          wpos = min(pos, 2047), the K/V write row is q = pos - base(pos), and
          the K/V read base starts at row max(0, q - 2047). Full layers
          (block3/4) are linear over 262144 rows. RoPE reads are constant
          (the host writes the current row into the table slot). Sample record
          offsets are a 2048-entry ring so they stay inside view348.

Word semantics (from holes.py, parse_ops): an op is [opcode, size, ...]. WRITE:
value at op+4. BLOCKWRITE: BD word0 length at op+4. DDR_PATCH: 64-bit plus at
op+10 (low) and op+11 (high). The varying words of every case are exactly the
holes in holes.json, so non-hole words are constant across positions.
"""

import argparse
import json
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import holes as H  # copy of npu-study/holes.py (parser and form table)  # noqa: E402

assert sys.byteorder == "little", "streams are little-endian uint32"

MAX_POS = 262143
WINDOW = 2048
SLIDING_INSTR = ("block0", "block1", "block2")
FULL_INSTR = ("block3", "block4")
FULL_LAYERS = (7, 15)              # full-attention layers (block3, block4)
POSITION_INSTR = ("block0", "block1", "block2", "block3", "block4", "block5", "inst28")
# KV DDR slot per instruction (args index): block0 uses view310 at slot 0; blocks 1-4 use slot 2.
KV_SLOT = {"block0": 0, "block1": 2, "block2": 2, "block3": 2, "block4": 2}
# pd-layout KV offsets inside each layer root (bytes)
PD_V_SLIDING = 0x100000          # sliding: K rows [0,4096) at +0, V at +1 MiB
PD_V_FULL = 0x4000000            # full: K rows [0,262144) at +0, V at +64 MiB
ROPE_FULL_OFFSET = 524288        # root0+524288 holds the full/YaRN table slot
REC_RING = 2048                  # block5 Sample records: ring over 2048 positions

WRITE_ROLE = {
    0x00301A84: "pk_k",          # K packet count = 8*blocks
    0x02201A84: "pk_v",          # V packet count = 8*blocks
    0x00409004: "qk_blocks",
    0x00409008: "qk_start",
    0x0050C000: "soft_blocks",
    0x0050C004: "soft_sq",
    0x0050C008: "soft_sk",
    0x0050C00C: "soft_start",
    0x02501804: "pv_blocks",
    0x02501808: "pv_sk",
    0x0250180C: "pv_start",
}


LAYER0_K_BASE = 3072   # layer 0 (block0) scratch occupies view310[0:3072]; K must start above it
SLIDING_KV_ROOT = 2 * 1024 * 1024


def kv_bases(layer):
    """(K base, V base) byte offsets inside the layer's KV root for the pd layout.

    DIAGNOSTIC: PD_DIAG_COMPAT_LAYOUT=1 selects the exported r34 offsets (views 310.. :
    layer 0 K 3072 / V 527360, others K 0 / V 524288) so per-layer roots can be tested with
    the exported formulas (compat mode).
    """
    import os
    if os.environ.get("PD_DIAG_COMPAT_LAYOUT"):
        return (3072, 527360) if layer == 0 else (0, 524288)
    if layer == 0:
        return LAYER0_K_BASE, LAYER0_K_BASE + PD_V_SLIDING
    if layer in FULL_LAYERS:
        return 0, PD_V_FULL
    return 0, PD_V_SLIDING


def instr_kv_bases(name):
    """(K base, V base) for an instruction's KV slot (block0 = layer 0, block3/4 = full)."""
    if name == "block0":
        return kv_bases(0)
    if name in FULL_INSTR:
        return 0, PD_V_FULL
    return kv_bases(1)


def kv_root_nbytes(layer):
    kb, vb = kv_bases(layer)
    if layer in FULL_LAYERS:
        return vb + 64 * 1024 * 1024
    return vb + 1024 * 1024


def window_base(pos):
    """Row offset of the sliding buffer: base(pos) = 2048*max(0, (pos-2048)//2048)."""
    return WINDOW * max(0, (pos - WINDOW) // WINDOW)


def sliding_row(pos):
    """Buffer row q that position pos writes (sliding layers, pd mode)."""
    return pos - window_base(pos)


def window_start(pos):
    """First buffer row of the 2048-row attention window (sliding, pd mode)."""
    return max(0, sliding_row(pos) - (WINDOW - 1))


def compaction_due(pos):
    """True when the host must move rows [2048,4096) to [0,2048) before step pos."""
    return pos >= 2 * WINDOW and pos % WINDOW == 0


def _check_pos(pos, mode):
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise ValueError("pos must be an int")
    limit = 2047 if mode == "compat" else MAX_POS
    if not 0 <= pos <= limit:
        raise ValueError(f"pos {pos} outside {mode} range 0..{limit}")


class Generator:
    """Stencil + hole formulas for the seven position-dependent instructions."""

    def __init__(self, package=H.DEFAULT_PACKAGE, mode="pd", loaded=None):
        if mode not in ("compat", "pd"):
            raise ValueError("mode must be compat or pd")
        self.mode = mode
        self.package = Path(package)
        # H2-L: `loaded` = (payload, blob) from the long package (pd-manifest.json + the r34-w4 transactions.bin); the PoC read package/manifest.json instead.
        self.payload, self.blob = loaded if loaded is not None else H.load_package(self.package)
        self.holes = json.loads((HERE / "holes.json").read_text())["holes"]
        self.stencil = {}
        self.dyn = {}  # name -> list of hole dicts
        self.extra = {}  # name -> list of pd-only DDR K/V read-base ops
        for name in POSITION_INSTR:
            raw = H.case_bytes(self.payload, self.blob, name, 0)
            self.stencil[name] = array("I", raw)
        self._relocate_rtps()
        for h in self.holes:
            self.dyn.setdefault(h["instruction"], []).append(h)
        for name in POSITION_INSTR:
            self.extra[name] = self._kv_read_ops(name) if name in KV_SLOT else []
        self.nbytes = {name: 4 * len(self.stencil[name]) for name in POSITION_INSTR}

    # ------------------------------------------------------------ RTP relocation (phase 3)
    def _relocate_rtps(self):
        """Phase 3: a rebuilt decode image with deeper attention FIFOs moves a few core-tile RTP
        buffers.  PD_RTP_MAP (json from attn/rtpmap.py) lists (col,row,old,new) for 16-byte RTP
        buffers; every WRITE op that targets one is re-addressed in the stencil.  Hole word
        indices and values are unchanged (holes.json keeps the original register names)."""
        import os
        path = os.environ.get("PD_RTP_MAP")
        self.rtp_moved = 0
        if not path:
            return
        table = json.loads(Path(path).read_text())
        for name in POSITION_INSTR:
            words, ops = H.parse_ops(self.stencil[name].tobytes())
            data = self.stencil[name]
            for op in ops:
                if op["opcode"] != 0:
                    continue
                col, row, off = H.decode_addr(op["words"][2])
                for t in table:
                    if (col, row) == (t["col"], t["row"]) and t["old"] <= off < t["old"] + t["nbytes"]:
                        data[op["at"] + 2] = (col << 25) | (row << 20) | (t["new"] + off - t["old"])
                        self.rtp_moved += 1
                        break

    # ------------------------------------------------------------ stencil checks
    def _kv_read_ops(self, name):
        """DDR_PATCH ops that set the K/V read BD base for the KV slot (not holes)."""
        slot = KV_SLOT[name]
        words, ops = H.parse_ops(self.stencil[name].tobytes())
        found = {"k": [], "v": []}
        for op in ops:
            if op["opcode"] != 129:
                continue
            w = op["words"]
            if w[8] != slot:
                continue
            if w[6] == 0x201D064:
                found["k"].append(op["at"] + 10)
            elif w[6] == 0x1D064:
                found["v"].append(op["at"] + 10)
        if len(found["k"]) != 1 or len(found["v"]) != 1:
            raise SystemExit(f"{name}: expected one K and one V read base, got {found}")
        return [("kread", found["k"][0]), ("vread", found["v"][0])]

    # ------------------------------------------------------------ hole formulas
    @staticmethod
    def _role(h):
        name, reg = h["instruction"], int(h["register"], 16)
        if name == "inst28":
            return "prefill"
        if h["op"] == "DDR_PATCH":
            if name == "block5":
                return "rec4" if h["b"] == 32768 else "rec16"
            if name == "block0" and reg == 0x0401D044:
                return "rope_s" if h["b"] == 0 else "rope_f"
            if reg == 0x0601D044:
                return "kw"
            if reg == 0x0401D024:
                return "vw"
            raise ValueError(f"unclassified DDR hole {h}")
        if h["op"] == "BLOCKWRITE":
            if reg == 0x0201D060:
                return "bdlen_k"
            if reg == 0x0001D060:
                return "bdlen_v"
            raise ValueError(f"unclassified BLOCKWRITE hole {h}")
        if h["op"] == "WRITE":
            if reg not in WRITE_ROLE:
                raise ValueError(f"unclassified WRITE hole {h}")
            return WRITE_ROLE[reg]
        raise ValueError(f"unclassified hole {h}")

    @staticmethod
    def _eval(h, x):
        return h["a"] * H.POS_FORMS[h["form"]](x) + h["b"]

    def values(self, name, pos):
        """Return [(word_index, value, width)] for the hole words of `name` at `pos`.

        width 1 is a 32-bit word, width 2 a 64-bit DDR plus (low word first).
        """
        mode = self.mode
        _check_pos(pos, mode)
        if name not in POSITION_INSTR:
            raise ValueError(f"{name} is not position-dependent")
        if mode == "pd" and name == "inst28":
            raise ValueError("inst28 is prefill-only and has no pd-mode stream")
        sliding = name in SLIDING_INSTR
        out = []
        for h in self.dyn.get(name, []):
            role = self._role(h)
            width = 2 if h["op"] == "DDR_PATCH" else 1
            if mode == "compat":
                value = self._eval(h, pos)
            elif role in ("kw", "vw"):
                x = sliding_row(pos) if sliding else pos
                kb, vb = instr_kv_bases(name)
                value = (kb if role == "kw" else vb) + 256 * x
            elif role in ("rope_s", "rope_f"):
                # Exported table row pos for pos < 2048 (exact, no host write). For pos >= 2048
                # the host writes row pos into table slot 2047 (pd_runtime) and the read uses it.
                base = 0 if role == "rope_s" else ROPE_FULL_OFFSET
                value = base + 256 * (pos if pos < 2048 else 2047)
            elif role in ("rec4", "rec16"):
                value = self._eval(h, pos % REC_RING)
            elif role == "prefill":
                raise AssertionError("unreachable")
            else:  # RTP roles and BD lengths
                x = min(pos, WINDOW - 1) if sliding else pos
                value = self._eval(h, x)
            if value < 0 or value >= (1 << (32 * width)):
                raise ValueError(f"{name} word {h['word_index']} value {value} out of range")
            out.append((h["word_index"], value, width))
        if mode == "pd" and name in KV_SLOT:
            start = window_start(pos) if sliding else 0
            kb, vb = instr_kv_bases(name)
            for kind, index in self.extra[name]:
                if kind == "kread":
                    value = kb + 256 * start
                else:
                    value = vb + 256 * start if sliding else vb
                out.append((index, value, 2))
        return out

    def words(self, name, pos):
        """Full uint32 stream of `name` at `pos` (a fresh array)."""
        data = array("I", self.stencil[name])
        for index, value, width in self.values(name, pos):
            data[index] = value & 0xFFFFFFFF
            if width == 2:
                if data[index + 1] != 0:
                    raise AssertionError(f"{name}: 64-bit plus high word not zero in stencil")
                data[index + 1] = value >> 32
        return data

    def bytes(self, name, pos):
        return self.words(name, pos).tobytes()


# ------------------------------------------------------------------- T1 test
def run_t1(package=H.DEFAULT_PACKAGE, limit=2048, verbose=True):
    """T1: compat-mode output equals every exported case for pos 0..limit-1."""
    gen = Generator(package, mode="compat")
    report = {}
    for name in POSITION_INSTR:
        exported_count = 0
        mismatches = []
        for pos in range(limit):
            if str(pos) not in gen.payload["instructions"][name]["cases"]:
                raise SystemExit(f"{name}: no exported case {pos}")
            want = H.case_bytes(gen.payload, gen.blob, name, pos)
            got = gen.bytes(name, pos)
            exported_count += 1
            if got != want:
                mismatches.append(pos)
                if len(mismatches) >= 5:
                    break
        report[name] = {"cases_checked": exported_count, "mismatches": mismatches[:5],
                        "pass": not mismatches}
        if verbose:
            print(f"T1 {name}: checked={exported_count} mismatches={len(mismatches)} "
                  f"{'PASS' if not mismatches else 'FAIL'}", flush=True)
    ok = all(v["pass"] for v in report.values())
    if verbose:
        print("T1", "PASS" if ok else "FAIL", flush=True)
    return ok, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--t1", action="store_true", help="run T1 (compat vs exported, pos 0..2047)")
    ap.add_argument("--limit", type=int, default=2048)
    ap.add_argument("--package", type=Path, default=H.DEFAULT_PACKAGE)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.t1:
        ok, report = run_t1(args.package, args.limit)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2))
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
