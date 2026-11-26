#!/usr/bin/env python3
"""Host RoPE rows for PD-256k: sliding (theta 1e4) and full/YaRN (theta 1e6, factor 256).

Each row is 128 BF16 values, interleaved (cos, sin) for 64 frequency pairs, exactly
the format of the r34-w4 tables (root0+0 = sliding, root0+524288 = full, 2048 rows
of 256 bytes each).

The arithmetic mirrors iron config.py `angles()` and `rope_config()` operation by
operation: float32 inverse frequencies, float32 phase = pos * inv_freq (one float32
multiply per element, same as np.outer), numpy float32 cos/sin times the scale, then
one round-to-nearest-even to BF16 (same as order_reference.bf).

T2 (`python3 rope.py --t2`) checks rows 0..2047 bitwise against initial.bin.
T3 (`python3 rope.py --t3`) reports bit differences against the HF
RotaryEmbedding in modeling_plamo.py at pos 65536 and 262143 (diagnostic only).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json" if (HERE / "config.json").exists() else HERE.parent / "p" / "model" / "config.json"
PACKAGE = Path("~/workspace/plamo3-fast-inference-by-xdna/runs/upload/hf/npu/plamo3-runtime/r34-w4").expanduser()
ROPE_WIDTH = 128
DIM = ROPE_WIDTH // 2
ROW_BYTES = 256


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def inv_freq(full, cfg=None):
    """float32 inverse frequencies, as config.angles() builds them."""
    cfg = cfg or load_config()
    kind = "full_attention" if full else "sliding_attention"
    p = cfg["rope_parameters"][kind]
    theta = p["rope_theta"]
    exponents = np.arange(0, ROPE_WIDTH, 2, dtype=np.float32) / ROPE_WIDTH
    freq = np.power(np.float32(theta), exponents)
    inv = 1.0 / freq
    if not full:
        return inv.astype(np.float32), 1.0
    if p["rope_type"] != "yarn" or p.get("truncate", False):
        raise ValueError("full RoPE must be YaRN with truncate=false")
    factor = p["factor"]
    original = p["original_max_position_embeddings"]
    low = DIM * math.log(original / (p["beta_fast"] * 2 * math.pi)) / math.log(theta)
    high = DIM * math.log(original / (p["beta_slow"] * 2 * math.pi)) / math.log(theta)
    if not 0 < low < high < DIM - 1:
        raise ValueError("YaRN ramp lies outside the rotary dimensions")
    mask = 1 - np.clip((np.arange(DIM, dtype=np.float32) - low) / (high - low), 0, 1)
    # iron applies the YaRN blend to the inverse frequencies (rope_scaling(inv_freq)).
    inv = inv / factor * (1 - mask) + inv * mask
    scale = 0.1 * math.log(factor) + 1
    return inv.astype(np.float32), scale


_CACHE = {}


def _tables(full):
    if full not in _CACHE:
        _CACHE[full] = inv_freq(full)
    return _CACHE[full]


def rows_f32(positions, full):
    """float32 values (pre-BF16) for an array of positions, shape (n, 128)."""
    inv, scale = _tables(full)
    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    phase = np.outer(pos, inv)
    out = np.empty((len(pos), ROPE_WIDTH), dtype=np.float32)
    out[:, ::2] = np.cos(phase) * scale
    out[:, 1::2] = np.sin(phase) * scale
    return out


def f32_to_bf16_bits(x):
    """Round-to-nearest-even float32 -> BF16 bit pattern (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def row_bits(pos, full):
    """BF16 bit pattern of one RoPE row, shape (128,) uint16."""
    return f32_to_bf16_bits(rows_f32([pos], full))[0]


def row_bytes(pos, full):
    """256 bytes (little-endian BF16) for one row, ready to write to a table slot."""
    return row_bits(pos, full).astype("<u2").tobytes()


def row_float(pos, full):
    """float32 values of the BF16 row (for the reference rotation)."""
    return bf16_bits_to_f32(row_bits(pos, full))


def table_bits(full, count=2048):
    return f32_to_bf16_bits(rows_f32(np.arange(count), full))


# ------------------------------------------------------------------- T2 / T3
def run_t2(package=PACKAGE, verbose=True):
    """Rows 0..2047 bitwise against initial.bin root0+0 (sliding) and root0+524288 (full)."""
    path = Path(package) / "initial.bin"
    with path.open("rb") as f:
        blob = f.read(2 * 524288)
    results = {}
    for full, start in ((False, 0), (True, 524288)):
        want = np.frombuffer(blob[start : start + 524288], dtype="<u2").reshape(2048, 128)
        got = table_bits(full, 2048)
        bad = np.nonzero(np.any(got != want, axis=1))[0]
        results["full" if full else "sliding"] = {
            "rows": 2048,
            "mismatched_rows": int(len(bad)),
            "first_mismatch": int(bad[0]) if len(bad) else None,
            "pass": len(bad) == 0,
        }
        if verbose:
            print(f"T2 {'full' if full else 'sliding'}: rows=2048 mismatched={len(bad)} "
                  f"{'PASS' if not len(bad) else 'FAIL'}", flush=True)
    ok = all(r["pass"] for r in results.values())
    if verbose:
        print("T2", "PASS" if ok else "FAIL", flush=True)
    return ok, results


def hf_rows(positions, full):
    """HF RotaryEmbedding (modeling_plamo.py) cos/sin for the given positions, as BF16 bits.

    Re-implements RotaryEmbedding._compute_buffers and forward verbatim (torch),
    because the HF module pulls in transformers. Rounding sequence kept: float32
    cos/sin -> bf16 -> * concentration -> bf16.
    """
    import torch

    cfg = load_config()
    kind = "full_attention" if full else "sliding_attention"
    p = cfg["rope_parameters"][kind]
    dim = ROPE_WIDTH
    base = p["rope_theta"]
    freq = base ** (torch.arange(0, dim, 2).float() / dim)
    if not full:
        inv_freq = 1.0 / freq
        concentration = 1.0
    else:
        factor = p["factor"]
        d_h = dim // 2
        low = d_h * math.log(p["original_max_position_embeddings"] / (p["beta_fast"] * 2 * math.pi)) / math.log(base)
        high = d_h * math.log(p["original_max_position_embeddings"] / (p["beta_slow"] * 2 * math.pi)) / math.log(base)
        interpolation = 1.0 / (factor * freq)
        extrapolation = 1.0 / freq
        ramp = (torch.arange(d_h).to(torch.float32) - low) / (high - low)
        mask = 1 - torch.clamp(ramp, 0, 1)
        inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
        concentration = 0.1 * math.log(factor) + 1.0
    pos = torch.as_tensor(np.asarray(positions), dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, :].to(dtype=torch.bfloat16) * concentration
    sin = emb.sin()[:, :].to(dtype=torch.bfloat16) * concentration
    # Interleave to the NPU layout: column 2j = cos[j], 2j+1 = sin[j] (j < 64).
    c = cos[:, :64]
    s = sin[:, :64]
    inter = torch.stack((c, s), dim=-1).reshape(len(pos), 128)
    return inter.view(torch.int16).numpy().view(np.uint16)


def hf_module_rows(pos, full, hf_dir=None):
    """Rows from the real HF RotaryEmbedding class in modeling_plamo.py (needs torch+transformers)."""
    import importlib.util
    import torch

    hf_dir = Path(hf_dir or HERE.parent / "p" / "model")
    spec = importlib.util.spec_from_file_location("hf_modeling_plamo", hf_dir / "modeling_plamo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = load_config()
    kind = "full_attention" if full else "sliding_attention"
    rot = mod.RotaryEmbedding(ROPE_WIDTH, max_position_embeddings=262144,
                              parameter=cfg["rope_parameters"][kind])
    x = torch.zeros(1, 1, 1, ROPE_WIDTH, dtype=torch.bfloat16)
    cos, sin = rot(x, seq_len=pos + 1)
    c = cos[0, 0, pos, :64].float()
    s = sin[0, 0, pos, :64].float()
    inter = torch.stack((c, s), dim=-1).reshape(128).to(torch.bfloat16)
    return inter.view(torch.int16).numpy().view(np.uint16)


def run_t3(verbose=True):
    """Diagnostic: bit differences of the NPU/host rows against HF at selected positions."""
    out = {}
    for pos in (65536, 262143):
        for full in (False, True):
            ours = row_bits(pos, full)
            hf = hf_module_rows(pos, full)
            replica = hf_rows([pos], full)[0]
            key = f"pos{pos}_{'full' if full else 'sliding'}"
            out.setdefault("replica_agrees_with_hf_class", True)
            out["replica_agrees_with_hf_class"] &= bool(np.array_equal(replica, hf))
            diff = np.nonzero(ours != hf)[0]
            ours_f = bf16_bits_to_f32(ours)
            hf_f = bf16_bits_to_f32(hf)
            max_ulp = 0
            if len(diff):
                du = np.abs(ours.astype(np.int32) - hf.astype(np.int32))
                max_ulp = int(du.max())
            out[key] = {
                "differing_elements": int(len(diff)),
                "of": 128,
                "max_abs_bf16_bit_diff": max_ulp,
                "max_abs_value_diff": float(np.max(np.abs(ours_f - hf_f))),
            }
            if verbose:
                print(f"T3 {key}: differing={len(diff)}/128 max_bits={max_ulp} "
                      f"max_value_diff={out[key]['max_abs_value_diff']:.6g}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--t2", action="store_true")
    ap.add_argument("--t3", action="store_true")
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    result = {}
    ok = True
    if args.t2:
        ok, result["T2"] = run_t2(args.package)
    if args.t3:
        result["T3"] = run_t3()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
