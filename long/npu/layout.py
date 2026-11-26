#!/usr/bin/env python3
"""PD-256k memory layout and PoC manifest builder (copy of the r34-w4 payload, never edited).

Layout (per layer, one root BO each; sizes in bytes)
  full layers 7, 15      pdkv{L}: 128 MiB. K rows 0..262143 at +0 (64 MiB), V at +64 MiB.
  sliding layers (rest)  pdkv{L}: 2 MiB.  K rows 0..4095 at +0 (1 MiB), V at +1 MiB.
  Each decode step args[slot] for layer L is pdkv{L}: slot 0 for block0 (layer 0),
  slot 2 for blocks 1-4. The DDR_PATCH plus constants are rewritten by gen.py in pd mode
  to match these offsets.
Unchanged: root0..root7 (weights, RoPE tables at root0+0 and root0+524288, block5 Sample
records in view348, which pd mode rings over 2048 entries), contexts, instructions, and
the exported prefill graphs (they still write the r34 KV layout in root0, and
pd_runtime.bridge_old_kv copies those rows into pdkv{L} once after prefill).

Outputs (under pd-package/ next to this file, never under runs/upload):
  manifest.json       PoC payload (version 1, decode graph position range [0, 262143],
                      max_seq 262144, no seal, no production validation lease)
  artifact-paths.json artifact name -> local path (laptop paths from the r34 manifest)
"""

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen as G  # noqa: E402

SRC = G.H.DEFAULT_PACKAGE  # workstation r34-w4 package (read only)
MAX_POS = G.MAX_POS
FULL_KV_BYTES = 128 * 1024 * 1024       # K 64 MiB + V 64 MiB
SLIDING_KV_BYTES = 2 * 1024 * 1024      # K 1 MiB + V 1 MiB
FULL_LAYERS = G.FULL_LAYERS
KV_STEP_SLOT = {0: 0}                   # layer 0 uses slot 0; others slot 2
OLD_KV = {}                             # filled from the r34 decode stencil in build()


def kv_layer_kind(layer):
    return "full" if layer in FULL_LAYERS else "sliding"


def kv_root_name(layer):
    return f"pdkv{layer}"


def kv_nbytes(layer):
    return G.kv_root_nbytes(layer)


def build(src=SRC, out_dir=HERE / "pd-package"):
    src = Path(src)
    raw = json.loads((src / "manifest.json").read_text())
    payload = copy.deepcopy(raw["payload"])
    views = payload["views"]
    roots = payload["roots"]
    graphs = payload["graphs"]
    decode = graphs["decode"]
    steps = decode["steps"]
    if len(steps) != 17:
        raise SystemExit("unexpected decode step count")

    old_kv_views = {}
    for layer in range(16):
        step = steps[layer]
        slot = KV_STEP_SLOT.get(layer, 2)
        old_view = step["args"][slot]
        if old_view not in views:
            raise SystemExit(f"layer {layer}: view {old_view} missing")
        old_kv_views[layer] = old_view
        name = kv_root_name(layer)
        if name in roots or name in views:
            raise SystemExit(f"{name} already exists")
        roots[name] = {"nbytes": kv_nbytes(layer), "flags": "host_only", "group_id": 0}
        views[name] = {"root": name, "offset": 0, "nbytes": kv_nbytes(layer)}
        step["args"][slot] = name

    # The decode embedding input (view0) aliased view310[0:3072] in the exported layout: block0
    # (slot 0 = layer-0 buffer) reads its embedding there and uses that region for scratch.
    # Layer 0 now lives in pdkv0, so the embedding must live at pdkv0[0:3072] too.
    if views["view0"]["root"] != "root0" or views["view0"]["offset"] != views["view310"]["offset"]:
        raise SystemExit("exported view0 no longer aliases view310")
    views["view0"] = {"root": kv_root_name(0), "offset": 0, "nbytes": views["view0"]["nbytes"]}

    # Re-point the other slots of the RoPE/KV steps untouched; verify slot-2 KV was the old KV view.
    decode["parameter_ranges"]["position"] = [0, MAX_POS]
    payload["model"]["max_seq"] = MAX_POS + 1

    # Keep the exported prefill graphs: they write the r34 KV layout (old views in root0),
    # which pd_runtime.bridge_old_kv copies into pdkv{L} after each prefill.
    payload["pd_layout"] = {
        "version": 1,
        "kv_roots": {str(l): {"root": kv_root_name(l), "kind": kv_layer_kind(l),
                               "nbytes": kv_nbytes(l)} for l in range(16)},
        "sliding": {"row_bytes": 256, "rows": 4096, "v_offset": G.PD_V_SLIDING,
                    "window": 2048, "compaction_every": 2048, "first_compaction_pos": 4096},
        "full": {"rows": 262144, "v_offset": G.PD_V_FULL},
        "rope_slots": {"sliding": 0, "full": G.ROPE_FULL_OFFSET,
                       "views": ["view2", "view4"], "rows_per_step": 1},
        "block5_record_ring": G.REC_RING,
        "old_kv_views": old_kv_views,
        "source_manifest_sha256": raw.get("sha256"),
        "generator": "poc/pd256k/d/gen.py (mode pd)",
    }
    payload["version"] = 1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=1))
    paths = {name: item["path"] for name, item in payload["artifacts"].items()}
    (out_dir / "artifact-paths.json").write_text(json.dumps(paths, indent=1))
    return payload, old_kv_views


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=HERE / "pd-package")
    args = ap.parse_args(argv)
    payload, old = build(args.src, args.out)
    print(json.dumps({"kv_roots": len(payload["pd_layout"]["kv_roots"]),
                      "old_kv_views": old, "decode_range": payload["graphs"]["decode"]["parameter_ranges"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
