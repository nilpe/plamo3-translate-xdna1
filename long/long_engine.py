# SPDX-License-Identifier: Apache-2.0
"""PLaMo 3 long mode, iGPU prefill step (Radeon 780M, gfx1103): hardened descendant of the PD-256k PoC engine (poc/pd256k/p/engine.py, v2 kernels).

Runs as a child process (python -I long_engine.py job.json) so that a driver fault or memory problem cannot take the NPU step down with it.
Differences from the PoC: HIP access and detection in long_hip, progress events, cooperative cancel (between chunks), full-attention dispatch splitting by
query-row tiles with a per-dispatch time cap (DispatchGovernor; every split boundary is a multiple of the kernel's block rows, so the output is bitwise identical),
optional per-kernel timing (H2L_KTIME), no v1 kernels, no dump-rows debugging. Arithmetic and kernels are the PoC's, unchanged.
"""
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_core import CHUNK, DispatchGovernor, LongModeError, Cancelled, errors
from long_hip import Hip

D = 1536
NH = 16
HD = 128
QKV = 2304
INTER = 6144
NLAYERS = 16
WINDOW = 2048
RING = 4096
VOCAB = 107520
EPS = 1e-6
ROW_BYTES = HD * 2


def is_full(i):
    return (i + 1) % 8 == 0


# ---------------------------------------------------------------- numerics helpers
def f32_to_bf16_bits(x):
    """Round-to-nearest-even fp32 -> BF16 bit pattern (uint16), NaN-safe for finite inputs."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return r.astype(np.uint16)


def rope_tables(base, factor, n_rows, yarn):
    """cos/sin BF16 bit tables [n_rows, 64] without concentration (mirrors RotaryEmbedding in fp32)."""
    f32 = np.float32
    dim, dh = HD, HD // 2
    freq = np.power(f32(base), np.arange(0, dim, 2, dtype=f32) / f32(dim)).astype(f32)
    if not yarn:
        inv = (f32(1.0) / freq).astype(f32)
    else:
        orig, beta_fast, beta_slow = 4096.0, 32.0, 1.0
        low = dh * math.log(orig / (beta_fast * 2 * math.pi)) / math.log(base)
        high = dh * math.log(orig / (beta_slow * 2 * math.pi)) / math.log(base)
        interp = (f32(1.0) / (f32(factor) * freq)).astype(f32)
        extrap = (f32(1.0) / freq).astype(f32)
        ramp = ((np.arange(dh, dtype=f32) - f32(low)) / f32(high - low)).astype(f32)
        mask = (f32(1.0) - np.clip(ramp, f32(0.0), f32(1.0))).astype(f32)
        inv = (interp * (f32(1.0) - mask) + extrap * mask).astype(f32)
    cos_t = np.empty((n_rows, dh), dtype=np.uint16)
    sin_t = np.empty((n_rows, dh), dtype=np.uint16)
    step = 32768
    for p0 in range(0, n_rows, step):
        p1 = min(n_rows, p0 + step)
        t = np.arange(p0, p1, dtype=f32)
        ang = (t[:, None] * inv[None, :]).astype(f32)
        cos_t[p0:p1] = f32_to_bf16_bits(np.cos(ang).astype(f32))
        sin_t[p0:p1] = f32_to_bf16_bits(np.sin(ang).astype(f32))
    return cos_t, sin_t


class Safetensors:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.base = 8 + n

    def read_u16(self, name):
        info = self.hdr[name]
        assert info["dtype"] == "BF16", info["dtype"]
        s, e = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + s)
            raw = f.read(e - s)
        return np.frombuffer(raw, dtype=np.uint16).reshape(info["shape"]).copy()


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, hip, co_path, model_dir, n_max, chunk, log, wmma_co=None, v2=None, governor=None):
        self.hip = hip
        self.v2 = v2  # dict(attn, attn_nw, fused, gemm, attn_full, attn_full_nw)
        self.gov = governor
        assert v2 is not None
        # WMMA kernels live in a second code object on the same device (second Hip handle).
        self.hw = Hip(lib=hip.h)
        self.hw.load_module(wmma_co or os.path.join(os.path.dirname(co_path), 'plamo3_wmma.co'))
        self.log = log
        self.chunk = chunk
        self.nrows = 1
        while self.nrows < n_max:
            self.nrows *= 2
        self.full_mask = self.nrows - 1
        self.ring_mask = RING - 1
        hip.load_module(co_path)
        st = Safetensors(os.path.join(model_dir, "model.safetensors"))
        self.embed = st.read_u16("model.embed_tokens.weight")  # host only
        self.w = []
        for i in range(NLAYERS):
            p = f"model.layers.layers.{i}."
            lw = {}
            lw["pre_mixer"] = self._up(st, p + "pre_mixer_norm.weight")
            lw["post_mixer"] = self._up(st, p + "post_mixer_norm.weight")
            lw["pre_mlp"] = self._up(st, p + "pre_mlp_norm.weight")
            lw["post_mlp"] = self._up(st, p + "post_mlp_norm.weight")
            lw["qnorm"] = self._up(st, p + "mixer.q_norm.weight")
            lw["knorm"] = self._up(st, p + "mixer.k_norm.weight")
            lw["qkv"] = self._up(st, p + "mixer.qkv_proj.weight")
            lw["o"] = self._up(st, p + "mixer.o_proj.weight")
            lw["gu"] = self._up(st, p + "mlp.gate_up_proj.weight")
            lw["down"] = self._up(st, p + "mlp.down_proj.weight")
            self.w.append(lw)
        self.final_norm = self._up(st, "model.norm.weight")
        self.lm_head = self._up(st, "lm_head.weight")
        # RoPE tables (BF16 bits) for every position up to nrows.
        cf, sf = rope_tables(1e6, 256.0, self.nrows, yarn=True)
        cs, ss = rope_tables(1e4, 1.0, self.nrows, yarn=False)
        self.cos_full, self.sin_full = self._up_arr(cf), self._up_arr(sf)
        self.cos_slide, self.sin_slide = self._up_arr(cs), self._up_arr(ss)
        self.conc_full = 0.1 * math.log(256.0) + 1.0
        # KV caches: full layers hold nrows rows; sliding layers a RING-row ring.
        self.kc, self.vc = {}, {}
        for i in range(NLAYERS):
            rows = self.nrows if is_full(i) else RING
            self.kc[i] = hip.malloc(rows * ROW_BYTES)
            self.vc[i] = hip.malloc(rows * ROW_BYTES)
            if v2:
                hip.memset(self.kc[i], 0, rows * ROW_BYTES)
                hip.memset(self.vc[i], 0, rows * ROW_BYTES)
        if v2:
            self.vct = {}
            for i in range(NLAYERS):
                rows = self.nrows if is_full(i) else RING
                self.vct[i] = hip.malloc(rows * ROW_BYTES)
                hip.memset(self.vct[i], 0, rows * ROW_BYTES)
            self.ha = Hip(lib=hip.h); self.ha.load_module(v2["attn"])
            if v2.get("attn_full"):
                self.haf = Hip(lib=hip.h); self.haf.load_module(v2["attn_full"])
            else:
                self.haf = self.ha
            self.hf = Hip(lib=hip.h); self.hf.load_module(v2["fused"])
            self.hg = Hip(lib=hip.h); self.hg.load_module(v2["gemm"])
            for i in range(NLAYERS):   # interleave gate/up rows per 128-row tile: [gate f0..f15 | up f0..f15] x 4 per tile
                gu = st.read_u16(f"model.layers.layers.{i}.mlp.gate_up_proj.weight")
                g, u = gu[:INTER].reshape(INTER // 16, 16, D), gu[INTER:].reshape(INTER // 16, 16, D)
                perm = np.stack([g, u], axis=1).reshape(2 * INTER, D)   # block b: gate rows of feats 16b..16b+15, then up rows
                self.hip.h2d(self.w[i]["gu"], np.ascontiguousarray(perm).ctypes.data, perm.nbytes)
        # Activations for one chunk (max rows = chunk).
        c = chunk
        self.A = {
            "x": hip.malloc(c * D * 2), "x2": hip.malloc(c * D * 2),
            "h": hip.malloc(c * D * 2), "qkv": hip.malloc(c * QKV * 2),
            "q": hip.malloc(c * NH * HD * 2), "att": hip.malloc(c * NH * HD * 2),
            "o": hip.malloc(c * D * 2), "o2": hip.malloc(c * D * 2),
            "gu": hip.malloc(c * 2 * INTER * 2), "act": hip.malloc(c * INTER * 2),
            "d": hip.malloc(c * D * 2), "d2": hip.malloc(c * D * 2),
            "lh": hip.malloc(D * 2), "logits": hip.malloc(VOCAB * 4),
        }
        self.host_bytes = 0

    def _up(self, st, name):
        arr = st.read_u16(name)
        return self._up_arr(arr)

    def _up_arr(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.uint16)
        dev = self.hip.malloc(arr.nbytes)
        self.hip.h2d(dev, arr.ctypes.data, arr.nbytes)
        return dev

    def _p(self, a, off=0):
        return ("p", a + off)

    def _chunk_forward2(self, toks, pos0, final):
        hip, A, C = self.hip, self.A, len(toks)
        ha, hf, hg = self.ha, self.hf, self.hg
        nw = self.v2["attn_nw"]
        x_host = np.ascontiguousarray(self.embed[toks]).view(np.uint16)
        hip.h2d(A["x"], x_host.ctypes.data, x_host.nbytes)
        rg = ((C + 7) // 8, 1)
        mg = (C + 127) // 128
        hf.launch("rms_row", rg, 256, [self._p(A["x"]), self._p(self.w[0]["pre_mixer"]), self._p(A["h"]), ("i", C), ("f", EPS)])
        for i in range(NLAYERS):
            w = self.w[i]
            full = is_full(i)
            kc, vc = self.kc[i], self.vc[i]
            slot_mask = self.full_mask if full else self.ring_mask
            cos_t = self.cos_full if full else self.cos_slide
            sin_t = self.sin_full if full else self.sin_slide
            conc = self.conc_full if full else 1.0
            self.hw.launch("gemm_wmma", (QKV // 128, mg), 256,
                           [self._p(A["h"]), self._p(w["qkv"]), self._p(A["qkv"]), ("i", C), ("i", QKV), ("i", D)])
            hip.launch("qk_prep", (C, NH + 2), 128,
                       [self._p(A["qkv"]), self._p(w["qnorm"]), self._p(w["knorm"]), self._p(cos_t), self._p(sin_t),
                        ("f", conc), ("f", EPS), ("i", pos0), ("i", slot_mask),
                        self._p(A["q"]), self._p(kc), self._p(vc)])
            ha.launch("vt_write", (C, 1), 128, [self._p(vc), self._p(self.vct[i]), ("i", pos0), ("i", slot_mask)])
            if full and self.v2.get("attn_full"):
                ah, anw = self.haf, self.v2["attn_full_nw"]
            else:
                ah, anw = ha, nw
            self._attention(ah, anw, i, C, pos0, full, slot_mask, kc)
            self.hw.launch("gemm_wmma", (D // 128, mg), 256,
                           [self._p(A["att"]), self._p(w["o"]), self._p(A["o"]), ("i", C), ("i", D), ("i", NH * HD)])
            hf.launch("norm_add_norm", rg, 256,
                      [self._p(A["o"]), self._p(A["x"]), self._p(w["post_mixer"]), ("f", 0.2), self._p(w["pre_mlp"]),
                       self._p(A["x2"]), self._p(A["h"]), ("i", C), ("f", EPS)])
            hg.launch("gemm_wmma_swiglu", ((2 * INTER) // 128, mg), 256,
                      [self._p(A["h"]), self._p(w["gu"]), self._p(A["act"]), ("i", C), ("i", 2 * INTER), ("i", D)])
            hg.launch("gemm_wmma", (D // 128, mg), 256,
                      [self._p(A["act"]), self._p(w["down"]), self._p(A["d"]), ("i", C), ("i", D), ("i", INTER)])
            nxt = self.w[i + 1]["pre_mixer"] if i + 1 < NLAYERS else 0
            hf.launch("norm_add_norm", rg, 256,
                      [self._p(A["d"]), self._p(A["x2"]), self._p(w["post_mlp"]), ("f", 1.0 / (5 ** 1.5)), ("p", nxt),
                       self._p(A["x"]), self._p(A["h"]), ("i", C), ("f", EPS)])
        if final:
            last = (C - 1) * D * 2
            hip.launch("rmsnorm_bf16", (1, 1), 256,
                       [self._p(A["x"], last), self._p(self.final_norm), self._p(A["lh"]), ("i", D), ("f", 1.0), ("f", EPS)])
            hip.launch("gemm_nt", (VOCAB // 64, 1), 256,
                       [self._p(A["lh"]), self._p(self.lm_head), self._p(A["logits"]), ("i", 1), ("i", VOCAB), ("i", D), ("i", 1)])

    def _attention(self, ah, anw, layer, C, pos0, full, slot_mask, kc):
        """attn2 for rows pos0..pos0+C-1. Full layers are split into query-row tiles when the predicted dispatch time exceeds the cap; the first launch of
        each chunk's first full layer is timed (sync before/after) to keep the prediction current. Sliding layers (bounded by the 2048-row window, about 20 ms) are never split."""
        A = self.A
        scale = 1.0 / math.sqrt(HD)
        gov = self.gov if (full and self.gov is not None) else None
        plan = gov.plan(C, anw) if gov is not None else [(0, C)]
        measure = gov is not None and layer == min(i for i in range(NLAYERS) if is_full(i)) and gov.want_measure()
        for k, (r0, rows) in enumerate(plan):
            args = [self._p(A["q"], r0 * NH * HD * 2), self._p(kc), self._p(self.vct[layer]), self._p(A["att"], r0 * NH * HD * 2), ("i", rows), ("i", pos0 + r0),
                    ("i", 1 if full else 0), ("i", slot_mask), ("f", scale)]
            if measure and k == 0:
                self.hip.sync()
                t0 = time.perf_counter()
                ah.launch("attn2", ((rows + anw - 1) // anw, 1), 32 * anw, args)
                self.hip.sync()
                gov.observe(rows, (time.perf_counter() - t0) * 1000.0)
            else:
                ah.launch("attn2", ((rows + anw - 1) // anw, 1), 32 * anw, args)

    def _download_rows(self, dev, row0, nrows):
        buf = np.empty((nrows, HD), dtype=np.uint16)
        self.hip.d2h(buf.ctypes.data, dev + row0 * ROW_BYTES, nrows * ROW_BYTES)
        return buf

    def _upload_rows(self, dev, row0, arr):
        arr = np.ascontiguousarray(arr, dtype=np.uint16)
        self.hip.h2d(dev + row0 * ROW_BYTES, arr.ctypes.data, arr.nbytes)




def kv_path(out, kind, i):
    return os.path.join(out, "kv", f"{kind}_{i}.bin")


def device_need_mb(n):
    """Device memory estimate: measured peak 2.65 GB at 262,144 tokens; weights 1.5 GB + about 3.3 KB per KV row (rows rounded up to a power of two) + 0.5 GB margin."""
    rows = 1
    while rows < n:
        rows *= 2
    return int(2000 + rows * 0.0033)


class _KernelTimer:
    """H2L_KTIME=<json>: sync after every launch and record count/total/max per kernel (diagnostic; slows nothing else)."""

    def __init__(self, path):
        self.path, self.stats = path, {}
        orig = Hip.launch

        def timed(hip, name, grid, block, args):
            t0 = time.perf_counter()
            orig(hip, name, grid, block, args)
            hip.sync()
            r = self.stats.setdefault(name, [0, 0.0, 0.0])
            ms = (time.perf_counter() - t0) * 1000.0
            r[0] += 1; r[1] += ms; r[2] = max(r[2], ms)
        Hip.launch = timed

    def dump(self):
        with open(self.path, "w") as f:
            json.dump({k: {"n": v[0], "total_ms": v[1], "max_ms": v[2]} for k, v in self.stats.items()}, f, indent=1)


def prefill(tokens, out, model_dir, co_dir, *, chunk=CHUNK, on_chunk=None, cancelled=lambda: False, cap_ms=200.0, split=True, hip=None, lib=None):
    """Chunked iGPU prefill of `tokens` into `out` (kv files + meta.json + last_logits.bin, the KV dump contract). Resumes from out/state.json.
    on_chunk(pos_done, n, ms) is called after every chunk; cancelled() is polled between chunks (raises Cancelled). Returns a summary dict."""
    n = len(tokens)
    os.makedirs(os.path.join(out, "kv"), exist_ok=True)
    os.makedirs(os.path.join(out, "ring"), exist_ok=True)
    t_proc = time.time()
    hip = hip or Hip(lib=lib)
    co = lambda name: os.path.join(co_dir, name)
    v2 = dict(attn=co("attn2.co"), attn_nw=8, fused=co("plamo3_fused.co"), gemm=co("plamo3_gemm2.co"), attn_full=co("attn2_full.co"), attn_full_nw=16)
    gov = DispatchGovernor(cap_ms=cap_ms) if split else None
    eng = Engine(hip, co("plamo3_prefill.co"), model_dir, n, chunk, None, wmma_co=co("plamo3_wmma.co"), v2=v2, governor=gov)
    t_init = time.time() - t_proc
    state_path = os.path.join(out, "state.json")
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        if state["n"] != n or state["chunk"] != chunk:
            raise LongModeError("the prefill work directory belongs to a different request")
    else:
        state = {"n": n, "chunk": chunk, "next_pos": 0}
        for i in range(NLAYERS):
            if is_full(i):
                for kind in ("k", "v"):
                    open(kv_path(out, kind, i), "wb").close()
    s0 = state["next_pos"]
    if s0 > 0:
        for i in range(NLAYERS):
            if is_full(i):
                for kind, dev in (("k", eng.kc[i]), ("v", eng.vc[i])):
                    raw = np.fromfile(kv_path(out, kind, i), dtype=np.uint16, count=s0 * HD)
                    if raw.size != s0 * HD:
                        raise LongModeError("the prefill work directory is damaged")
                    eng._upload_rows(dev, 0, raw.reshape(s0, HD))
            else:
                for kind, dev in (("k", eng.kc[i]), ("v", eng.vc[i])):
                    ring = np.fromfile(os.path.join(out, "ring", f"{kind}_{i}.bin"), dtype=np.uint16)
                    eng._upload_rows(dev, 0, ring.reshape(RING, HD))
        for i in range(NLAYERS):
            cnt = s0 if is_full(i) else RING
            m_ = eng.full_mask if is_full(i) else eng.ring_mask
            eng.ha.launch("vt_write", (cnt, 1), 128, [("p", eng.vc[i]), ("p", eng.vct[i]), ("i", 0), ("i", m_)])
        hip.sync()
    fh = {(kind, i): open(kv_path(out, kind, i), "r+b") for i in range(NLAYERS) if is_full(i) for kind in ("k", "v")}
    t_loop = time.time()
    s = s0
    chunk_ms = []
    try:
        while s < n:
            if cancelled():
                raise Cancelled("cancelled")
            C = min(chunk, n - s)
            t0 = time.time()
            eng._chunk_forward2(tokens[s:s + C], s, s + C == n)
            hip.sync()
            ms = (time.time() - t0) * 1e3
            for i in range(NLAYERS):
                if is_full(i):
                    for kind, dev in (("k", eng.kc[i]), ("v", eng.vc[i])):
                        f = fh[(kind, i)]
                        f.seek(s * ROW_BYTES)
                        f.write(eng._download_rows(dev, s, C).tobytes())
            for f in fh.values():
                f.flush()
            s += C
            chunk_ms.append(ms)
            if on_chunk:
                on_chunk(s, n, ms)
        # checkpoint rings + state (resume support), then the dump contract
        for i in range(NLAYERS):
            if not is_full(i):
                for kind, dev in (("k", eng.kc[i]), ("v", eng.vc[i])):
                    with open(os.path.join(out, "ring", f"{kind}_{i}.bin"), "wb") as f:
                        f.write(eng._download_rows(dev, 0, RING).tobytes())
    finally:
        for f in fh.values():
            f.close()
        state["next_pos"] = s
        tmp = state_path + ".tmp"
        json.dump(state, open(tmp, "w"))
        os.replace(tmp, state_path)
    argmax = finalize(eng, hip, out, n)
    return {"n": n, "from_pos": s0, "chunks": len(chunk_ms), "loop_s": round(time.time() - t_loop, 3), "init_s": round(t_init, 3), "total_s": round(time.time() - t_proc, 3),
            "argmax_last": argmax, "max_chunk_ms": max(chunk_ms) if chunk_ms else 0.0, "gov": {"splits": gov.splits if gov else 0, "max_launch_ms": round(gov.max_launch_ms, 1) if gov else 0.0,
            "per_row_ms": gov.per_row_ms if gov else None}}


def finalize(eng, hip, out, n):
    first = max(0, n - WINDOW)
    rows_idx = np.arange(first, n) & (RING - 1)
    meta_layers = []
    for i in range(NLAYERS):
        if is_full(i):
            kind_name, first_row, rows = "full", 0, n
        else:
            kind_name, first_row, rows = "sliding", first, n - first
            for kind, dev in (("k", eng.kc[i]), ("v", eng.vc[i])):
                ring = eng._download_rows(dev, 0, RING)
                with open(kv_path(out, kind, i), "wb") as f:
                    f.write(ring[rows_idx].tobytes())
        meta_layers.append({"layer": i, "kind": kind_name, "first_row": first_row, "rows": rows})
    lg = np.empty(VOCAB, dtype=np.float32)
    hip.d2h(lg.ctypes.data, eng.A["logits"], lg.nbytes)
    lh = np.empty(D, dtype=np.uint16)
    hip.d2h(lh.ctypes.data, eng.A["lh"], lh.nbytes)
    with open(os.path.join(out, "last_logits.bin"), "wb") as f:
        f.write(lg.tobytes())
    with open(os.path.join(out, "last_hidden.bin"), "wb") as f:
        f.write(lh.tobytes())
    meta = {"n_tokens": n, "dtype": "bf16 (kv, last_hidden); fp32 (last_logits)", "head_dim": HD, "n_kv_heads": 1, "n_layers": NLAYERS, "window": WINDOW,
            "layer_kinds": ["full" if is_full(i) else "sliding" for i in range(NLAYERS)], "layers": meta_layers, "vocab": VOCAB, "argmax_last": int(np.argmax(lg)),
            "source": "long_engine.py (HIP v2 kernels)"}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta["argmax_last"]


def main(argv):
    """Child entry: argv[1] = job.json {tokens_npy, out, model_dir, co_dir, cancel_file, split, cap_ms, ktime}. Events as JSON lines on stdout; exit code = error code."""
    job = json.load(open(argv[1]))
    def emit(**kw):
        print(json.dumps(kw), flush=True)
    kt = _KernelTimer(job["ktime"]) if job.get("ktime") else None
    try:
        tokens = np.load(job["tokens_npy"]).astype(np.int64)
        from long_hip import detect
        info = detect(job["co_dir"], need_mb=device_need_mb(len(tokens)))
        emit(ev="device", **info)
        cancel_file = job.get("cancel_file")
        res = prefill(tokens, job["out"], job["model_dir"], job["co_dir"], cap_ms=job.get("cap_ms", 200.0), split=job.get("split", True),
                      on_chunk=lambda s, n, ms: emit(ev="chunk", pos=s, n=n, ms=round(ms, 1)),
                      cancelled=lambda: bool(cancel_file) and os.path.exists(cancel_file))
        emit(ev="done", **res)
        return 0
    except Cancelled:
        emit(ev="cancelled")
        return errors.INTERRUPTED
    except errors.PlamoError as error:
        emit(ev="error", code=error.code, message=error.message)
        return error.code
    except MemoryError:
        emit(ev="error", code=errors.ENVIRONMENT, message="the machine ran out of memory during the iGPU prefill; close other programs and try again")
        return errors.ENVIRONMENT
    except Exception as error:  # never a traceback to the user
        emit(ev="error", code=errors.INTERNAL, message="internal error in the iGPU prefill: %s: %s" % (type(error).__name__, " ".join(str(error).split())[:200]))
        return errors.INTERNAL
    finally:
        if kt:
            kt.dump()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
