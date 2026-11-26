#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-mode NPU gate harness (runs on the Windows machine with the installed tree; uses the SHIPPED long runtime: long_decode.LongDecoder, npu\\long_pdrt.py):

  t4    regression: prefill with the exported graph, bridge, teacher-forced decode of the 4 reference prompts (96 rows) bitwise vs reference_rows.bin   (L2)
  step  one teacher-forced decode step at --pos on an injected KV dump; writes the logits bits                                                        (L3)
  l4    one CONTINUOUS decode: iGPU prefill of --prompt tokens (the product's long_engine child), then teacher-forced decode for --steps steps through the
        sliding-window compaction points; at each --capture position the K/V window a decode there sees is dumped BEFORE the step and the step's logits are
        saved, so the CPU order-matched reference (ref_step.py) can be compared bitwise offline                                                        (L4)
The CPU reference runs on the workstation: python ref_step.py --expect --pos P --token T --kv-dump DIR --out FILE.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def open_decoder(root):
    sys.path.insert(0, str(Path(root) / 'long'))
    import long_decode
    dec = long_decode.LongDecoder(root).open()
    return long_decode, dec


def session_of(dec, runtime=None):
    return dec._session_mod.RuntimeSession(dec.runtime)


def dump_window(runtime, pos, out_dir, n_prompt=0):
    """Same extraction as the PoC dumpkv: the K/V a decode step at `pos` sees; buffers hold the layout after step pos-1."""
    import gen
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = []
    base_pos = gen.window_base(max(0, pos - 1))
    for layer in range(16):
        full = layer in gen.FULL_LAYERS
        lo = 0 if full else max(0, pos - 2047)
        v_off, k_off, view = runtime._v_offset(layer), runtime._k_base(layer), runtime._kv_view(layer)
        row0 = lo if full else lo - base_pos
        nbytes = (pos - lo) * 256

        def rd(off):
            if view in runtime._dirty:
                return bytes(runtime.views[view][off: off + nbytes])
            return runtime.read(view, offset=off, nbytes=nbytes)
        (out / f'k_{layer}.bin').write_bytes(bytes(rd(v_off + row0 * 256)))
        (out / f'v_{layer}.bin').write_bytes(bytes(rd(k_off + row0 * 256)))
        layers.append({'layer': layer, 'kind': 'full' if full else 'sliding', 'first_row': lo, 'rows': pos - lo})
    meta = {'n_tokens': pos, 'dtype': 'bf16 (kv); NPU-produced history (long gate)', 'head_dim': 128, 'n_kv_heads': 1, 'n_layers': 16, 'window': 2048,
            'layer_kinds': ['full' if l in gen.FULL_LAYERS else 'sliding' for l in range(16)], 'layers': layers, 'prompt_tokens': n_prompt}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))


def cmd_step(a):
    ld, dec = open_decoder(a.root)
    try:
        s = session_of(dec); s.length = a.pos; s.last_graph = None
        for layer, (lo, k, v) in ld.load_dump(a.kv_dump, a.pos).items():
            dec.runtime.inject_rows(a.pos - 1, layer, lo, k, v)
        out = s.decode(int(a.token), position=a.pos)
        Path(a.out).write_bytes(s.read_logits())
        print(json.dumps(dict(mode='step', pos=a.pos, token=out['token'])))
    finally:
        dec.close()
    return 0


def cmd_t4(a):
    ld, dec = open_decoder(a.root)
    tpl = json.loads(Path(a.template).read_text())
    refs = tpl['references']
    blob = (Path(a.root) / 'plamo3-runtime' / 'r34-w4' / 'reference_rows.bin').read_bytes()
    row_bytes = 107520 * 2
    bad_total, rows_total = 0, 0
    try:
        for ref in refs['prompts']:
            pid = int(ref['prompt']); prompt = tpl['model']['prompts'][pid]; teacher = [int(t) for t in ref['tokens']]
            s = session_of(dec)
            s.prefill(prompt)
            dec.runtime.bridge_old_kv(len(prompt))
            bad = []
            for r in range(1, int(ref['rows'])):
                out = s.decode(teacher[r - 1])
                want = blob[int(ref['offset']) + r * row_bytes: int(ref['offset']) + (r + 1) * row_bytes]
                if s.read_logits() != want or out['token'] != int(ref['actual_tokens'][r]):
                    bad.append(r)
            rows_total += int(ref['rows']) - 1; bad_total += len(bad)
            print(json.dumps(dict(prompt=pid, rows=int(ref['rows']) - 1, mismatched_rows=bad)), flush=True)
    finally:
        dec.close()
    res = dict(mode='t4', rows=rows_total, mismatched=bad_total, result='PASS' if bad_total == 0 and rows_total > 0 else 'FAIL')
    print(json.dumps(res)); return 0 if res['result'] == 'PASS' else 1


def cmd_l4(a):
    import numpy as np
    root = Path(a.root)
    toks = np.load(a.tokens).astype('int64')
    P, steps = a.prompt, a.steps
    assert len(toks) >= P + steps, 'token file too short'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'prompt.npy', toks[:P])
    kv = out / 'kv-out'
    job = dict(tokens_npy=str(out / 'prompt.npy'), out=str(kv), model_dir=str(root / 'model'), co_dir=str(root / 'plamo3-long' / 'prefill'), split=True, cap_ms=200.0)
    (out / 'job.json').write_text(json.dumps(job))
    t0 = time.time()
    r = subprocess.run([sys.executable, '-I', '-B', str(root / 'long' / 'long_engine.py'), str(out / 'job.json')], capture_output=True, text=True)
    done = [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{') and '"done"' in l]
    if r.returncode != 0 or not done:
        print('prefill failed', r.returncode, r.stdout[-300:]); return 2
    prefill_s = time.time() - t0
    ld, dec = open_decoder(root)
    captured = []
    caps = set(int(x) for x in a.capture.split(',')) if a.capture else set()
    try:
        s = session_of(dec); s.length = P; s.last_graph = None
        for layer, (lo, k, v) in ld.load_dump(kv, P).items():
            dec.runtime.inject_rows(P - 1, layer, lo, k, v)
        t1 = time.time()
        for i in range(steps):
            pos = P + i
            tok = int(toks[pos])
            if pos in caps:
                dump_window(dec.runtime, pos, out / f'dump_{pos}', P)
            res = s.decode(tok)
            if pos in caps:
                (out / f'logits_{pos}.bin').write_bytes(s.read_logits())
                captured.append(dict(pos=pos, token=tok, npu_token=int(res['token'])))
        decode_s = time.time() - t1
    finally:
        dec.close()
    summary = dict(mode='l4', prompt=P, steps=steps, captured=captured, prefill_s=round(prefill_s, 1), decode_s=round(decode_s, 1), last_pos=P + steps - 1)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)); return 0


def cmd_prefill(a):
    """iGPU prefill (the product's long_engine child) of the first --prompt tokens into --out (a KV dump usable by `step` and by the CPU reference)."""
    import numpy as np
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    toks = np.load(a.tokens).astype('int64')[:a.prompt]; np.save(out / 'prompt.npy', toks)
    job = dict(tokens_npy=str(out / 'prompt.npy'), out=str(out / 'kv-out'), model_dir=str(root / 'model'), co_dir=str(root / 'plamo3-long' / 'prefill'), split=True, cap_ms=200.0)
    (out / 'job.json').write_text(json.dumps(job))
    r = subprocess.run([sys.executable, '-I', '-B', str(root / 'long' / 'long_engine.py'), str(out / 'job.json')], capture_output=True, text=True)
    if r.returncode != 0:
        print('prefill failed', r.returncode, r.stdout[-300:]); return 2
    print(json.dumps(dict(mode='prefill', prompt=a.prompt, dump=str(out / 'kv-out')))); return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--mode', choices=('t4', 'step', 'l4', 'prefill'), required=True)
    ap.add_argument('--template'); ap.add_argument('--pos', type=int); ap.add_argument('--token', type=int); ap.add_argument('--kv-dump'); ap.add_argument('--out')
    ap.add_argument('--tokens'); ap.add_argument('--prompt', type=int); ap.add_argument('--steps', type=int); ap.add_argument('--capture', default='')
    a = ap.parse_args(argv)
    return {'t4': cmd_t4, 'step': cmd_step, 'l4': cmd_l4, 'prefill': cmd_prefill}[a.mode](a)


if __name__ == '__main__':
    sys.exit(main())
