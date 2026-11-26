#!/usr/bin/env python3
"""L6 (laptop side): a full-length prefill on the SHIPPED long engine (long_engine.prefill, dispatch splitting on) that saves, at the given positions, the post-RoPE query row and the GPU attention output row
of layers 7 (full) and 8 (sliding) (+ the sliding window K/V) exactly like the PoC dump hook; the float64 CPU check (PoC longcheck.py) runs on the workstation.
  python gate_l6.py <root> <out dir> <n tokens> <positions comma list> <tokens.npy>"""
import os, sys
root, out, n, rows, tokfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(',')], sys.argv[5]
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(root, 'long'))
import numpy as np
import long_engine as E
DUMP = os.path.join(out, 'dump'); os.makedirs(DUMP, exist_ok=True)
class DumpEngine(E.Engine):
    def _attention(self, ah, anw, layer, C, pos0, full, slot_mask, kc):
        super()._attention(ah, anw, layer, C, pos0, full, slot_mask, kc)
        if layer in (7, 8):
            for t in rows:
                if not (pos0 <= t < pos0 + C): continue
                self.hip.sync(); r = t - pos0
                q = np.empty((E.NH, E.HD), np.uint16); self.hip.d2h(q.ctypes.data, self.A['q'] + r * E.NH * E.HD * 2, q.nbytes)
                a = np.empty((E.NH, E.HD), np.uint16); self.hip.d2h(a.ctypes.data, self.A['att'] + r * E.NH * E.HD * 2, a.nbytes)
                d = {'q': q, 'att': a, 'pos': t, 'layer': layer, 'full': int(full)}
                if not full:
                    ring = self._download_rows(kc, 0, E.RING); start = max(0, t - (E.WINDOW - 1)); idx = np.arange(start, t + 1) & (E.RING - 1)
                    d.update(start=start, wk=ring[idx], wv=self._download_rows(self.vc[layer], 0, E.RING)[idx])
                np.savez(os.path.join(DUMP, f'L{layer}_t{t}.npz'), **d)
E.Engine = DumpEngine
toks = np.load(tokfile).astype(np.int64)[:n]
res = E.prefill(toks, out, os.path.join(root, 'model'), os.path.join(root, 'plamo3-long', 'prefill'), split=True)
print(res)
