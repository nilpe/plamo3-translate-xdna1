# H2-L: long-mode copy of poc/pd256k/d/pd_runtime.py with the compaction fix (flush dirty host views to the device before compacting).
#!/usr/bin/env python3
"""PD-256k runtime hook: a DirectRuntime subclass; IRON files are imported read-only.

What the subclass changes, and nothing else:
  _cases       decode steps return synthetic "pdgen" parts for the 7 position-dependent
               instructions, so any pos in 0..262143 is accepted (base class only knows
               the exported 0..2047 cases). Non-position steps use the exported cases.
  _instruction_size / _instruction_bytes
               serve pdgen parts from gen.Generator (pd mode). Each step's stream is
               generated from the stencil, so only hole words differ between positions.
  execute      before a decode step: (1) at pos >= 4096 with pos % 2048 == 0, compact the
               sliding buffers (rows 2048..4095 -> 0..2047, K and V), (2) write this
               position's RoPE row into the table slots root0+0 (sliding) and
               root0+524288 (full). The write marks view2/view4 dirty, so the base
               execute syncs them.
  bridge_old_kv  after a prefill (<= 2048 rows), copy the exported r34 KV rows from the old
               views into the pdkv{L} roots, so decode can continue from the prefill.
  inject_rows  host write of absolute KV rows (dump format) at the right buffer rows.

Not changed: BO allocation, runlists, xclbin contexts, the exported prefill graphs, and
validation (PdRuntime is constructed without a validation lease, per REPORT.md section 1).
"""

import importlib.util
from pathlib import Path

import gen as G
import layout as L
import rope

ROW = 256
SLIDING_ROWS = 4096


def load_iron_direct_runtime(iron_root):
    """Import iron/applications/plamo3/direct_runtime.py by path (read-only; no package init)."""
    import sys
    sys.dont_write_bytecode = True  # never write __pycache__ into the IRON tree
    path = Path(iron_root) / "iron" / "applications" / "plamo3" / "direct_runtime.py"
    spec = importlib.util.spec_from_file_location("iron_direct_runtime_ro", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DirectRuntime


def make_pd_runtime_class(DirectRuntime):
    class PdRuntime(DirectRuntime):
        def __init__(self, manifest, artifact_paths, *, generator, **kwargs):
            if manifest.get("pd_layout") is None:
                raise ValueError("not a PD-256k manifest (run layout.py)")
            super().__init__(manifest, artifact_paths, **kwargs)
            self.gen = generator
            self.kv_old = manifest["pd_layout"]["old_kv_views"]
            self._rope_saved = None
            self._rope_clobbered = False
            self.t_pre = 0.0   # host work before decode (compaction, RoPE rows), cumulative
            self.t_gen = 0.0   # stream generation (gen.bytes), cumulative

        # ------------------------------------------------------- cases / streams
        def _cases(self, graph_name, values):
            if graph_name != "decode":
                return super()._cases(graph_name, values)
            graph = self.manifest["graphs"][graph_name]
            if set(values) != set(graph.get("parameters", [])):
                raise ValueError("decode scalar keys differ from exported graph parameters")
            pos = values["position"]
            if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos <= G.MAX_POS:
                raise ValueError(f"decode position {pos!r} outside 0..{G.MAX_POS}")
            parts = []
            for step in graph["steps"]:
                name = step["instructions"]
                if name in G.POSITION_INSTR:
                    size = self.gen.nbytes[name]
                    parts.append({"codec": "pdgen", "instruction": name, "position": pos,
                                  "artifact": "transactions", "nbytes": 0,
                                  "decoded_nbytes": size})
                else:
                    desc = self.manifest["instructions"][name]
                    key = ",".join(str(values[p]) for p in desc.get("params", []))
                    parts.append(desc["cases"][key])
            return parts

        def _instruction_size(self, part):
            if part.get("codec") == "pdgen":
                return int(part["decoded_nbytes"])
            return super()._instruction_size(part)

        def _instruction_bytes(self, part):
            if part.get("codec") == "pdgen":
                import time as _t
                t0 = _t.perf_counter()
                data = self.gen.bytes(part["instruction"], part["position"])
                self.t_gen += _t.perf_counter() - t0
                return data
            return super()._instruction_bytes(part)

        # ------------------------------------------------------- per-step host work
        def execute(self, graph_name, values):
            if graph_name == "decode":
                import time as _t
                t0 = _t.perf_counter()
                self.pd_before_decode(int(values["position"]))
                self.t_pre += _t.perf_counter() - t0
            elif self._rope_clobbered:
                # Prefill (or any non-decode graph) reads the exported table: restore slot 2047.
                self.restore_rope_slot()
            return super().execute(graph_name, values)

        def _save_rope_slot(self):
            if self._rope_saved is None:
                self._rope_saved = {
                    "view2": bytes(self.views["view2"][2047 * ROW : 2048 * ROW]),
                    "view4": bytes(self.views["view4"][2047 * ROW : 2048 * ROW]),
                }

        def restore_rope_slot(self):
            if self._rope_saved is not None and self._rope_clobbered:
                self.write("view2", self._rope_saved["view2"], offset=2047 * ROW)
                self.write("view4", self._rope_saved["view4"], offset=2047 * ROW)
            self._rope_clobbered = False

        def pd_before_decode(self, pos):
            if G.compaction_due(pos):
                # H2-L Phase 1 fix: flush every dirty host view to the device first (the device is idle between steps, the host copy is the newest),
                # otherwise read() in compact_sliding raises 'cannot overwrite unsynchronized host changes' when a dirty view shares a root.
                for _v in list(self._dirty):
                    self.sync_to(_v)
                for layer in range(16):
                    if layer not in G.FULL_LAYERS:
                        self.compact_sliding(layer)
            import os
            if pos >= 2048 and not os.environ.get("PD_DIAG_NO_ROPE"):
                # Positions < 2048 read the exported table directly; beyond that the row goes to slot 2047.
                self._save_rope_slot()
                self.write("view2", rope.row_bytes(pos, False), offset=2047 * ROW)
                self.write("view4", rope.row_bytes(pos, True), offset=2047 * ROW)
                self._rope_clobbered = True

        def _kv_view(self, layer):
            return L.kv_root_name(layer)

        def _v_offset(self, layer):
            return G.kv_bases(layer)[1]

        def _k_base(self, layer):
            return G.kv_bases(layer)[0]

        def compact_sliding(self, layer):
            """Rows 2048..4095 -> 0..2047 for K (at +0) and V (at +V offset).

            If the view has unsynced host writes (e.g. injected rows), the host copy is
            the newest data; otherwise read it back from the device first.
            """
            view = self._kv_view(layer)
            if view in self._dirty:
                data = bytearray(self.views[view])
            else:
                data = bytearray(self.read(view))
            half = 2048 * ROW
            kb, v = self._k_base(layer), self._v_offset(layer)
            data[kb : kb + half] = data[kb + half : kb + 2 * half]
            data[v : v + half] = data[v + half : v + 2 * half]
            self.write(view, data)
            # H2-L Phase 1 fix: push the compacted view at once. A view left dirty overlaps (same root) the next layer's
            # view, and its read() then raises 'cannot overwrite unsynchronized host changes' in a continuous decode.
            self.sync_to(view)

        def inject_rows(self, state_pos, layer, first_abs, k_bytes, v_bytes):
            """Write absolute KV rows [first_abs, first_abs+n) into the buffer layout that exists
            BEFORE the decode of state_pos+1 (i.e. the layout after the last executed step,
            state_pos). The next decode then performs any compaction itself, so injected
            rows are compacted with everything else. Full layers: buffer row = absolute row.
            Sliding: buffer row = absolute - window_base(state_pos).
            """
            full = layer in G.FULL_LAYERS
            base = 0 if full else G.window_base(max(0, state_pos))
            row0 = first_abs - base
            n = len(k_bytes) // ROW
            if len(v_bytes) != n * ROW or row0 < 0 or row0 + n > (262144 if full else SLIDING_ROWS):
                raise ValueError("injected rows do not fit the layer buffer")
            view = self._kv_view(layer)
            # Device roles (verified bitwise, phase 2): attention keys live at the region named
            # "V" in the exported stream (_v_offset), values at the "K" region (_k_base). The dump
            # file k_L.bin holds keys (post-RoPE K) and v_L.bin holds values, so keys go to _v_offset.
            self.write(view, k_bytes, offset=self._v_offset(layer) + row0 * ROW)
            self.write(view, v_bytes, offset=self._k_base(layer) + row0 * ROW)

        def bridge_old_kv(self, nrows):
            """After prefill of nrows (<= 2048), copy exported r34 KV rows into pdkv roots."""
            if not 0 < nrows <= 2048:
                raise ValueError("bridge supports prefill lengths 1..2048 only")
            for layer in range(16):
                old = self.kv_old[str(layer)]
                k_old = 3072 if layer == 0 else 0      # exported r34 layout (decode K base)
                v_old = 527360 if layer == 0 else 524288
                k = bytes(self.read(old, offset=k_old, nbytes=nrows * ROW))
                v = bytes(self.read(old, offset=v_old, nbytes=nrows * ROW))
                self.write(self._kv_view(layer), k, offset=self._k_base(layer))
                self.write(self._kv_view(layer), v, offset=self._v_offset(layer))

    return PdRuntime
