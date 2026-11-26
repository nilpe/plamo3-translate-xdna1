# SPDX-License-Identifier: Apache-2.0
"""PLaMo 3 long mode, NPU decode step: the PD-256k copy-and-patch decode (v4: fast K/V pack, FIFO depths 2/2/4) driven through the installed, sealed DirectRuntime.

Serial W4 greedy decode from the iGPU-produced KV. Differences from the PoC runner: install-relative paths, errors mapped to the exit-code table, EOS / caps /
total-length stop, cooperative cancel, progress, and the sliding-window compaction fix (npu/long_pdrt.py). Arithmetic and kernels are the PoC's.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from long_core import Cancelled, EOS_IDS, KV_LAYERS, MAX_TOTAL, LongModeError, errors, install_root, sealed_paths

HERE = Path(__file__).resolve().parent
NPU_DIR = HERE / 'npu'


def _debug(error):
    if os.environ.get('PLAMO3_DEBUG') == '1':
        import traceback
        traceback.print_exception(type(error), error, error.__traceback__)


def rebase_paths(paths, root):
    """artifact-paths.json holds '@ROOT@\\...' strings; returns absolute paths under the install root (refuses anything else)."""
    out = {}
    for key, value in paths.items():
        if not isinstance(value, str) or not value.startswith('@ROOT@'):
            raise errors.InstallProblem('the installation is incomplete or modified (long-mode artifact list). Run install.ps1 -Verify.')
        rel = value[len('@ROOT@'):].lstrip('\\/')
        if '..' in rel.replace('\\', '/').split('/'):
            raise errors.InstallProblem('the installation is incomplete or modified (long-mode artifact path). Run install.ps1 -Verify.')
        out[key] = str(Path(root) / rel)
    return out


def load_dump(dump_dir, pos):
    """KV dump (iGPU output) -> {layer: (first_row, K_bytes, V_bytes)} covering rows needed by a decode at `pos`."""
    meta = json.loads((Path(dump_dir) / 'meta.json').read_text())
    out = {}
    for item in meta['layers']:
        layer = int(item['layer'])
        full = item['kind'] == 'full'
        lo = 0 if full else max(0, pos - 2047)
        first, rows = int(item['first_row']), int(item['rows'])
        if first > lo or first + rows < pos:
            raise LongModeError('the prefill result does not cover the needed rows (layer %d)' % layer)
        base = Path(dump_dir)
        if not (base / f'k_{layer}.bin').exists() and (base / 'kv' / f'k_{layer}.bin').exists():
            base = base / 'kv'
        k = (base / f'k_{layer}.bin').read_bytes()
        v = (base / f'v_{layer}.bin').read_bytes()
        a, b = (lo - first) * 256, (pos - first) * 256
        out[layer] = (lo, k[a:b], v[a:b])
    return out


class LongDecoder:
    def __init__(self, root=None):
        self.root = Path(root) if root else install_root()
        self.runtime = None
        self.session = None
        self.open_s = None

    # -------------------------------------------------------------- open
    def open(self):
        plamo = sealed_paths(self.root)
        if not (plamo / 'direct_runtime.py').exists():
            raise errors.InstallProblem('the installation is incomplete (the NPU runtime is missing). Run install.ps1 -Verify.')
        for p in (str(plamo), str(NPU_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        sys.dont_write_bytecode = True
        try:
            import startup_r20
            startup_r20.bootstrap()                      # our pyxrt (<root>\xrt\python), deterministic DLL search; same as the default CLI
            import long_pdrt
            DirectRuntime = long_pdrt.load_iron_direct_runtime(self.root / 'IRON')
            spec = importlib.util.spec_from_file_location('plamo_runtime_session_long', plamo / 'runtime_session.py')
            self._session_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._session_mod)
            pkg = self.root / 'plamo3-long' / 'decode'
            for name in ('pd-manifest.json', 'artifact-paths.json', 'rtp-map.json', 'final.xclbin'):
                if not (pkg / name).exists():
                    raise errors.InstallProblem('the installation is incomplete (missing long-mode file %s). Run install.ps1 -Verify.' % name)
            os.environ['PD_RTP_MAP'] = str(pkg / 'rtp-map.json')
            manifest = json.loads((pkg / 'pd-manifest.json').read_text())
            paths = rebase_paths(json.loads((pkg / 'artifact-paths.json').read_text()), self.root)
            import gen
            r34 = self.root / 'plamo3-runtime' / 'r34-w4'
            payload = manifest['payload'] if 'payload' in manifest else manifest
            generator = gen.Generator(r34, mode='pd', loaded=(payload, (r34 / 'transactions.bin').read_bytes()))
            Pd = long_pdrt.make_pd_runtime_class(DirectRuntime)
            self.runtime = Pd(manifest, paths, generator=generator, validation_barrier=None)
            t0 = time.perf_counter()
            self.runtime.open(device_index=0)
            self.open_s = time.perf_counter() - t0
        except errors.PlamoError:
            raise
        except Exception as error:
            _debug(error)
            raise errors.classify(error) from error
        return self

    # -------------------------------------------------------------- decode
    def run(self, dump_dir, n_prompt, first_token, cap, *, on_token=None, cancelled=lambda: False, eos=EOS_IDS, stop=None):
        """Greedy decode. `first_token` is the prefill argmax (token 1 of the output). Returns dict(tokens, steps_ms, inject_s)."""
        if n_prompt + cap > MAX_TOTAL:
            raise LongModeError('prompt plus output would exceed %d tokens' % MAX_TOTAL)
        rt = self.runtime
        session = self._session_mod.RuntimeSession(rt)
        session.length = n_prompt
        session.last_graph = None
        dump = load_dump(dump_dir, n_prompt)
        t0 = time.perf_counter()
        for layer, (lo, k, v) in dump.items():
            rt.inject_rows(n_prompt - 1, layer, lo, k, v)
        inject_s = time.perf_counter() - t0
        tokens = [int(first_token)]
        steps = []
        repeat = None
        cur = tokens[0]
        try:
            while len(tokens) < cap and cur not in eos:
                if cancelled():
                    raise Cancelled('cancelled')
                t1 = time.perf_counter()
                out = session.decode(cur)
                steps.append((time.perf_counter() - t1) * 1000.0)
                cur = int(out['token'])
                tokens.append(cur)
                if on_token:
                    on_token(tokens, steps)
                if stop is not None:
                    repeat = stop(tokens)
                    if repeat:
                        tokens = tokens[:repeat['keep']]
                        break
        except (Cancelled, errors.PlamoError):
            raise
        except Exception as error:
            _debug(error)
            raise errors.classify(error) from error
        return dict(tokens=tokens, steps_ms=steps, inject_s=inject_s, repeat=repeat)

    def close(self):
        rt, self.runtime = self.runtime, None
        if rt is not None:
            try:
                rt.close()
            except Exception:
                pass
