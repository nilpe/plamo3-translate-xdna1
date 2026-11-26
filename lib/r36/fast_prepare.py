# SPDX-License-Identifier: Apache-2.0
"""Opt-in sparse updates of exhaustively checked, position-indexed TXN words."""
import time
from types import MethodType

def install(runtime):
    spec = runtime.manifest.get('r36_patch')
    if spec is None or hasattr(runtime, '_r36_original_prepare'):
        return
    runtime._r36_original_prepare = runtime.prepare
    runtime._r36_patch_state = None
    runtime.prepare = MethodType(prepare, runtime)

def prepare(rt, graph_name, values):
    spec = rt.manifest['r36_patch']
    if graph_name != spec['graph']:
        return rt._r36_original_prepare(graph_name, values)
    with rt._lock:
        rt._ready()
        parts = rt._cases(graph_name, values)  # Preserve exact domain/type checks.
        start = time.perf_counter()
        graph = rt.manifest['graphs'][graph_name]
        if rt._r36_patch_state is None:
            state = rt._r36_original_prepare(graph_name, values)
            records = []
            for entry, current in zip(graph['steps'], state['steps']):
                desc = spec['steps'][entry['instructions']]
                storage = current['storage']
                if len(storage['host']) != desc['nbytes']:
                    raise ValueError('patch stream extent differs from generated TXN')
                source = rt._slice(desc['values']).cast('I')
                if len(source) != len(desc['positions']) * len(desc['word_indices']):
                    raise ValueError('patch table extent differs')
                records.append((storage, desc, source, storage['host'].cast('I')))
            rt._r36_patch_state = state, records
            state['last_prepare']['prepare_s'] = time.perf_counter() - start
            return state
        state, records = rt._r36_patch_state
        updated = uploaded = read_bytes = words = 0
        position = str(values['position'])
        for part, (storage, desc, source, host) in zip(parts, records):
            if storage['part'] == part:
                continue
            row = desc['positions'][position]
            width = len(desc['word_indices'])
            begin = row * width
            for i, offset in enumerate(desc['word_indices']):
                host[offset] = source[begin + i]
            lo, size = desc['sync_span']
            if size:
                storage['bo'].sync(rt.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, lo)
            storage['part'] = part
            updated += 1
            uploaded += size
            read_bytes += width * 4
            words += width
        state['last_prepare'] = dict(
            prepare_s=rt._span('prepare:' + graph_name, start),
            instruction_updates=updated, instruction_upload_bytes=uploaded,
            instruction_read_bytes=read_bytes, instruction_bos=len(records),
            runlist_rebuilt=False, patched_words=words,
            patch_criterion='every reconstructed stream byte-equal to exported R35 cases',
        )
        return state
