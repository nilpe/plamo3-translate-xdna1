"""R83 host change (experiment tree only): the in-place instruction patchers of R36 (verify2, manifest r36_patch) and R40 (r40_patches: serial bulk taps, drafter joins, and since R83 `decode`)
generalised from ONE sync span per step to a LIST of spans (`desc['sync_spans']`, falling back to the single `desc['sync_span']`). Everything else is the original code, line for line:
the word patch, the part test, the first-prepare record building (so every R80 spec keeps working unchanged). apply() replaces `fast_prepare.prepare` and `cycle_runtime.prepare` before any install()."""
import time

def _spans(desc):
    s = desc.get('sync_spans')
    return s if s is not None else [desc['sync_span']]

def prepare_r36(rt, graph_name, values):
    spec = rt.manifest['r36_patch']
    if graph_name != spec['graph']:
        return rt._r36_original_prepare(graph_name, values)
    with rt._lock:
        rt._ready()
        parts = rt._cases(graph_name, values)
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
            for lo, size in _spans(desc):
                if size:
                    storage['bo'].sync(rt.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, lo)
                    uploaded += size
            storage['part'] = part
            updated += 1
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

def prepare_r40(rt, graph_name, values):
    spec = rt.manifest.get('r40_patches', {}).get(graph_name)
    if spec is None:
        return rt._r40_original_prepare(graph_name, values)
    with rt._lock:
        rt._ready()
        parts = rt._cases(graph_name, values)
        start = time.perf_counter()
        graph = rt.manifest['graphs'][graph_name]
        if graph_name not in rt._r40_patch_states:
            state = rt._r40_original_prepare(graph_name, values)
            records = []
            for entry, current in zip(graph['steps'], state['steps']):
                desc = spec['steps'][entry['instructions']]
                storage = current['storage']
                assert len(storage['host']) == desc['nbytes']
                source = rt._slice(desc['values']).cast('I')
                assert len(source) == len(desc['positions']) * len(desc['word_indices'])
                records.append((storage, desc, source, storage['host'].cast('I')))
            rt._r40_patch_states[graph_name] = (state, records)
            return state
        state, records = rt._r40_patch_states[graph_name]
        updated = uploaded = read = words = 0
        position = str(values['position'])
        for part, (storage, desc, source, host) in zip(parts, records):
            if storage['part'] == part:
                continue
            width = len(desc['word_indices'])
            begin = desc['positions'][position] * width
            for i, off in enumerate(desc['word_indices']):
                host[off] = source[begin + i]
            for lo, size in _spans(desc):
                if size:
                    storage['bo'].sync(rt.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, lo)
                    uploaded += size
            storage['part'] = part
            updated += 1
            read += width * 4
            words += width
        state['last_prepare'] = dict(prepare_s=rt._span('prepare:' + graph_name, start), instruction_updates=updated, instruction_upload_bytes=uploaded, instruction_read_bytes=read,
                                     instruction_bos=len(records), runlist_rebuilt=False, patched_words=words, patch_criterion='every reconstructed stream byte-equal to full generated position catalog')
        return state

def apply():
    """Call after the source roots are on sys.path and before any install()."""
    import fast_prepare, cycle_runtime
    fast_prepare.prepare = prepare_r36
    cycle_runtime.prepare = prepare_r40
    return dict(fast_prepare=fast_prepare.__file__, cycle_runtime=cycle_runtime.__file__)
