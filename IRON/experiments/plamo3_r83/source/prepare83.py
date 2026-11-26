"""R83 exact host-side change: stable runs for the prefill graphs (DirectRuntime.prepare replacement) + optional start-up pre-creation.

Why: DirectRuntime.prepare creates, per graph, one exactly-sized instruction BO per (context, instruction family) and one run per step, and REBUILDS every run of the graph
whenever any family's instruction size changes (sizes depend on rows/start: 27 -> 128 -> 3 rows ...). That is ~20 ms in a fresh process (first prefill) and ~10 ms at the first
and last chunk of every later prefill (R72 attribution).
What: the prefill graphs (taps 0-3 and the head graph) get ONE capacity-sized instruction BO per (context, family) (next power of two >= the first stream, grown on demand) and one run per step, created once.
A later case only copies the new stream into the BO, flushes exactly those bytes, and -- when its size differs -- calls run.set_arg(2, size) on the (idle, re-executable) runs of that
family. run.set_arg on a run already in a runlist is honoured at the next execute (runs/r83 setarg83/setarg2_83: slot 1 BO swap and slot 2 size change, outputs equal to fresh runs).
Nothing the device executes changes: same instruction bytes, same count per execute, same buffers, same order. Decode/verify graphs keep the original prepare.
Modes (env R83_PREPARE, default `cap`): off = original prepare everywhere (the R80 timed path); cap = capacity prepare, states created lazily by the first request;
warm = cap + a daemon thread that creates all prefill states right after the runtime is open (inside the start-up window); warm2 = cap + the same thread started after spec-ready (end of DefaultRuntime.__init__).
"""
import os
import threading
import time
import types

MODES = ('off', 'cap', 'warm', 'warm2')
DEFAULT_MODE = 'cap'


def mode():
    value = os.environ.get('R83_PREPARE', DEFAULT_MODE)
    return value if value in MODES else DEFAULT_MODE


def prefill_graphs(runtime):
    cfg = runtime.manifest['r37_features']
    return tuple(dict.fromkeys([x['graph'] for x in cfg['prefill']] + [cfg['prefill_head']]))


def _capacity(size):
    """Capacity for a BO that must hold `size` bytes: next power of two, at least 4 KiB. Grown on demand (see _prepare); the exported cases are NOT scanned (the compact catalog
    expands every case it is asked for: a scan of all ~8000 cases costs ~55 ms)."""
    return max(4096, 1 << (size - 1).bit_length())


def _create(self, graph_name, graph, sizes):
    """First use of a graph: capacity BOs, all runs and runlists (the original lazily did this inside prepare)."""
    xrt = self.xrt
    state = {'steps': [], 'segments': [], 'instructions': {}}
    for entry, first_size in zip(graph['steps'], sizes):
        _, context, kernel, nargs = self.contexts[entry['context']]
        if len(entry['args']) > nargs - 3:
            raise ValueError(f'{graph_name} has too many NPU buffer arguments')
        if nargs > 3 and not entry['args']:
            raise ValueError(f'{graph_name} has no valid BO for unused pointer slots')
        key = (entry['context'], entry['instructions'])
        storage = state['instructions'].get(key)
        if storage is None:
            capacity = _capacity(first_size)
            bo = xrt.bo(self.device, capacity, xrt.bo.cacheable, kernel.group_id(1))
            host = memoryview(bo.map()).cast('B')
            storage = state['instructions'].setdefault(key, {'bo': bo, 'host': host, 'part': None, 'kernel': kernel, 'generation': 0, 'size': capacity, 'capacity': capacity, 'runs': []})
        current = {'run': None, 'storage': storage, 'generation': 0, 'context': entry['context'], 'kernel': kernel, 'args': list(entry['args']), 'nargs': nargs}
        state['steps'].append(current)
        if not state['segments'] or state['segments'][-1]['context_name'] != entry['context']:
            state['segments'].append({'context_name': entry['context'], 'context': context, 'steps': [], 'runlist': None})
        state['segments'][-1]['steps'].append(current)
    for segment in state['segments']:
        runlist = xrt.runlist(segment['context'])
        for current in segment['steps']:
            storage = current['storage']
            run = xrt.run(current['kernel'])
            run.set_arg(0, 3)
            run.set_arg(1, storage['bo'])
            run.set_arg(2, storage['size'])
            buffers = [self._bo(view) for view in current['args']]
            if len(buffers) < current['nargs'] - 3:
                buffers.extend([buffers[0]] * (current['nargs'] - 3 - len(buffers)))
            for index, bo in enumerate(buffers, start=3):
                run.set_arg(index, bo)
            runlist.add(run)
            current['run'] = run
            storage['runs'].append(run)
        segment['runlist'] = runlist
    self._graphs[graph_name] = state
    return state


def _prepare(self, graph_name, values):
    with self._lock:
        self._ready()
        parts = self._cases(graph_name, values)
        start = time.perf_counter()
        graph = self.manifest['graphs'][graph_name]
        state = self._graphs.get(graph_name)
        created = state is None
        if created:
            state = _create(self, graph_name, graph, [self._instruction_size(part) for part in parts])
        updated = uploaded = read_bytes = resized = 0
        to_device = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        for current, part in zip(state['steps'], parts):
            storage = current['storage']
            if storage['part'] != part:
                payload = self._instruction_bytes(part)
                size = self._instruction_size(part)
                if size > storage['capacity']:
                    # Grow: a larger BO replaces the old one in every run of this family (run.set_arg slot 1 is honoured, see the module docstring). The old BO stays alive until the swap is done.
                    capacity = _capacity(size)
                    bo = self.xrt.bo(self.device, capacity, self.xrt.bo.cacheable, storage['kernel'].group_id(1))
                    host = memoryview(bo.map()).cast('B')
                    for run in storage['runs']:
                        run.set_arg(1, bo)
                    storage['old'] = (storage.get('old') or []) + [(storage['bo'], storage['host'])]
                    storage['bo'], storage['host'], storage['capacity'] = bo, host, capacity
                storage['host'][:size] = payload
                storage['bo'].sync(to_device, min(storage['capacity'], (size + 63) // 64 * 64), 0)
                storage['part'] = part
                if storage['size'] != size:
                    for run in storage['runs']:
                        run.set_arg(2, size)
                    storage['size'] = size
                    resized += 1
                updated += 1
                uploaded += size
                read_bytes += part['nbytes']
        state['last_prepare'] = {
            'prepare_s': self._span(f'prepare:{graph_name}', start),
            'instruction_updates': updated,
            'instruction_upload_bytes': uploaded,
            'instruction_read_bytes': read_bytes,
            'instruction_bos': len(state['instructions']),
            'runlist_rebuilt': created,
            'size_updates': resized,
        }
        return state


def install(runtime, requested=None):
    """Called once, before any request (the validation callback). Returns the mode actually applied."""
    chosen = requested or mode()
    if chosen == 'off':
        return 'off'
    graphs = frozenset(prefill_graphs(runtime))
    original = runtime.prepare

    def prepare(self, graph_name, values):
        if graph_name in graphs:
            return _prepare(self, graph_name, values)
        return original(graph_name, values)

    runtime.prepare = types.MethodType(prepare, runtime)
    runtime._r83_prepare = chosen
    return chosen


def warm(runtime):
    """Create every prefill state with a harmless valid case (rows 1, start 0). Never raises; a failure leaves lazy creation to the first request."""
    try:
        for name in prefill_graphs(runtime):
            runtime.prepare(name, {'position': 0, 'rows': 1, 'start': 0})
        runtime._r83_warm_done = time.perf_counter()
    except BaseException as error:  # noqa: BLE001 -- the request path recreates anything missing
        runtime._r83_warm_error = repr(error)[:300]


def start_warm(runtime):
    thread = threading.Thread(target=warm, args=(runtime,), name='r83-prepare', daemon=True)
    runtime._r83_warm_thread = thread
    thread.start()
    return thread
