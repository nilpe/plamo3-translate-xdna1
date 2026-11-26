"""Strict retained validation and quiescent cleanup on initialization failure."""
import importlib.abc
import json
import os
import sys
import time
from pathlib import Path
from settings45 import APP, ROOT, paths
paths()

class NoCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'aie','mlir','torch','transformers'}:
            raise ImportError('cached default runtime forbids ' + fullname)

class Environment:
    def __init__(self):
        self.runtime = self.lease = self.session = None
        self.events = []
        self.closed = False

    def close(self):
        if self.closed:
            return
        if self.runtime is not None:
            runtime = self.runtime
            with runtime._lock:
                if runtime._active or runtime._completion_uncertain:
                    raise RuntimeError('cannot close a runtime with uncertain device completion')
                if self.session is not None:
                    self.session.embedding.release()
                    self.session = None
                # The certified sparse-patch adapters borrow source-map and
                # instruction-BO views outside DirectRuntime's base graph table.
                # Drop these only after completion and before freeing their owners.
                states = list(getattr(runtime, '_r40_patch_states', {}).values())
                if getattr(runtime, '_r36_patch_state', None) is not None:
                    states.append(runtime._r36_patch_state)
                for state, records in states:
                    for storage, desc, source, host in records:
                        source.release()
                        host.release()
                if hasattr(runtime, '_r40_patch_states'):
                    runtime._r40_patch_states.clear()
                if hasattr(runtime, '_r36_patch_state'):
                    runtime._r36_patch_state = None
                runtime.close()
            self.runtime = None
        if self.lease is not None:
            self.lease.close()
            self.lease = None
        self.closed = True

def initialize(manifest, *, on_validated=None):
    for name in sys.modules:
        if name.split('.')[0] in {'aie','mlir','torch','transformers'}:
            raise RuntimeError('preexisting compiler/model framework ' + name)
    if not any(isinstance(x, NoCompiler) for x in sys.meta_path):
        sys.meta_path.insert(0, NoCompiler())
    manifest = Path(manifest)
    # The same frozen configuration is checked again by the retained lease.
    # H1: the configuration is exported into the environment ONLY from a manifest whose envelope digest verifies (a corrupted file used to export its damaged keys, e.g. a bit-flipped IRON_PLAMO_ name, and then made the serial fallback refuse the polluted environment)
    import runtime_manifest as _rm
    envelope = dict(payload=_rm._read_manifest_raw(manifest))
    import plamo_root
    for key, value in envelope['payload']['configuration'].items():
        # H1: machine keys of older manifests (XRT_ROOT, XILINX_XRT, SYSTEMROOT) are never written into the environment.
        if key in plamo_root.LEGACY_CONFIG_KEYS:continue
        os.environ[key] = value
    import startup_r20
    startup_r20.bootstrap()
    if manifest.name == 'certificate.json':
        from identity43 import IdentityLease
        from direct43 import DirectRuntime
    else:
        from runtime_identity import IdentityLease
        from direct_runtime import DirectRuntime
    from runtime_session import RuntimeSession
    from concurrent.futures import ThreadPoolExecutor
    env = Environment()
    start = time.perf_counter()
    def event(name, seconds):
        env.events.append(dict(phase=name, ms=seconds*1000, end_ms=(time.perf_counter()-start)*1000))
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='r45-early-xrt') as pool:
            def executable_ready(lease):
                env.runtime = DirectRuntime(lease.manifest, lease.paths, event=event,
                    artifact_lease=lease, validation_barrier=lease.require_validated, copy_mode='readinto')
                return env.runtime.begin_contexts(pool).result
            env.lease = IdentityLease(manifest, workers=2, event=event, executable_ready=executable_ready)
            def commit_validation():
                env.lease.wait()
                if on_validated is not None:
                    on_validated(env.runtime)
            committed = pool.submit(commit_validation)
            env.runtime.open(concurrent_setup=True)
            committed.result()
            env.lease.require_validated()
        env.session = RuntimeSession(env.runtime)
        env.runtime.event = None  # Startup trace is bounded; requests keep their own profiles.
        env.setup_s = time.perf_counter()-start
        return env
    except BaseException:
        # IdentityLease cleans its own partially constructed retained streams.
        # Wait for early context setup before releasing its owner.
        env.close()
        raise
