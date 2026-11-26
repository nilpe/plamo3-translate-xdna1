"""Default exact W4/M2 controller, with validated serial W4 fallbacks."""
import json
import sys
import time
from pathlib import Path
import settings45 as settings
settings.paths()

class DefaultRuntime:
    def __init__(self, mode=None, *, spec_manifest=None, bundle=None, log=None, primer_factory=None, initialize=None):
        self.requested_mode = mode or settings.DEFAULT_MODE
        if self.requested_mode not in ('w4-m2','w4-serial','w8-serial'):
            raise ValueError('unsupported runtime mode')
        self.mode = self.requested_mode
        self.env = self.drafter = self.primer = None
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self.fallbacks = []
        self.startup = {}
        self._closed = False
        self._hook = None  # Fault-gate observer; never set by the public CLI.
        if initialize is None:
            from initialize45 import initialize
        self._initialize = initialize
        if self.mode == 'w4-m2':
            try:
                from primer45 import Primer
                at = time.perf_counter()
                self.primer = (primer_factory or Primer)(bundle or settings.DRAFTER_PACKAGE)
                self.startup['worker_launch_ms'] = (time.perf_counter()-at)*1000
                at = time.perf_counter()
                self._authorization_error = None
                def validated(runtime):
                    try:
                        self.primer.authorize(runtime)
                    except Exception as error:
                        self._authorization_error = error.with_traceback(None)
                self.env = initialize(spec_manifest or settings.SPEC_MANIFEST, on_validated=validated)
                self.startup['runtime_initialize_ms'] = (time.perf_counter()-at)*1000
                if self._authorization_error is not None:
                    error=self._authorization_error;self._authorization_error=None;raise error
                runtime = self.env.runtime
                self.env.lease.require_validated()
                validated_bundle = Path(runtime.manifest['r40_cycle']['bundle'])
                if validated_bundle != Path(bundle or settings.DRAFTER_PACKAGE):
                    raise ValueError('configured package is not the package bound by the validated runtime')
                if runtime.manifest['r40_cycle']['package_manifest_sha256'] != settings.PACKAGE_MANIFEST_SHA256:
                    raise ValueError('configured package manifest hash is not approved')
                if not self.primer.authorized:
                    self.primer.authorize(runtime)
                runtime._r42_primer = self.primer
                at = time.perf_counter()
                from draft43 import make_direct as factory
                from host44 import make_direct
                Native, self.module = make_direct(validated_bundle, runtime, factory)
                self.startup['factory_ms'] = (time.perf_counter()-at)*1000
                at = time.perf_counter()
                self.drafter = Native(validated_bundle)
                self.startup['constructor_and_worker_wait_ms'] = (time.perf_counter()-at)*1000
                self.startup['worker'] = self.primer.ready_record
                self.startup['worker_authorized_after_spawn_ms'] = (self.primer.authorized_ns-self.primer.started_ns)/1e6
                self.startup['worker_ready_after_spawn_ms'] = (self.primer.ready_record['ready_ns']-self.primer.started_ns)/1e6
            except Exception as error:
                self._fallback('initialization', error)
                if self.primer is not None:
                    self.primer.close()
                    self.primer = None
                self.drafter = None
                if self.env is None:
                    # A failed speculative lease is never used for serial work.
                    self.env = initialize(settings.W4_MANIFEST)
        else:
            self.env = initialize(settings.W4_MANIFEST if self.mode == 'w4-serial' else settings.W8_MANIFEST)
        self.env.lease.require_validated()
        self.session = self.env.session
        from readiness import process_age_ms
        self.spec_ready_ms = process_age_ms() if self.mode == 'w4-m2' else None
        self.runtime_ready_ms = process_age_ms()
        self.startup['events'] = self.env.events

    def _fallback(self, phase, error):
        if self.mode == 'w4-serial':
            return
        reason = (type(error).__name__ + ': ' + ' '.join(str(error).split()))[:300]
        self.fallbacks.append(dict(phase=phase, reason=reason))
        self.mode = 'w4-serial'
        self.log('PLaMo: W4 serial fallback (' + phase + '): ' + reason)

    def generate(self, prompt, limit=256, emit=None):
        request_start = time.perf_counter()
        if self._closed:
            raise RuntimeError('runtime is closed')
        # Match the checkpoint tokenizer's BOS convention for a raw empty list.
        # Text translation, including empty text, uses its normal chat template.
        prompt = list(prompt)
        if not prompt:
            prompt = [1]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('positive integer output cap required')
        if len(prompt) > self.session.max_seq:
            raise ValueError('prompt exceeds validated context capacity')
        for token in prompt:
            self.session._token(token)
        self.env.lease.require_validated()
        if self.mode == 'w4-m2':
            from primer45 import WorkerUnavailable
            try:
                self.primer.check()
                self.primer.quiesce()
            except WorkerUnavailable as error:
                self._fallback('worker before request', error)
                self.primer.close()
            if self.mode == 'w4-m2':
                # Profiles are per request. Keep persistent BOs and command caches.
                for name in ('cycle_profile','refresh_profile','head_profile','_fast_profile'):
                    getattr(self.drafter, name).clear()
                self.primer.records.clear()
                from loop45 import generate
                prepared = time.perf_counter()
                result = generate(self.session, self.drafter, prompt, limit, emit,
                    on_fallback=lambda error: self._fallback('worker during request', error), hook=self._hook)
                prefix_s = prepared-request_start
                result['host_before_prefill_s'] = prefix_s
                result['ttft_s'] += prefix_s
                result['wall_s'] += prefix_s
                result['call_return_s'] += prefix_s
                result['mode'] = self.mode
                return result
        start = request_start
        got = self.session.prefill(prompt)
        times = [time.perf_counter()]
        tokens = [int(got['token'])]
        if emit is not None:
            emit(tokens[0])
        while len(tokens) < limit and tokens[-1] not in self.session.model['eos'] and self.session.length < self.session.max_seq:
            got = self.session.decode(tokens[-1])
            tokens.append(int(got['token']))
            times.append(time.perf_counter())
            if emit is not None:
                emit(tokens[-1])
        return dict(mode=self.mode, tokens=tokens, ttft_s=times[0]-start, wall_s=times[-1]-start,
            decode_s=times[-1]-times[0], logical_length=self.session.length,
            stopped_eos=tokens[-1] in self.session.model['eos'], cycles=[],
            decode_ms_per_token=(times[-1]-times[0])*1000/(len(tokens)-1) if len(tokens)>1 else None)

    def close(self):
        if self._closed:
            return
        if self.primer is not None:
            self.primer.close()
        if self.drafter is not None:
            self.drafter._state = None
            self.drafter._root_pending = None
        self.drafter = None
        import gc
        gc.collect()
        # Release per-drafter NumPy views before releasing the target BO owner.
        if self.env is not None:
            self.session = None
            self.env.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
