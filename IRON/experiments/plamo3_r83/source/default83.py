"""Default exact W4/M2 controller, with validated serial W4 fallbacks."""
import json
import sys
import time
from pathlib import Path
import settings83 as settings
settings.paths()

class DefaultRuntime:
    def __init__(self, mode=None, *, spec_manifest=None, bundle=None, log=None, primer_factory=None, initialize=None, keep_warm=None, allow_fallback=True):
        # H1: allow_fallback=True keeps the R83 behaviour for harnesses that construct DefaultRuntime directly (fault-injection gates). The public CLI passes False unless
        # the user gave --allow-serial-fallback: a failed speculative initialization is then refused (exit 3) instead of silently running in W4 serial.
        self.allow_fallback = bool(allow_fallback)
        # R83: after the last hw_context is created (startup event xclbin_hw_contexts) the remaining startup threads (identity
        # commit, BO setup, validation, package preload) are Python-bound and convoy on the GIL with the default 5 ms switch interval.
        # A shorter interval ONLY in that window changes thread scheduling only (no bytes, order or arithmetic). The interval is the
        # default during device_open/xclbin load/hw_context creation (pyxrt holds the GIL there; the R53 whole-construction interval
        # coincided with ~500 ms hw_context_create:context0 stalls) and the previous value is restored on every exit path.
        self.startup = {}
        previous = sys.getswitchinterval()
        self.startup['switch_interval_default_s'] = previous
        self._window = None
        if mode in (None, 'w4-m2') and settings.STARTUP_SWITCH_INTERVAL_S:
            import switchwin83 as switchwin
            self._window = switchwin.Window(settings.STARTUP_SWITCH_INTERVAL_S)
            switchwin.install(self._window)
        try:
            self._construct(mode, spec_manifest, bundle, log, primer_factory, initialize)
        finally:
            if self._window is not None:
                import switchwin83 as switchwin
                switchwin.uninstall(self._window)
                self.startup['switch_window'] = self._window.close()
                if self.startup['switch_window'].get('opened'):
                    self.startup['switch_interval_startup_s'] = self.startup['switch_window']['short_s']
            sys.setswitchinterval(previous)
            self.startup['switch_interval_restored_s'] = sys.getswitchinterval()
        # R83: keep-warm (IDLE-priority spinner helper on the decode thread's CPU, see keepwarm83.py). Started only now, after spec-ready was recorded and the
        # GIL switch interval was restored; it never runs during startup and its construction never raises (any failure = R80 behaviour).
        self._init_keepwarm(keep_warm)
        self._init_hybrid()

    def _init_hybrid(self):
        # R83: hybrid wait (hybridwait83.py): completion hint + the ordinary runlist.wait(); w4-m2 only, keep-warm off, PLAMO3_HYBRID_WAIT=0 disables. Never raises.
        self.hybrid = None
        try:
            if self.mode != 'w4-m2' or self.drafter is None or getattr(self, 'keepwarm', None) is not None or self.keep_warm_mode not in ('off',):
                return
            import hybridwait83
            if hybridwait83.enabled():
                self.hybrid = hybridwait83.install(self.env.runtime, self.drafter, log=self.log)
        except Exception as error:
            self.hybrid = None
            self.log('PLaMo: hybrid wait unavailable (' + type(error).__name__ + ': ' + str(error)[:200] + '); running with the ordinary wait')

    def _init_keepwarm(self, keep_warm):
        self.keepwarm = None
        self.keep_warm_mode = 'off'
        try:
            import os
            if keep_warm == 'off' or (keep_warm is None and not os.environ.get('PLAMO3_KEEP_WARM') and settings.DEFAULT_KEEP_WARM == 'off'):
                return  # off: nothing is imported, started, queried or pinned (the R80 path)
            import keepwarm83
            mode = keep_warm if keep_warm in keepwarm83.MODES else keepwarm83.resolve_mode(None, None, settings.DEFAULT_KEEP_WARM, self.log)
            self.keep_warm_mode = mode
            if mode == 'off' or self.mode != 'w4-m2':
                return
            self.keepwarm = keepwarm83.make(mode, log=self.log, collect_stats=bool(__import__('os').environ.get('R83_KEEP_WARM_STATS')))
            if mode == 'on' or keepwarm83.power_status() == 'ac':
                self.keepwarm.start()
        except Exception as error:
            self.keepwarm = None
            self.log('PLaMo: keep-warm unavailable (' + type(error).__name__ + ': ' + str(error)[:200] + '); running without it')

    def _construct(self, mode, spec_manifest, bundle, log, primer_factory, initialize):
        self.requested_mode = mode or settings.DEFAULT_MODE
        if self.requested_mode not in ('w4-m2','w4-serial','w8-serial'):
            raise ValueError('unsupported runtime mode')
        self.mode = self.requested_mode
        self.env = self.drafter = self.primer = None
        self._tokpre = None
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self.fallbacks = []
        self._closed = False
        self._hook = None  # Fault-gate observer; never set by the public CLI.
        if initialize is None:
            from initialize83 import initialize
        self._initialize = initialize
        if self.mode == 'w4-m2':
            try:
                from primer83 import Primer
                at = time.perf_counter()
                self.primer = (primer_factory or Primer)(bundle or settings.DRAFTER_PACKAGE)
                self.startup['worker_launch_ms'] = (time.perf_counter()-at)*1000
                at = time.perf_counter()
                self._authorization_error = None
                def validated(runtime):
                    # R83: span-list generalisation of the R36/R40 in-place patchers (patcher83); must precede every install(). Not guarded: a failure falls back through the normal initialization fallback.
                    import patcher83
                    patcher83.apply()
                    try:
                        # R83: stable prefill runs (prepare83) + optional creation thread; R83_PREPARE=off restores the R80 prepare. Never fails start-up.
                        import prepare83
                        self._prepare_mode = prepare83.install(runtime)
                        if self._prepare_mode == 'warm':prepare83.start_warm(runtime)
                    except Exception:
                        pass
                    try:
                        self.primer.authorize(runtime)
                        # Package-dependent construction work starts at the same barrier as the worker.
                        from target83_base import start_preload
                        start_preload(runtime)
                    except Exception as error:
                        self._authorization_error = error.with_traceback(None)
                    # R83: the tokenizer's file reads + hashes (GIL-free) run in a thread (tokfast83); the constructor itself only casts buffers. Where the thread starts is tokfast83.start_point():
                    # 'validated' (this barrier), 'root0' (after the 520 MB initial copy), 'init_end' (right after the runtime initializer, overlapping only the worker wait).
                    self._tok_manifest = runtime.manifest['tokenizer_manifest']
                    try:
                        import tokfast83
                        if tokfast83.start_point() == 'validated':
                            self._tokpre = tokfast83.start(self._tok_manifest)
                        elif tokfast83.start_point() == 'root0':
                            def hook(phase, self=self):
                                if phase == 'bo_upload:root0' and self._tokpre is None:
                                    self._tokpre = tokfast83.start(self._tok_manifest)
                            import initialize83
                            initialize83.TOK_EVENT_HOOK = hook
                    except Exception:
                        self._tokpre = None
                if self._window is not None:self._window.arm()
                self.env = initialize(spec_manifest or settings.SPEC_MANIFEST, on_validated=validated)
                self.startup['runtime_initialize_ms'] = (time.perf_counter()-at)*1000
                try:
                    import tokfast83
                    if tokfast83.start_point() == 'init_end' and self._tokpre is None:
                        self._tokpre = tokfast83.start(self.env.runtime.manifest['tokenizer_manifest'])
                    import initialize83
                    initialize83.TOK_EVENT_HOOK = None
                except Exception:
                    pass
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
                from adapter83 import make_direct as factory
                from host44 import make_direct
                Native0, self.module = make_direct(validated_bundle, runtime, factory)
                import os as _os
                import numpy as _np
                _check = _os.environ.get('R83_AUDIT_CHECK') == '1'
                class Native(Native0):
                    # R83: the controller calls propose(audit='logits') and only needs the 512 exact candidate logits from the audit result. The base audit also re-converts
                    # the 24576 coarse scores (unused). The same bytes are read here from the exact row buffer, exactly as the base does.
                    # R83_AUDIT_CHECK=1 (gate run) runs the full base audit and asserts bitwise equality of the logits.
                    def propose(self, state, anchor, audit=False):
                        if audit != 'logits':
                            return super().propose(state, anchor, audit=bool(audit))
                        if _check:
                            full = super().propose(state, anchor, audit=True)
                            lean = (self._exact_host.astype(_np.uint32) << 16).view(_np.float32)
                            assert full['candidate_logits'].tobytes() == lean.tobytes(), 'lean audit logits differ'
                            return full
                        result = super().propose(state, anchor, audit=False)
                        result['candidate_logits'] = (self._exact_host.astype(_np.uint32) << 16).view(_np.float32)
                        return result
                self.startup['factory_ms'] = (time.perf_counter()-at)*1000
                at = time.perf_counter()
                self.drafter = Native(validated_bundle)
                self.startup['constructor_and_worker_wait_ms'] = (time.perf_counter()-at)*1000
                self.startup['worker'] = self.primer.ready_record
                self.startup['worker_authorized_after_spawn_ms'] = (self.primer.authorized_ns-self.primer.started_ns)/1e6
                self.startup['worker_ready_after_spawn_ms'] = (self.primer.ready_record['ready_ns']-self.primer.started_ns)/1e6
            except Exception as error:
                if self._window is not None:self._window.close()
                self._refuse_initialization(error)  # H1: raises for NPU/driver/environment errors always, and for any error unless allow_fallback
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
        if self.drafter is not None and hasattr(self.drafter, 'prefault'):
            self.drafter.prefault()
        self.startup['events'] = self.env.events

    def make_tokenizer(self):
        """R83: the pinned exact tokenizer (same checks, same encode/decode results; see tokcompact83 / tokfast83). R77_TOK=off restores FastPlamo3Tokenizer(manifest)."""
        import tokfast83
        pre, self._tokpre = self._tokpre, None
        return tokfast83.make(self.env.runtime.manifest['tokenizer_manifest'], pre)

    @staticmethod
    def fast_exit_allowed(*, turbo_on, service):
        """R83: the fast exit is only for the plain one-shot CLI: never with --service (more requests follow), never with turbo on (the release helper is spawned by the session exit)."""
        import os
        return (not turbo_on) and (not service) and os.environ.get('R77_EXIT', 'fast') == 'fast'

    def fast_exit(self, rc=0):
        """R83, one-shot CLI only: called after the result was printed and flushed. Does what close() does for everything the OS cannot do (stops the priming worker, keep-warm
        helper) and then ends the process without the Python-level BO/context/lease teardown, gc.collect and interpreter finalization, which cost ~130 ms and are repeated by the
        kernel at process exit anyway. Returns False without doing anything when the runtime is not in the plain idle state (then the normal close() runs, including its
        'uncertain device completion' refusal). R77_EXIT=off disables it."""
        import os
        if os.environ.get('R77_EXIT', 'fast') != 'fast' or self._closed or self.env is None or getattr(self.env, 'runtime', None) is None:
            return False
        try:
            runtime = self.env.runtime
            with runtime._lock:
                if runtime._active or runtime._completion_uncertain:
                    return False
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
        except Exception:
            return False
        try:
            if getattr(self, 'keepwarm', None) is not None:
                self.keepwarm.close()
            if self.primer is not None:
                self.primer.close()
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
        except Exception:
            return False
        os._exit(rc)

    def _refuse_initialization(self, error):
        """H1 loud downgrade (BUG 10). A serial fallback cannot help when the NPU itself is unavailable or the driver/environment is broken: those errors always end
        the run with their own message (exit 5/6). Every other initialization error of the speculative path ends the run with exit 3 unless fallbacks are allowed."""
        import plamo_errors
        classified = plamo_errors.classify(error)
        if classified.code in (plamo_errors.NPU, plamo_errors.ENVIRONMENT):
            self._abort()
            raise classified from error
        if not self.allow_fallback:
            reason = (type(error).__name__ + ': ' + ' '.join(str(error).split()))[:300]
            if 'configuration changed' in reason:
                names = sorted(k for k in __import__('os').environ if k.startswith('IRON_PLAMO_'))
                if names:reason += ' [environment variable(s) set that change the certified configuration: ' + ', '.join(names) + '; remove them]'
            self._abort()
            raise plamo_errors.InstallProblem(
                'speculative decoding could not be started (' + reason + '). The installation may be incomplete or modified. '
                'Run install.ps1 -Verify, or pass --allow-serial-fallback to continue in W4 serial (about 25% slower; results are identical).') from error

    def _abort(self):
        """Release what a failed construction holds (priming worker, validated environment); best effort, never raises."""
        try:
            if self.primer is not None:
                self.primer.close()
                self.primer = None
        except Exception:
            pass
        try:
            if self.env is not None:
                self.env.close()
                self.env = None
        except Exception:
            pass
        self.drafter = None
        self._closed = True

    def _fallback(self, phase, error):
        if self.mode == 'w4-serial':
            return
        reason = (type(error).__name__ + ': ' + ' '.join(str(error).split()))[:300]
        self.fallbacks.append(dict(phase=phase, reason=reason))
        self.mode = 'w4-serial'
        self.log('PLaMo: W4 serial fallback (' + phase + '): ' + reason)

    def generate(self, prompt, limit=256, emit=None, *, target_language=None):
        # R83: the decode thread is pinned to the keep-warm CPU only for the duration of this call (affinity restored in finally); enter/leave never raise.
        kw = self.keepwarm if getattr(self, 'keepwarm', None) is not None and self.mode == 'w4-m2' else None
        token = kw.enter() if kw is not None else None
        try:
            return self._generate(prompt, limit, emit, target_language=target_language)
        finally:
            if kw is not None:
                kw.leave(token)

    def _generate(self, prompt, limit=256, emit=None, *, target_language=None):
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
        if self.mode == 'w4-m2' and hasattr(self.drafter,'select_language'):
            try:self.drafter.select_language(target_language)
            except (ValueError,KeyError) as error:
                if not self.allow_fallback:raise ValueError('target_language must be ja or en') from error  # H1: a bad request is an error, never a permanent downgrade (BUG 5)
                self._fallback('request language',error)
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
                from loop83 import generate
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
        if getattr(self, 'hybrid', None) is not None:
            try:self.hybrid.uninstall()
            except Exception:pass
            self.hybrid = None
        if getattr(self, 'keepwarm', None) is not None:
            self.keepwarm.close()
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
