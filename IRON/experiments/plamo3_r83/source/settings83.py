"""R83 candidate settings (R80 default + the keep-warm switch, DEFAULT_KEEP_WARM below): the R57 default (exact M2 with the shipped drafter package, confidence-gated C(1)) with the R83 exact W4 GEMV kernels (decode image + M2 verifier image swapped, same interfaces). R83_PACKAGE=v4C selects the v4C drafter package (unused)."""
import hashlib,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'plamo3_r45'/'source'))
import settings45
settings45.paths()
from settings45 import ROOT,APP,DEFAULT_MODE,W4_MANIFEST,W8_MANIFEST
# The drafter package is data, not code: release.json (the selection record next to the install root) names it with 'drafter_package' (relative to
# the install root), so a new drafter package needs a new certificate and a new release.json, not a code change. The fallback is the shipped package.
def _drafter_relpath():
 try:
  import json
  rel=json.loads((ROOT/'release.json').read_text(encoding='utf-8-sig')).get('drafter_package')
  if isinstance(rel,str) and rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts:return rel
 except (OSError,ValueError,AttributeError):pass
 return 'plamo3-drafter-port-v7/PLaMo-D20-M2'
DRAFTER_PACKAGE=ROOT/_drafter_relpath()
def __getattr__(name):
 # Read inside the constructor/authorization guard, never during module import.
 # Missing package data must leave W4/W8 options and exact fallback available.
 if name=='PACKAGE_MANIFEST_SHA256':return hashlib.sha256((DRAFTER_PACKAGE/'PLaMo-manifest.json').read_bytes()).hexdigest()
 raise AttributeError(name)
# R83 keep-warm default policy: ONE constant (user decision 2026-11-18: opt-in, like turbo). 'off' | 'on' | 'ac' (on only on AC power, checked at each generate). With 'off' and no env/CLI override the timed path is exactly R80's (no helper, no affinity call, no power-status query, no thread). Env PLAMO3_KEEP_WARM=0|1|auto and CLI --keep-warm/--no-keep-warm override it.
DEFAULT_KEEP_WARM='off'
CANDIDATE=ROOT/'plamo3-default/r45-spec'   # release: the exported bundle lives next to the shipped certificate
CANDIDATE_SPEC=ROOT/'plamo3-default/r45-spec/certificate.json'   # release: one certificate (the candidate IS the shipped default)
W4_CANDIDATE=ROOT/'plamo3-default/r45-w4/manifest.json'
SERIAL_VIEW=CANDIDATE/'serial-view/manifest.json'
ACTIVE_SPEC=settings45.SPEC_MANIFEST
ACTIVE_W4=settings45.W4_MANIFEST
SPEC_MANIFEST=settings45.SPEC_MANIFEST
# R83: GIL switch interval (100 microseconds) used ONLY in the window opened at startup event xclbin_hw_contexts (after the last hw_context) and closed at end of speculative construction; see switchwin83.py. Always restored before the first request.
STARTUP_SWITCH_INTERVAL_S=0.0001
def paths():
 settings45.paths()
 import sys
 sys.path.insert(0,str(Path(__file__).parent))
paths()
# R83 (Phase B staging): the flat-table tokenizer artifact derived from the bound marshal payload (tokcompact83.build). Not bound/attested yet; the loader falls back to the original constructor if absent.
TOKENIZER_COMPACT=ROOT/'plamo3-r74/tokenizer/tokenizer-manifest.compact.json'

