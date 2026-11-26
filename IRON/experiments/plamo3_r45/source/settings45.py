"""One deployment configuration; a package change requires export and all gates."""
import sys
from pathlib import Path
# H1: the install root comes from the location of plamo_root.py (plamo_root.ROOT), never from an environment variable.
APP = Path(__file__).resolve().parents[3] / 'iron' / 'applications' / 'plamo3'
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from plamo_root import ROOT
DEFAULT_MODE = 'w4-m2'
# DRAFTER_PACKAGE: see settings83 (selected by release.json, default in settings83)
SPEC_MANIFEST = ROOT / 'plamo3-default/r45-spec/certificate.json'
W4_MANIFEST = ROOT / 'plamo3-default/r45-w4/manifest.json'
W8_MANIFEST = ROOT / 'plamo3-default/r45-w8/manifest.json'

def paths():
    # Keep the certified modules frozen. R45 adds lifecycle/fallback control.
    import sys
    # lib/rNN holds the certified helper modules (a flat layout; r39a = source_front, r39b = target_runtime_k).
    folders = [APP] + [ROOT / 'lib' / f'r{n}' for n in (36,37)]
    folders += [ROOT / 'lib' / 'r39a', ROOT / 'lib' / 'r39b']
    folders += [ROOT / 'lib' / f'r{n}' for n in (40,41,42,43,44)]
    folders += [Path(__file__).parent]
    for folder in folders:
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
