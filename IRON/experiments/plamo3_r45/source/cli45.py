"""Production CLI entry shim (R83): W4 + exact M2 with the shipped drafter package is the default.
The public entry (iron/applications/plamo3/translate.py) imports `cli45.main`; the real, certificate-bound implementation is
cli83 in experiments/plamo3_r83/source. Serial W4 / W8 stay selectable through --mode / --spec-off. Previous entry: plamo3-default/r83-active-archive/cli45.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'plamo3_r83'/'source'))
from cli83 import main as _main, real_process_entry
def main():
    # the fast exit (R83) is allowed only when this is the real process entry (public translate.py), never inside an in-process harness
    return _main(process_entry=real_process_entry())
if __name__=='__main__':sys.exit(main())
