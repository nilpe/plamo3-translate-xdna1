# SPDX-License-Identifier: Apache-2.0
"""Cached translation entry point. One line on stderr and exit code 3 when the installation is incomplete (never a traceback)."""
from pathlib import Path
import sys


def main():
    source = Path(__file__).resolve().parents[3] / 'experiments' / 'plamo3_r45' / 'source'
    try:
        sys.path.insert(0, str(source))
        from cli45 import main as cached_main
    except Exception as error:
        import os
        detail = ' '.join(str(error).split())[:200]
        sys.stderr.write('PLaMo: error: the installation is incomplete or modified (%s: %s). Run install.ps1 -Verify.\n' % (type(error).__name__, detail))
        if os.environ.get('PLAMO3_DEBUG') == '1':
            import traceback
            traceback.print_exc()
        return 3
    return cached_main()


if __name__=='__main__':
    sys.exit(main())
