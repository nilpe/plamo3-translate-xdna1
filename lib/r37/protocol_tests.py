import argparse,json,sys,unittest
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py');r=unittest.TextTestRunner(verbosity=2).run(suite);a.output.write_text(json.dumps(dict(result='PASS'if r.wasSuccessful()else'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)),indent=2));sys.exit(0 if r.wasSuccessful()else 1)
