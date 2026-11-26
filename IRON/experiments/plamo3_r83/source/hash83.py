"""Standalone inventory inspection; no runtime/model/device is constructed."""
import argparse,json
from pathlib import Path
import settings83
from initialize83 import package_hash
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=dict(result='PASS',no_model_or_NPU=True,**package_hash());a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
