"""Language-aware (v5-family format) request selection and mapped gathers; reuse certified R43/R44 services."""
import json
from pathlib import Path
def make_direct(bundle,runtime):
 import target43_io,target83_base,draft43
 from target43_base import make_direct as old
 v5=json.loads((Path(bundle)/'PLaMo-manifest.json').read_text())['format'] in ('PLaMo-M2-port-v7',)
 previous=target43_io.base_factory
 try:
  if v5:target43_io.base_factory=target83_base.make_direct
  return draft43.make_direct(bundle,runtime)
 finally:target43_io.base_factory=previous
