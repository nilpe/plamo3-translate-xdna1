"""R83 Phase B test hook (NOT a bundle change): hot-swap the head context of a constructed DirectRuntime.

context1 of the default bundle is used only by the 6 head steps (PFLast, RMSNorm, PFRepeat, PFLinear N=107520, PFLogits, Sample; graphs r37_prefill_head and the legacy `prefill`).
It currently holds the R34-array image (PDI afd519d8) whose W8 path uses the software-permuting r12 kernels (head matrix 32.7 ms). The W8 option's original-array image (PDI b5633dde,
r24 DMA-packed kernels, memtile does the W8 permutation) runs the same op in 20.3 ms (runs/r83 probe83p). This hook loads that image as a hw_context in place of context1 and
points the six instruction families at the W8 option's exported streams for that image (same params, same case keys, same buffer order; they differ from the current streams in 8-32 bytes).
A Phase C export would put the artifact and the streams in the bundle instead; this hook only exists to measure and gate the idea without touching any bound file.
"""
import json,zlib
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()))
W8_MANIFEST=ROOT/'plamo3-default/r45-w8/manifest.json'
FAMILIES=('inst23','inst24','inst25','inst26','inst27','inst28')

def apply(rt,w8_manifest=W8_MANIFEST):
    raw=json.loads(Path(w8_manifest).read_text(encoding='utf-8-sig'));w8=raw.get('payload',raw)
    tx=memoryview(Path(w8['artifacts']['transactions']['path']).read_bytes())
    image_path=w8['artifacts'][w8['contexts']['context1']['artifact']]['path']
    with rt._lock:
        assert not any(seg['context_name']=='context1' for state in rt._graphs.values() for seg in state['segments']),'head swap must happen before any head graph is prepared'
        image=rt.xrt.xclbin(str(image_path));uuid=image.get_uuid();rt.device.register_xclbin(image)
        context=rt.xrt.hw_context(rt.device,uuid);kernel=rt.xrt.kernel(context,'MLIR_AIE')
        nargs=next(k.get_num_args() for k in image.get_kernels() if k.get_name()=='MLIR_AIE')
        old=rt.contexts['context1'];rt.contexts['context1']=(image,context,kernel,nargs)
        for fam in FAMILIES:
            d=rt.manifest['instructions'][fam];wd=w8['instructions'][fam]
            assert d['params']==wd['params'] and set(d['cases'])==set(wd['cases']),fam
            d['cases']={key:dict(part,r83_w8=True) for key,part in wd['cases'].items()}   # the catalog's lazy Cases mapping is read-only: replace it by a plain dict (parts are decoded lazily from the W8 transactions)
        orig_bytes,orig_size=rt._instruction_bytes,rt._instruction_size
        def ibytes(part):
            if part.get('r83_w8'):
                b=bytes(tx[part['offset']:part['offset']+part['nbytes']])
                return zlib.decompress(b) if part.get('codec')=='zlib' else b
            return orig_bytes(part)
        def isize(part):
            return part['decoded_nbytes'] if part.get('r83_w8') else orig_size(part)
        rt._instruction_bytes,rt._instruction_size=ibytes,isize
        rt._r83_headswap=dict(image=str(image_path),old_context_kept=old is not None,nargs=nargs)
    return rt._r83_headswap
