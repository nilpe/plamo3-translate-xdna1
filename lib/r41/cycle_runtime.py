"""Sparse updates only after exhaustive byte equality in the exporter."""
import time
from types import MethodType

def install(rt):
 if hasattr(rt,'_r40_original_prepare'):return
 rt._r40_original_prepare=rt.prepare;rt._r40_patch_states={};rt.prepare=MethodType(prepare,rt)

def prepare(rt,graph_name,values):
 spec=rt.manifest.get('r40_patches',{}).get(graph_name)
 if spec is None:return rt._r40_original_prepare(graph_name,values)
 with rt._lock:
  rt._ready();parts=rt._cases(graph_name,values);start=time.perf_counter();graph=rt.manifest['graphs'][graph_name]
  if graph_name not in rt._r40_patch_states:
   state=rt._r40_original_prepare(graph_name,values);records=[]
   for entry,current in zip(graph['steps'],state['steps']):
    desc=spec['steps'][entry['instructions']];storage=current['storage'];assert len(storage['host'])==desc['nbytes'];source=rt._slice(desc['values']).cast('I');assert len(source)==len(desc['positions'])*len(desc['word_indices']);records.append((storage,desc,source,storage['host'].cast('I')))
   rt._r40_patch_states[graph_name]=(state,records);return state
  state,records=rt._r40_patch_states[graph_name];updated=uploaded=read=words=0;position=str(values['position'])
  for part,(storage,desc,source,host)in zip(parts,records):
   if storage['part']==part:continue
   width=len(desc['word_indices']);begin=desc['positions'][position]*width
   for i,off in enumerate(desc['word_indices']):host[off]=source[begin+i]
   lo,size=desc['sync_span']
   if size:storage['bo'].sync(rt.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,size,lo)
   storage['part']=part;updated+=1;uploaded+=size;read+=width*4;words+=width
  state['last_prepare']=dict(prepare_s=rt._span('prepare:'+graph_name,start),instruction_updates=updated,instruction_upload_bytes=uploaded,instruction_read_bytes=read,instruction_bos=len(records),runlist_rebuilt=False,patched_words=words,patch_criterion='every reconstructed stream byte-equal to full generated position catalog')
  return state
