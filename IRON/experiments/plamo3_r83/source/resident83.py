"""One default instance: 15 full fixtures, 100 prefixes, limits/EOS/idle and RSS."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from settings83 import ROOT, paths
paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
rows=[];memory=[]
try:
    from default83 import DefaultRuntime
    from loop_gate import serial_generate
    from memory45 import snapshot,Sampler
    meta=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text())
    para=json.loads((ROOT/'gate-data/long-translation.json').read_text())
    paragraph=next(x for x in meta['long_cases']if x['id']=='en-512')['prompt']
    holder=[None];sampler=Sampler(lambda:holder[0].primer if holder[0]is not None else None);sampler.start()
    app=DefaultRuntime();holder[0]=app;assert app.mode=='w4-m2'
    worker_pid=app.primer.child.pid
    memory.append(dict(block='initialized',**snapshot(app.primer)))
    def run(name,prompt,cap,expected=None,language="ja"):
        normalized=prompt or[1]
        baseline=serial_generate(app.session,normalized,cap)
        delivered=[];got=app.generate(prompt,cap,emit=delivered.append,target_language=language)
        assert app.mode=='w4-m2' and app.primer.child.pid==worker_pid,app.fallbacks
        assert got['tokens']==delivered==baseline['tokens'],name
        if expected is not None:assert got['tokens']==expected,name
        assert app.session.length==len(normalized)+len(got['tokens'])-1,(name,app.session.length,len(got['tokens']))
        row=dict(id=name,result='PASS',prompt_tokens=len(prompt),normalized_prompt_tokens=len(normalized),tokens=got['tokens'],serial_wall_s=baseline['wall_s'],default_wall_s=got['wall_s'],ttft_ms=got['ttft_s']*1000,M2_cycles=sum(c['mode']=='M2'for c in got['cycles']),EOS_C1_cycles=sum(c['mode']=='EOS-C1'for c in got['cycles']),serial_phase_rows=got.get('serial_phase_rows'),logical_length=got['logical_length'],stopped_eos=got['stopped_eos'])
        rows.append(row);print(json.dumps({k:v for k,v in row.items()if k!='tokens'}),flush=True)
        return row
    originals={}
    for block in range(3):
        for i,prompt in enumerate(meta['prompts']):
            row=run(f'fixtures-{block}-translation-{i}',prompt,256,language=('en'if i<2 else'ja'))
            originals[i]=row['tokens']
        run(f'fixtures-{block}-paragraph',paragraph,1024,para['generated_tokens'])
        memory.append(dict(block=f'fixtures-{block}',**snapshot(app.primer)))
    for pi,prompt in enumerate(meta['prompts']):
        for j in range(25):
            run(f'prefix-{pi}-{j}',prompt+meta['teacher_tokens'][pi*25:pi*25+j],25,language=('en'if pi<2 else'ja'))
        memory.append(dict(block=f'prefixes-{pi}',**snapshot(app.primer)))
    for name,prompt,cap in [('empty-raw',[],8),('one-token',[1],8),('cap-one',meta['prompts'][0],1),('cap-two',paragraph,2),('cap-three',paragraph,3)]:run(name,prompt,cap)
    assert originals[0][-1]in(1,16)
    eos_prompt=meta['prompts'][0]+originals[0][:-1]
    immediate=run('immediate-EOS',eos_prompt,20)
    assert len(immediate['tokens'])==1 and immediate['stopped_eos'] and immediate['M2_cycles']==0
    for N in(2045,2046,2047,2048):
        prompt=(paragraph*((N+len(paragraph)-1)//len(paragraph)))[:N]
        run(f'context-{N}',prompt,8)
    memory.append(dict(block='limits',**snapshot(app.primer)))
    time.sleep(30)
    run('after-idle-30',meta['prompts'][0],256,originals[0])
    memory.append(dict(block='idle',**snapshot(app.primer)))
    warmed=[x for x in memory if x['block'].startswith('fixtures-')or x['block'].startswith('prefixes-')]
    values=[x['rss_total_bytes']for x in warmed]
    # Report all samples, including peak counters. This is a growth diagnostic,
    # not a threshold weakened to excuse a failed numerical gate.
    sampled= sampler.finish()
    report=dict(sampled_memory=sampled,result='PASS',requests=len(rows),fixture_requests=15,prefix_requests=100,rows=rows,memory=memory,warm_rss_range_bytes=max(values)-min(values),warm_first_to_last_bytes=values[-1]-values[0],spec_ready_ms=app.spec_ready_ms,fallbacks=app.fallbacks,profile_lengths={n:len(getattr(app.drafter,n))for n in('cycle_profile','refresh_profile','head_profile','_fast_profile')},scope='one parent, one CPU worker, same contexts/BOs for every request; serial oracle before each request; exact callbacks/tokens/cursor; no per-request gc or model reload; RSS includes driver records and command caches')
    app.close();report['close_passed']=True
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k not in('rows','memory')}),flush=True)
except BaseException:
    import traceback
    error=traceback.format_exc();a.output.write_text(json.dumps(dict(result='FAIL',rows=rows,memory=memory,error=error),indent=2));print(error,flush=True);sys.exit(1)
