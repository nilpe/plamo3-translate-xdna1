"""Release gate index (H1). Indexes the step list of the release suite (the complete traced suite run on the installed tree after the freeze) and
fails unless every listed step has a fresh, parseable PASS record of the frozen epoch.

For the epoch of <root>\\state\\gates\\freeze.json:
  * every record folder of the epoch (summary.json, package 'rel') must parse: an unparseable record fails the index;
  * every listed (step, index) has at least as many PASS records (exit 0, this epoch, written after the freeze) as it is listed, and its NEWEST record is a PASS;
  * a record of another epoch never counts and a record older than the freeze fails (a step with only such records is missing);
  * the step output json next to a summary (when the step writes one) must say PASS;
  * the startup records are distinct processes by (pid, process creation time) (R84 rule, distinct_processes);
  * the export record's audits and the 15-case fallback record must be PASS.
usage: gate_index_rel.py --output FILE.json [--root DIR]    (default root: the installation this file is in)"""
import argparse, collections, hashlib, json, sys
from pathlib import Path

SUITE_A = (
    'freeze,export,hybrid-wait-gate,sentinel-proof,sentinel-dyn-decode-0of6,sentinel-dyn-decode-1of6,sent'
    'inel-dyn-decode-2of6,sentinel-dyn-decode-3of6,sentinel-dyn-decode-4of6,sentinel-dyn-decode-5of6,sent'
    'inel-dyn-bulk-0of6,sentinel-dyn-bulk-1of6,sentinel-dyn-bulk-2of6,sentinel-dyn-bulk-3of6,sentinel-dyn'
    '-bulk-4of6,sentinel-dyn-bulk-5of6,sentinel-dyn-verify-0of6,sentinel-dyn-verify-1of6,sentinel-dyn-ver'
    'ify-2of6,sentinel-dyn-verify-3of6,sentinel-dyn-verify-4of6,sentinel-dyn-verify-5of6,sentinel-dyn-dra'
    'fter-0of6,sentinel-dyn-drafter-1of6,sentinel-dyn-drafter-2of6,sentinel-dyn-drafter-3of6,sentinel-dyn'
    '-drafter-4of6,sentinel-dyn-drafter-5of6,final-table,cli-check,lease-integrity,smoke,services,native,'
    'fallback,language-fallback,protocol,cap-protocol,eos-protocol,fallback-protocol,switch-protocol,gate'
    '-mutants,startup-mutants,confidence-run,turbo-gate,turbo-e2e,keepwarm-gate,keepwarm-e2e,prefill-gate'
    ',head-swap-gate,prepare-gate,overlap-protocol,mailbox,single,cache,eos-native,verify-teacher,verify-'
    'rollback,verify-generation,verify-features,integrity,hash,startup,generate,tokenizer-gate,fast-exit-'
    'gate,startup@0,generate,resident,memory,regress-args,regress-service,regress-wrapper,regress-lock,re'
    'gress-orphan,regress-multibyte,regress-fallback'
)
SUITE_B = (
    'relgate-n1b,relgate-n1c,relgate-n2,relgate-n3,relgate-n4,relgate-n5,relgate-n7,relgate-n8,relgate-n1'
    '5,relgate-n14'
)
SUITE_C = (
    'fast-exit-soak@0,fast-exit-soak@1,fast-exit-soak@2,fast-exit-soak@3,startup@1,generate,fast-exit-soa'
    'k-check'
)

SUITE = tuple(SUITE_A.split(',') + SUITE_B.split(',') + SUITE_C.split(','))


def distinct_processes(records,min_gap_ms=100.0):
    """R84: a process is identified by (pid, process creation time), not by the pid alone (Windows reused pid 385376 for two startups 2.7 s apart in R83).
    records = [(pid, proc_start_epoch_ms or None)]. Records with a creation time are distinct when their pids differ or their creation times differ by more than min_gap_ms
    (one process cannot appear twice with different creation times; two processes cannot start within one gap of each other and both finish a startup).
    A record WITHOUT a creation time (frozen older epochs) keeps the R83 rule: its pid must be unique among all records of that pid."""
    by={}
    for pid,start in records:by.setdefault(pid,[]).append(start)
    for pid,starts in by.items():
        if len(starts)==1:continue
        if any(s is None for s in starts):return False
        s=sorted(starts)
        if any(b-a<=min_gap_ms for a,b in zip(s,s[1:])):return False
    return True


def build_index(folder,suite=SUITE):
    """Raises AssertionError (with the reason) unless the records in `folder` (= <root>/state/gates) index the suite. Returns the report dict."""
    freeze=json.loads((folder/'freeze.json').read_text(encoding='utf-8-sig'))
    epoch=freeze['epoch'];t0=freeze['started_ns']/1e9
    want=collections.Counter()
    for s in suite:
        name,_,idx=s.partition('@');want[(name,int(idx or 0))]+=1
    records=collections.defaultdict(list)
    for path in sorted(folder.glob('*/summary.json')):
        try:row=json.loads(path.read_text(encoding='utf-8-sig'))
        except Exception as e:raise AssertionError('unparseable record %s (%s)'%(path.parent.name,type(e).__name__))
        if not isinstance(row,dict) or 'step' not in row or 'result' not in row:raise AssertionError('malformed record %s'%path.parent.name)
        if row.get('package')!='rel':continue
        if row['step'] not in ('freeze','export'):
            if row.get('epoch')!=epoch:continue                       # another epoch: never counts
            if path.stat().st_mtime<t0-1:raise AssertionError('stale record %s: older than the freeze'%path.parent.name)
        records[(row['step'],int(row.get('index',0)))].append((path,row))
    rows=[];pids=[];missing=[]
    for key,n in sorted(want.items()):
        got=records.get(key,[])
        passes=[(p,r) for p,r in got if r['result']=='PASS' and r.get('exit_code')==0]
        if len(passes)<n or got[-1][1]['result']!='PASS':missing.append((key,n,len(passes)));continue
        newest=got[-1][0]
        out=newest.parent/('%s.json'%key[0])
        if out.is_file():
            try:rec=json.loads(out.read_text(encoding='utf-8-sig'))
            except Exception as e:raise AssertionError('unparseable step output %s/%s (%s)'%(newest.parent.name,out.name,type(e).__name__))
            assert rec.get('result','PASS')=='PASS',('step output not PASS',key,newest.parent.name)
        rows.append(dict(step=key[0],index=key[1],listed=n,passes=len(passes),summary=str(newest),sha256=hashlib.sha256(newest.read_bytes()).hexdigest(),
                         receipts=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(newest.parent.glob('*.json')) if f!=newest]))
    assert not missing,('steps without enough fresh PASS records (step, listed, passes)',missing)
    for key in sorted(k for k in want if k[0]=='startup'):
        for path,row in records[key]:
            if row['result']!='PASS':continue
            r=json.loads((path.parent/'startup.json').read_text(encoding='utf-8-sig'));assert r['result']=='PASS' and r['token_identity'],('startup record',path.parent.name)
            pids.append((r['pid'],r.get('proc_start_epoch_ms')))
    assert len(pids)>=sum(n for k,n in want.items() if k[0]=='startup'),'startup records'
    assert distinct_processes(pids),('startup records are not distinct processes',pids)
    export=json.loads((records[('export',0)][-1][0].parent/'export.json').read_text(encoding='utf-8-sig'));assert export['result']=='PASS'
    for value in export.get('audits',{}).values():
        if isinstance(value,dict) and 'result' in value:assert value['result']=='PASS'
    fb=json.loads((records[('fallback',0)][-1][0].parent/'fallback.json').read_text(encoding='utf-8-sig'));assert fb['result']=='PASS' and fb['cases']==15
    return dict(result='PASS',epoch=epoch,steps=len(want),listed_runs=sum(want.values()),gates=rows,startup_distinct_processes=len(pids))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--root',type=Path);a=p.parse_args()
    if a.root is None:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'IRON'/'iron'/'applications'/'plamo3'))
        from plamo_root import ROOT
        a.root=ROOT
    try:report=build_index(a.root/'state'/'gates')
    except AssertionError as e:
        a.output.write_text(json.dumps(dict(result='FAIL',error=str(e)[:2000]),indent=1));print('FAIL',str(e)[:600]);return 1
    a.output.write_text(json.dumps(report,indent=1));print(json.dumps(dict(result='PASS',steps=report['steps'],listed_runs=report['listed_runs'],epoch=report['epoch'])),flush=True);return 0


if __name__=='__main__':sys.exit(main())
