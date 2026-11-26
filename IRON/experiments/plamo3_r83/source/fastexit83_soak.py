"""R83 fast-exit-soak: 120 consecutive FRESH-PROCESS one-shot runs through the candidate CLI (cli83, fast exit on, the exported bundle) = 30 x each of: short ja->en, short en->ja, paragraph en->ja, paragraph ja->en
(run n uses fixture n mod 4), in 4 blocks of 30 (step fast-exit-soak, Index 0..3), then `--check` (step fast-exit-soak-check). Each run must exit 0 with tokens identical to the reference (made from the ACTIVE R80 default
through translate.py, frozen below), leave no python process behind (0.4 s after exit, versus the PIDs alive at block start), print no ERT/0xc01e0009/timeout message, and records hw_context_create:context0 (R77_REPORT_STARTUP=1).
The check requires: all 120 present and clean; median context0 of the last 20 runs <= median of the first 20 + 20 ms; available physical memory after the soak within 300 MB of before; a generate step and a startup step of
this epoch that PASSED AFTER the last soak block (they are the next steps of the gate sequence); pmode Default."""
import argparse,ctypes,json,os,re,statistics as st,subprocess as sp,sys,time,traceback
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()));FOLDER=ROOT/'state/gates'
FIX={"s_ja_en": [None, "ja", "en"], "s_en_ja": ["The meeting starts at nine tomorrow morning, so please send the updated report before then.", "en", "ja"], "p_en_ja": [None, "en", "ja"], "p_ja_en": ["昨日の会議では、新しい翻訳システムの導入について長い時間をかけて議論しました。参加者の多くは、処理速度が大幅に向上する点を高く評価しましたが、一部の担当者は、既存のワークフローとの互換性や、導入後の保守にかかる費用について懸念を示しました。最終的に、来月から小規模な試験運用を始め、その結果を見て全社への展開を判断することで合意しました。", "ja", "en"]}
REF={"s_ja_en": [1100, 5457, 48097, 29425, 1781, 44, 4566, 1106, 63, 19105, 1378, 1490, 3704, 46, 16], "s_en_ja": [52499, 50104, 51797, 50100, 50552, 57, 50391, 54411, 53397, 88329, 50639, 50100, 58713, 58854, 50166, 16], "p_en_ja": [90873, 50104, 55287, 50100, 50535, 50101, 55, 48, 52285, 57288, 56236, 50106, 50341, 60916, 51370, 55756, 50100, 62135, 52729, 50178, 59412, 50433, 51245, 50100, 91687, 50230, 50116, 51475, 50650, 50105, 52003, 50100, 77290, 50990, 67635, 51570, 58761, 50360, 71668, 54644, 62397, 50470, 50105, 66623, 86178, 55837, 50684, 52513, 50839, 50105, 68951, 50106, 58769, 51189, 50100, 52428, 54382, 57215, 70587, 50680, 54629, 50433, 55603, 51930, 50104, 65847, 50100, 57566, 51357, 52257, 84387, 86807, 50144, 50, 57112, 51890, 56177, 50105, 56620, 50938, 52581, 50101, 50952, 50105, 51, 48, 50011, 51004, 52763, 55837, 50106, 57703, 63074, 50102, 52631, 53980, 50220, 50144, 52865, 51896, 55756, 50103, 81023, 57928, 60937, 51455, 50103, 53087, 51149, 50922, 82468, 50108, 74054, 51838, 100657, 52727, 50848, 50116, 51419, 57363, 50100, 56313, 50508, 50106, 81173, 50142, 51479, 56255, 50131, 82335, 81902, 50112, 58121, 50341, 50101, 50232, 50720, 50220, 50745, 50100, 65480, 51260, 50780, 51785, 67058, 50227, 50103, 63281, 50670, 52515, 50197, 50108, 50494, 50433, 51, 50164, 51004, 64315, 55258, 50100, 53007, 50101, 90873, 50104, 51245, 52923, 50106, 52853, 52796, 50108, 58701, 52436, 50100, 55668, 54464, 53684, 51798, 60407, 56036, 50106, 50328, 51984, 56236, 50105, 50232, 52729, 55916, 50197, 50593, 52499, 50103, 60578, 50397, 57315, 59797, 50433, 58835, 50220, 50100, 51095, 50116, 51201, 52162, 50218, 52848, 52470, 90873, 50105, 53513, 50100, 53433, 50407, 50100, 52041, 50100, 57703, 74598, 62256, 50105, 86618, 50377, 50137, 55258, 50106, 55446, 59244, 51000, 50108, 59354, 50102, 16], "p_ja_en": [1804, 6815, 1121, 3560, 44, 1139, 4436, 15644, 15380, 1087, 14129, 18745, 10041, 1366, 5010, 7695, 14636, 1087, 3066, 31073, 5287, 3744, 1181, 1165, 7077, 5951, 1160, 21503, 1091, 3799, 22249, 1224, 12339, 7057, 37259, 8532, 1059, 29752, 44, 1139, 11870, 15490, 3579, 13060, 6678, 33787, 48092, 47763, 6052, 1479, 2045, 19490, 1227, 29008, 1705, 46, 16]}
def verify_followups(folder,epoch,end,steps=('generate','startup')):
    """The PASSed generate / startup steps after the soak must have REAL records: summary.json PASS of this epoch, written after the last soak block, AND the step's own output file
    (<step>.json) next to it must exist, parse, say PASS and be newer than the soak end (a step that exits 0 without writing its record is not a pass)."""
    follow={}
    for step in steps:
        ok=[]
        for sj in sorted(Path(folder).glob('*/summary.json')):
            try:row=json.loads(sj.read_text(encoding='utf-8-sig'))
            except Exception:continue
            if not(row.get('step')==step and row.get('package')=='rel' and row.get('result')=='PASS' and row.get('epoch')==epoch and sj.stat().st_mtime>end):continue
            rec=sj.parent/(step+'.json')
            if not rec.is_file() or rec.stat().st_mtime<=end:continue
            try:d=json.loads(rec.read_text(encoding='utf-8-sig'))
            except Exception:continue
            if d.get('result')!='PASS':continue
            ok.append(str(sj))
        follow[step]=ok[:1]
    return follow
def main():
    global a,rep
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--block',type=int);p.add_argument('--check',action='store_true');p.add_argument('--model');a=p.parse_args()
    def avail_mb():
        class M(ctypes.Structure):_fields_=[('l',ctypes.c_ulong),('ml',ctypes.c_ulong),('tp',ctypes.c_ulonglong),('ap',ctypes.c_ulonglong),('tpf',ctypes.c_ulonglong),('apf',ctypes.c_ulonglong),('tv',ctypes.c_ulonglong),('av',ctypes.c_ulonglong),('ae',ctypes.c_ulonglong)]
        m=M();m.l=ctypes.sizeof(M);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return m.ap/1048576
    def ps(cmd):return sp.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,encoding='utf-8',errors='replace').stdout
    def pythons():
        out=ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }")
        return {int(l.split(' ',1)[0]):l for l in out.splitlines() if l.strip() and l.split(' ',1)[0].isdigit()}
    def text(name):
        t=FIX[name][0]
        if t is None:t=(ROOT/('gate-data/short66.txt' if name=='s_ja_en' else 'gate-data/para63.txt')).read_text(encoding='utf-8-sig').strip()
        return t
    def pmode():
        t=sp.run([r'C:\Windows\System32\AMD\xrt-smi.exe','examine','-r','platform'],capture_output=True,text=True,errors='replace').stdout
        m=re.search(r'Power Mode\s*:\s*(\w+)',t);return m.group(1) if m else 'unknown'
    def blockfile(b):return FOLDER/('soak_block_%d.json'%b)
    rep=dict(result='FAIL')
    try:
        if a.check:
            runs=[]
            for b in range(4):
                f=blockfile(b);assert f.exists(),f;d=json.loads(f.read_text(encoding='utf-8'));assert d['result']=='PASS',(b,d.get('error'));runs+=d['runs']
            assert len(runs)==120,len(runs);runs.sort(key=lambda r:r['n']);assert [r['n'] for r in runs]==list(range(120))
            badr=[r for r in runs if not(r['exit']==0 and r['tokens_match'] and not r['leftover'] and not r['bad_stderr'])];assert not badr,badr[:3]
            c0=[r['context0_ms'] for r in runs];first,last=st.median(c0[:20]),st.median(c0[-20:]);assert last<=first+20,(first,last)
            b0=json.loads(blockfile(0).read_text(encoding='utf-8'));b3=json.loads(blockfile(3).read_text(encoding='utf-8'))
            before,after=b0['mem_before_mb'],b3['mem_after_mb'];assert after>=before-300,(before,after)
            end=blockfile(3).stat().st_mtime;epoch=json.loads((FOLDER/'freeze.json').read_text())['epoch'];follow=verify_followups(FOLDER,epoch,end)
            assert follow['generate'] and follow['startup'],('no PASSed generate/startup step after the soak yet',follow)
            assert pmode()=='Default'
            rep=dict(result='PASS',runs=len(runs),context0_first20_median_ms=first,context0_last20_median_ms=last,context0_drift_ms=last-first,context0_all=[round(x,1) for x in c0],mem_available_before_mb=before,mem_available_after_mb=after,
                free_physical_kb_before=b0.get('free_kb_before'),free_physical_kb_after=b3.get('free_kb_after'),following_gates=follow,wall_ms_median={k:st.median(r['wall_ms'] for r in runs if r['fixture']==k) for k in FIX},leftovers=0,bad_stderr=0)
        else:
            b=a.block;assert b in range(4)
            cli=ROOT/'IRON/experiments/plamo3_r83/source/cli83.py';py=ROOT/'venv/Scripts/python.exe'
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',R77_REPORT_STARTUP='1')
            for k in('R83_STAGE','R77_EXIT','R77_TOK','PLAMO3_KEEP_WARM','PLAMO3_NPU_TURBO'):env.pop(k,None)
            assert pmode()=='Default'
            mem0=avail_mb();base=set(pythons());runs=[];free0=ps('(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory').strip()
            for n in range(30*b,30*b+30):
                name=list(FIX)[n%4];sl,tl=FIX[name][1],FIX[name][2];t0=time.perf_counter()
                r=sp.run([str(py),str(cli),'--source-lang',sl,'--target-lang',tl,'--num-tokens','700','--json','--',text(name)],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=120)
                wall=(time.perf_counter()-t0)*1000;time.sleep(0.4);j=None
                for l in r.stdout.splitlines():
                    if l.startswith('{'):j=json.loads(l)
                left=[l for pid,l in pythons().items() if pid not in base]
                ev=[e['ms'] for e in (j or {}).get('startup_events',[]) if e['phase']=='hw_context_create:context0']
                bad=any(x in r.stderr.lower() for x in('c01e0009','ert_cmd','timeout','traceback'))
                runs.append(dict(n=n,fixture=name,exit=r.returncode,wall_ms=wall,tokens_match=bool(j) and j['tokens']==REF[name],spec_ready_ms=(j or {}).get('spec_ready_ms'),ttft_ms=(j or {}).get('ttft_ms'),context0_ms=sum(ev),leftover=left,bad_stderr=bad,stderr_tail=r.stderr[-300:] if (bad or r.returncode) else ''))
                assert r.returncode==0 and runs[-1]['tokens_match'] and not left and not bad,runs[-1]
                time.sleep(1.0)
            rep=dict(result='PASS',block=b,runs=runs,mem_before_mb=mem0,mem_after_mb=avail_mb(),free_kb_before=free0,free_kb_after=ps('(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory').strip(),pmode_after=pmode())
            assert rep['pmode_after']=='Default'
        a.output.write_text(json.dumps(rep,indent=1));print(json.dumps({k:v for k,v in rep.items() if k!='runs'})[:1500],flush=True)
    except BaseException:
        rep['error']=traceback.format_exc();a.output.write_text(json.dumps(rep,indent=1));traceback.print_exc()
        if a.block is not None and not a.check:blockfile(a.block).write_text(json.dumps(rep,indent=1))
        sys.exit(1)
    else:
        if a.block is not None and not a.check:blockfile(a.block).write_text(json.dumps(rep,indent=1))

if __name__=='__main__':main()
