# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
# Release gate runner (ported from the author's R83 runner): one supervised step per call; records go to <root>\state\gates. One NPU job at a time (state\npu-job.lock).
param([Parameter(Mandatory=$true)][string]$Step,[ValidateSet('rel','off','on','auto')][string]$Package='rel',[int]$Index=0,[int]$IdleSeconds=0,[switch]$Trace)
$ErrorActionPreference='Stop';$root=(Split-Path -Parent $PSScriptRoot);$src="$root\IRON\experiments\plamo3_r83\source"
New-Item -ItemType Directory -Force "$root\state\gates","$root\tmp"|Out-Null
$lock=[IO.File]::Open("$root\state\npu-job.lock",'OpenOrCreate','ReadWrite','None')
try{
 $env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8';$env:PYTHONUTF8='1';$env:OPENBLAS_NUM_THREADS='4';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4';$env:NPU2='0';$env:USE_TORCH='0';$env:R42_BOOTSTRAP='fastnorm'
 $folder="$root\state\gates\$((Get-Date).ToString('yyyyMMdd-HHmmss-fff'))-$Step-$Package-$Index";New-Item -ItemType Directory -Force $folder|Out-Null
 # Arms (release): rel = the shipped default; off/on/auto = the same runtime with PLAMO3_KEEP_WARM 0/1/auto.
 Remove-Item Env:PLAMO3_KEEP_WARM -ErrorAction SilentlyContinue;Remove-Item Env:PLAMO3_KEEP_WARM_CPU -ErrorAction SilentlyContinue;Remove-Item Env:PLAMO3_KEEP_WARM_SPAWN -ErrorAction SilentlyContinue
 if($true){Remove-Item Env:R83_PACKAGE,Env:R83_STAGE -ErrorAction SilentlyContinue;Remove-Item Env:R83_KEEP_WARM_STATS -ErrorAction SilentlyContinue;if($Package -eq 'off'){$env:PLAMO3_KEEP_WARM='0'}elseif($Package -eq 'on'){$env:PLAMO3_KEEP_WARM='1';$env:R83_KEEP_WARM_STATS='1'}elseif($Package -eq 'auto'){$env:PLAMO3_KEEP_WARM='auto';$env:R83_KEEP_WARM_STATS='1'}
  if($env:R83_KW_CPU){$env:PLAMO3_KEEP_WARM_CPU=$env:R83_KW_CPU};if($env:R83_KW_SPAWN){$env:PLAMO3_KEEP_WARM_SPAWN=$env:R83_KW_SPAWN}}
 $map=@{'lease-integrity'='lease_gate83.py';export='export_check_rel.py';services='service83.py';native='native83.py';generate='compare83.py';startup='startup83.py';fallback='fallback83.py';hash='hash83.py';freeze='freeze_rel.py';resident='resident83.py';'language-fallback'='language_fallback83.py';smoke='smoke83.py';single='single83.py';mailbox='mailbox83.py';memory='memory83.py';'cli-check'='cli_check83.py';'final-table'='sentence_final83.py';'confidence-run'='compare83.py';'confidence-gate'='confidence83_gate.py';'turbo-gate'='turbo83_gate.py';'keepwarm-gate'='keepwarm83_gate.py';'keepwarm-e2e'='keepwarm83_e2e.py';'turbo-e2e'='turbo83_e2e.py';'tokenizer-gate'='tokenizer83_gate.py';'fast-exit-gate'='fastexit83_gate.py';'fast-exit-soak'='fastexit83_soak.py';'fast-exit-soak-check'='fastexit83_soak.py'}
 $legacy=@{cache='cache_gate.py';'eos-native'='eos_native.py';integrity='integrity_gate.py';'verify-teacher'='verify_run.py';'verify-rollback'='verify_run.py';'verify-generation'='verify_run.py';'verify-features'='verify_run.py'}
 $protocol=@{protocol='lib\r37\protocol_tests.py';'overlap-protocol'='lib\r43\overlap_protocol.py'}
 $local=@{'relgate-n2'='relgates_rel.py';'relgate-n3'='relgates_rel.py';'relgate-n4'='relgates_rel.py';'relgate-n5'='relgates_rel.py';'relgate-n7'='relgates_rel.py';'relgate-n1b'='relgates_rel.py';'relgate-n1c'='relgates_rel.py';'relgate-n14'='relgates_rel.py';'relgate-n15'='relgates_rel.py';'relgate-n8'='relgates_rel.py';'regress-args'='relregress_rel.py';'regress-service'='relregress_rel.py';'regress-wrapper'='relregress_rel.py';'regress-lock'='relregress_rel.py';'regress-orphan'='relregress_rel.py';'regress-fallback'='relregress_rel.py';'regress-multibyte'='relregress_rel.py';'hybrid-wait-gate'='hybridwait83_gate.py';'sentinel-proof'='sentinelproof83_gate.py';'sentinel-dyn-decode-0of6'='sentineldyn83_gate.py';'sentinel-dyn-decode-1of6'='sentineldyn83_gate.py';'sentinel-dyn-decode-2of6'='sentineldyn83_gate.py';'sentinel-dyn-decode-3of6'='sentineldyn83_gate.py';'sentinel-dyn-decode-4of6'='sentineldyn83_gate.py';'sentinel-dyn-decode-5of6'='sentineldyn83_gate.py';'sentinel-dyn-bulk-0of6'='sentineldyn83_gate.py';'sentinel-dyn-bulk-1of6'='sentineldyn83_gate.py';'sentinel-dyn-bulk-2of6'='sentineldyn83_gate.py';'sentinel-dyn-bulk-3of6'='sentineldyn83_gate.py';'sentinel-dyn-bulk-4of6'='sentineldyn83_gate.py';'sentinel-dyn-bulk-5of6'='sentineldyn83_gate.py';'sentinel-dyn-verify-0of6'='sentineldyn83_gate.py';'sentinel-dyn-verify-1of6'='sentineldyn83_gate.py';'sentinel-dyn-verify-2of6'='sentineldyn83_gate.py';'sentinel-dyn-verify-3of6'='sentineldyn83_gate.py';'sentinel-dyn-verify-4of6'='sentineldyn83_gate.py';'sentinel-dyn-verify-5of6'='sentineldyn83_gate.py';'sentinel-dyn-drafter-0of6'='sentineldyn83_gate.py';'sentinel-dyn-drafter-1of6'='sentineldyn83_gate.py';'sentinel-dyn-drafter-2of6'='sentineldyn83_gate.py';'sentinel-dyn-drafter-3of6'='sentineldyn83_gate.py';'sentinel-dyn-drafter-4of6'='sentineldyn83_gate.py';'sentinel-dyn-drafter-5of6'='sentineldyn83_gate.py';'head-swap-gate'='headswap83_gate.py';'prepare-gate'='prepare83_gate.py';'cap-protocol'='cap83_protocol.py';'eos-protocol'='eos83_protocol.py';'fallback-protocol'='fallback83_protocol.py';'switch-protocol'='switch83_protocol.py';'prefill-gate'='prefill_gate83.py';'gate-mutants'='gate_mutants83.py';'startup-mutants'='startup_mutants83.py';'hoist-gate-sdecode-0of8'='hoistgate83.py';'hoist-gate-sdecode-1of8'='hoistgate83.py';'hoist-gate-sdecode-2of8'='hoistgate83.py';'hoist-gate-sdecode-3of8'='hoistgate83.py';'hoist-gate-sdecode-4of8'='hoistgate83.py';'hoist-gate-sdecode-5of8'='hoistgate83.py';'hoist-gate-sdecode-6of8'='hoistgate83.py';'hoist-gate-sdecode-7of8'='hoistgate83.py';'hoist-gate-sbulk-0of8'='hoistgate83.py';'hoist-gate-sbulk-1of8'='hoistgate83.py';'hoist-gate-sbulk-2of8'='hoistgate83.py';'hoist-gate-sbulk-3of8'='hoistgate83.py';'hoist-gate-sbulk-4of8'='hoistgate83.py';'hoist-gate-sbulk-5of8'='hoistgate83.py';'hoist-gate-sbulk-6of8'='hoistgate83.py';'hoist-gate-sbulk-7of8'='hoistgate83.py';'hoist-gate-sverifya-0of8'='hoistgate83.py';'hoist-gate-sverifya-1of8'='hoistgate83.py';'hoist-gate-sverifya-2of8'='hoistgate83.py';'hoist-gate-sverifya-3of8'='hoistgate83.py';'hoist-gate-sverifya-4of8'='hoistgate83.py';'hoist-gate-sverifya-5of8'='hoistgate83.py';'hoist-gate-sverifya-6of8'='hoistgate83.py';'hoist-gate-sverifya-7of8'='hoistgate83.py';'hoist-gate-sverifyb'='hoistgate83.py';'hoist-gate-drafter-0of4'='hoistgate83.py';'hoist-gate-drafter-1of4'='hoistgate83.py';'hoist-gate-drafter-2of4'='hoistgate83.py';'hoist-gate-drafter-3of4'='hoistgate83.py';'hoist-gate-patch'='hoistgate83.py';'hoist-gate-mutants'='hoistgate83.py';'hoist-gate-device'='hoistgate83.py'}
 foreach($key in $legacy.Keys){$map[$key]="$root\lib\r43\$($legacy[$key])"}
 foreach($key in $protocol.Keys){$map[$key]="$root\$($protocol[$key])"}
 foreach($key in $local.Keys){$map[$key]="$src\$($local[$key])"}
 if(-not $map.ContainsKey($Step)){throw "unknown R83 step $Step"}
 $script=$map[$Step];if(-not[IO.Path]::IsPathRooted($script)){$script="$src\$script"}
 $launch=@($script);$extra=@()
 if($Step -eq 'single' -or $legacy.ContainsKey($Step)){$launch=@("$src\bridge83.py",'legacy',$script)}
 if($protocol.ContainsKey($Step) -or ($local.ContainsKey($Step) -and $Step -ne 'gate-mutants' -and $Step -ne 'startup-mutants')){$launch=@("$src\bridge83.py",'direct',$script)}
 if($Step.StartsWith('verify-')){$extra=@('--mode',$Step.Substring(7))}
 if($Step -like 'relgate-*'){$extra=@('--gate',$Step.Substring(8))}
 if($Step -like 'regress-*'){$extra=@('--group',$Step.Substring(8))}
 if($Step -like 'hoist-gate-*'){$extra=@('--part',$Step.Substring(11))}
 if($Step -like 'sentinel-dyn-*'){$extra=@('--part',$Step.Substring(13))}
 if($Step -eq 'fast-exit-soak'){$extra=@('--block',"$Index")}
 if($Step -eq 'fast-exit-soak-check'){$extra=@('--check')}
 if($Step -eq 'confidence-run'){$env:R83_LOG_LOGITS='1';$env:R83_AUDIT_CHECK='1'}
 if($Step -eq 'confidence-gate'){
  $cal=@(Get-ChildItem "$root\gate-data\r55" -Directory -Filter '*-calibrate-*'|Sort-Object Name|ForEach-Object{"$($_.FullName)\calibrate.json"})
  $dec=@(Get-ChildItem "$root\gate-data\r55" -Directory -Filter '*-generate-rel-*'|Sort-Object Name|ForEach-Object{"$($_.FullName)\generate.json"}|Where-Object{Test-Path $_})
  $fresh=@(Get-ChildItem "$root\state\gates" -Directory -Filter '*-confidence-run-rel-*'|Sort-Object Name|ForEach-Object{"$($_.FullName)\confidence-run.json"}|Where-Object{Test-Path $_})
  if($fresh.Count -lt 1){throw 'confidence-gate needs at least one confidence-run (fresh R83 logits)'}
  $extra=@('--min-calibration-cycles','3097','--min-decision-cycles','14554','--calibration')+$cal+@('--decision')+$dec+@('--fresh')+$fresh}
 if($Step -eq 'generate'){$extra=@('--idle',"$IdleSeconds");if($IdleSeconds){Start-Sleep -Seconds $IdleSeconds}}
 if($Step -eq 'memory'){$primary=Get-ChildItem "$root\state\gates" -Directory -Filter '*-resident-rel-*'|Sort-Object Name|Select-Object -Last 1;$extra=@('--resident',"$($primary.FullName)\resident.json")}
 if($Step -ne 'freeze' -and $Step -ne 'export' -and (Test-Path "$root\state\gates\freeze.json")){
  $frozen=Get-Content "$root\state\gates\freeze.json" -Raw|ConvertFrom-Json
  foreach($prop in $frozen.source_hashes.PSObject.Properties){if((Get-FileHash $prop.Name).Hash.ToLowerInvariant() -ne $prop.Value){throw "frozen source changed: $($prop.Name)"}}
  if((Get-FileHash "$root\plamo3-default\r45-spec\certificate.json").Hash.ToLowerInvariant() -ne $frozen.candidate_sha256){throw 'installed certificate changed since the freeze'}
 }
 $epoch='';if(Test-Path "$root\state\gates\freeze.json"){$epoch=(Get-Content "$root\state\gates\freeze.json" -Raw|ConvertFrom-Json).epoch}
 if($Trace){
  # Diagnostic: audit-hook trace of every package open (diag\tracesite); gate steps are only logged, runtime steps also DENY gate-only opens.
  $env:PYTHONPATH="$root\IRON\experiments\plamo3_r83\diag\tracesite;"+$env:PYTHONPATH;$env:R83_TRACE_DIR="$folder\trace"
  if(@('native','services','single','hash','integrity') -notcontains $Step){$env:R83_TRACE_DENY='1'}
 }
 $args47=@("$root\IRON\iron\applications\plamo3\r20_supervisor.py",'--log',"$folder\foreground.log",'--timeout',$(if($env:H1_GATE_TIMEOUT){$env:H1_GATE_TIMEOUT}else{'550'}),'--')+$launch+@('--output',"$folder\$Step.json")+$extra
 $at=Get-Date
 & "$root\venv\Scripts\python.exe" @args47 2>&1|Tee-Object "$folder\supervisor.log";$code=$LASTEXITCODE
 # R83: a step that exits 0 but did not write its own record (e.g. a harness ended by os._exit) is NOT a pass: the step's output file must exist, be newer than the step start, parse and say PASS.
 if($code -eq 0 -and ((@('startup','generate','confidence-run','tokenizer-gate','fast-exit-gate','fast-exit-soak','fast-exit-soak-check') -contains $Step -or $Step -like 'hoist-gate-*'))){
  $o="$folder\$Step.json";$okrec=$false
  if((Test-Path $o) -and ((Get-Item $o).LastWriteTime -ge $at)){try{$j=(Get-Content $o -Raw -Encoding UTF8)|ConvertFrom-Json;$okrec=($j.result -eq 'PASS')}catch{$okrec=$false}}
  if(-not $okrec){"record check FAILED: $o missing, stale, unparseable or not PASS";$code=3}
 }
 @{result=$(if($code -eq 0){'PASS'}else{'FAIL'});step=$Step;package=$Package;index=$Index;seconds=((Get-Date)-$at).TotalSeconds;folder=$folder;exit_code=$code;idle_seconds=$IdleSeconds;epoch=$epoch;arguments=$args47}|ConvertTo-Json -Depth 8|Set-Content "$folder\summary.json" -Encoding UTF8
 exit $code
}finally{$lock.Dispose()}

