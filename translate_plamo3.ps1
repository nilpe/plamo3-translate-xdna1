# SPDX-License-Identifier: Apache-2.0
# PLaMo 3 Translate on the XDNA1 NPU: public wrapper. Works from Windows PowerShell 5.1 and PowerShell 7, with or without 2>&1.
# Exit codes: 0 ok | 1 internal error | 2 usage / invalid request | 3 installation problem | 4 degraded (-Strict) | 5 NPU busy / unavailable | 6 environment | 130 Ctrl+C
param(
 [Parameter(Position=0)][string]$Text,
 [ValidateSet('ja','en')][string]$SourceLang='ja',
 [ValidateSet('ja','en')][string]$TargetLang='en',
 [ValidateSet('w4-m2','w4-serial','w8-serial')][string]$Mode,
 [int]$NumTokens=256,
 [int]$Wait=-1,
 [switch]$SpecOff,[switch]$Json,[switch]$Service,[switch]$Stats,[switch]$Quiet,[switch]$Strict,
 [switch]$Turbo,[switch]$KeepWarm,[switch]$AllowSerialFallback,[switch]$NoLock,
 [switch]$Version,[switch]$EnvReport,[switch]$Diag,
 # pipeline lines for -Service (in-process pipelines only). The script never references $input: that makes Windows PowerShell 5.1 wait for stdin EOF at start when another program launches it with an open stdin pipe.
 [Parameter(ValueFromPipeline=$true)][string[]]$PipeLine
)
begin{$script:fed=@()}
process{if($null -ne $PipeLine){$script:fed+=$PipeLine}}
end{
$root=$PSScriptRoot
$python="$root\venv\Scripts\python.exe"
if(-not(Test-Path $python)){$python="$root\IRON\.venv\Scripts\python.exe"}
if(-not(Test-Path $python)){
 if(-not(Test-Path "$root\IRON\iron\applications\plamo3\translate.py")){$root=[IO.Path]::GetFullPath("$PSScriptRoot\..\..\..\..")}
 $python="$root\IRON\.venv\Scripts\python.exe"
}
$entry="$root\IRON\iron\applications\plamo3\translate.py"
if(-not((Test-Path $python) -and (Test-Path $entry))){
 [Console]::Error.WriteLine("PLaMo: error: the installation is incomplete (Python environment or entry point missing). Run install.ps1 -Verify.")
 exit 3
}
$info=$Version -or $EnvReport -or $Diag
if($Service -and $Text){
 [Console]::Error.WriteLine("PLaMo: error: text must not be given together with -Service (requests are read from stdin).")
 exit 2
}
if(-not $Service -and -not $info){
 # PowerShell 5.1 drops empty-string native arguments, mangles embedded quotes and Windows limits a command line to 32 KB: the text is therefore passed on stdin (UTF-8).
 if([string]::IsNullOrWhiteSpace($Text)){
  [Console]::Error.WriteLine("PLaMo: error: text is required (and must not be empty). Text that starts with '-' must be passed as -Text '...'.")
  exit 2
 }
}
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);$OutputEncoding=[Console]::OutputEncoding
$env:PYTHONIOENCODING='utf-8';$env:PYTHONUTF8='1';$env:OPENBLAS_NUM_THREADS='4';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$arguments=@($entry)
if($info){
 if($Version){$arguments+='--version'}
 if($EnvReport){$arguments+='--env-report'}
 if($Diag){$arguments+='--diag'}
}else{
 $arguments+=@('--source-lang',$SourceLang,'--target-lang',$TargetLang,'--num-tokens',"$NumTokens")
 if($PSBoundParameters.ContainsKey('Mode')){$arguments+=@('--mode',$Mode)}
 if($SpecOff){$arguments+='--spec-off'}
 if($Json){$arguments+='--json'}
 if($Stats){$arguments+='--stats'}
 if($Quiet){$arguments+='--quiet'}
 if($Strict){$arguments+='--strict'}
 if($Turbo){$arguments+='--turbo'}
 if($KeepWarm){$arguments+='--keep-warm'}
 if($AllowSerialFallback){$arguments+='--allow-serial-fallback'}
 if($NoLock){$arguments+='--no-lock'}
 if($Wait -ge 0){$arguments+=@('--wait',"$Wait")}
 if($Service){$arguments+='--service'}else{$arguments+='--text-stdin'}
}
# Native stderr must never become a terminating error: under $ErrorActionPreference='Stop' PowerShell 5.1 turns merged native stderr into NativeCommandError.
$previous=$ErrorActionPreference
$ErrorActionPreference='Continue'
try{
 if($info){& $python @arguments}
 elseif($Service){ if($script:fed.Count -gt 0){$script:fed | & $python @arguments}else{& $python @arguments} }
 else{ $Text | & $python @arguments }
 $code=$LASTEXITCODE
}finally{
 $ErrorActionPreference=$previous
}
exit $code
}
