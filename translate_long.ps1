# SPDX-License-Identifier: Apache-2.0
# PLaMo 3 Translate, any input length: one command. Requests that fit the normal path (prompt + output <= 2,048 tokens, no -Long) are handed UNCHANGED to
# translate_plamo3.ps1 (same arguments, same exit code; nothing is re-implemented). Longer requests run long mode (iGPU prefill + NPU decode, up to 262,144 tokens).
# Works from Windows PowerShell 5.1 and PowerShell 7. Exit codes are those of translate_plamo3.ps1 (0 ok | 2 usage | 3 installation | 5 NPU busy | 6 environment | 130 Ctrl+C).
param(
 [Parameter(Position=0)][string]$Text,
 [string]$InputFile,
 [ValidateSet('ja','en')][string]$SourceLang='ja',
 [ValidateSet('ja','en')][string]$TargetLang='en',
 [ValidateSet('w4-m2','w4-serial','w8-serial')][string]$Mode,
 [int]$NumTokens=256,
 [int]$Wait=-1,
 [switch]$Long,[switch]$NoRepeatStop,
 [switch]$SpecOff,[switch]$Json,[switch]$Service,[switch]$Stats,[switch]$Quiet,[switch]$Strict,
 [switch]$Turbo,[switch]$KeepWarm,[switch]$AllowSerialFallback,[switch]$NoLock,
 [switch]$Version,[switch]$EnvReport,[switch]$Diag,
 [Parameter(ValueFromPipeline=$true)][string[]]$PipeLine
)
begin{$script:fed=@()}
process{if($null -ne $PipeLine){$script:fed+=$PipeLine}}
end{
$root=$PSScriptRoot
$default="$root\translate_plamo3.ps1"
function Invoke-Default {
 # Hand over to the unchanged wrapper with the caller's own arguments (splatting; stdin/-Service pipeline lines are passed on).
 $splat=@{}
 foreach($k in $script:BoundOriginal.Keys){
  if(@('Long','NoRepeatStop','InputFile','PipeLine','Text') -contains $k){continue}
  $splat[$k]=$script:BoundOriginal[$k]
 }
 if($null -ne $script:TextToUse -and -not $script:BoundOriginal.ContainsKey('Service')){$splat['Text']=$script:TextToUse}
 if($script:BoundOriginal.ContainsKey('Service') -and $script:fed.Count -gt 0){ $script:fed | & $default @splat } else { & $default @splat }
 $script:rc=$LASTEXITCODE
}
$script:BoundOriginal=$PSBoundParameters
$script:TextToUse=$Text
$info=$Version -or $EnvReport -or $Diag
if($info -or $Service){ Invoke-Default; exit $script:rc }
# text: -InputFile (UTF-8) > -Text > pipeline
if($InputFile){
 if(-not(Test-Path -LiteralPath $InputFile)){ [Console]::Error.WriteLine("PLaMo: error: input file not found: $InputFile"); exit 2 }
 $script:TextToUse=[System.Text.Encoding]::UTF8.GetString([System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $InputFile).Path))
 if($script:TextToUse.Length -gt 0 -and $script:TextToUse[0] -eq [char]0xFEFF){$script:TextToUse=$script:TextToUse.Substring(1)}
 $script:TextToUse=$script:TextToUse -replace "(\r?\n)$",""
}elseif([string]::IsNullOrEmpty($Text) -and $script:fed.Count -gt 0){
 $script:TextToUse=($script:fed -join "`n")
}
if([string]::IsNullOrWhiteSpace($script:TextToUse)){ Invoke-Default; exit $script:rc }   # the default wrapper owns the usage errors
$bytes=[System.Text.Encoding]::UTF8.GetByteCount($script:TextToUse)
$explicit=$PSBoundParameters.ContainsKey('NumTokens')
$cap= if($explicit){$NumTokens}else{256}
# Cheap pre-filter (proof: every UTF-8 byte is at most one byte-fallback token, plus about 40 template tokens): small requests never start Python here.
if(-not $Long -and ($bytes + 40 + $cap) -le 2048){ Invoke-Default; exit $script:rc }
$python="$root\venv\Scripts\python.exe"
$entry="$root\long\long_main.py"
if(-not((Test-Path $python) -and (Test-Path $entry))){
 [Console]::Error.WriteLine("PLaMo: error: this input needs long mode, which is not installed. Run: python install_long.py --with-model (see LONG.md).")
 exit 3
}
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);$OutputEncoding=[Console]::OutputEncoding
$env:PYTHONIOENCODING='utf-8';$env:OPENBLAS_NUM_THREADS='4';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$arguments=@('-I','-X','utf8','-B',$entry,'--text-stdin','--source-lang',$SourceLang,'--target-lang',$TargetLang)
if($explicit){$arguments+=@('--num-tokens',"$NumTokens")}
if($Long){$arguments+='--long'}else{$arguments+='--delegate-if-default'}
if($NoRepeatStop){$arguments+='--no-repeat-stop'}
if($Json){$arguments+='--json'}
if($Stats){$arguments+='--stats'}
if($Quiet){$arguments+='--quiet'}
if($NoLock){$arguments+='--no-lock'}
if($Wait -ge 0){$arguments+=@('--wait',"$Wait")}
$previous=$ErrorActionPreference
$ErrorActionPreference='Continue'
try{
 $script:TextToUse | & $python @arguments
 $code=$LASTEXITCODE
}finally{
 $ErrorActionPreference=$previous
}
if($code -eq 90){ Invoke-Default; $code=$script:rc }
exit $code
}
