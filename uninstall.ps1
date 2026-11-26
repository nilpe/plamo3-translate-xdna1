# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
<#
.SYNOPSIS
  Removes what install.ps1 created (venv, downloaded payload and base model, state) from the install root.
.DESCRIPTION
  Only files recorded in state\install-manifest.json are removed; the code of the cloned repository, the registry,
  PATH and every other place are never touched (the installer never changed them). A recorded turbo power mode is
  restored first. -KeepModel keeps the 1.8 GB base model. -DryRun lists what would go. Asks for confirmation unless -Yes.
.EXAMPLE
  .\uninstall.ps1 -DryRun
#>
[CmdletBinding()]
param(
  [string]$Root,
  [string]$Python,
  [switch]$KeepModel,
  [switch]$Yes,
  [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
if (-not $Root) { $Root = $PSScriptRoot }
$Root = [IO.Path]::GetFullPath($Root)

# 1. stop processes that run from this installation's venv (they would hold files open)
$venvPrefix = (Join-Path $Root 'venv') + '\'
$procs = @()
try {
  $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venvPrefix, [StringComparison]::OrdinalIgnoreCase) })
} catch { }
if ($procs.Count -gt 0) {
  Write-Host "Python processes running from this installation:"
  $procs | ForEach-Object { Write-Host ("  pid {0}  {1}" -f $_.ProcessId, $_.CommandLine) }
  if ($DryRun) {
    Write-Host "(dry run: they would be stopped)"
  } else {
    $go = $Yes
    if (-not $go) { $go = ((Read-Host 'Stop them? Type YES') -ceq 'YES') }
    if (-not $go) { [Console]::Error.WriteLine('ERROR: processes still running; aborted.'); exit 2 }
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
  }
}

# 2. find a Python to run the removal (the venv itself is going away, so use another interpreter)
function Find-AnyPython {
  if ($Python) { return $Python }
  foreach ($name in 'py.exe', 'python.exe', 'python3.exe') {
    foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
      if ($cmd.Source -like '*\WindowsApps\*') { continue }
      if ($cmd.Source.StartsWith($venvPrefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
      return $cmd.Source
    }
  }
  return $null
}
$exe = Find-AnyPython
if (-not $exe) { [Console]::Error.WriteLine('ERROR: no Python found; pass -Python <path>.'); exit 6 }

# 3. restore a turbo power mode that a crashed run may have left behind (best effort, soft-fail)
$turboDir = Join-Path $Root 'IRON\experiments\plamo3_r83\source'
$venvPy = Join-Path $Root 'venv\Scripts\python.exe'
if ((Test-Path -LiteralPath (Join-Path $turboDir 'turbo83.py')) -and (Test-Path -LiteralPath $venvPy) -and -not $DryRun) {
  try {
    $code = "import sys; sys.path.insert(0, sys.argv[1]); import turbo83; turbo83.stale_check()"
    & $venvPy -c $code $turboDir 2>$null
  } catch { }
}

# 4. remove what the installer created
$pyArgs = @((Join-Path $Root 'install.py'), '--uninstall', '--root', $Root)
if ($KeepModel) { $pyArgs += '--keep-model' }
if ($Yes)       { $pyArgs += '--yes' }
if ($DryRun)    { $pyArgs += '--dry-run' }
$ErrorActionPreference = 'Continue'
if ($exe -like '*\py.exe') { & $exe -3 @pyArgs } else { & $exe @pyArgs }
$rc = $LASTEXITCODE
if ($rc -eq 0 -and -not $DryRun) {
  Write-Host 'If you ever enabled the turbo NPU power mode by hand, restore it with:  & "$env:SystemRoot\System32\AMD\xrt-smi.exe" configure --pmode default'
}
exit $rc
