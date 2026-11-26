# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
<#
.SYNOPSIS
  Installer bootstrap for the PLaMo 3 Translate XDNA1-NPU runtime (Windows PowerShell 5.1 or PowerShell 7).
.DESCRIPTION
  Finds a 64-bit CPython 3.13, then runs install.py (all logic is in Python). No administrator rights needed.
  Run it from the cloned repository; that folder is the install root unless -Root is given.
  Exit codes: 0 ok, 2 usage, 3 verification, 5 no NPU/driver, 6 environment, 7 network, 8 licence not accepted,
  10+N self-test step N failed. See INSTALL.md.
.EXAMPLE
  .\install.ps1 -AcceptLicense
.EXAMPLE
  .\install.ps1 -Plan                      # checks only, changes nothing
.EXAMPLE
  .\install.ps1 -Verify                    # re-verify an installation
#>
[CmdletBinding()]
param(
  [string]$Root,
  [string]$Python,
  [switch]$AcceptLicense,
  [string]$PayloadSource,
  [string]$ModelSource,
  [switch]$NoModel,
  [switch]$WithGateData,
  [switch]$Verify,
  [switch]$Reattest,
  [switch]$SelfTestOnly,
  [switch]$SkipSelfTest,
  [switch]$Plan,
  [switch]$NoNpuCheck,
  [switch]$RequireSignature,
  [string]$Fingerprint,
  [string]$SumsSig,
  [switch]$Quiet,
  [string]$LogPath
)
$ErrorActionPreference = 'Stop'
try {
  [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
  $OutputEncoding = [Console]::OutputEncoding
} catch { }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Test-Python313([string]$exe, [string[]]$prefix) {
  # returns the interpreter's own path when it is CPython 3.13, 64-bit; otherwise $null
  try {
    # no double quotes inside the code: Windows PowerShell 5.1 strips them from native command arguments
    $code = 'import sys;print(sys.implementation.cache_tag+chr(124)+str(sys.maxsize>2**32)+chr(124)+sys.executable)'
    $out = & $exe @prefix -c $code 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    $parts = ("$out").Trim().Split('|')
    if ($parts.Length -ge 3 -and $parts[0] -eq 'cpython-313' -and $parts[1] -eq 'True') { return $parts[2] }
  } catch { }
  return $null
}

function Find-Python313 {
  if ($Python) {
    $p = Test-Python313 $Python @()
    if ($p) { return $p }
    throw "-Python '$Python' is not a 64-bit CPython 3.13."
  }
  $py = Get-Command py.exe -ErrorAction SilentlyContinue
  if ($py) {
    $p = Test-Python313 $py.Source @('-3.13')
    if ($p) { return $p }
  }
  if ($py) {
    # every interpreter the launcher knows (python.org, Store, uv/Astral builds appear here with their own tags)
    foreach ($line in @(& $py.Source -0p 2>$null)) {
      if ("$line" -match '([A-Za-z]:\\[^\r\n]*python[0-9.]*\.exe)\s*$') {
        $p = Test-Python313 $Matches[1] @()
        if ($p) { return $p }
      }
    }
  }
  foreach ($name in 'python.exe', 'python3.exe') {
    foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
      if ($cmd.Source -like '*\WindowsApps\*') { continue }   # Store alias stubs
      $p = Test-Python313 $cmd.Source @()
      if ($p) { return $p }
    }
  }
  return $null
}

$exe = $null
try { $exe = Find-Python313 } catch {
  [Console]::Error.WriteLine("ERROR: $($_.Exception.Message)")
  exit 6
}
if (-not $exe) {
  [Console]::Error.WriteLine(@"
ERROR: no 64-bit CPython 3.13 was found.
  The NPU module of this runtime is built for Python 3.13 (64-bit). Install it without administrator rights:
    - python.org installer: https://www.python.org/downloads/windows/  ("Windows installer (64-bit)", choose "Install for me only"),
    - or:  winget install Python.Python.3.13
    - or:  conda create -n plamo313 python=3.13  (and run this script from that environment).
  Then run install.ps1 again, or pass -Python <path to python.exe>.
"@)
  exit 6
}

$script = Join-Path $PSScriptRoot 'install.py'
if (-not (Test-Path -LiteralPath $script)) {
  [Console]::Error.WriteLine("ERROR: install.py not found next to install.ps1 ($PSScriptRoot).")
  exit 3
}
$pyArgs = New-Object System.Collections.Generic.List[string]
$pyArgs.Add($script)
if ($Root)             { $pyArgs.Add('--root'); $pyArgs.Add($Root) }
if ($AcceptLicense)    { $pyArgs.Add('--accept-license') }
if ($PayloadSource)    { $pyArgs.Add('--payload-source'); $pyArgs.Add($PayloadSource) }
if ($ModelSource)      { $pyArgs.Add('--model-source'); $pyArgs.Add($ModelSource) }
if ($NoModel)          { $pyArgs.Add('--no-model') }
if ($WithGateData)     { $pyArgs.Add('--with-gate-data') }
if ($Verify)           { $pyArgs.Add('--verify') }
if ($Reattest)         { $pyArgs.Add('--reattest') }
if ($SelfTestOnly)     { $pyArgs.Add('--selftest-only') }
if ($SkipSelfTest)     { $pyArgs.Add('--skip-selftest') }
if ($Plan)             { $pyArgs.Add('--plan') }
if ($NoNpuCheck)       { $pyArgs.Add('--no-npu-check') }
if ($RequireSignature) { $pyArgs.Add('--require-signature') }
if ($Fingerprint)      { $pyArgs.Add('--fingerprint'); $pyArgs.Add($Fingerprint) }
if ($SumsSig)          { $pyArgs.Add('--sums-sig'); $pyArgs.Add($SumsSig) }
if ($Quiet)            { $pyArgs.Add('--quiet') }
if ($LogPath)          { $pyArgs.Add('--log'); $pyArgs.Add($LogPath) }

# Native stderr must not turn into terminating PowerShell errors.
$ErrorActionPreference = 'Continue'
& $exe @pyArgs
exit $LASTEXITCODE
