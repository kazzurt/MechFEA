# Recreate the authorized isolated CalculiX build. No global PATH modification.
$ErrorActionPreference = 'Stop'
$solverDirectory = $PSScriptRoot
$archivePath = Join-Path $solverDirectory 'CalculiX-2.23.0-win-x64.zip'
$archiveUrl = 'https://raw.githubusercontent.com/calculix/CalculiX-Windows/60c9a10b350ec813f4d493098f47f7a8b96cd276/releases/CalculiX-2.23.0-win-x64.zip'
$expectedArchiveHash = '79a674fdee735a22720bbd12bb9ccbadad1e8c8ad5b1de74e1c62f8808b2d395'
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath
}
$actualArchiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualArchiveHash -ne $expectedArchiveHash) { throw 'CalculiX archive hash mismatch' }
$executablePath = Join-Path $solverDirectory 'CalculiX-2.23.0-win-x64/bin/ccx.exe'
if (-not (Test-Path -LiteralPath $executablePath)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $solverDirectory
}
# Validate bundled runtime libraries against the pinned archive as well as EXEs.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    foreach ($entry in $archive.Entries) {
        if ($entry.FullName -notmatch '^CalculiX-2\.23\.0-win-x64/bin/[^/]+\.(exe|dll)$') { continue }
        $installedPath = Join-Path $solverDirectory $entry.FullName
        if (-not (Test-Path -LiteralPath $installedPath)) { throw "Missing bundled binary: $($entry.FullName)" }
        $entryStream = $entry.Open()
        $entryDigest = [System.Security.Cryptography.SHA256]::Create()
        try { $archiveEntryHash = [System.BitConverter]::ToString($entryDigest.ComputeHash($entryStream)).Replace('-', '').ToLowerInvariant() }
        finally { $entryStream.Dispose(); $entryDigest.Dispose() }
        if ((Get-FileHash -LiteralPath $installedPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $archiveEntryHash) {
            throw "Bundled binary differs from pinned archive: $($entry.FullName)"
        }
    }
}
finally { $archive.Dispose() }
$expectedExecutableHashes = @{
    'ccx.exe' = '27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f'
    'ccx_MT.exe' = '62349096f3b76cef2f0e094d03f747187016683a233ec7f07a363c7089804876'
}
foreach ($name in @('ccx.exe', 'ccx_MT.exe')) {
    $verifiedExecutable = Join-Path $solverDirectory "CalculiX-2.23.0-win-x64/bin/$name"
    if (-not (Test-Path -LiteralPath $verifiedExecutable)) { throw "Incomplete solver installation: missing $name" }
    $actualExecutableHash = (Get-FileHash -LiteralPath $verifiedExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualExecutableHash -ne $expectedExecutableHashes[$name]) { throw "CalculiX executable hash mismatch: $name" }
    $versionOutput = & $verifiedExecutable -v 2>&1
    # This pinned build returns 201 for a successful -v informational query.
    # Actual analyses still require exit code 0 in CalculixRunner.
    if ($LASTEXITCODE -notin @(0, 201) -or ($versionOutput -join "`n") -notmatch 'Version\s+2\.23(?:\s|$)') {
        throw "CalculiX version execution check failed: $name"
    }
    Write-Output "$name verified: $($versionOutput -join ' ')"
}
$licensePath = Join-Path $solverDirectory 'CalculiX-2.23.0-win-x64/bin/LICENSE.txt'
if ((Get-FileHash -LiteralPath $licensePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'da0eca0fb517ac939d167924c9d4b3f8750a6b7191932ef2cb145acfa624ac7e') {
    throw 'Bundled solver license is missing or changed'
}
