param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 is required'"
if ($LASTEXITCODE -ne 0) { throw 'Pass -Python with a Python 3.12 executable.' }
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    & $Python -m venv --without-pip (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
& uv --cache-dir (Join-Path $PSScriptRoot '.cache') pip sync --python $ProjectPython (Join-Path $PSScriptRoot 'structural-fea-environment/requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& uv --cache-dir (Join-Path $PSScriptRoot '.cache') pip install --python $ProjectPython --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'MechFEA installation failed' }
& uv --cache-dir (Join-Path $PSScriptRoot '.cache') pip check --python $ProjectPython
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed' }
