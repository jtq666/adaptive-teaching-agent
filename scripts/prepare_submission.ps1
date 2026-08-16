param([switch]$KeepDemoData)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$forbidden = @(
    (Join-Path $projectRoot ".env"),
    (Join-Path $projectRoot ".coverage"),
    (Join-Path $projectRoot ".pytest_cache"),
    (Join-Path $projectRoot ".pytest-tmp")
)

Write-Host "Submission audit for $projectRoot"
foreach ($path in $forbidden) {
    if (Test-Path -LiteralPath $path) {
        Write-Warning "Exclude from submission: $path"
    }
}

if (-not $KeepDemoData) {
    Write-Host "Runtime archives are excluded by .gitignore; no user data was deleted."
}

python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
Write-Host "Audit passed. Copy only source/docs plus one manually selected demo bundle."
