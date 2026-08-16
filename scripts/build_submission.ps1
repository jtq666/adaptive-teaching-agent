param(
    [string]$DemoSessionPath = "",
    [string]$ReleaseRoot = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
if (-not $ReleaseRoot) {
    $ReleaseRoot = Join-Path $projectRoot "release"
}
$releaseRootPath = [System.IO.Path]::GetFullPath($ReleaseRoot)
if (-not $releaseRootPath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ReleaseRoot must stay inside the exam2 project directory"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$packageRoot = Join-Path $releaseRootPath "exam2-submission-$stamp"
$zipPath = "$packageRoot.zip"
if ((Test-Path -LiteralPath $packageRoot) -or (Test-Path -LiteralPath $zipPath)) {
    throw "Target already exists: $packageRoot"
}
New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null

function Copy-CleanTree([string]$relativeSource) {
    $source = Join-Path $projectRoot $relativeSource
    if (-not (Test-Path -LiteralPath $source)) { return }
    Get-ChildItem -LiteralPath $source -Recurse -File -Force |
        Where-Object {
            $_.FullName -notmatch '[\\/](__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)[\\/]' -and
            $_.Extension -ne '.pyc' -and $_.Name -ne '.env'
        } |
        ForEach-Object {
            $relative = $_.FullName.Substring($projectRoot.Length).TrimStart([char[]]('\', '/'))
            $target = Join-Path $packageRoot $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $target
        }
}

foreach ($directory in @("src", "app_pages", "config", "data", "scripts", "tests", "docs", ".github", ".streamlit")) {
    Copy-CleanTree $directory
}

$rootFiles = @(
    ".env.example", ".gitignore", "streamlit_app.py", "pyproject.toml", "pytest.ini",
    "requirements.txt", "requirements-lock.txt", "requirements-e2e.txt", "README.md"
)
foreach ($name in $rootFiles) {
    $source = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $packageRoot $name)
    }
}

if (-not $DemoSessionPath) {
    $DemoSessionPath = (
        Get-ChildItem -LiteralPath (Join-Path $projectRoot "output\sessions") -File -Filter "*.json" |
            Where-Object Name -ne "index.json" |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    ).FullName
}
$resolvedDemo = (Resolve-Path -LiteralPath $DemoSessionPath).Path
$sessionTarget = Join-Path $packageRoot "output\sessions\demo_session.json"
New-Item -ItemType Directory -Path (Split-Path -Parent $sessionTarget) -Force | Out-Null
Copy-Item -LiteralPath $resolvedDemo -Destination $sessionTarget

$onlineAcceptance = Join-Path $projectRoot ".e2e-runtime\online-api-multicase.json"
if (Test-Path -LiteralPath $onlineAcceptance) {
    $acceptanceTarget = Join-Path $packageRoot "output\acceptance\online-api-multicase.json"
    New-Item -ItemType Directory -Path (Split-Path -Parent $acceptanceTarget) -Force | Out-Null
    Copy-Item -LiteralPath $onlineAcceptance -Destination $acceptanceTarget
}
$contextAcceptance = Join-Path $projectRoot ".e2e-runtime\online-context-acceptance.json"
if (Test-Path -LiteralPath $contextAcceptance) {
    $contextTarget = Join-Path $packageRoot "output\acceptance\online-context-acceptance.json"
    New-Item -ItemType Directory -Path (Split-Path -Parent $contextTarget) -Force | Out-Null
    Copy-Item -LiteralPath $contextAcceptance -Destination $contextTarget
}
$adaptiveAcceptance = Join-Path $projectRoot "output\evaluations\online_adaptive_teaching_acceptance.json"
if (Test-Path -LiteralPath $adaptiveAcceptance) {
    $adaptiveTarget = Join-Path $packageRoot "output\evaluations\online_adaptive_teaching_acceptance.json"
    New-Item -ItemType Directory -Path (Split-Path -Parent $adaptiveTarget) -Force | Out-Null
    Copy-Item -LiteralPath $adaptiveAcceptance -Destination $adaptiveTarget
}
foreach ($demoName in @("demo_physics_run.json", "demo_derivative_run.json")) {
    $demoSource = Join-Path $projectRoot "output\evaluations\$demoName"
    if (Test-Path -LiteralPath $demoSource) {
        $demoTarget = Join-Path $packageRoot "output\evaluations\$demoName"
        New-Item -ItemType Directory -Path (Split-Path -Parent $demoTarget) -Force | Out-Null
        Copy-Item -LiteralPath $demoSource -Destination $demoTarget
    }
}

$evaluationSource = Join-Path $projectRoot "output\evaluations"
$latestEvaluation = $null
$evaluationData = $null
foreach ($candidate in @(
    Get-ChildItem -LiteralPath $evaluationSource -File -Filter "evaluation_*.json" |
        Sort-Object LastWriteTime -Descending
)) {
    try {
        $candidateData = Get-Content -LiteralPath $candidate.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        continue
    }
    if ($candidateData.evaluation_protocol.mode -eq "full" -and $candidateData.case_results.Count -eq 810) {
        $latestEvaluation = $candidate
        $evaluationData = $candidateData
        break
    }
}
if (-not $latestEvaluation) { throw "No complete 810-cell evaluation bundle is available" }
$token = $latestEvaluation.BaseName.Substring("evaluation_".Length)
$evaluationTarget = Join-Path $packageRoot "output\evaluations"
New-Item -ItemType Directory -Path $evaluationTarget -Force | Out-Null
foreach ($name in @(
    "evaluation_$token.json", "evaluation_$token.csv", "evaluation_$token.md",
    "annotation_blind_$token.csv", "annotation_key_$token.csv"
)) {
    $source = Join-Path $evaluationSource $name
    if (-not (Test-Path -LiteralPath $source)) { throw "Evaluation bundle is missing: $name" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $evaluationTarget $name)
}

$forbidden = Get-ChildItem -LiteralPath $packageRoot -Recurse -File -Force |
    Where-Object { $_.Name -eq '.env' -or $_.Extension -eq '.pyc' -or $_.FullName -match '__pycache__' }
if ($forbidden) { throw "Submission contains forbidden files: $($forbidden.FullName -join ', ')" }
if ((Get-ChildItem -LiteralPath (Join-Path $packageRoot "output\sessions") -File -Filter "*.json").Count -ne 1) {
    throw "Submission must contain exactly one demo session"
}
if ((Get-ChildItem -LiteralPath $evaluationTarget -File -Filter "evaluation_*.json").Count -ne 1) {
    throw "Submission must contain exactly one evaluation suite"
}

Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$shaPath = Join-Path $releaseRootPath "$($packageRoot | Split-Path -Leaf)-SHA256.txt"
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath $shaPath -Value "$hash  $(Split-Path -Leaf $zipPath)" -Encoding UTF8
Write-Host "Submission package created: $packageRoot"
Write-Host "ZIP created: $zipPath"
Write-Host "SHA-256 manifest created: $shaPath"
