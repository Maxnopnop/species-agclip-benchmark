param(
    [ValidateSet('doctor','prepare','download','train','evaluate','predict','smoke')]
    [string]$Action = 'doctor',
    [string]$ImagePath = ''
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$projectMain = Join-Path $PSScriptRoot 'run.py'
$projectData = Join-Path $PSScriptRoot 'data_tools.py'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Project Python is missing. See README.md for environment setup.' }
Set-Location -LiteralPath $PSScriptRoot
if ($Action -in @('prepare', 'download')) {
    & $projectPython -u $projectData $Action
} elseif ($Action -eq 'train') {
    & $projectPython -u $projectMain train --device cuda --epochs 20 --batch-size 16 --workers 2
} elseif ($Action -in @('evaluate','predict')) {
    $projectCandidates = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'runs') -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $configPath = Join-Path $_.FullName 'config.json'
            $checkpointPath = Join-Path $_.FullName 'best.pt'
            if ((Test-Path -LiteralPath $configPath) -and (Test-Path -LiteralPath $checkpointPath)) {
                $projectConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
                -not $projectConfig.synthetic
            } else { $false }
        } | Sort-Object LastWriteTime -Descending)
    if ($projectCandidates.Count -eq 0) { throw 'No real-data model found. Download data and run train_baseline.cmd first. Synthetic smoke models are excluded.' }
    $projectCheckpoint = Join-Path $projectCandidates[0].FullName 'best.pt'
    Write-Host "Using checkpoint: $projectCheckpoint"
    if ($Action -eq 'evaluate') {
        & $projectPython -u $projectMain evaluate --checkpoint $projectCheckpoint --split test --device cuda
    } else {
        if (-not $ImagePath) { $ImagePath = Read-Host 'Enter image path (or drag an image onto predict_image.cmd)' }
        $ImagePath = $ImagePath.Trim('"')
        if (-not (Test-Path -LiteralPath $ImagePath -PathType Leaf)) { throw "Image not found: $ImagePath" }
        & $projectPython -u $projectMain predict --checkpoint $projectCheckpoint --image $ImagePath --device cuda
    }
} else {
    & $projectPython -u $projectMain $Action
}
exit $LASTEXITCODE
