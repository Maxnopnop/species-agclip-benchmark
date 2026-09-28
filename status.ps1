$ErrorActionPreference = 'Stop'
function Read-LiveTail([string]$filePath) {
    if (Test-Path -LiteralPath $filePath) {
        $stream = [System.IO.File]::Open($filePath,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::ReadWrite)
        try {
            [void]$stream.Seek([Math]::Max(0,$stream.Length-4096),[System.IO.SeekOrigin]::Begin)
            $reader = New-Object System.IO.StreamReader($stream)
            $reader.ReadToEnd() -split "`n" | Select-Object -Last 8
        } finally { $stream.Dispose() }
    }
}
Write-Host 'DATA DOWNLOAD'
Read-LiveTail (Join-Path $PSScriptRoot 'work\image_download.log')
Write-Host 'PILOT WORKFLOW'
$projectStatus = Join-Path $PSScriptRoot 'runs\pilot_multimodal\status.json'
if (Test-Path -LiteralPath $projectStatus) { Get-Content -LiteralPath $projectStatus }
Read-LiveTail (Join-Path $PSScriptRoot 'work\pilot_multimodal.log')
Write-Host 'DOWNLOAD FILE SIZES'
foreach ($projectFile in @('data\archives\train_mini.tar.gz.part','data\archives\val.tar.gz.part','cache\clip\ViT-B-32.pt')) {
    $projectPath = Join-Path $PSScriptRoot $projectFile
    if (Test-Path -LiteralPath $projectPath) {
        $projectStream = [System.IO.File]::Open($projectPath,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::ReadWrite)
        try { Write-Host ($projectFile + ': ' + [Math]::Round($projectStream.Length/1MB,1) + ' MiB') }
        finally { $projectStream.Dispose() }
    }
}
