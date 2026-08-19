# Pack the Overleaf drop zip: compile inputs only.
# Uses forward-slash zip entries — Compress-Archive writes backslashes
# that Overleaf (Linux) will not unpack as folders.
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stamp = Get-Date -Format "yyyy-MM-dd"
$dest = Join-Path $root "overleaf-paper-$stamp.zip"
if (Test-Path $dest) { Remove-Item $dest -Force }

$files = @()
$files += Get-Item (Join-Path $root "main.tex")
$files += Get-Item (Join-Path $root "references.bib")
$files += Get-Item (Join-Path $root "pending_macros.tex")
$files += Get-ChildItem (Join-Path $root "sections\*.tex") |
    Where-Object { $_.Name -notlike "_*" }
$files += Get-ChildItem (Join-Path $root "generated\*.tex")
$files += Get-ChildItem (Join-Path $root "figures\*") -File |
    Where-Object { $_.Extension -match '\.(png|pdf|jpg)$' -and $_.Name -notlike "crash_*" -and $_.Name -notlike "make_*" }

$zip = [IO.Compression.ZipFile]::Open($dest, [IO.Compression.ZipArchiveMode]::Create)
foreach ($f in $files) {
    $rel = $f.FullName.Substring($root.Length).TrimStart('\', '/')
    $entry = ($rel -replace '\\', '/')
    [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $zip, $f.FullName, $entry, [IO.Compression.CompressionLevel]::Optimal)
}
$zip.Dispose()
Write-Output $dest
