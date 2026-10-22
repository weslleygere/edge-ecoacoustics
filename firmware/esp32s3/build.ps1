param(
    [ValidateSet('Build', 'Prepare', 'CheckChip')]
    [string]$Action = 'Build',
    [string]$Port = 'COM10',
    [string]$IdfPath = 'C:\esp\v6.0.2\esp-idf',
    [string]$ToolsPath = 'C:\Espressif\tools',
    [string]$StageRoot = 'C:\esp\edge-ecoacoustics-build'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ToolsPath 'python\v6.0.2\venv\Scripts\python.exe'
$env:IDF_PATH = $IdfPath
$env:IDF_TOOLS_PATH = $ToolsPath
$env:IDF_PYTHON_ENV_PATH = Split-Path (Split-Path $python)
$env:ESP_IDF_VERSION = '6.0.2'
$env:IDF_TARGET = 'esp32s3'
$env:ESP_ROM_ELF_DIR = Join-Path $ToolsPath 'esp-rom-elfs\20241011'
$env:PATH = (@(
    (Split-Path $python),
    (Join-Path $ToolsPath 'cmake\4.0.3\bin'),
    (Join-Path $ToolsPath 'ninja\1.12.1'),
    (Join-Path $ToolsPath 'xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin'),
    (Join-Path $ToolsPath 'esp32ulp-elf\2.38_20240113\esp32ulp-elf\bin'),
    $env:PATH
) -join ';')

if ($Action -eq 'CheckChip') {
    & $python -m esptool --chip esp32s3 --port $Port chip-id
    exit $LASTEXITCODE
}

# Keep compiler inputs and outputs at stable ASCII paths. The ESP-IDF
# extension invokes Prepare before its own CMake/Ninja tasks.
if ($StageRoot -match '[^\x21-\x7E]') {
    throw 'StageRoot must be an ASCII path without spaces.'
}
$sourcePath = [System.IO.Path]::GetFullPath((Join-Path $StageRoot 'source'))
$buildPath = [System.IO.Path]::GetFullPath((Join-Path $StageRoot 'build'))
$mainPath = [System.IO.Path]::GetFullPath((Join-Path $sourcePath 'main'))
# Mirroring may delete obsolete files, so validate its absolute destination
# and reject redirects before touching the managed source copy.
if (-not $mainPath.StartsWith("$sourcePath\", [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The managed main directory must be inside the staged source directory.'
}
foreach ($directory in @($StageRoot, $sourcePath, $mainPath)) {
    if ((Test-Path -LiteralPath $directory) -and
        ((Get-Item -LiteralPath $directory).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "The staging destination must not be a junction or symbolic link: $directory"
    }
}
$null = New-Item -ItemType Directory -Force -Path $sourcePath, $buildPath
foreach ($name in @('CMakeLists.txt', 'sdkconfig.defaults')) {
    $inputFile = Join-Path $PSScriptRoot $name
    Copy-Item -LiteralPath $inputFile -Destination $sourcePath -Force
}
& robocopy.exe (Join-Path $PSScriptRoot 'main') $mainPath /MIR /XJ /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -ge 8) { throw 'Could not synchronize the firmware main directory.' }
Write-Host "Firmware source: $sourcePath"
Write-Host "Firmware build:  $buildPath"
if ($Action -eq 'Prepare') { exit 0 }
& cmake -S $sourcePath -B $buildPath -G Ninja -DIDF_TARGET=esp32s3 "-DPYTHON=$python"
if ($LASTEXITCODE -ne 0) { throw 'ESP-IDF configuration failed.' }
& cmake --build $buildPath
if ($LASTEXITCODE -ne 0) { throw 'ESP-IDF build failed.' }
