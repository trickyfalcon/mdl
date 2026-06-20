# Build llama.cpp's llama-quantize.exe (and llama-cli/llama-server) on Windows with NO admin.
#
# Toolchain (all user-scope, no UAC):
#   * CMake + Ninja   -> installed via `uv tool install`
#   * C/C++ compiler  -> w64devkit (portable MinGW-w64 GCC, extracted to C:\tools\w64devkit)
#
# Produces (Ninja single-config layout):  <llamacpp>\build\bin\llama-quantize.exe
# (MSVC would put it under build\bin\Release\ -- mdl's config is updated to match whatever we build.)

param(
    [string]$LlamaDir = 'C:\src\llama.cpp',
    [string]$ToolsDir = 'C:\tools',
    [string]$DevkitDir = 'C:\tools\w64devkit'
)

$ErrorActionPreference = 'Stop'
function Log($m) { Write-Output ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m) }

$uvbin = "$env:USERPROFILE\.local\bin"
$env:PATH = "$uvbin;$env:PATH"

Log "1/5 installing CMake + Ninja via uv tool (user scope)"
uv tool install cmake  | Out-Host
uv tool install ninja  | Out-Host
$env:PATH = "$uvbin;$env:PATH"

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $cmake) { throw "cmake not found after install" }
if (-not $ninja) { throw "ninja not found after install" }
Log ("    cmake = {0} ({1})" -f $cmake, (& cmake --version | Select-Object -First 1))
Log ("    ninja = {0} ({1})" -f $ninja, (& ninja --version))

Log "2/5 resolving latest w64devkit (MinGW GCC) release"
if (-not (Test-Path $ToolsDir)) { New-Item -ItemType Directory -Force $ToolsDir | Out-Null }
$gcc = Join-Path $DevkitDir 'bin\gcc.exe'
if (-not (Test-Path $gcc)) {
    $rel = Invoke-RestMethod 'https://api.github.com/repos/skeeto/w64devkit/releases/latest' -Headers @{ 'User-Agent' = 'mdl-build' }
    # Newer releases ship only a 7-Zip self-extractor (w64devkit-x64-*.7z.exe); extract it silently with -y -o.
    $asset = $rel.assets | Where-Object { $_.name -match '^w64devkit-x64-.*\.7z\.exe$' } | Select-Object -First 1
    if (-not $asset) { $asset = $rel.assets | Where-Object { $_.name -match '^w64devkit-x64-.*\.zip$' } | Select-Object -First 1 }
    if (-not $asset) { throw "could not find an x64 w64devkit asset in $($rel.tag_name)" }
    $pkg = Join-Path $ToolsDir $asset.name
    Log ("    downloading {0} ({1:N0} MB)" -f $asset.name, ($asset.size / 1MB))
    Invoke-WebRequest $asset.browser_download_url -OutFile $pkg
    Log "    extracting to $ToolsDir"
    if ($pkg -match '\.7z\.exe$') {
        # 7-Zip SFX: -y (assume yes), -o<dir> (no space). Archive contains a top-level w64devkit\ folder.
        & $pkg -y "-o$ToolsDir" | Out-Host
    } else {
        Expand-Archive -Path $pkg -DestinationPath $ToolsDir -Force
    }
    Remove-Item $pkg -Force -ErrorAction SilentlyContinue
}
if (-not (Test-Path $gcc)) { throw "gcc not found at $gcc after extract" }
$env:PATH = "$DevkitDir\bin;$env:PATH"
Log ("    gcc = {0}" -f (& gcc --version | Select-Object -First 1))

Log "3/5 configuring llama.cpp (Ninja + GCC, Release, CPU)"
$build = Join-Path $LlamaDir 'build'
& cmake -S $LlamaDir -B $build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ `
    -DLLAMA_CURL=OFF -DGGML_NATIVE=ON | Out-Host
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

Log "4/5 building targets: llama-quantize, llama-cli, llama-server"
& cmake --build $build --config Release -j --target llama-quantize llama-cli llama-server | Out-Host
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

Log "5/5 locating llama-quantize.exe"
$exe = Get-ChildItem -Path $build -Recurse -Filter 'llama-quantize.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) { throw "build finished but llama-quantize.exe not found under $build" }
Log ("BUILT: {0}" -f $exe.FullName)
Write-Output ("LLAMA_QUANTIZE_PATH={0}" -f $exe.FullName)
