# AVM SYCL GPU Acceleration - One-Line Installer for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/hbliu007/avm-sycl-gpu-acceleration/main/install.ps1 | iex

param(
    [string]$InstallDir = "$env:USERPROFILE\avm-sycl-gpu-acceleration"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║     🎬 AVM SYCL GPU Acceleration Installer                ║" -ForegroundColor Blue
Write-Host "║     Cross-platform GPU acceleration for AV2 codec          ║" -ForegroundColor Blue
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

$RepoUrl = "https://github.com/hbliu007/avm-sycl-gpu-acceleration.git"

# Check for Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git not found. Please install Git first." -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check for CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "❌ CMake not found. Please install CMake first." -ForegroundColor Red
    Write-Host "   Download from: https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

# Check for Intel oneAPI
$OneApiPath = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if (-not (Test-Path $OneApiPath)) {
    Write-Host "⚠ Intel oneAPI not found." -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Would you like to download Intel oneAPI DPC++? [Y/n]"
    if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
        Write-Host "Opening Intel oneAPI download page..." -ForegroundColor Blue
        Start-Process "https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html"
        Write-Host ""
        Write-Host "Please install Intel oneAPI and run this script again." -ForegroundColor Yellow
        exit 0
    } else {
        Write-Host "❌ Cannot proceed without Intel oneAPI." -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Intel oneAPI found" -ForegroundColor Green

# Clone repository
Write-Host ""
Write-Host "Step 2/5: Cloning repository..." -ForegroundColor Yellow

if (Test-Path $InstallDir) {
    Write-Host "Directory $InstallDir exists. Updating..." -ForegroundColor Yellow
    Set-Location $InstallDir
    git pull
} else {
    git clone $RepoUrl $InstallDir
    Set-Location $InstallDir
}

Write-Host "✓ Repository ready at $InstallDir" -ForegroundColor Green

# Build
Write-Host ""
Write-Host "Step 3/5: Building project..." -ForegroundColor Yellow

$BuildDir = Join-Path $InstallDir "build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Set-Location $BuildDir

# Run CMake with oneAPI environment
$BuildCmd = @"
call "$OneApiPath"
cmake .. -G "Visual Studio 17 2022" -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DAVM_ENABLE_SYCL=ON -DAVM_BUILD_TESTS=ON
cmake --build . --config Release
"@

$BuildCmd | cmd.exe

Write-Host "✓ Build completed" -ForegroundColor Green

# Run tests
Write-Host ""
Write-Host "Step 4/5: Running tests..." -ForegroundColor Yellow

ctest -C Release --output-on-failure

Write-Host "✓ All tests passed" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✅ Installation Successful!                   ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Install location: $InstallDir" -ForegroundColor Blue
Write-Host ""
Write-Host "Quick start commands:" -ForegroundColor Blue
Write-Host "  cd $BuildDir"
Write-Host "  .\tests\Release\sycl_context_test.exe"
Write-Host ""
Write-Host "📚 Documentation: https://github.com/hbliu007/avm-sycl-gpu-acceleration" -ForegroundColor Blue
