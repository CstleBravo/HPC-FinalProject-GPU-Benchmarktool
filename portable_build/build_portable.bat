@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "BUILD_PYTHON=%ROOT_DIR%\.venv-portable-build\Scripts\python.exe"

if not exist "%BUILD_PYTHON%" (
    echo Build environment not found at:
    echo   %BUILD_PYTHON%
    echo.
    echo Create the local build environment before running this script.
    exit /b 1
)

pushd "%ROOT_DIR%"

echo Building portable GPUBenchmark folder...

if exist "build" rmdir /s /q "build"
if exist "dist\GPUBenchmark" rmdir /s /q "dist\GPUBenchmark"

"%BUILD_PYTHON%" -m PyInstaller --noconfirm "%SCRIPT_DIR%gpubenchmark_portable.spec"
if errorlevel 1 (
    echo.
    echo Build failed.
    popd
    exit /b 1
)

echo.
echo Portable build created at:
echo   %ROOT_DIR%\dist\GPUBenchmark
echo.
echo Copy the entire GPUBenchmark folder to a USB drive and run GPUBenchmark.exe.

popd
endlocal
