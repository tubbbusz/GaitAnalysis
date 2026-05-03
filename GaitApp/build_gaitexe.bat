@echo off
title Build GaitApp EXE
setlocal

REM === Go to this script's folder ===
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo -----------------------------------------
    echo Using existing virtual environment
    echo -----------------------------------------
    goto :have_venv
)

set "PY_TAG="
for %%V in (3.13 3.12 3.11 3.10) do (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY_TAG=%%V"
        goto :found_python
    )
)

echo No compatible Python found.
echo Install Python 3.13, 3.12, 3.11, or 3.10 (64-bit) and make sure the py launcher is available.
pause
exit /b 1

:found_python
echo -----------------------------------------
echo Creating Python %PY_TAG% virtual environment
echo -----------------------------------------
py -%PY_TAG% -I -m venv venv
if errorlevel 1 (
    echo Failed to create venv with Python %PY_TAG%.
    echo If you still see import errors here, your Python installation may have a broken global site customization.
    pause
    exit /b 1
)

:have_venv

echo.
echo Activating venv
call venv\Scripts\activate

echo.
echo Checking Python stdlib health
python -c "import argparse,sys; sys.exit(0 if hasattr(argparse, 'HelpFormatter') else 1)"
if errorlevel 1 (
    echo Your Python stdlib argparse module is broken and PyInstaller cannot run.
    echo Reinstall Python 3.13 or restore Lib\argparse.py from a clean copy.
    pause
    exit /b 1
)

echo.
echo Upgrading pip
python -m pip install --upgrade pip

echo.
echo Installing required libraries
pip install mediapipe opencv-python numpy pandas scipy matplotlib pillow pyinstaller reportlab pyglet
if errorlevel 1 (
    echo Library installation failed.
    pause
    exit /b 1
)

echo.
echo -----------------------------------------
echo Patching mediapipe C bindings
echo (fixes 'free not found' in frozen builds)
echo -----------------------------------------
python patch_mediapipe.py
if errorlevel 1 (
    echo Patching failed.
    pause
    exit /b 1
)

REM Clear pycache so PyInstaller picks up the patched source
for /d /r "venv\Lib\site-packages\mediapipe\tasks\python\core" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)
echo Cleared mediapipe core pycache

echo.
echo -----------------------------------------
echo Building EXE with PyInstaller
echo -----------------------------------------
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

pyinstaller --noconfirm --clean Gaitapp.spec
if errorlevel 1 (
    echo PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo -----------------------------------------
echo Build finished.
echo EXE is in: %cd%\dist\Gaitapp.exe
echo.
echo IMPORTANT: copy pose_landmarker_full.task
echo into the same folder as Gaitapp.exe
echo before running it.
echo -----------------------------------------
pause
endlocal
