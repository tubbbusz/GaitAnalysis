@echo off
title Build GaitApp EXE
setlocal

REM === Go to this script's folder ===
cd /d "%~dp0"

echo -----------------------------------------
echo Creating Python 3.13 virtual environment
echo -----------------------------------------
py -3.13 -m venv venv
if errorlevel 1 (
    echo Failed to create venv. Check that Python 3.13 is installed.
    pause
    exit /b 1
)

echo.
echo Activating venv
call venv\Scripts\activate

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
