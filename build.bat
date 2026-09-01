@echo off
setlocal

set "PYTHON=venv\Scripts\python.exe"
set "APP_NAME=GeminiTranscribe"
set "ENTRY=gui.py"
set "ICON=app.ico"
set "ENV_FILE=.env"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found: %PYTHON%
    exit /b 1
)

if not exist "%ENTRY%" (
    echo [ERROR] Entry file not found: %ENTRY%
    exit /b 1
)

if not exist "%ICON%" (
    echo [ERROR] Icon file not found: %ICON%
    exit /b 1
)

if not exist "%ENV_FILE%" (
    echo [ERROR] Environment file not found: %ENV_FILE%
    echo [INFO] Create .env with GEMINI_API_KEY=your_key before building.
    exit /b 1
)

echo [INFO] Building %APP_NAME%...
"%PYTHON%" -m PyInstaller --name "%APP_NAME%" --onefile --windowed --clean --noconfirm --paths . --icon "%ICON%" --collect-all qtawesome --collect-all google.genai --hidden-import PySide6.QtSvg "%ENTRY%"

set "BUILD_EXIT=%ERRORLEVEL%"
if not "%BUILD_EXIT%"=="0" (
    echo [ERROR] Build failed with code %BUILD_EXIT%
    exit /b %BUILD_EXIT%
)

copy /Y "%ENV_FILE%" "dist\%ENV_FILE%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy %ENV_FILE% beside the EXE.
    exit /b 1
)

echo [SUCCESS] Created dist\%APP_NAME%.exe
echo [INFO] Copied %ENV_FILE% beside the EXE.
exit /b 0
