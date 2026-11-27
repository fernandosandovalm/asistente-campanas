@echo off
chcp 65001 >nul
title Asistente de campañas - Ben ^& Frank
REM Doble clic para abrir la interfaz. Debe estar en la raíz del proyecto (junto a app.py).

cd /d "%~dp0"

REM 1) Entorno virtual (vive en notebooks\.venv)
if not exist "notebooks\.venv\Scripts\activate.bat" (
    echo [ERROR] No encuentro el entorno virtual en notebooks\.venv
    echo Crea el entorno e instala requirements.txt antes de usar este archivo.
    pause
    exit /b 1
)
call "notebooks\.venv\Scripts\activate.bat"

REM 2) Ollama: si no responde, se inicia en segundo plano
ollama list >nul 2>&1
if errorlevel 1 (
    echo Iniciando Ollama...
    start "" /min ollama serve
    timeout /t 5 /nobreak >nul
)

REM 3) Interfaz (se abre sola en el navegador: http://localhost:8501)
echo.
echo  Asistente de campañas Ben ^& Frank
echo  Abriendo http://localhost:8501 ...
echo  Para cerrar: cierra esta ventana o presiona Ctrl+C.
echo.
python -m streamlit run app.py --server.headless false

pause
