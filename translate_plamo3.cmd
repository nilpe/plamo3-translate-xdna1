@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set OPENBLAS_NUM_THREADS=4
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set "PLAMO_PY=%~dp0venv\Scripts\python.exe"
if not exist "%PLAMO_PY%" set "PLAMO_PY=%~dp0IRON\.venv\Scripts\python.exe"
"%PLAMO_PY%" "%~dp0IRON\iron\applications\plamo3\translate.py" %*
exit /b %ERRORLEVEL%
