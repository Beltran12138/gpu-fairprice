@echo off
rem One-off scheduled run of the pre-registered 24h test. Output: data\verdict_<date>.txt
cd /d "%~dp0\.."
if "%PYTHON%"=="" set PYTHON=python
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STAMP=%%d
"%PYTHON%" analysis\first_pass.py data\vast.sqlite 24 sustained 2 > data\verdict_%STAMP%.txt 2>&1
"%PYTHON%" analysis\first_pass.py data\vast.sqlite 24 any 0 >> data\verdict_%STAMP%.txt 2>&1
