@echo off
rem Scheduled entry point: one snapshot, appended log. Set PYTHON to override the interpreter.
cd /d "%~dp0"
if not exist data mkdir data
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" collector.py >> data\collector.log 2>&1
