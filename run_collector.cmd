@echo off
rem Scheduled entry point: one snapshot, appended log. Proxy bypass is in collector.py.
cd /d "%~dp0"
if not exist data mkdir data
"C:\Users\lenovo\AppData\Local\Programs\Python\Python311\python.exe" collector.py >> data\collector.log 2>&1
