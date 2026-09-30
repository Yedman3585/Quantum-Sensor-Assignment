@echo off
setlocal

cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_prc_qubo_c_article_suite.ps1" -RunConflictOnlyStressCurve

pause
