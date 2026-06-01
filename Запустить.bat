@echo off
REM Запуск автопилота от имени администратора (иначе ввод не дойдёт до Roblox).
cd /d "%~dp0"
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Запрашиваю права администратора...
    powershell -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
REM Убиваем прежние копии автопилота, чтобы два бота не дрались за мышь.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*autopilot.py*' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }"
python "%~dp0autopilot.py"
pause
