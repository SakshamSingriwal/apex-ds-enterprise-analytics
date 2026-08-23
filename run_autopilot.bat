@echo off
REM ── DS Autopilot — fully offline launcher ─────────────────────────────
REM Double-click this file. Runs on your machine only: no cloud, no upload,
REM data never leaves this PC. Opens http://localhost:8501 in your browser.
cd /d "%~dp0"
echo Starting DS Autopilot (offline, local)...
python -m streamlit run autopilot_ui.py --server.headless false --browser.gatherUsageStats false --server.address 127.0.0.1
pause
