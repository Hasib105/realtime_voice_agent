@echo off
REM Voice Agent - single browser call mode

echo.
echo ========================================
echo.  VOICE AGENT
echo ========================================
echo.
echo Starting the real-time voice call server...
echo Open: http://127.0.0.1:7860
echo Terminal debug will show mic and agent audio activity.
echo.

uv run python voice_agent_server.py
pause
