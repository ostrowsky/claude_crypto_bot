@echo off
setlocal

cd /d "%~dp0"
call "%~dp0run_full_pipeline_catboost.bat"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo Pipeline failed with exit code %EXITCODE%.
  pause
  exit /b %EXITCODE%
)

echo.
echo Full CatBoost pipeline completed.
endlocal
