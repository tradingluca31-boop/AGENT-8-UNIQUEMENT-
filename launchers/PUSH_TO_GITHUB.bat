@echo off
REM ================================================================================
REM PUSH AGENT 8 TO GITHUB
REM ================================================================================

echo.
echo ================================================================================
echo PUSHING AGENT 8 TO GITHUB
echo ================================================================================
echo.
echo Repository: https://github.com/tradingluca31-boop/AGENT-8-UNIQUEMENT-
echo.
echo IMPORTANT:
echo   - This will push ALL changes to GitHub
echo   - .gitignore will exclude checkpoints and logs
echo.
pause

cd /d "%~dp0\.."

echo.
echo [1/6] Adding all files (respecting .gitignore)...
git add .

echo.
echo [2/6] Creating commit...
git commit -m "Agent 8 - RL Trading Gold (XAUUSD) - Organized structure"

echo.
echo [3/6] Checking remote...
git remote -v

echo.
echo [4/6] Pulling latest changes...
git pull origin main --rebase --allow-unrelated-histories

echo.
echo [5/6] Pushing to GitHub...
git push origin main

echo.
echo [6/6] Done!
echo.
echo ================================================================================
echo PUSH COMPLETE
echo ================================================================================
echo.
echo Your code is now on GitHub:
echo https://github.com/tradingluca31-boop/AGENT-8-UNIQUEMENT-
echo.

pause
