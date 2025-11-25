@echo off
REM ================================================================================
REM AGENT 8 - INTERVIEW DIAGNOSTIC
REM ================================================================================

echo.
echo ================================================================================
echo AGENT 8 - INTERVIEW DIAGNOSTIC
echo ================================================================================
echo.
echo Cet interview va poser 8 QUESTIONS CRITIQUES pour diagnostiquer:
echo   - Fixes vraiment activés ?
echo   - Demonstration Learning actif ?
echo   - Trading Action Rewards appliqués ?
echo   - Pourquoi HOLD malgré rewards ?
echo   - Entropy correcte ?
echo   - Observations valides ?
echo   - Critic apprend ?
echo   - VRAIE CAUSE ?
echo.
echo Durée: ~3 minutes
echo.
echo Appuyez sur une touche pour commencer l'interview...
pause > nul

cd /d "%~dp0\.."

echo.
echo ================================================================================
echo LANCEMENT INTERVIEW...
echo ================================================================================
echo.

python analysis\interview.py

echo.
echo ================================================================================
echo INTERVIEW TERMINÉE
echo ================================================================================
echo.
echo Le rapport a été sauvegardé dans:
echo   DIAGNOSTIC_REPORT_*.txt
echo.
echo PROCHAINES ÉTAPES:
echo   1. Lire le rapport complet
echo   2. Appliquer les fixes DIRECTEMENT dans environment\trading_env.py
echo   3. Re-tester avec training ou interview
echo.

pause
