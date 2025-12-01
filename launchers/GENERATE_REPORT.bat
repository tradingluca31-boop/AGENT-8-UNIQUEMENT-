@echo off
REM ================================================================================
REM AGENT 8 - Generer le rapport quotidien
REM ================================================================================
REM Usage: Double-cliquer sur ce fichier pour generer le rapport du jour
REM ================================================================================

echo.
echo ================================================================================
echo AGENT 8 - GENERATION DU RAPPORT QUOTIDIEN
echo ================================================================================
echo.

cd /d "%~dp0\.."

echo Generation du rapport en cours...
echo.

python analysis/modification_tracker.py --action report

echo.
echo ================================================================================
echo RAPPORT GENERE!
echo ================================================================================
echo.
echo Le rapport quotidien a ete genere dans docs/daily_reports/
echo Le fichier ACTUALITE_MISE_A_JOUR.md a ete mis a jour.
echo.
echo ================================================================================
echo.

pause
