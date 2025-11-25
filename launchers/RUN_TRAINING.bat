@echo off
REM ================================================================================
REM AGENT 8 - TRAINING LAUNCHER (500K VALIDATION)
REM ================================================================================

echo.
echo ================================================================================
echo AGENT 8 - MODE COLLAPSE FIX TRAINING
echo ================================================================================
echo.
echo 7 NUCLEAR FIXES + 3 CRITICAL UPDATES
echo ENTROPY SCHEDULE: 0.40 → 0.20 (linear decay)
echo DURATION: ~40 minutes
echo CHECKPOINTS: Every 50K steps (10 total)
echo.
echo ================================================================================
echo.
echo Appuyez sur une touche pour commencer le training...
pause > nul

cd /d "%~dp0\.."

echo.
echo ================================================================================
echo LANCEMENT TRAINING...
echo ================================================================================
echo.

python training\train.py

echo.
echo ================================================================================
echo TRAINING TERMINÉ
echo ================================================================================
echo.
echo Résultats disponibles dans:
echo   - outputs\checkpoints_analysis\checkpoint_*.csv
echo   - outputs\checkpoints\
echo.
echo PROCHAINES ÉTAPES:
echo   1. Analyser les CSVs pour vérifier action distribution
echo   2. Vérifier si mode collapse résolu
echo   3. Si succès: Full training 1M+ steps
echo   4. Si échec: Modifier DIRECTEMENT environment\trading_env.py
echo.

pause
