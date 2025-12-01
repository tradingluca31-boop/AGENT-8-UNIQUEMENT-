@echo off
REM ================================================================================
REM AGENT 8 - SMOKE TEST (10K steps - 5 minutes)
REM ================================================================================
REM
REM Quick test to verify that the agent TRADES (not 0 trades!)
REM
REM WHAT IT TESTS:
REM   - Does the agent open positions?
REM   - Are Trading Action Rewards working (+5.0)?
REM   - Is Demonstration Learning forcing trades?
REM   - Action distribution (SELL/HOLD/BUY)
REM
REM SUCCESS CRITERIA:
REM   - Total trades > 5
REM   - At least 2 actions used (not 100% HOLD)
REM
REM DURATION: ~5 minutes
REM ================================================================================

echo.
echo ================================================================================
echo AGENT 8 - SMOKE TEST (10K steps)
echo ================================================================================
echo.
echo Testing if agent TRADES with the new fixes:
echo   [1] Trading Action Rewards: +5.0 (protected at END)
echo   [2] RSI thresholds: 40/60 (widened from 30/70)
echo   [3] Demonstration Learning: Phase 1 forces smart trades
echo.
echo SUCCESS CRITERIA:
echo   - Total trades ^> 5
echo   - At least 2 actions used
echo.
echo DURATION: ~5 minutes
echo.
echo ================================================================================
echo.
echo Appuyez sur une touche pour lancer le smoke test...
pause > nul

cd /d "%~dp0\.."

echo.
echo ================================================================================
echo LANCEMENT SMOKE TEST (10K steps)...
echo ================================================================================
echo.

python training\train_smoke_test.py

echo.
echo ================================================================================
echo SMOKE TEST TERMINÉ
echo ================================================================================
echo.
echo Vérifiez les résultats:
echo   - outputs\checkpoints_analysis\checkpoint_10000_stats.csv
echo.
echo SI total_trades ^> 5 = SUCCESS! Continue avec full training
echo SI total_trades = 0 = ÉCHEC! Voir docs\DIAGNOSTIC_URGENT.md
echo.

pause
