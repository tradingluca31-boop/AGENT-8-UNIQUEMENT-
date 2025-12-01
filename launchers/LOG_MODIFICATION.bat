@echo off
REM ================================================================================
REM AGENT 8 - Logger une modification rapidement
REM ================================================================================
REM Usage: Double-cliquer sur ce fichier et suivre les instructions
REM ================================================================================

echo.
echo ================================================================================
echo AGENT 8 - LOGGER UNE MODIFICATION
echo ================================================================================
echo.

cd /d "%~dp0\.."

echo Quelle categorie de modification?
echo.
echo [1] FIX       - Correction de bug
echo [2] FEAT      - Nouvelle fonctionnalite
echo [3] REFACTOR  - Refactoring de code
echo [4] DOCS      - Documentation
echo [5] TEST      - Tests
echo [6] PERF      - Optimisation de performance
echo [7] CONFIG    - Configuration
echo [8] DATA      - Donnees/Features
echo.

set /p choice="Votre choix (1-8): "

if "%choice%"=="1" set category=FIX
if "%choice%"=="2" set category=FEAT
if "%choice%"=="3" set category=REFACTOR
if "%choice%"=="4" set category=DOCS
if "%choice%"=="5" set category=TEST
if "%choice%"=="6" set category=PERF
if "%choice%"=="7" set category=CONFIG
if "%choice%"=="8" set category=DATA

if not defined category (
    echo Choix invalide!
    pause
    exit /b 1
)

echo.
set /p message="Description de la modification: "

if "%message%"=="" (
    echo Description requise!
    pause
    exit /b 1
)

echo.
set /p files="Fichiers modifies (separes par des virgules, optionnel): "

echo.
echo Enregistrement de la modification...
echo.

if "%files%"=="" (
    python analysis/modification_tracker.py --action log --category %category% --message "%message%"
) else (
    python analysis/modification_tracker.py --action log --category %category% --message "%message%" --files "%files%"
)

echo.
echo ================================================================================
echo.

pause
