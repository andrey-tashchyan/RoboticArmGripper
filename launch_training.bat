@echo off
REM YOLOv8 Raspberry Detection - Script de lancement pour Windows
REM Usage: launch_training.bat [fast|full]

setlocal EnableDelayedExpansion

echo ================================================================================
echo   YOLOv8 Raspberry Detection - Training with Negatives
echo ================================================================================
echo.

REM Recuperer l'argument (fast ou full)
set MODE=%1
if "%MODE%"=="" set MODE=full

if not "%MODE%"=="fast" if not "%MODE%"=="full" (
    echo [ERREUR] Mode invalide '%MODE%'
    echo Usage: %0 [fast^|full]
    echo.
    echo Modes disponibles:
    echo   fast - Entrainement rapide ^(30 epochs, ~1-2h^)
    echo   full - Entrainement complet ^(120 epochs, ~5-8h^)
    exit /b 1
)

REM Afficher le mode selectionne
if "%MODE%"=="fast" (
    echo [MODE RAPIDE]
    echo   - 30 epochs
    echo   - Patience: 10 epochs
    echo   - Duree estimee: 1-2 heures
    echo   - Ideal pour: tests et validation
) else (
    echo [MODE COMPLET]
    echo   - 120 epochs
    echo   - Patience: 30 epochs
    echo   - Duree estimee: 5-8 heures
    echo   - Ideal pour: production finale
)

echo.
echo --------------------------------------------------------------------------------
echo.

REM Verifier que l'environnement virtuel existe
if not exist ".venv311" (
    if not exist "venv" (
        echo [ERREUR] Environnement virtuel introuvable
        echo Creez-en un avec: python -m venv venv
        exit /b 1
    )
    set VENV_DIR=venv
) else (
    set VENV_DIR=.venv311
)

REM Verifier que le dataset existe
if not exist "data\raspberries\images\train" (
    echo [ATTENTION] Dataset non prepare. Lancement de la preparation...
    echo.
    call %VENV_DIR%\Scripts\activate.bat
    python prepare_dataset_with_negatives.py
    echo.
    echo [OK] Dataset prepare avec succes
    echo.
)

REM Demander confirmation
set /p CONFIRM="Pret a demarrer l'entrainement en mode %MODE%. Continuer? (o/n) "
if /i not "%CONFIRM%"=="o" if /i not "%CONFIRM%"=="y" (
    echo [ANNULE] Entrainement annule
    exit /b 0
)

echo.
echo [DEMARRAGE] Lancement de l'entrainement...
echo.

REM Activer l'environnement et lancer l'entrainement
call %VENV_DIR%\Scripts\activate.bat
python train_with_negatives.py --mode %MODE%

REM Resume final
echo.
echo ================================================================================
echo   Entrainement termine!
echo ================================================================================
echo.

set OUTPUT_DIR=runs\raspberry_detect\train_%MODE%
echo Resultats sauvegardes dans: %OUTPUT_DIR%\
echo.
echo Fichiers generes:
echo   - %OUTPUT_DIR%\weights\best.pt     (meilleur modele)
echo   - %OUTPUT_DIR%\weights\last.pt     (dernier epoch)
echo   - %OUTPUT_DIR%\results.png         (courbes d'entrainement)
echo   - %OUTPUT_DIR%\confusion_matrix.png
echo.
echo Pour utiliser le modele:
echo   yolo predict model=%OUTPUT_DIR%\weights\best.pt source=image.jpg
echo.

endlocal
