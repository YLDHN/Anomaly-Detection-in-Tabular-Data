@echo off
REM ###########################################################################
REM Setup Autoencodeur - Installation TensorFlow avec Python 3.9-3.12
REM 
REM Ce script installe l'Autoencodeur dans un environnement séparé
REM ###########################################################################

echo.
echo ========================================================================
echo.
echo    INSTALLATION AUTOENCODEUR - TensorFlow
echo.
echo ========================================================================
echo.

REM Vérifier Python
echo [ETAPE 1/5] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Telechargez Python 3.9-3.12 depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Version Python detectee: %PYTHON_VERSION%

REM Créer environnement virtuel
echo.
echo [ETAPE 2/5] Creation de l'environnement virtuel...
if exist venv_autoencoder (
    echo L'environnement venv_autoencoder existe deja
    set /p RECREATE="Voulez-vous le recreer? (O/N): "
    if /i "%RECREATE%"=="O" (
        rmdir /s /q venv_autoencoder
        python -m venv venv_autoencoder
        echo Environnement recree
    ) else (
        echo Utilisation de l'environnement existant
    )
) else (
    python -m venv venv_autoencoder
    echo Environnement virtuel cree
)

REM Activer l'environnement
echo.
echo [ETAPE 3/5] Activation de l'environnement...
call venv_autoencoder\Scripts\activate.bat

REM Mettre à jour pip
echo.
echo [ETAPE 4/5] Mise a jour de pip...
python -m pip install --upgrade pip setuptools wheel --quiet

REM Installer les dépendances
echo.
echo [ETAPE 5/5] Installation des dependances...
echo Installation des dependances de base...
pip install --quiet numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0
pip install --quiet matplotlib>=3.7.0 seaborn>=0.12.0 scipy>=1.10.0 joblib>=1.2.0

echo Installation de TensorFlow (peut prendre quelques minutes)...
pip install --quiet tensorflow>=2.13.0 keras>=2.13.0

if errorlevel 1 (
    echo ERREUR lors de l'installation de TensorFlow
    echo Verifiez que vous utilisez Python 3.9-3.12
    pause
    exit /b 1
)

echo.
echo Verification de l'installation...
python -c "import tensorflow as tf; import keras; print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__)"

if errorlevel 1 (
    echo ERREUR: TensorFlow n'est pas correctement installe
    pause
    exit /b 1
)

REM Test de l'Autoencodeur
echo.
echo Test de l'Autoencodeur...
python -c "import sys; sys.path.insert(0, 'src'); from src.anomaly_detector import AutoencoderDetector; print('Autoencodeur OK')"

if errorlevel 1 (
    echo ERREUR lors du test de l'Autoencodeur
    pause
    exit /b 1
)

REM Instructions finales
echo.
echo ========================================================================
echo.
echo    INSTALLATION REUSSIE !
echo.
echo ========================================================================
echo.
echo L'environnement 'venv_autoencoder' a ete cree avec TensorFlow
echo.
echo UTILISATION:
echo.
echo 1. Activer l'environnement Autoencodeur:
echo    venv_autoencoder\Scripts\activate
echo.
echo 2. Utiliser l'Autoencodeur:
echo    python main.py --synthetic --model autoencoder
echo.
echo 3. Comparer tous les modeles:
echo    python main.py --synthetic --model all
echo.
echo 4. Pour revenir a l'environnement normal:
echo    deactivate
echo    venv\Scripts\activate
echo.
echo ========================================================================
echo.
pause
