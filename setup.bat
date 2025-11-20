@echo off
REM Script d'installation pour Windows
REM Détection d'Anomalies dans des Données Tabulaires (DADT)

echo ========================================================================
echo.
echo    INSTALLATION AUTOMATIQUE - DETECTION D'ANOMALIES
echo.
echo ========================================================================
echo.

REM Vérification de Python
echo [ETAPE 1/6] Verification de Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    echo Telechargez Python depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python installe: %PYTHON_VERSION%

REM Création de l'environnement virtuel
echo.
echo [ETAPE 2/6] Creation de l'environnement virtuel
if exist venv (
    echo [WARNING] Environnement virtuel existant detecte
    set /p RECREATE="Voulez-vous le recreer ? (o/N): "
    if /i "%RECREATE%"=="o" (
        rmdir /s /q venv
        python -m venv venv
        echo [OK] Environnement virtuel recree
    ) else (
        echo [OK] Utilisation de l'environnement existant
    )
) else (
    python -m venv venv
    echo [OK] Environnement virtuel cree
)

REM Activation de l'environnement
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel active

REM Mise à jour de pip
echo.
echo [ETAPE 3/6] Mise a jour de pip
python -m pip install --upgrade pip --quiet
echo [OK] pip mis a jour

REM Installation des dépendances
echo.
echo [ETAPE 4/6] Installation des dependances Python
echo Installation des packages essentiels...
pip install numpy pandas scikit-learn matplotlib seaborn plotly scipy jupyter ipykernel joblib tqdm --quiet

echo Installation de TensorFlow (peut prendre quelques minutes)...
pip install tensorflow keras --quiet 2>nul
if %errorlevel% equ 0 (
    echo [OK] TensorFlow et Keras installes - Tous les modeles disponibles
) else (
    echo [WARNING] TensorFlow non installe - Autoencodeur non disponible
    echo Isolation Forest et One-Class SVM fonctionneront normalement
)

echo [OK] Toutes les dependances installees

REM Création des répertoires
echo.
echo [ETAPE 5/6] Creation de la structure du projet
if not exist data mkdir data
if not exist models mkdir models
if not exist results mkdir results
if not exist notebooks mkdir notebooks
type nul > data\.gitkeep
type nul > models\.gitkeep
echo [OK] Structure du projet creee

REM Tests d'installation
echo.
echo [ETAPE 6/6] Verification de l'installation
echo Test des imports...
python -c "import sys; sys.path.insert(0, 'src'); from src.data_loader import create_sample_dataset; from src.preprocessor import preprocess_data; from src.anomaly_detector import IsolationForestDetector; print('[OK] Tous les modules importes avec succes')"

if %errorlevel% neq 0 (
    echo [ERREUR] Echec des tests de validation
    pause
    exit /b 1
)

echo [OK] Tests de validation reussis

REM Résumé
echo.
echo ========================================================================
echo.
echo    INSTALLATION TERMINEE AVEC SUCCES
echo.
echo ========================================================================
echo.
echo Pour commencer:
echo   1. Activer l'environnement: venv\Scripts\activate
echo   2. Lancer le test complet:   python test_complet.py
echo   3. Ou utiliser le CLI:       python main.py --help
echo.
echo Documentation:
echo   - README.md              : Vue d'ensemble
echo   - QUICKSTART.md          : Guide de demarrage
echo   - RAPPORT_VALIDATION.md  : Validation complete
echo.
echo Exemples:
echo   python main.py --synthetic --model isolation_forest
echo   python main.py --data data\fichier.csv --model onesvm --output results\
echo.
echo Bon codage!
echo.
pause
