#!/bin/bash

# Script d'installation et de lancement automatique du projet
# Détection d'Anomalies dans des Données Tabulaires (DADT)

set -e  # Arrêt en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   🚀 INSTALLATION AUTOMATIQUE - DÉTECTION D'ANOMALIES 🚀              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Fonction pour afficher les étapes
print_step() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[ÉTAPE $1] $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Étape 1: Vérification Python
print_step "1/6" "Vérification de Python"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python installé: $PYTHON_VERSION"
    
    # Vérifier la version pour TensorFlow
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 12 ]; then
        print_success "Version compatible pour tous les modèles (y compris Autoencodeur)"
        INSTALL_TENSORFLOW=true
    else
        print_warning "Python $PYTHON_VERSION détecté - Autoencodeur nécessite Python 3.9-3.12"
        print_warning "Isolation Forest et One-Class SVM fonctionneront normalement"
        INSTALL_TENSORFLOW=false
    fi
else
    print_error "Python 3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Étape 2: Création de l'environnement virtuel
print_step "2/6" "Création de l'environnement virtuel"

if [ -d "venv" ]; then
    print_warning "Environnement virtuel existant détecté"
    read -p "Voulez-vous le recréer ? (o/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        print_success "Environnement virtuel recréé"
    else
        print_success "Utilisation de l'environnement existant"
    fi
else
    python3 -m venv venv
    print_success "Environnement virtuel créé"
fi

# Activation de l'environnement
source venv/bin/activate
print_success "Environnement virtuel activé"

# Étape 3: Mise à jour de pip
print_step "3/6" "Mise à jour de pip"
pip install --upgrade pip --quiet
print_success "pip mis à jour"

# Étape 4: Installation des dépendances
print_step "4/6" "Installation des dépendances Python"

echo "Installation des packages essentiels..."
pip install numpy pandas scikit-learn matplotlib seaborn plotly scipy jupyter ipykernel joblib tqdm --quiet

if [ "$INSTALL_TENSORFLOW" = true ]; then
    echo "Installation de TensorFlow et Keras..."
    pip install tensorflow keras --quiet
    print_success "TensorFlow et Keras installés - Tous les modèles disponibles"
else
    print_warning "TensorFlow non installé - Autoencodeur non disponible"
fi

print_success "Toutes les dépendances installées"

# Étape 5: Création des répertoires nécessaires
print_step "5/6" "Création de la structure du projet"

mkdir -p data models results notebooks
touch data/.gitkeep models/.gitkeep

print_success "Structure du projet créée"

# Étape 6: Tests d'installation
print_step "6/6" "Vérification de l'installation"

echo "Test des imports..."
python3 << 'PYTHON_TEST'
import sys
sys.path.insert(0, 'src')

try:
    from src.data_loader import create_sample_dataset
    from src.preprocessor import preprocess_data
    from src.anomaly_detector import IsolationForestDetector, OneClassSVMDetector
    from src.evaluator import evaluate_predictions
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Test rapide
print("Création d'un dataset de test...")
df = create_sample_dataset(n_samples=100, contamination=0.1)
print(f"✅ Dataset créé: {df.shape}")

print("Test du prétraitement...")
X, preprocessor = preprocess_data(df, exclude_columns=['id', 'true_label'], return_preprocessor=True)
print(f"✅ Données prétraitées: {X.shape}")

print("Test d'Isolation Forest...")
detector = IsolationForestDetector(contamination=0.1)
predictions = detector.fit_predict(X)
print(f"✅ Isolation Forest fonctionne: {sum(predictions == -1)} anomalies détectées")

print("\n🎉 Tous les tests passés avec succès!")
PYTHON_TEST

if [ $? -eq 0 ]; then
    print_success "Tests de validation réussis"
else
    print_error "Échec des tests de validation"
    exit 1
fi

# Affichage du résumé
echo -e "\n${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ✅ INSTALLATION TERMINÉE AVEC SUCCÈS ✅                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}📦 Packages installés:${NC}"
echo "   ✅ numpy, pandas, scikit-learn"
echo "   ✅ matplotlib, seaborn, plotly"
echo "   ✅ scipy, joblib, tqdm"
if [ "$INSTALL_TENSORFLOW" = true ]; then
    echo "   ✅ tensorflow, keras"
fi

echo -e "\n${GREEN}🚀 Pour commencer:${NC}"
echo "   1. Activer l'environnement: source venv/bin/activate"
echo "   2. Lancer le test complet:   python test_complet.py"
echo "   3. Ou utiliser le CLI:       python main.py --help"

echo -e "\n${BLUE}📚 Documentation:${NC}"
echo "   - README.md              : Vue d'ensemble"
echo "   - QUICKSTART.md          : Guide de démarrage"
echo "   - RAPPORT_VALIDATION.md  : Validation complète"
echo "   - EXECUTION_TESTS.md     : Résultats des tests"

echo -e "\n${YELLOW}💡 Exemples d'utilisation:${NC}"
echo "   python main.py --synthetic --model isolation_forest"
echo "   python main.py --data data/fichier.csv --model onesvm --output results/"
echo "   python test_complet.py"

echo -e "\n${GREEN}Bon codage! 🎉${NC}\n"
