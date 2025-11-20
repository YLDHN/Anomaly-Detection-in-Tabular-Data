#!/bin/bash

###############################################################################
# Setup Autoencodeur - Installation TensorFlow avec Python 3.9-3.12
# 
# Ce script installe l'Autoencodeur dans un environnement séparé pour éviter
# les conflits de dépendances avec TensorFlow
###############################################################################

set -e  # Arrêter en cas d'erreur

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║   INSTALLATION AUTOENCODEUR - TensorFlow                               ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Fonction pour afficher les messages
print_step() {
    echo -e "${BLUE}► $1${NC}"
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

# Vérifier Python
print_step "Vérification de la version de Python..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)

echo "Version Python détectée: $PYTHON_VERSION"

# Vérifier si la version est compatible avec TensorFlow
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 12 ]; then
    print_success "Python $PYTHON_VERSION est compatible avec TensorFlow"
else
    print_error "Python $PYTHON_VERSION n'est PAS compatible avec TensorFlow"
    echo ""
    echo "TensorFlow nécessite Python 3.9, 3.10, 3.11 ou 3.12"
    echo ""
    echo "Options:"
    echo "1. Installer Python 3.12 avec Homebrew:"
    echo "   brew install python@3.12"
    echo ""
    echo "2. Utiliser pyenv pour installer Python 3.12:"
    echo "   pyenv install 3.12.0"
    echo "   pyenv local 3.12.0"
    echo ""
    echo "3. Utiliser Isolation Forest et One-Class SVM (déjà installés)"
    exit 1
fi

# Créer un environnement virtuel séparé pour TensorFlow
print_step "Création de l'environnement virtuel 'venv_autoencoder'..."

if [ -d "venv_autoencoder" ]; then
    print_warning "L'environnement venv_autoencoder existe déjà"
    read -p "Voulez-vous le recréer? (o/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        rm -rf venv_autoencoder
        python3 -m venv venv_autoencoder
        print_success "Environnement recréé"
    else
        print_warning "Utilisation de l'environnement existant"
    fi
else
    python3 -m venv venv_autoencoder
    print_success "Environnement virtuel créé"
fi

# Activer l'environnement
print_step "Activation de l'environnement..."
source venv_autoencoder/bin/activate

# Mettre à jour pip
print_step "Mise à jour de pip..."
pip install --upgrade pip setuptools wheel --quiet

# Installer les dépendances de base
print_step "Installation des dépendances de base..."
pip install --quiet \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    scipy>=1.10.0 \
    joblib>=1.2.0

print_success "Dépendances de base installées"

# Installer TensorFlow
print_step "Installation de TensorFlow (cela peut prendre quelques minutes)..."

# Détecter l'architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "Architecture ARM64 (Apple Silicon) détectée"
    # Pour Apple Silicon, utiliser tensorflow-macos
    if [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 11 ]; then
        pip install --quiet tensorflow-macos>=2.13.0
        # Optionnel: tensorflow-metal pour accélération GPU
        pip install --quiet tensorflow-metal
        print_success "TensorFlow pour Apple Silicon installé"
    else
        pip install --quiet tensorflow>=2.13.0
        print_success "TensorFlow installé"
    fi
else
    echo "Architecture x86_64 détectée"
    pip install --quiet tensorflow>=2.13.0
    print_success "TensorFlow installé"
fi

# Installer Keras si nécessaire
pip install --quiet keras>=2.13.0
print_success "Keras installé"

# Vérifier l'installation
print_step "Vérification de l'installation TensorFlow..."

python3 << 'PYTHON_EOF'
import sys
try:
    import tensorflow as tf
    import keras
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ Keras version: {keras.__version__}")
    
    # Vérifier GPU (optionnel)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU détecté: {len(gpus)} dispositif(s)")
    else:
        print("ℹ️  Pas de GPU détecté (CPU sera utilisé)")
    
    sys.exit(0)
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    print_success "TensorFlow fonctionne correctement"
else
    print_error "Problème avec TensorFlow"
    exit 1
fi

# Tester l'Autoencodeur
print_step "Test de l'Autoencodeur..."

python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, 'src')

try:
    from src.data_loader import create_sample_dataset, handle_missing_values
    from src.preprocessor import preprocess_data
    from src.anomaly_detector import AutoencoderDetector
    import numpy as np
    
    print("📊 Création d'un dataset de test...")
    df = create_sample_dataset(n_samples=500, contamination=0.1)
    df = handle_missing_values(df, strategy='auto')
    
    print("⚙️  Prétraitement...")
    X, _ = preprocess_data(df, exclude_columns=['id', 'true_label'])
    
    print("🧠 Entraînement de l'Autoencodeur...")
    detector = AutoencoderDetector(
        encoding_dim=8,
        epochs=10,  # Rapide pour le test
        batch_size=32,
        contamination=0.1,
        verbose=0
    )
    
    predictions = detector.fit_predict(X)
    scores = detector.get_anomaly_scores(X)
    
    n_anomalies = np.sum(predictions == -1)
    print(f"✅ Anomalies détectées: {n_anomalies}/{len(predictions)} ({n_anomalies/len(predictions)*100:.1f}%)")
    
    # Évaluation
    y_true = df['true_label'].values
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    y_true_binary = y_true
    y_pred_binary = np.where(predictions == -1, 1, 0)
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    print(f"✅ Précision: {precision:.3f}")
    print(f"✅ Rappel: {recall:.3f}")
    print(f"✅ F1-Score: {f1:.3f}")
    
    print("\n🎉 L'Autoencodeur fonctionne parfaitement!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    print_success "Test de l'Autoencodeur réussi"
else
    print_error "Échec du test de l'Autoencodeur"
    exit 1
fi

# Instructions finales
echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║   ✅ INSTALLATION RÉUSSIE !                                            ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "📦 L'environnement 'venv_autoencoder' a été créé avec TensorFlow"
echo ""
echo "🚀 UTILISATION:"
echo ""
echo "1. Activer l'environnement Autoencodeur:"
echo "   ${BLUE}source venv_autoencoder/bin/activate${NC}"
echo ""
echo "2. Utiliser l'Autoencodeur:"
echo "   ${BLUE}python main.py --synthetic --model autoencoder${NC}"
echo ""
echo "3. Comparer tous les modèles (IF + OCSVM + Autoencoder):"
echo "   ${BLUE}python main.py --synthetic --model all${NC}"
echo ""
echo "4. Pour revenir à l'environnement normal (sans TensorFlow):"
echo "   ${BLUE}deactivate${NC}"
echo "   ${BLUE}source venv/bin/activate${NC}"
echo ""
echo "💡 NOTE: Utilisez 'venv' pour IF et OCSVM, 'venv_autoencoder' pour les 3 modèles"
echo ""
print_success "Configuration terminée!"
