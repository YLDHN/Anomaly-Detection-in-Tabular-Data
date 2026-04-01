#!/bin/bash

###############################################################################
# Démonstration Complète - Tous les modèles incluant l'Autoencodeur
# 
# Ce script exécute une démonstration complète des 3 algorithmes
###############################################################################

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║   DÉMONSTRATION COMPLÈTE - 3 ALGORITHMES                               ║"
echo "║   Isolation Forest + One-Class SVM + Autoencodeur                      ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Vérifier si l'environnement Autoencodeur existe
if [ ! -d "venv_autoencoder" ]; then
    echo -e "${YELLOW}⚠️  L'environnement venv_autoencoder n'existe pas${NC}"
    echo ""
    echo "L'Autoencodeur nécessite TensorFlow avec Python 3.9-3.12"
    echo ""
    read -p "Voulez-vous installer l'Autoencodeur maintenant? (o/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo "Lancement de l'installation..."
        ./setup_autoencoder.sh
    else
        echo ""
        echo "Démonstration avec Isolation Forest et One-Class SVM uniquement"
        echo ""
        source venv/bin/activate
        python main.py --synthetic --model isolation_forest --no-visualizations
        echo ""
        python main.py --synthetic --model onesvm --no-visualizations --n-samples 500
        exit 0
    fi
fi

# Activer l'environnement avec TensorFlow
echo -e "${BLUE}► Activation de l'environnement avec TensorFlow...${NC}"
source venv_autoencoder/bin/activate

# Vérifier TensorFlow avec gestion d'erreur
echo -e "${BLUE}► Vérification de TensorFlow...${NC}"
if python3 << 'EOF' 2>/dev/null
import tensorflow as tf
print(f"✅ TensorFlow {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU disponible: {len(gpus)} dispositif(s)")
else:
    print("ℹ️  Mode CPU (pas de GPU)")
EOF
then
    # TensorFlow fonctionne
    USE_AUTOENCODER=1
else
    # TensorFlow ne fonctionne pas
    USE_AUTOENCODER=0
    echo -e "${YELLOW}⚠️  TensorFlow ne fonctionne pas sur cet environnement${NC}"
    echo "Cause possible: Dépendances binaires manquantes sur macOS ARM64"
    echo ""
    echo "Continuation avec Isolation Forest et One-Class SVM..."
    echo ""
    USE_AUTOENCODER=0
fi

echo ""

if [ $USE_AUTOENCODER -eq 1 ]; then
    # Démo 1: Dataset synthétique avec les 3 modèles
    echo -e "${GREEN}"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  DÉMO 1: Comparaison des 3 algorithmes sur données synthétiques"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"

    python main.py --synthetic --model all --contamination 0.1 --no-visualizations --n-samples 1000

    echo ""
    echo -e "${BLUE}Appuyez sur Entrée pour continuer...${NC}"
    read

    # Démo 2: Autoencodeur avec différents hyperparamètres
    echo -e "${GREEN}"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  DÉMO 2: Autoencodeur avec optimisation des hyperparamètres"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"

    echo "Test 1: Autoencodeur avec 50 epochs"
    python main.py --synthetic --model autoencoder --epochs 50 --encoding-dim 8 --no-visualizations --n-samples 500

    echo ""
    echo "Test 2: Autoencodeur avec dimension latente plus grande"
    python main.py --synthetic --model autoencoder --epochs 30 --encoding-dim 16 --no-visualizations --n-samples 500

    echo ""
    echo -e "${BLUE}Appuyez sur Entrée pour continuer...${NC}"
    read
else
    # Mode sans Autoencodeur - Utiliser les modèles qui fonctionnent
    echo -e "${GREEN}"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  DÉMO 1: Isolation Forest et One-Class SVM (synthétique)"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"

    python main.py --synthetic --model isolation_forest --contamination 0.1 --no-visualizations --n-samples 1000

    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  DÉMO 2: One-Class SVM${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"

    python main.py --synthetic --model onesvm --contamination 0.1 --no-visualizations --n-samples 1000

    echo ""
    echo -e "${BLUE}Appuyez sur Entrée pour continuer...${NC}"
    read
fi

# Démo 3: Comparaison sur fichier CSV
echo -e "${GREEN}"
echo "═══════════════════════════════════════════════════════════════════════"
if [ $USE_AUTOENCODER -eq 1 ]; then
    echo "  DÉMO 3: Analyse d'un fichier CSV avec les 3 modèles"
else
    echo "  DÉMO 3: Analyse d'un fichier CSV avec IF et OCSVM"
fi
echo "═══════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Créer un dataset de test si nécessaire
if [ ! -f "data/demo_data.csv" ]; then
    echo "Création d'un dataset de démonstration..."
    python3 << 'EOF'
from src.data_loader import create_sample_dataset
df = create_sample_dataset(n_samples=800, contamination=0.12)
df.to_csv('data/demo_data.csv', index=False)
print("✅ Dataset créé: data/demo_data.csv")
EOF
fi

echo ""
echo "Analyse avec Isolation Forest..."
python main.py --data data/demo_data.csv --model isolation_forest --true-label-column true_label --output results_demo --no-visualizations

echo ""
echo "Analyse avec One-Class SVM..."
python main.py --data data/demo_data.csv --model onesvm --true-label-column true_label --output results_demo --no-visualizations

if [ $USE_AUTOENCODER -eq 1 ]; then
    echo ""
    echo "Analyse avec Autoencodeur..."
    python main.py --data data/demo_data.csv --model autoencoder --true-label-column true_label --output results_demo --no-visualizations --epochs 40
fi

echo ""
echo -e "${BLUE}Appuyez sur Entrée pour voir les résultats...${NC}"
read

# Afficher les résultats
if [ -f "results_demo/model_comparison.csv" ]; then
    echo -e "${GREEN}"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  RÉSULTATS DE LA COMPARAISON"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    cat results_demo/model_comparison.csv | column -t -s,
fi

echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║   ✅ DÉMONSTRATION TERMINÉE !                                          ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "📁 Fichiers générés:"
echo "   - results_demo/anomaly_report_isolation_forest.csv"
echo "   - results_demo/anomaly_report_onesvm.csv"
if [ $USE_AUTOENCODER -eq 1 ]; then
    echo "   - results_demo/anomaly_report_autoencoder.csv"
fi
echo "   - results_demo/model_comparison.csv"
echo ""

if [ $USE_AUTOENCODER -eq 1 ]; then
    echo "🎯 Les 3 algorithmes sont maintenant pleinement fonctionnels!"
else
    echo "🎯 Isolation Forest et One-Class SVM sont fonctionnels!"
    echo ""
    echo "⚠️  Note: TensorFlow/Autoencodeur ne fonctionne pas sur cet environnement"
    echo "   Pour activer l'Autoencodeur sur macOS ARM64:"
    echo "   https://github.com/issues/tensorflow"
fi

echo ""
echo "Pour utiliser le projet dans vos propres scripts:"
echo "  1. Activez un environnement: source venv/bin/activate"
echo "  2. Lancez votre script Python"
echo ""
