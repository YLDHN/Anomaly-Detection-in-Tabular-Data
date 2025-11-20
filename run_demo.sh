#!/bin/bash

# Script de démonstration rapide du projet
# Lance tous les exemples pour montrer les capacités du système

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   🎬 DÉMONSTRATION - DÉTECTION D'ANOMALIES 🎬                         ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Activer l'environnement
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ Environnement virtuel activé${NC}\n"
else
    echo -e "${YELLOW}⚠️  Environnement virtuel non trouvé. Exécutez d'abord ./setup.sh${NC}"
    exit 1
fi

# Démonstration 1: Dataset synthétique avec Isolation Forest
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[DEMO 1] Dataset synthétique - Isolation Forest${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

python main.py --synthetic --model isolation_forest --no-visualizations --n-samples 500

echo -e "\n${GREEN}✅ Démonstration 1 terminée${NC}\n"
sleep 2

# Démonstration 2: One-Class SVM
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[DEMO 2] Dataset synthétique - One-Class SVM${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

python main.py --synthetic --model onesvm --no-visualizations --n-samples 500

echo -e "\n${GREEN}✅ Démonstration 2 terminée${NC}\n"
sleep 2

# Démonstration 3: Test complet avec comparaison
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[DEMO 3] Test complet avec comparaison des modèles${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

python test_complet.py

echo -e "\n${GREEN}✅ Démonstration 3 terminée${NC}\n"

# Résumé final
echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ✅ DÉMONSTRATION TERMINÉE ✅                                         ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}📁 Fichiers générés:${NC}"
echo "   - results/anomalies_IF.csv      : Top anomalies Isolation Forest"
echo "   - results/anomalies_SVM.csv     : Top anomalies One-Class SVM"
echo "   - models/isolation_forest_model.pkl"
echo "   - models/onesvm_model.pkl"
echo "   - models/preprocessor.pkl"

echo -e "\n${YELLOW}💡 Pour utiliser vos propres données:${NC}"
echo "   python main.py --data votre_fichier.csv --model isolation_forest --output results/"

echo -e "\n${GREEN}Merci d'avoir essayé le projet! 🎉${NC}\n"
