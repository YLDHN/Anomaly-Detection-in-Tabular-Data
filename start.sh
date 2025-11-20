#!/bin/bash

###############################################################################
# 🚀 LANCEUR COMPLET - Projet Détection d'Anomalies
# 
# Ce script lance l'installation complète et une démonstration
# Version: 2.0 - Avec Autoencodeur
###############################################################################

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   🚀 LANCEUR COMPLET - DÉTECTION D'ANOMALIES                           ║
║                                                                        ║
║   Installation + Configuration + Démonstration                        ║
║   Version 2.0 avec Autoencodeur                                       ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Menu principal
echo ""
echo -e "${CYAN}Que souhaitez-vous faire ?${NC}"
echo ""
echo "1) 🔧 Installation Standard (Isolation Forest + One-Class SVM)"
echo "2) 🧠 Installation Complète (+ Autoencodeur avec TensorFlow)"
echo "3) 🎮 Démonstration Rapide (si déjà installé)"
echo "4) 🎯 Démonstration Complète (tous les modèles)"
echo "5) 🧪 Lancer les Tests Unitaires"
echo "6) 📊 Utilisation Personnalisée (CLI interactive)"
echo "7) ℹ️  Afficher l'Aide et la Documentation"
echo "8) ❌ Quitter"
echo ""
read -p "Votre choix (1-8): " choice

case $choice in
    1)
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  INSTALLATION STANDARD${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Installation de l'environnement standard avec:"
        echo "  ✅ Isolation Forest"
        echo "  ✅ One-Class SVM"
        echo ""
        read -p "Continuer? (o/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            chmod +x setup.sh
            ./setup.sh
            
            echo ""
            echo -e "${GREEN}✅ Installation terminée !${NC}"
            echo ""
            echo "Pour utiliser le projet:"
            echo "  source venv/bin/activate"
            echo "  python main.py --synthetic --model isolation_forest"
        fi
        ;;
        
    2)
        echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${MAGENTA}  INSTALLATION COMPLÈTE AVEC AUTOENCODEUR${NC}"
        echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Installation de l'environnement complet avec:"
        echo "  ✅ Isolation Forest"
        echo "  ✅ One-Class SVM"
        echo "  ✅ Autoencodeur (TensorFlow)"
        echo ""
        echo "⚠️  Nécessite Python 3.9-3.12"
        echo ""
        read -p "Continuer? (o/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            chmod +x setup_autoencoder.sh
            ./setup_autoencoder.sh
            
            echo ""
            echo -e "${GREEN}✅ Installation terminée !${NC}"
            echo ""
            echo "Pour utiliser les 3 modèles:"
            echo "  source venv_autoencoder/bin/activate"
            echo "  python main.py --synthetic --model all"
        fi
        ;;
        
    3)
        echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}  DÉMONSTRATION RAPIDE${NC}"
        echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        
        # Vérifier quel environnement est disponible
        if [ -d "venv" ]; then
            echo "Utilisation de l'environnement standard (venv)"
            source venv/bin/activate
            
            echo ""
            echo "🌲 Test avec Isolation Forest..."
            python main.py --synthetic --model isolation_forest --no-visualizations --n-samples 500
            
            echo ""
            echo "🎯 Test avec One-Class SVM..."
            python main.py --synthetic --model onesvm --no-visualizations --n-samples 500
            
        elif [ -d "venv_autoencoder" ]; then
            echo "Utilisation de l'environnement avec TensorFlow"
            source venv_autoencoder/bin/activate
            
            echo ""
            python main.py --synthetic --model all --no-visualizations --n-samples 500
        else
            echo -e "${RED}❌ Aucun environnement trouvé !${NC}"
            echo "Veuillez d'abord installer le projet (option 1 ou 2)"
            exit 1
        fi
        
        echo ""
        echo -e "${GREEN}✅ Démonstration terminée !${NC}"
        ;;
        
    4)
        echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${MAGENTA}  DÉMONSTRATION COMPLÈTE${NC}"
        echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        
        if [ -d "venv_autoencoder" ]; then
            chmod +x demo_all_models.sh
            ./demo_all_models.sh
        else
            echo -e "${YELLOW}⚠️  L'environnement avec Autoencodeur n'est pas installé${NC}"
            echo ""
            read -p "Installer maintenant? (o/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Oo]$ ]]; then
                ./setup_autoencoder.sh
                ./demo_all_models.sh
            else
                echo "Démonstration avec les modèles standards uniquement..."
                source venv/bin/activate
                chmod +x run_demo.sh
                ./run_demo.sh
            fi
        fi
        ;;
        
    5)
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  TESTS UNITAIRES${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        
        if [ -d "venv" ]; then
            source venv/bin/activate
        elif [ -d "venv_autoencoder" ]; then
            source venv_autoencoder/bin/activate
        else
            echo -e "${RED}❌ Aucun environnement trouvé !${NC}"
            exit 1
        fi
        
        echo "Lancement des tests avec pytest..."
        echo ""
        
        if command -v pytest &> /dev/null; then
            pytest tests/ -v --tb=short
        else
            echo "Installation de pytest..."
            pip install pytest
            pytest tests/ -v --tb=short
        fi
        
        echo ""
        echo -e "${GREEN}✅ Tests terminés !${NC}"
        ;;
        
    6)
        echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}  UTILISATION PERSONNALISÉE${NC}"
        echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        
        # Activer l'environnement
        if [ -d "venv_autoencoder" ]; then
            echo "Environnement: venv_autoencoder (3 modèles)"
            source venv_autoencoder/bin/activate
            MODELS="isolation_forest, onesvm, autoencoder, all"
        elif [ -d "venv" ]; then
            echo "Environnement: venv (2 modèles)"
            source venv/bin/activate
            MODELS="isolation_forest, onesvm"
        else
            echo -e "${RED}❌ Aucun environnement trouvé !${NC}"
            exit 1
        fi
        
        echo ""
        echo "Configuration de votre analyse:"
        echo ""
        
        # Type de données
        echo "1. Source des données:"
        echo "   a) Dataset synthétique"
        echo "   b) Fichier CSV"
        read -p "Choix (a/b): " data_choice
        
        if [ "$data_choice" = "b" ]; then
            read -p "Chemin du fichier CSV: " data_file
            DATA_ARG="--data $data_file"
        else
            read -p "Nombre d'échantillons (défaut 1000): " n_samples
            n_samples=${n_samples:-1000}
            DATA_ARG="--synthetic --n-samples $n_samples"
        fi
        
        # Modèle
        echo ""
        echo "2. Modèle à utiliser:"
        echo "   1) Isolation Forest (rapide, précis)"
        echo "   2) One-Class SVM (frontière de décision)"
        if [ "$MODELS" = "isolation_forest, onesvm, autoencoder" ]; then
            echo "   3) Autoencodeur (deep learning)"
        fi
        read -p "Choix (1-2): " model_choice
        
        case $model_choice in
            1) model="isolation_forest" ;;
            2) model="onesvm" ;;
            3) model="autoencoder" ;;
            *) model="isolation_forest" ;;
        esac
        
        # Contamination
        echo ""
        read -p "3. Proportion d'anomalies attendue (ex: 0.1): " contamination
        contamination=${contamination:-0.1}
        
        # Sortie
        echo ""
        read -p "4. Répertoire de sortie (défaut: results): " output
        output=${output:-results}
        
        # Construire la commande
        CMD="python main.py $DATA_ARG --model $model --contamination $contamination --output $output"
        
        echo ""
        echo -e "${BLUE}Commande à exécuter:${NC}"
        echo "$CMD"
        echo ""
        read -p "Lancer? (o/N) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            eval $CMD
            echo ""
            echo -e "${GREEN}✅ Analyse terminée !${NC}"
            echo "Résultats dans: $output/"
        fi
        ;;
        
    7)
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  AIDE ET DOCUMENTATION${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "📚 Documentation disponible:"
        echo ""
        echo "  📄 README.md                  - Vue d'ensemble du projet"
        echo "  📄 QUICKSTART.md              - Guide de démarrage rapide"
        echo "  📄 INSTALL_AUTOENCODER.md     - Installation Autoencodeur"
        echo "  📄 NOTE_FINALE_10_10.md       - Récapitulatif complet"
        echo "  📄 RAPPORT_VALIDATION.md      - Validation du projet"
        echo ""
        echo "🔧 Scripts disponibles:"
        echo ""
        echo "  ./start.sh                    - Ce script (lanceur complet)"
        echo "  ./setup.sh                    - Installation standard"
        echo "  ./setup_autoencoder.sh        - Installation avec TensorFlow"
        echo "  ./demo_all_models.sh          - Démo des 3 modèles"
        echo "  ./run_demo.sh                 - Démo rapide"
        echo ""
        echo "💻 Utilisation en ligne de commande:"
        echo ""
        echo "  python main.py --help         - Aide complète"
        echo "  python main.py --synthetic --model isolation_forest"
        echo "  python main.py --data data/fichier.csv --model all"
        echo ""
        read -p "Ouvrir README.md? (o/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            if command -v bat &> /dev/null; then
                bat README.md
            elif command -v less &> /dev/null; then
                less README.md
            else
                cat README.md
            fi
        fi
        ;;
        
    8)
        echo ""
        echo "👋 Au revoir !"
        exit 0
        ;;
        
    *)
        echo ""
        echo -e "${RED}❌ Choix invalide${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║   ✅ OPÉRATION TERMINÉE                                                ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Pour relancer ce menu: ./start.sh"
echo ""
