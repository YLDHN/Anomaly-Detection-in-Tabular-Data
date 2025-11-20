# Guide d'Installation de l'Autoencodeur

## 🎯 Objectif

Ce guide vous aide à installer l'Autoencodeur avec TensorFlow pour compléter votre projet de détection d'anomalies.

## ⚠️ Prérequis

**Important:** TensorFlow nécessite **Python 3.9, 3.10, 3.11 ou 3.12**

Si vous utilisez Python 3.13+, vous devrez installer une version compatible de Python.

## 🚀 Installation Rapide

### Option 1: Script Automatique (Recommandé)

#### macOS / Linux
```bash
./setup_autoencoder.sh
```

#### Windows
```cmd
setup_autoencoder.bat
```

Le script va :
- ✅ Vérifier votre version de Python
- ✅ Créer un environnement virtuel séparé `venv_autoencoder`
- ✅ Installer TensorFlow et toutes les dépendances
- ✅ Tester l'installation
- ✅ Vérifier que l'Autoencodeur fonctionne

### Option 2: Installation Manuelle

1. **Vérifier votre version de Python**
   ```bash
   python3 --version
   ```
   Vous devez avoir Python 3.9, 3.10, 3.11 ou 3.12

2. **Créer un environnement virtuel séparé**
   ```bash
   python3 -m venv venv_autoencoder
   source venv_autoencoder/bin/activate  # macOS/Linux
   # ou
   venv_autoencoder\Scripts\activate  # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install --upgrade pip
   pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib
   ```

4. **Installer TensorFlow**
   
   **Pour Apple Silicon (M1/M2/M3):**
   ```bash
   pip install tensorflow-macos tensorflow-metal
   ```
   
   **Pour Intel/AMD (macOS, Linux, Windows):**
   ```bash
   pip install tensorflow keras
   ```

5. **Vérifier l'installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
   ```

## 📝 Si vous n'avez pas Python 3.9-3.12

### Option A: Installer Python 3.12 avec Homebrew (macOS)
```bash
brew install python@3.12
```

Puis créer l'environnement avec cette version:
```bash
python3.12 -m venv venv_autoencoder
```

### Option B: Utiliser pyenv
```bash
# Installer pyenv
curl https://pyenv.run | bash

# Installer Python 3.12
pyenv install 3.12.0

# Utiliser Python 3.12 pour ce projet
cd DADT
pyenv local 3.12.0

# Créer l'environnement
python -m venv venv_autoencoder
```

### Option C: Continuer sans l'Autoencodeur

Vous pouvez utiliser le projet avec seulement **Isolation Forest** et **One-Class SVM** qui fonctionnent avec toutes les versions de Python:

```bash
source venv/bin/activate  # Environnement normal
python main.py --synthetic --model isolation_forest
python main.py --synthetic --model onesvm
```

## 🎮 Utilisation

### Activer l'environnement Autoencodeur
```bash
source venv_autoencoder/bin/activate  # macOS/Linux
# ou
venv_autoencoder\Scripts\activate  # Windows
```

### Utiliser l'Autoencodeur seul
```bash
python main.py --synthetic --model autoencoder
```

### Comparer les 3 modèles
```bash
python main.py --synthetic --model all
```

### Avec vos propres données
```bash
python main.py --data data/fichier.csv --model autoencoder --output results/
```

### Désactiver l'environnement
```bash
deactivate
```

## 🔄 Basculer entre les environnements

Le projet dispose de **2 environnements virtuels** :

1. **`venv`** - Environnement standard (Isolation Forest + One-Class SVM)
   ```bash
   source venv/bin/activate
   python main.py --synthetic --model isolation_forest
   ```

2. **`venv_autoencoder`** - Environnement avec TensorFlow (tous les modèles)
   ```bash
   source venv_autoencoder/bin/activate
   python main.py --synthetic --model all
   ```

## 📊 Tester l'installation

```bash
source venv_autoencoder/bin/activate

python << 'EOF'
from src.data_loader import create_sample_dataset, handle_missing_values
from src.preprocessor import preprocess_data
from src.anomaly_detector import AutoencoderDetector

# Créer des données de test
df = create_sample_dataset(n_samples=500)
df = handle_missing_values(df)
X, _ = preprocess_data(df, exclude_columns=['id', 'true_label'])

# Tester l'Autoencodeur
detector = AutoencoderDetector(epochs=20, verbose=1)
predictions = detector.fit_predict(X)

print(f"✅ Autoencodeur fonctionne! {sum(predictions == -1)} anomalies détectées")
EOF
```

## 🐛 Dépannage

### Erreur: "No module named 'tensorflow'"
- Vérifiez que vous avez activé `venv_autoencoder`
- Réinstallez TensorFlow: `pip install tensorflow`

### Erreur: "No compatible version of tensorflow"
- Vérifiez votre version de Python: `python --version`
- Installez Python 3.9-3.12

### Apple Silicon: Erreur avec tensorflow
- Utilisez `tensorflow-macos` au lieu de `tensorflow`
- Installez aussi `tensorflow-metal` pour l'accélération GPU

### GPU non détecté
- C'est normal, le CPU sera utilisé
- Pour utiliser le GPU, installez CUDA (NVIDIA) ou tensorflow-metal (Apple)

## 📚 Ressources

- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [Keras Documentation](https://keras.io/)
- [Python Downloads](https://www.python.org/downloads/)

## ✅ Vérification finale

Une fois installé, vous devriez pouvoir exécuter:

```bash
source venv_autoencoder/bin/activate
python main.py --synthetic --model all --no-visualizations
```

Et voir les 3 modèles s'exécuter:
- ✅ Isolation Forest
- ✅ One-Class SVM  
- ✅ Autoencodeur

## 🎉 Félicitations !

Vous avez maintenant accès aux **3 algorithmes de détection d'anomalies** !

Le projet est maintenant **100% complet** avec:
- Isolation Forest (sklearn)
- One-Class SVM (sklearn)
- Autoencodeur (TensorFlow/Keras)
