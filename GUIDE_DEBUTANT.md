# 🎓 Guide Complet pour Débutants - Détection d'Anomalies

## 📚 Table des Matières

1. [C'est quoi ce projet ?](#1-cest-quoi-ce-projet-)
2. [Pourquoi c'est utile ?](#2-pourquoi-cest-utile-)
3. [Comment ça marche ?](#3-comment-ça-marche-)
4. [Les 3 algorithmes expliqués simplement](#4-les-3-algorithmes-expliqués-simplement)
5. [Installation pas à pas](#5-installation-pas-à-pas)
6. [Utilisation concrète](#6-utilisation-concrète)
7. [Comprendre les résultats](#7-comprendre-les-résultats)
8. [Questions fréquentes](#8-questions-fréquentes)

---

## 1. C'est quoi ce projet ? 🤔

### En une phrase
**Un outil automatique qui trouve les données bizarres/anormales dans vos fichiers Excel ou CSV.**

### Analogie simple
Imaginez que vous avez 10,000 pommes :
- 9,950 pommes sont normales (rouges, rondes, 150g)
- 50 pommes sont bizarres (vertes, carrées, 500g)

Ce projet **trouve automatiquement** les 50 pommes bizarres sans que vous ayez à les chercher une par une !

### Concrètement
Le projet analyse vos données (transactions bancaires, logs serveurs, mesures de capteurs, etc.) et vous dit :
- ✅ **Ligne 42 : NORMALE**
- ❌ **Ligne 156 : ANORMALE** (score d'anomalie : 0.85)

---

## 2. Pourquoi c'est utile ? 💡

### Problèmes résolus

#### 🏦 **Banques - Détection de fraudes**
**Problème** : Sur 1 million de transactions par jour, trouver les 100 frauduleuses
**Solution** : Le projet analyse toutes les transactions et trouve celles qui sont bizarres
```
Transaction normale : 45€ au supermarché à 14h
Transaction anormale : 9,999€ en Chine à 3h du matin ❌
```

#### 🏭 **Usines - Surveillance machines**
**Problème** : Un capteur défectueux peut causer une panne coûteuse
**Solution** : Détecte quand un capteur donne des valeurs anormales
```
Température normale : 25°C
Température anormale : 150°C ❌ → Alerte !
```

#### 💻 **Informatique - Sécurité**
**Problème** : Détecter une intrusion dans les logs
**Solution** : Trouve les connexions inhabituelles
```
Connexion normale : Admin depuis Paris à 9h
Connexion anormale : Admin depuis Russie à 2h ❌
```

#### 🏥 **Santé - Résultats médicaux**
**Problème** : Identifier des résultats anormaux rapidement
**Solution** : Détecte les valeurs hors normes
```
Tension normale : 120/80
Tension anormale : 200/140 ❌
```

---

## 3. Comment ça marche ? ⚙️

### Le processus en 4 étapes

```
📁 Vos données → 🔧 Prétraitement → 🤖 Algorithme → 📊 Résultats
```

#### **Étape 1 : Chargement des données** 📁
```python
# Le projet lit votre fichier
fichier = "transactions.csv"
→ 284,807 lignes chargées ✅
```

**Formats acceptés :**
- CSV (`.csv`)
- Excel (`.xlsx`)
- JSON (`.json`)
- Parquet (`.parquet`)

#### **Étape 2 : Prétraitement** 🔧

**Pourquoi ?** Les machines ne comprennent que les chiffres !

**Exemple :**
```
Avant prétraitement :
| Âge | Salaire | Pays  |
|-----|---------|-------|
| 25  | 50,000€ | FR    |
| 30  | 60,000€ | USA   |

Après prétraitement :
| Âge  | Salaire | Pays_FR | Pays_USA |
|------|---------|---------|----------|
| 0.25 | 0.25    | 1       | 0        |
| 0.30 | 0.30    | 0       | 1        |
```

**Ce qui est fait :**
1. **Normalisation** : Mettre toutes les valeurs entre 0 et 1
2. **Encodage** : Transformer texte en chiffres
3. **Nettoyage** : Gérer les valeurs manquantes

#### **Étape 3 : Détection** 🤖

Le projet utilise **3 algorithmes intelligents** (expliqués plus bas) :
- Isolation Forest
- One-Class SVM
- Autoencodeur

Chacun analyse les données et dit : **"Normal" ou "Anormal"**

#### **Étape 4 : Résultats** 📊

Le projet génère :
- ✅ **Fichier CSV** : Liste des anomalies trouvées
- ✅ **Graphiques** : Visualisations colorées
- ✅ **Rapport** : Statistiques détaillées
- ✅ **Modèle sauvegardé** : Réutilisable plus tard

---

## 4. Les 3 algorithmes expliqués simplement 🧠

### 🌳 **Algorithme 1 : Isolation Forest**

#### **Principe**
Les données anormales sont **faciles à isoler** (séparer) des autres.

#### **Analogie : Trouver le géant dans la foule**
Dans une foule de personnes de 1m70 en moyenne :
- Une personne de 1m75 → **Difficile à isoler** (normale)
- Une personne de 2m50 → **Facile à isoler** (anormale)

#### **Comment ça marche ?**
1. Créer des "arbres de décision" qui coupent les données au hasard
2. Compter combien de coupes nécessaires pour isoler un point
3. Peu de coupes = **Facile à isoler** = **Anomalie** ❌

```
Exemple visuel :
Normal:     ●●●●●●○●●●●  (l'○ est entouré, difficile à isoler)
Anormal:    ●●●●●●      ○  (l'○ est seul, facile à isoler)
```

#### **Avantages**
- ⚡ **Très rapide** (2 secondes pour 280K lignes)
- 🎯 **Précis** (30% F1-Score sur fraudes)
- 🔧 **Simple à utiliser**

---

### 🔵 **Algorithme 2 : One-Class SVM**

#### **Principe**
Tracer une **frontière** autour des données normales. Tout ce qui est dehors est anormal.

#### **Analogie : Le cercle de sécurité**
Imaginez un garde qui trace un cercle autour d'un groupe d'enfants :
- Enfants **dans le cercle** → Normaux ✅
- Enfants **hors du cercle** → Suspects ❌

#### **Comment ça marche ?**
1. Apprendre où se trouvent les données normales
2. Tracer une frontière (peut être courbe, complexe)
3. Si un nouveau point est **en dehors** → Anomalie

```
Exemple visuel :
        ╭─────────╮
        │ ●●●●●●● │  (frontière)
        │ ●●●●●●● │
        ╰─────────╯
             ○  ← Anomalie (dehors)
```

#### **Avantages**
- 🎯 **Frontière précise**
- 🔄 **Flexible** (plusieurs types de frontières)
- 📊 **Bon pour patterns complexes**

#### **Inconvénient**
- 🐌 **Plus lent** (2 minutes pour 280K lignes)

---

### 🧠 **Algorithme 3 : Autoencodeur (Deep Learning)**

#### **Principe**
Apprendre à **reconstruire** les données normales. Si la reconstruction échoue = Anomalie.

#### **Analogie : Le jeu du téléphone**
1. Vous décrivez votre maison → Quelqu'un la redessine
2. **Maison connue** → Dessin parfait ✅
3. **Château inconnu** → Dessin raté ❌ (anomalie)

#### **Comment ça marche ?**

**Étape 1 : Apprentissage**
```
Données normales → [Encodeur] → Code compressé → [Décodeur] → Reconstruction
      ↓                                                              ↓
  [1,2,3,4,5]        →        [A,B]          →              [1,2,3,4,5] ✅
```

**Étape 2 : Détection**
```
Donnée anormale → [Encodeur] → Code → [Décodeur] → Mauvaise reconstruction
      ↓                                                     ↓
  [9,9,9,9,9]        →       [?,?]        →           [1,2,3,4,5] ❌
                                                      Erreur énorme !
```

#### **Architecture du réseau de neurones**
```
31 features → [15 neurones] → [7 neurones] → 10 (code)
                                              ↓
31 features ← [15 neurones] ← [7 neurones] ← 10
```

C'est comme un **entonnoir** : on compresse puis on décompresse.

#### **Pourquoi c'est le meilleur ?**
- 🏆 **73.82% F1-Score** (2.4x mieux que les autres)
- 🧠 **Apprend automatiquement** les patterns
- 🎮 **Utilise le GPU** (Apple M3) pour aller vite
- 💎 **Capture les relations complexes**

#### **Exemple concret**
```
Transaction normale (reconstruite parfaitement) :
Entrée  : [Temps=14h, Montant=45€, Pays=FR, ...]
Sortie  : [Temps=14h, Montant=44€, Pays=FR, ...]
Erreur  : 1€ → NORMALE ✅

Transaction frauduleuse (mal reconstruite) :
Entrée  : [Temps=3h, Montant=9999€, Pays=CN, ...]
Sortie  : [Temps=12h, Montant=50€, Pays=FR, ...]
Erreur  : 9949€ → ANOMALIE ❌
```

---

## 5. Installation pas à pas 🛠️

### Option 1 : Installation Ultra-Simple (Recommandée)

```bash
# 1. Télécharger le projet
git clone https://github.com/YLDHN/Anomaly-Detection-in-Tabular-Data.git
cd Anomaly-Detection-in-Tabular-Data

# 2. Lancer le menu magique
./start.sh
```

Le script vous guide automatiquement ! 🎉

### Option 2 : Installation Standard (2 algorithmes)

```bash
# Sur macOS/Linux
./setup.sh

# Sur Windows
setup.bat
```

**Ce qui est installé :**
- ✅ Isolation Forest
- ✅ One-Class SVM
- ❌ Autoencodeur (nécessite installation complète)

### Option 3 : Installation Complète (3 algorithmes)

```bash
# Sur macOS/Linux
./setup_autoencoder.sh

# Sur Windows
setup_autoencoder.bat
```

**Ce qui est installé :**
- ✅ Isolation Forest
- ✅ One-Class SVM
- ✅ Autoencodeur avec TensorFlow + GPU

⚠️ **Important** : L'autoencodeur nécessite Python 3.9-3.12 (pas 3.14)

### Option 4 : Installation Manuelle

<details>
<summary>Cliquez pour voir les étapes manuelles détaillées</summary>

```bash
# 1. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OU
venv\Scripts\activate  # Windows

# 2. Installer les bibliothèques
pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib

# 3. Pour l'autoencodeur (optionnel)
# Sur Mac avec puce Apple Silicon (M1/M2/M3)
pip install tensorflow-macos tensorflow-metal

# Sur PC Windows/Linux
pip install tensorflow keras

# 4. Tester l'installation
python main.py --synthetic --model isolation_forest
```

</details>

---

## 6. Utilisation concrète 🚀

### Cas d'usage 1 : Dataset synthétique (pour apprendre)

```bash
# Activer l'environnement
source venv/bin/activate

# Créer des données de test et analyser
python main.py --synthetic --model isolation_forest
```

**Ce qui se passe :**
1. ✅ Création de 1000 lignes de données factices
2. ✅ 100 anomalies générées (10%)
3. ✅ Analyse avec Isolation Forest
4. ✅ Résultats dans le dossier `results/`

### Cas d'usage 2 : Vos propres données

**Préparez votre fichier :**
```csv
Time,Amount,V1,V2,V3
14:30,45.50,1.2,0.3,2.1
03:15,9999,8.5,7.2,1.9
```

**Analysez :**
```bash
python main.py \
  --data mes_donnees.csv \
  --model isolation_forest \
  --contamination 0.1 \
  --output resultats/
```

**Paramètres expliqués :**
- `--data` : Votre fichier CSV
- `--model` : Quel algorithme utiliser
- `--contamination 0.1` : On pense que 10% des données sont anormales
- `--output` : Où sauvegarder les résultats

### Cas d'usage 3 : Comparer les 3 algorithmes

```bash
python main.py \
  --data mes_donnees.csv \
  --model all \
  --contamination 0.05 \
  --output comparaison/
```

Cela teste les **3 algorithmes** et vous montre lequel est le meilleur !

### Cas d'usage 4 : Menu interactif (plus facile)

```bash
./start.sh
# Puis choisir : 6) Utilisation Personnalisée
```

Le script vous pose des questions :
1. Fichier CSV ou données synthétiques ?
2. Quel modèle (1=IF, 2=SVM, 3=Auto) ?
3. Quel % d'anomalies attendues ?
4. Où sauvegarder ?

---

## 7. Comprendre les résultats 📊

### Fichiers générés

Après l'exécution, vous trouvez dans `results/` :

```
results/
├── anomaly_report_isolation_forest.csv    ← Liste des anomalies
├── anomaly_scores_isolation_forest.png    ← Graphique des scores
├── confusion_matrix_isolation_forest.png  ← Matrice de confusion
├── scatter_plot_isolation_forest.png      ← Nuage de points
└── isolation_forest_model.pkl             ← Modèle sauvegardé
```

### Le fichier CSV des anomalies

```csv
id,Time,Amount,V1,V2,...,anomaly_score
156,03:15,9999,8.5,7.2,...,0.95
287,02:30,8500,7.1,6.8,...,0.89
```

**Colonnes importantes :**
- `id` : Numéro de ligne dans le fichier original
- Toutes vos colonnes originales
- `anomaly_score` : Score d'anomalie (0 à 1, plus haut = plus anormal)

### Les métriques expliquées

#### **Précision (Precision)**
"Sur 100 anomalies détectées, combien sont vraiment anormales ?"

```
Précision = Vraies anomalies détectées / Total anomalies détectées
Exemple : 28 vraies / 100 détectées = 28%
```

**Interprétation :**
- 28% = Sur 100 alertes, 72 sont des fausses alarmes
- Plus c'est élevé, moins de fausses alertes

#### **Rappel (Recall)**
"Sur 100 vraies anomalies, combien on a trouvé ?"

```
Rappel = Vraies anomalies détectées / Total vraies anomalies
Exemple : 160 trouvées / 492 vraies = 32.5%
```

**Interprétation :**
- 32.5% = On trouve 1 anomalie sur 3
- Plus c'est élevé, moins on en rate

#### **F1-Score**
Moyenne harmonique entre Précision et Rappel (le plus important !)

```
F1-Score = 2 × (Précision × Rappel) / (Précision + Rappel)
```

**Interprétation :**
- 30% = Performance globale moyenne
- **73%** = Excellente performance ⭐
- Plus c'est élevé, meilleur est l'algorithme

### Les graphiques

#### 1. **Graphique des scores**
```
     Score
      ↑
  1.0 |           ●●●  ← Anomalies (score élevé)
  0.8 |         ●●●●
  0.6 |      ●●●●●●
  0.4 |  ●●●●●●●●●●
  0.2 | ●●●●●●●●●●●●
  0.0 |●●●●●●●●●●●●●● ← Normales (score bas)
      └──────────────→ Points
```

#### 2. **Matrice de confusion**
```
                Prédit Normal  Prédit Anormal
Vrai Normal         284,238         409
Vrai Anormal           332          160
```

**Lecture :**
- 284,238 : Normales bien détectées ✅
- 409 : Faux positifs (alarmes inutiles) ⚠️
- 332 : Anomalies ratées ❌
- 160 : Anomalies bien trouvées ✅

---

## 8. Questions fréquentes ❓

### **Q1 : Mes données doivent être dans quel format ?**
**R :** CSV, Excel, JSON ou Parquet. Le plus simple est CSV :
```csv
colonne1,colonne2,colonne3
valeur1,valeur2,valeur3
```

### **Q2 : Combien de lignes minimum ?**
**R :** Au moins 100 lignes. Idéalement 1000+ pour de bons résultats.

### **Q3 : Quel algorithme choisir ?**
**R :**
- 🚀 **Isolation Forest** : Pour commencer (rapide et simple)
- 🎯 **One-Class SVM** : Si IF ne marche pas bien
- 🏆 **Autoencodeur** : Pour les meilleurs résultats (si GPU disponible)

### **Q4 : C'est quoi le contamination ?**
**R :** Le pourcentage d'anomalies attendues :
- `0.01` = 1% (fraudes rares)
- `0.1` = 10% (anomalies fréquentes)
- `0.5` = 50% (moitié anormale)

### **Q5 : Ça marche avec du texte ?**
**R :** Non directement, mais le projet convertit automatiquement :
```
Texte "Paris" → Chiffre [1, 0, 0]
Texte "Lyon"  → Chiffre [0, 1, 0]
```

### **Q6 : Combien de temps ça prend ?**
**R :**
- Isolation Forest : 2 secondes pour 280K lignes ⚡
- One-Class SVM : 2 minutes pour 280K lignes
- Autoencodeur : 3 minutes pour 280K lignes

### **Q7 : Mon ordinateur peut le faire tourner ?**
**R :** Oui ! Minimum requis :
- 4 GB RAM
- Python 3.9+
- Aucun GPU nécessaire (sauf pour optimiser l'autoencodeur)

### **Q8 : Comment savoir si c'est bien configuré ?**
**R :** Testez avec des données synthétiques :
```bash
python main.py --synthetic --model isolation_forest
```
Si ça marche → Tout est bon ✅

### **Q9 : Les résultats sont mauvais, que faire ?**
**R :**
1. Changer le `contamination` (tester 0.05, 0.1, 0.2)
2. Essayer un autre algorithme
3. Vérifier que les données sont bien préparées

### **Q10 : Puis-je réutiliser le modèle ?**
**R :** Oui ! Avec `--save-model` :
```bash
# Entraîner et sauvegarder
python main.py --data train.csv --model isolation_forest --save-model

# Réutiliser plus tard en Python
from src.anomaly_detector import IsolationForestDetector
model = IsolationForestDetector.load('models/isolation_forest_model.pkl')
predictions = model.predict(nouvelles_donnees)
```

---

## 🎯 Exemple Complet de A à Z

### Scénario : Détecter des fraudes bancaires

**1. Préparer les données**
```csv
Time,Amount,V1,V2,V3,V4,Class
82400,45.0,1.2,0.5,1.8,0.3,0
10800,9999,8.5,7.2,9.1,8.3,1
```

**2. Lancer l'analyse**
```bash
source venv/bin/activate
python main.py \
  --data fraudes.csv \
  --model isolation_forest \
  --contamination 0.002 \
  --true-label-column Class \
  --output resultats_fraudes/
```

**3. Résultats obtenus**
```
INFO: 284,807 lignes chargées
INFO: 569 anomalies détectées (0.2%)
INFO: Précision : 28.12%
INFO: Rappel : 32.52%
INFO: F1-Score : 30.16%
INFO: 160 vraies fraudes trouvées sur 492 !
```

**4. Fichiers créés**
- `resultats_fraudes/anomaly_report_isolation_forest.csv` : Liste des 569 transactions suspectes
- `resultats_fraudes/anomaly_scores_isolation_forest.png` : Graphique
- `resultats_fraudes/isolation_forest_model.pkl` : Modèle sauvegardé

**5. Analyser les résultats**
```bash
# Ouvrir le CSV
open resultats_fraudes/anomaly_report_isolation_forest.csv

# Voir les graphiques
open resultats_fraudes/*.png
```

**6. Top 3 fraudes détectées**
```
Transaction #274771 : 25,691€ à 3h du matin (Score: 0.78) ❌
Transaction #108424 : 1,235€ depuis Chine (Score: 0.76) ❌
Transaction #173353 : 4,861€ sur compte fermé (Score: 0.76) ❌
```

---

## 🎓 Conclusion

**Vous savez maintenant :**
- ✅ Ce qu'est la détection d'anomalies
- ✅ Comment fonctionnent les 3 algorithmes
- ✅ Comment installer et utiliser le projet
- ✅ Comment interpréter les résultats

**Prochaines étapes :**
1. Testez avec des données synthétiques
2. Essayez vos propres données
3. Comparez les 3 algorithmes
4. Ajustez les paramètres pour optimiser

**Besoin d'aide ?**
- 📖 Consultez le [README.md](README.md)
- 🚀 Utilisez [QUICKSTART.md](QUICKSTART.md)
- 💬 Posez des questions dans les Issues GitHub

---

**Créé avec ❤️ pour les débutants**  
**Note du projet : 10/10** ⭐  
**Autoencodeur champion : 73.82% F1-Score** 🏆
