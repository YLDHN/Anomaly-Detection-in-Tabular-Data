# Guide de Contribution

Merci de votre intérêt pour contribuer au projet de Détection d'Anomalies ! 🎉

## 🤝 Comment contribuer

### 1. Signaler un bug

Si vous trouvez un bug, veuillez :

1. Vérifier que le bug n'a pas déjà été signalé dans les [issues](https://github.com/votre-repo/issues)
2. Ouvrir une nouvelle issue avec :
   - Un titre clair et descriptif
   - Les étapes pour reproduire le bug
   - Le comportement attendu vs observé
   - Votre environnement (OS, version Python, etc.)
   - Les logs d'erreur si disponibles

**Template :**
```markdown
### Description du bug
[Description claire du problème]

### Étapes de reproduction
1. ...
2. ...
3. ...

### Comportement attendu
[Ce qui devrait se passer]

### Environnement
- OS: [e.g., macOS 13.0]
- Python: [e.g., 3.11.0]
- Version du projet: [e.g., 1.0.0]
```

### 2. Proposer une fonctionnalité

Pour proposer une nouvelle fonctionnalité :

1. Ouvrir une issue avec le label `enhancement`
2. Décrire la fonctionnalité souhaitée
3. Expliquer le cas d'usage
4. (Optionnel) Proposer une implémentation

### 3. Soumettre une Pull Request

#### Pré-requis

1. Fork le repository
2. Clone votre fork :
   ```bash
   git clone https://github.com/votre-username/DADT.git
   cd DADT
   ```
3. Créer une branche :
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

#### Développement

1. **Installer l'environnement de dev :**
   ```bash
   ./setup.sh
   source venv/bin/activate
   pip install pytest pytest-cov black ruff
   ```

2. **Faire vos modifications**

3. **Ajouter des tests :**
   ```python
   # tests/test_mon_module.py
   def test_ma_fonctionnalite():
       # Votre test
       assert True
   ```

4. **Lancer les tests :**
   ```bash
   pytest tests/ -v
   ```

5. **Formater le code :**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   ```

6. **Commit et Push :**
   ```bash
   git add .
   git commit -m "feat: ajout de ma fonctionnalité"
   git push origin feature/ma-fonctionnalite
   ```

7. **Ouvrir une Pull Request**

#### Convention de commits

Utilisez [Conventional Commits](https://www.conventionalcommits.org/) :

- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `docs:` documentation uniquement
- `style:` formatage (pas de changement de code)
- `refactor:` refactoring (ni bug ni feature)
- `test:` ajout ou modification de tests
- `chore:` tâches de maintenance

**Exemples :**
```bash
feat: ajout du modèle DBSCAN pour la détection
fix: correction du calcul de contamination
docs: mise à jour du README avec exemples
test: ajout tests pour OneClassSVM
```

## 📋 Checklist PR

Avant de soumettre votre PR, vérifiez que :

- [ ] Le code suit les conventions du projet
- [ ] Les tests passent (`pytest tests/`)
- [ ] Le code est formaté (`black` et `ruff`)
- [ ] La documentation est à jour
- [ ] Les nouveaux fichiers ont les headers appropriés
- [ ] Les commits suivent la convention
- [ ] La PR a une description claire

## 🧪 Tests

### Lancer tous les tests
```bash
pytest tests/ -v
```

### Avec couverture
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Tests spécifiques
```bash
pytest tests/test_data_loader.py -v
pytest tests/test_data_loader.py::TestLoadData::test_load_csv -v
```

## 📝 Style de code

### Python (PEP 8)

- Indentation : 4 espaces
- Longueur de ligne : max 100 caractères
- Imports : groupés et triés (stdlib, third-party, local)
- Docstrings : format Google ou NumPy
- Type hints : utilisés partout où c'est pertinent

**Exemple :**
```python
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import helper_function


def process_data(
    df: pd.DataFrame,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Traite les données selon un seuil donné.
    
    Args:
        df: DataFrame à traiter
        threshold: Seuil de filtrage (0-1)
        verbose: Afficher les logs
    
    Returns:
        Tuple contenant les données traitées et les métadonnées
    
    Raises:
        ValueError: Si threshold n'est pas dans [0, 1]
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold doit être dans [0, 1], reçu {threshold}")
    
    # Implementation
    return processed_data, metadata
```

### Formatage automatique

```bash
# Black (formatter)
black src/ tests/

# Ruff (linter)
ruff check src/ tests/
ruff check --fix src/ tests/  # Correction auto
```

## 🏗️ Architecture

### Ajout d'un nouveau détecteur

1. **Créer la classe** dans `src/anomaly_detector.py` :

```python
class MonNouveauDetecteur(BaseAnomalyDetector):
    """
    Description du détecteur.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="MonDetecteur")
        # Initialisation
    
    def fit(self, X: np.ndarray):
        # Entraînement
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Prédiction
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        # Scores
        return scores
```

2. **Ajouter des tests** dans `tests/test_anomaly_detector.py`

3. **Mettre à jour la documentation**

### Ajout d'un format de données

1. **Modifier** `src/data_loader.py` :

```python
def load_data(filepath: str, file_format: Optional[str] = None):
    # ...
    if file_format == 'nouveau_format':
        df = pd.read_nouveauformat(filepath, **kwargs)
    # ...
```

2. **Ajouter des tests**

3. **Documenter dans le README**

## 📚 Documentation

### Docstrings

Utilisez le format Google :

```python
def ma_fonction(param1: int, param2: str) -> bool:
    """
    Brève description de la fonction.
    
    Description plus détaillée si nécessaire,
    sur plusieurs lignes.
    
    Args:
        param1: Description du premier paramètre
        param2: Description du second paramètre
    
    Returns:
        Description de la valeur de retour
    
    Raises:
        ValueError: Quand param1 est négatif
        TypeError: Quand param2 n'est pas une chaîne
    
    Example:
        >>> ma_fonction(5, "test")
        True
    """
    pass
```

### README et guides

- Mettre à jour le README si vous ajoutez une fonctionnalité
- Ajouter des exemples d'utilisation
- Inclure des captures d'écran si pertinent

## 🎯 Priorités actuelles

Domaines où les contributions sont particulièrement bienvenues :

1. **Nouveaux algorithmes** : DBSCAN, LOF, HDBSCAN
2. **Visualisations** : Graphiques interactifs, dashboards
3. **Performance** : Optimisations, parallélisation
4. **Documentation** : Tutoriels, exemples avancés
5. **Tests** : Augmenter la couverture à 95%+

## 💬 Communication

- **Issues** : Pour les bugs et features
- **Discussions** : Pour les questions et idées
- **Pull Requests** : Pour les contributions de code

## 🙏 Remerciements

Merci à tous les contributeurs qui rendent ce projet possible !

---

**Questions ?** N'hésitez pas à ouvrir une issue ou une discussion !
