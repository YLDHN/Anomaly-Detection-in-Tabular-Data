"""
Tests unitaires pour le module preprocessor.
Exécuter avec: pytest tests/test_preprocessor.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.preprocessor import (
    DataPreprocessor,
    preprocess_data,
    split_train_test,
    create_feature_summary
)


class TestDataPreprocessor:
    """Tests pour la classe DataPreprocessor."""
    
    @pytest.fixture
    def sample_df(self):
        """Crée un DataFrame de test."""
        return pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X']
        })
    
    def test_preprocessor_initialization(self):
        """Test de l'initialisation du préprocesseur."""
        preprocessor = DataPreprocessor(
            numeric_scaling='standard',
            categorical_encoding='onehot'
        )
        
        assert preprocessor.numeric_scaling == 'standard'
        assert preprocessor.categorical_encoding == 'onehot'
        assert not preprocessor.is_fitted
    
    def test_fit_transform(self, sample_df):
        """Test de fit_transform."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df)
        
        assert preprocessor.is_fitted
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] > 0
    
    def test_standard_scaling(self, sample_df):
        """Test du StandardScaler."""
        preprocessor = DataPreprocessor(numeric_scaling='standard')
        X = preprocessor.fit_transform(sample_df)
        
        # Vérifier que les colonnes numériques sont standardisées
        # (moyenne ≈ 0, écart-type ≈ 1)
        num_cols = X[:, :2]  # Les 2 premières colonnes sont numériques
        assert abs(np.mean(num_cols[:, 0])) < 1e-10
        assert abs(np.std(num_cols[:, 0]) - 1.0) < 1e-10
    
    def test_minmax_scaling(self, sample_df):
        """Test du MinMaxScaler."""
        preprocessor = DataPreprocessor(numeric_scaling='minmax')
        X = preprocessor.fit_transform(sample_df)
        
        # Vérifier que les valeurs sont entre 0 et 1
        num_cols = X[:, :2]
        assert np.min(num_cols) >= 0
        assert np.max(num_cols) <= 1
    
    def test_onehot_encoding(self, sample_df):
        """Test de l'encodage OneHot."""
        preprocessor = DataPreprocessor(categorical_encoding='onehot')
        X = preprocessor.fit_transform(sample_df)
        
        # OneHot: cat1 a 3 valeurs (A,B,C), cat2 a 2 valeurs (X,Y)
        # Total: 2 num + 3 + 2 = 7 features
        assert X.shape[1] == 7
    
    def test_label_encoding(self, sample_df):
        """Test de l'encodage Label."""
        preprocessor = DataPreprocessor(categorical_encoding='label')
        X = preprocessor.fit_transform(sample_df)
        
        # Label: 2 num + 2 cat = 4 features
        assert X.shape[1] == 4
    
    def test_exclude_columns(self, sample_df):
        """Test de l'exclusion de colonnes."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df, exclude_columns=['num1'])
        
        assert 'num1' not in preprocessor.numeric_columns
    
    def test_get_feature_names(self, sample_df):
        """Test de la récupération des noms de features."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_df)
        
        feature_names = preprocessor.get_feature_names()
        assert len(feature_names) > 0
        assert isinstance(feature_names, list)


class TestPreprocessData:
    """Tests pour la fonction preprocess_data."""
    
    def test_preprocess_data_basic(self):
        """Test basique de preprocess_data."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': ['X', 'Y', 'Z']
        })
        
        X, preprocessor = preprocess_data(df, return_preprocessor=True)
        
        assert X.shape[0] == 3
        assert isinstance(preprocessor, DataPreprocessor)
    
    def test_preprocess_data_without_preprocessor(self):
        """Test sans retourner le preprocessor."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        X = preprocess_data(df, return_preprocessor=False)
        
        assert isinstance(X, np.ndarray)


class TestSplitTrainTest:
    """Tests pour split_train_test."""
    
    def test_split_basic(self):
        """Test basique du split."""
        df = pd.DataFrame({'a': range(100), 'b': range(100)})
        
        train_df, test_df = split_train_test(df, test_size=0.2)
        
        assert len(train_df) == 80
        assert len(test_df) == 20
    
    def test_split_stratify(self):
        """Test du split avec stratification."""
        df = pd.DataFrame({
            'a': range(100),
            'label': [0] * 80 + [1] * 20
        })
        
        train_df, test_df = split_train_test(
            df, 
            test_size=0.2, 
            stratify_column='label'
        )
        
        # Vérifier que les proportions sont maintenues
        train_ratio = sum(train_df['label']) / len(train_df)
        test_ratio = sum(test_df['label']) / len(test_df)
        
        assert abs(train_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05


class TestCreateFeatureSummary:
    """Tests pour create_feature_summary."""
    
    def test_feature_summary(self):
        """Test du résumé des features."""
        df = pd.DataFrame({
            'num': [1, 2, 3, 4, 5],
            'cat': ['A', 'B', 'A', 'B', 'C'],
            'with_nan': [1.0, np.nan, 3.0, 4.0, 5.0]
        })
        
        summary = create_feature_summary(df)
        
        assert len(summary) == 3
        assert 'Feature' in summary.columns
        assert 'Type' in summary.columns
        assert 'Null' in summary.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
