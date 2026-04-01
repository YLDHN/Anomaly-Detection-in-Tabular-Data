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
from src.data_loader import load_data


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


class TestPreprocessingWithRealData:
    """Tests du prétraitement avec le dataset réel Fraud Detection."""
    
    @pytest.fixture
    def real_data(self):
        """Charge le dataset réel Fraud Detection."""
        data_path = Path(__file__).parent.parent / 'data' / 'Fraud Detection Transactions Dataset.csv'
        if not data_path.exists():
            pytest.skip("Fichier Fraud Detection Dataset non trouvé")
        
        df = load_data(str(data_path))
        return df.iloc[:500].copy()  # Prendre un échantillon
    
    def test_preprocess_real_data(self, real_data):
        """Test du prétraitement avec données réelles."""
        exclude_cols = ['Transaction_ID', 'User_ID', 'Timestamp', 'IP_Address_Flag', 'Fraud_Label']
        
        X = preprocess_data(real_data, exclude_columns=exclude_cols, numeric_scaling='standard', return_preprocessor=False)
        
        assert X.shape[0] == real_data.shape[0]
        assert X.shape[1] > 0
        assert isinstance(X, np.ndarray)
    
    def test_preprocessor_with_real_data(self, real_data):
        """Test de la classe DataPreprocessor avec données réelles."""
        exclude_cols = ['Transaction_ID', 'User_ID', 'Timestamp', 'IP_Address_Flag', 'Fraud_Label']
        
        preprocessor = DataPreprocessor(numeric_scaling='standard', categorical_encoding='onehot')
        X = preprocessor.fit_transform(real_data, exclude_columns=exclude_cols)
        
        assert preprocessor.is_fitted
        assert X.shape[0] == real_data.shape[0]
        
        # Vérifier les noms de features
        feature_names = preprocessor.get_feature_names()
        assert len(feature_names) == X.shape[1]
    
    def test_train_test_split_real_data(self, real_data):
        """Test du split train/test avec données réelles."""
        train_df, test_df = split_train_test(real_data, test_size=0.2, random_state=42)
        
        assert len(train_df) + len(test_df) == len(real_data)
        assert len(train_df) == int(len(real_data) * 0.8)
        assert len(test_df) == int(len(real_data) * 0.2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
