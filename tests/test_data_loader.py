"""
Tests unitaires pour le module data_loader.
Exécuter avec: pytest tests/test_data_loader.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data_loader import (
    load_data,
    handle_missing_values,
    analyze_missing_values,
    get_data_summary,
    create_sample_dataset
)


class TestCreateSampleDataset:
    """Tests pour la création de datasets synthétiques."""
    
    def test_create_sample_dataset_basic(self):
        """Test de création basique d'un dataset."""
        df = create_sample_dataset(n_samples=100, n_features=5, contamination=0.1)
        
        assert len(df) == 100
        assert 'true_label' in df.columns
        assert sum(df['true_label']) == 10  # 10% d'anomalies
    
    def test_create_sample_dataset_shapes(self):
        """Test des dimensions du dataset."""
        df = create_sample_dataset(n_samples=200, n_features=3)
        
        # 3 features + 2 catégorielles + id + true_label = 7 colonnes
        assert df.shape == (200, 7)
    
    def test_create_sample_dataset_contamination(self):
        """Test de différents niveaux de contamination."""
        for contamination in [0.05, 0.1, 0.15, 0.2]:
            df = create_sample_dataset(n_samples=1000, contamination=contamination)
            expected_anomalies = int(1000 * contamination)
            assert sum(df['true_label']) == expected_anomalies


class TestLoadData:
    """Tests pour le chargement de données."""
    
    def test_load_csv(self):
        """Test de chargement d'un fichier CSV."""
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('a,b,c\n1,2,3\n4,5,6\n')
            temp_path = f.name
        
        try:
            df = load_data(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ['a', 'b', 'c']
        finally:
            os.unlink(temp_path)
    
    def test_load_fraud_detection_dataset(self):
        """Test de chargement du dataset Fraud Detection réel."""
        data_path = Path(__file__).parent.parent / 'data' / 'Fraud Detection Transactions Dataset.csv'
        if data_path.exists():
            df = load_data(str(data_path))
            assert len(df) > 0
            assert 'Fraud_Label' in df.columns
            assert 'Transaction_ID' in df.columns
            assert df.shape[1] > 10  # Vérifier qu'il y a plusieurs colonnes
        else:
            pytest.skip("Fichier Fraud Detection Dataset non trouvé")
    
    def test_load_unsupported_format(self):
        """Test avec un format non supporté."""
        with pytest.raises(ValueError):
            load_data('test.unknown')


class TestHandleMissingValues:
    """Tests pour la gestion des valeurs manquantes."""
    
    def test_drop_strategy(self):
        """Test de la stratégie 'drop'."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        
        df_clean = handle_missing_values(df, strategy='drop')
        assert len(df_clean) == 2  # Seules 2 lignes sans NaN
    
    def test_impute_strategy(self):
        """Test de la stratégie 'impute'."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [5.0, np.nan, 7.0, 8.0]
        })
        
        df_clean = handle_missing_values(df, strategy='impute', numeric_method='mean')
        assert df_clean.isnull().sum().sum() == 0
        assert abs(df_clean.loc[2, 'a'] - 2.333) < 0.01  # Moyenne de 1,2,4
    
    def test_auto_strategy(self):
        """Test de la stratégie 'auto'."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [np.nan] * 5  # 100% de valeurs manquantes
        })
        
        df_clean = handle_missing_values(df, strategy='auto', threshold=0.5)
        assert 'b' not in df_clean.columns  # Colonne supprimée


class TestAnalyzeMissingValues:
    """Tests pour l'analyse des valeurs manquantes."""
    
    def test_analyze_missing_values(self):
        """Test de l'analyse des valeurs manquantes."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1, np.nan, 3, np.nan, 5]
        })
        
        missing_df = analyze_missing_values(df)
        assert len(missing_df) == 1
        assert missing_df.loc[0, 'Colonne'] == 'b'
        assert missing_df.loc[0, 'Valeurs_Manquantes'] == 2
        assert missing_df.loc[0, 'Pourcentage'] == 40.0


class TestGetDataSummary:
    """Tests pour le résumé des données."""
    
    def test_get_data_summary(self):
        """Test du résumé des données."""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4.5, 5.5, 6.5],
            'cat': ['A', 'B', 'C']
        })
        
        summary = get_data_summary(df)
        
        assert summary['n_rows'] == 3
        assert summary['n_columns'] == 3
        assert summary['n_numeric_columns'] == 2
        assert summary['n_categorical_columns'] == 1
        assert summary['n_missing_values'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
