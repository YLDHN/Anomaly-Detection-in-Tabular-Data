"""
Tests unitaires pour le module anomaly_detector.
Exécuter avec: pytest tests/test_anomaly_detector.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.anomaly_detector import (
    IsolationForestDetector,
    OneClassSVMDetector,
    compare_detectors
)
from src.data_loader import load_data
from src.preprocessor import preprocess_data


class TestIsolationForestDetector:
    """Tests pour Isolation Forest."""
    
    @pytest.fixture
    def sample_data(self):
        """Crée des données de test."""
        np.random.seed(42)
        # Données normales
        normal = np.random.randn(100, 5)
        # Anomalies
        anomalies = np.random.uniform(-10, 10, (10, 5))
        X = np.vstack([normal, anomalies])
        return X
    
    def test_initialization(self):
        """Test de l'initialisation."""
        detector = IsolationForestDetector(contamination=0.1)
        
        assert detector.contamination == 0.1
        assert not detector.is_fitted
    
    def test_fit_predict(self, sample_data):
        """Test de fit_predict."""
        detector = IsolationForestDetector(contamination=0.1)
        predictions = detector.fit_predict(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert detector.is_fitted
        assert -1 in predictions  # Au moins une anomalie
        assert 1 in predictions   # Au moins un point normal
    
    def test_get_anomaly_scores(self, sample_data):
        """Test des scores d'anomalie."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(sample_data)
        
        scores = detector.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)
        assert all(scores >= 0)  # Les scores sont positifs
    
    def test_contamination_effect(self, sample_data):
        """Test de l'effet du paramètre contamination."""
        detector1 = IsolationForestDetector(contamination=0.05)
        detector2 = IsolationForestDetector(contamination=0.15)
        
        pred1 = detector1.fit_predict(sample_data)
        pred2 = detector2.fit_predict(sample_data)
        
        n_anomalies1 = sum(pred1 == -1)
        n_anomalies2 = sum(pred2 == -1)
        
        # Plus de contamination = plus d'anomalies détectées
        assert n_anomalies2 > n_anomalies1


class TestOneClassSVMDetector:
    """Tests pour One-Class SVM."""
    
    @pytest.fixture
    def sample_data(self):
        """Crée des données de test."""
        np.random.seed(42)
        normal = np.random.randn(100, 5)
        anomalies = np.random.uniform(-10, 10, (10, 5))
        X = np.vstack([normal, anomalies])
        return X
    
    def test_initialization(self):
        """Test de l'initialisation."""
        detector = OneClassSVMDetector(nu=0.1, kernel='rbf')
        
        assert detector.nu == 0.1
        assert not detector.is_fitted
    
    def test_fit_predict(self, sample_data):
        """Test de fit_predict."""
        detector = OneClassSVMDetector(nu=0.1)
        predictions = detector.fit_predict(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert detector.is_fitted
        assert -1 in predictions
        assert 1 in predictions
    
    def test_get_anomaly_scores(self, sample_data):
        """Test des scores d'anomalie."""
        detector = OneClassSVMDetector(nu=0.1)
        detector.fit(sample_data)
        
        scores = detector.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)
        assert all(scores >= 0)
    
    def test_different_kernels(self, sample_data):
        """Test avec différents kernels."""
        kernels = ['linear', 'rbf', 'poly']
        
        for kernel in kernels:
            detector = OneClassSVMDetector(nu=0.1, kernel=kernel)
            predictions = detector.fit_predict(sample_data)
            assert len(predictions) == len(sample_data)


class TestCompareDetectors:
    """Tests pour la comparaison des détecteurs."""
    
    def test_compare_detectors_without_labels(self):
        """Test de comparaison sans vraies étiquettes."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        results = compare_detectors(X, contamination=0.1)
        
        assert 'IsolationForest' in results
        assert 'OneClassSVM' in results
        assert len(results['IsolationForest']) == 100
    
    def test_compare_detectors_with_labels(self):
        """Test de comparaison avec vraies étiquettes."""
        np.random.seed(42)
        normal = np.random.randn(90, 5)
        anomalies = np.random.uniform(-10, 10, (10, 5))
        X = np.vstack([normal, anomalies])
        y_true = np.array([1] * 90 + [-1] * 10)
        
        results = compare_detectors(X, y_true=y_true, contamination=0.1)
        
        assert len(results) >= 2


class TestModelPersistence:
    """Tests pour la sauvegarde/chargement des modèles."""
    
    def test_save_load_isolation_forest(self, tmp_path):
        """Test sauvegarde/chargement Isolation Forest."""
        X = np.random.randn(100, 5)
        
        # Entraîner et sauvegarder
        detector1 = IsolationForestDetector(contamination=0.1)
        detector1.fit(X)
        
        model_path = tmp_path / "if_model.pkl"
        detector1.save(str(model_path))
        
        # Charger et vérifier
        detector2 = IsolationForestDetector.load(str(model_path))
        
        pred1 = detector1.predict(X)
        pred2 = detector2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_save_load_onesvm(self, tmp_path):
        """Test sauvegarde/chargement One-Class SVM."""
        X = np.random.randn(100, 5)
        
        detector1 = OneClassSVMDetector(nu=0.1)
        detector1.fit(X)
        
        model_path = tmp_path / "svm_model.pkl"
        detector1.save(str(model_path))
        
        detector2 = OneClassSVMDetector.load(str(model_path))
        
        pred1 = detector1.predict(X)
        pred2 = detector2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestDetectorsWithRealData:
    """Tests des détecteurs avec le dataset réel Fraud Detection."""
    
    @pytest.fixture
    def real_data(self):
        """Charge le dataset réel Fraud Detection et le prétraite."""
        data_path = Path(__file__).parent.parent / 'data' / 'Fraud Detection Transactions Dataset.csv'
        if not data_path.exists():
            pytest.skip("Fichier Fraud Detection Dataset non trouvé")
        
        df = load_data(str(data_path))
        # Ajouter true_label si Fraud_Label existe
        if 'Fraud_Label' in df.columns:
            df['true_label'] = df['Fraud_Label']
        
        # Prétraiter les données
        exclude_cols = ['Transaction_ID', 'User_ID', 'Timestamp', 'IP_Address_Flag', 'true_label', 'Fraud_Label']
        X = preprocess_data(df, exclude_columns=exclude_cols, numeric_scaling='standard', return_preprocessor=False)
        
        # Prendre un échantillon pour les tests (plus rapide)
        n_samples = min(1000, X.shape[0])
        X_sample = X[:n_samples]
        y_true = df['true_label'].values[:n_samples] if 'true_label' in df.columns else None
        
        return X_sample, y_true
    
    def test_isolation_forest_with_real_data(self, real_data):
        """Test Isolation Forest avec données réelles."""
        X, y_true = real_data
        
        detector = IsolationForestDetector(contamination=0.1)
        predictions = detector.fit_predict(X)
        
        assert len(predictions) == X.shape[0]
        assert detector.is_fitted
        assert -1 in predictions or 1 in predictions
    
    def test_ocsvm_with_real_data(self, real_data):
        """Test One-Class SVM avec données réelles."""
        X, y_true = real_data
        
        detector = OneClassSVMDetector(nu=0.1)
        predictions = detector.fit_predict(X)
        
        assert len(predictions) == X.shape[0]
        assert detector.is_fitted
        assert -1 in predictions or 1 in predictions
    
    def test_compare_with_real_data(self, real_data):
        """Test la comparaison des modèles avec données réelles."""
        X, y_true = real_data
        
        results = compare_detectors(X, y_true=y_true, contamination=0.1)
        
        assert len(results) >= 2
        assert isinstance(results, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
