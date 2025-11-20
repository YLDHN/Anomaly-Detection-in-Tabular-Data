"""
Tests unitaires pour le module evaluator.
Exécuter avec: pytest tests/test_evaluator.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.evaluator import (
    evaluate_predictions,
    generate_anomaly_report
)


class TestEvaluatePredictions:
    """Tests pour l'évaluation des prédictions."""
    
    def test_evaluate_perfect_predictions(self):
        """Test avec des prédictions parfaites."""
        y_true = np.array([1, 1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, -1, -1])
        
        metrics = evaluate_predictions(y_true, y_pred, "TestModel")
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_evaluate_with_binary_labels(self):
        """Test avec des étiquettes 0/1."""
        y_true = np.array([0, 0, 0, 1, 1])  # 0=normal, 1=anomalie
        y_pred = np.array([1, 1, 1, -1, -1])
        
        metrics = evaluate_predictions(y_true, y_pred, "TestModel")
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_evaluate_no_anomalies_detected(self):
        """Test quand aucune anomalie n'est détectée."""
        y_true = np.array([1, 1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, 1, 1])  # Toutes prédites normales
        
        metrics = evaluate_predictions(y_true, y_pred, "TestModel")
        
        assert metrics['n_anomalies_pred'] == 0
        assert metrics['recall'] == 0.0
    
    def test_evaluate_counts(self):
        """Test des comptages."""
        y_true = np.array([1] * 80 + [-1] * 20)
        y_pred = np.array([1] * 85 + [-1] * 15)
        
        metrics = evaluate_predictions(y_true, y_pred, "TestModel")
        
        assert metrics['total_samples'] == 100
        assert metrics['n_anomalies_true'] == 20
        assert metrics['n_anomalies_pred'] == 15


class TestGenerateAnomalyReport:
    """Tests pour la génération de rapports."""
    
    @pytest.fixture
    def sample_data(self):
        """Crée des données de test."""
        df = pd.DataFrame({
            'id': range(1, 11),
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'category': ['A', 'B'] * 5
        })
        predictions = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1])
        scores = np.random.rand(10)
        
        return df, predictions, scores
    
    def test_generate_anomaly_report_basic(self, sample_data):
        """Test basique de génération de rapport."""
        df, predictions, scores = sample_data
        
        report = generate_anomaly_report(df, predictions, scores, top_n=5)
        
        assert len(report) <= 5  # Max top_n anomalies
        assert 'anomaly_score' in report.columns
        assert all(report['anomaly_score'].notna())
    
    def test_generate_anomaly_report_save(self, sample_data, tmp_path):
        """Test de sauvegarde du rapport."""
        df, predictions, scores = sample_data
        
        save_path = tmp_path / "anomalies.csv"
        report = generate_anomaly_report(
            df, predictions, scores, 
            top_n=5, 
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        loaded_report = pd.read_csv(save_path)
        assert len(loaded_report) == len(report)
    
    def test_generate_anomaly_report_ordering(self, sample_data):
        """Test que les anomalies sont triées par score."""
        df, predictions, scores = sample_data
        
        report = generate_anomaly_report(df, predictions, scores, top_n=10)
        
        # Vérifier que les scores sont en ordre décroissant
        scores_list = report['anomaly_score'].tolist()
        assert scores_list == sorted(scores_list, reverse=True)
    
    def test_generate_anomaly_report_only_anomalies(self, sample_data):
        """Test que seules les anomalies sont dans le rapport."""
        df, predictions, scores = sample_data
        
        report = generate_anomaly_report(df, predictions, scores, top_n=10)
        
        # Toutes les lignes du rapport devraient être des anomalies
        n_anomalies = sum(predictions == -1)
        assert len(report) == min(n_anomalies, 10)


class TestMetricsEdgeCases:
    """Tests des cas limites pour les métriques."""
    
    def test_all_normal_predictions(self):
        """Test quand tout est prédit normal."""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, 1])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        assert metrics['recall'] == 0.0
        assert metrics['n_anomalies_pred'] == 0
    
    def test_all_anomaly_predictions(self):
        """Test quand tout est prédit anomalie."""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([-1, -1, -1, -1])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        assert metrics['n_anomalies_pred'] == 4
    
    def test_empty_predictions(self):
        """Test avec des prédictions vides."""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Ne devrait pas lever d'exception
        metrics = evaluate_predictions(y_true, y_pred)
        assert metrics['total_samples'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
