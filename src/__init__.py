"""
Package pour la détection d'anomalies dans des données tabulaires.

Ce package contient des modules pour :
- Charger et nettoyer des données
- Prétraiter et transformer les données
- Détecter les anomalies avec différents algorithmes
- Évaluer et visualiser les résultats
"""

__version__ = "1.0.0"
__author__ = "Votre nom"

from .data_loader import (
    load_data,
    handle_missing_values,
    analyze_missing_values,
    get_data_summary,
    create_sample_dataset
)

from .preprocessor import (
    DataPreprocessor,
    preprocess_data,
    split_train_test,
    create_feature_summary
)

from .anomaly_detector import (
    IsolationForestDetector,
    OneClassSVMDetector,
    AutoencoderDetector,
    compare_detectors
)

from .evaluator import (
    evaluate_predictions,
    evaluate_and_visualize,
    plot_confusion_matrix,
    plot_anomaly_scores,
    plot_scatter_2d,
    generate_anomaly_report,
    compare_models_visualization
)

__all__ = [
    # Data loading
    'load_data',
    'handle_missing_values',
    'analyze_missing_values',
    'get_data_summary',
    'create_sample_dataset',
    
    # Preprocessing
    'DataPreprocessor',
    'preprocess_data',
    'split_train_test',
    'create_feature_summary',
    
    # Anomaly detection
    'IsolationForestDetector',
    'OneClassSVMDetector',
    'AutoencoderDetector',
    'compare_detectors',
    
    # Evaluation
    'evaluate_predictions',
    'evaluate_and_visualize',
    'plot_confusion_matrix',
    'plot_anomaly_scores',
    'plot_scatter_2d',
    'generate_anomaly_report',
    'compare_models_visualization',
]
