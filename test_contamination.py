#!/usr/bin/env python3
"""
Script rapide pour tester différents niveaux de contamination
et trouver les meilleurs paramètres
"""
import sys
sys.path.insert(0, 'src')

from src.data_loader import load_data, handle_missing_values
from src.preprocessor import preprocess_data
from src.anomaly_detector import IsolationForestDetector, OneClassSVMDetector
from src.evaluator import evaluate_predictions
import numpy as np

# Charger les données
print("📥 Chargement des données...")
df = load_data('data/Fraud Detection Transactions Dataset.csv')
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Ajouter true_label
df['true_label'] = df['Fraud_Label'].copy()
print(f"✅ True fraud rate: {df['true_label'].sum() / len(df) * 100:.2f}%")

# Nettoyer
df_clean = handle_missing_values(df, strategy='auto')

# Prétraiter
exclude_cols = ['Transaction_ID', 'User_ID', 'Timestamp', 'IP_Address_Flag', 'true_label', 'Fraud_Label']
X = preprocess_data(df_clean, exclude_columns=exclude_cols, numeric_scaling='standard', return_preprocessor=False)

# Convertir true_label en -1/1
y_true = np.where(df_clean['true_label'] == 0, 1, -1)

print(f"\n📊 Testing different contamination levels:\n")
print("=" * 80)

# Tester différentes contaminations
contaminations = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35]

for contamination in contaminations:
    print(f"\n🧪 Contamination = {contamination:.2f} ({int(contamination * len(X))} samples)")
    print("-" * 80)
    
    # Isolation Forest
    detector_if = IsolationForestDetector(contamination=contamination, n_estimators=50)
    predictions_if = detector_if.fit_predict(X)
    metrics_if = evaluate_predictions(y_true, predictions_if, f"IF-{contamination}")
    
    print(f"  Isolation Forest:")
    print(f"    Precision: {metrics_if['precision']:.3f}")
    print(f"    Recall:    {metrics_if['recall']:.3f}")
    print(f"    F1-Score:  {metrics_if['f1_score']:.3f}")
    
    # One-Class SVM (plus rapide avec subset)
    if contamination <= 0.2:  # Tester rapide
        detector_svm = OneClassSVMDetector(nu=contamination, kernel='rbf')
        predictions_svm = detector_svm.fit_predict(X)
        metrics_svm = evaluate_predictions(y_true, predictions_svm, f"SVM-{contamination}")
        
        print(f"  One-Class SVM:")
        print(f"    Precision: {metrics_svm['precision']:.3f}")
        print(f"    Recall:    {metrics_svm['recall']:.3f}")
        print(f"    F1-Score:  {metrics_svm['f1_score']:.3f}")

print("\n" + "=" * 80)
print("✅ Test terminé!")
print("\nRecommandation: utiliser la contamination avec le meilleur F1-Score")
