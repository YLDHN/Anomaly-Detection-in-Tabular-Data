#!/usr/bin/env python3
"""
Script principal pour exécuter la détection d'anomalies en ligne de commande.

Utilisation :
    python main.py --data data/dataset.csv --model isolation_forest --contamination 0.1
    python main.py --help pour voir toutes les options
"""

import argparse
import sys
import os
import logging
import numpy as np
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import (
    load_data,
    handle_missing_values,
    analyze_missing_values,
    get_data_summary,
    create_sample_dataset
)
from src.preprocessor import preprocess_data
from src.anomaly_detector import (
    IsolationForestDetector,
    OneClassSVMDetector,
    AutoencoderDetector
)
from src.evaluator import (
    evaluate_predictions,
    evaluate_and_visualize,
    generate_anomaly_report
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Détection d\'anomalies dans des données tabulaires',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  # Créer et analyser un dataset synthétique
  python main.py --synthetic --model all
  
  # Analyser vos propres données avec Isolation Forest
  python main.py --data data/transactions.csv --model isolation_forest
  
  # Comparer tous les modèles avec visualisations
  python main.py --data data/sensors.csv --model all --output results/
  
  # Ajuster les hyperparamètres
  python main.py --data data/data.csv --model isolation_forest --contamination 0.05 --n-estimators 200
        """
    )
    
    # Données
    data_group = parser.add_argument_group('Données')
    data_group.add_argument(
        '--data',
        type=str,
        help='Chemin vers le fichier de données (CSV, Excel, JSON, Parquet)'
    )
    data_group.add_argument(
        '--synthetic',
        action='store_true',
        help='Créer un dataset synthétique pour la démonstration'
    )
    data_group.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Nombre d\'échantillons pour le dataset synthétique (défaut: 1000)'
    )
    data_group.add_argument(
        '--true-label-column',
        type=str,
        help='Nom de la colonne contenant les vraies étiquettes (optionnel)'
    )
    data_group.add_argument(
        '--exclude-columns',
        type=str,
        nargs='+',
        help='Colonnes à exclure du prétraitement'
    )
    
    # Prétraitement
    preprocess_group = parser.add_argument_group('Prétraitement')
    preprocess_group.add_argument(
        '--missing-strategy',
        type=str,
        choices=['auto', 'drop', 'impute'],
        default='auto',
        help='Stratégie pour gérer les valeurs manquantes (défaut: auto)'
    )
    preprocess_group.add_argument(
        '--scaling',
        type=str,
        choices=['standard', 'minmax', 'none'],
        default='standard',
        help='Type de normalisation (défaut: standard)'
    )
    preprocess_group.add_argument(
        '--encoding',
        type=str,
        choices=['onehot', 'label', 'none'],
        default='onehot',
        help='Type d\'encodage pour les catégories (défaut: onehot)'
    )
    
    # Modèle
    model_group = parser.add_argument_group('Modèle')
    model_group.add_argument(
        '--model',
        type=str,
        choices=['isolation_forest', 'onesvm', 'autoencoder', 'all'],
        default='isolation_forest',
        help='Algorithme de détection à utiliser (défaut: isolation_forest)'
    )
    model_group.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Proportion attendue d\'anomalies (défaut: 0.1)'
    )
    
    # Hyperparamètres Isolation Forest
    if_group = parser.add_argument_group('Isolation Forest')
    if_group.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Nombre d\'arbres pour Isolation Forest (défaut: 100)'
    )
    
    # Hyperparamètres One-Class SVM
    svm_group = parser.add_argument_group('One-Class SVM')
    svm_group.add_argument(
        '--kernel',
        type=str,
        choices=['linear', 'poly', 'rbf', 'sigmoid'],
        default='rbf',
        help='Type de kernel pour SVM (défaut: rbf)'
    )
    svm_group.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Coefficient du kernel (défaut: scale)'
    )
    
    # Hyperparamètres Autoencodeur
    ae_group = parser.add_argument_group('Autoencodeur')
    ae_group.add_argument(
        '--encoding-dim',
        type=int,
        default=8,
        help='Dimension de l\'espace latent (défaut: 8)'
    )
    ae_group.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Nombre d\'époques d\'entraînement (défaut: 50)'
    )
    ae_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Taille des batchs (défaut: 32)'
    )
    
    # Sortie
    output_group = parser.add_argument_group('Sortie')
    output_group.add_argument(
        '--output',
        type=str,
        help='Répertoire pour sauvegarder les résultats'
    )
    output_group.add_argument(
        '--save-model',
        action='store_true',
        help='Sauvegarder le(s) modèle(s) entraîné(s)'
    )
    output_group.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Ne pas afficher les visualisations'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.data and not args.synthetic:
        parser.error("Vous devez spécifier --data ou --synthetic")
    
    return args


def main():
    """Fonction principale."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("DÉTECTION D'ANOMALIES DANS DES DONNÉES TABULAIRES")
    logger.info("="*80)
    
    # Créer le répertoire de sortie si nécessaire
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        logger.info(f"Résultats seront sauvegardés dans : {args.output}")
    
    # === 1. CHARGEMENT DES DONNÉES ===
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
    logger.info("="*80)
    
    if args.synthetic:
        logger.info("Création d'un dataset synthétique...")
        df = create_sample_dataset(
            n_samples=args.n_samples,
            n_features=5,
            contamination=args.contamination
        )
        y_true = df['true_label'].values
    else:
        logger.info(f"Chargement des données depuis : {args.data}")
        df = load_data(args.data)
        
        # Extraire les vraies étiquettes si spécifiées
        if args.true_label_column:
            if args.true_label_column in df.columns:
                y_true = df[args.true_label_column].values
                logger.info(f"Vraies étiquettes chargées depuis '{args.true_label_column}'")
            else:
                logger.warning(f"Colonne '{args.true_label_column}' non trouvée")
                y_true = None
        else:
            y_true = None
    
    # Résumé des données
    summary = get_data_summary(df)
    
    # === 2. PRÉTRAITEMENT ===
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 2 : PRÉTRAITEMENT DES DONNÉES")
    logger.info("="*80)
    
    # Analyser les valeurs manquantes
    analyze_missing_values(df)
    
    # Traiter les valeurs manquantes
    df_cleaned = handle_missing_values(
        df,
        strategy=args.missing_strategy
    )
    
    # Déterminer les colonnes à exclure
    exclude_cols = args.exclude_columns or []
    if args.synthetic or args.true_label_column:
        exclude_cols.extend(['id', 'true_label'])
    exclude_cols = list(set(exclude_cols))  # Supprimer les doublons
    
    # Prétraiter les données
    X, preprocessor = preprocess_data(
        df_cleaned,
        numeric_scaling=args.scaling,
        categorical_encoding=args.encoding,
        exclude_columns=exclude_cols,
        return_preprocessor=True
    )
    
    logger.info(f"Données transformées : {X.shape}")
    
    # === 3. DÉTECTION D'ANOMALIES ===
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 3 : DÉTECTION D'ANOMALIES")
    logger.info("="*80)
    
    models_to_run = []
    
    if args.model == 'all':
        models_to_run = ['isolation_forest', 'onesvm', 'autoencoder']
    else:
        models_to_run = [args.model]
    
    results = {}
    
    for model_name in models_to_run:
        logger.info(f"\n--- {model_name.upper()} ---")
        
        if model_name == 'isolation_forest':
            detector = IsolationForestDetector(
                contamination=args.contamination,
                n_estimators=args.n_estimators
            )
        elif model_name == 'onesvm':
            detector = OneClassSVMDetector(
                nu=args.contamination,
                kernel=args.kernel,
                gamma=args.gamma
            )
        elif model_name == 'autoencoder':
            detector = AutoencoderDetector(
                encoding_dim=args.encoding_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                contamination=args.contamination,
                verbose=1 if not args.no_visualizations else 0
            )
        
        # Entraîner et prédire
        predictions = detector.fit_predict(X)
        scores = detector.get_anomaly_scores(X)
        
        results[model_name] = {
            'detector': detector,
            'predictions': predictions,
            'scores': scores
        }
        
        # Sauvegarder le modèle si demandé
        if args.save_model:
            model_path = f"{args.output or 'models'}/{model_name}_model.pkl"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            detector.save(model_path)
    
    # === 4. ÉVALUATION ===
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 4 : ÉVALUATION ET VISUALISATION")
    logger.info("="*80)
    
    for model_name, result in results.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"RÉSULTATS : {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        predictions = result['predictions']
        scores = result['scores']
        
        # Évaluation si les vraies étiquettes sont disponibles
        if y_true is not None:
            metrics = evaluate_predictions(y_true, predictions, model_name)
        
        # Générer le rapport des anomalies
        report = generate_anomaly_report(
            df_cleaned,
            predictions,
            scores,
            top_n=20,
            save_path=f"{args.output}/anomaly_report_{model_name}.csv" if args.output else None
        )
        
        # Visualisations
        if not args.no_visualizations:
            evaluate_and_visualize(
                df_cleaned,
                predictions,
                X,
                scores=scores,
                y_true=y_true,
                model_name=model_name,
                feature_names=preprocessor.get_feature_names(),
                output_dir=args.output
            )
    
    # === 5. COMPARAISON (si plusieurs modèles) ===
    if len(results) > 1 and y_true is not None:
        logger.info("\n" + "="*80)
        logger.info("COMPARAISON DES MODÈLES")
        logger.info("="*80)
        
        import pandas as pd
        
        comparison_data = []
        for model_name, result in results.items():
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            predictions = result['predictions']
            y_true_sklearn = np.where(y_true == 1, -1, 1) if set(np.unique(y_true)) == {0, 1} else y_true
            y_pred_sklearn = np.where(predictions == -1, 1, 0)
            y_true_sklearn_binary = np.where(y_true_sklearn == -1, 1, 0)
            
            comparison_data.append({
                'Modèle': model_name,
                'Précision': precision_score(y_true_sklearn_binary, y_pred_sklearn, zero_division=0),
                'Rappel': recall_score(y_true_sklearn_binary, y_pred_sklearn, zero_division=0),
                'F1-Score': f1_score(y_true_sklearn_binary, y_pred_sklearn, zero_division=0),
                'Anomalies Détectées': np.sum(predictions == -1)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        if args.output:
            comparison_df.to_csv(f"{args.output}/model_comparison.csv", index=False)
            logger.info(f"\nComparaison sauvegardée dans {args.output}/model_comparison.csv")
    
    # === 6. CONCLUSION ===
    logger.info("\n" + "="*80)
    logger.info("✅ TRAITEMENT TERMINÉ")
    logger.info("="*80)
    
    if args.output:
        logger.info(f"\nTous les résultats ont été sauvegardés dans : {args.output}")
    
    logger.info("\nMerci d'avoir utilisé ce système de détection d'anomalies !")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interruption par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Erreur : {str(e)}", exc_info=True)
        sys.exit(1)
