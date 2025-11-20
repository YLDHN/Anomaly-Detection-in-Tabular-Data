"""
Module d'évaluation et visualisation des résultats de détection d'anomalies.

Ce module fournit des fonctions pour :
- Évaluer les performances des détecteurs
- Visualiser les anomalies détectées
- Générer des rapports détaillés
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Évalue les prédictions d'un détecteur d'anomalies.
    
    Args:
        y_true: Vraies étiquettes (1=normal, -1=anomalie ou 0=normal, 1=anomalie)
        y_pred: Prédictions (1=normal, -1=anomalie)
        model_name: Nom du modèle
    
    Returns:
        Dictionnaire avec les métriques
    """
    # Convertir les étiquettes si nécessaire (0/1 -> 1/-1)
    if set(np.unique(y_true)) == {0, 1}:
        y_true_converted = np.where(y_true == 1, -1, 1)
    else:
        y_true_converted = y_true
    
    # Convertir pour sklearn (1/-1 -> 0/1)
    y_true_sklearn = np.where(y_true_converted == -1, 1, 0)
    y_pred_sklearn = np.where(y_pred == -1, 1, 0)
    
    metrics = {
        'precision': precision_score(y_true_sklearn, y_pred_sklearn, zero_division=0),
        'recall': recall_score(y_true_sklearn, y_pred_sklearn, zero_division=0),
        'f1_score': f1_score(y_true_sklearn, y_pred_sklearn, zero_division=0),
        'n_anomalies_true': np.sum(y_true_sklearn == 1),
        'n_anomalies_pred': np.sum(y_pred_sklearn == 1),
        'total_samples': len(y_true)
    }
    
    logger.info(f"\n=== Évaluation : {model_name} ===")
    logger.info(f"Précision : {metrics['precision']:.4f}")
    logger.info(f"Rappel : {metrics['recall']:.4f}")
    logger.info(f"F1-Score : {metrics['f1_score']:.4f}")
    logger.info(f"Anomalies vraies : {metrics['n_anomalies_true']}")
    logger.info(f"Anomalies prédites : {metrics['n_anomalies_pred']}")
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Affiche la matrice de confusion.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        model_name: Nom du modèle
        save_path: Chemin pour sauvegarder la figure
    """
    # Convertir les étiquettes
    if set(np.unique(y_true)) == {0, 1}:
        y_true_converted = np.where(y_true == 1, -1, 1)
    else:
        y_true_converted = y_true
    
    y_true_sklearn = np.where(y_true_converted == -1, 1, 0)
    y_pred_sklearn = np.where(y_pred == -1, 1, 0)
    
    cm = confusion_matrix(y_true_sklearn, y_pred_sklearn)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomalie'],
                yticklabels=['Normal', 'Anomalie'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraies Étiquettes')
    plt.xlabel('Prédictions')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matrice de confusion sauvegardée dans {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_anomaly_scores(
    scores: np.ndarray,
    predictions: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Visualise la distribution des scores d'anomalie.
    
    Args:
        scores: Scores d'anomalie
        predictions: Prédictions (-1 ou 1)
        y_true: Vraies étiquettes si disponibles
        model_name: Nom du modèle
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogramme des scores
    pred_labels = np.where(predictions == -1, 'Anomalie', 'Normal')
    
    for label in ['Normal', 'Anomalie']:
        mask = pred_labels == label
        axes[0].hist(scores[mask], bins=50, alpha=0.6, label=label)
    
    axes[0].set_xlabel('Score d\'anomalie')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title(f'Distribution des Scores - {model_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    df_scores = pd.DataFrame({
        'Score': scores,
        'Prédiction': pred_labels
    })
    
    if y_true is not None:
        if set(np.unique(y_true)) == {0, 1}:
            y_true_converted = np.where(y_true == 1, -1, 1)
        else:
            y_true_converted = y_true
        df_scores['Vérité'] = np.where(y_true_converted == -1, 'Anomalie', 'Normal')
        
        sns.boxplot(data=df_scores, x='Vérité', y='Score', hue='Prédiction', ax=axes[1])
        axes[1].set_title(f'Scores par Vérité et Prédiction - {model_name}')
    else:
        sns.boxplot(data=df_scores, x='Prédiction', y='Score', ax=axes[1])
        axes[1].set_title(f'Scores par Prédiction - {model_name}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique des scores sauvegardé dans {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_scatter_2d(
    X: np.ndarray,
    predictions: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    features_to_plot: Tuple[int, int] = (0, 1),
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Visualise les anomalies dans un espace 2D.
    
    Args:
        X: Données (features)
        predictions: Prédictions (-1 ou 1)
        y_true: Vraies étiquettes si disponibles
        feature_names: Noms des features
        features_to_plot: Indices des deux features à afficher
        model_name: Nom du modèle
        save_path: Chemin pour sauvegarder la figure
    """
    if X.shape[1] < 2:
        logger.warning("Pas assez de features pour un scatter plot 2D")
        return
    
    idx1, idx2 = features_to_plot
    
    if feature_names:
        xlabel = feature_names[idx1]
        ylabel = feature_names[idx2]
    else:
        xlabel = f'Feature {idx1}'
        ylabel = f'Feature {idx2}'
    
    # Créer subplots
    if y_true is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Vraies étiquettes
        if set(np.unique(y_true)) == {0, 1}:
            y_true_converted = np.where(y_true == 1, -1, 1)
        else:
            y_true_converted = y_true
        
        colors_true = np.where(y_true_converted == -1, 'red', 'blue')
        axes[0].scatter(X[:, idx1], X[:, idx2], c=colors_true, alpha=0.6, s=50)
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(f'Vraies Étiquettes')
        axes[0].legend(['Normal', 'Anomalie'])
        
        # Prédictions
        colors_pred = np.where(predictions == -1, 'red', 'blue')
        axes[1].scatter(X[:, idx1], X[:, idx2], c=colors_pred, alpha=0.6, s=50)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_title(f'Prédictions - {model_name}')
        axes[1].legend(['Normal', 'Anomalie'])
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_pred = np.where(predictions == -1, 'red', 'blue')
        ax.scatter(X[:, idx1], X[:, idx2], c=colors_pred, alpha=0.6, s=50)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Anomalies Détectées - {model_name}')
        ax.legend(['Normal', 'Anomalie'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Scatter plot sauvegardé dans {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_feature_boxplots(
    df: pd.DataFrame,
    predictions: np.ndarray,
    numeric_columns: Optional[List[str]] = None,
    max_features: int = 6,
    save_path: Optional[str] = None
):
    """
    Affiche des boxplots des features numériques par classe.
    
    Args:
        df: DataFrame original
        predictions: Prédictions (-1 ou 1)
        numeric_columns: Liste des colonnes numériques à afficher
        max_features: Nombre maximum de features à afficher
        save_path: Chemin pour sauvegarder la figure
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limiter le nombre de features
    numeric_columns = numeric_columns[:max_features]
    
    n_features = len(numeric_columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    pred_labels = np.where(predictions == -1, 'Anomalie', 'Normal')
    
    for i, col in enumerate(numeric_columns):
        df_plot = pd.DataFrame({
            col: df[col],
            'Classe': pred_labels
        })
        
        sns.boxplot(data=df_plot, x='Classe', y=col, ax=axes[i])
        axes[i].set_title(f'Distribution de {col}')
        axes[i].grid(True, alpha=0.3)
    
    # Masquer les axes vides
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Boxplots sauvegardés dans {save_path}")
    
    plt.tight_layout()
    plt.show()


def generate_anomaly_report(
    df: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Génère un rapport détaillé des anomalies détectées.
    
    Args:
        df: DataFrame original
        predictions: Prédictions (-1 ou 1)
        scores: Scores d'anomalie
        top_n: Nombre d'anomalies à inclure dans le rapport
        save_path: Chemin pour sauvegarder le rapport CSV
    
    Returns:
        DataFrame avec le rapport des anomalies
    """
    # Filtrer les anomalies
    anomaly_mask = predictions == -1
    
    if np.sum(anomaly_mask) == 0:
        logger.warning("Aucune anomalie détectée")
        return pd.DataFrame()
    
    # Créer le rapport
    report_df = df[anomaly_mask].copy()
    report_df['anomaly_score'] = scores[anomaly_mask]
    
    # Trier par score décroissant
    report_df = report_df.sort_values('anomaly_score', ascending=False).head(top_n)
    
    logger.info(f"\n=== TOP {top_n} ANOMALIES DÉTECTÉES ===")
    logger.info(f"\n{report_df.to_string()}")
    
    if save_path:
        report_df.to_csv(save_path, index=False)
        logger.info(f"Rapport sauvegardé dans {save_path}")
    
    return report_df


def compare_models_visualization(
    X: np.ndarray,
    results: Dict[str, np.ndarray],
    y_true: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    features_to_plot: Tuple[int, int] = (0, 1),
    save_path: Optional[str] = None
):
    """
    Compare visuellement les résultats de plusieurs modèles.
    
    Args:
        X: Données (features)
        results: Dictionnaire {nom_modèle: prédictions}
        y_true: Vraies étiquettes si disponibles
        feature_names: Noms des features
        features_to_plot: Indices des deux features à afficher
        save_path: Chemin pour sauvegarder la figure
    """
    n_models = len(results)
    n_cols = 2 if y_true is not None else 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    idx1, idx2 = features_to_plot
    
    if feature_names:
        xlabel = feature_names[idx1]
        ylabel = feature_names[idx2]
    else:
        xlabel = f'Feature {idx1}'
        ylabel = f'Feature {idx2}'
    
    # Afficher les vraies étiquettes si disponibles
    if y_true is not None:
        if set(np.unique(y_true)) == {0, 1}:
            y_true_converted = np.where(y_true == 1, -1, 1)
        else:
            y_true_converted = y_true
        
        colors_true = np.where(y_true_converted == -1, 'red', 'blue')
        axes[0].scatter(X[:, idx1], X[:, idx2], c=colors_true, alpha=0.6, s=50)
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title('Vraies Étiquettes')
        start_idx = 1
    else:
        start_idx = 0
    
    # Afficher les prédictions de chaque modèle
    for i, (model_name, predictions) in enumerate(results.items()):
        ax_idx = start_idx + i
        colors_pred = np.where(predictions == -1, 'red', 'blue')
        axes[ax_idx].scatter(X[:, idx1], X[:, idx2], c=colors_pred, alpha=0.6, s=50)
        axes[ax_idx].set_xlabel(xlabel)
        axes[ax_idx].set_ylabel(ylabel)
        axes[ax_idx].set_title(f'{model_name}')
    
    # Masquer les axes vides
    for i in range(start_idx + n_models, len(axes)):
        axes[i].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparaison sauvegardée dans {save_path}")
    
    plt.tight_layout()
    plt.show()


def evaluate_and_visualize(
    df: pd.DataFrame,
    predictions: np.ndarray,
    X: np.ndarray,
    scores: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    model_name: str = "Model",
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
):
    """
    Pipeline complet d'évaluation et visualisation.
    
    Args:
        df: DataFrame original
        predictions: Prédictions du modèle
        X: Données transformées (features)
        scores: Scores d'anomalie
        y_true: Vraies étiquettes si disponibles
        model_name: Nom du modèle
        feature_names: Noms des features
        output_dir: Répertoire pour sauvegarder les visualisations
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ÉVALUATION ET VISUALISATION - {model_name}")
    logger.info(f"{'='*60}")
    
    # Évaluation
    if y_true is not None:
        metrics = evaluate_predictions(y_true, predictions, model_name)
        plot_confusion_matrix(
            y_true, predictions, model_name,
            save_path=f"{output_dir}/confusion_matrix_{model_name}.png" if output_dir else None
        )
    
    # Visualisation des scores
    if scores is not None:
        plot_anomaly_scores(
            scores, predictions, y_true, model_name,
            save_path=f"{output_dir}/anomaly_scores_{model_name}.png" if output_dir else None
        )
    
    # Scatter plot 2D
    plot_scatter_2d(
        X, predictions, y_true, feature_names,
        model_name=model_name,
        save_path=f"{output_dir}/scatter_plot_{model_name}.png" if output_dir else None
    )
    
    # Boxplots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        plot_feature_boxplots(
            df, predictions, numeric_cols,
            save_path=f"{output_dir}/boxplots_{model_name}.png" if output_dir else None
        )
    
    # Rapport des anomalies
    if scores is not None:
        report = generate_anomaly_report(
            df, predictions, scores, top_n=20,
            save_path=f"{output_dir}/anomaly_report_{model_name}.csv" if output_dir else None
        )


if __name__ == "__main__":
    # Exemple d'utilisation
    from data_loader import create_sample_dataset, handle_missing_values
    from preprocessor import preprocess_data
    from anomaly_detector import IsolationForestDetector
    
    print("Création d'un dataset synthétique...")
    df = create_sample_dataset(n_samples=1000, n_features=5, contamination=0.1)
    
    print("\nTraitement et prétraitement...")
    df_cleaned = handle_missing_values(df, strategy='auto')
    X, preprocessor = preprocess_data(
        df_cleaned,
        exclude_columns=['id', 'true_label'],
        return_preprocessor=True
    )
    
    print("\nDétection d'anomalies...")
    detector = IsolationForestDetector(contamination=0.1)
    predictions = detector.fit_predict(X)
    scores = detector.get_anomaly_scores(X)
    
    print("\nÉvaluation et visualisation...")
    y_true = df_cleaned['true_label'].values
    evaluate_and_visualize(
        df_cleaned,
        predictions,
        X,
        scores=scores,
        y_true=y_true,
        model_name="IsolationForest",
        feature_names=preprocessor.get_feature_names()
    )
