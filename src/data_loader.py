"""
Module de chargement et gestion des données pour la détection d'anomalies.

Ce module fournit des fonctions pour :
- Charger des données tabulaires depuis différents formats
- Gérer les valeurs manquantes (imputation ou suppression)
- Afficher des statistiques de base
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(
    filepath: str,
    file_format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Charge un dataset tabulaire depuis un fichier.
    
    Args:
        filepath: Chemin vers le fichier de données
        file_format: Format du fichier ('csv', 'excel', 'json', 'parquet')
                    Si None, déduit du nom de fichier
        **kwargs: Arguments supplémentaires passés à la fonction de lecture pandas
    
    Returns:
        DataFrame pandas contenant les données
    
    Raises:
        ValueError: Si le format de fichier n'est pas supporté
        FileNotFoundError: Si le fichier n'existe pas
    """
    if file_format is None:
        # Déduire le format depuis l'extension
        if filepath.endswith('.csv'):
            file_format = 'csv'
        elif filepath.endswith(('.xls', '.xlsx')):
            file_format = 'excel'
        elif filepath.endswith('.json'):
            file_format = 'json'
        elif filepath.endswith('.parquet'):
            file_format = 'parquet'
        else:
            raise ValueError(f"Impossible de déduire le format du fichier : {filepath}")
    
    logger.info(f"Chargement des données depuis {filepath} (format: {file_format})")
    
    try:
        if file_format == 'csv':
            df = pd.read_csv(filepath, **kwargs)
        elif file_format == 'excel':
            df = pd.read_excel(filepath, **kwargs)
        elif file_format == 'json':
            df = pd.read_json(filepath, **kwargs)
        elif file_format == 'parquet':
            df = pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Format de fichier non supporté : {file_format}")
        
        logger.info(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé : {filepath}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {str(e)}")
        raise


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les valeurs manquantes dans le DataFrame.
    
    Args:
        df: DataFrame à analyser
    
    Returns:
        DataFrame avec le nombre et le pourcentage de valeurs manquantes par colonne
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs_Manquantes': missing_count.values,
        'Pourcentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Valeurs_Manquantes'] > 0].sort_values(
        'Valeurs_Manquantes', ascending=False
    )
    
    if len(missing_df) > 0:
        logger.info(f"\nValeurs manquantes détectées dans {len(missing_df)} colonnes")
        logger.info(f"\n{missing_df.to_string()}")
    else:
        logger.info("Aucune valeur manquante détectée")
    
    return missing_df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto',
    threshold: float = 0.5,
    numeric_method: str = 'mean',
    categorical_method: str = 'mode'
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le DataFrame.
    
    Args:
        df: DataFrame à traiter
        strategy: Stratégie globale ('auto', 'drop', 'impute')
                 'auto' : supprime les colonnes avec >threshold de valeurs manquantes,
                         impute le reste
                 'drop' : supprime toutes les lignes avec valeurs manquantes
                 'impute' : impute toutes les valeurs manquantes
        threshold: Seuil (0-1) pour supprimer les colonnes (uniquement pour 'auto')
        numeric_method: Méthode d'imputation pour colonnes numériques 
                       ('mean', 'median', 'mode', 'constant')
        categorical_method: Méthode d'imputation pour colonnes catégorielles
                           ('mode', 'constant')
    
    Returns:
        DataFrame avec valeurs manquantes traitées
    """
    df_cleaned = df.copy()
    initial_shape = df_cleaned.shape
    
    logger.info(f"Traitement des valeurs manquantes (stratégie: {strategy})")
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
        logger.info(f"Lignes supprimées : {initial_shape[0] - df_cleaned.shape[0]}")
    
    elif strategy in ['auto', 'impute']:
        # Pour la stratégie 'auto', supprimer d'abord les colonnes avec trop de valeurs manquantes
        if strategy == 'auto':
            missing_ratio = df_cleaned.isnull().sum() / len(df_cleaned)
            cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            
            if cols_to_drop:
                logger.info(f"Suppression de {len(cols_to_drop)} colonnes avec >{threshold*100}% de valeurs manquantes : {cols_to_drop}")
                df_cleaned = df_cleaned.drop(columns=cols_to_drop)
        
        # Imputer les valeurs manquantes restantes
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        
        # Imputation des colonnes numériques
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if numeric_method == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif numeric_method == 'median':
                    fill_value = df_cleaned[col].median()
                elif numeric_method == 'mode':
                    fill_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else df_cleaned[col].mean()
                else:  # constant
                    fill_value = 0
                
                df_cleaned[col].fillna(fill_value, inplace=True)
                logger.info(f"Colonne '{col}' : {df_cleaned[col].isnull().sum()} valeurs imputées avec {numeric_method} = {fill_value:.2f}")
        
        # Imputation des colonnes catégorielles
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if categorical_method == 'mode':
                    fill_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                else:  # constant
                    fill_value = 'Unknown'
                
                df_cleaned[col].fillna(fill_value, inplace=True)
                logger.info(f"Colonne '{col}' : {df_cleaned[col].isnull().sum()} valeurs imputées avec {categorical_method} = {fill_value}")
    
    else:
        raise ValueError(f"Stratégie non reconnue : {strategy}")
    
    logger.info(f"Forme finale : {df_cleaned.shape} (était : {initial_shape})")
    return df_cleaned


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Génère un résumé statistique des données.
    
    Args:
        df: DataFrame à analyser
    
    Returns:
        Dictionnaire contenant diverses statistiques
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'n_categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
        'n_missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict()
    }
    
    logger.info("\n=== RÉSUMÉ DES DONNÉES ===")
    logger.info(f"Lignes : {summary['n_rows']}")
    logger.info(f"Colonnes : {summary['n_columns']} ({summary['n_numeric_columns']} numériques, {summary['n_categorical_columns']} catégorielles)")
    logger.info(f"Valeurs manquantes : {summary['n_missing_values']} ({summary['missing_percentage']:.2f}%)")
    logger.info(f"Utilisation mémoire : {summary['memory_usage_mb']:.2f} MB")
    
    return summary


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    contamination: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Crée un dataset synthétique pour tester la détection d'anomalies.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        n_features: Nombre de features numériques
        contamination: Proportion d'anomalies à injecter
        random_state: Seed pour la reproductibilité
    
    Returns:
        DataFrame avec données normales et anomalies
    """
    np.random.seed(random_state)
    
    # Données normales (distribution normale)
    n_normal = int(n_samples * (1 - contamination))
    normal_data = np.random.randn(n_normal, n_features)
    
    # Anomalies (valeurs extrêmes)
    n_anomalies = n_samples - n_normal
    anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))
    
    # Combiner
    data = np.vstack([normal_data, anomalies])
    labels = np.array([0] * n_normal + [1] * n_anomalies)
    
    # Créer DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    
    # Ajouter quelques colonnes catégorielles
    df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    df['category_B'] = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
    
    # Ajouter une colonne d'ID
    df.insert(0, 'id', range(1, n_samples + 1))
    
    # Ajouter la vraie étiquette (pour évaluation)
    df['true_label'] = labels
    
    # Injecter quelques valeurs manquantes
    for col in feature_names[:2]:
        missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    logger.info(f"Dataset synthétique créé : {n_samples} échantillons, {n_features} features, {contamination*100}% d'anomalies")
    
    return df


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Création d'un dataset synthétique...")
    df = create_sample_dataset(n_samples=1000, n_features=5, contamination=0.1)
    
    print("\nAnalyse des valeurs manquantes...")
    analyze_missing_values(df)
    
    print("\nTraitement des valeurs manquantes...")
    df_cleaned = handle_missing_values(df, strategy='auto')
    
    print("\nRésumé des données...")
    summary = get_data_summary(df_cleaned)
