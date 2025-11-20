"""
Module de prétraitement des données pour la détection d'anomalies.

Ce module fournit des fonctions pour :
- Normaliser les colonnes numériques (min-max ou standardisation)
- Encoder les colonnes catégorielles (one-hot ou label encoding)
- Séparer les données en train/test
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union, List
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Classe pour prétraiter les données avant la détection d'anomalies.
    """
    
    def __init__(
        self,
        numeric_scaling: str = 'standard',
        categorical_encoding: str = 'onehot',
        handle_categorical: bool = True
    ):
        """
        Initialise le préprocesseur.
        
        Args:
            numeric_scaling: Type de normalisation ('standard', 'minmax', 'none')
            categorical_encoding: Type d'encodage ('onehot', 'label', 'none')
            handle_categorical: Si False, ignore les colonnes catégorielles
        """
        self.numeric_scaling = numeric_scaling
        self.categorical_encoding = categorical_encoding
        self.handle_categorical = handle_categorical
        
        self.scaler = None
        self.encoders = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Apprend les transformations sur les données.
        
        Args:
            df: DataFrame à prétraiter
            exclude_columns: Colonnes à exclure du prétraitement
        
        Returns:
            self
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Identifier les colonnes numériques et catégorielles
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Exclure les colonnes spécifiées
        self.numeric_columns = [col for col in self.numeric_columns if col not in exclude_columns]
        self.categorical_columns = [col for col in self.categorical_columns if col not in exclude_columns]
        
        logger.info(f"Colonnes numériques : {len(self.numeric_columns)}")
        logger.info(f"Colonnes catégorielles : {len(self.categorical_columns)}")
        
        # Préparer le scaler pour les colonnes numériques
        if self.numeric_scaling == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.numeric_columns])
            logger.info("StandardScaler appliqué aux colonnes numériques")
        
        elif self.numeric_scaling == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.numeric_columns])
            logger.info("MinMaxScaler appliqué aux colonnes numériques")
        
        # Préparer les encoders pour les colonnes catégorielles
        if self.handle_categorical and len(self.categorical_columns) > 0:
            if self.categorical_encoding == 'label':
                for col in self.categorical_columns:
                    encoder = LabelEncoder()
                    encoder.fit(df[col].astype(str))
                    self.encoders[col] = encoder
                logger.info(f"LabelEncoder appliqué à {len(self.categorical_columns)} colonnes catégorielles")
            
            elif self.categorical_encoding == 'onehot':
                # OneHotEncoder pour toutes les colonnes catégorielles
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(df[self.categorical_columns].astype(str))
                self.encoders['onehot'] = encoder
                logger.info(f"OneHotEncoder appliqué à {len(self.categorical_columns)} colonnes catégorielles")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applique les transformations apprises sur les données.
        
        Args:
            df: DataFrame à transformer
        
        Returns:
            Array numpy avec les données transformées
        """
        if not self.is_fitted:
            raise ValueError("Le préprocesseur doit d'abord être ajusté avec fit()")
        
        transformed_parts = []
        
        # Transformer les colonnes numériques
        if len(self.numeric_columns) > 0:
            if self.scaler is not None:
                numeric_transformed = self.scaler.transform(df[self.numeric_columns])
            else:
                numeric_transformed = df[self.numeric_columns].values
            transformed_parts.append(numeric_transformed)
        
        # Transformer les colonnes catégorielles
        if self.handle_categorical and len(self.categorical_columns) > 0:
            if self.categorical_encoding == 'label':
                categorical_transformed = np.zeros((len(df), len(self.categorical_columns)))
                for i, col in enumerate(self.categorical_columns):
                    categorical_transformed[:, i] = self.encoders[col].transform(df[col].astype(str))
                transformed_parts.append(categorical_transformed)
            
            elif self.categorical_encoding == 'onehot':
                categorical_transformed = self.encoders['onehot'].transform(df[self.categorical_columns].astype(str))
                transformed_parts.append(categorical_transformed)
        
        # Combiner toutes les parties
        if len(transformed_parts) > 0:
            result = np.hstack(transformed_parts)
        else:
            raise ValueError("Aucune colonne à transformer")
        
        logger.info(f"Données transformées : shape {result.shape}")
        return result
    
    def fit_transform(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Apprend et applique les transformations en une seule étape.
        
        Args:
            df: DataFrame à prétraiter
            exclude_columns: Colonnes à exclure du prétraitement
        
        Returns:
            Array numpy avec les données transformées
        """
        self.fit(df, exclude_columns)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        Retourne les noms des features après transformation.
        
        Returns:
            Liste des noms de features
        """
        if not self.is_fitted:
            raise ValueError("Le préprocesseur doit d'abord être ajusté avec fit()")
        
        feature_names = []
        
        # Noms des features numériques
        feature_names.extend(self.numeric_columns)
        
        # Noms des features catégorielles
        if self.handle_categorical and len(self.categorical_columns) > 0:
            if self.categorical_encoding == 'label':
                feature_names.extend(self.categorical_columns)
            elif self.categorical_encoding == 'onehot':
                onehot_features = self.encoders['onehot'].get_feature_names_out(self.categorical_columns)
                feature_names.extend(onehot_features)
        
        return feature_names
    
    def save(self, filepath: str):
        """
        Sauvegarde le préprocesseur.
        
        Args:
            filepath: Chemin où sauvegarder le préprocesseur
        """
        joblib.dump(self, filepath)
        logger.info(f"Préprocesseur sauvegardé dans {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """
        Charge un préprocesseur sauvegardé.
        
        Args:
            filepath: Chemin du préprocesseur à charger
        
        Returns:
            Préprocesseur chargé
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Préprocesseur chargé depuis {filepath}")
        return preprocessor


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les données en ensembles train et test.
    
    Args:
        df: DataFrame à séparer
        test_size: Proportion des données pour le test (0-1)
        random_state: Seed pour la reproductibilité
        stratify_column: Colonne pour stratification (optionnel)
    
    Returns:
        Tuple (train_df, test_df)
    """
    stratify = df[stratify_column] if stratify_column else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Données séparées : Train={len(train_df)}, Test={len(test_df)}")
    return train_df, test_df


def preprocess_data(
    df: pd.DataFrame,
    numeric_scaling: str = 'standard',
    categorical_encoding: str = 'onehot',
    exclude_columns: Optional[List[str]] = None,
    return_preprocessor: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, DataPreprocessor]]:
    """
    Fonction helper pour prétraiter rapidement les données.
    
    Args:
        df: DataFrame à prétraiter
        numeric_scaling: Type de normalisation ('standard', 'minmax', 'none')
        categorical_encoding: Type d'encodage ('onehot', 'label', 'none')
        exclude_columns: Colonnes à exclure du prétraitement
        return_preprocessor: Si True, retourne aussi le preprocessor
    
    Returns:
        Array numpy transformé (et optionnellement le preprocessor)
    """
    preprocessor = DataPreprocessor(
        numeric_scaling=numeric_scaling,
        categorical_encoding=categorical_encoding
    )
    
    X_transformed = preprocessor.fit_transform(df, exclude_columns)
    
    if return_preprocessor:
        return X_transformed, preprocessor
    else:
        return X_transformed


def inverse_transform_numeric(
    X: np.ndarray,
    preprocessor: DataPreprocessor
) -> np.ndarray:
    """
    Applique la transformation inverse sur les colonnes numériques.
    
    Args:
        X: Données transformées
        preprocessor: Préprocesseur utilisé
    
    Returns:
        Données dans l'échelle originale
    """
    if preprocessor.scaler is not None:
        n_numeric = len(preprocessor.numeric_columns)
        X_numeric = X[:, :n_numeric]
        X_numeric_original = preprocessor.scaler.inverse_transform(X_numeric)
        
        # Remplacer les colonnes numériques
        X_inverse = X.copy()
        X_inverse[:, :n_numeric] = X_numeric_original
        return X_inverse
    else:
        return X


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un résumé des features du DataFrame.
    
    Args:
        df: DataFrame à analyser
    
    Returns:
        DataFrame avec statistiques par feature
    """
    summary_data = []
    
    for col in df.columns:
        col_info = {
            'Feature': col,
            'Type': str(df[col].dtype),
            'Non-Null': df[col].count(),
            'Null': df[col].isnull().sum(),
            'Unique': df[col].nunique()
        }
        
        if np.issubdtype(df[col].dtype, np.number):
            col_info.update({
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max()
            })
        
        summary_data.append(col_info)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


if __name__ == "__main__":
    # Exemple d'utilisation
    from data_loader import create_sample_dataset
    
    print("Création d'un dataset synthétique...")
    df = create_sample_dataset(n_samples=1000, n_features=5)
    
    print("\nRésumé des features...")
    summary = create_feature_summary(df)
    print(summary)
    
    print("\nPrétraitement des données...")
    # Exclure 'id' et 'true_label' du prétraitement
    X, preprocessor = preprocess_data(
        df,
        numeric_scaling='standard',
        categorical_encoding='onehot',
        exclude_columns=['id', 'true_label'],
        return_preprocessor=True
    )
    
    print(f"\nForme des données transformées : {X.shape}")
    print(f"Noms des features : {preprocessor.get_feature_names()}")
    
    print("\nSéparation train/test...")
    train_df, test_df = split_train_test(df, test_size=0.2)
