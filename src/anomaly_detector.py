"""
Module de détection d'anomalies.

Ce module implémente plusieurs algorithmes de détection d'anomalies :
- Isolation Forest
- One-Class SVM
- Autoencodeur (Deep Learning)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Tuple, Dict
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAnomalyDetector:
    """
    Classe de base pour les détecteurs d'anomalies.
    """
    
    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.anomaly_scores_ = None
        self.predictions_ = None
    
    def fit(self, X: np.ndarray):
        """
        Entraîne le modèle sur les données.
        
        Args:
            X: Données d'entraînement
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les anomalies (-1 pour anomalie, 1 pour normal).
        
        Args:
            X: Données à prédire
        
        Returns:
            Prédictions (-1 ou 1)
        """
        raise NotImplementedError
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Entraîne et prédit en une seule étape.
        
        Args:
            X: Données d'entraînement
        
        Returns:
            Prédictions (-1 ou 1)
        """
        self.fit(X)
        return self.predict(X)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les scores d'anomalie pour chaque échantillon.
        
        Args:
            X: Données
        
        Returns:
            Scores d'anomalie
        """
        raise NotImplementedError
    
    def save(self, filepath: str):
        """
        Sauvegarde le modèle.
        
        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        joblib.dump(self, filepath)
        logger.info(f"{self.name} sauvegardé dans {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'BaseAnomalyDetector':
        """
        Charge un modèle sauvegardé.
        
        Args:
            filepath: Chemin du modèle à charger
        
        Returns:
            Modèle chargé
        """
        model = joblib.load(filepath)
        logger.info(f"Modèle chargé depuis {filepath}")
        return model


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Détecteur d'anomalies basé sur Isolation Forest.
    
    Isolation Forest détecte les anomalies en isolant les observations.
    Les anomalies sont plus faciles à isoler que les points normaux.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialise Isolation Forest.
        
        Args:
            contamination: Proportion attendue d'anomalies dans les données
            n_estimators: Nombre d'arbres
            max_samples: Nombre d'échantillons pour entraîner chaque arbre
            random_state: Seed pour reproductibilité
            **kwargs: Autres paramètres pour IsolationForest
        """
        super().__init__(name="IsolationForest")
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            **kwargs
        )
        
        self.contamination = contamination
        logger.info(f"IsolationForest initialisé (contamination={contamination}, n_estimators={n_estimators})")
    
    def fit(self, X: np.ndarray):
        """Entraîne Isolation Forest."""
        logger.info(f"Entraînement de {self.name} sur {X.shape[0]} échantillons...")
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Entraînement terminé")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les anomalies."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné avec fit()")
        
        predictions = self.model.predict(X)
        self.predictions_ = predictions
        
        n_anomalies = np.sum(predictions == -1)
        logger.info(f"Anomalies détectées : {n_anomalies}/{len(predictions)} ({n_anomalies/len(predictions)*100:.2f}%)")
        
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les scores d'anomalie (plus négatif = plus anormal).
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné avec fit()")
        
        scores = self.model.score_samples(X)
        self.anomaly_scores_ = -scores  # Inverser pour que plus haut = plus anormal
        return self.anomaly_scores_


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    Détecteur d'anomalies basé sur One-Class SVM.
    
    One-Class SVM apprend une frontière qui englobe les données normales.
    Les points en dehors de cette frontière sont considérés comme des anomalies.
    """
    
    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        **kwargs
    ):
        """
        Initialise One-Class SVM.
        
        Args:
            nu: Limite supérieure de la fraction d'erreurs d'entraînement
                et limite inférieure de la fraction de vecteurs de support
            kernel: Type de kernel ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Coefficient du kernel
            **kwargs: Autres paramètres pour OneClassSVM
        """
        super().__init__(name="OneClassSVM")
        
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            **kwargs
        )
        
        self.nu = nu
        logger.info(f"OneClassSVM initialisé (nu={nu}, kernel={kernel}, gamma={gamma})")
    
    def fit(self, X: np.ndarray):
        """Entraîne One-Class SVM."""
        logger.info(f"Entraînement de {self.name} sur {X.shape[0]} échantillons...")
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Entraînement terminé")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les anomalies."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné avec fit()")
        
        predictions = self.model.predict(X)
        self.predictions_ = predictions
        
        n_anomalies = np.sum(predictions == -1)
        logger.info(f"Anomalies détectées : {n_anomalies}/{len(predictions)} ({n_anomalies/len(predictions)*100:.2f}%)")
        
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les scores d'anomalie (distance signée à la frontière).
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné avec fit()")
        
        scores = self.model.decision_function(X)
        self.anomaly_scores_ = -scores  # Inverser pour que plus haut = plus anormal
        return self.anomaly_scores_


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Détecteur d'anomalies basé sur un Autoencodeur.
    
    L'autoencodeur apprend à reconstruire les données normales.
    Les anomalies ont une erreur de reconstruction plus élevée.
    """
    
    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_layers: Optional[list] = None,
        epochs: int = 50,
        batch_size: int = 32,
        contamination: float = 0.1,
        verbose: int = 0,
        random_state: int = 42
    ):
        """
        Initialise l'Autoencodeur.
        
        Args:
            encoding_dim: Dimension de l'espace latent
            hidden_layers: Liste des tailles des couches cachées (None = auto)
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des batchs
            contamination: Proportion d'anomalies (pour le seuil)
            verbose: Niveau de verbosité (0, 1, ou 2)
            random_state: Seed pour reproductibilité
        """
        super().__init__(name="Autoencoder")
        
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.verbose = verbose
        self.random_state = random_state
        
        self.threshold_ = None
        
        logger.info(f"Autoencodeur initialisé (encoding_dim={encoding_dim}, epochs={epochs})")
    
    def _build_model(self, input_dim: int):
        """
        Construit l'architecture de l'autoencodeur.
        
        Args:
            input_dim: Dimension des données d'entrée
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            import tensorflow as tf
            
            # Fixer le seed pour reproductibilité
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
            
        except ImportError:
            raise ImportError("TensorFlow et Keras sont requis pour l'Autoencodeur")
        
        # Définir les couches cachées par défaut si non spécifiées
        if self.hidden_layers is None:
            self.hidden_layers = [input_dim // 2, input_dim // 4]
        
        # Encoder
        encoder_layers = [layers.Input(shape=(input_dim,))]
        
        for units in self.hidden_layers:
            encoder_layers.append(layers.Dense(units, activation='relu'))
        
        encoder_layers.append(layers.Dense(self.encoding_dim, activation='relu'))
        
        # Decoder (symétrique)
        decoder_layers = []
        for units in reversed(self.hidden_layers):
            decoder_layers.append(layers.Dense(units, activation='relu'))
        
        decoder_layers.append(layers.Dense(input_dim, activation='linear'))
        
        # Modèle complet
        all_layers = encoder_layers + decoder_layers
        
        x = all_layers[0]
        for layer in all_layers[1:]:
            x = layer(x)
        
        self.model = keras.Model(inputs=encoder_layers[0], outputs=x)
        self.model.compile(optimizer='adam', loss='mse')
        
        logger.info(f"Architecture de l'autoencodeur : {input_dim} -> {self.hidden_layers} -> {self.encoding_dim} -> {self.hidden_layers[::-1]} -> {input_dim}")
    
    def fit(self, X: np.ndarray):
        """Entraîne l'Autoencodeur."""
        logger.info(f"Entraînement de {self.name} sur {X.shape[0]} échantillons...")
        
        # Construire le modèle
        self._build_model(X.shape[1])
        
        # Entraîner
        self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            validation_split=0.1
        )
        
        # Calculer le seuil basé sur l'erreur de reconstruction
        reconstructions = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Le seuil est le percentile correspondant à (1 - contamination)
        self.threshold_ = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        
        self.is_fitted = True
        logger.info(f"Entraînement terminé (seuil={self.threshold_:.4f})")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les anomalies."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné avec fit()")
        
        # Calculer les erreurs de reconstruction
        scores = self.get_anomaly_scores(X)
        
        # Comparer au seuil
        predictions = np.where(scores > self.threshold_, -1, 1)
        self.predictions_ = predictions
        
        n_anomalies = np.sum(predictions == -1)
        logger.info(f"Anomalies détectées : {n_anomalies}/{len(predictions)} ({n_anomalies/len(predictions)*100:.2f}%)")
        
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les erreurs de reconstruction (scores d'anomalie).
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné avec fit()")
        
        reconstructions = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        self.anomaly_scores_ = reconstruction_errors
        
        return self.anomaly_scores_


def compare_detectors(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    contamination: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Compare les trois détecteurs d'anomalies.
    
    Args:
        X: Données à analyser
        y_true: Vraies étiquettes si disponibles (1=normal, -1=anomalie)
        contamination: Proportion attendue d'anomalies
    
    Returns:
        Dictionnaire avec les prédictions de chaque modèle
    """
    logger.info("\n=== COMPARAISON DES DÉTECTEURS ===")
    
    results = {}
    
    # Isolation Forest
    logger.info("\n--- Isolation Forest ---")
    if_detector = IsolationForestDetector(contamination=contamination)
    if_predictions = if_detector.fit_predict(X)
    results['IsolationForest'] = if_predictions
    
    # One-Class SVM
    logger.info("\n--- One-Class SVM ---")
    svm_detector = OneClassSVMDetector(nu=contamination)
    svm_predictions = svm_detector.fit_predict(X)
    results['OneClassSVM'] = svm_predictions
    
    # Autoencodeur
    logger.info("\n--- Autoencodeur ---")
    ae_detector = AutoencoderDetector(contamination=contamination, epochs=30, verbose=0)
    ae_predictions = ae_detector.fit_predict(X)
    results['Autoencoder'] = ae_predictions
    
    # Évaluation si les vraies étiquettes sont disponibles
    if y_true is not None:
        logger.info("\n=== ÉVALUATION ===")
        
        # Convertir y_true si nécessaire (0/1 -> 1/-1)
        if set(np.unique(y_true)) == {0, 1}:
            y_true_converted = np.where(y_true == 1, -1, 1)
        else:
            y_true_converted = y_true
        
        for name, predictions in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"Matrice de confusion:\n{confusion_matrix(y_true_converted, predictions)}")
            logger.info(f"\n{classification_report(y_true_converted, predictions, target_names=['Normal', 'Anomalie'])}")
    
    return results


if __name__ == "__main__":
    # Exemple d'utilisation
    from data_loader import create_sample_dataset, handle_missing_values
    from preprocessor import preprocess_data
    
    print("Création d'un dataset synthétique...")
    df = create_sample_dataset(n_samples=1000, n_features=5, contamination=0.1)
    
    print("\nTraitement des valeurs manquantes...")
    df_cleaned = handle_missing_values(df, strategy='auto')
    
    print("\nPrétraitement des données...")
    X, preprocessor = preprocess_data(
        df_cleaned,
        exclude_columns=['id', 'true_label'],
        return_preprocessor=True
    )
    
    print("\nComparaison des détecteurs...")
    y_true = df_cleaned['true_label'].values
    results = compare_detectors(X, y_true=y_true, contamination=0.1)
