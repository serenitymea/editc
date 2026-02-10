import joblib
import numpy as np
from typing import Iterable, Optional

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARN] CatBoost not available, falling back to LogisticRegression")

from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    """
    Trains a binary classifier from feature vectors.
    Uses CatBoost if available, otherwise LogisticRegression.
    """

    def __init__(self, use_catboost=True):
        self.use_catboost = use_catboost and CATBOOST_AVAILABLE
        
        if self.use_catboost:
            self.model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function='Logloss',
                eval_metric='AUC',
                early_stopping_rounds=50,
                verbose=False,
                random_state=42
            )
            print("[ModelTrainer] Using CatBoost")
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
            print("[ModelTrainer] Using LogisticRegression")

    def train(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[int],
        X_val: Optional[Iterable[Iterable[float]]] = None,
        y_val: Optional[Iterable[int]] = None
    ) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features (for CatBoost early stopping)
            y_val: Optional validation labels
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if len(X) == 0:
            raise ValueError("Empty training set")
        
        print(f"[ModelTrainer] Training on {len(X)} samples, {len(X[0])} features")
        print(f"[ModelTrainer] Class distribution: {np.bincount(y)}")

        if self.use_catboost:

            if X_val is not None and y_val is not None:
                X_val = np.asarray(X_val, dtype=np.float32)
                y_val = np.asarray(y_val, dtype=np.int32)
                
                self.model.fit(
                    X, y,
                    eval_set=(X_val, y_val),
                    verbose=50
                )
            else:
                self.model.fit(X, y)
        else:

            self.model.fit(X, y)
        
        print("[ModelTrainer] Training complete")

        self._print_feature_importance()

    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels"""
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"[ModelTrainer] Model saved to {path}")

    def load(self, model) -> None:
        """Load model from disk or object"""
        if isinstance(model, str):

            self.model = joblib.load(model)
            print(f"[ModelTrainer] Model loaded from {model}")
        elif isinstance(model, (CatBoostClassifier, LogisticRegression)) if CATBOOST_AVAILABLE else isinstance(model, LogisticRegression):

            self.model = model
            print(f"[ModelTrainer] Model loaded from object")
        else:
            print(f"[WARN] Invalid model type: {type(model)}")
    
    def _print_feature_importance(self):
        """Print top feature importances if available"""
        try:
            if self.use_catboost:
                importance = self.model.get_feature_importance()
                feature_names = self._get_feature_names()
                
                if len(feature_names) == len(importance):
                    sorted_features = sorted(
                        zip(feature_names, importance),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    print("\n[ModelTrainer] Top 10 Feature Importances:")
                    for name, imp in sorted_features[:10]:
                        print(f"  {name:30s}: {imp:6.2f}")
            elif hasattr(self.model, 'coef_'):

                coef = np.abs(self.model.coef_[0])
                feature_names = self._get_feature_names()
                
                if len(feature_names) == len(coef):
                    sorted_features = sorted(
                        zip(feature_names, coef),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    print("\n[ModelTrainer] Top 10 Feature Coefficients:")
                    for name, imp in sorted_features[:10]:
                        print(f"  {name:30s}: {imp:6.4f}")
        except Exception as e:
            print(f"[WARN] Could not print feature importance: {e}")
    
    def _get_feature_names(self):
        """Get standard feature names"""
        return [
            # Motion (5)
            "motion_mean", "motion_max", "motion_p90", "motion_std", "motion_peak_ratio",
            
            # Audio (10)
            "rms_mean", "rms_peak", "rms_contrast",
            "spectral_centroid", "spectral_rolloff", "mfcc_variance",
            "bass_energy", "vocal_probability", "onset_density",
            "beat_alignment_score",
            
            # Visual (10)
            "blur_score", "edge_density", "color_variance",
            "avg_brightness", "contrast_mean",
            "face_present", "face_size_ratio",
            "symmetry", "rule_of_thirds", "text_presence",
            
            # Context (5)
            "motion_momentum", "audio_momentum", "combined_buildup",
            "relative_position", "motion_derivative",
            
            # Other
            "duration"
        ]