import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Iterable, Tuple


class ModelTrainer:
    """
    Trains a binary classifier from feature vectors.
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    def train(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[int]
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if len(X) == 0:
            raise ValueError("Empty training set")

        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
