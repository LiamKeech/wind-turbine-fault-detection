"""Local Outlier Factor (LOF) model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass
class LofDetector:
    """Local Outlier Factor wrapper with scaling and persistence.

    Args:
        n_neighbors: Number of neighbors used for LOF.
        contamination: Expected fraction of anomalies in the data.
        novelty: Enables scoring of new, unseen samples.
        scaler: Optional pre-fit scaler for features.
        model: Optional pre-fit LOF model.
    """

    n_neighbors: int = 20
    contamination: float = 0.05
    novelty: bool = True
    scaler: StandardScaler | None = None
    model: LocalOutlierFactor | None = None

    def fit(self, X: np.ndarray) -> "LofDetector":
        """Fit the scaler and LOF model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Self for chaining.
        """
        self.scaler = self.scaler or StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty,
        )
        self.model.fit(X_scaled)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores where higher means more anomalous.
        """
        self._ensure_fitted()
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)
        return np.asarray(scores)

    def predict(self, X: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores and flags.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            threshold: Score threshold for anomaly flagging.

        Returns:
            Tuple of scores and binary flags (1=anomaly, 0=normal).
        """
        scores = self.score_samples(X)
        flags = (scores > threshold).astype(int)
        return scores, flags

    def save(self, path: Path) -> None:
        """Persist the detector to disk.

        Args:
            path: File path to save the model artifact.

        Returns:
            None
        """
        self._ensure_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: Path) -> "LofDetector":
        """Load a detector from disk.

        Args:
            path: File path to a saved detector.

        Returns:
            Loaded LofDetector instance.
        """
        payload = joblib.load(path)
        return cls(
            n_neighbors=payload["model"].n_neighbors,
            contamination=payload["model"].contamination,
            novelty=payload["model"].novelty,
            scaler=payload["scaler"],
            model=payload["model"],
        )

    @staticmethod
    def compute_threshold(scores: np.ndarray, quantile: float = 0.95) -> float:
        """Compute a score threshold based on a quantile.

        Args:
            scores: Array of anomaly scores.
            quantile: Quantile value in the range (0, 1).

        Returns:
            Threshold value.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")
        return float(np.quantile(scores, quantile))

    @staticmethod
    def select_features(
        data: pd.DataFrame, feature_columns: Iterable[str]
    ) -> np.ndarray:
        """Select feature columns as a numpy array.

        Args:
            data: Input DataFrame.
            feature_columns: Iterable of feature column names.

        Returns:
            Feature matrix as numpy array.
        """
        return data.loc[:, list(feature_columns)].to_numpy()

    def _ensure_fitted(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("LofDetector must be fitted before use.")
