"""LOF model wrapper for anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class LOFConfig:
	n_neighbors: int = 20
	contamination: float = 0.01
	metric: str = "minkowski"


class LOFAnomalyDetector:
	"""Small wrapper around LocalOutlierFactor for novelty detection."""

	def __init__(self, config: Optional[LOFConfig] = None) -> None:
		self.config = config or LOFConfig()
		self.model = LocalOutlierFactor(
			n_neighbors=self.config.n_neighbors,
			contamination=self.config.contamination,
			metric=self.config.metric,
			novelty=True,
		)
		self.threshold_: Optional[float] = None

	def fit(self, X: np.ndarray) -> "LOFAnomalyDetector":
		"""Fit the LOF model and set the anomaly threshold from training scores."""
		self.model.fit(X)
		train_scores = -self.model.score_samples(X)
		self.threshold_ = float(np.quantile(train_scores, 1 - self.config.contamination))
		return self

	def score_samples(self, X: np.ndarray) -> np.ndarray:
		"""Return anomaly scores where higher values indicate more anomalous points."""
		return -self.model.score_samples(X)

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict anomaly labels (1 = anomaly, 0 = normal)."""
		if self.threshold_ is None:
			raise ValueError("Model must be fitted before calling predict.")
		scores = self.score_samples(X)
		return (scores >= self.threshold_).astype(int)
