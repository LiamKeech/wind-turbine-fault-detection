from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

@dataclass
class LOFConfig:
	"""
	Configuration for Local Outlier Factor anomaly detection.

	Attributes:
		n_neighbors (int): Number of neighbors to consider. Defaults to 20.
		contamination (float): Expected proportion of outliers. Defaults to 0.01.
		metric (str): Distance metric to use. Defaults to "minkowski".
	"""
 
	n_neighbors: int = 20
	contamination: float = 0.01
	metric: str = "minkowski"

class LOFAnomalyDetector:

	def __init__(self, config: Optional[LOFConfig] = None) -> None:
		"""
		Initialize LOF anomaly detector.

		Args:
			config (Optional[LOFConfig]): Configuration object for LOF parameters. If None, default configuration is used. Defaults to None.

		Returns:
			None
		"""
  
		self.config = config or LOFConfig()
		self.model = LocalOutlierFactor(
			n_neighbors=self.config.n_neighbors,
			contamination=self.config.contamination,
			metric=self.config.metric,
			novelty=True,
		)
		self.threshold_: Optional[float] = None

	def fit(self, X: np.ndarray) -> "LOFAnomalyDetector":
		"""
		Fit the LOF model on training data.

		Args:
			X (np.ndarray): Training data of shape (n_samples, n_features).

		Returns:
			LOFAnomalyDetector: The fitted detector instance.
		"""
  
		self.model.fit(X)
		train_scores = -self.model.score_samples(X)
		self.threshold_ = float(np.quantile(train_scores, 1 - self.config.contamination))
		return self

	def score_samples(self, X: np.ndarray) -> np.ndarray:
		"""
		Compute anomaly scores for samples.

		Args:
			X (np.ndarray): Data of shape (n_samples, n_features).

		Returns:
			np.ndarray: Anomaly scores for each sample. Higher scores indicate more anomalous samples.
		"""
  
		return -self.model.score_samples(X)

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict anomaly labels for samples.

		Args:
			X (np.ndarray): Data of shape (n_samples, n_features).

		Raises:
			ValueError: If the model has not been fitted before calling predict.

		Returns:
			np.ndarray: Binary labels (0=normal, 1=anomaly) for each sample.
		"""
  
		if self.threshold_ is None:
			raise ValueError("Model must be fitted before calling predict.")
		scores = self.score_samples(X)
		return (scores >= self.threshold_).astype(int)