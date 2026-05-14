"""Training utilities for LOF anomaly detection."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.lof_preprocessing import clean_data, select_feature_columns, split_time_series
from features.lof_features import add_rolling_features, build_feature_matrix
from models.lof import LOFAnomalyDetector, LOFConfig


def train_lof(
	df: pd.DataFrame,
	timestamp_col: str = "timestamp",
	feature_cols: Optional[Iterable[str]] = None,
	rolling_window: int = 60,
	n_neighbors: int = 20,
	contamination: float = 0.01,
	train_frac: float = 0.7,
	val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
	"""Run the full LOF pipeline and return results, summary, and artifacts."""
	base_features = (
		list(feature_cols)
		if feature_cols is not None
		else select_feature_columns(df, timestamp_col=timestamp_col)
	)

	cleaned_df = clean_data(df, base_features, timestamp_col=timestamp_col)
	feature_df = add_rolling_features(cleaned_df, base_features, window=rolling_window)
	X_df, feature_columns = build_feature_matrix(feature_df, base_features)

	train_df, val_df, test_df = split_time_series(feature_df, train_frac, val_frac)
	train_end = len(train_df)
	val_end = train_end + len(val_df)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_df.iloc[:train_end])
	X_all = scaler.transform(X_df)

	detector = LOFAnomalyDetector(LOFConfig(n_neighbors=n_neighbors, contamination=contamination))
	detector.fit(X_train)
	scores = detector.score_samples(X_all)
	labels = (scores >= detector.threshold_).astype(int)

	results_df = feature_df.copy()
	results_df["anomaly_score"] = scores
	results_df["is_anomaly"] = labels

	summary_df = summarize_anomalies(results_df)

	artifacts = {
		"model": detector,
		"scaler": scaler,
		"base_features": base_features,
		"feature_columns": feature_columns,
		"threshold": detector.threshold_,
		"train_end": train_end,
		"val_end": val_end,
	}

	return results_df, summary_df, artifacts


def summarize_anomalies(df: pd.DataFrame, anomaly_col: str = "is_anomaly") -> pd.DataFrame:
	"""Summarize anomaly counts and rates."""
	total_rows = len(df)
	anomaly_count = int(df[anomaly_col].sum()) if total_rows > 0 else 0
	anomaly_rate = anomaly_count / total_rows if total_rows > 0 else 0.0

	return pd.DataFrame(
		{
			"total_rows": [total_rows],
			"anomalies": [anomaly_count],
			"anomaly_rate": [anomaly_rate],
		}
	)


def top_anomalies(
	df: pd.DataFrame,
	n: int = 10,
	timestamp_col: str = "timestamp",
	score_col: str = "anomaly_score",
) -> pd.DataFrame:
	"""Return the top-N anomalies sorted by score."""
	columns = [col for col in [timestamp_col, score_col, "is_anomaly"] if col in df.columns]
	preview = df.sort_values(score_col, ascending=False).head(n)
	if columns:
		return preview[columns]
	return preview.head(n)
