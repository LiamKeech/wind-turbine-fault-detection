from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from data.lof_preprocessing import clean_data, select_feature_columns, split_time_series
from features.lof_features import add_rolling_features, build_feature_matrix
from models.lof import LOFAnomalyDetector, LOFConfig
import pandas as pd

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
	"""
	Train a Local Outlier Factor model and generate anomaly predictions.

	Args:
		df (pd.DataFrame): Input DataFrame with features and optional timestamp column.
		timestamp_col (str, optional): Name of the timestamp column. Defaults to "timestamp".
		feature_cols (Optional[Iterable[str]]): Feature columns to use. If None, all numeric columns are selected. Defaults to None.
		rolling_window (int, optional): Window size for rolling statistics. Defaults to 60.
		n_neighbors (int, optional): Number of neighbors for LOF algorithm. Defaults to 20.
		contamination (float, optional): Expected proportion of outliers. Defaults to 0.01.
		train_frac (float, optional): Fraction of data for training. Defaults to 0.7.
		val_frac (float, optional): Fraction of data for validation. Defaults to 0.15.

	Returns:
		Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]: Results DataFrame with predictions, summary statistics, and model artifacts.
	"""
 
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
	"""
	Summarize anomaly statistics for a DataFrame.

	Args:
		df (pd.DataFrame): DataFrame containing anomaly predictions.
		anomaly_col (str, optional): Name of the anomaly column. Defaults to "is_anomaly".

	Returns:
		pd.DataFrame: Summary DataFrame with total_rows, anomalies count, and anomaly_rate.
	"""
 
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
	"""
	Retrieve the top N anomalies by score.

	Args:
		df (pd.DataFrame): DataFrame containing anomaly scores.
		n (int, optional): Number of top anomalies to return. Defaults to 10.
		timestamp_col (str, optional): Name of the timestamp column. Defaults to "timestamp".
		score_col (str, optional): Name of the anomaly score column. Defaults to "anomaly_score".

	Returns:
		pd.DataFrame: DataFrame with top N anomalies, showing timestamp, score, and anomaly label columns.
	"""
 
	columns = [col for col in [timestamp_col, score_col, "is_anomaly"] if col in df.columns]
	preview = df.sort_values(score_col, ascending=False).head(n)
	if columns:
		return preview[columns]
	return preview.head(n)