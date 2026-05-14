from pathlib import Path
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_raw_data(csv_path: Path, timestamp_col: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if timestamp_col in df.columns:
		df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
	return df


def clean_raw_data(
	df: pd.DataFrame,
	timestamp_col: str,
	label_col: Optional[str] = None,
) -> pd.DataFrame:
	cleaned = df.copy()
	if timestamp_col in cleaned.columns:
		cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
		cleaned = cleaned.dropna(subset=[timestamp_col])
		cleaned = cleaned.sort_values(timestamp_col)
		cleaned = cleaned.drop_duplicates(subset=[timestamp_col], keep="last")

	if label_col is None and "is_anomaly" in cleaned.columns:
		label_col = "is_anomaly"
	numeric_cols = [
		col
		for col in cleaned.columns
		if col not in {timestamp_col, label_col}
	]
	cleaned[numeric_cols] = cleaned[numeric_cols].apply(pd.to_numeric, errors="coerce")
	cleaned = cleaned.reset_index(drop=True)
	if numeric_cols:
		cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(method="linear").ffill().bfill()
	return cleaned


def split_train_val(
	df: pd.DataFrame,
	val_fraction: float,
	timestamp_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if not 0 < val_fraction < 1:
		raise ValueError("val_fraction must be between 0 and 1.")
	if timestamp_col is None and "timestamp" in df.columns:
		timestamp_col = "timestamp"
	ordered = df.sort_values(timestamp_col) if timestamp_col in df.columns else df.sort_index()
	ordered = ordered.reset_index(drop=True)
	split_idx = int(len(ordered) * (1 - val_fraction))
	return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def fit_scaler(
	df: pd.DataFrame,
	feature_cols: Iterable[str],
	label_col: Optional[str] = None,
	normal_value: int = 0,
) -> StandardScaler:
	fit_df = df
	if label_col is None and "is_anomaly" in df.columns:
		label_col = "is_anomaly"
	if label_col and label_col in df.columns:
		fit_df = df.loc[df[label_col] == normal_value]
		if fit_df.empty:
			raise ValueError("No normal samples available for scaler fitting.")
	scaler = StandardScaler()
	scaler.fit(fit_df[list(feature_cols)].values)
	return scaler


def scale_features(df: pd.DataFrame, feature_cols: Iterable[str], scaler: StandardScaler) -> np.ndarray:
	values = scaler.transform(df[list(feature_cols)].values)
	return values.astype("float32")


def create_sequences(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
	if window_size <= 0 or stride <= 0:
		raise ValueError("window_size and stride must be positive.")
	if values.shape[0] < window_size:
		raise ValueError("Not enough rows to build one sequence.")

	num_sequences = 1 + (values.shape[0] - window_size) // stride
	sequences = np.zeros((num_sequences, window_size, values.shape[1]), dtype=np.float32)
	for i in range(num_sequences):
		start = i * stride
		sequences[i] = values[start:start + window_size]
	return sequences


def save_processed_data(
	df: pd.DataFrame,
	output_dir: Path,
	filename: str = "processed_features.csv",
) -> Path:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / filename
	df.to_csv(output_path, index=False)
	return output_path


def save_scaler(scaler: StandardScaler, output_path: Path) -> Path:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(scaler, output_path)
	return output_path
