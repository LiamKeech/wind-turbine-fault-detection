from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import pandas as pd

def add_rolling_features(
	df: pd.DataFrame,
	feature_cols: Iterable[str],
	window: int = 60,
	min_periods: Optional[int] = None,
) -> pd.DataFrame:
	"""
	Add rolling mean and standard deviation features to a DataFrame.

	Args:
		df (pd.DataFrame): The DataFrame to add rolling features to.
		feature_cols (Iterable[str]): Column names to compute rolling statistics for.
		window (int, optional): Size of the rolling window. Defaults to 60.
		min_periods (Optional[int]): Minimum number of observations in window required. Defaults to None (uses window size).

	Returns:
		pd.DataFrame: DataFrame with rolling features added and missing values removed.
	"""
 
	if min_periods is None:
		min_periods = window

	df_features = df.copy()
	for column in feature_cols:
		if column not in df_features.columns:
			continue
		rolling = df_features[column].rolling(window=window, min_periods=min_periods)
		df_features[f"{column}_roll_mean"] = rolling.mean()
		df_features[f"{column}_roll_std"] = rolling.std()

	return df_features.dropna().reset_index(drop=True)

def build_feature_matrix(
	df: pd.DataFrame,
	base_features: Iterable[str],
) -> Tuple[pd.DataFrame, List[str]]:
	"""
	Build a feature matrix by combining base features with their rolling statistics.

	Args:
		df (pd.DataFrame): DataFrame containing base features and their rolling statistics.
		base_features (Iterable[str]): Names of the base feature columns.

	Returns:
		Tuple[pd.DataFrame, List[str]]: Feature matrix DataFrame and list of feature column names.
	"""
 
	base_features = [col for col in base_features if col in df.columns]
	rolling_mean = [f"{col}_roll_mean" for col in base_features if f"{col}_roll_mean" in df.columns]
	rolling_std = [f"{col}_roll_std" for col in base_features if f"{col}_roll_std" in df.columns]
	feature_columns = [*base_features, *rolling_mean, *rolling_std]

	return df[feature_columns], feature_columns