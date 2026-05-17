from typing import List, Tuple
import numpy as np
import pandas as pd

BASE_FEATURES = [
	"gearbox_oil_temp",
	"gearbox_bearing_temp",
	"vibration_x",
	"vibration_y",
	"vibration_z",
	"oil_pressure",
]

ENGINEERED_FEATURES = [
	"temp_diff",
	"vibration_rms",
	"vibration_mag",
	"oil_pressure_delta",
	"particle_count_delta",
	"hour_sin",
	"hour_cos",
	"month_sin",
	"month_cos",
]

def _clean_particle_delta(
	delta: pd.Series,
	reset_quantile: float,
	clip_quantile: float,
) -> pd.Series:
	"""
	Clean particle count delta series by resetting negative spikes and clipping outliers.

	Args:
		delta (pd.Series): Particle count delta series to clean.
		reset_quantile (float): Quantile threshold for resetting negative deltas. Must be between 0 and 1.
		clip_quantile (float): Quantile threshold for clipping extreme values. Must be in (0, 1].

	Returns:
		pd.Series: Cleaned particle delta series with NaN values filled as 0.
	"""
 
	cleaned = delta.copy()

	if not 0 < reset_quantile < 1:
		raise ValueError("reset_quantile must be between 0 and 1.")
	neg_delta = cleaned[cleaned < 0]
	if not neg_delta.empty:
		reset_threshold = neg_delta.quantile(reset_quantile)
		cleaned = cleaned.mask(cleaned < reset_threshold, 0.0)

	if not 0 < clip_quantile <= 1:
		raise ValueError("clip_quantile must be in (0, 1].")
	clip_value = cleaned.abs().quantile(clip_quantile)
	if clip_value > 0:
		cleaned = cleaned.clip(-clip_value, clip_value)

	return cleaned.fillna(0.0)

def engineer_features(
	df: pd.DataFrame,
	particle_reset_quantile: float = 0.01,
	particle_delta_clip_quantile: float = 0.99,
) -> Tuple[pd.DataFrame, List[str]]:
	"""
	Engineer features from raw turbine sensor data.

	Args:
		df (pd.DataFrame): Raw DataFrame containing base features and timestamp column.
		particle_reset_quantile (float): Quantile for resetting negative particle count deltas. Default is 0.01.
		particle_delta_clip_quantile (float): Quantile for clipping particle count delta outliers. Default is 0.99.

	Returns:
		Tuple[pd.DataFrame, List[str]]: DataFrame with engineered features and list of all feature column names.
	"""
 
	features = df.copy()

	features["temp_diff"] = features["gearbox_bearing_temp"] - features["gearbox_oil_temp"]
	features["vibration_rms"] = np.sqrt(
		(
			features["vibration_x"] ** 2
			+ features["vibration_y"] ** 2
			+ features["vibration_z"] ** 2
		)
		/ 3.0
	)
	features["vibration_mag"] = np.sqrt(
		features["vibration_x"] ** 2
		+ features["vibration_y"] ** 2
		+ features["vibration_z"] ** 2
	)
	features["oil_pressure_delta"] = features["oil_pressure"].diff().fillna(0.0)
	particle_delta = features["particle_count"].diff().fillna(0.0)
	features["particle_count_delta"] = _clean_particle_delta(
		particle_delta,
		particle_reset_quantile,
		particle_delta_clip_quantile,
	)
	timestamps = pd.to_datetime(features["timestamp"], errors="coerce")
	hour = timestamps.dt.hour + (timestamps.dt.minute / 60.0)
	month = timestamps.dt.month.astype(float)
	features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
	features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
	features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
	features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

	feature_cols = BASE_FEATURES + ENGINEERED_FEATURES
	return features, feature_cols