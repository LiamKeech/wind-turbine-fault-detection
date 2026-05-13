from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class WindowedFeatures:
    """
    Container for windowed features used in LSTM autoencoder training.
    """    

    windows: np.ndarray
    window_start_indices: np.ndarray
    window_end_timestamps: Optional[np.ndarray]


@dataclass(frozen=True)
class WindowedSplits:
    """
    Container for windowed features for train/validation/test splits.
    """    

    train: WindowedFeatures
    val: WindowedFeatures
    test: WindowedFeatures


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from multivariate time series data.

    Args:
        data (np.ndarray): Input data of shape (n_samples, n_features).
        window_size (int): Length of each sequence window.
        step_size (int): Stride between windows.

    Raises:
        ValueError: If window_size is less than 1.
        ValueError: If step_size is less than 1.
        ValueError: If data does not have the correct shape.
        ValueError: If window_size is greater than the number of samples.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of shape (n_windows, window_size, n_features) and start indices.
    """    
    
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")
    if data.ndim != 2:
        raise ValueError("data must have shape (n_samples, n_features).")

    n_samples = data.shape[0]
    if window_size > n_samples:
        raise ValueError("window_size must be <= number of samples.")
    n_windows = 1 + (n_samples - window_size) // step_size

    windows = np.empty((n_windows, window_size, data.shape[1]), dtype=data.dtype)
    start_indices = np.empty(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]
        start_indices[i] = start

    return windows, start_indices

def build_windowed_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    window_size: int,
    step_size: int,
    *,
    timestamp_col: str = "timestamp",
) -> WindowedFeatures:
    """
    Build windowed features from a dataframe for LSTM autoencoder training.

    Args:
        df (pd.DataFrame): Input dataframe with scaled features.
        feature_columns (Sequence[str]): Columns to include in the window tensors.
        window_size (int): Length of each sequence window.
        step_size (int): Stride between windows.
        timestamp_col (str, optional): Timestamp column name used to record window end times. Defaults to "timestamp".

    Raises:
        ValueError: If any feature column is not found in the dataframe.
        ValueError: If window_size is less than 1.
        ValueError: If step_size is less than 1.
        ValueError: If data does not have the correct shape.
        ValueError: If window_size is greater than the number of samples.

    Returns:
        WindowedFeatures: Container for windowed features.
    """    
    
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    data = df[list(feature_columns)].to_numpy(dtype=np.float32)
    windows, start_indices = create_sliding_windows(data, window_size, step_size)

    window_end_timestamps = None
    if timestamp_col in df.columns:
        end_indices = start_indices + window_size - 1
        window_end_timestamps = df[timestamp_col].iloc[end_indices].to_numpy()

    return WindowedFeatures(
        windows=windows,
        window_start_indices=start_indices,
        window_end_timestamps=window_end_timestamps,
    )

def build_windowed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window_size: int = 60,
    step_size: int = 5,
    timestamp_col: str = "timestamp",
) -> WindowedSplits:
    """
    Build windowed features for train/validation/test splits.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data.
        val_df (pd.DataFrame): DataFrame containing validation data.
        test_df (pd.DataFrame): DataFrame containing test data.
        feature_columns (Sequence[str]): Columns to include in the window tensors.
        window_size (int, optional): Length of each sequence window. Defaults to 60.
        step_size (int, optional): Stride between windows. Defaults to 5.
        timestamp_col (str, optional): Timestamp column name used to record window end times. Defaults to "timestamp".

    Returns:
        WindowedSplits: Container for windowed features for each split.
    """    
    
    return WindowedSplits(
        train=build_windowed_features(
            train_df,
            feature_columns,
            window_size,
            step_size,
            timestamp_col=timestamp_col,
        ),
        val=build_windowed_features(
            val_df,
            feature_columns,
            window_size,
            step_size,
            timestamp_col=timestamp_col,
        ),
        test=build_windowed_features(
            test_df,
            feature_columns,
            window_size,
            step_size,
            timestamp_col=timestamp_col,
        ),
    )