from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from sklearn.preprocessing import StandardScaler
import pandas as pd

@dataclass(frozen=True)
class AutoencoderPreprocessResult:
    """
    Result of preprocessing for LSTM autoencoder training, including train/val/test splits and fitted scaler.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: List[str]
    scaler: StandardScaler

def preprocess_autoencoder_data(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    feature_columns: Optional[Sequence[str]] = None,
    exclude_columns: Optional[Sequence[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    missing_row_threshold: float = 0.2,
    fill_method: Optional[str] = "ffill_bfill",
) -> AutoencoderPreprocessResult:
    """
    Preprocess raw time series data for LSTM autoencoder training.
    
    Process:
    1. Validate input parameters and timestamp column.
    2. Sort data by timestamp and remove duplicates.
    3. Select feature columns based on user input and exclusions.
    4. Handle missing values by dropping rows with too many missing features and applying fill methods.
    5. Split data into train/validation/test sets based on time.
    6. Scale features using StandardScaler fitted on the training set.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        timestamp_col (str, optional): Name of the timestamp column. Defaults to "timestamp".
        feature_columns (Optional[Sequence[str]], optional): List of column names to include as features. Defaults to None.
        exclude_columns (Optional[Sequence[str]], optional): List of column names to exclude from features. Defaults to None.
        train_ratio (float, optional): Proportion of data to use for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of data to use for validation. Defaults to 0.15.
        missing_row_threshold (float, optional): Maximum proportion of missing values allowed in a row. Defaults to 0.2.
        fill_method (Optional[str], optional): Method for filling missing values. Defaults to "ffill_bfill".

    Raises:
        ValueError: If input parameters are invalid.
        ValueError: If the timestamp column is missing.

    Returns:
        AutoencoderPreprocessResult: Preprocessed data and fitted scaler.
    """    
    
    _validate_split_ratios(train_ratio, val_ratio)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    working = df.copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working = working.dropna(subset=[timestamp_col])
    working = working.sort_values(timestamp_col)
    working = working.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)

    features = _select_feature_columns(
        working,
        timestamp_col,
        feature_columns,
        exclude_columns,
    )

    if missing_row_threshold is not None:
        if not 0.0 <= missing_row_threshold <= 1.0:
            raise ValueError("missing_row_threshold must be in [0.0, 1.0].")
        row_missing = working[features].isna().mean(axis=1)
        working = working.loc[row_missing <= missing_row_threshold].reset_index(drop=True)

    working = _fill_missing_values(working, features, fill_method)
    working = working.dropna(subset=features).reset_index(drop=True)

    train_df, val_df, test_df = _split_by_time(working, train_ratio, val_ratio)

    scaler = StandardScaler()
    train_df = _apply_scaler(train_df, features, scaler, fit=True)
    val_df = _apply_scaler(val_df, features, scaler, fit=False)
    test_df = _apply_scaler(test_df, features, scaler, fit=False)

    return AutoencoderPreprocessResult(
        train=train_df,
        val=val_df,
        test=test_df,
        feature_columns=list(features),
        scaler=scaler,
    )

def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    """
    Validate the train and validation split ratios.

    Args:
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.

    Raises:
        ValueError: If train_ratio or val_ratio are not between 0 and 1.
        ValueError: If train_ratio + val_ratio is not less than 1.
    """    
    
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

def _select_feature_columns(
    df: pd.DataFrame,
    timestamp_col: str,
    feature_columns: Optional[Sequence[str]],
    exclude_columns: Optional[Sequence[str]],
) -> List[str]:
    """
    Select feature columns based on user input and exclusions.

    Args:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): Name of the timestamp column.
        feature_columns (Optional[Sequence[str]]): List of feature column names to include.
        exclude_columns (Optional[Sequence[str]]): List of column names to exclude.

    Raises:
        ValueError: If any specified feature columns are not found in the DataFrame.
        ValueError: If no feature columns remain after exclusions.
        ValueError: If no numeric feature columns are found after exclusions.

    Returns:
        List[str]: List of selected feature column names.
    """    
    
    exclusions = set(exclude_columns or [])
    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        selected = [col for col in feature_columns if col not in exclusions]
        if not selected:
            raise ValueError("No feature columns remain after exclusions.")
        return selected

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    selected = [
        col for col in numeric_cols if col != timestamp_col and col not in exclusions
    ]
    if not selected:
        raise ValueError("No numeric feature columns found after exclusions.")
    return selected

def _fill_missing_values(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    fill_method: Optional[str],
) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame using the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (Sequence[str]): List of column names to fill missing values for.
        fill_method (Optional[str]): Method to use for filling missing values.

    Raises:
        ValueError: If an invalid fill_method is specified.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """    
    
    if fill_method is None or fill_method == "none":
        return df

    working = df.copy()
    if fill_method == "ffill":
        working[feature_columns] = working[feature_columns].ffill()
    elif fill_method == "bfill":
        working[feature_columns] = working[feature_columns].bfill()
    elif fill_method == "ffill_bfill":
        working[feature_columns] = working[feature_columns].ffill().bfill()
    elif fill_method == "interpolate":
        working[feature_columns] = working[feature_columns].interpolate(
            method="linear",
            limit_direction="both",
        )
    else:
        raise ValueError(
            "fill_method must be one of: ffill, bfill, ffill_bfill, interpolate, none."
        )
    return working

def _split_by_time(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train, validation, and test sets based on time.

    Args:
        df (pd.DataFrame): Input DataFrame sorted by timestamp.
        train_ratio (float): Ratio of samples to include in the training set.
        val_ratio (float): Ratio of samples to include in the validation set.

    Raises:
        ValueError: If the DataFrame does not have enough samples or if the split ratios are invalid.
        ValueError: If the resulting splits are empty or if the validation set does not come after the training set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Split DataFrames for training, validation, and testing.
    """    
    
    n_samples = len(df)
    if n_samples < 3:
        raise ValueError("Not enough samples to split into train/val/test.")

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    if train_end == 0 or val_end <= train_end or val_end >= n_samples:
        raise ValueError("Invalid split sizes; adjust train_ratio and val_ratio.")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

def _apply_scaler(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    scaler: StandardScaler,
    *,
    fit: bool,
) -> pd.DataFrame:
    """
    Apply a StandardScaler to the specified feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (Sequence[str]): List of column names to scale.
        scaler (StandardScaler): The scaler to apply.
        fit (bool): Whether to fit the scaler on the data.

    Returns:
        pd.DataFrame: DataFrame with scaled feature columns
    """    
    
    working = df.copy()
    if fit:
        working[feature_columns] = scaler.fit_transform(working[feature_columns])
    else:
        working[feature_columns] = scaler.transform(working[feature_columns])
    return working