from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

def load_raw_data(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Load raw data from a CSV file.

    Args:
        path (Path): The path to the CSV file.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to "timestamp".

    Raises:
        FileNotFoundError: If the specified file is not found.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """    
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    header = pd.read_csv(path, nrows=0).columns
    parse_dates = [timestamp_col] if timestamp_col in header else None
    df = pd.read_csv(path, parse_dates=parse_dates)

    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
        df = df.drop_duplicates(subset=timestamp_col)

    return df.reset_index(drop=True)

def select_feature_columns(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    include_cols: Optional[Iterable[str]] = None,
    exclude_cols: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Select feature columns from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to select columns from.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to "timestamp".
        include_cols (Optional[Iterable[str]]): Specific columns to include. If None, all numeric columns are selected. Defaults to None.
        exclude_cols (Optional[Iterable[str]]): Columns to exclude from the selection. Defaults to None.

    Returns:
        List[str]: A list of selected feature column names.
    """
    
    if include_cols is not None:
        columns = [col for col in include_cols if col in df.columns]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    columns = [col for col in columns if col != timestamp_col]
    if exclude_cols:
        columns = [col for col in columns if col not in set(exclude_cols)]

    return columns

def clean_data(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Clean data by sorting, removing duplicates, and handling missing values.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        feature_cols (Iterable[str]): Feature columns to process for missing values.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to "timestamp".

    Returns:
        pd.DataFrame: The cleaned DataFrame with reset index.
    """
    
    df_clean = df.copy()

    if timestamp_col in df_clean.columns:
        df_clean = df_clean.sort_values(timestamp_col)
        df_clean = df_clean.drop_duplicates(subset=timestamp_col)

    feature_cols = [col for col in feature_cols if col in df_clean.columns]
    if feature_cols:
        df_clean[feature_cols] = df_clean[feature_cols].apply(pd.to_numeric, errors="coerce")
        df_clean = df_clean.dropna(subset=feature_cols)

    return df_clean.reset_index(drop=True)

def split_time_series(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time series DataFrame into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_frac (float, optional): Fraction of data for training. Defaults to 0.7.
        val_frac (float, optional): Fraction of data for validation. Defaults to 0.15.

    Raises:
        ValueError: If train_frac and val_frac do not satisfy 0 < train_frac + val_frac < 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test DataFrames.
    """
    
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must satisfy 0 < train_frac + val_frac < 1")

    n_samples = len(df)
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def dataset_overview(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Generate an overview of the dataset including row count, feature count, and time range.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        feature_cols (Optional[Iterable[str]]): Feature columns to include. If None, all numeric columns are selected. Defaults to None.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to "timestamp".

    Returns:
        pd.DataFrame: A DataFrame containing rows, features, start, and end columns.
    """
    
    if feature_cols is None:
        feature_cols = select_feature_columns(df, timestamp_col=timestamp_col)
    else:
        feature_cols = [col for col in feature_cols if col in df.columns]

    if timestamp_col in df.columns:
        start = df[timestamp_col].min()
        end = df[timestamp_col].max()
    else:
        start = pd.NaT
        end = pd.NaT

    return pd.DataFrame(
        {
            "rows": [df.shape[0]],
            "features": [len(feature_cols)],
            "start": [start],
            "end": [end],
        }
    )

def missing_rate_summary(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    top_n: Optional[int] = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Generate a summary of missing values for each column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        feature_cols (Optional[Iterable[str]]): Columns to check for missing values. If None, all columns are checked. Defaults to None.
        top_n (Optional[int]): Return only the top N columns with highest missing rate. Defaults to None.
        sort (bool, optional): Whether to sort by missing rate in descending order. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with missing_count and missing_rate for each column.
    """
    
    if feature_cols is None:
        columns = df.columns
    else:
        columns = [col for col in feature_cols if col in df.columns]

    missing_count = df[columns].isna().sum()
    missing_rate = df[columns].isna().mean()
    summary = pd.DataFrame(
        {"missing_count": missing_count, "missing_rate": missing_rate}
    )
    summary.index.name = "column"
    if sort:
        summary = summary.sort_values("missing_rate", ascending=False)
    if top_n is not None:
        summary = summary.head(top_n)
    return summary