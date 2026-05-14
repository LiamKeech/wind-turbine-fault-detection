"""Basic preprocessing utilities for LOF anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_raw_data(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load a CSV file and parse the timestamp column when available."""
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
    """Choose numeric feature columns, optionally constrained by include/exclude lists."""
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
    """Sort, de-duplicate, coerce numeric types, and drop rows with missing features."""
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
    """Split a time-ordered dataframe into train/val/test segments."""
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
    """Return a compact overview of rows, feature count, and time span."""
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
    """Summarize missing counts and rates for selected columns."""
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