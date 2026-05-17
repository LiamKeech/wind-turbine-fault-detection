from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def select_top_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    n_features: int = 4,
) -> List[str]:
    """
    Select the top N features by variance.

    Args:
        df (pd.DataFrame): The DataFrame containing feature columns.
        feature_cols (Iterable[str]): Feature column names to consider.
        n_features (int, optional): Number of top features to select. Defaults to 4.

    Returns:
        List[str]: List of top feature column names ordered by variance (highest first).
    """
    
    columns = [col for col in feature_cols if col in df.columns]
    if not columns:
        return []

    numeric_df = df[columns].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return []

    variances = numeric_df.var().sort_values(ascending=False)
    return variances.head(n_features).index.tolist()

def plot_feature_anomalies(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    feature_cols: Optional[Iterable[str]] = None,
    anomaly_col: str = "is_anomaly",
    n_features: int = 4,
    split_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot top features with anomalies highlighted.

    Args:
        df (pd.DataFrame): DataFrame containing feature data and anomaly labels.
        timestamp_col (str, optional): Name of the timestamp column. Defaults to "timestamp".
        feature_cols (Optional[Iterable[str]]): Feature columns to plot. If None, numeric columns are selected. Defaults to None.
        anomaly_col (str, optional): Name of the anomaly label column. Defaults to "is_anomaly".
        n_features (int, optional): Number of top features to plot. Defaults to 4.
        split_indices (Optional[Sequence[int]]): Indices where vertical lines should be drawn (e.g., train/val/test splits). Defaults to None.
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (12, 8).

    Returns:
        Tuple: Matplotlib figure and axes objects.

    Raises:
        ValueError: If no numeric feature columns are available to plot.
    """
    
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [
            col for col in feature_cols if col not in {anomaly_col, "anomaly_score"}
        ]

    features_to_plot = select_top_features(df, feature_cols, n_features=n_features)
    if not features_to_plot:
        raise ValueError("No numeric feature columns available to plot.")

    time_axis = _resolve_time_axis(df, timestamp_col)
    anomaly_mask = df[anomaly_col] == 1 if anomaly_col in df.columns else pd.Series(False, index=df.index)

    fig, axes = plt.subplots(
        nrows=len(features_to_plot),
        ncols=1,
        figsize=figsize,
        sharex=True,
    )
    if len(features_to_plot) == 1:
        axes = [axes]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        ax.plot(time_axis, df[feature], color="steelblue", linewidth=1)
        if anomaly_mask.any():
            ax.scatter(
                time_axis[anomaly_mask],
                df.loc[anomaly_mask, feature],
                color="crimson",
                s=12,
                alpha=0.7,
                label="Anomaly" if idx == 0 else None,
            )
        _add_split_lines(ax, time_axis, split_indices)
        ax.set_ylabel(feature)

    axes[0].set_title("Top feature trends with anomalies")
    axes[-1].set_xlabel("Timestamp" if timestamp_col in df.columns else "Index")
    if anomaly_mask.any():
        axes[0].legend(loc="upper right")
    plt.tight_layout()
    return fig, axes

def plot_anomaly_score_timeline(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    score_col: str = "anomaly_score",
    threshold: Optional[float] = None,
    split_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot anomaly scores over time.

    Args:
        df (pd.DataFrame): DataFrame containing anomaly scores.
        timestamp_col (str, optional): Name of the timestamp column. Defaults to "timestamp".
        score_col (str, optional): Name of the anomaly score column. Defaults to "anomaly_score".
        threshold (Optional[float]): Threshold value to display as a reference line. Defaults to None.
        split_indices (Optional[Sequence[int]]): Indices where vertical lines should be drawn. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (12, 4).

    Returns:
        Tuple: Matplotlib figure and axes objects.

    Raises:
        ValueError: If the score column is not found in the DataFrame.
    """
    
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")

    time_axis = _resolve_time_axis(df, timestamp_col)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, df[score_col], color="slategray", linewidth=1)
    if threshold is not None:
        ax.axhline(threshold, color="crimson", linestyle="--", label="Threshold")
    _add_split_lines(ax, time_axis, split_indices)
    ax.set_title("Anomaly score over time")
    ax.set_xlabel("Timestamp" if timestamp_col in df.columns else "Index")
    ax.set_ylabel("Anomaly score")
    if threshold is not None:
        ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_anomaly_score_histogram(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    threshold: Optional[float] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (6, 4),
):
    """
    Plot histogram of anomaly scores.

    Args:
        df (pd.DataFrame): DataFrame containing anomaly scores.
        score_col (str, optional): Name of the anomaly score column. Defaults to "anomaly_score".
        threshold (Optional[float]): Threshold value to display as a reference line. Defaults to None.
        bins (int, optional): Number of histogram bins. Defaults to 50.
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (6, 4).

    Returns:
        Tuple: Matplotlib figure and axes objects.

    Raises:
        ValueError: If the score column is not found in the DataFrame.
    """
    
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[score_col], bins=bins, color="slategray", alpha=0.8)
    if threshold is not None:
        ax.axvline(threshold, color="crimson", linestyle="--", label="Threshold")
    ax.set_title("Anomaly score distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    if threshold is not None:
        ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_anomaly_rate_timeline(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    anomaly_col: str = "is_anomaly",
    window: int = 144,
    split_indices: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot rolling anomaly rate over time.

    Args:
        df (pd.DataFrame): DataFrame containing anomaly labels.
        timestamp_col (str, optional): Name of the timestamp column. Defaults to "timestamp".
        anomaly_col (str, optional): Name of the anomaly label column. Defaults to "is_anomaly".
        window (int, optional): Window size for rolling mean calculation. Defaults to 144.
        split_indices (Optional[Sequence[int]]): Indices where vertical lines should be drawn. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (12, 4).

    Returns:
        Tuple: Matplotlib figure and axes objects.

    Raises:
        ValueError: If the anomaly column is not found in the DataFrame.
    """
    
    if anomaly_col not in df.columns:
        raise ValueError(f"Missing anomaly column: {anomaly_col}")

    time_axis = _resolve_time_axis(df, timestamp_col)
    rate = df[anomaly_col].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, rate, color="teal", linewidth=1)
    _add_split_lines(ax, time_axis, split_indices)
    ax.set_title(f"Rolling anomaly rate (window={window})")
    ax.set_xlabel("Timestamp" if timestamp_col in df.columns else "Index")
    ax.set_ylabel("Anomaly rate")

    max_rate = float(rate.max()) if len(rate) else 0.0
    upper = min(1.0, max(0.05, max_rate * 1.2))
    ax.set_ylim(0, upper)
    plt.tight_layout()
    return fig, ax

def plot_anomaly_rate_by_split(
    df: pd.DataFrame,
    train_end: int,
    val_end: int,
    anomaly_col: str = "is_anomaly",
    figsize: Tuple[int, int] = (6, 4),
):
    """
    Plot anomaly rates for train, validation, and test splits.

    Args:
        df (pd.DataFrame): DataFrame containing anomaly labels.
        train_end (int): Index where training set ends.
        val_end (int): Index where validation set ends.
        anomaly_col (str, optional): Name of the anomaly label column. Defaults to "is_anomaly".
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (6, 4).

    Returns:
        Tuple: Matplotlib figure and axes objects.

    Raises:
        ValueError: If the anomaly column is not found in the DataFrame.
    """
    
    if anomaly_col not in df.columns:
        raise ValueError(f"Missing anomaly column: {anomaly_col}")

    splits = [df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]]
    labels = ["train", "val", "test"]
    rates = [float(split[anomaly_col].mean()) if len(split) else 0.0 for split in splits]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, rates, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Anomaly rate by split")
    ax.set_ylabel("Anomaly rate")

    upper = min(1.0, max(0.05, max(rates) * 1.2 if rates else 0.05))
    ax.set_ylim(0, upper)
    plt.tight_layout()
    return fig, ax

def _resolve_time_axis(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    """
    Resolve the time axis for plotting, using timestamp column if available.

    Args:
        df (pd.DataFrame): The DataFrame to extract time axis from.
        timestamp_col (str): Name of the timestamp column.

    Returns:
        pd.Series: Series containing timestamps or row indices if timestamp column not found.
    """
    
    if timestamp_col in df.columns:
        return df[timestamp_col]
    return pd.Series(df.index, index=df.index, name="index")

def _add_split_lines(
    ax: plt.Axes,
    time_axis: pd.Series,
    split_indices: Optional[Sequence[int]],
) -> None:
    """
    Add vertical lines to plot axes at specified split indices.

    Args:
        ax (plt.Axes): Matplotlib axes object to modify.
        time_axis (pd.Series): Time values corresponding to the split indices.
        split_indices (Optional[Sequence[int]]): Indices where vertical lines should be drawn.

    Returns:
        None
    """
    
    if not split_indices or len(time_axis) == 0:
        return

    for idx in split_indices:
        if idx is None:
            continue
        safe_idx = min(max(int(idx), 0), len(time_axis) - 1)
        ax.axvline(time_axis.iloc[safe_idx], color="gray", linestyle="--", alpha=0.6)