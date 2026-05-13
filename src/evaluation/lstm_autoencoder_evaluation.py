from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from features.lstm_autoencoder_features import WindowedFeatures
from models.lstm_autoencoder import LSTMAutoencoder
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class EvaluationResult:
    """
    Result of evaluating an autoencoder, including reconstruction errors, anomaly flags, threshold, and metrics.
    """    

    errors: np.ndarray
    threshold: float
    flags: np.ndarray
    metrics: Dict[str, float]

def evaluate_autoencoder(
    model: LSTMAutoencoder,
    loader: DataLoader,
    windowed: WindowedFeatures,
    *,
    device: Optional[torch.device] = None,
    threshold_percentile: float = 95.0,
    label_series: Optional[Sequence[int]] = None,
) -> EvaluationResult:
    """
    Evaluate the autoencoder using reconstruction errors.

    Args:
        model (LSTMAutoencoder): Trained autoencoder model to evaluate.
        loader (DataLoader): DataLoader with test windows.
        windowed (WindowedFeatures): Windowed features for test data.
        device (Optional[torch.device], optional): Torch device for evaluation. Defaults to None.
        threshold_percentile (float, optional): Percentile for anomaly threshold. Defaults to 95.0.
        label_series (Optional[Sequence[int]], optional): Optional label series aligned with the test split. Defaults to None.

    Returns:
        EvaluationResult: Container with reconstruction errors, threshold, anomaly flags, and metrics.
    """    
    
    errors = reconstruction_errors(model, loader, device=device)
    threshold = float(np.percentile(errors, threshold_percentile))
    flags = errors > threshold

    metrics: Dict[str, float] = {
        "threshold": threshold,
        "anomaly_rate": float(flags.mean()),
    }

    if label_series is not None:
        window_size = int(windowed.windows.shape[1])
        window_labels = _window_labels(label_series, windowed.window_start_indices, window_size)
        metrics.update(_supervised_metrics(window_labels, flags, errors))

    return EvaluationResult(
        errors=errors,
        threshold=threshold,
        flags=flags,
        metrics=metrics,
    )

def reconstruction_errors(
    model: LSTMAutoencoder,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute reconstruction errors for all windows in the loader.

    Args:
        model (LSTMAutoencoder): Trained autoencoder model.
        loader (DataLoader): DataLoader with test windows.
        device (Optional[torch.device], optional): Torch device for evaluation. Defaults to None.

    Returns:
        np.ndarray: Array of reconstruction errors for each window.
    """    
    
    device = device or torch.device("cpu")
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            batch_errors = model.reconstruction_error(batch, reduction="sample")
            errors.append(batch_errors.detach().cpu().numpy())
    return np.concatenate(errors)

def plot_training_history(history: Dict[str, Sequence[float]]) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history (Dict[str, Sequence[float]]): Dictionary containing 'train' and 'val' loss sequences.s
    """        

    plt.figure(figsize=(6, 4))
    plt.plot(history.get("train", []), label="train")
    plt.plot(history.get("val", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_distribution(errors: np.ndarray, threshold: float) -> None:
    """
    Plot the distribution of reconstruction errors with the anomaly threshold.

    Args:
        errors (np.ndarray): Array of reconstruction errors.
        threshold (float): Anomaly threshold.
    """    

    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(threshold, color="red", linestyle="--", label="threshold")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

def build_anomaly_table(
    reference_df: pd.DataFrame,
    windowed: WindowedFeatures,
    errors: np.ndarray,
    threshold: float,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Build a table of detected anomalies with their reconstruction errors and timestamps.

    Args:
        reference_df (pd.DataFrame): DataFrame to pull rows from (raw or processed).
        windowed (WindowedFeatures): Windowed features used for scoring.
        errors (np.ndarray): Reconstruction errors per window.
        threshold (float): Anomaly threshold.
        timestamp_col (str, optional): Timestamp column name for matching. Defaults to "timestamp".

    Raises:
        ValueError: If the length of errors does not match the number of windows.

    Returns:
        pd.DataFrame: DataFrame containing anomalous rows plus error metadata.
    """    
    
    if len(errors) != len(windowed.window_start_indices):
        raise ValueError("errors length must match number of windows.")

    anomaly_mask = errors > threshold
    if not anomaly_mask.any():
        return reference_df.head(0).assign(reconstruction_error=[], is_anomaly=[])

    if windowed.window_end_timestamps is not None and timestamp_col in reference_df.columns:
        ref = reference_df.copy()
        ref[timestamp_col] = pd.to_datetime(ref[timestamp_col], errors="coerce")
        ref = ref.dropna(subset=[timestamp_col])
        ref = ref.sort_values(timestamp_col)
        ref = ref.drop_duplicates(subset=[timestamp_col], keep="last")

        anomaly_times = pd.to_datetime(
            windowed.window_end_timestamps[anomaly_mask],
            errors="coerce",
        )
        score_table = pd.DataFrame(
            {
                timestamp_col: anomaly_times,
                "reconstruction_error": errors[anomaly_mask],
            }
        ).dropna(subset=[timestamp_col])
        score_table = score_table.groupby(timestamp_col, as_index=False)[
            "reconstruction_error"
        ].max()
        score_table["is_anomaly"] = True

        merged = ref.merge(score_table, on=timestamp_col, how="inner")
        return merged.sort_values(timestamp_col).reset_index(drop=True)

    ref = reference_df.reset_index(drop=True)
    end_indices = windowed.window_start_indices + windowed.windows.shape[1] - 1
    anomaly_indices = end_indices[anomaly_mask]
    rows = ref.iloc[anomaly_indices].copy()
    rows["reconstruction_error"] = errors[anomaly_mask]
    rows["is_anomaly"] = True
    return rows.reset_index(drop=True)

def predict_window_anomaly(
    model: LSTMAutoencoder,
    window: np.ndarray,
    threshold: float,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Predict whether a single window is anomalous based on its reconstruction error.

    Args:
        model (LSTMAutoencoder): Trained autoencoder model.
        window (np.ndarray): Array shaped (seq_len, n_features) or (1, seq_len, n_features).
        threshold (float): Anomaly threshold.
        device (Optional[torch.device], optional): Torch device for scoring. Defaults to None.

    Raises:
        ValueError: If the window has an invalid shape.

    Returns:
        Dict[str, float]: Dictionary with reconstruction error and anomaly flag.
    """    
    
    device = device or torch.device("cpu")
    model.eval()

    window_array = np.asarray(window, dtype=np.float32)
    if window_array.ndim == 2:
        window_array = window_array[None, :, :]
    if window_array.ndim != 3:
        raise ValueError("window must have shape (seq_len, features) or (1, seq_len, features).")

    tensor = torch.tensor(window_array, dtype=torch.float32, device=device)
    with torch.no_grad():
        error = model.reconstruction_error(tensor, reduction="sample")[0].item()

    return {
        "reconstruction_error": float(error),
        "is_anomaly": bool(error > threshold),
    }

def _window_labels(
    labels: Sequence[int],
    start_indices: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Generate binary labels for each window based on the presence of any positive labels within the window.

    Args:
        labels (Sequence[int]): Sequence of binary labels.
        start_indices (np.ndarray): Array of starting indices for each window.
        window_size (int): Size of each window.

    Returns:
        np.ndarray: Array of binary labels for each window.
    """    
    
    label_array = np.asarray(labels)
    window_labels = []
    for start in start_indices:
        end = start + window_size
        window_labels.append(int(label_array[start:end].max() > 0))
    return np.array(window_labels, dtype=int)

def _supervised_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute supervised classification metrics based on true labels, predicted anomaly flags, and reconstruction error scores.

    Args:
        y_true (np.ndarray): Array of true binary labels.
        y_pred (np.ndarray): Array of predicted binary labels.
        scores (np.ndarray): Array of reconstruction error scores.

    Returns:
        Dict[str, float]: Dictionary with classification metrics.
    """    
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred.astype(int),
        average="binary",
        zero_division=0,
    )
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }