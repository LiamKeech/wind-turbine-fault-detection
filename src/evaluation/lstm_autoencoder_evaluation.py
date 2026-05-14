from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)


def reconstruction_errors(
	model: torch.nn.Module,
	sequences: np.ndarray,
	batch_size: int,
	device: torch.device,
) -> np.ndarray:
	model.eval()
	dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	criterion = torch.nn.MSELoss(reduction="none")

	errors = []
	with torch.no_grad():
		for (batch,) in loader:
			batch = batch.to(device)
			recon = model(batch)
			loss = criterion(recon, batch)
			batch_errors = loss.mean(dim=(1, 2)).detach().cpu().numpy()
			errors.append(batch_errors)
	return np.concatenate(errors, axis=0)


def compute_threshold(errors: np.ndarray, quantile: float) -> float:
	return float(np.quantile(errors, quantile))


def flag_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
	return errors > threshold


def compute_classification_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
	y_true = np.asarray(y_true).astype(int)
	y_pred = np.asarray(y_pred).astype(int)

	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()

	precision = float(precision_score(y_true, y_pred, zero_division=0))
	recall = float(recall_score(y_true, y_pred, zero_division=0))
	f1 = float(f1_score(y_true, y_pred, zero_division=0))
	accuracy = float(accuracy_score(y_true, y_pred))

	auc = float("nan")
	if y_scores is not None:
		try:
			auc = float(roc_auc_score(y_true, np.asarray(y_scores)))
		except ValueError:
			auc = float("nan")

	true_positive_rate = tp / (tp + fn) if (tp + fn) else 0.0
	false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
	specificity = tn / (tn + fp) if (tn + fp) else 0.0

	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1_score": f1,
		"auc": auc,
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
		"tpr": float(true_positive_rate),
		"fpr": float(false_positive_rate),
		"specificity": float(specificity),
	}


def evaluate_anomaly_scores(
	y_true: np.ndarray,
	scores: np.ndarray,
	threshold: float,
) -> Dict[str, float]:
	y_pred = (np.asarray(scores) > threshold).astype(int)
	return compute_classification_metrics(y_true, y_pred, y_scores=scores)


def align_scores_with_timestamps(
	timestamps: Iterable,
	errors: np.ndarray,
	window_size: int,
	stride: int,
) -> pd.DataFrame:
	ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
	indices = np.arange(0, len(errors) * stride, stride) + (window_size - 1)
	indices = indices[indices < len(ts)]
	aligned_errors = errors[: len(indices)]
	return pd.DataFrame(
		{
			"timestamp": ts.iloc[indices].values,
			"reconstruction_error": aligned_errors,
		}
	)


def plot_reconstruction_errors(score_df: pd.DataFrame, threshold: float):
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(score_df["timestamp"], score_df["reconstruction_error"], label="Reconstruction error")
	ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
	ax.set_xlabel("Timestamp")
	ax.set_ylabel("Reconstruction error")
	ax.legend()
	fig.autofmt_xdate()
	return ax


def plot_feature_with_anomalies(
	df: pd.DataFrame,
	feature_col: str,
	anomaly_timestamps: Iterable,
):
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(df["timestamp"], df[feature_col], label=feature_col)

	anomaly_times = pd.to_datetime(pd.Series(anomaly_timestamps))
	if not anomaly_times.empty:
		values = df.set_index("timestamp")[feature_col].reindex(anomaly_times)
		ax.scatter(anomaly_times, values.values, color="red", s=12, label="Anomaly")

	ax.set_xlabel("Timestamp")
	ax.set_ylabel(feature_col)
	ax.legend()
	fig.autofmt_xdate()
	return ax
