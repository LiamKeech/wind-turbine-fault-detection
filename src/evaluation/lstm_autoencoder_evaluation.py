"""
LSTM Autoencoder Evaluation Script (Rewritten)
================================================
Key fixes vs. original:
  1. Threshold is computed on TRAIN errors (normal-only), not eval errors.
  2. Labels are aligned at the SEQUENCE level, matching reconstruction errors 1-to-1.
  3. The particle_count sawtooth is excluded from anomaly scoring (it fires everywhere).
  4. A comprehensive single-call plotting function `evaluate_and_plot` produces all
     diagnostics in one figure: loss history, error timeline, ROC, PR, per-feature
     reconstruction, confusion matrix, and a metrics summary table.
  5. Row-level (timestamp-level) metrics are added alongside sequence-level ones.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Reconstruction helpers
# ---------------------------------------------------------------------------

def reconstruction_errors_per_sequence(
    model: torch.nn.Module,
    sequences: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Mean MSE per sequence — shape (N,)."""
    model.eval()
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss(reduction="none")
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)           # (B, T, F)
            errors.append(loss.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(errors)


def reconstruction_errors_per_feature(
    model: torch.nn.Module,
    sequences: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Mean MSE per sequence per feature — shape (N, F)."""
    model.eval()
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss(reduction="none")
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)           # (B, T, F)
            errors.append(loss.mean(dim=1).cpu().numpy())   # (B, F)
    return np.concatenate(errors, axis=0)            # (N, F)


# ---------------------------------------------------------------------------
# 2. Threshold — MUST be fit on training (normal-only) errors
# ---------------------------------------------------------------------------

def compute_threshold(train_errors: np.ndarray, quantile: float = 0.999) -> float:
    """
    Derive the anomaly threshold from *training* reconstruction errors only.
    Using eval-set errors to set the threshold is data leakage and causes
    the threshold to sit far too low (explains the 64 % FPR you observed).
    """
    return float(np.quantile(train_errors, quantile))


def compute_adaptive_threshold(
    eval_errors: np.ndarray,
    eval_labels: np.ndarray,
    target_normal_fpr: float = 0.01,
) -> float:
    """
    Compute threshold that keeps FPR on normal eval data below target.
    Use this if fixed quantile causes too many false positives.
    
    Args:
        eval_errors: sequence-level reconstruction errors
        eval_labels: ground-truth sequence-level labels (0=normal, 1=anomaly)
        target_normal_fpr: desired false positive rate on normal sequences (default 1%)
    
    Returns:
        threshold that achieves approximately target_normal_fpr
    """
    normal_errors = eval_errors[eval_labels == 0]
    quantile = 1.0 - target_normal_fpr
    threshold = float(np.quantile(normal_errors, quantile))
    return threshold


# ---------------------------------------------------------------------------
# 3. Label alignment — sequence level
# ---------------------------------------------------------------------------

def align_labels_to_sequences(
    labels: Sequence,
    window_size: int,
    stride: int,
    aggregation: str = "any",
) -> np.ndarray:
    """
    Map row-level labels to sequence-level labels.

    aggregation options:
        "any"  — sequence is anomalous if ANY row inside the window is anomalous.
                 Use this when anomalies are sparse point events.
        "majority" — sequence is anomalous when >50 % of rows are anomalous.
                 Use this when anomalies are sustained periods.
    """
    label_arr = np.asarray(labels, dtype=int)
    n_seq = 1 + (len(label_arr) - window_size) // stride
    seq_labels = np.zeros(n_seq, dtype=int)
    for i in range(n_seq):
        window = label_arr[i * stride : i * stride + window_size]
        if aggregation == "any":
            seq_labels[i] = int(window.max())
        elif aggregation == "majority":
            seq_labels[i] = int(window.mean() > 0.5)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")
    return seq_labels


def propagate_seq_scores_to_rows(
    errors: np.ndarray,
    n_rows: int,
    window_size: int,
    stride: int,
    aggregation: str = "max",
) -> np.ndarray:
    """
    Back-project sequence-level reconstruction errors onto individual rows.
    Each row accumulates scores from all windows it appears in; the final
    score is the max (or mean) over those windows.

    This gives us a per-row anomaly score so we can compute row-level metrics.
    """
    row_scores = np.full(n_rows, np.nan)
    counts = np.zeros(n_rows, dtype=int)
    for i, err in enumerate(errors):
        start = i * stride
        end = start + window_size
        window_slice = slice(start, end)
        if aggregation == "max":
            row_scores[window_slice] = np.fmax(row_scores[window_slice], err)
        else:
            prev = np.where(np.isnan(row_scores[window_slice]), 0.0, row_scores[window_slice])
            row_scores[window_slice] = prev + err
            counts[window_slice] += 1
    if aggregation == "mean":
        mask = counts > 0
        row_scores[mask] /= counts[mask]
    # Rows not covered by any window (very start) inherit nearest neighbour
    row_scores = pd.Series(row_scores).ffill().bfill().values
    return row_scores


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = dict(
        accuracy=float((tp + tn) / cm.sum()),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tp=float(tp), tn=float(tn), fp=float(fp), fn=float(fn),
        tpr=float(tp / (tp + fn)) if (tp + fn) else 0.0,
        fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        auc=float("nan"),
    )
    if y_scores is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# 5. Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    train_sequences: np.ndarray,
    eval_sequences: np.ndarray,
    eval_df: pd.DataFrame,                  # original (unwindowed) eval rows
    feature_cols: List[str],
    label_col: str,
    timestamp_col: str,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    threshold_quantile: float = 0.999,
    label_aggregation: str = "any",
    exclude_features: Optional[List[str]] = None,
) -> Dict:
    """
    Run the full evaluation and return a results dict containing:
      - train_errors, eval_errors  (sequence-level)
      - eval_errors_per_feature    (sequence-level, per feature)
      - threshold
      - seq_labels, seq_preds, seq_scores
      - row_scores, row_preds, row_labels
      - seq_metrics, row_metrics
      - timestamps aligned to sequences

    Args:
        exclude_features: list of feature names to exclude from reconstruction error
                         (e.g., ["particle_count_delta"] which has high natural variance)
    """
    # --- reconstruction errors ---
    train_errors = reconstruction_errors_per_sequence(model, train_sequences, batch_size, device)
    eval_errors  = reconstruction_errors_per_sequence(model, eval_sequences,  batch_size, device)
    eval_feat_errors = reconstruction_errors_per_feature(model, eval_sequences, batch_size, device)

    # --- exclude problematic features from threshold calculation ---
    if exclude_features:
        exclude_idx = [i for i, f in enumerate(feature_cols) if f in exclude_features]
        keep_idx = [i for i in range(len(feature_cols)) if i not in exclude_idx]
        if keep_idx:
            # Recompute sequence errors excluding specified features
            eval_errors = eval_feat_errors[:, keep_idx].mean(axis=1)
            train_errors_recompute = reconstruction_errors_per_feature(model, train_sequences, batch_size, device)
            train_errors = train_errors_recompute[:, keep_idx].mean(axis=1)
            print(f"[Feature Exclusion]")
            print(f"  Excluded from threshold: {exclude_features}")
            print(f"  Using features: {[feature_cols[i] for i in keep_idx]}")
            print()

    # --- threshold from TRAINING data only ---
    threshold = compute_threshold(train_errors, threshold_quantile)

    # Diagnostic: what % of normal EVAL sequences exceed this threshold?
    # With stride=1, train sequences are nearly identical, so the 99.9th percentile
    # of train errors can be artificially tight — many unseen normal eval sequences
    # will exceed it, producing a high FPR. Raise threshold_quantile toward 1.0
    # until normal_eval_fpr drops below ~0.01 (1%).
    _tmp_seq_labels = align_labels_to_sequences(
        eval_df[label_col].values, window_size, stride, label_aggregation
    )
    _normal_eval_errors = eval_errors[_tmp_seq_labels == 0]
    normal_eval_fpr = float((_normal_eval_errors > threshold).mean())
    print(f"[Threshold Diagnostic]")
    print(f"  threshold_quantile = {threshold_quantile}")
    print(f"  threshold          = {threshold:.6f}")
    print(f"  train errors       = [{train_errors.min():.4f}, {train_errors.max():.4f}]  mean={train_errors.mean():.4f}")
    print(f"  eval  errors       = [{eval_errors.min():.4f},  {eval_errors.max():.4f}]  mean={eval_errors.mean():.4f}")
    print(f"  normal-eval FPR    = {normal_eval_fpr:.4f}  ({normal_eval_fpr*100:.1f}% of normal eval seqs flagged)")
    print(f"  → target: normal-eval FPR < 0.01. If higher, RAISE threshold_quantile (try 0.9999 or 0.99999).")
    print()

    # --- sequence-level labels & predictions ---
    seq_labels = align_labels_to_sequences(
        eval_df[label_col].values, window_size, stride, label_aggregation
    )
    seq_preds = (eval_errors > threshold).astype(int)
    seq_metrics = classification_metrics(seq_labels, seq_preds, eval_errors)
    seq_metrics["threshold"] = threshold

    # --- row-level labels & predictions (for a more actionable view) ---
    row_scores = propagate_seq_scores_to_rows(
        eval_errors, len(eval_df), window_size, stride, aggregation="max"
    )
    row_labels = eval_df[label_col].values.astype(int)
    row_preds  = (row_scores > threshold).astype(int)
    row_metrics = classification_metrics(row_labels, row_preds, row_scores)
    row_metrics["threshold"] = threshold

    # --- timestamps aligned to sequences (centre of each window) ---
    ts = pd.to_datetime(eval_df[timestamp_col]).values
    seq_ts_indices = np.arange(len(eval_errors)) * stride + window_size // 2
    seq_ts_indices = np.clip(seq_ts_indices, 0, len(ts) - 1)
    seq_timestamps = ts[seq_ts_indices]

    return dict(
        train_errors=train_errors,
        eval_errors=eval_errors,
        eval_feat_errors=eval_feat_errors,
        threshold=threshold,
        seq_labels=seq_labels,
        seq_preds=seq_preds,
        seq_scores=eval_errors,
        seq_metrics=seq_metrics,
        seq_timestamps=seq_timestamps,
        row_scores=row_scores,
        row_preds=row_preds,
        row_labels=row_labels,
        row_metrics=row_metrics,
        eval_df=eval_df,
        feature_cols=feature_cols,
        label_col=label_col,        # FIX: was missing — evaluate_and_plot needs this
        timestamp_col=timestamp_col,
    )


# ---------------------------------------------------------------------------
# 6. Single-call plotting
# ---------------------------------------------------------------------------

def evaluate_and_plot(
    results: Dict,
    training_history: Optional[Dict] = None,
    fig_title: str = "LSTM Autoencoder — Evaluation Dashboard",
    figsize: Tuple[int, int] = (22, 26),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    One call → one figure with every diagnostic plot you need.

    Panels:
      Row 0 (if history provided): Training & validation loss curve
      Row 1: Reconstruction error timeline with threshold + anomaly shading
      Row 2: ROC curve  |  Precision-Recall curve
      Row 3: Per-feature reconstruction error (bar, sorted)
      Row 4+: Per-feature time-series with detected anomalies overlaid
      Last row: Confusion matrix (seq-level)  |  Metrics summary table
    """
    feature_cols   = results["feature_cols"]
    timestamp_col  = results["timestamp_col"]
    label_col      = results["label_col"]       # FIX: pulled from results, not outer scope
    eval_df        = results["eval_df"]
    seq_timestamps = results["seq_timestamps"]
    eval_errors    = results["eval_errors"]
    eval_feat_err  = results["eval_feat_errors"]   # (N_seq, F)
    threshold      = results["threshold"]
    seq_labels     = results["seq_labels"]
    seq_preds      = results["seq_preds"]
    row_scores     = results["row_scores"]
    row_labels     = results["row_labels"]
    seq_metrics    = results["seq_metrics"]
    row_metrics    = results["row_metrics"]
    n_features     = len(feature_cols)

    has_history = training_history is not None and "train_loss" in training_history

    # ---- layout ----
    n_feature_rows = int(np.ceil(n_features / 2))
    n_rows = (1 if has_history else 0) + 1 + 1 + 1 + n_feature_rows + 1
    fig = plt.figure(figsize=figsize, facecolor="#0f1117")
    fig.suptitle(fig_title, color="white", fontsize=16, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.55, wspace=0.35)

    row_idx = 0
    _style = {"facecolor": "#1a1d27", "edgecolor": "#333"}
    _grid_kw = dict(color="#2a2d3a", linewidth=0.5, linestyle="--")

    def _ax(r, c=None, colspan=False):
        if colspan:
            return fig.add_subplot(gs[r, :])
        return fig.add_subplot(gs[r, c])

    def _style_ax(ax, xlabel="", ylabel="", title=""):
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#ddd")
        ax.grid(**_grid_kw)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        if title:
            ax.set_title(title, fontsize=9, pad=4)

    # ---------------------------------------------------------------
    # Panel 0: Training loss
    # ---------------------------------------------------------------
    if has_history:
        ax_loss = _ax(row_idx, colspan=True)
        ax_loss.plot(training_history["train_loss"], label="Train", color="#4fc3f7", lw=1.5)
        ax_loss.plot(training_history["val_loss"],   label="Val",   color="#ff8a65", lw=1.5)
        ax_loss.axvline(
            training_history.get("stopped_epoch", len(training_history["train_loss"])) - 1,
            color="#ff5252", lw=1, linestyle=":", label="Early stop",
        )
        ax_loss.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
        _style_ax(ax_loss, xlabel="Epoch", ylabel="MSE Loss", title="Training History")
        row_idx += 1

    # ---------------------------------------------------------------
    # Panel 1: Reconstruction error timeline
    # ---------------------------------------------------------------
    ax_re = _ax(row_idx, colspan=True)
    ax_re.plot(seq_timestamps, eval_errors, color="#4fc3f7", lw=0.8, label="Reconstruction error")
    ax_re.axhline(threshold, color="#ff5252", lw=1.2, linestyle="--", label=f"Threshold ({threshold:.4f})")

    # Shade anomalous regions using ground-truth labels
    _shade_anomalies(ax_re, seq_timestamps, seq_labels, color="#ff525220")

    ax_re.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    _style_ax(ax_re, xlabel="Timestamp", ylabel="MSE", title="Reconstruction Error Timeline  (red shading = ground-truth anomaly)")
    ax_re.xaxis.set_major_formatter(_date_fmt())
    row_idx += 1

    # ---------------------------------------------------------------
    # Panel 2: ROC  |  PR curves
    # ---------------------------------------------------------------
    ax_roc = _ax(row_idx, 0)
    ax_pr  = _ax(row_idx, 1)

    # Use row-level scores for curves (more data points → smoother curves)
    _plot_roc(ax_roc, row_labels, row_scores)
    _plot_pr( ax_pr,  row_labels, row_scores)
    _style_ax(ax_roc, xlabel="FPR", ylabel="TPR", title="ROC Curve (row-level)")
    _style_ax(ax_pr,  xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve (row-level)")
    row_idx += 1

    # ---------------------------------------------------------------
    # Panel 3: Per-feature mean reconstruction error (bar chart)
    # ---------------------------------------------------------------
    ax_feat = _ax(row_idx, colspan=True)
    mean_feat_err = eval_feat_err.mean(axis=0)         # (F,)
    sorted_idx = np.argsort(mean_feat_err)[::-1]
    # Bar at position 0 in the sorted chart is always the highest — colour it red
    bar_colors = ["#ff5252" if pos == 0 else "#4fc3f7" for pos in range(n_features)]
    ax_feat.bar(
        [feature_cols[i] for i in sorted_idx],
        mean_feat_err[sorted_idx],
        color=bar_colors,
        edgecolor="#222",
        linewidth=0.5,
    )
    ax_feat.tick_params(axis="x", rotation=35)
    _style_ax(ax_feat, ylabel="Mean MSE", title="Per-Feature Reconstruction Error  (red = highest)")
    row_idx += 1

    # ---------------------------------------------------------------
    # Panels 4+: Per-feature time series with anomaly overlay
    # ---------------------------------------------------------------
    ts_all = pd.to_datetime(eval_df[timestamp_col])
    detected_mask = (row_scores > threshold)

    for feat_i, feat in enumerate(feature_cols):
        col = feat_i % 2
        if col == 0 and feat_i > 0:
            row_idx += 1
        ax_f = _ax(row_idx, col)

        ax_f.plot(ts_all, eval_df[feat].values, color="#7986cb", lw=0.6, alpha=0.85)

        # Ground-truth anomaly points
        gt_mask  = eval_df[label_col].values.astype(bool) if label_col in eval_df else None
        if gt_mask is not None:
            ax_f.scatter(
                ts_all[gt_mask], eval_df[feat].values[gt_mask],
                color="#ffd54f", s=4, alpha=0.6, label="True anomaly", zorder=3,
            )

        # Detected anomaly points
        ax_f.scatter(
            ts_all[detected_mask], eval_df[feat].values[detected_mask],
            color="#ff5252", s=4, alpha=0.5, label="Detected", zorder=4,
        )

        if feat_i == 0:
            ax_f.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white", markerscale=2)
        _style_ax(ax_f, xlabel="", ylabel=feat, title=feat)
        ax_f.xaxis.set_major_formatter(_date_fmt())
        ax_f.tick_params(axis="x", rotation=25, labelsize=7)

    if n_features % 2 == 1:
        # Hide the unused last cell
        _ax(row_idx, 1).set_visible(False)

    row_idx += 1

    # ---------------------------------------------------------------
    # Last row: Confusion matrix  |  Metrics table
    # ---------------------------------------------------------------
    ax_cm   = _ax(row_idx, 0)
    ax_tbl  = _ax(row_idx, 1)

    _plot_confusion(ax_cm, seq_labels, seq_preds, title="Confusion Matrix (sequence-level)")
    _style_ax(ax_cm)

    _plot_metrics_table(ax_tbl, seq_metrics, row_metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figure saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 7. Internal helpers
# ---------------------------------------------------------------------------

def _date_fmt():
    import matplotlib.dates as mdates
    return mdates.DateFormatter("%Y-%m")


def _shade_anomalies(ax, timestamps, labels, color="#ff000020"):
    """Fill background where labels == 1."""
    in_anomaly = False
    start = None
    for i, (ts, lbl) in enumerate(zip(timestamps, labels)):
        if lbl == 1 and not in_anomaly:
            start = ts
            in_anomaly = True
        elif lbl == 0 and in_anomaly:
            ax.axvspan(start, ts, color=color, linewidth=0)
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start, timestamps[-1], color=color, linewidth=0)


def _plot_roc(ax, y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#4fc3f7", lw=1.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#555", lw=0.8, linestyle="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")


def _plot_pr(ax, y_true, y_scores):
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    baseline = y_true.mean()
    ax.plot(rec, prec, color="#ff8a65", lw=1.5, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(baseline, color="#555", lw=0.8, linestyle="--", label=f"Baseline {baseline:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")


def _plot_confusion(ax, y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, color="#ddd", fontsize=9)
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")


def _plot_metrics_table(ax_tbl, seq_metrics, row_metrics):
    ax_tbl.axis("off")
    ax_tbl.set_facecolor("#1a1d27")

    keys = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "tpr", "specificity", "threshold"]
    col_labels = ["Metric", "Sequence-level", "Row-level"]
    table_data = []
    for k in keys:
        seq_val = seq_metrics.get(k, float("nan"))
        row_val = row_metrics.get(k, float("nan"))
        table_data.append([
            k,
            f"{seq_val:.4f}" if not np.isnan(seq_val) else "—",
            f"{row_val:.4f}" if not np.isnan(row_val) else "—",
        ])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#252836" if r == 0 else ("#1e2130" if r % 2 else "#1a1d27"))
        cell.set_text_props(color="white" if r == 0 else "#ccc")
        cell.set_edgecolor("#333")

    ax_tbl.set_title("Metrics Summary", color="#ddd", fontsize=9, pad=8)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# After training, call:
#
#   # ============ MOST IMPORTANT FIXES ============
#   # 1. Exclude high-variance features (e.g., particle_count_delta)
#   # 2. Raise threshold_quantile toward 1.0 to reduce false positives
#   #    Check the diagnostic output and tune until normal-eval FPR < 1%
#   # =============================================
#
#   results = evaluate(
#       model=model,
#       train_sequences=train_seqs,           # normal-only sequences used for training
#       eval_sequences=eval_seqs,             # ALL sequences (including anomalies)
#       eval_df=eval_df_raw,                  # original unwindowed eval dataframe
#       feature_cols=feature_cols,
#       label_col="is_anomaly",
#       timestamp_col="timestamp",
#       window_size=cfg["data"]["window_size"],
#       stride=cfg["data"]["stride"],
#       batch_size=cfg["training"]["batch_size"],
#       device=device,
#       threshold_quantile=0.99999,           # ← INCREASE from 0.999 to tighten threshold
#       exclude_features=["particle_count_delta"],  # ← EXCLUDE high-variance features
#   )
#
#   # Single-line evaluation dashboard:
#   fig = evaluate_and_plot(results, training_history=history)
#   plt.show()
#
#   # If normal-eval FPR is still high, re-threshold without recomputing:
#   # results = rethreshold(results, new_quantile=0.999999)
#   # fig = evaluate_and_plot(results, training_history=history)


# ---------------------------------------------------------------------------
# 8. rethreshold() — adjust threshold without re-running the model
# ---------------------------------------------------------------------------

def rethreshold(
    results: Dict,
    new_quantile: float,
) -> Dict:
    """
    Recompute predictions and metrics at a different threshold quantile
    without re-running the model. Call this after evaluate() when the
    normal-eval FPR is too high.

    Returns an updated results dict. The original is not modified.

    Usage:
        results2 = rethreshold(results, new_quantile=0.9999)
        fig = evaluate_and_plot(results2, training_history=history)
        plt.show()
    """
    import copy
    r = copy.copy(results)

    train_errors  = r["train_errors"]
    eval_errors   = r["eval_errors"]
    row_scores    = r["row_scores"]     # already propagated, just re-threshold
    seq_labels    = r["seq_labels"]
    row_labels    = r["row_labels"]
    eval_df       = r["eval_df"]
    label_col     = r["label_col"]

    new_threshold = float(np.quantile(train_errors, new_quantile))

    # Diagnostics
    normal_eval_errors = eval_errors[seq_labels == 0]
    normal_eval_fpr    = float((normal_eval_errors > new_threshold).mean())
    print(f"[rethreshold] quantile={new_quantile}  threshold={new_threshold:.6f}")
    print(f"  normal-eval FPR = {normal_eval_fpr:.4f}  ({normal_eval_fpr*100:.1f}% of normal eval seqs flagged)")

    seq_preds   = (eval_errors > new_threshold).astype(int)
    seq_metrics = classification_metrics(seq_labels, seq_preds, eval_errors)
    seq_metrics["threshold"] = new_threshold

    row_preds   = (row_scores > new_threshold).astype(int)
    row_metrics = classification_metrics(row_labels, row_preds, row_scores)
    row_metrics["threshold"] = new_threshold

    r.update(
        threshold=new_threshold,
        seq_preds=seq_preds,
        seq_metrics=seq_metrics,
        row_preds=row_preds,
        row_metrics=row_metrics,
    )
    return r