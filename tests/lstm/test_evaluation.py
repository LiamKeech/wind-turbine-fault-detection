import numpy as np
import pytest

from evaluation.lstm_autoencoder_evaluation import (
    align_labels_to_sequences,
    classification_metrics,
    compute_adaptive_threshold,
    compute_threshold,
)


# --------------------------------------------------------------------------
# Threshold tuning
# --------------------------------------------------------------------------

def test_compute_threshold_matches_numpy_quantile():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = compute_threshold(errors, quantile=0.8)
    assert threshold == pytest.approx(np.quantile(errors, 0.8))


def test_compute_threshold_default_quantile_is_near_max():
    errors = np.arange(1, 1001, dtype=float)  # 1..1000
    threshold = compute_threshold(errors)  # default quantile=0.999
    assert threshold == pytest.approx(np.quantile(errors, 0.999))
    assert threshold > np.median(errors)


def test_compute_threshold_all_equal_errors():
    errors = np.full(10, 3.5)
    threshold = compute_threshold(errors, quantile=0.5)
    assert threshold == pytest.approx(3.5)


def test_compute_adaptive_threshold_targets_normal_fpr():
    rng = np.random.default_rng(0)
    normal_errors = rng.normal(loc=0.0, scale=1.0, size=1000)
    anomaly_errors = rng.normal(loc=10.0, scale=1.0, size=50)
    errors = np.concatenate([normal_errors, anomaly_errors])
    labels = np.concatenate([np.zeros(1000), np.ones(50)])

    threshold = compute_adaptive_threshold(errors, labels, target_normal_fpr=0.01)

    normal_fpr = float((normal_errors > threshold).mean())
    assert normal_fpr == pytest.approx(0.01, abs=0.01)


# --------------------------------------------------------------------------
# Label alignment (row -> sequence), used by the threshold/eval pipeline
# --------------------------------------------------------------------------

def test_align_labels_to_sequences_any_aggregation_flags_window_with_anomaly():
    labels = [0, 0, 0, 1, 0, 0, 0, 0]
    seq_labels = align_labels_to_sequences(labels, window_size=4, stride=4, aggregation="any")
    # window 0 = indices [0:4] contains the anomaly at index 3 -> 1
    # window 1 = indices [4:8] all normal -> 0
    np.testing.assert_array_equal(seq_labels, [1, 0])


def test_align_labels_to_sequences_majority_aggregation():
    labels = [1, 1, 1, 0, 0]  # window_size=5, majority of 1s -> 1
    seq_labels = align_labels_to_sequences(labels, window_size=5, stride=5, aggregation="majority")
    assert seq_labels[0] == 1


def test_align_labels_to_sequences_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        align_labels_to_sequences([0, 1, 0], window_size=3, stride=3, aggregation="bogus")


# --------------------------------------------------------------------------
# Evaluation metrics (precision/recall/F1) — fixed predictions, no model
# --------------------------------------------------------------------------

def test_classification_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_classification_metrics_known_mixed_predictions():
    # tp=2 (idx 1,4), fn=1 (idx 2), fp=0, tn=2 (idx 0,3)
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["tp"] == 2.0
    assert metrics["fp"] == 0.0
    assert metrics["fn"] == 1.0
    assert metrics["tn"] == 2.0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["accuracy"] == pytest.approx(0.8)


def test_classification_metrics_no_positive_predictions_zero_division_safe():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["precision"] == pytest.approx(0.0)  # zero_division=0
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_classification_metrics_includes_auc_when_scores_given():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = classification_metrics(y_true, y_pred, y_scores=y_scores)

    assert metrics["auc"] == pytest.approx(1.0)


def test_classification_metrics_auc_is_nan_without_scores():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    metrics = classification_metrics(y_true, y_pred)

    assert np.isnan(metrics["auc"])
