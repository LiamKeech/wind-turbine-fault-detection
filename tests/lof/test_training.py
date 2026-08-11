import numpy as np
import pandas as pd
import pytest

from training.lof_training import summarize_anomalies, top_anomalies, train_lof


def _make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": rng.normal(50, 1, n),
            "pressure": rng.normal(4, 0.1, n),
        }
    )
    # inject an obvious anomaly near the end so scoring has something to find
    df.loc[n - 1, "temp"] = 500.0
    df.loc[n - 1, "pressure"] = 500.0
    return df


def test_train_lof_returns_expected_shapes_and_artifacts():
    df = _make_df(100)

    results_df, summary_df, artifacts = train_lof(
        df,
        rolling_window=5,
        n_neighbors=5,
        contamination=0.1,
        train_frac=0.7,
        val_frac=0.15,
    )

    assert "anomaly_score" in results_df.columns
    assert "is_anomaly" in results_df.columns
    assert len(results_df) == len(df) - (5 - 1)  # rows dropped by the rolling window

    assert summary_df.loc[0, "total_rows"] == len(results_df)
    assert set(artifacts.keys()) >= {
        "model", "scaler", "base_features", "feature_columns", "threshold", "train_end", "val_end",
    }
    assert artifacts["base_features"] == ["temp", "pressure"]
    assert artifacts["train_end"] + (len(results_df) - artifacts["val_end"]) <= len(results_df)


def test_train_lof_flags_injected_anomaly():
    df = _make_df(100)

    results_df, _summary_df, _artifacts = train_lof(
        df,
        rolling_window=5,
        n_neighbors=5,
        contamination=0.1,
        train_frac=0.7,
        val_frac=0.15,
    )

    # the injected extreme row should be the top-scoring anomaly
    top_row = results_df.sort_values("anomaly_score", ascending=False).iloc[0]
    assert top_row["is_anomaly"] == 1
    assert top_row["temp"] == 500.0


def test_train_lof_respects_explicit_feature_cols():
    df = _make_df(100)
    df["unused_col"] = 0.0

    _results_df, _summary_df, artifacts = train_lof(
        df,
        feature_cols=["temp", "pressure"],
        rolling_window=5,
        n_neighbors=5,
        contamination=0.1,
    )

    assert artifacts["base_features"] == ["temp", "pressure"]
    assert "unused_col" not in artifacts["feature_columns"]


def test_summarize_anomalies_counts_and_rate():
    df = pd.DataFrame({"is_anomaly": [0, 0, 1, 0, 1]})
    summary = summarize_anomalies(df)

    assert summary.loc[0, "total_rows"] == 5
    assert summary.loc[0, "anomalies"] == 2
    assert summary.loc[0, "anomaly_rate"] == pytest.approx(0.4)


def test_summarize_anomalies_handles_empty_dataframe():
    df = pd.DataFrame({"is_anomaly": pd.Series(dtype=int)})
    summary = summarize_anomalies(df)

    assert summary.loc[0, "total_rows"] == 0
    assert summary.loc[0, "anomalies"] == 0
    assert summary.loc[0, "anomaly_rate"] == 0.0


def test_top_anomalies_returns_sorted_subset():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
            "anomaly_score": [0.1, 0.9, 0.5, 0.2, 0.8],
            "is_anomaly": [0, 1, 0, 0, 1],
        }
    )

    top = top_anomalies(df, n=2)

    assert len(top) == 2
    assert list(top["anomaly_score"]) == [0.9, 0.8]
    assert list(top.columns) == ["timestamp", "anomaly_score", "is_anomaly"]
