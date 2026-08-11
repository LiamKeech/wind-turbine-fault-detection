from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.lof_preprocessing import (
    clean_data,
    dataset_overview,
    load_raw_data,
    missing_rate_summary,
    select_feature_columns,
    split_time_series,
)


def _make_df(n=20):
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": rng.normal(50, 1, n),
            "pressure": rng.normal(4, 0.1, n),
        }
    )


def test_load_raw_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_load_raw_data_sorts_and_dedups(tmp_path):
    df = _make_df(5)
    shuffled = pd.concat([df, df.iloc[[2]]]).sample(frac=1, random_state=1)
    csv_path = tmp_path / "raw.csv"
    shuffled.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path, timestamp_col="timestamp")

    assert list(loaded["timestamp"]) == sorted(loaded["timestamp"])
    assert loaded["timestamp"].is_unique
    assert list(loaded.index) == list(range(len(loaded)))


def test_select_feature_columns_defaults_to_numeric(tmp_path):
    df = _make_df()
    columns = select_feature_columns(df, timestamp_col="timestamp")
    assert set(columns) == {"temp", "pressure"}


def test_select_feature_columns_respects_include_and_exclude():
    df = _make_df()
    included = select_feature_columns(df, include_cols=["temp", "pressure", "missing_col"])
    assert included == ["temp", "pressure"]

    excluded = select_feature_columns(df, exclude_cols=["pressure"])
    assert excluded == ["temp"]


def test_clean_data_coerces_and_drops_missing():
    df = _make_df(5)
    df["temp"] = df["temp"].astype(object)
    df.loc[1, "temp"] = "not-a-number"
    df.loc[3, "pressure"] = np.nan

    cleaned = clean_data(df, feature_cols=["temp", "pressure"], timestamp_col="timestamp")

    assert len(cleaned) == 3
    assert cleaned["temp"].dtype.kind == "f"
    assert list(cleaned.index) == list(range(len(cleaned)))


def test_split_time_series_respects_fractions():
    df = _make_df(100)
    train_df, val_df, test_df = split_time_series(df, train_frac=0.7, val_frac=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert list(train_df["timestamp"]) == list(df["timestamp"].iloc[:70])


@pytest.mark.parametrize("train_frac,val_frac", [(0.0, 0.5), (0.6, 0.5), (-0.1, 0.2)])
def test_split_time_series_rejects_invalid_fractions(train_frac, val_frac):
    df = _make_df(10)
    with pytest.raises(ValueError):
        split_time_series(df, train_frac=train_frac, val_frac=val_frac)


def test_dataset_overview_reports_shape_and_range():
    df = _make_df(10)
    overview = dataset_overview(df, timestamp_col="timestamp")

    assert overview.loc[0, "rows"] == 10
    assert overview.loc[0, "features"] == 2
    assert overview.loc[0, "start"] == df["timestamp"].min()
    assert overview.loc[0, "end"] == df["timestamp"].max()


def test_missing_rate_summary_counts_and_sorts():
    df = _make_df(10)
    df.loc[0:4, "temp"] = np.nan

    summary = missing_rate_summary(df, feature_cols=["temp", "pressure"])

    assert summary.loc["temp", "missing_count"] == 5
    assert summary.loc["temp", "missing_rate"] == pytest.approx(0.5)
    assert summary.index[0] == "temp"  # sorted descending by missing_rate


def test_missing_rate_summary_top_n():
    df = _make_df(10)
    df.loc[0, "temp"] = np.nan
    summary = missing_rate_summary(df, feature_cols=["temp", "pressure"], top_n=1)
    assert len(summary) == 1
