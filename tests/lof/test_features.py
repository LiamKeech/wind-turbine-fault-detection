import numpy as np
import pandas as pd
import pytest

from features.lof_features import add_rolling_features, build_feature_matrix


def _make_df(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "temp": rng.normal(50, 1, n),
            "pressure": rng.normal(4, 0.1, n),
        }
    )


def test_add_rolling_features_adds_mean_and_std_columns():
    df = _make_df(20)
    result = add_rolling_features(df, feature_cols=["temp", "pressure"], window=5)

    for col in ["temp_roll_mean", "temp_roll_std", "pressure_roll_mean", "pressure_roll_std"]:
        assert col in result.columns

    # rows with insufficient window history are dropped, index is reset
    assert len(result) == 20 - (5 - 1)
    assert list(result.index) == list(range(len(result)))
    assert not result.isna().any().any()


def test_add_rolling_features_skips_unknown_columns():
    df = _make_df(10)
    result = add_rolling_features(df, feature_cols=["temp", "not_a_column"], window=3)

    assert "temp_roll_mean" in result.columns
    assert "not_a_column_roll_mean" not in result.columns


def test_add_rolling_features_matches_manual_rolling_mean():
    df = _make_df(10)
    result = add_rolling_features(df, feature_cols=["temp"], window=3)

    expected = df["temp"].rolling(window=3, min_periods=3).mean().dropna().reset_index(drop=True)
    np.testing.assert_allclose(result["temp_roll_mean"].to_numpy(), expected.to_numpy())


def test_build_feature_matrix_combines_base_and_rolling_columns():
    df = _make_df(20)
    feature_df = add_rolling_features(df, feature_cols=["temp", "pressure"], window=5)

    matrix, columns = build_feature_matrix(feature_df, base_features=["temp", "pressure"])

    assert columns == [
        "temp",
        "pressure",
        "temp_roll_mean",
        "pressure_roll_mean",
        "temp_roll_std",
        "pressure_roll_std",
    ]
    assert list(matrix.columns) == columns
    assert len(matrix) == len(feature_df)


def test_build_feature_matrix_ignores_missing_base_features():
    df = _make_df(20)
    feature_df = add_rolling_features(df, feature_cols=["temp"], window=5)

    matrix, columns = build_feature_matrix(feature_df, base_features=["temp", "not_present"])

    assert "not_present" not in columns
    assert columns == ["temp", "temp_roll_mean", "temp_roll_std"]
