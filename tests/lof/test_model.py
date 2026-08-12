import numpy as np
import pytest

from models.lof import LOFAnomalyDetector, LOFConfig


def _make_cluster(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, 2))


def test_lof_config_defaults():
    config = LOFConfig()
    assert config.n_neighbors == 20
    assert config.contamination == 0.01
    assert config.metric == "minkowski"


def test_detector_uses_default_config_when_none_given():
    detector = LOFAnomalyDetector()
    assert detector.config.n_neighbors == 20
    assert detector.threshold_ is None


def test_predict_before_fit_raises():
    detector = LOFAnomalyDetector(LOFConfig(n_neighbors=5))
    with pytest.raises(ValueError):
        detector.predict(_make_cluster(10))


def test_fit_returns_self_and_sets_threshold():
    detector = LOFAnomalyDetector(LOFConfig(n_neighbors=5, contamination=0.1))
    X = _make_cluster(50)

    result = detector.fit(X)

    assert result is detector
    assert detector.threshold_ is not None
    assert isinstance(detector.threshold_, float)


def test_score_samples_returns_one_score_per_row():
    detector = LOFAnomalyDetector(LOFConfig(n_neighbors=5, contamination=0.1)).fit(_make_cluster(50))
    scores = detector.score_samples(_make_cluster(10, seed=1))
    assert scores.shape == (10,)


def test_predict_flags_clear_outlier_as_anomaly():
    train_X = _make_cluster(50)
    detector = LOFAnomalyDetector(LOFConfig(n_neighbors=5, contamination=0.1)).fit(train_X)

    test_X = np.vstack([_make_cluster(5, seed=2), np.array([[100.0, 100.0]])])
    labels = detector.predict(test_X)

    assert labels.shape == (6,)
    assert set(np.unique(labels)).issubset({0, 1})
    assert labels[-1] == 1  # the far-away point must be flagged anomalous


def test_predict_matches_manual_threshold_comparison():
    detector = LOFAnomalyDetector(LOFConfig(n_neighbors=5, contamination=0.1)).fit(_make_cluster(50))
    test_X = _make_cluster(10, seed=3)

    scores = detector.score_samples(test_X)
    labels = detector.predict(test_X)

    np.testing.assert_array_equal(labels, (scores >= detector.threshold_).astype(int))
