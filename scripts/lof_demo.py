"""Run a minimal LOF training and scoring demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PathConfig, SENSOR_FEATURE_COLUMNS, TIMESTAMP_COLUMN
from models import LofDetector


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="LOF demo for turbine data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PathConfig.RAW_DATA / "turbine_5yr_complex_data.csv",
        help="Path to input CSV.",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Row limit.")
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--output-version", type=str, default="v1.0.0")
    return parser.parse_args()


def time_based_split(
    data: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets by time.

    Args:
        data: Sorted DataFrame.
        train_frac: Fraction for training split.
        val_frac: Fraction for validation split.

    Returns:
        Tuple of train, validation, and test DataFrames.
    """
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac and val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1")
    n_samples = len(data)
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    return train, val, test


def main() -> None:
    """Run LOF training, thresholding, and scoring."""
    args = parse_args()
    PathConfig.ensure_dirs_exist()

    data = pd.read_csv(args.data_path, parse_dates=[TIMESTAMP_COLUMN])
    data = data.sort_values(TIMESTAMP_COLUMN)
    data = data.dropna(subset=SENSOR_FEATURE_COLUMNS)

    if args.limit > 0:
        data = data.head(args.limit)

    train, val, test = time_based_split(data)

    detector = LofDetector(
        n_neighbors=args.n_neighbors, contamination=args.contamination
    ).fit(LofDetector.select_features(train, SENSOR_FEATURE_COLUMNS))

    val_scores = detector.score_samples(
        LofDetector.select_features(val, SENSOR_FEATURE_COLUMNS)
    )
    threshold = detector.compute_threshold(val_scores, args.threshold_quantile)

    test_scores, test_flags = detector.predict(
        LofDetector.select_features(test, SENSOR_FEATURE_COLUMNS), threshold
    )

    output_dir = PathConfig.LOF_MODEL_DIR / args.output_version
    model_path = output_dir / "model.joblib"
    detector.save(model_path)

    metadata = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "threshold": threshold,
        "feature_columns": SENSOR_FEATURE_COLUMNS,
        "n_neighbors": args.n_neighbors,
        "contamination": args.contamination,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("LOF demo complete")
    print(f"Validation threshold: {threshold:.6f}")
    print(f"Test anomalies: {int(test_flags.sum())} / {len(test_flags)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
