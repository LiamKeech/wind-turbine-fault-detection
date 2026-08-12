"""CLI entrypoint for the wind turbine fault detection project.

Wires the existing LOF and LSTM-autoencoder pipeline modules under `src/`
into runnable commands, so the project can be driven without opening a
notebook: preprocessing, training, evaluation, and report generation for
either track.

Examples:
    python main.py preprocess --track lof
    python main.py train --track lof --n-neighbors 25 --contamination 0.02
    python main.py train --track lstm
    python main.py evaluate --track lstm
    python main.py report --track lof
    python main.py report --track lstm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_LOF_RAW_PATH = ROOT / "data" / "raw" / "turbine_5yr_complex_data.csv"
DEFAULT_LOF_OUTPUT_DIR = ROOT / "data" / "processed" / "lof"
DEFAULT_LSTM_CONFIG_PATH = SRC / "config" / "lstm_autoencoder_config.yaml"


# --------------------------------------------------------------------------
# LOF track
# --------------------------------------------------------------------------

def _lof_load_raw(data_path: Path, timestamp_col: str):
    from data.lof_preprocessing import load_raw_data
    return load_raw_data(data_path, timestamp_col=timestamp_col)


def lof_preprocess(args: argparse.Namespace) -> None:
    from data.lof_preprocessing import (
        clean_data,
        dataset_overview,
        missing_rate_summary,
        select_feature_columns,
    )
    from features.lof_features import add_rolling_features, build_feature_matrix

    raw_df = _lof_load_raw(args.data_path, args.timestamp_col)
    base_features = select_feature_columns(raw_df, timestamp_col=args.timestamp_col)

    print("=== Dataset overview (raw) ===")
    print(dataset_overview(raw_df, base_features, timestamp_col=args.timestamp_col).to_string(index=False))
    print("\n=== Missing value summary (raw) ===")
    print(missing_rate_summary(raw_df, base_features).to_string())

    cleaned_df = clean_data(raw_df, base_features, timestamp_col=args.timestamp_col)
    feature_df = add_rolling_features(cleaned_df, base_features, window=args.rolling_window)
    feature_matrix, feature_columns = build_feature_matrix(feature_df, base_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "features.csv"
    feature_matrix.to_csv(out_path, index=False)
    print(f"\nSaved {len(feature_matrix)} rows x {len(feature_columns)} feature columns to {out_path}")


def lof_train(args: argparse.Namespace) -> None:
    import joblib
    from training.lof_training import train_lof

    raw_df = _lof_load_raw(args.data_path, args.timestamp_col)
    results_df, summary_df, artifacts = train_lof(
        raw_df,
        timestamp_col=args.timestamp_col,
        rolling_window=args.rolling_window,
        n_neighbors=args.n_neighbors,
        contamination=args.contamination,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("=== Anomaly summary ===")
    print(summary_df.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_dir / "results.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    joblib.dump(artifacts["model"], args.output_dir / "model.joblib")
    joblib.dump(artifacts["scaler"], args.output_dir / "scaler.joblib")

    metadata = {
        "base_features": artifacts["base_features"],
        "feature_columns": artifacts["feature_columns"],
        "threshold": artifacts["threshold"],
        "train_end": artifacts["train_end"],
        "val_end": artifacts["val_end"],
        "rolling_window": args.rolling_window,
        "timestamp_col": args.timestamp_col,
        "data_path": str(args.data_path),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved model, scaler, results, and metadata to {args.output_dir}")


def _lof_load_artifacts(output_dir: Path):
    import joblib

    model = joblib.load(output_dir / "model.joblib")
    scaler = joblib.load(output_dir / "scaler.joblib")
    with open(output_dir / "metadata.json", "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return model, scaler, metadata


def _lof_score(args: argparse.Namespace):
    from data.lof_preprocessing import clean_data
    from features.lof_features import add_rolling_features, build_feature_matrix

    model, scaler, metadata = _lof_load_artifacts(args.output_dir)
    base_features = metadata["base_features"]
    timestamp_col = metadata["timestamp_col"]

    raw_df = _lof_load_raw(args.data_path, timestamp_col)
    cleaned_df = clean_data(raw_df, base_features, timestamp_col=timestamp_col)
    feature_df = add_rolling_features(cleaned_df, base_features, window=metadata["rolling_window"])
    feature_matrix, _ = build_feature_matrix(feature_df, base_features)

    X = scaler.transform(feature_matrix)
    scores = model.score_samples(X)
    labels = model.predict(X)

    results_df = feature_df.copy()
    results_df["anomaly_score"] = scores
    results_df["is_anomaly"] = labels
    return results_df, metadata


def lof_evaluate(args: argparse.Namespace) -> None:
    from training.lof_training import summarize_anomalies, top_anomalies

    results_df, _ = _lof_score(args)
    summary_df = summarize_anomalies(results_df)

    print("=== Anomaly summary (scored with saved model) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Top anomalies ===")
    print(top_anomalies(results_df, n=args.top_n).to_string(index=False))


def lof_report(args: argparse.Namespace) -> None:
    from visualization.lof_plots import (
        plot_anomaly_rate_by_split,
        plot_anomaly_rate_timeline,
        plot_anomaly_score_histogram,
        plot_anomaly_score_timeline,
        plot_feature_anomalies,
    )
    from training.lof_training import summarize_anomalies, top_anomalies

    results_df, metadata = _lof_score(args)
    summary_df = summarize_anomalies(results_df)
    threshold = metadata["threshold"]
    train_end, val_end = metadata["train_end"], metadata["val_end"]

    report_dir = args.output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "feature_anomalies.png": plot_feature_anomalies(
            results_df, timestamp_col=metadata["timestamp_col"],
            feature_cols=metadata["base_features"], split_indices=(train_end, val_end),
        ),
        "score_timeline.png": plot_anomaly_score_timeline(
            results_df, timestamp_col=metadata["timestamp_col"],
            threshold=threshold, split_indices=(train_end, val_end),
        ),
        "score_histogram.png": plot_anomaly_score_histogram(results_df, threshold=threshold),
        "rate_timeline.png": plot_anomaly_rate_timeline(
            results_df, timestamp_col=metadata["timestamp_col"], split_indices=(train_end, val_end),
        ),
        "rate_by_split.png": plot_anomaly_rate_by_split(results_df, train_end=train_end, val_end=val_end),
    }
    for filename, (fig, _axes) in figures.items():
        fig.savefig(report_dir / filename, dpi=150, bbox_inches="tight")

    summary_df.to_csv(report_dir / "summary.csv", index=False)
    top_anomalies(results_df, n=args.top_n).to_csv(report_dir / "top_anomalies.csv", index=False)

    print(f"Saved {len(figures)} figures and summary/top-anomaly CSVs to {report_dir}")


# --------------------------------------------------------------------------
# LSTM autoencoder track
# --------------------------------------------------------------------------

def _lstm_prepare(config_path: Path, save_processed: bool = False):
    """Load config + raw data and run cleaning/feature engineering.

    Returns (config, data_cfg, train_cfg, feat_df, feature_cols).
    """
    from data.lstm_autoencoder_preprocessing import (
        clean_raw_data,
        load_raw_data,
        save_processed_data,
    )
    from features.lstm_autoencoder_features import engineer_features
    from models.lstm_autoencoder import load_config

    config = load_config(config_path)
    data_cfg, train_cfg = config["data"], config["training"]
    timestamp_col, label_col = data_cfg["timestamp_col"], data_cfg["label_col"]

    raw_df = load_raw_data(ROOT / data_cfg["raw_path"], timestamp_col)
    clean_df = clean_raw_data(raw_df, timestamp_col, label_col)
    feat_df, feature_cols = engineer_features(clean_df)

    if save_processed:
        out_path = save_processed_data(feat_df, ROOT / data_cfg["processed_dir"], data_cfg["processed_filename"])
        print(f"Saved processed features to {out_path}")

    return config, data_cfg, train_cfg, feat_df, feature_cols


def _lstm_normal_train_val_split(feat_df, data_cfg, timestamp_col, label_col):
    import numpy as np
    from data.lstm_autoencoder_preprocessing import split_train_val

    anomaly_mask = feat_df[label_col].astype(int) == 1
    first_anomaly_pos = int(np.where(anomaly_mask.values)[0][0]) if anomaly_mask.any() else len(feat_df)
    normal_df = feat_df.iloc[:first_anomaly_pos].copy()
    if normal_df.empty:
        raise ValueError("No normal data found before the first labeled anomaly.")
    return split_train_val(normal_df, float(data_cfg["val_split"]), timestamp_col=timestamp_col)


def lstm_preprocess(args: argparse.Namespace) -> None:
    _lstm_prepare(args.config, save_processed=True)


def lstm_train(args: argparse.Namespace) -> None:
    import torch
    from data.lstm_autoencoder_preprocessing import (
        create_sequences,
        fit_scaler,
        save_scaler,
        scale_features,
    )
    from models.lstm_autoencoder import LSTMAutoencoder
    from training.lstm_autoencoder_training import (
        prepare_dataloaders,
        resolve_device,
        save_model,
        train_lstm_autoencoder,
    )

    config, data_cfg, train_cfg, feat_df, feature_cols = _lstm_prepare(args.config, save_processed=True)
    timestamp_col, label_col = data_cfg["timestamp_col"], data_cfg["label_col"]

    train_df, val_df = _lstm_normal_train_val_split(feat_df, data_cfg, timestamp_col, label_col)
    scaler = fit_scaler(train_df, feature_cols, label_col=label_col, normal_value=0)

    window_size, stride = int(data_cfg["window_size"]), int(data_cfg["stride"])
    train_seq = create_sequences(scale_features(train_df, feature_cols, scaler), window_size, stride)
    val_seq = create_sequences(scale_features(val_df, feature_cols, scaler), window_size, stride)
    train_loader, val_loader = prepare_dataloaders(train_seq, val_seq, train_cfg["batch_size"])

    device = resolve_device(train_cfg["device"])
    model = LSTMAutoencoder.from_config(args.config).to(device)
    history = train_lstm_autoencoder(model, train_loader, val_loader, train_cfg, device)

    model_path = save_model(model, ROOT / train_cfg["model_path"])
    scaler_path = save_scaler(scaler, ROOT / train_cfg["scaler_path"])
    history_path = model_path.parent / "history.json"
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved training history to {history_path}")


def _lstm_load_trained(config_path: Path):
    import joblib
    import torch
    from models.lstm_autoencoder import LSTMAutoencoder
    from training.lstm_autoencoder_training import resolve_device

    config, data_cfg, train_cfg, feat_df, feature_cols = _lstm_prepare(config_path, save_processed=False)
    device = resolve_device(train_cfg["device"])

    model = LSTMAutoencoder.from_config(config_path).to(device)
    model.load_state_dict(torch.load(ROOT / train_cfg["model_path"], map_location=device))
    model.eval()
    scaler = joblib.load(ROOT / train_cfg["scaler_path"])

    history_path = (ROOT / train_cfg["model_path"]).parent / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else None

    return config, data_cfg, train_cfg, feat_df, feature_cols, device, model, scaler, history


def _lstm_run_evaluate(args: argparse.Namespace):
    from data.lstm_autoencoder_preprocessing import create_sequences, scale_features
    from evaluation.lstm_autoencoder_evaluation import evaluate

    config, data_cfg, train_cfg, feat_df, feature_cols, device, model, scaler, history = _lstm_load_trained(args.config)
    timestamp_col, label_col = data_cfg["timestamp_col"], data_cfg["label_col"]
    window_size, stride = int(data_cfg["window_size"]), int(data_cfg["stride"])

    train_df, _val_df = _lstm_normal_train_val_split(feat_df, data_cfg, timestamp_col, label_col)
    train_seq = create_sequences(scale_features(train_df, feature_cols, scaler), window_size, stride)

    eval_df = feat_df.copy()
    eval_seq = create_sequences(scale_features(eval_df, feature_cols, scaler), window_size, stride)

    exclude_features = args.exclude_features.split(",") if args.exclude_features else None
    results = evaluate(
        model=model,
        train_sequences=train_seq,
        eval_sequences=eval_seq,
        eval_df=eval_df,
        feature_cols=feature_cols,
        label_col=label_col,
        timestamp_col=timestamp_col,
        window_size=window_size,
        stride=stride,
        batch_size=train_cfg["batch_size"],
        device=device,
        threshold_quantile=args.threshold_quantile,
        label_aggregation="any",
        exclude_features=exclude_features,
    )
    return results, history


def lstm_evaluate(args: argparse.Namespace) -> None:
    results, _history = _lstm_run_evaluate(args)

    print("=== Sequence-level metrics ===")
    for key, value in results["seq_metrics"].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n=== Row-level metrics ===")
    for key, value in results["row_metrics"].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def lstm_report(args: argparse.Namespace) -> None:
    from evaluation.lstm_autoencoder_evaluation import evaluate_and_plot
    from models.lstm_autoencoder import load_config

    results, history = _lstm_run_evaluate(args)

    processed_dir = load_config(args.config)["data"]["processed_dir"]
    report_dir = ROOT / processed_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    evaluate_and_plot(
        results,
        training_history=history,
        fig_title="Wind Turbine LSTM Autoencoder - Evaluation",
        save_path=str(report_dir / "evaluation_dashboard.png"),
    )

    metrics = {"seq_metrics": results["seq_metrics"], "row_metrics": results["row_metrics"]}
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved evaluation dashboard and metrics to {report_dir}")


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------

def _add_lof_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=Path, default=DEFAULT_LOF_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOF_OUTPUT_DIR)
    parser.add_argument("--timestamp-col", default="timestamp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Wind turbine fault detection: LOF and LSTM-autoencoder pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("preprocess", "Load raw data and materialize cleaned/engineered features."),
        ("train", "Train a model end-to-end and save its artifacts."),
        ("evaluate", "Score data with a previously trained model and print metrics."),
        ("report", "Evaluate and write plots/metrics/CSVs to disk."),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--track", choices=["lof", "lstm"], required=True)
        sub.set_defaults(_command=name)

        if name in ("train", "evaluate", "report", "preprocess"):
            # LOF-specific args (ignored for --track lstm)
            _add_lof_data_args(sub)
        # LSTM-specific args (ignored for --track lof)
        sub.add_argument("--config", type=Path, default=DEFAULT_LSTM_CONFIG_PATH)

        if name == "train":
            sub.add_argument("--rolling-window", type=int, default=60)
            sub.add_argument("--n-neighbors", type=int, default=20)
            sub.add_argument("--contamination", type=float, default=0.01)
            sub.add_argument("--train-frac", type=float, default=0.7)
            sub.add_argument("--val-frac", type=float, default=0.15)
        elif name == "preprocess":
            sub.add_argument("--rolling-window", type=int, default=60)
        elif name in ("evaluate", "report"):
            sub.add_argument("--top-n", type=int, default=10)
            sub.add_argument("--threshold-quantile", type=float, default=0.999)
            sub.add_argument("--exclude-features", default="particle_count_delta",
                              help="Comma-separated LSTM feature columns to exclude from thresholding; "
                                   "empty string to include all.")

    return parser


DISPATCH = {
    ("preprocess", "lof"): lof_preprocess,
    ("preprocess", "lstm"): lstm_preprocess,
    ("train", "lof"): lof_train,
    ("train", "lstm"): lstm_train,
    ("evaluate", "lof"): lof_evaluate,
    ("evaluate", "lstm"): lstm_evaluate,
    ("report", "lof"): lof_report,
    ("report", "lstm"): lstm_report,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = DISPATCH[(args._command, args.track)]
    handler(args)


if __name__ == "__main__":
    main()
