# Wind Turbine Fault Detection - Project Overview

## Purpose
This project detects anomalies in wind turbine drivetrain signals to surface early fault indicators and reduce unplanned downtime. The system focuses on time series sensor data and supports offline training, evaluation, and batch scoring.

## Primary Stack
- Python 3.10+
- Scikit-Learn
- Pandas
- NumPy

## Dataset Schema
Core columns (features and metadata):
- timestamp
- gearbox_oil_temp
- gearbox_bearing_temp
- vibration_x
- vibration_y
- vibration_z
- oil_pressure
- particle_count

## Software Architecture
Layered, pipeline-oriented design with clean boundaries between data, features, models, and services:

1) Data Layer
	 - Ingestion: read raw sensor data from CSV/Parquet.
	 - Validation: schema checks, missing values, monotonic timestamps.
	 - Persistence: write curated datasets and feature tables.

2) Feature Layer
	 - Windowing and aggregation (rolling stats, percentiles).
	 - Signal transforms (z-score, lag features, deltas).
	 - Feature store artifacts saved with metadata.

3) Modeling Layer
	 - Unsupervised or semi-supervised anomaly detection.
	 - Baselines: IsolationForest, OneClassSVM, LocalOutlierFactor.
	 - Model selection with time-based cross-validation.

4) Evaluation Layer
	 - Threshold tuning and stability metrics.
	 - Backtesting on historical windows.
	 - Drift detection and monitoring reports.

5) Serving Layer (Batch)
	 - Batch scoring jobs for new data.
	 - Artifact versioning and reproducible runs.

6) Orchestration and Experiments
	 - Reproducible pipelines and experiment tracking.
	 - Config-driven runs (YAML/JSON).

## Professional Folder Structure
```
wind-turbine-fault-detection/
	data/
		raw/
		interim/
		processed/
		external/
	docs/
		reports/
		diagrams/
	notebooks/
		exploration/
		experiments/
	src/
		fault_detection/
			__init__.py
			config/
				settings.py
			data/
				ingestion.py
				validation.py
				schemas.py
			features/
				build_features.py
				transforms.py
			models/
				train.py
				evaluate.py
				predict.py
				registry.py
			monitoring/
				drift.py
				metrics.py
			pipelines/
				train_pipeline.py
				score_pipeline.py
			utils/
				io.py
				logging.py
				time.py
	tests/
		unit/
		integration/
	scripts/
		train.py
		score.py
	configs/
		base.yaml
		train.yaml
		score.yaml
	outputs/
		models/
		reports/
		figures/
	pyproject.toml
	README.md
	OVERVIEW.md
```

## Typing and Docstrings
- All functions must use type hints.
- Use Google style docstrings for all public functions and modules.

Example:
```python
from typing import Iterable
import numpy as np

def train(X: np.ndarray, *, contamination: float) -> None:
		"""Train the anomaly detection model.

		Args:
				X: Feature matrix of shape (n_samples, n_features).
				contamination: Expected fraction of anomalies.

		Returns:
				None
		"""
		...
```

## Modeling Guidelines
- Prefer time-based splits to avoid leakage.
- Normalize or standardize features consistently between train and score.
- Keep a baseline model for regression testing.
- Use robust metrics for imbalanced data (e.g., precision at k, PR AUC).

## Data Quality and Validation
- Enforce schema: dtypes, ranges, and monotonic timestamps.
- Record missingness and outlier statistics per batch.
- Keep raw data immutable; only write to processed folders.

## Reproducibility
- Fix random seeds for model training.
- Log configs, code version, and feature definitions with each run.
- Store artifacts using semantic versioning.

## Security and Safety
- Validate file paths and input configs.
- Avoid executing untrusted notebooks in automated pipelines.

## Notes
- This project targets anomaly detection for wind turbine drivetrain signals.
- The dataset columns used for features are listed under Dataset Schema.
