# Wind Turbine Fault Detection - Machine Learning Project Blueprint

## Project Purpose
This project implements two anomaly detection methods to identify faults in wind turbine gearbox data, enabling early fault detection and reducing unplanned downtime. The system processes multivariate time series sensor data using both deep learning (LSTM Autoencoder) and statistical (Local Outlier Factor) approaches for robust fault detection.

## Core Objectives
1. **LSTM Autoencoder**: Capture temporal dependencies and nonlinear patterns in multivariate time series
2. **Local Outlier Factor (LOF)**: Detect local density-based anomalies in feature space
3. **Comparative Analysis**: Evaluate method performance, false positive rates, and deployment trade-offs
4. **Production Readiness**: Config-driven pipeline with reproducible experiments and batch scoring

## Technology Stack
- **Python**: 3.10+
- **Deep Learning**: PyTorch (LSTM Autoencoder)
- **Traditional ML**: Scikit-Learn (Local Outlier Factor (LOF))
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Containerization**: Docker

## Dataset Schema
### Core Sensor Columns
| Column | Type | Description | Unit |
|--------|------|-------------|------|
| timestamp | datetime | UTC timestamp of measurement | ISO 8601 |
| gearbox_oil_temp | float | Temperature of gearbox oil | Celsius |
| gearbox_bearing_temp | float | Temperature of gearbox bearing | Celsius |
| vibration_x | float | Vibration amplitude along X-axis | mm/s |
| vibration_y | float | Vibration amplitude along Y-axis | mm/s |
| vibration_z | float | Vibration amplitude along Z-axis | mm/s |
| oil_pressure | float | Hydraulic pressure in oil system | bar |
| particle_count | int | Number of particles in oil (contamination indicator) | count |

## Software Architecture
Six-layer pipeline-oriented design with clean boundaries and configuration-driven execution:

### 1) Data Ingestion Layer
- **CSV/Parquet Loading**: Read raw sensor data from `data/raw/` with configurable paths
- **Schema Validation**: Enforce dtypes, value ranges, and timestamp monotonicity
- **Missing Value Handling**: Document, impute, or exclude rows based on thresholds
- **Persistence**: Write validated data to `data/interim/`

### 2) Data Preprocessing Layer
- **Deduplication**: Remove duplicate records based on timestamp
- **Outlier Detection**: Flag statistical outliers (e.g., 3σ rule) for review
- **Time Alignment**: Resample irregular sampling intervals
- **Stationarity Checks**: Log-transform skewed distributions
- **Output**: `data/processed/` with metadata tracking

### 3) Feature Engineering Layer
- **Windowing**: Create sliding windows (e.g., 60-second, 5-minute windows)
- **Rolling Statistics**: Min, max, mean, std, median per window across all sensor columns
- **Lag Features**: Previous values for time-lagged dependencies (LOF method)
- **Signal Transforms**:
  - Z-score normalization per sensor
  - First/second derivatives (rate of change)
  - FFT-based frequency domain features (optional)
  - Entropy and approximate entropy (signal complexity)
- **Feature Store**: Save normalized feature matrices with column metadata

### 4) Anomaly Detection Methods

#### Method A: LSTM Autoencoder
- **Architecture**:
  - Encoder: 2-3 stacked LSTM layers → compressed latent representation
  - Decoder: 2-3 stacked LSTM layers → reconstructed time series
  - Reconstruction loss (MSE) as anomaly score
- **Hyperparameters**:
  - Sequence length (window size): 30-60 timesteps
  - Latent dimension: 8-16
  - LSTM hidden units: 32-64 per layer
  - Dropout: 0.2-0.3 (regularization)
  - Batch size: 32-64
  - Learning rate: 1e-3 to 1e-4 (Adam optimizer)
  - Epochs: Early stopping on validation loss
- **Advantages**: Captures temporal patterns; handles multivariate correlations
- **Training**:
  - Time-based split: 70% train, 15% validation, 15% test
  - Normal data only (unsupervised)
  - Standardize features using training set statistics
- **Inference**: Reconstruction error threshold tuned on validation data

#### Method B: Local Outlier Factor (LOF)
- **Algorithm**:
  - Scikit-learn LOF with k-nearest neighbors
  - Density-based local outlier detection
  - Anomaly scores: LOF value > threshold → anomaly
- **Hyperparameters**:
  - n_neighbors: 20-50 (typically 5-10% of training set)
  - contamination: 0.05-0.15 (expected fraction of anomalies)
  - metric: Euclidean (scaled features)
- **Advantages**: Computationally efficient; interpretable; no temporal assumption
- **Training**:
  - Fit on rolling windows or entire training set
  - Standardize features (StandardScaler)
- **Inference**: Direct LOF score output for new samples

### 5) Model Evaluation Layer
- **Train/Validation/Test Splits**: Time-based (no future leakage)
- **Metrics**:
  - Reconstruction error distribution (LSTM): mean, std, percentiles
  - LOF score distribution: mean, std, percentiles
  - If labeled data available: Precision, Recall, F1-score, ROC-AUC, PR-AUC
  - Threshold optimization: Maximize F1 or precision at k
- **Validation**:
  - Cross-validation: Time-series aware (e.g., expanding window CV)
  - Stability: Scores consistent across time windows
  - Drift detection: Monitor score distribution shift over time
- **Comparative Analysis**:
  - Side-by-side method performance
  - False positive rate analysis
  - Computational cost and latency
  - Ensemble strategy (e.g., voting, weighted combination)
- **Outputs**: Reports, confusion matrices, ROC/PR curves

### 6) Batch Scoring and Serving Layer
- **Batch Prediction Pipeline**:
  - Load new data from `data/raw/` or streaming source
  - Apply preprocessing and feature engineering (consistent with training)
  - Load trained models (LSTM + LOF)
  - Generate anomaly scores and flags
  - Write results to `outputs/predictions/`
- **Artifact Versioning**:
  - Model checkpoints with timestamp/semantic version (v1.0.0)
  - Feature transformers (StandardScaler, PCA, etc.) saved as pickle
  - Configuration snapshots (model hyperparams, feature list)
- **Reproducibility**:
  - Fixed random seeds (NumPy, PyTorch, Scikit-Learn)
  - Logged config, data versions, code commit hash
  - Deterministic preprocessing order

### 7) Orchestration and Experiment Tracking
- **Config-Driven Execution**:
  - `configs/base.yaml`: Default paths, common hyperparams
  - `configs/train.yaml`: Training-specific overrides (epochs, learning rate)
  - `configs/score.yaml`: Inference-specific config (batch size, model path)
- **Environment Variables**: All file paths via `${DATA_DIR}`, `${MODEL_DIR}` patterns
- **Experiment Logging**:
  - MLflow or manual JSON logging: model params, metrics, artifacts
  - Notebook-based exploration captured in `notebooks/`

## Professional Folder Structure
```
wind-turbine-fault-detection/
│
├── README.md                           # Project documentation
├── OVERVIEW.md                         # This file
├── pyproject.toml                      # Package and dependency configuration
├── Dockerfile                          # Container image definition
├── .dockerignore                       # Files to exclude from Docker build
│
├── data/
│   ├── raw/                            # Immutable raw sensor data (CSV/Parquet)
│   ├── interim/                        # Validated, deduplicated data
│   ├── processed/                      # Normalized, feature-engineered data
│   └── external/                       # Reference datasets (optional)
│
├── src/
│   └── fault_detection/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py             # Pydantic config models for YAML loading
│       │   ├── paths.py                # Centralized path management (no hardcoding)
│       │   └── constants.py            # Feature names, column mappings
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py              # CSV/Parquet ingestion functions
│       │   ├── validation.py           # Schema and data quality checks
│       │   ├── preprocessing.py        # Deduplication, outlier handling, alignment
│       │   └── schemas.py              # Pydantic schemas for validation
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── engineering.py          # Windowing, rolling stats, transforms
│       │   ├── normalization.py        # StandardScaler, feature scaling
│       │   └── store.py                # Feature storage and retrieval
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lstm_autoencoder.py     # LSTM Autoencoder architecture (PyTorch)
│       │   ├── lof.py                  # LOF wrapper with training/prediction
│       │   ├── train.py                # Training loops for both methods
│       │   ├── evaluate.py             # Evaluation metrics and threshold tuning
│       │   ├── predict.py              # Inference for both models
│       │   └── registry.py             # Model loading/saving with versioning
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── drift.py                # Data/concept drift detection
│       │   ├── metrics.py              # Runtime metric computation
│       │   └── alerting.py             # Threshold-based alerts (optional)
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── train_pipeline.py       # End-to-end training orchestration
│       │   ├── score_pipeline.py       # End-to-end batch scoring
│       │   └── evaluation_pipeline.py  # Evaluation and comparison
│       │
│       └── utils/
│           ├── __init__.py
│           ├── io.py                   # File I/O wrappers (paths from config)
│           ├── logging.py              # Structured logging utilities
│           ├── time.py                 # Timestamp handling, window creation
│           └── device.py               # PyTorch device management (CPU/GPU)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: load, visualize sensor data
│   ├── 02_preprocessing.ipynb          # Deduplication, outlier analysis
│   ├── 03_feature_engineering.ipynb    # Feature creation and analysis
│   ├── 04_lstm_autoencoder_demo.ipynb  # LSTM training, tuning, visualization
│   ├── 05_lof_demo.ipynb               # LOF training, tuning, visualization
│   ├── 06_model_comparison.ipynb       # Side-by-side evaluation of methods
│   └── 07_deployment_guide.ipynb       # Instructions for batch scoring
│
├── scripts/
│   ├── train.py                        # CLI entry point for model training
│   ├── score.py                        # CLI entry point for batch scoring
│   └── evaluate.py                     # CLI entry point for evaluation
│
├── configs/
│   ├── base.yaml                       # Base configuration (paths, common hyperparams)
│   ├── train_lstm.yaml                 # LSTM-specific training config
│   ├── train_lof.yaml                  # LOF-specific training config
│   └── score.yaml                      # Batch scoring configuration
│
├── outputs/
│   ├── models/                         # Saved model artifacts (v1.0.0/, v1.0.1/, etc.)
│   │   ├── lstm_v1.0.0/
│   │   │   ├── model.pt                # PyTorch state dict
│   │   │   ├── config.json             # Training hyperparameters
│   │   │   ├── scaler.pkl              # Feature standardizer
│   │   │   └── metadata.json           # Training data info, timestamp
│   │   └── lof_v1.0.0/
│   │       ├── model.pkl               # Scikit-learn LOF model
│   │       ├── config.json
│   │       ├── scaler.pkl
│   │       └── metadata.json
│   │
│   ├── predictions/                    # Batch scoring output
│   │   ├── scores_2024_01_15.csv       # Timestamps, scores, flags
│   │   └── report_2024_01_15.json      # Aggregated metrics, summary
│   │
│   └── reports/                        # Analysis reports, figures
│       ├── model_evaluation_report.md
│       ├── drift_analysis_2024_01.pdf
│       └── figures/
│
├── tests/
│   ├── unit/
│   │   ├── test_data_loaders.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_models.py
│   │   ├── test_evaluation.py
│   │   └── test_config.py
│   │
│   └── integration/
│       ├── test_train_pipeline.py
│       └── test_score_pipeline.py
│
└── .gitignore                          # Standard Python + data files exclusions
```

## Configuration Management (No Hardcoded Paths)
All file paths are centralized in `src/fault_detection/config/paths.py`:

```python
from pathlib import Path
from typing import Dict
import os

class PathConfig:
    """Centralized path management with environment variable support."""
    
    # Root directories from environment or defaults
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
    DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    CONFIG_DIR = Path(os.getenv("CONFIG_DIR", PROJECT_ROOT / "configs"))
    
    # Data subdirectories
    RAW_DATA = DATA_DIR / "raw"
    INTERIM_DATA = DATA_DIR / "interim"
    PROCESSED_DATA = DATA_DIR / "processed"
    EXTERNAL_DATA = DATA_DIR / "external"
    
    # Model artifacts
    MODELS_DIR = OUTPUT_DIR / "models"
    LSTM_MODEL_DIR = MODELS_DIR / "lstm"
    LOF_MODEL_DIR = MODELS_DIR / "lof"
    
    # Outputs
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"
    
    @staticmethod
    def ensure_dirs_exist() -> None:
        """Create all required directories if missing."""
        for path in [PathConfig.RAW_DATA, PathConfig.INTERIM_DATA, 
                     PathConfig.PROCESSED_DATA, PathConfig.MODELS_DIR,
                     PathConfig.PREDICTIONS_DIR, PathConfig.REPORTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
```

Usage in code:
```python
from fault_detection.config.paths import PathConfig

df = pd.read_csv(PathConfig.RAW_DATA / "sensor_data.csv")
model_path = PathConfig.LSTM_MODEL_DIR / "v1.0.0" / "model.pt"
```

## Typing and Docstrings
All functions use strict type hints and Google-style docstrings:

```python
from typing import Tuple, Optional
import numpy as np
import pandas as pd

def create_sliding_windows(
    data: pd.DataFrame,
    window_size: int,
    step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series data.
    
    Args:
        data: Input DataFrame with shape (n_samples, n_features).
        window_size: Size of each window in samples.
        step_size: Stride for sliding window.
    
    Returns:
        windows: Array of shape (n_windows, window_size, n_features).
        indices: Start indices for each window.
    
    Raises:
        ValueError: If window_size > len(data).
    """
    ...
```

## Model Training Guidelines

### LSTM Autoencoder
1. **Data Preparation**:
   - Use normal operation data only (unsupervised)
   - Create overlapping sliding windows (sequence_length=60)
   - Standardize with training set statistics (fit StandardScaler on train only)
   - Time-based split: 70/15/15 (train/val/test)

2. **Training**:
   - Loss function: Reconstruction MSE (no labels)
   - Optimizer: Adam with learning rate decay (initial 1e-3)
   - Early stopping: Monitor validation loss (patience=5 epochs)
   - Checkpointing: Save best model by validation loss
   - Batch size: 32-64 (depending on GPU memory)

3. **Threshold Tuning**:
   - Compute reconstruction error on validation set
   - Set threshold to percentile (e.g., 95th) for unsupervised setting
   - Optimize threshold on test set if labeled anomalies available

### Local Outlier Factor (LOF)
1. **Data Preparation**:
   - Aggregate features to sample level or use windowed features
   - Standardize features (StandardScaler fit on training set)
   - Time-based split: 70/15/15

2. **Training**:
   - Fit LOF on training set: `lof.fit(X_train)`
   - n_neighbors: 20-50 (data-dependent)
   - contamination: 0.05-0.15 (tune or set from domain knowledge)

3. **Scoring**:
   - Predict anomaly labels: `-1 = anomaly, 1 = normal`
   - Output LOF scores (negative distance to LOF value)
   - Threshold: Adjust contamination parameter or use score quantile

## Model Evaluation and Comparison
- **Metrics** (if labeled data):
  - Precision, Recall, F1-score, ROC-AUC, PR-AUC
  - Confusion matrix, precision-at-k
- **Metrics** (unsupervised):
  - Error/score distributions (mean, std, percentiles)
  - Temporal consistency (scores stable across similar operational modes)
  - Computational efficiency (training time, inference latency)
- **Cross-Validation**: Time-series aware expanding window CV
- **Drift Monitoring**: Detect distribution shifts in score outputs
- **Ensemble Strategy** (optional): Combine both methods (e.g., voting)

## Data Quality and Validation

### Ingestion Checks
- Schema validation: dtypes, non-null counts per column
- Timestamp monotonicity and duplication detection
- Value range checks (e.g., temperature bounds, pressure limits)
- Missing value statistics logged per batch

### Processing Checks
- Outlier flagging: 3-sigma rule or IQR method
- Stationarity assessment (ADF test, optional)
- Feature correlation analysis (multicollinearity)
- Data leakage detection (no future data in training)

### Output Validation
- Model predictions shape and type checks
- Score distribution sanity checks
- Reproducibility tests (identical inputs → identical outputs)

## Reproducibility
- **Random Seeds**: Set seed in NumPy, PyTorch, and Scikit-Learn at pipeline start
- **Deterministic Operations**: Use `torch.backends.cudnn.deterministic = True`
- **Config Snapshots**: Save YAML/JSON with every trained model
- **Data Versioning**: Track input data hash or git commit
- **Experiment Logs**: Record hyperparameters, metrics, code version with each run
- **Artifact Management**: Semantic versioning (v1.0.0, v1.0.1, v1.1.0)

## Security and Safety
- **Input Validation**: File path checks, prevent path traversal
- **Config Validation**: Use Pydantic to validate YAML/JSON structure
- **Untrusted Code**: Never auto-execute notebooks in production pipelines
- **Error Handling**: Graceful failures with informative logging
- **Data Privacy**: Ensure compliance with data retention policies (optional)

## Containerization

### Docker Overview
Containerization enables consistent execution across development, testing, and production environments. All dependencies are pinned, and the application runs in isolation.

### Dockerfile Strategy
```dockerfile
# Multi-stage build for optimized image size
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage: runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data/output directories
RUN mkdir -p data/raw data/interim data/processed outputs/models outputs/predictions

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    OUTPUT_DIR=/app/outputs \
    CONFIG_DIR=/app/configs

# Default command: training pipeline
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/train.yaml"]
```

### .dockerignore Configuration
Exclude unnecessary files to reduce image size and build context:
```
# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.pytest_cache/
.coverage
.tox/
.hypothesis/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Data (large files, not needed in container)
data/raw/
data/interim/
outputs/predictions/
outputs/reports/

# Git and documentation
.git/
.gitignore
.github/
README.md
OVERVIEW.md

# Jupyter
.ipynb_checkpoints/
notebooks/

# OS
.DS_Store
Thumbs.db
```

### Docker Execution Examples
```bash
# Build image
docker build -t wind-turbine-fault-detection:latest .

# Training with custom config
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  wind-turbine-fault-detection:latest \
  python scripts/train.py --config configs/train_lstm.yaml

# Batch scoring
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  wind-turbine-fault-detection:latest \
  python scripts/score.py --config configs/score.yaml

# Interactive shell for debugging
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  wind-turbine-fault-detection:latest /bin/bash
```

### Dependencies Management
`requirements.txt` (pinned versions for reproducibility):
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
torch==2.0.1
pydantic==2.0.0
pyyaml==6.0
pytest==7.4.0
matplotlib==3.7.2
seaborn==0.12.2
mlflow==2.5.0  # Optional
```

## Notes
- This project targets production-grade anomaly detection using two complementary methods
- Deep learning (LSTM) captures temporal patterns; statistical (LOF) provides interpretability
- All paths are config-driven; no hardcoded values
- Reproducibility is enforced through seed management and artifact versioning
- Docker enables seamless deployment and environment consistency

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
