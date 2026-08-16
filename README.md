# Wind Turbine Fault Detection

Two complementary anomaly-detection pipelines for identifying faults in wind
turbine gearbox/drivetrain sensor data: an **LSTM Autoencoder** (deep
learning, sequence reconstruction) and a **Local Outlier Factor (LOF)**
model (density-based, statistical). Both process multivariate time-series
sensor data and flag anomalous readings that indicate early-stage faults,
aiming to reduce unplanned downtime.

## Dataset

Raw sensor readings live in `data/raw/` (Git LFS-tracked CSVs, one 10-minute
sample per row over ~5 years):

| Column | Type | Description | Unit |
|---|---|---|---|
| `timestamp` | datetime | UTC timestamp of measurement | ISO 8601 |
| `gearbox_oil_temp` | float | Gearbox oil temperature | °C |
| `gearbox_bearing_temp` | float | Gearbox bearing temperature | °C |
| `vibration_x` / `_y` / `_z` | float | Vibration amplitude per axis | mm/s |
| `oil_pressure` | float | Hydraulic oil system pressure | bar |
| `particle_count` | int | Particle count in oil (contamination indicator) | count |

- `turbine_5yr_complex_data.csv` — unlabeled, used by the LOF track.
- `turbine_5yr_labeled_data.csv` — includes an `is_anomaly` label column, used by the LSTM track for threshold tuning and evaluation.

## Repository layout

```
wind-turbine-fault-detection/
├── main.py                      # CLI entrypoint for both tracks
├── requirements.txt             # pinned runtime dependencies
├── requirements-dev.txt         # + pytest, for running tests/
├── Dockerfile / docker-compose.yml
├── conftest.py                  # puts src/ on sys.path for pytest
├── data/
│   ├── raw/                     # source CSVs (Git LFS)
│   └── processed/               # pipeline outputs (LFS: *.csv/.pt/.pkl/.joblib)
│       ├── lof/                 # created by `main.py ... --track lof`
│       └── lstm_autoencoder/    # processed_features.csv, model.pt, scaler.pkl
├── notebooks/
│   ├── LOF Anomaly Detection.ipynb
│   └── LSTM Autoencoder.ipynb
├── src/
│   ├── config/lstm_autoencoder_config.yaml
│   ├── data/                    # loading, cleaning, splitting
│   ├── features/                # feature engineering
│   ├── models/                  # LOFAnomalyDetector, LSTMAutoencoder
│   ├── training/                # training loops
│   ├── evaluation/              # LSTM metrics + evaluation dashboard
│   └── visualization/           # LOF plots
└── tests/
    ├── lof/                     # unit tests for the LOF track
    └── lstm/                    # unit tests for the LSTM autoencoder track
```

## Setup

```bash
git clone <repo-url> && cd wind-turbine-fault-detection
git lfs install && git lfs pull
```

**Required:** run `git lfs pull` before running anything in this repo — the
dataset CSVs and saved model artefacts are Git LFS-tracked, and without it
you'll only have pointer stub files instead of real data.

```bash
python -m venv .venv
source .venv/bin/activate              # .venv\Scripts\activate on Windows

pip install -r requirements.txt        # add requirements-dev.txt to also run tests
```

Requires Python 3.10 or 3.11 (pinned versions are chosen for PyTorch wheel
availability on that range; check for a matching `torch` wheel before using
a newer interpreter).

## CLI usage

Everything runs through `main.py`, so no notebook is required. Every
subcommand takes `--track {lof,lstm}`.

```bash
# LOF track — unsupervised, no config file needed
python main.py preprocess --track lof                 # inspect + materialize features
python main.py train --track lof                      # train + save model/scaler/results
python main.py evaluate --track lof                    # re-score with the saved model
python main.py report --track lof                      # plots + summary CSVs

# LSTM autoencoder track — driven by src/config/lstm_autoencoder_config.yaml
python main.py preprocess --track lstm
python main.py train --track lstm
python main.py evaluate --track lstm
python main.py report --track lstm                      # evaluation dashboard PNG + metrics.json
```

Key optional flags:
- LOF: `--data-path`, `--output-dir`, `--rolling-window`, `--n-neighbors`, `--contamination`, `--train-frac`, `--val-frac`
- LSTM: `--config` (path to an alternate YAML config), `--threshold-quantile`, `--exclude-features` (comma-separated, defaults to `particle_count_delta`)

Outputs are written under `data/processed/lof/` and
`data/processed/lstm_autoencoder/` respectively (models, scalers, results
CSVs, and a `report/` subfolder with plots and metrics).

### Known limitations

- **LOF**: the anomaly threshold is fit as a score quantile on the training
  split only, then applied across the full (non-stationary, 5-year) series.
  On the bundled dataset this produces a much higher overall anomaly rate
  than the `--contamination` value alone would suggest — tune
  `--contamination`/`--n-neighbors` or retrain periodically for production use.
- **LSTM**: the config's default `threshold_quantile` (`0.999`) is very
  sensitive to which features are included; the original notebook run used
  `0.9999` and excluded `particle_count_delta` from thresholding (the CLI's
  `evaluate`/`report` default matches that exclusion — override with
  `--exclude-features ""` to include everything).

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/lof tests/lstm
```

- `tests/lof/`: preprocessing, feature engineering, model, training.
- `tests/lstm/`: autoencoder forward pass, threshold tuning, and evaluation
  metrics — tested in isolation with fixed/mocked inputs, not a freshly
  trained model (training the LSTM is too slow/flaky for CI).

## Docker

```bash
git lfs pull                      # make sure real data files are on disk first
docker compose build
docker compose run --rm app train --track lof
docker compose run --rm app train --track lstm
docker compose run --rm app report --track lof
```

The image only bakes in `src/`, `main.py`, and dependencies; `data/` is
mounted as a volume from the host so it always reflects whatever you've
already `git lfs pull`-ed, without duplicating multi-GB LFS content into the
image. See [Dockerfile](Dockerfile) / [docker-compose.yml](docker-compose.yml).

## Contributions

- **LOF anomaly-detection track** (`src/**/lof_*.py`,
  `src/visualization/lof_plots.py`, `notebooks/LOF Anomaly Detection.ipynb`,
  `tests/lof/`): written end-to-end — preprocessing, features, model,
  training, evaluation, and documentation.
- **LSTM autoencoder track** (`src/**/lstm_autoencoder_*.py`,
  `src/evaluation/`, `src/config/`, `notebooks/LSTM Autoencoder.ipynb`):
  written by my collaborator.
- `main.py`, `README.md`, `requirements*.txt`, `Dockerfile`,
  `docker-compose.yml`, `tests/lof/`, and `tests/lstm/` (this pass) cover
  both tracks at the wiring/infra level but do not modify either track's
  modeling code.
