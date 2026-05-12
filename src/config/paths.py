"""Path configuration for the project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class PathConfig:
    """Centralized path management with environment variable support."""

    PROJECT_ROOT = Path(
        os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    CONFIG_DIR = Path(os.getenv("CONFIG_DIR", PROJECT_ROOT / "configs"))

    RAW_DATA = DATA_DIR / "raw"
    INTERIM_DATA = DATA_DIR / "interim"
    PROCESSED_DATA = DATA_DIR / "processed"
    EXTERNAL_DATA = DATA_DIR / "external"

    MODELS_DIR = OUTPUT_DIR / "models"
    LSTM_MODEL_DIR = MODELS_DIR / "lstm"
    LOF_MODEL_DIR = MODELS_DIR / "lof"

    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    @staticmethod
    def ensure_dirs_exist(paths: Iterable[Path] | None = None) -> None:
        """Create all required directories if missing.

        Args:
            paths: Optional iterable of paths to create. When omitted, a
                standard set of project paths is created.

        Returns:
            None
        """
        if paths is None:
            paths = [
                PathConfig.RAW_DATA,
                PathConfig.INTERIM_DATA,
                PathConfig.PROCESSED_DATA,
                PathConfig.MODELS_DIR,
                PathConfig.PREDICTIONS_DIR,
                PathConfig.REPORTS_DIR,
            ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
