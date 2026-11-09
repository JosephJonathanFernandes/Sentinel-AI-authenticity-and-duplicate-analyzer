"""Application configuration for Sentinel.

All paths are resolved relative to this file so the project is portable.
CPU-first, memory-conscious defaults suitable for an i5 / 8-16 GB RAM laptop.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Persistent runtime state
DATABASE_PATH = PROJECT_ROOT / "sentinel.db"
QUARANTINE_DIR = PROJECT_ROOT / "quarantine"
SCAN_STATE_DIR = PROJECT_ROOT / ".scan_state"

# Ensure directories exist
for _d in (
    MODELS_DIR,
    DATASETS_DIR,
    SAMPLE_IMAGES_DIR,
    REPORTS_DIR,
    QUARANTINE_DIR,
    SCAN_STATE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Runtime settings. Override via environment variables or a .env file."""

    # ---- Scanning / batching ----
    batch_size: int = 50
    image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )

    # ---- Hashing ----
    hash_size: int = 8          # pHash / dHash / aHash dimension (hash_size x hash_size)
    highhash_size: int = 16     # Larger pHash for finer granularity
    hamming_threshold_exact: int = 3      # near-duplicate threshold (pHash)
    hamming_threshold_loose: int = 8      # looser near-duplicate threshold

    # ---- ORB feature matching ----
    orb_max_features: int = 500
    orb_match_ratio: float = 0.75         # Lowe's ratio test
    orb_min_match_count: int = 10         # minimum good matches to declare near-duplicate

    # ---- Similarity engine ----
    similarity_combined_threshold: float = 0.65

    # ---- AI detection (baseline) ----
    ai_image_size: int = 128              # lightweight classifier input
    ai_batch_size: int = 32
    ai_model_path: str = str(MODELS_DIR / "ai_classifier.pt")

    # ---- Forensics ----
    fft_size: int = 256
    noise_sigma: float = 1.0

    # ---- Management ----
    default_quarantine: bool = True       # quarantine instead of delete by default

    model_config = {"env_prefix": "SENTINEL_"}


settings = Settings()


# Logging configuration dictionary (used by logging.config.dictConfig)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}
