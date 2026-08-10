"""
Centralized configuration. Kept as plain env-driven constants rather than a
settings library (pydantic-settings, etc.) to avoid an extra dependency for
a project this size — see README trade-offs.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where uploaded images are persisted on disk.
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite file. SQLite is intentionally chosen for this take-home: zero setup,
# file-based, trivially inspectable, good enough for the concurrency levels
# a take-home / demo will see. See README for the Postgres migration path.
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "pipeline.db"))

# Upload constraints
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 15 * 1024 * 1024))  # 15MB

# Async processing
QUEUE_WORKER_COUNT = int(os.getenv("QUEUE_WORKER_COUNT", 2))
MAX_PROCESSING_RETRIES = int(os.getenv("MAX_PROCESSING_RETRIES", 2))

# Analysis thresholds (tunable; documented in README)
BLUR_VARIANCE_THRESHOLD = float(os.getenv("BLUR_VARIANCE_THRESHOLD", 100.0))
LOW_LIGHT_MEAN_THRESHOLD = float(os.getenv("LOW_LIGHT_MEAN_THRESHOLD", 50.0))
OVEREXPOSED_MEAN_THRESHOLD = float(os.getenv("OVEREXPOSED_MEAN_THRESHOLD", 200.0))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", 480))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", 360))
DUPLICATE_HAMMING_THRESHOLD = int(os.getenv("DUPLICATE_HAMMING_THRESHOLD", 5))
