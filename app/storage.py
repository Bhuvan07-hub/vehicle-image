"""
Local-disk file storage. Swappable behind this same function signature for
S3/GCS later (see README) — nothing above this module knows the file lives
on local disk.
"""
import hashlib
from pathlib import Path

from app.config import STORAGE_DIR

CHUNK_SIZE = 1024 * 1024  # 1MB


def build_storage_path(image_id: str, extension: str) -> Path:
    return STORAGE_DIR / f"{image_id}{extension}"


def save_upload_bytes(data: bytes, image_id: str, extension: str) -> Path:
    path = build_storage_path(image_id, extension)
    path.write_bytes(data)
    return path


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extension_for_content_type(content_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(content_type, ".bin")
