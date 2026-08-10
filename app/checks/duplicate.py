"""
Duplicate detection, two layers:

1. Exact duplicate: SHA-256 of the raw file bytes. Catches byte-identical
   re-uploads (same file uploaded twice) cheaply and with zero false positives.
2. Near-duplicate: a difference hash (dHash) computed over the decoded pixels,
   compared via Hamming distance against recently-seen images. Catches
   re-compressed/resized/lightly-cropped re-uploads of the same photo, which
   a byte hash would miss.

dHash is implemented directly (no `imagehash` dependency) — it's ~10 lines
and keeps the dependency footprint small. Algorithm: downscale to 9x8
grayscale, threshold each row on "is this pixel brighter than its left
neighbor", flatten to a 64-bit string.
"""
from PIL import Image
import numpy as np

from app.config import DUPLICATE_HAMMING_THRESHOLD

CHECK_NAME = "duplicate_detection"


def compute_dhash(pil_image: Image.Image, hash_size: int = 8) -> str:
    gray = pil_image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join("1" if v else "0" for v in diff.flatten())


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if len(hash_a) != len(hash_b):
        return max(len(hash_a), len(hash_b))
    return sum(a != b for a, b in zip(hash_a, hash_b))


def run(
    sha256_hash: str,
    perceptual_hash: str,
    existing: list[tuple[str, str, str]],  # (id, sha256_hash, perceptual_hash)
    threshold: int = DUPLICATE_HAMMING_THRESHOLD,
) -> dict:
    for other_id, other_sha, other_phash in existing:
        if other_sha and other_sha == sha256_hash:
            return {
                "check": CHECK_NAME,
                "issue_detected": True,
                "issue_code": "exact_duplicate",
                "confidence": 1.0,
                "metrics": {"matched_image_id": other_id, "match_type": "exact_sha256"},
            }

    best_match_id, best_distance = None, None
    for other_id, _, other_phash in existing:
        if not other_phash:
            continue
        distance = hamming_distance(perceptual_hash, other_phash)
        if best_distance is None or distance < best_distance:
            best_distance, best_match_id = distance, other_id

    if best_distance is not None and best_distance <= threshold:
        confidence = round(max(0.0, 1.0 - (best_distance / (threshold + 1))), 3)
        return {
            "check": CHECK_NAME,
            "issue_detected": True,
            "issue_code": "likely_duplicate",
            "confidence": confidence,
            "metrics": {
                "matched_image_id": best_match_id,
                "match_type": "perceptual_hash",
                "hamming_distance": best_distance,
                "threshold": threshold,
            },
        }

    return {
        "check": CHECK_NAME,
        "issue_detected": False,
        "issue_code": None,
        "confidence": 0.0,
        "metrics": {"closest_hamming_distance": best_distance, "threshold": threshold},
    }
