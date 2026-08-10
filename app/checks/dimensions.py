"""Minimum-resolution validation. Cheap, deterministic, catches thumbnails
and heavily downscaled re-uploads that won't be useful for downstream review."""
from app.config import MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH

CHECK_NAME = "dimension_validation"


def run(width: int, height: int, min_width: int = MIN_IMAGE_WIDTH, min_height: int = MIN_IMAGE_HEIGHT) -> dict:
    too_small = width < min_width or height < min_height
    return {
        "check": CHECK_NAME,
        "issue_detected": too_small,
        "issue_code": "resolution_too_low" if too_small else None,
        "confidence": 1.0 if too_small else 0.0,  # deterministic check, not probabilistic
        "metrics": {
            "width": width,
            "height": height,
            "min_width": min_width,
            "min_height": min_height,
        },
    }
