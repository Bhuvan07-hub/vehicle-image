"""
Blur detection via variance of the Laplacian.

A sharp image has lots of high-frequency edge content, which shows up as
high variance after a Laplacian (2nd derivative) filter. A blurry image has
smoothed-out edges, so the variance collapses. This is a standard, cheap
heuristic (no ML model, no training data needed) — well-suited for a
"reasoning under uncertainty" style check.

Known limitation: the raw variance is scale- and content-dependent (a photo
of a flat wall is "low variance" without being blurry). We accept this
trade-off for this assignment; a production version would normalize by
image content/edge density rather than a single global threshold.
"""
import cv2
import numpy as np

from app.config import BLUR_VARIANCE_THRESHOLD

CHECK_NAME = "blur_detection"


def run(image_bgr: np.ndarray, threshold: float = BLUR_VARIANCE_THRESHOLD) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = variance < threshold

    # Confidence grows the further the variance is from the threshold, capped at 1.0.
    if is_blurry:
        confidence = min(1.0, (threshold - variance) / threshold) if threshold > 0 else 0.0
    else:
        confidence = min(1.0, (variance - threshold) / (threshold * 3)) if threshold > 0 else 0.0

    return {
        "check": CHECK_NAME,
        "issue_detected": is_blurry,
        "issue_code": "blurry_image" if is_blurry else None,
        "confidence": round(max(0.0, confidence), 3),
        "metrics": {"laplacian_variance": round(variance, 2), "threshold": threshold},
    }
