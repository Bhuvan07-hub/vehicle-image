"""
Brightness analysis: flags low-light and overexposed images using mean
grayscale intensity, plus the fraction of near-black pixels (a better signal
for "shot at night" than the mean alone, since a mean can be dragged up by a
single bright headlight/streetlamp in an otherwise dark frame).
"""
import cv2
import numpy as np

from app.config import LOW_LIGHT_MEAN_THRESHOLD, OVEREXPOSED_MEAN_THRESHOLD

CHECK_NAME = "brightness_analysis"


def run(
    image_bgr: np.ndarray,
    low_threshold: float = LOW_LIGHT_MEAN_THRESHOLD,
    high_threshold: float = OVEREXPOSED_MEAN_THRESHOLD,
) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    dark_pixel_ratio = float(np.mean(gray < 40))

    issue_code = None
    confidence = 0.0
    if mean_brightness < low_threshold or dark_pixel_ratio > 0.6:
        issue_code = "low_light"
        confidence = round(min(1.0, max(
            (low_threshold - mean_brightness) / low_threshold if low_threshold else 0,
            dark_pixel_ratio,
        )), 3)
    elif mean_brightness > high_threshold:
        issue_code = "overexposed"
        confidence = round(min(1.0, (mean_brightness - high_threshold) / (255 - high_threshold)), 3)

    return {
        "check": CHECK_NAME,
        "issue_detected": issue_code is not None,
        "issue_code": issue_code,
        "confidence": confidence,
        "metrics": {
            "mean_brightness": round(mean_brightness, 2),
            "dark_pixel_ratio": round(dark_pixel_ratio, 3),
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
    }
