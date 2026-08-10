"""
Screenshot detection and photo-of-photo detection.

Neither of these has one clean signal, so both are implemented as a small
weighted-vote of independent heuristics. This is deliberately structured as
"evidence accumulation" rather than a single if/else, because in the real
problem each individual signal is weak and unreliable on its own — the
point of this check is to combine several weak signals into one calibrated-
ish confidence rather than pretend any single rule is authoritative.

Screenshot signals:
  - No camera EXIF (Make/Model) at all. Real camera/phone photos almost
    always carry this; most screenshot tools strip it or never write it.
  - Aspect ratio matches a common device screen ratio (16:9, 19.5:9, 3:4...).
  - A thin, near-uniform-color band at the very top and/or bottom of the
    frame (status bar / home indicator / letterboxing), detected via low
    pixel standard-deviation in those strips.
  - File saved as PNG (common for screenshots; camera photos are usually
    JPEG). Weighted lightly since this is weak on its own.

Photo-of-photo ("recapture") signals:
  - A moire-like high-frequency periodic pattern, approximated cheaply via
    the ratio of high-frequency to low-frequency energy in the FFT magnitude
    spectrum (re-photographing a screen/print introduces pixel-grid /
    printing-halftone interference patterns that raise this ratio).
  - A visible bezel/frame: strong straight edges concentrated near the image
    border, which happens when a screen or printed photo edge lands inside
    the outer photograph.

These are stated as heuristics, not certainties, in the README trade-offs.
"""
import cv2
import numpy as np
from PIL import Image, ExifTags

CHECK_NAME_SCREENSHOT = "screenshot_detection"
CHECK_NAME_RECAPTURE = "photo_of_photo_detection"

_COMMON_SCREEN_RATIOS = [9 / 16, 9 / 19.5, 3 / 4, 4 / 3, 16 / 9, 19.5 / 9, 1.0]
_RATIO_TOLERANCE = 0.03


def _has_camera_exif(pil_image: Image.Image) -> bool:
    try:
        exif = pil_image.getexif()
        if not exif:
            return False
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        return bool(tags.get("Make") or tags.get("Model"))
    except Exception:
        return False


def _matches_screen_ratio(width: int, height: int) -> bool:
    if height == 0:
        return False
    ratio = width / height
    return any(abs(ratio - r) < _RATIO_TOLERANCE for r in _COMMON_SCREEN_RATIOS)


def _uniform_border_bands(gray: np.ndarray, band_frac: float = 0.04, std_threshold: float = 12.0) -> bool:
    h, _ = gray.shape
    band_h = max(2, int(h * band_frac))
    top_band, bottom_band = gray[:band_h, :], gray[-band_h:, :]
    return bool(np.std(top_band) < std_threshold or np.std(bottom_band) < std_threshold)


def detect_screenshot(pil_image: Image.Image, gray: np.ndarray, file_extension: str) -> dict:
    width, height = pil_image.size
    signals = {
        "missing_camera_exif": not _has_camera_exif(pil_image),
        "screen_aspect_ratio": _matches_screen_ratio(width, height),
        "uniform_border_band": _uniform_border_bands(gray),
        "png_format": file_extension.lower() == ".png",
    }
    weights = {
        "missing_camera_exif": 0.35,
        "screen_aspect_ratio": 0.30,
        "uniform_border_band": 0.25,
        "png_format": 0.10,
    }
    score = sum(weights[k] for k, v in signals.items() if v)
    is_screenshot = score >= 0.55

    return {
        "check": CHECK_NAME_SCREENSHOT,
        "issue_detected": is_screenshot,
        "issue_code": "possible_screenshot" if is_screenshot else None,
        "confidence": round(min(1.0, score), 3),
        "metrics": {"signals": signals},
    }


def _high_frequency_energy_ratio(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(small.astype(np.float32))
    magnitude = np.abs(np.fft.fftshift(f))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    low_energy = float(magnitude[mask_low].sum())
    total_energy = float(magnitude.sum()) + 1e-6
    high_energy = total_energy - low_energy
    return high_energy / total_energy


def _border_edge_density(gray: np.ndarray, band_frac: float = 0.05) -> float:
    edges = cv2.Canny(gray, 80, 160)
    h, w = edges.shape
    bh, bw = max(2, int(h * band_frac)), max(2, int(w * band_frac))
    border_mask = np.zeros_like(edges, dtype=bool)
    border_mask[:bh, :] = border_mask[-bh:, :] = True
    border_mask[:, :bw] = border_mask[:, -bw:] = True
    border_pixels = int(max(1, border_mask.sum()))
    return float(edges[border_mask].sum() / 255) / border_pixels


def detect_photo_of_photo(gray: np.ndarray) -> dict:
    hf_ratio = _high_frequency_energy_ratio(gray)
    border_density = _border_edge_density(gray)

    hf_signal = bool(hf_ratio > 0.90)
    border_signal = bool(border_density > 0.12)

    score = float(0.6 * hf_signal + 0.4 * border_signal)
    is_recapture = bool(score >= 0.6)

    return {
        "check": CHECK_NAME_RECAPTURE,
        "issue_detected": is_recapture,
        "issue_code": "possible_photo_of_photo" if is_recapture else None,
        "confidence": round(min(1.0, score), 3),
        "metrics": {
            "high_frequency_energy_ratio": round(hf_ratio, 4),
            "border_edge_density": round(border_density, 4),
        },
    }
