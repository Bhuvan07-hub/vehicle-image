"""
Suspicious-editing heuristic, two independent signals combined:

1. EXIF "Software" tag. If present and matches a known editor (Photoshop,
   GIMP, Snapseed, Lightroom, etc.) that's a strong, cheap signal — most
   camera/phone JPEGs either omit this tag or set it to the OS camera app.
   This is the more reliable of the two signals.

2. Error Level Analysis (ELA): re-encode the image as JPEG at a fixed
   quality and diff it against the (decoded) original. Regions that were
   pasted in or heavily re-touched tend to compress differently than the
   rest of the photo, showing up as localized high-error patches rather
   than a uniform low-level pattern across the whole frame.

   Important, explicitly documented limitation: ELA is a heuristic that
   works best on *originally uncompressed* images. Since almost everything
   arriving here will already be a JPEG straight off a phone camera, ELA's
   signal is weak and prone to false positives from ordinary JPEG
   recompression noise. We therefore weight it lower than the EXIF signal
   and always surface the raw metric so a human reviewer can judge it
   rather than trusting a binary verdict.
"""
import io

import numpy as np
from PIL import Image, ExifTags

CHECK_NAME = "editing_heuristic"

_EDITOR_KEYWORDS = ["photoshop", "gimp", "snapseed", "lightroom", "picsart", "facetune", "canva"]


def _exif_software_tag(pil_image: Image.Image) -> str:
    try:
        exif = pil_image.getexif()
        if not exif:
            return ""
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        return str(tags.get("Software", "")).lower()
    except Exception:
        return ""


def _error_level_analysis(pil_image: Image.Image, quality: int = 90) -> dict:
    rgb = pil_image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    original_arr = np.asarray(rgb, dtype=np.int16)
    resaved_arr = np.asarray(resaved, dtype=np.int16)
    diff = np.abs(original_arr - resaved_arr)

    mean_diff = float(np.mean(diff))
    # Localized-hotspot signal: fraction of pixels far above the mean error,
    # which is more indicative of a pasted/edited region than overall mean error.
    pixel_diff = diff.mean(axis=2)
    hotspot_ratio = float(np.mean(pixel_diff > (mean_diff + 3 * np.std(pixel_diff) + 1e-6)))

    return {"mean_error": round(mean_diff, 3), "hotspot_ratio": round(hotspot_ratio, 4)}


def run(pil_image: Image.Image, quality: int = 90) -> dict:
    software_tag = _exif_software_tag(pil_image)
    editor_detected = any(keyword in software_tag for keyword in _EDITOR_KEYWORDS)

    ela = _error_level_analysis(pil_image, quality=quality)
    ela_signal = ela["hotspot_ratio"] > 0.015

    score = (0.7 if editor_detected else 0.0) + (0.3 if ela_signal else 0.0)
    is_suspicious = score >= 0.3  # either signal alone is enough to flag for review

    return {
        "check": CHECK_NAME,
        "issue_detected": is_suspicious,
        "issue_code": "possible_editing" if is_suspicious else None,
        "confidence": round(min(1.0, score), 3),
        "metrics": {
            "exif_software_tag": software_tag or None,
            "editor_signature_matched": editor_detected,
            "ela": ela,
            "note": "ELA is unreliable on already-compressed JPEGs; treat as a weak, human-reviewable signal.",
        },
    }
