"""
Number-plate extraction + Indian format validation.

Full pipeline:
  1. Locate plate candidates: edge-detect the whole frame, find contours,
     keep ones shaped like a plate (aspect ratio ~2:1-6:1, plausible area).
     This is a coarse OpenCV heuristic, not a trained detector — good enough
     to narrow OCR's search space, which matters a lot for OCR accuracy.
  2. If no plausible candidate region is found, fall back to running OCR on
     the full frame (lower accuracy, but still attempts extraction rather
     than giving up).
  3. Run Tesseract OCR on each candidate (best candidates first), clean the
     text (uppercase, strip non-alphanumerics), and check it against the
     standard Indian registration format: 2 letters (state), 1-2 digits
     (RTO code), 1-3 letters (series), 4 digits (unique number) —
     e.g. KA01AB1234.

Known limitation, stated explicitly rather than hidden: plate OCR on
in-the-wild vehicle photos (angle, distance, dirt, non-standard fonts,
regional/BH-series plates) is genuinely hard, and Tesseract is a general-
purpose OCR engine, not a plate-specialized model. `valid_format: False`
here means "we could not confirm a valid plate", not "this vehicle has an
invalid plate" — this check is reported as a low/medium confidence signal
for human review, not a ground-truth verdict. A production system would
swap this module for a dedicated ANPR model/API behind the same interface.
"""
import re

import cv2
import numpy as np

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TESSERACT_AVAILABLE = False

CHECK_NAME = "plate_format_validation"

INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def _find_plate_candidates(gray: np.ndarray, max_candidates: int = 5) -> list[np.ndarray]:
    h, w = gray.shape
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if ch == 0:
            continue
        aspect_ratio = cw / ch
        area_frac = (cw * ch) / (w * h)
        if 2.0 <= aspect_ratio <= 6.0 and 0.005 <= area_frac <= 0.25:
            candidates.append((cw * ch, x, y, cw, ch))

    candidates.sort(key=lambda c: c[0], reverse=True)
    crops = []
    for _, x, y, cw, ch in candidates[:max_candidates]:
        pad_x, pad_y = int(cw * 0.05), int(ch * 0.15)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + cw + pad_x), min(h, y + ch + pad_y)
        crops.append(gray[y0:y1, x0:x1])
    return crops


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    scale = 3 if max(crop.shape) < 300 else 1
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _ocr_text(image: np.ndarray) -> str:
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(
            image, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
    except Exception:
        return ""


def run(gray: np.ndarray) -> dict:
    if not _TESSERACT_AVAILABLE:
        return {
            "check": CHECK_NAME,
            "issue_detected": True,
            "issue_code": "ocr_unavailable",
            "confidence": 0.0,
            "metrics": {"note": "pytesseract/tesseract binary not available in this environment"},
        }

    candidates = _find_plate_candidates(gray)
    used_fallback_full_frame = False
    if not candidates:
        candidates = [gray]
        used_fallback_full_frame = True

    best = {"raw_text": "", "cleaned_text": "", "valid_format": False}
    for crop in candidates:
        processed = _preprocess_for_ocr(crop)
        raw_text = _ocr_text(processed)
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        valid = bool(INDIAN_PLATE_REGEX.match(cleaned))
        if valid:
            best = {"raw_text": raw_text.strip(), "cleaned_text": cleaned, "valid_format": True}
            break
        if len(cleaned) > len(best["cleaned_text"]):
            best = {"raw_text": raw_text.strip(), "cleaned_text": cleaned, "valid_format": False}

    issue_detected = not best["valid_format"]
    # Low confidence when we couldn't localize a plate region at all — the
    # OCR ran "blind" on the full frame, so a failed match is weaker evidence.
    confidence = 0.4 if used_fallback_full_frame else 0.75

    return {
        "check": CHECK_NAME,
        "issue_detected": issue_detected,
        "issue_code": "invalid_or_unreadable_plate" if issue_detected else None,
        "confidence": confidence if issue_detected else 0.9,
        "metrics": {
            "raw_ocr_text": best["raw_text"],
            "cleaned_text": best["cleaned_text"],
            "valid_format": best["valid_format"],
            "plate_region_localized": not used_fallback_full_frame,
            "candidates_tried": len(candidates),
        },
    }
