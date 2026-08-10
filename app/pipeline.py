"""
Runs the full analysis pipeline for one image and persists the result.

Design notes:
  - Each check is independent and wrapped in its own try/except. A single
    check crashing (corrupt input, an OpenCV edge case, etc.) should not
    fail the entire job — we record it as a failed sub-check with an error
    note and keep going. The job as a whole only moves to `failed` status
    if we can't even decode the image, or if an unexpected exception escapes
    everything (defense in depth, logged with the image id for debugging).
  - This function is synchronous/blocking (OpenCV + Tesseract are CPU-bound,
    not I/O-bound), so the queue worker runs it in a thread pool
    (`asyncio.to_thread`) rather than awaiting it directly — see queue_worker.py.
"""
import logging

import cv2
import numpy as np
from PIL import Image

from app import repository
from app.checks import blur, brightness, dimensions, duplicate, plate_ocr, screenshot, tamper

logger = logging.getLogger("pipeline")


def _load_images(file_path: str) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Returns (bgr, gray, pil_image). Raises if the file can't be decoded."""
    pil_image = Image.open(file_path)
    pil_image.load()  # force decode now so corrupt files fail here, not later
    bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image at {file_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray, pil_image


def _run_check_safely(name: str, fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
        logger.exception("check '%s' raised an exception", name)
        return {
            "check": name,
            "issue_detected": None,
            "issue_code": "check_error",
            "confidence": 0.0,
            "metrics": {"error": str(exc)},
        }


def process_image(image_id: str) -> None:
    record = repository.get_image(image_id)
    if record is None:
        logger.error("process_image called for unknown image_id=%s", image_id)
        return

    repository.update_status(image_id, "processing")
    file_path = record["file_path"]

    try:
        bgr, gray, pil_image = _load_images(file_path)
    except Exception as exc:
        logger.exception("failed to decode image_id=%s", image_id)
        repository.update_status(image_id, "failed", error_message=f"could not decode image: {exc}")
        return

    try:
        existing_hashes = repository.get_recent_hashes(exclude_image_id=image_id)

        checks: list[dict] = [
            _run_check_safely("blur_detection", blur.run, bgr),
            _run_check_safely("brightness_analysis", brightness.run, bgr),
            _run_check_safely("dimension_validation", dimensions.run, record["width"], record["height"]),
            _run_check_safely(
                "duplicate_detection",
                duplicate.run,
                record["sha256_hash"],
                record["perceptual_hash"],
                existing_hashes,
            ),
            _run_check_safely(
                "screenshot_detection",
                screenshot.detect_screenshot,
                pil_image,
                gray,
                _extension_from_path(file_path),
            ),
            _run_check_safely("photo_of_photo_detection", screenshot.detect_photo_of_photo, gray),
            _run_check_safely("editing_heuristic", tamper.run, pil_image),
            _run_check_safely("plate_format_validation", plate_ocr.run, gray),
        ]

        issues_detected = [c["issue_code"] for c in checks if c.get("issue_detected") and c.get("issue_code")]
        overall_risk_score = _aggregate_risk_score(checks)

        repository.save_analysis_result(
            image_id=image_id,
            issues_detected=issues_detected,
            checks=checks,
            overall_risk_score=overall_risk_score,
        )
        repository.update_status(image_id, "completed")
    except Exception as exc:
        logger.exception("unexpected failure processing image_id=%s", image_id)
        repository.update_status(image_id, "failed", error_message=f"unexpected processing error: {exc}")


def _aggregate_risk_score(checks: list[dict]) -> float:
    """Simple aggregate: highest confidence among detected issues. Deliberately
    not a weighted sum across checks — a single high-confidence issue (e.g. an
    exact duplicate) should dominate the score rather than get diluted by five
    unrelated "no issue" checks contributing zero."""
    confidences = [c["confidence"] for c in checks if c.get("issue_detected") and isinstance(c.get("confidence"), (int, float))]
    return round(max(confidences), 3) if confidences else 0.0


def _extension_from_path(file_path: str) -> str:
    idx = file_path.rfind(".")
    return file_path[idx:] if idx != -1 else ""
