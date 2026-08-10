"""
Unit tests for the analysis checks, using small synthetically-generated
images rather than fixture files, so the test suite has no external asset
dependencies and runs anywhere.
"""
import numpy as np
import cv2
import pytest
from PIL import Image

from app.checks import blur, brightness, dimensions, duplicate, plate_ocr, screenshot, tamper


def _bgr_from_gray_value(value: int, size=(600, 800)) -> np.ndarray:
    h, w = size
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return arr


def _sharp_noise_bgr(size=(600, 800)) -> np.ndarray:
    h, w = size
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


class TestBlur:
    def test_sharp_image_not_flagged(self):
        bgr = _sharp_noise_bgr()
        result = blur.run(bgr)
        assert result["issue_detected"] is False

    def test_blurred_image_flagged(self):
        bgr = _sharp_noise_bgr()
        blurred = cv2.GaussianBlur(bgr, (31, 31), 15)
        result = blur.run(blurred)
        assert result["issue_detected"] is True
        assert result["issue_code"] == "blurry_image"


class TestBrightness:
    def test_dark_image_flagged_low_light(self):
        bgr = _bgr_from_gray_value(5)
        result = brightness.run(bgr)
        assert result["issue_detected"] is True
        assert result["issue_code"] == "low_light"

    def test_bright_image_flagged_overexposed(self):
        bgr = _bgr_from_gray_value(250)
        result = brightness.run(bgr)
        assert result["issue_detected"] is True
        assert result["issue_code"] == "overexposed"

    def test_mid_brightness_not_flagged(self):
        bgr = _bgr_from_gray_value(120)
        result = brightness.run(bgr)
        assert result["issue_detected"] is False


class TestDimensions:
    def test_small_image_flagged(self):
        result = dimensions.run(100, 100)
        assert result["issue_detected"] is True

    def test_normal_image_not_flagged(self):
        result = dimensions.run(1920, 1080)
        assert result["issue_detected"] is False


class TestDuplicate:
    def test_identical_perceptual_hash_flagged(self):
        h = "1" * 64
        result = duplicate.run("shaA", h, [("other", "shaB", h)])
        assert result["issue_detected"] is True
        assert result["issue_code"] == "likely_duplicate"

    def test_exact_sha_match_takes_priority(self):
        result = duplicate.run("shaA", "1" * 64, [("other", "shaA", "0" * 64)])
        assert result["issue_code"] == "exact_duplicate"

    def test_unrelated_hash_not_flagged(self):
        result = duplicate.run("shaA", "1" * 64, [("other", "shaB", "0" * 64)])
        assert result["issue_detected"] is False

    def test_dhash_is_deterministic(self):
        img = Image.fromarray(_sharp_noise_bgr())
        assert duplicate.compute_dhash(img) == duplicate.compute_dhash(img)

    def test_hamming_distance_zero_for_identical(self):
        assert duplicate.hamming_distance("1010", "1010") == 0

    def test_hamming_distance_counts_differences(self):
        assert duplicate.hamming_distance("1010", "1111") == 2


class TestScreenshot:
    def test_screenshot_like_image_flagged(self):
        arr = np.random.randint(100, 180, (867, 400, 3), dtype=np.uint8)
        arr[:35, :] = 20
        arr[-20:, :] = 20
        pil = Image.fromarray(arr)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        result = screenshot.detect_screenshot(pil, gray, ".png")
        assert result["issue_detected"] is True

    def test_photo_of_photo_returns_serializable_types(self):
        gray = cv2.cvtColor(_sharp_noise_bgr(), cv2.COLOR_BGR2GRAY)
        result = screenshot.detect_photo_of_photo(gray)
        assert isinstance(result["issue_detected"], bool)
        assert isinstance(result["confidence"], float)


class TestTamper:
    def test_returns_expected_shape(self):
        pil = Image.fromarray(_sharp_noise_bgr())
        result = tamper.run(pil)
        assert result["check"] == "editing_heuristic"
        assert "ela" in result["metrics"]
        assert isinstance(result["issue_detected"], bool)


class TestPlateOcr:
    def test_random_noise_does_not_crash_and_flags_unreadable(self):
        gray = cv2.cvtColor(_sharp_noise_bgr(), cv2.COLOR_BGR2GRAY)
        result = plate_ocr.run(gray)
        assert result["check"] == "plate_format_validation"
        assert result["metrics"]["valid_format"] is False

    def test_regex_accepts_standard_format(self):
        assert plate_ocr.INDIAN_PLATE_REGEX.match("KA01AB1234")
        assert plate_ocr.INDIAN_PLATE_REGEX.match("DL3CAF1234")

    def test_regex_rejects_malformed(self):
        assert not plate_ocr.INDIAN_PLATE_REGEX.match("1234ABCD")
        assert not plate_ocr.INDIAN_PLATE_REGEX.match("KA01AB123")  # only 3 digits
        assert not plate_ocr.INDIAN_PLATE_REGEX.match("ka01ab1234")  # lowercase


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
