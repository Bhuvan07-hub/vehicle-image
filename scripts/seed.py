"""
Demo/seed script: generates a handful of synthetic images (each engineered
to trip a specific check) and uploads them to a running instance of the API,
then polls and prints each result. Useful as a smoke test after `docker
compose up` / `uvicorn app.main:app`.

Usage:
    python scripts/seed.py [--base-url http://localhost:8000]
"""
import argparse
import io
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


def _to_jpeg_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def build_sample_images() -> dict[str, bytes]:
    samples: dict[str, bytes] = {}

    sharp = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)
    samples["normal_sharp.jpg"] = _to_jpeg_bytes(sharp)

    blurry = cv2.GaussianBlur(sharp, (31, 31), 15)
    samples["blurry.jpg"] = _to_jpeg_bytes(blurry)

    dark = np.random.randint(0, 20, (600, 800, 3), dtype=np.uint8)
    samples["low_light.jpg"] = _to_jpeg_bytes(dark)

    bright = np.random.randint(230, 255, (600, 800, 3), dtype=np.uint8)
    samples["overexposed.jpg"] = _to_jpeg_bytes(bright)

    small = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    samples["too_small.jpg"] = _to_jpeg_bytes(small)

    screenshot = np.random.randint(100, 180, (867, 400, 3), dtype=np.uint8)
    screenshot[:35, :] = 20
    screenshot[-20:, :] = 20
    samples["screenshot_like.png"] = _to_png_bytes(screenshot)

    # Exact duplicate of the first "normal" image
    samples["duplicate_of_normal.jpg"] = samples["normal_sharp.jpg"]

    frame = np.full((600, 900, 3), 90, dtype=np.uint8)
    cv2.rectangle(frame, (350, 400), (600, 470), (255, 255, 255), -1)
    cv2.rectangle(frame, (350, 400), (600, 470), (0, 0, 0), 3)
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    draw.text((365, 412), "KA01AB1234", fill=(0, 0, 0), font=font)
    samples["with_plate.jpg"] = _to_jpeg_bytes(np.array(pil))

    return samples


def upload(base_url: str, filename: str, data: bytes) -> str:
    content_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    resp = requests.post(
        f"{base_url}/api/v1/images",
        files={"file": (filename, data, content_type)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def wait_and_print(base_url: str, filename: str, image_id: str, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    status = "pending"
    while time.time() < deadline:
        status_resp = requests.get(f"{base_url}/api/v1/images/{image_id}", timeout=10)
        status_resp.raise_for_status()
        status = status_resp.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)

    print(f"\n--- {filename} ({image_id}) -> {status} ---")
    if status == "completed":
        result = requests.get(f"{base_url}/api/v1/images/{image_id}/result", timeout=10).json()
        print("issues_detected:", result["issues_detected"])
        print("overall_risk_score:", result["overall_risk_score"])
    elif status == "failed":
        detail = requests.get(f"{base_url}/api/v1/images/{image_id}", timeout=10).json()
        print("error_message:", detail.get("error_message"))
    else:
        print(f"still '{status}' after {timeout}s timeout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        requests.get(f"{args.base_url}/health", timeout=5).raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach API at {args.base_url}. Is it running? ({exc})"
        )

    samples = build_sample_images()
    print(f"Uploading {len(samples)} synthetic sample images to {args.base_url} ...")
    uploads = [(name, upload(args.base_url, name, data)) for name, data in samples.items()]

    for filename, image_id in uploads:
        wait_and_print(args.base_url, filename, image_id)


if __name__ == "__main__":
    main()
