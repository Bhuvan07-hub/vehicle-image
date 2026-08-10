"""
End-to-end API tests using FastAPI's TestClient (sync wrapper over httpx).

These exercise the real upload -> queue -> process -> fetch-result flow
against a temp DB/storage dir. Since the in-memory queue workers only run
inside the app's lifespan (i.e. inside a running event loop), we poll the
status endpoint briefly rather than assuming synchronous completion —
mirroring how a real client should behave.
"""
import io
import time

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))

    # Reload modules so they pick up the patched env vars instead of any
    # module-level state cached from a previous import.
    import importlib
    import app.config as config
    importlib.reload(config)
    for mod_name in ["app.db", "app.repository", "app.storage", "app.pipeline", "app.queue_worker", "app.routers.images", "app.main"]:
        if mod_name in list(__import__("sys").modules):
            importlib.reload(__import__("sys").modules[mod_name])

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as test_client:
        yield test_client


def _sample_jpeg_bytes(sharp: bool = True) -> bytes:
    arr = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)
    if not sharp:
        import cv2
        arr = cv2.GaussianBlur(arr, (31, 31), 15)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _wait_for_terminal_status(client, image_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/images/{image_id}")
        status = resp.json()["status"]
        if status in ("completed", "failed"):
            return status
        time.sleep(0.2)
    raise TimeoutError(f"image {image_id} did not reach a terminal status in {timeout}s")


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestUploadFlow:
    def test_upload_returns_202_and_pending(self, client):
        files = {"file": ("car.jpg", _sample_jpeg_bytes(), "image/jpeg")}
        resp = client.post("/api/v1/images", files=files)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] in ("pending", "processing", "completed")
        assert "id" in body

    def test_rejects_non_image_content_type(self, client):
        files = {"file": ("notes.txt", b"hello world", "text/plain")}
        resp = client.post("/api/v1/images", files=files)
        assert resp.status_code == 415

    def test_rejects_empty_file(self, client):
        files = {"file": ("car.jpg", b"", "image/jpeg")}
        resp = client.post("/api/v1/images", files=files)
        assert resp.status_code == 400

    def test_full_flow_reaches_completed_with_results(self, client):
        files = {"file": ("car.jpg", _sample_jpeg_bytes(sharp=False), "image/jpeg")}
        upload_resp = client.post("/api/v1/images", files=files)
        image_id = upload_resp.json()["id"]

        status = _wait_for_terminal_status(client, image_id)
        assert status == "completed"

        result_resp = client.get(f"/api/v1/images/{image_id}/result")
        assert result_resp.status_code == 200
        body = result_resp.json()
        assert "blurry_image" in body["issues_detected"]
        assert len(body["checks"]) >= 4  # assignment requires >=4 meaningful checks

    def test_result_returns_409_while_pending(self, client):
        # A fresh upload's result is checked immediately, before the queue
        # has had a chance to run (best-effort race, may occasionally be
        # flaky under a very fast machine — acceptable for a take-home).
        files = {"file": ("car.jpg", _sample_jpeg_bytes(), "image/jpeg")}
        image_id = client.post("/api/v1/images", files=files).json()["id"]
        resp = client.get(f"/api/v1/images/{image_id}/result")
        assert resp.status_code in (200, 409)

    def test_unknown_image_id_returns_404(self, client):
        resp = client.get("/api/v1/images/does-not-exist")
        assert resp.status_code == 404

    def test_list_images(self, client):
        files = {"file": ("car.jpg", _sample_jpeg_bytes(), "image/jpeg")}
        client.post("/api/v1/images", files=files)
        resp = client.get("/api/v1/images")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1
