"""
Repository tests. Uses a temp SQLite file per test session via monkeypatching
app.config.DB_PATH before importing app.db, so tests never touch the
project's real pipeline.db.
"""
import importlib
import os
import uuid

import pytest


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))

    import app.config as config
    importlib.reload(config)
    import app.db as db
    importlib.reload(db)
    import app.repository as repository
    importlib.reload(repository)

    db.init_db()
    return repository


def _make_image_id() -> str:
    return str(uuid.uuid4())


class TestImageLifecycle:
    def test_create_and_get(self, repo):
        image_id = _make_image_id()
        repo.create_image(
            image_id=image_id, original_filename="a.jpg", stored_filename="a.jpg",
            file_path="/tmp/a.jpg", content_type="image/jpeg", file_size_bytes=100,
            width=800, height=600, sha256_hash="sha", perceptual_hash="hash",
        )
        record = repo.get_image(image_id)
        assert record["status"] == "pending"
        assert record["retry_count"] == 0

    def test_get_missing_returns_none(self, repo):
        assert repo.get_image("does-not-exist") is None

    def test_update_status_transitions(self, repo):
        image_id = _make_image_id()
        repo.create_image(
            image_id=image_id, original_filename="a.jpg", stored_filename="a.jpg",
            file_path="/tmp/a.jpg", content_type="image/jpeg", file_size_bytes=100,
            width=800, height=600, sha256_hash="sha", perceptual_hash="hash",
        )
        repo.update_status(image_id, "processing")
        assert repo.get_image(image_id)["status"] == "processing"
        repo.update_status(image_id, "failed", error_message="boom")
        record = repo.get_image(image_id)
        assert record["status"] == "failed"
        assert record["error_message"] == "boom"

    def test_invalid_status_rejected(self, repo):
        with pytest.raises(ValueError):
            repo.update_status("whatever", "not-a-real-status")

    def test_retry_count_increments(self, repo):
        image_id = _make_image_id()
        repo.create_image(
            image_id=image_id, original_filename="a.jpg", stored_filename="a.jpg",
            file_path="/tmp/a.jpg", content_type="image/jpeg", file_size_bytes=100,
            width=800, height=600, sha256_hash="sha", perceptual_hash="hash",
        )
        assert repo.increment_retry_count(image_id) == 1
        assert repo.increment_retry_count(image_id) == 2

    def test_incomplete_jobs_excludes_completed(self, repo):
        pending_id, done_id = _make_image_id(), _make_image_id()
        for image_id in (pending_id, done_id):
            repo.create_image(
                image_id=image_id, original_filename="a.jpg", stored_filename="a.jpg",
                file_path="/tmp/a.jpg", content_type="image/jpeg", file_size_bytes=100,
                width=800, height=600, sha256_hash="sha", perceptual_hash="hash",
            )
        repo.update_status(done_id, "completed")
        incomplete = repo.get_incomplete_image_ids()
        assert pending_id in incomplete
        assert done_id not in incomplete


class TestAnalysisResults:
    def test_save_and_get_result(self, repo):
        image_id = _make_image_id()
        repo.create_image(
            image_id=image_id, original_filename="a.jpg", stored_filename="a.jpg",
            file_path="/tmp/a.jpg", content_type="image/jpeg", file_size_bytes=100,
            width=800, height=600, sha256_hash="sha", perceptual_hash="hash",
        )
        repo.save_analysis_result(
            image_id=image_id, issues_detected=["blurry_image"],
            checks=[{"check": "blur_detection", "issue_detected": True}],
            overall_risk_score=0.9,
        )
        result = repo.get_analysis_result(image_id)
        assert result["issues_detected"] == ["blurry_image"]
        assert result["overall_risk_score"] == 0.9

    def test_save_is_idempotent_upsert(self, repo):
        image_id = _make_image_id()
        repo.create_image(
            image_id=image_id, original_filename="a.jpg", stored_filename="a.jpg",
            file_path="/tmp/a.jpg", content_type="image/jpeg", file_size_bytes=100,
            width=800, height=600, sha256_hash="sha", perceptual_hash="hash",
        )
        repo.save_analysis_result(image_id=image_id, issues_detected=["a"], checks=[], overall_risk_score=0.1)
        repo.save_analysis_result(image_id=image_id, issues_detected=["b"], checks=[], overall_risk_score=0.2)
        result = repo.get_analysis_result(image_id)
        assert result["issues_detected"] == ["b"]
        assert result["overall_risk_score"] == 0.2
