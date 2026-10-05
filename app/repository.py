"""
Data-access functions for the `images` and `analysis_results` tables.
Kept as plain functions (no repository class/interface) — there's only one
storage backend in scope, so an abstraction layer would be speculative.
"""
import json
from datetime import datetime, timezone
from typing import Any, Optional

from app.db import connection, write_connection

VALID_STATUSES = {"pending", "processing", "completed", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_image(
    *,
    image_id: str,
    original_filename: str,
    stored_filename: str,
    file_path: str,
    content_type: str,
    file_size_bytes: int,
    width: Optional[int],
    height: Optional[int],
    sha256_hash: str,
    perceptual_hash: Optional[str],
) -> None:
    now = _now()
    with write_connection() as conn:
        conn.execute(
            """
            INSERT INTO images (
                id, original_filename, stored_filename, file_path, content_type,
                file_size_bytes, width, height, sha256_hash, perceptual_hash,
                status, retry_count, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?)
            """,
            (
                image_id, original_filename, stored_filename, file_path, content_type,
                file_size_bytes, width, height, sha256_hash, perceptual_hash,
                now, now,
            ),
        )


def get_image(image_id: str) -> Optional[dict]:
    with connection() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(row) if row else None


def list_images(status: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[dict]:
    with connection() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM images WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM images ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]


def get_recent_hashes(exclude_image_id: str, limit: int = 5000) -> list[tuple[str, str, str]]:
    """Returns (id, sha256_hash, perceptual_hash) for duplicate comparison."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, sha256_hash, perceptual_hash FROM images
            WHERE id != ? AND perceptual_hash IS NOT NULL
            ORDER BY created_at DESC LIMIT ?
            """,
            (exclude_image_id, limit),
        ).fetchall()
        return [(r["id"], r["sha256_hash"], r["perceptual_hash"]) for r in rows]


def update_status(image_id: str, status: str, error_message: Optional[str] = None) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    with write_connection() as conn:
        conn.execute(
            "UPDATE images SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
            (status, error_message, _now(), image_id),
        )


def increment_retry_count(image_id: str) -> int:
    with write_connection() as conn:
        conn.execute(
            "UPDATE images SET retry_count = retry_count + 1, updated_at = ? WHERE id = ?",
            (_now(), image_id),
        )
        row = conn.execute("SELECT retry_count FROM images WHERE id = ?", (image_id,)).fetchone()
        return row["retry_count"] if row else 0


def save_analysis_result(
    *, image_id: str, issues_detected: list[str], checks: list[dict[str, Any]], overall_risk_score: float
) -> None:
    with write_connection() as conn:
        conn.execute(
            """
            INSERT INTO analysis_results (id, image_id, issues_detected, checks, overall_risk_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                issues_detected = excluded.issues_detected,
                checks = excluded.checks,
                overall_risk_score = excluded.overall_risk_score,
                created_at = excluded.created_at
            """,
            (
                f"result_{image_id}",
                image_id,
                json.dumps(issues_detected),
                json.dumps(checks),
                overall_risk_score,
                _now(),
            ),
        )


def get_analysis_result(image_id: str) -> Optional[dict]:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_results WHERE image_id = ?", (image_id,)
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["issues_detected"] = json.loads(result["issues_detected"])
        result["checks"] = json.loads(result["checks"])
        return result


def get_incomplete_image_ids() -> list[str]:
    """Used on startup to re-enqueue jobs that were mid-flight on last shutdown/crash."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT id FROM images WHERE status IN ('pending', 'processing') ORDER BY created_at ASC"
        ).fetchall()
        return [r["id"] for r in rows]

import json

from app.db import connection


def get_analytics_summary() -> dict:
    """Aggregate stats across every image ever processed: counts by status,
    how often each issue code has fired, and the average risk score among
    analyzed images. All computed from existing tables — no new storage
    needed, since the data is already there."""
    with connection() as conn:
        status_rows = conn.execute(
            "SELECT status, COUNT(*) as c FROM images GROUP BY status"
        ).fetchall()
        by_status = {r["status"]: r["c"] for r in status_rows}
        total_images = sum(by_status.values())

        result_rows = conn.execute(
            "SELECT issues_detected, overall_risk_score FROM analysis_results"
        ).fetchall()

        issue_counts: dict[str, int] = {}
        risk_scores: list[float] = []
        clean_count = 0
        for row in result_rows:
            issues = json.loads(row["issues_detected"])
            if not issues:
                clean_count += 1
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            risk_scores.append(row["overall_risk_score"])

        average_risk_score = round(sum(risk_scores) / len(risk_scores), 3) if risk_scores else 0.0
        # Most-common issues first — this is what you'd actually want to see
        # at a glance ("what's going wrong most often"), not an arbitrary order.
        issue_frequency = dict(sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True))

        return {
            "total_images": total_images,
            "by_status": by_status,
            "analyzed_count": len(result_rows),
            "clean_count": clean_count,
            "average_risk_score": average_risk_score,
            "issue_frequency": issue_frequency,
        }


def get_export_rows() -> list[dict]:
    """One row per image, joined with its analysis result if it has one
    (LEFT JOIN — a pending/processing/failed image still gets a row, just
    with empty result fields). Used by the CSV export endpoint."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT i.id, i.original_filename, i.status, i.created_at, i.error_message,
                   r.issues_detected, r.overall_risk_score
            FROM images i
            LEFT JOIN analysis_results r ON r.image_id = i.id
            ORDER BY i.created_at DESC
            """
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["issues_detected"] = json.loads(d["issues_detected"]) if d["issues_detected"] else []
            out.append(d)
        return out

