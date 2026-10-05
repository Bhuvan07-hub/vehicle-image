import csv
import io

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app import repository
from app.schemas import AnalyticsSummaryResponse

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
def get_summary():
    """Aggregate view across every image the system has ever processed:
    counts by status, how often each issue has fired, and the average risk
    score. Useful as the one endpoint a reviewer/manager would actually want
    to check rather than paging through individual results."""
    return repository.get_analytics_summary()


@router.get("/export")
def export_csv():
    """Streams a CSV of every image and its result (if any) as a file
    download. This is the 'report' a non-technical stakeholder could open
    in Excel/Sheets without touching the API directly."""
    rows = repository.get_export_rows()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "filename", "status", "created_at", "issues_detected", "overall_risk_score", "error_message"])
    for r in rows:
        writer.writerow([
            r["id"],
            r["original_filename"],
            r["status"],
            r["created_at"],
            ";".join(r["issues_detected"]),
            r["overall_risk_score"] if r["overall_risk_score"] is not None else "",
            r["error_message"] or "",
        ])
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vehicle_image_report.csv"},
    )
