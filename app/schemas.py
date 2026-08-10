from typing import Any, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    id: str
    status: str
    created_at: str


class ImageStatusResponse(BaseModel):
    id: str
    original_filename: str
    content_type: Optional[str]
    file_size_bytes: Optional[int]
    width: Optional[int]
    height: Optional[int]
    status: str
    retry_count: int
    error_message: Optional[str]
    created_at: str
    updated_at: str


class CheckResult(BaseModel):
    check: str
    issue_detected: Optional[bool]
    issue_code: Optional[str]
    confidence: float
    metrics: dict[str, Any]


class AnalysisResultResponse(BaseModel):
    image_id: str
    status: str
    issues_detected: list[str]
    overall_risk_score: float
    checks: list[CheckResult]
    created_at: str


class ErrorResponse(BaseModel):
    detail: str


class ImageListItem(BaseModel):
    id: str
    original_filename: str
    status: str
    created_at: str


class ImageListResponse(BaseModel):
    items: list[ImageListItem]
    limit: int
    offset: int
