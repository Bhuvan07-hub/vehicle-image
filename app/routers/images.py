import io
import logging
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app import repository
from app.checks.duplicate import compute_dhash
from app.config import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_SIZE_BYTES
from app.queue_worker import enqueue
from app.schemas import (
    AnalysisResultResponse,
    BatchUploadItem,
    BatchUploadResponse,
    ImageListResponse,
    ImageStatusResponse,
    UploadResponse,
)
from app.storage import extension_for_content_type, sha256_of_bytes, save_upload_bytes

logger = logging.getLogger("images_router")
router = APIRouter(prefix="/api/v1/images", tags=["images"])


async def _save_and_enqueue(file: UploadFile) -> dict:
    """Shared validation + persistence logic for a single file. Used by both
    the single-upload and batch-upload endpoints so they can never drift —
    one codepath decides what counts as a valid upload. Raises HTTPException
    on any validation failure; the caller decides whether that's a hard
    failure (single upload) or just one bad item in a batch."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds max size of {MAX_UPLOAD_SIZE_BYTES} bytes",
        )

    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(data))  # re-open after verify()
        width, height = pil_image.size
        perceptual_hash = compute_dhash(pil_image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="file is not a valid/decodable image")

    image_id = str(uuid.uuid4())
    extension = extension_for_content_type(file.content_type)
    file_path = save_upload_bytes(data, image_id, extension)
    sha256_hash = sha256_of_bytes(data)

    repository.create_image(
        image_id=image_id,
        original_filename=file.filename or f"{image_id}{extension}",
        stored_filename=file_path.name,
        file_path=str(file_path),
        content_type=file.content_type,
        file_size_bytes=len(data),
        width=width,
        height=height,
        sha256_hash=sha256_hash,
        perceptual_hash=perceptual_hash,
    )

    await enqueue(image_id)
    logger.info("enqueued image_id=%s for processing", image_id)

    record = repository.get_image(image_id)
    return {"id": image_id, "status": record["status"], "created_at": record["created_at"]}


@router.post("", response_model=UploadResponse, status_code=202)
async def upload_image(file: UploadFile = File(...)):
    result = await _save_and_enqueue(file)
    return UploadResponse(**result)


@router.post("/batch", response_model=BatchUploadResponse, status_code=202)
async def upload_batch(files: List[UploadFile] = File(...)):
    """Accepts multiple files in one request. Each file is validated and
    enqueued independently — one bad file in the batch (wrong format, too
    large, corrupt) doesn't block the rest. The response reports a per-file
    outcome so the caller knows exactly which uploads succeeded and which
    didn't, rather than an all-or-nothing failure."""
    items: list[BatchUploadItem] = []
    accepted = 0
    for file in files:
        try:
            result = await _save_and_enqueue(file)
            items.append(BatchUploadItem(filename=file.filename or "unknown", id=result["id"], status=result["status"]))
            accepted += 1
        except HTTPException as exc:
            items.append(BatchUploadItem(filename=file.filename or "unknown", error=str(exc.detail)))

    return BatchUploadResponse(accepted=accepted, rejected=len(files) - accepted, items=items)


@router.get("", response_model=ImageListResponse)
def list_images(
    status: str | None = Query(default=None, pattern="^(pending|processing|completed|failed)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    rows = repository.list_images(status=status, limit=limit, offset=offset)
    return ImageListResponse(items=rows, limit=limit, offset=offset)


@router.get("/{image_id}", response_model=ImageStatusResponse)
def get_image_status(image_id: str):
    record = repository.get_image(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no image found with id '{image_id}'")
    return ImageStatusResponse(**record)


@router.get("/{image_id}/result", response_model=AnalysisResultResponse)
def get_image_result(image_id: str):
    record = repository.get_image(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no image found with id '{image_id}'")

    if record["status"] == "failed":
        raise HTTPException(
            status_code=409,
            detail=f"processing failed: {record['error_message'] or 'unknown error'}",
        )
    if record["status"] in ("pending", "processing"):
        raise HTTPException(
            status_code=409,
            detail=f"analysis not ready yet, current status: '{record['status']}'",
        )

    result = repository.get_analysis_result(image_id)
    if result is None:
        raise HTTPException(status_code=500, detail="image marked completed but no result was found")

    return AnalysisResultResponse(
        image_id=image_id,
        status=record["status"],
        issues_detected=result["issues_detected"],
        overall_risk_score=result["overall_risk_score"],
        checks=result["checks"],
        created_at=result["created_at"],
    )
