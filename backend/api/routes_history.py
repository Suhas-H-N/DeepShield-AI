"""Detection history and statistics endpoints (real DB-backed, replacing the old TODO stub)."""

import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

import models_db
from auth import get_current_user
from database import get_db
from api.schemas import DetectionRecordOut, PaginatedDetections, UserStatistics

router = APIRouter(prefix="/api/v1", tags=["history"])


@router.get("/history", response_model=PaginatedDetections)
def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    media_type: Optional[str] = Query(None, description="Filter: image | video"),
    result: Optional[str] = Query(None, description="Filter: deepfake | authentic"),
    search: Optional[str] = Query(None, description="Search by filename"),
    sort: str = Query("newest", description="newest | oldest | confidence_desc | confidence_asc"),
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user),
):
    query = db.query(models_db.Detection).filter(models_db.Detection.owner_id == current_user.id)

    if media_type:
        query = query.filter(models_db.Detection.media_type == media_type)
    if result == "deepfake":
        query = query.filter(models_db.Detection.is_deepfake.is_(True))
    elif result == "authentic":
        query = query.filter(models_db.Detection.is_deepfake.is_(False))
    if search:
        query = query.filter(models_db.Detection.filename.ilike(f"%{search}%"))

    sort_map = {
        "newest": models_db.Detection.created_at.desc(),
        "oldest": models_db.Detection.created_at.asc(),
        "confidence_desc": models_db.Detection.confidence.desc(),
        "confidence_asc": models_db.Detection.confidence.asc(),
    }
    query = query.order_by(sort_map.get(sort, models_db.Detection.created_at.desc()))

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedDetections(
        items=[DetectionRecordOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, math.ceil(total / page_size)),
    )


@router.delete("/history/{detection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_detection(
    detection_id: str,
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user),
):
    record = (
        db.query(models_db.Detection)
        .filter(models_db.Detection.id == detection_id, models_db.Detection.owner_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection record not found")
    db.delete(record)
    db.commit()


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user),
):
    db.query(models_db.Detection).filter(models_db.Detection.owner_id == current_user.id).delete()
    db.commit()


@router.get("/stats", response_model=UserStatistics)
def get_statistics(
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user),
):
    base = db.query(models_db.Detection).filter(models_db.Detection.owner_id == current_user.id)
    total = base.count()

    if total == 0:
        return UserStatistics(
            total_detections=0,
            deepfakes_detected=0,
            authentic_detected=0,
            average_confidence=0.0,
            average_processing_time_ms=0.0,
        )

    deepfakes = base.filter(models_db.Detection.is_deepfake.is_(True)).count()
    avg_confidence = base.with_entities(func.avg(models_db.Detection.confidence)).scalar() or 0.0
    avg_time = base.with_entities(func.avg(models_db.Detection.processing_time_ms)).scalar() or 0.0

    return UserStatistics(
        total_detections=total,
        deepfakes_detected=deepfakes,
        authentic_detected=total - deepfakes,
        average_confidence=round(float(avg_confidence), 4),
        average_processing_time_ms=round(float(avg_time), 1),
    )
