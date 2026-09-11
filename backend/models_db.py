"""
ORM models. Named `models_db` (not `models`) to avoid clashing with the
existing `backend/models/` package that holds the ML architectures.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    detections = relationship("Detection", back_populates="owner", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # "image" | "video"
    is_deepfake = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    model_type = Column(String, default="ensemble")
    model_predictions = Column(Text, nullable=True)  # JSON-encoded dict
    processing_time_ms = Column(Integer, default=0)
    demo_mode = Column(Boolean, default=True)  # True if untrained/demo weights were used
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    owner = relationship("User", back_populates="detections")
