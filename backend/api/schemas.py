"""
API Request and Response Schemas
Pydantic models for FastAPI validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    timestamp: str


class FaceLocation(BaseModel):
    """Face bounding box location"""
    x: int
    y: int
    width: int
    height: int


class FrameAnalysis(BaseModel):
    """Video frame analysis results"""
    total_frames: int
    analyzed_frames: int
    deepfake_frames: int
    average_confidence: float
    confidence_variance: float


class DetectionResult(BaseModel):
    """Detection result for single file"""
    is_deepfake: bool = Field(..., description="Whether the content is detected as deepfake")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model_predictions: Optional[Dict[str, float]] = Field(None, description="Individual model predictions")
    face_detected: Optional[bool] = Field(None, description="Whether a face was detected")
    face_location: Optional[FaceLocation] = Field(None, description="Face bounding box")
    visualization: Optional[str] = Field(None, description="Grad-CAM visualization as base64")
    frame_analysis: Optional[FrameAnalysis] = Field(None, description="Frame-by-frame analysis for videos")
    temporal_score: Optional[float] = Field(None, description="Temporal consistency score")
    audio_score: Optional[float] = Field(None, description="Audio deepfake score")
    timestamp: str = Field(..., description="Detection timestamp")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


class BatchResult(BaseModel):
    """Batch processing result"""
    total_files: int
    processed: int
    failed: int
    results: List[Dict]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    architecture: str
    parameters: Optional[int] = None
    accuracy: Optional[float] = None


class EnsembleInfo(BaseModel):
    """Ensemble configuration"""
    models: List[str]
    weights: Dict[str, float]


class StatisticsResponse(BaseModel):
    """Detection statistics"""
    total_detections: int
    deepfakes_detected: int
    average_confidence: float
    uptime: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str
