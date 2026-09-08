"""
Advanced Deepfake Detection System - Main API
FastAPI backend with multi-model ensemble detection.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db, init_db
import models_db
from auth import get_optional_user
from api.routes_auth import router as auth_router
from api.routes_history import router as history_router
from api.schemas import DetectionResult, BatchResult, HealthCheck
from models.ensemble import EnsembleDetector
from utils.preprocessing import preprocess_image
from utils.face_detector import FaceDetector
from utils.video_processor import VideoProcessor
from utils.audio_analyzer import AudioAnalyzer
from utils.explainability import generate_gradcam
from utils.file_validation import validate_upload

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Deepfake detection system with a multi-model ensemble backend",
    version=settings.app_version,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - restricted to configured origins instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)
app.include_router(history_router)

# Global model instances (loaded once at startup)
ensemble_detector: Optional[EnsembleDetector] = None
face_detector: Optional[FaceDetector] = None
video_processor: Optional[VideoProcessor] = None
audio_analyzer: Optional[AudioAnalyzer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize DB and models on startup."""
    global ensemble_detector, face_detector, video_processor, audio_analyzer

    init_db()
    os.makedirs(settings.upload_temp_dir, exist_ok=True)

    logger.info("Loading deepfake detection models...")
    try:
        ensemble_detector = EnsembleDetector(model_dir=settings.model_weights_dir)
        face_detector = FaceDetector()
        video_processor = VideoProcessor()
        audio_analyzer = AudioAnalyzer()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def _generic_error_detail(e: Exception) -> str:
    """Avoid leaking internal stack traces / implementation details to
    clients in production; show full detail only in debug mode."""
    if settings.debug:
        return f"Processing error: {str(e)}"
    return "An internal error occurred while processing your request."


def _save_detection_record(
    db: Session,
    user: Optional[models_db.User],
    filename: str,
    media_type: str,
    is_deepfake: bool,
    confidence: float,
    model_type: str,
    model_predictions: Optional[dict],
    processing_time_ms: int,
):
    """Persist a detection to history, but only for authenticated users."""
    if user is None:
        return
    record = models_db.Detection(
        owner_id=user.id,
        filename=filename[:255],
        media_type=media_type,
        is_deepfake=is_deepfake,
        confidence=confidence,
        model_type=model_type,
        model_predictions=json.dumps(model_predictions) if model_predictions else None,
        processing_time_ms=processing_time_ms,
        demo_mode=ensemble_detector.demo_mode if ensemble_detector else True,
    )
    db.add(record)
    db.commit()


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": settings.app_version,
        "models_loaded": ensemble_detector is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/detect/image", response_model=DetectionResult)
@limiter.limit(settings.rate_limit_detect)
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    return_visualization: bool = True,
    model_type: Optional[str] = "ensemble",
    db: Session = Depends(get_db),
    current_user: Optional[models_db.User] = Depends(get_optional_user),
):
    """
    Detect deepfake in a single image.

    Detections are saved to history automatically when the request is
    authenticated (send an `Authorization: Bearer <token>` header).
    """
    start_time = time.perf_counter()
    try:
        contents = await file.read()

        validation = validate_upload(contents, settings.max_image_size_mb, settings.max_video_size_mb)
        if not validation.ok:
            raise HTTPException(status_code=400, detail=validation.reason)
        if validation.media_type != "image":
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid, readable image")
        image_array = np.array(image)

        logger.info("Detecting faces...")
        faces = face_detector.detect_faces(image_array)
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")

        face = faces[0]
        face_crop = face_detector.extract_face(image_array, face)
        preprocessed = preprocess_image(face_crop)

        logger.info(f"Running {model_type} detection...")
        if model_type == "ensemble":
            prediction, confidence, model_predictions = ensemble_detector.predict(preprocessed)
        else:
            prediction, confidence = ensemble_detector.predict_single_model(preprocessed, model_type)
            model_predictions = {model_type: confidence}

        visualization = None
        if return_visualization:
            logger.info("Generating Grad-CAM visualization...")
            visualization = generate_gradcam(ensemble_detector.get_model(model_type), preprocessed)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = {
            "is_deepfake": bool(prediction),
            "confidence": float(confidence),
            "model_predictions": model_predictions,
            "face_detected": True,
            "face_location": {
                "x": int(face["box"][0]),
                "y": int(face["box"][1]),
                "width": int(face["box"][2]),
                "height": int(face["box"][3]),
            },
            "visualization": visualization,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": processing_time_ms,
            "demo_mode": ensemble_detector.demo_mode,
        }

        _save_detection_record(
            db, current_user, file.filename, "image", bool(prediction), float(confidence),
            model_type, model_predictions, processing_time_ms,
        )

        logger.info(f"Detection complete: {'DEEPFAKE' if prediction else 'REAL'} ({confidence:.2%})")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=_generic_error_detail(e))


@app.post("/api/v1/detect/video", response_model=DetectionResult)
@limiter.limit(settings.rate_limit_detect)
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    sample_rate: int = 5,
    analyze_audio: bool = True,
    temporal_analysis: bool = True,
    db: Session = Depends(get_db),
    current_user: Optional[models_db.User] = Depends(get_optional_user),
):
    """
    Detect deepfake in a video file.
    """
    start_time = time.perf_counter()
    video_path = None
    try:
        contents = await file.read()

        validation = validate_upload(contents, settings.max_image_size_mb, settings.max_video_size_mb)
        if not validation.ok:
            raise HTTPException(status_code=400, detail=validation.reason)
        if validation.media_type != "video":
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid video")

        # Use a random, sandboxed filename rather than the client-supplied
        # one to avoid path traversal / collision issues.
        safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename or 'upload')}"
        video_path = os.path.join(settings.upload_temp_dir, safe_name)

        with open(video_path, "wb") as f:
            f.write(contents)

        logger.info(f"Processing video: {file.filename}")

        frames = video_processor.extract_frames(video_path, sample_rate=sample_rate)
        logger.info(f"Extracted {len(frames)} frames")

        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="No frames could be extracted")

        frame_predictions = []
        frame_confidences = []

        for idx, frame in enumerate(frames):
            try:
                faces = face_detector.detect_faces(frame)
                if len(faces) == 0:
                    continue
                face_crop = face_detector.extract_face(frame, faces[0])
                preprocessed = preprocess_image(face_crop)
                prediction, confidence, _ = ensemble_detector.predict(preprocessed)
                frame_predictions.append(prediction)
                frame_confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Error processing frame {idx}: {e}")
                continue

        if len(frame_predictions) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in video frames")

        temporal_score = None
        if temporal_analysis and len(frame_predictions) > 10:
            temporal_score = ensemble_detector.analyze_temporal_consistency(frame_predictions, frame_confidences)

        audio_score = None
        if analyze_audio:
            try:
                audio_score = audio_analyzer.analyze(video_path)
                logger.info(f"Audio analysis score: {audio_score:.2%}")
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")

        avg_confidence = float(np.mean(frame_confidences))
        deepfake_ratio = float(np.mean(frame_predictions))
        final_prediction = deepfake_ratio > 0.5

        final_confidence = avg_confidence
        if temporal_score is not None:
            final_confidence = (final_confidence + temporal_score) / 2
        if audio_score is not None:
            final_confidence = (final_confidence + audio_score) / 2

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = {
            "is_deepfake": bool(final_prediction),
            "confidence": float(final_confidence),
            "frame_analysis": {
                "total_frames": len(frames),
                "analyzed_frames": len(frame_predictions),
                "deepfake_frames": int(sum(frame_predictions)),
                "average_confidence": avg_confidence,
                "confidence_variance": float(np.var(frame_confidences)),
            },
            "temporal_score": float(temporal_score) if temporal_score else None,
            "audio_score": float(audio_score) if audio_score else None,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": processing_time_ms,
            "demo_mode": ensemble_detector.demo_mode,
        }

        _save_detection_record(
            db, current_user, file.filename, "video", bool(final_prediction), float(final_confidence),
            "ensemble", None, processing_time_ms,
        )

        logger.info(f"Video detection complete: {'DEEPFAKE' if final_prediction else 'REAL'} ({final_confidence:.2%})")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=_generic_error_detail(e))
    finally:
        # Always clean up the temp file, success or failure.
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


@app.post("/api/v1/detect/batch", response_model=BatchResult)
@limiter.limit("3/minute")
async def detect_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: Optional[models_db.User] = Depends(get_optional_user),
):
    """Process multiple files in batch."""
    try:
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 files per batch")

        results = []
        for file in files:
            try:
                file_ext = (file.filename or "").lower().split(".")[-1]
                if file_ext in ["jpg", "jpeg", "png"]:
                    result = await detect_image(
                        request, file, return_visualization=False, db=db, current_user=current_user
                    )
                elif file_ext in ["mp4", "avi", "mov"]:
                    result = await detect_video(
                        request, file, sample_rate=10, analyze_audio=False, db=db, current_user=current_user
                    )
                else:
                    result = {"filename": file.filename, "error": "Unsupported file format"}

                result["filename"] = file.filename
                results.append(result)
            except HTTPException as e:
                results.append({"filename": file.filename, "error": e.detail})
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({"filename": file.filename, "error": _generic_error_detail(e)})

        return {
            "total_files": len(files),
            "processed": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=_generic_error_detail(e))


@app.get("/api/v1/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "ensemble": {
            "models": ensemble_detector.get_model_list(),
            "weights": ensemble_detector.get_ensemble_weights(),
        },
        "face_detector": {"type": "MTCNN", "min_face_size": 20},
        "supported_formats": {"image": ["jpg", "jpeg", "png"], "video": ["mp4", "avi", "mov"]},
        "demo_mode": ensemble_detector.demo_mode,
        "demo_mode_notice": (
            "Models are running with randomly-initialized weights and predictions are "
            "not meaningful. Add trained checkpoints to the configured model_weights_dir "
            "to enable real detection."
        ) if ensemble_detector.demo_mode else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug, log_level="info")
