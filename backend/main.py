"""
Advanced Deepfake Detection System - Main API
FastAPI backend with multi-model ensemble detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import logging
from datetime import datetime
import json

# Import our models and utilities
from models.ensemble import EnsembleDetector
from utils.preprocessing import preprocess_image, preprocess_video
from utils.face_detector import FaceDetector
from utils.video_processor import VideoProcessor
from utils.audio_analyzer import AudioAnalyzer
from utils.explainability import generate_gradcam
from api.schemas import DetectionResult, BatchResult, HealthCheck

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Advanced multi-model deepfake detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (loaded once at startup)
ensemble_detector = None
face_detector = None
video_processor = None
audio_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global ensemble_detector, face_detector, video_processor, audio_analyzer
    
    logger.info("Loading deepfake detection models...")
    try:
        ensemble_detector = EnsembleDetector(model_dir="models")
        face_detector = FaceDetector()
        video_processor = VideoProcessor()
        audio_analyzer = AudioAnalyzer()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "models_loaded": ensemble_detector is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/detect/image", response_model=DetectionResult)
async def detect_image(
    file: UploadFile = File(...),
    return_visualization: bool = True,
    model_type: Optional[str] = "ensemble"
):
    """
    Detect deepfake in a single image
    
    Args:
        file: Image file (JPEG, PNG)
        return_visualization: Whether to return Grad-CAM visualization
        model_type: Model to use (cnn, efficientnet, xception, vit, ensemble)
    
    Returns:
        DetectionResult with prediction and confidence
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Detect and extract face
        logger.info("Detecting faces...")
        faces = face_detector.detect_faces(image_array)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Use the largest face
        face = faces[0]
        face_crop = face_detector.extract_face(image_array, face)
        
        # Preprocess
        preprocessed = preprocess_image(face_crop)
        
        # Run detection
        logger.info(f"Running {model_type} detection...")
        if model_type == "ensemble":
            prediction, confidence, model_predictions = ensemble_detector.predict(preprocessed)
        else:
            prediction, confidence = ensemble_detector.predict_single_model(
                preprocessed, model_type
            )
            model_predictions = {model_type: confidence}
        
        # Generate visualization if requested
        visualization = None
        if return_visualization:
            logger.info("Generating Grad-CAM visualization...")
            visualization = generate_gradcam(
                ensemble_detector.get_model(model_type),
                preprocessed
            )
        
        # Prepare response
        result = {
            "is_deepfake": bool(prediction),
            "confidence": float(confidence),
            "model_predictions": model_predictions,
            "face_detected": True,
            "face_location": {
                "x": int(face['box'][0]),
                "y": int(face['box'][1]),
                "width": int(face['box'][2]),
                "height": int(face['box'][3])
            },
            "visualization": visualization,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": 0  # TODO: Add timing
        }
        
        logger.info(f"Detection complete: {'DEEPFAKE' if prediction else 'REAL'} ({confidence:.2%})")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/v1/detect/video", response_model=DetectionResult)
async def detect_video(
    file: UploadFile = File(...),
    sample_rate: int = 5,
    analyze_audio: bool = True,
    temporal_analysis: bool = True
):
    """
    Detect deepfake in a video file
    
    Args:
        file: Video file (MP4, AVI, MOV)
        sample_rate: Frames to skip between samples
        analyze_audio: Whether to analyze audio track
        temporal_analysis: Whether to perform temporal consistency check
    
    Returns:
        DetectionResult with frame-by-frame and overall predictions
    """
    try:
        # Save uploaded video temporarily
        video_path = f"/tmp/{file.filename}"
        contents = await file.read()
        
        with open(video_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Processing video: {file.filename}")
        
        # Extract frames
        frames = video_processor.extract_frames(video_path, sample_rate=sample_rate)
        logger.info(f"Extracted {len(frames)} frames")
        
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="No frames could be extracted")
        
        # Analyze each frame
        frame_predictions = []
        frame_confidences = []
        
        for idx, frame in enumerate(frames):
            try:
                # Detect face
                faces = face_detector.detect_faces(frame)
                if len(faces) == 0:
                    continue
                
                # Extract and preprocess face
                face_crop = face_detector.extract_face(frame, faces[0])
                preprocessed = preprocess_image(face_crop)
                
                # Predict
                prediction, confidence, _ = ensemble_detector.predict(preprocessed)
                frame_predictions.append(prediction)
                frame_confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error processing frame {idx}: {e}")
                continue
        
        if len(frame_predictions) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in video frames")
        
        # Temporal analysis
        temporal_score = None
        if temporal_analysis and len(frame_predictions) > 10:
            temporal_score = ensemble_detector.analyze_temporal_consistency(
                frame_predictions, frame_confidences
            )
        
        # Audio analysis
        audio_score = None
        if analyze_audio:
            try:
                audio_score = audio_analyzer.analyze(video_path)
                logger.info(f"Audio analysis score: {audio_score:.2%}")
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")
        
        # Aggregate predictions
        avg_confidence = np.mean(frame_confidences)
        deepfake_ratio = np.mean(frame_predictions)
        
        # Final decision (majority vote with confidence weighting)
        final_prediction = deepfake_ratio > 0.5
        
        # Adjust confidence based on temporal and audio scores
        final_confidence = avg_confidence
        if temporal_score is not None:
            final_confidence = (final_confidence + temporal_score) / 2
        if audio_score is not None:
            final_confidence = (final_confidence + audio_score) / 2
        
        result = {
            "is_deepfake": bool(final_prediction),
            "confidence": float(final_confidence),
            "frame_analysis": {
                "total_frames": len(frames),
                "analyzed_frames": len(frame_predictions),
                "deepfake_frames": int(sum(frame_predictions)),
                "average_confidence": float(avg_confidence),
                "confidence_variance": float(np.var(frame_confidences))
            },
            "temporal_score": float(temporal_score) if temporal_score else None,
            "audio_score": float(audio_score) if audio_score else None,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Video detection complete: {'DEEPFAKE' if final_prediction else 'REAL'} ({final_confidence:.2%})")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/v1/detect/batch", response_model=BatchResult)
async def detect_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process multiple files in batch
    
    Args:
        files: List of image/video files
        background_tasks: Background task processor
    
    Returns:
        BatchResult with individual results
    """
    try:
        if len(files) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 files per batch"
            )
        
        results = []
        
        for file in files:
            try:
                # Determine file type
                file_ext = file.filename.lower().split('.')[-1]
                
                if file_ext in ['jpg', 'jpeg', 'png']:
                    result = await detect_image(file, return_visualization=False)
                elif file_ext in ['mp4', 'avi', 'mov']:
                    result = await detect_video(file, sample_rate=10, analyze_audio=False)
                else:
                    result = {
                        "filename": file.filename,
                        "error": "Unsupported file format"
                    }
                
                result["filename"] = file.filename
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "total_files": len(files),
            "processed": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


@app.get("/api/v1/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "ensemble": {
            "models": ensemble_detector.get_model_list(),
            "weights": ensemble_detector.get_ensemble_weights()
        },
        "face_detector": {
            "type": "MTCNN",
            "min_face_size": 20
        },
        "supported_formats": {
            "image": ["jpg", "jpeg", "png"],
            "video": ["mp4", "avi", "mov"]
        }
    }


@app.get("/api/v1/stats")
async def get_statistics():
    """Get detection statistics"""
    # TODO: Implement statistics tracking
    return {
        "total_detections": 0,
        "deepfakes_detected": 0,
        "average_confidence": 0.0,
        "uptime": "N/A"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
