# Project Architecture Documentation

## System Overview

The Advanced Deepfake Detection System is a comprehensive solution for detecting AI-generated and manipulated media content. The system employs a multi-model ensemble approach combining CNNs, EfficientNet, Xception, and Vision Transformers.

## Architecture Components

### 1. Backend (FastAPI)

**Main Application** (`backend/main.py`)
- RESTful API with FastAPI framework
- Async request handling
- CORS middleware for frontend integration
- Health checks and monitoring endpoints

**Core Endpoints:**
- `POST /api/v1/detect/image` - Single image detection
- `POST /api/v1/detect/video` - Video analysis with frame-by-frame detection
- `POST /api/v1/detect/batch` - Batch processing
- `GET /api/v1/models/info` - Model information
- `GET /api/v1/stats` - System statistics

### 2. Models

**Ensemble Architecture** (`models/ensemble.py`)
- Weighted voting system combining multiple models
- Model predictions: CNN (15%), EfficientNet (30%), Xception (35%), ViT (20%)
- Confidence thresholding and temporal consistency checks

**Individual Models:**

1. **CNN Base Model** (`models/cnn_model.py`)
   - Custom architecture with 10 convolutional layers
   - Batch normalization and dropout regularization
   - ~92.3% accuracy

2. **EfficientNet-B4** (`models/efficientnet.py`)
   - Pre-trained on ImageNet
   - Custom classification head
   - ~95.7% accuracy

3. **XceptionNet** (`models/xception.py`)
   - Depthwise separable convolutions
   - Modified for deepfake detection
   - ~96.2% accuracy

4. **Vision Transformer** (`models/vision_transformer.py`)
   - Attention-based architecture
   - Pre-trained ViT-Base
   - ~94.8% accuracy

5. **LSTM Temporal Analyzer** (`models/lstm_temporal.py`)
   - Bidirectional LSTM for video analysis
   - Temporal consistency detection
   - Attention mechanism

### 3. Utilities

**Preprocessing** (`utils/preprocessing.py`)
- Image normalization with ImageNet statistics
- Data augmentation (rotation, flip, color jitter)
- Frequency analysis for artifact detection

**Face Detection** (`utils/face_detector.py`)
- MTCNN-based face detection and alignment
- Landmark detection for precise alignment
- Face extraction with margin control

**Video Processing** (`utils/video_processor.py`)
- Frame extraction at configurable sample rates
- Keyframe detection for scene changes
- Optical flow analysis
- Audio track extraction

**Audio Analysis** (`utils/audio_analyzer.py`)
- Mel-spectrogram feature extraction
- MFCC (Mel-Frequency Cepstral Coefficients)
- Voice consistency analysis
- Spectral anomaly detection

**Explainability** (`utils/explainability.py`)
- Grad-CAM visualization
- Attention map generation for ViT
- Layer-CAM for refined visualizations

### 4. Frontend (React)

**Components:**

1. **Upload Component** (`components/Upload.jsx`)
   - Drag-and-drop file upload
   - File preview (image/video)
   - Detection options configuration
   - Real-time progress indication

2. **Results Component** (`components/Results.jsx`)
   - Verdict display with confidence scores
   - Model-specific predictions
   - Frame analysis for videos
   - Grad-CAM visualizations
   - Temporal and audio analysis results

3. **History Component** (`components/History.jsx`)
   - Detection history tracking
   - Statistics dashboard
   - Clear history functionality

4. **Header Component** (`components/Header.jsx`)
   - Branding and navigation
   - Feature highlights
   - Model accuracy badges

**Services:**

- **API Service** (`services/api.js`)
  - HTTP client for backend communication
  - File upload handling
  - Error handling and retries

## Data Flow

### Image Detection Flow

```
User Upload → Frontend
     ↓
API Endpoint
     ↓
Face Detection (MTCNN)
     ↓
Preprocessing (Resize, Normalize)
     ↓
Parallel Model Inference
  ├─ CNN
  ├─ EfficientNet
  ├─ Xception
  └─ ViT
     ↓
Ensemble Aggregation (Weighted Voting)
     ↓
Grad-CAM Generation (Optional)
     ↓
Response to Frontend
```

### Video Detection Flow

```
User Upload → Frontend
     ↓
API Endpoint
     ↓
Frame Extraction (Sample Rate)
     ↓
Audio Extraction (Optional)
     ↓
For Each Frame:
  ├─ Face Detection
  ├─ Preprocessing
  └─ Model Inference
     ↓
Temporal Analysis (LSTM)
     ↓
Audio Analysis (Mel-Spectrogram)
     ↓
Aggregated Results
     ↓
Response to Frontend
```

## Model Training Pipeline

### 1. Dataset Preparation
- Download: FaceForensics++, Celeb-DF, DFDC
- Face extraction using MTCNN
- Train/Val/Test split (70/15/15)
- Data augmentation

### 2. Training Process
- Transfer learning from ImageNet
- Fine-tuning all layers
- Loss: Binary Cross-Entropy with label smoothing
- Optimizer: AdamW with cosine annealing
- Early stopping based on validation accuracy

### 3. Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Confusion matrix
- Cross-validation

## Deployment Architecture

### Development Environment
```
Frontend (React) → Port 3000
     ↓
Backend (FastAPI) → Port 8000
     ↓
Models (PyTorch) → GPU/CPU
```

### Production Recommendations
- Docker containerization
- Nginx reverse proxy
- Load balancing for multiple backend instances
- Redis caching for model predictions
- PostgreSQL for detection history
- S3 for media storage

## Security Considerations

1. **Input Validation**
   - File type verification
   - File size limits
   - Malware scanning

2. **Rate Limiting**
   - Per-IP request throttling
   - API key authentication

3. **Data Privacy**
   - No permanent storage of uploaded media
   - Encrypted data transmission (HTTPS)
   - GDPR compliance

## Performance Optimization

1. **Model Optimization**
   - Model quantization (INT8)
   - TensorRT acceleration
   - ONNX export for cross-platform

2. **Caching**
   - Redis for frequent predictions
   - CDN for static assets

3. **Async Processing**
   - Background task queue (Celery)
   - WebSocket for real-time updates

## Monitoring and Logging

1. **Metrics**
   - Request latency
   - Model inference time
   - Error rates
   - Detection accuracy over time

2. **Logging**
   - Structured JSON logging
   - ELK stack integration
   - Alert system for anomalies

## Future Enhancements

1. **Technical**
   - Real-time webcam detection
   - Edge device deployment (TensorFlow Lite)
   - Federated learning for privacy

2. **Features**
   - Multi-language support
   - Blockchain-based verification
   - API rate limiting and monetization

3. **Models**
   - Diffusion model detection (Stable Diffusion)
   - Audio deepfake detection (voice cloning)
   - Text-based deepfake detection

## References

1. Rossler et al. (2019) - FaceForensics++
2. Chollet (2017) - Xception Architecture
3. Tan & Le (2019) - EfficientNet
4. Dosovitskiy et al. (2021) - Vision Transformer
5. Selvaraju et al. (2017) - Grad-CAM
