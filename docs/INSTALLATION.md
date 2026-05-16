# Installation and Setup Guide

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores (Intel i5 or equivalent)
- RAM: 8 GB
- Storage: 20 GB free space
- OS: Ubuntu 20.04+, Windows 10+, macOS 10.15+

**Recommended:**
- CPU: 8 cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 16 GB+
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 2060 or better)
- Storage: 50 GB+ SSD
- OS: Ubuntu 22.04 LTS

### Software Dependencies

- Python 3.8 or higher
- Node.js 16.x or higher
- npm 8.x or higher
- Git
- CUDA 11.x (for GPU acceleration, optional but recommended)
- FFmpeg (for video processing)

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/deepfake-detection-system.git
cd deepfake-detection-system
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
cd backend
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support)
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

#### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Download Model Weights

#### Option A: Pre-trained Models (Recommended)

```bash
# Download pre-trained models
python scripts/download_models.py
```

This will download:
- CNN model (~50 MB)
- EfficientNet-B4 (~75 MB)
- Xception (~85 MB)
- Vision Transformer (~300 MB)

#### Option B: Train Your Own Models

See "Training Models" section below.

### 4. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create environment file
echo "REACT_APP_API_URL=http://localhost:8000" > .env
```

### 5. Dataset Setup (Optional - for training)

```bash
cd ../scripts

# Create directory structure
python create_dataset.py --action setup

# Show download instructions
python create_dataset.py --action info

# After downloading datasets, preprocess them
python create_dataset.py --action preprocess

# Split into train/val/test
python create_dataset.py --action split

# Analyze dataset statistics
python create_dataset.py --action analyze
```

For quick testing without real data:

```bash
# Create a small sample dataset
python create_dataset.py --action sample --num_samples 100
```

## Running the Application

### Method 1: Development Mode

#### Terminal 1 - Backend

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

#### Terminal 2 - Frontend

```bash
cd frontend
npm start
```

Frontend will open automatically at: http://localhost:3000

### Method 2: Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up --build
```

Access application at: http://localhost:3000

### Method 3: Production Build

#### Backend

```bash
cd backend
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Frontend

```bash
cd frontend
npm run build
npx serve -s build -l 3000
```

## Training Models

### Prepare Dataset

```bash
cd scripts

# Setup dataset directory
python create_dataset.py --action setup

# Download datasets (follow instructions from --action info)
# Place real images in: data/raw/real/
# Place fake images in: data/raw/fake/

# Preprocess (extract faces)
python create_dataset.py --action preprocess

# Split dataset
python create_dataset.py --action split --train_ratio 0.7 --val_ratio 0.15
```

### Train Models

#### Train All Models

```bash
python train_models.py \
    --model all \
    --train_dir ../data/train \
    --val_dir ../data/val \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

#### Train Individual Model

```bash
# Train only CNN
python train_models.py --model cnn --epochs 30

# Train only EfficientNet
python train_models.py --model efficientnet --epochs 50

# Train only Xception
python train_models.py --model xception --epochs 50

# Train only ViT
python train_models.py --model vit --epochs 50
```

### Evaluate Models

```bash
python evaluate.py --test_dir ../data/test --batch_size 32
```

This will generate:
- Performance metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrices
- ROC curves
- Model comparison charts

## Troubleshooting

### Common Issues

#### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size
python train_models.py --batch_size 16

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

#### Issue: Port Already in Use

**Solution:**
```bash
# Backend - use different port
uvicorn main:app --port 8001

# Frontend - use different port
PORT=3001 npm start
```

#### Issue: Module Not Found Errors

**Solution:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or for specific package
pip install --upgrade <package-name>
```

#### Issue: FFmpeg Not Found (Video Processing)

**Solution:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from: https://ffmpeg.org/download.html
Add to PATH environment variable

#### Issue: Face Detection Failing

**Solution:**
```bash
# Install MTCNN properly
pip uninstall facenet-pytorch
pip install facenet-pytorch

# Or use OpenCV fallback (automatic)
```

#### Issue: Frontend Not Connecting to Backend

**Solution:**
1. Check backend is running: http://localhost:8000
2. Check CORS settings in `backend/main.py`
3. Verify `.env` file in frontend has correct API URL
4. Clear browser cache

## Configuration

### Backend Configuration

Edit `backend/main.py`:

```python
# CORS settings
allow_origins=["http://localhost:3000", "http://yourdomain.com"]

# Model directory
model_dir = "path/to/your/models"

# Device selection
device = "cuda"  # or "cpu"
```

### Frontend Configuration

Edit `frontend/.env`:

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_MAX_FILE_SIZE=100  # MB
REACT_APP_SUPPORTED_FORMATS=jpg,png,mp4,avi,mov
```

### Ensemble Weights

Edit `backend/models/ensemble.py`:

```python
self.ensemble_weights = {
    'cnn': 0.15,
    'efficientnet': 0.30,
    'xception': 0.35,
    'vit': 0.20
}
```

## Performance Optimization

### GPU Acceleration

```bash
# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available())"

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU (not yet implemented)
# export CUDA_VISIBLE_DEVICES=0,1
```

### Model Optimization

```python
# Enable TensorFlow mixed precision (coming soon)
# Enable model quantization for faster inference
```

### Caching

Install Redis for caching predictions:

```bash
# Ubuntu
sudo apt-get install redis-server

# Start Redis
redis-server

# Configure in backend
REDIS_URL=redis://localhost:6379
```

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Integration Tests

```bash
# Start backend and frontend, then:
npm run test:e2e
```

## Next Steps

1. **Add your datasets** - Follow dataset preparation guide
2. **Train models** - Or use pre-trained weights
3. **Customize** - Adjust models, weights, and UI
4. **Deploy** - Use Docker for production deployment

## Getting Help

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for API details
- Open an issue on GitHub
- Contact: your.email@university.edu

## License

MIT License - see LICENSE file for details
