"""
Ensemble Deepfake Detection Model
Combines CNN, EfficientNet, Xception, and Vision Transformer
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
import logging

from .cnn_model import DeepfakeCNN
from .efficientnet import EfficientNetDetector
from .xception import XceptionDetector
from .vision_transformer import ViTDetector
from .lstm_temporal import TemporalAnalyzer

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble model combining multiple architectures for robust deepfake detection
    """
    
    def __init__(self, model_dir: str = "models", device: str = None):
        """
        Initialize ensemble detector
        
        Args:
            model_dir: Directory containing model weights
            device: Device to run inference on (cuda/cpu)
        """
        self.model_dir = model_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing ensemble on {self.device}")
        
        # Initialize models
        self.models = {}
        self.load_models()
        
        # Ensemble weights (based on validation performance)
        self.ensemble_weights = {
            'cnn': 0.15,
            'efficientnet': 0.30,
            'xception': 0.35,
            'vit': 0.20
        }
        
        # Temporal analyzer
        self.temporal_analyzer = TemporalAnalyzer().to(self.device)
        
    def load_models(self):
        """Load all detection models"""
        try:
            # CNN Model
            logger.info("Loading CNN model...")
            self.models['cnn'] = DeepfakeCNN().to(self.device)
            self.models['cnn'].eval()
            
            # EfficientNet-B4
            logger.info("Loading EfficientNet-B4...")
            self.models['efficientnet'] = EfficientNetDetector(version='b4').to(self.device)
            self.models['efficientnet'].eval()
            
            # Xception
            logger.info("Loading Xception...")
            self.models['xception'] = XceptionDetector().to(self.device)
            self.models['xception'].eval()
            
            # Vision Transformer
            logger.info("Loading Vision Transformer...")
            self.models['vit'] = ViTDetector().to(self.device)
            self.models['vit'].eval()
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """
        Predict whether image is deepfake using ensemble
        
        Args:
            image: Preprocessed image (H, W, C)
        
        Returns:
            Tuple of (is_deepfake, confidence, model_predictions)
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        
        # Add batch dimension and move to device
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions from each model
        model_predictions = {}
        
        with torch.no_grad():
            # CNN
            cnn_out = self.models['cnn'](image_tensor)
            cnn_prob = torch.sigmoid(cnn_out).item()
            model_predictions['cnn'] = cnn_prob
            
            # EfficientNet
            eff_out = self.models['efficientnet'](image_tensor)
            eff_prob = torch.sigmoid(eff_out).item()
            model_predictions['efficientnet'] = eff_prob
            
            # Xception
            xcp_out = self.models['xception'](image_tensor)
            xcp_prob = torch.sigmoid(xcp_out).item()
            model_predictions['xception'] = xcp_prob
            
            # ViT
            vit_out = self.models['vit'](image_tensor)
            vit_prob = torch.sigmoid(vit_out).item()
            model_predictions['vit'] = vit_prob
        
        # Weighted ensemble
        ensemble_score = (
            self.ensemble_weights['cnn'] * cnn_prob +
            self.ensemble_weights['efficientnet'] * eff_prob +
            self.ensemble_weights['xception'] * xcp_prob +
            self.ensemble_weights['vit'] * vit_prob
        )
        
        # Prediction and confidence
        is_deepfake = ensemble_score > 0.5
        confidence = ensemble_score if is_deepfake else (1 - ensemble_score)
        
        return is_deepfake, confidence, model_predictions
    
    def predict_single_model(self, image: np.ndarray, model_name: str) -> Tuple[bool, float]:
        """
        Predict using a single model
        
        Args:
            image: Preprocessed image
            model_name: Name of model to use
        
        Returns:
            Tuple of (is_deepfake, confidence)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        image_tensor = torch.from_numpy(image).float()
        
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.models[model_name](image_tensor)
            prob = torch.sigmoid(output).item()
        
        is_deepfake = prob > 0.5
        confidence = prob if is_deepfake else (1 - prob)
        
        return is_deepfake, confidence
    
    def analyze_temporal_consistency(
        self, 
        predictions: List[bool], 
        confidences: List[float]
    ) -> float:
        """
        Analyze temporal consistency across video frames
        
        Args:
            predictions: List of frame predictions
            confidences: List of frame confidences
        
        Returns:
            Temporal consistency score (0-1)
        """
        if len(predictions) < 10:
            return None
        
        # Convert to tensors
        pred_tensor = torch.tensor(predictions, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        conf_tensor = torch.tensor(confidences, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        # Combine predictions and confidences
        sequence = torch.cat([pred_tensor, conf_tensor], dim=-1).to(self.device)
        
        # Run through LSTM
        with torch.no_grad():
            temporal_score = self.temporal_analyzer(sequence)
        
        return temporal_score.item()
    
    def get_model(self, model_name: str) -> nn.Module:
        """Get a specific model"""
        return self.models.get(model_name)
    
    def get_model_list(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get ensemble weights"""
        return self.ensemble_weights
    
    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """
        Update ensemble weights
        
        Args:
            new_weights: Dictionary of new weights
        """
        # Validate weights sum to 1
        total = sum(new_weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.ensemble_weights = new_weights
        logger.info(f"Updated ensemble weights: {new_weights}")
