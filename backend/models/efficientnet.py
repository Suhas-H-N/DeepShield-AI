"""
EfficientNet-based Deepfake Detector
Uses pre-trained EfficientNet with custom classification head
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-based deepfake detection model
    """
    
    def __init__(self, version: str = 'b4', pretrained: bool = True, num_classes: int = 1):
        """
        Initialize EfficientNet detector
        
        Args:
            version: EfficientNet version (b0-b7)
            pretrained: Use ImageNet pre-trained weights
            num_classes: Number of output classes
        """
        super(EfficientNetDetector, self).__init__()
        
        # Load pre-trained EfficientNet
        model_name = f'efficientnet_{version}'
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Logits for classification
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x):
        """Extract feature embeddings"""
        return self.backbone(x)


if __name__ == "__main__":
    # Test the model
    model = EfficientNetDetector(version='b4')
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
