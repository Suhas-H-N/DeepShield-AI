"""
Explainability Module using Grad-CAM
Generates visual explanations for model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import base64
from io import BytesIO
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which regions the model focuses on for detection
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for visualization
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        if target_layer:
            self.register_hooks(target_layer)
        else:
            self.register_hooks_auto()
    
    def register_hooks(self, layer_name: str):
        """Register forward and backward hooks on target layer"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def register_hooks_auto(self):
        """Automatically find and register hooks on last conv layer"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find last convolutional layer
        target_module = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_module = module
        
        if target_module:
            target_module.register_forward_hook(forward_hook)
            target_module.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None for highest scoring class)
        
        Returns:
            Heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[:, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize(
        self,
        input_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            input_image: Original image (H, W, C)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay
        
        Returns:
            Visualization image
        """
        # Resize heatmap to match input image
        heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure input image is in correct format
        if input_image.dtype != np.uint8:
            input_image = (input_image * 255).astype(np.uint8)
        
        # Overlay heatmap on image
        visualization = cv2.addWeighted(
            input_image,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return visualization


def generate_gradcam(
    model: torch.nn.Module,
    input_image: np.ndarray,
    return_base64: bool = True
) -> Optional[str]:
    """
    Generate Grad-CAM visualization for an image
    
    Args:
        model: PyTorch model
        input_image: Preprocessed image (C, H, W) or (H, W, C)
        return_base64: Return as base64 encoded string
    
    Returns:
        Visualization as base64 string or numpy array
    """
    try:
        # Prepare input
        if len(input_image.shape) == 3:
            if input_image.shape[0] == 3:  # (C, H, W)
                input_tensor = torch.from_numpy(input_image).unsqueeze(0).float()
            else:  # (H, W, C)
                input_tensor = torch.from_numpy(
                    np.transpose(input_image, (2, 0, 1))
                ).unsqueeze(0).float()
        else:
            return None
        
        # Create Grad-CAM instance
        grad_cam = GradCAM(model)
        
        # Generate heatmap
        heatmap = grad_cam.generate_cam(input_tensor)
        
        # Denormalize input for visualization
        from utils.preprocessing import denormalize_image
        
        if input_image.shape[0] == 3:  # (C, H, W)
            img_for_vis = denormalize_image(input_image)
        else:  # (H, W, C)
            img_for_vis = input_image
        
        # Create visualization
        visualization = grad_cam.visualize(img_for_vis, heatmap)
        
        if return_base64:
            # Convert to base64
            pil_image = Image.fromarray(visualization)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        else:
            return visualization
            
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None


class LayerCAM:
    """
    Layer-CAM: More refined version of Grad-CAM
    """
    
    def __init__(self, model: torch.nn.Module):
        """Initialize Layer-CAM"""
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Generate Layer-CAM heatmap
        
        Args:
            input_tensor: Input tensor
        
        Returns:
            Heatmap
        """
        # Similar to Grad-CAM but with element-wise multiplication
        # instead of global pooling
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Element-wise multiplication and sum
        cam = torch.sum(gradients * activations, dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy()


def create_attention_map(
    attention_weights: torch.Tensor,
    input_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Create attention map visualization for Vision Transformer
    
    Args:
        attention_weights: Attention weights from ViT
        input_size: Size to resize attention map
    
    Returns:
        Attention map visualization
    """
    # Average attention across heads
    attention_avg = torch.mean(attention_weights, dim=0)
    
    # Get attention for CLS token
    cls_attention = attention_avg[0, 1:]  # Exclude CLS token itself
    
    # Reshape to spatial dimensions
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_map = cls_attention.reshape(grid_size, grid_size)
    
    # Resize to input size
    attention_map = cv2.resize(
        attention_map.cpu().numpy(),
        input_size,
        interpolation=cv2.INTER_LINEAR
    )
    
    # Normalize
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    return attention_map


if __name__ == "__main__":
    # Test Grad-CAM
    print("Testing Grad-CAM...")
    
    # Create a simple dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(32, 1)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    grad_cam = GradCAM(model)
    heatmap = grad_cam.generate_cam(dummy_input)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print("Grad-CAM test successful!")
