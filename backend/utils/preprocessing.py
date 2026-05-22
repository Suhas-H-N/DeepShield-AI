"""
Image and Video Preprocessing Utilities
Handles normalization, augmentation, and preparation for model input
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, List, Optional


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: Input image (H, W, C) in RGB format
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize using ImageNet statistics
    
    Returns:
        Preprocessed image ready for model input
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Define preprocessing pipeline
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    
    transform = transforms.Compose(transform_list)
    
    # Apply transformations
    preprocessed = transform(image)
    
    # Convert to numpy for compatibility
    return preprocessed.numpy()


def preprocess_video(
    video_path: str,
    sample_rate: int = 5,
    max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """
    Extract and preprocess frames from video
    
    Args:
        video_path: Path to video file
        sample_rate: Extract every Nth frame
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of preprocessed frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample at specified rate
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def augment_image(image: np.ndarray, mode: str = 'train') -> np.ndarray:
    """
    Apply data augmentation
    
    Args:
        image: Input image
        mode: 'train' or 'test'
    
    Returns:
        Augmented image
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transform(image).numpy()


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image for visualization
    
    Args:
        image: Normalized image tensor (C, H, W)
    
    Returns:
        Denormalized image (H, W, C) in [0, 255]
    """
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    
    # Denormalize
    image = image * std + mean
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    # Convert to uint8 and transpose
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    
    return image


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio with padding
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        pad_color: Color for padding
    
    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Resize
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # Calculate padding offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def apply_frequency_analysis(image: np.ndarray) -> np.ndarray:
    """
    Apply FFT-based frequency analysis to detect artifacts
    
    Args:
        image: Input image in grayscale
    
    Returns:
        Frequency spectrum image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Normalize to 0-255
    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    return magnitude_spectrum


def batch_preprocess(
    images: List[np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess a batch of images
    
    Args:
        images: List of images
        target_size: Target size
    
    Returns:
        Batched tensor (B, C, H, W)
    """
    preprocessed = []
    
    for image in images:
        proc_img = preprocess_image(image, target_size)
        preprocessed.append(proc_img)
    
    # Stack into batch
    batch = torch.from_numpy(np.array(preprocessed)).float()
    
    return batch


if __name__ == "__main__":
    # Test preprocessing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Testing preprocessing...")
    preprocessed = preprocess_image(dummy_image)
    print(f"Original shape: {dummy_image.shape}")
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Test augmentation
    augmented = augment_image(dummy_image, mode='train')
    print(f"Augmented shape: {augmented.shape}")
    
    # Test denormalization
    denorm = denormalize_image(preprocessed)
    print(f"Denormalized shape: {denorm.shape}")
    print(f"Denormalized range: [{denorm.min()}, {denorm.max()}]")
