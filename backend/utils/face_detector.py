"""
Face Detection and Extraction Utility
Uses MTCNN for robust face detection and alignment
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image


class FaceDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks)
    """
    
    def __init__(
        self,
        min_face_size: int = 20,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        device: str = 'cpu'
    ):
        """
        Initialize face detector
        
        Args:
            min_face_size: Minimum face size to detect
            thresholds: Detection thresholds for each stage
            device: Device to run on (cpu/cuda)
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.device = device
        
        # Try to import MTCNN, fallback to OpenCV if not available
        try:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(
                min_face_size=min_face_size,
                thresholds=thresholds,
                device=device,
                keep_all=True
            )
            self.method = 'mtcnn'
        except ImportError:
            print("MTCNN not available, using OpenCV Haar Cascade")
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.method = 'opencv'
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image (H, W, C) in RGB format
        
        Returns:
            List of face detections with bounding boxes and landmarks
        """
        if self.method == 'mtcnn':
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN"""
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
        
        faces = []
        
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                face_info = {
                    'box': box.astype(int).tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(prob),
                    'landmarks': landmark.tolist() if landmark is not None else None
                }
                faces.append(face_info)
        
        # Sort by confidence
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces_cv = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        faces = []
        
        for (x, y, w, h) in faces_cv:
            face_info = {
                'box': [x, y, x+w, y+h],
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'landmarks': None
            }
            faces.append(face_info)
        
        return faces
    
    def extract_face(
        self,
        image: np.ndarray,
        face: Dict,
        margin: float = 0.2,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Extract and align face from image
        
        Args:
            image: Input image
            face: Face detection dictionary
            margin: Margin around face (as fraction of face size)
            target_size: Output size
        
        Returns:
            Cropped and resized face image
        """
        box = face['box']
        x1, y1, x2, y2 = box
        
        # Calculate face dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Add margin
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Expand bounding box
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_resized = cv2.resize(
            face_crop,
            target_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        return face_resized
    
    def extract_aligned_face(
        self,
        image: np.ndarray,
        face: Dict,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Extract face with alignment based on landmarks
        
        Args:
            image: Input image
            face: Face detection with landmarks
            target_size: Output size
        
        Returns:
            Aligned face image or None if landmarks not available
        """
        if face.get('landmarks') is None:
            # Fallback to simple extraction
            return self.extract_face(image, face, target_size=target_size)
        
        landmarks = np.array(face['landmarks'])
        
        # Get eye positions
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle for alignment
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center between eyes
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        h, w = image.shape[:2]
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # Extract face from aligned image
        aligned_face = self.extract_face(aligned, face, target_size=target_size)
        
        return aligned_face
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on image
        
        Args:
            image: Input image
            faces: List of face detections
        
        Returns:
            Image with drawn faces
        """
        output = image.copy()
        
        for face in faces:
            box = face['box']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                output,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),
                2
            )
            
            # Draw confidence
            cv2.putText(
                output,
                f"{confidence:.2f}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw landmarks if available
            if face.get('landmarks') is not None:
                landmarks = face['landmarks']
                for (x, y) in landmarks:
                    cv2.circle(output, (int(x), int(y)), 2, (255, 0, 0), -1)
        
        return output
    
    def filter_faces(
        self,
        faces: List[Dict],
        min_confidence: float = 0.9,
        min_size: int = 50
    ) -> List[Dict]:
        """
        Filter faces based on confidence and size
        
        Args:
            faces: List of face detections
            min_confidence: Minimum confidence threshold
            min_size: Minimum face size (width or height)
        
        Returns:
            Filtered list of faces
        """
        filtered = []
        
        for face in faces:
            box = face['box']
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            if (face['confidence'] >= min_confidence and
                width >= min_size and height >= min_size):
                filtered.append(face)
        
        return filtered


class FaceAligner:
    """
    Advanced face alignment using facial landmarks
    """
    
    def __init__(self):
        """Initialize face aligner"""
        # Standard face template points (normalized coordinates)
        self.template = np.array([
            [0.34, 0.46],  # Left eye
            [0.66, 0.46],  # Right eye
            [0.50, 0.62],  # Nose
            [0.37, 0.82],  # Left mouth
            [0.63, 0.82]   # Right mouth
        ], dtype=np.float32)
    
    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Align face using similarity transform
        
        Args:
            image: Input image
            landmarks: Facial landmarks (5 points)
            target_size: Output size
        
        Returns:
            Aligned face image
        """
        # Scale template to target size
        template_scaled = self.template * target_size
        
        # Estimate similarity transform
        tform = cv2.estimateAffinePartial2D(landmarks, template_scaled)[0]
        
        # Apply transform
        aligned = cv2.warpAffine(
            image,
            tform,
            target_size,
            flags=cv2.INTER_LINEAR
        )
        
        return aligned


if __name__ == "__main__":
    # Test face detector
    print("Testing Face Detector...")
    
    detector = FaceDetector()
    
    # Create dummy image with a face-like pattern
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    faces = detector.detect_faces(dummy_image)
    print(f"Detected {len(faces)} face(s)")
    
    if len(faces) > 0:
        print(f"First face: {faces[0]}")
        
        # Extract face
        face_crop = detector.extract_face(dummy_image, faces[0])
        print(f"Extracted face shape: {face_crop.shape}")
