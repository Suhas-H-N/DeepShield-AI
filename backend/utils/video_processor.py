"""
Video Processing Utilities
Handles video frame extraction, analysis, and processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import subprocess
import json
from pathlib import Path


class VideoProcessor:
    """
    Video processing and frame extraction utility
    """
    
    def __init__(self):
        """Initialize video processor"""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0.0
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        
        return info
    
    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 5,
        max_frames: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            List of frames as numpy arrays (RGB)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set start position
        if start_time > 0:
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate end frame
        end_frame = None
        if end_time is not None:
            end_frame = int(end_time * fps)
        
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Check if we've reached the end time
            if end_frame and current_frame > end_frame:
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
    
    def extract_frames_uniform(
        self,
        video_path: str,
        num_frames: int = 30
    ) -> List[np.ndarray]:
        """
        Extract uniformly distributed frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        
        Returns:
            List of uniformly sampled frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        return frames
    
    def extract_keyframes(
        self,
        video_path: str,
        threshold: float = 30.0
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract keyframes based on scene changes
        
        Args:
            video_path: Path to video file
            threshold: Threshold for scene change detection
        
        Returns:
            List of (frame_index, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        keyframes = []
        prev_frame = None
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                # If difference exceeds threshold, it's a keyframe
                if mean_diff > threshold:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append((frame_idx, frame_rgb))
            else:
                # First frame is always a keyframe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.append((frame_idx, frame_rgb))
            
            prev_frame = gray
            frame_idx += 1
        
        cap.release()
        
        return keyframes
    
    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame"
    ):
        """
        Save frames to disk
        
        Args:
            frames: List of frames
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            filename = output_path / f"{prefix}_{idx:04d}.jpg"
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filename), frame_bgr)
    
    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v'
    ):
        """
        Create video from frames
        
        Args:
            frames: List of frames
            output_path: Output video path
            fps: Frames per second
            codec: Video codec
        """
        if len(frames) == 0:
            raise ValueError("No frames to write")
        
        height, width = frames[0].shape[:2]
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create video writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
    
    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Extract audio track from video
        
        Args:
            video_path: Path to video file
            output_path: Output audio file path
        
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.wav'))
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM codec
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            return None
        except FileNotFoundError:
            print("ffmpeg not found. Install ffmpeg to extract audio.")
            return None
    
    def calculate_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate optical flow between two frames
        
        Args:
            frame1: First frame
            frame2: Second frame
        
        Returns:
            Optical flow visualization
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Convert flow to HSV for visualization
        hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return flow_rgb
    
    def detect_scene_changes(
        self,
        video_path: str,
        threshold: float = 30.0
    ) -> List[int]:
        """
        Detect scene change frame indices
        
        Args:
            video_path: Path to video
            threshold: Change detection threshold
        
        Returns:
            List of frame indices where scenes change
        """
        cap = cv2.VideoCapture(video_path)
        
        scene_changes = []
        prev_frame = None
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    scene_changes.append(frame_idx)
            
            prev_frame = gray
            frame_idx += 1
        
        cap.release()
        
        return scene_changes


if __name__ == "__main__":
    # Test video processor
    print("Testing Video Processor...")
    
    processor = VideoProcessor()
    
    # Note: This would need an actual video file to test
    # Uncomment and provide a video path to test
    
    # video_path = "sample_video.mp4"
    # info = processor.get_video_info(video_path)
    # print(f"Video info: {info}")
    
    # frames = processor.extract_frames(video_path, sample_rate=10, max_frames=5)
    # print(f"Extracted {len(frames)} frames")
    
    print("Video processor initialized successfully")
