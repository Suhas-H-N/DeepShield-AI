"""
Audio Analysis for Voice Deepfake Detection
Analyzes audio tracks for synthetic speech patterns
"""

import numpy as np
from typing import Tuple, Dict, Optional
import librosa
import librosa.display


class AudioAnalyzer:
    """
    Audio deepfake detection using mel-spectrogram analysis
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Initialize audio analyzer
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Audio signal as numpy array
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features
        
        Args:
            audio: Audio signal
        
        Returns:
            Mel-spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features
        
        Args:
            audio: Audio signal
            n_mfcc: Number of MFCCs to extract
        
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """
        Extract various spectral features
        
        Args:
            audio: Audio signal
        
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Root mean square energy
        features['rms'] = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return features
    
    def detect_anomalies(self, features: Dict) -> Dict:
        """
        Detect anomalies in audio features that may indicate synthetic speech
        
        Args:
            features: Dictionary of audio features
        
        Returns:
            Anomaly detection results
        """
        anomalies = {}
        
        # Check for unnatural spectral patterns
        for feature_name, feature_values in features.items():
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            variance = np.var(feature_values)
            
            anomalies[feature_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'variance': float(variance),
                'suspicious': False
            }
            
            # Simple heuristics for anomaly detection
            # Real implementation would use trained models
            if feature_name == 'zero_crossing_rate':
                # Synthetic speech often has abnormal ZCR
                if std_val < 0.05 or std_val > 0.3:
                    anomalies[feature_name]['suspicious'] = True
            
            elif feature_name == 'spectral_centroid':
                # Check for unnatural frequency distribution
                if mean_val < 500 or mean_val > 5000:
                    anomalies[feature_name]['suspicious'] = True
        
        return anomalies
    
    def analyze(self, audio_path: str) -> Optional[float]:
        """
        Analyze audio file for deepfake indicators
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Deepfake probability score (0-1) or None if analysis fails
        """
        try:
            # Load audio
            audio = self.load_audio(audio_path)
            
            if audio is None or len(audio) == 0:
                return None
            
            # Extract features
            mel_spec = self.extract_mel_spectrogram(audio)
            mfcc = self.extract_mfcc(audio)
            spectral_features = self.extract_spectral_features(audio)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(spectral_features)
            
            # Calculate suspiciousness score
            suspicious_count = sum(
                1 for feat in anomalies.values() if feat['suspicious']
            )
            total_features = len(anomalies)
            
            # Simple scoring (real implementation would use ML model)
            suspiciousness_ratio = suspicious_count / total_features
            
            # Additional checks on mel-spectrogram
            mel_mean = np.mean(mel_spec)
            mel_std = np.std(mel_spec)
            
            # Synthetic speech often has lower variance
            if mel_std < 10:
                suspiciousness_ratio += 0.2
            
            # Calculate final score
            deepfake_score = min(1.0, suspiciousness_ratio)
            
            return deepfake_score
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return None
    
    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch (F0) contour
        
        Args:
            audio: Audio signal
        
        Returns:
            Tuple of (pitch values, voiced flag)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        return f0, voiced_flag
    
    def compare_voice_consistency(
        self,
        audio_segments: list
    ) -> float:
        """
        Compare voice consistency across multiple audio segments
        Inconsistency may indicate deepfake manipulation
        
        Args:
            audio_segments: List of audio arrays
        
        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(audio_segments) < 2:
            return 1.0
        
        # Extract MFCC for each segment
        mfcc_features = []
        for segment in audio_segments:
            mfcc = self.extract_mfcc(segment)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_features.append(mfcc_mean)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(mfcc_features)):
            for j in range(i + 1, len(mfcc_features)):
                # Cosine similarity
                similarity = np.dot(mfcc_features[i], mfcc_features[j]) / (
                    np.linalg.norm(mfcc_features[i]) * 
                    np.linalg.norm(mfcc_features[j])
                )
                similarities.append(similarity)
        
        # Average similarity as consistency score
        consistency = np.mean(similarities)
        
        return float(consistency)


class VoiceActivityDetector:
    """
    Detect voice activity in audio
    """
    
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        """Initialize VAD"""
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def detect(
        self,
        audio: np.ndarray,
        threshold: float = 0.02
    ) -> np.ndarray:
        """
        Detect voice activity
        
        Args:
            audio: Audio signal
            threshold: Energy threshold
        
        Returns:
            Binary mask (1 = voice, 0 = silence)
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # Threshold
        voice_mask = (rms > threshold).astype(int)
        
        return voice_mask


if __name__ == "__main__":
    # Test audio analyzer
    print("Testing Audio Analyzer...")
    
    analyzer = AudioAnalyzer()
    
    # Create dummy audio signal
    duration = 2.0  # seconds
    dummy_audio = np.random.randn(int(duration * analyzer.sample_rate))
    
    # Test feature extraction
    mel_spec = analyzer.extract_mel_spectrogram(dummy_audio)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    
    mfcc = analyzer.extract_mfcc(dummy_audio)
    print(f"MFCC shape: {mfcc.shape}")
    
    spectral_features = analyzer.extract_spectral_features(dummy_audio)
    print(f"Extracted {len(spectral_features)} spectral features")
    
    anomalies = analyzer.detect_anomalies(spectral_features)
    print(f"Anomaly detection results: {len(anomalies)} features analyzed")
    
    print("Audio analyzer initialized successfully")
