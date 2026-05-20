"""
LSTM Temporal Analyzer for Video Deepfake Detection
Analyzes temporal inconsistencies across video frames
"""

import torch
import torch.nn as nn


class TemporalAnalyzer(nn.Module):
    """
    LSTM-based temporal consistency analyzer for video sequences
    """
    
    def __init__(
        self,
        input_size: int = 2,  # predictions + confidences
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize temporal analyzer
        
        Args:
            input_size: Size of input features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(TemporalAnalyzer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequence (B, T, F) where T is sequence length
        
        Returns:
            Temporal consistency score (0-1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (B, T, H*2)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # (B, T, 1)
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (B, H*2)
        
        # Classify
        output = self.classifier(context)  # (B, 1)
        
        return output.squeeze(-1)
    
    def analyze_consistency(self, predictions, confidences):
        """
        Analyze temporal consistency of predictions
        
        Args:
            predictions: List of frame predictions
            confidences: List of frame confidences
        
        Returns:
            Consistency score and anomaly detection
        """
        # Convert to tensor
        pred_tensor = torch.tensor(predictions, dtype=torch.float32)
        conf_tensor = torch.tensor(confidences, dtype=torch.float32)
        
        # Calculate variance (high variance = inconsistent)
        pred_variance = torch.var(pred_tensor).item()
        conf_variance = torch.var(conf_tensor).item()
        
        # Calculate transition frequency (many flips = suspicious)
        transitions = torch.sum(torch.abs(torch.diff(pred_tensor))).item()
        transition_rate = transitions / len(predictions)
        
        # Overall consistency score (lower is more consistent)
        consistency_score = 1.0 - (pred_variance + conf_variance + transition_rate) / 3.0
        
        return {
            'score': max(0.0, min(1.0, consistency_score)),
            'variance': pred_variance,
            'transitions': transition_rate,
            'is_consistent': consistency_score > 0.7
        }


class TemporalAttentionBlock(nn.Module):
    """Attention block for temporal features"""
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttentionBlock, self).__init__()
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = hidden_dim ** 0.5
        
    def forward(self, x):
        """
        Compute self-attention
        
        Args:
            x: Input tensor (B, T, H)
        
        Returns:
            Attended features
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attention, V)
        
        return output


if __name__ == "__main__":
    # Test temporal analyzer
    model = TemporalAnalyzer()
    
    # Simulate frame predictions and confidences
    batch_size = 4
    sequence_length = 30
    
    dummy_sequence = torch.randn(batch_size, sequence_length, 2)
    output = model(dummy_sequence)
    
    print(f"Input shape: {dummy_sequence.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Test consistency analysis
    predictions = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    confidences = [0.6, 0.8, 0.9, 0.7, 0.85, 0.9, 0.88, 0.65, 0.9, 0.87]
    
    consistency = model.analyze_consistency(predictions, confidences)
    print(f"\nConsistency analysis: {consistency}")
