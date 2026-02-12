"""CNN + Transformer model for gravitational wave signal detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return x


class CNNEncoder(nn.Module):
    """CNN encoder for feature extraction from time series."""
    
    def __init__(
        self,
        input_channels: int = 1,
        channels: list = [64, 128, 256],
        kernel_sizes: list = [7, 5, 3],
        pool_sizes: list = [2, 2, 2],
    ):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, pool_size in zip(channels, kernel_sizes, pool_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_channels = channels[-1]
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, channels, sequence_length]
        Returns:
            Tensor, shape [batch_size, output_channels, reduced_length]
        """
        return self.conv_layers(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling."""
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # seq_len first for positional encoding
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        # Reshape for transformer: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Reshape back: [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)
        
        return x


class GWDetectorModel(nn.Module):
    """
    Combined CNN + Transformer model for gravitational wave detection.
    
    Architecture:
    1. CNN layers extract local features from time series
    2. Transformer layers model long-range dependencies
    3. Classification head outputs signal/noise prediction
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: list = [64, 128, 256],
        cnn_kernel_sizes: list = [7, 5, 3],
        transformer_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        
        # CNN encoder
        self.cnn_encoder = CNNEncoder(
            input_channels=input_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
        )
        
        # Projection from CNN output to transformer dimension
        cnn_output_dim = cnn_channels[-1]
        self.projection = nn.Linear(cnn_output_dim, transformer_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )
        
        # Classification head
        # Use global average pooling over sequence dimension
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, channels, sequence_length]
        Returns:
            Tensor, shape [batch_size, num_classes]
        """
        # CNN feature extraction
        # x: [batch_size, channels, seq_len]
        cnn_features = self.cnn_encoder(x)  # [batch_size, cnn_channels[-1], reduced_len]
        
        # Reshape for transformer: [batch_size, seq_len, features]
        batch_size, channels, seq_len = cnn_features.shape
        cnn_features = cnn_features.transpose(1, 2)  # [batch_size, seq_len, channels]
        
        # Project to transformer dimension
        x = self.projection(cnn_features)  # [batch_size, seq_len, transformer_dim]
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)  # [batch_size, seq_len, transformer_dim]
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, transformer_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights to prevent numerical instability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


def create_model(config: dict = None):
    """Create model from configuration."""
    if config is None:
        from utils.config_loader import load_config
        config = load_config()
    
    model_config = config.get("model", {}).get("architecture", {})
    
    model = GWDetectorModel(
        input_channels=1,
        cnn_channels=model_config.get("cnn_channels", [64, 128, 256]),
        transformer_dim=model_config.get("transformer_dim", 512),
        transformer_heads=model_config.get("transformer_heads", 8),
        transformer_layers=model_config.get("transformer_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        num_classes=2,
    )
    
    return model
