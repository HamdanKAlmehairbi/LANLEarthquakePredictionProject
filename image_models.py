"""
MLPER-Inspired Image-Based Model for Earthquake Prediction
Based on Damikoukas & Lagaros (2023) - Encoder-Decoder CNN architecture
that processes spectrograms (magnitude and phase) as 2D images.
"""
import torch
import torch.nn as nn


class MLPERRegressionModel(nn.Module):
    """
    Encoder-Decoder CNN model that processes magnitude and phase spectrograms
    as separate 2D image channels and predicts time-to-failure.
    """
    def __init__(self, num_classes=1):
        super(MLPERRegressionModel, self).__init__()
        
        # Encoder: Process magnitude spectrogram
        self.mag_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),  # Downsample by 2
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),  # Downsample by 2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),  # Downsample by 2
        )
        
        # Encoder: Process phase spectrogram
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
        )
        
        # Decoder: Combine encoded features
        # Using adaptive pooling to handle variable input sizes, then fixed FC layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Fixed output size
        self.decoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 8 * 8, 512),  # 128 channels * 2 (mag+phase) * 8 * 8
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, mag_spec, phase_spec):
        """
        Forward pass through the model.
        
        Args:
            mag_spec: Magnitude spectrogram (batch, freq_bins, time_frames)
            phase_spec: Phase spectrogram (batch, freq_bins, time_frames)
        
        Returns:
            Regression output (batch, 1)
        """
        # Add channel dimension: (batch, freq, time) -> (batch, 1, freq, time)
        mag_spec = mag_spec.unsqueeze(1)
        phase_spec = phase_spec.unsqueeze(1)
        
        # Encode both spectrograms
        mag_encoded = self.mag_encoder(mag_spec)
        phase_encoded = self.phase_encoder(phase_spec)
        
        # Concatenate encoded features along channel dimension
        combined = torch.cat([mag_encoded, phase_encoded], dim=1)
        
        # Use adaptive pooling to get fixed size, then decode
        pooled = self.adaptive_pool(combined)
        output = self.decoder_fc(pooled)
        
        return output

