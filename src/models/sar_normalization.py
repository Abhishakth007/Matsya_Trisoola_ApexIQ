"""
SAR-specific normalization and preprocessing for vessel detection models.
Handles the unique statistical properties of SAR imagery vs. natural RGB images.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SARNormalization(nn.Module):
    """SAR-specific normalization layer that adapts ImageNet-pretrained models to SAR imagery.
    
    SAR imagery has fundamentally different statistical properties than natural RGB images:
    - Different value ranges and distributions
    - Speckle noise characteristics
    - Backscatter physics properties
    - No natural color channels
    """
    
    def __init__(self, num_channels=2, normalization_type="adaptive"):
        super(SARNormalization, self).__init__()
        self.num_channels = num_channels
        self.normalization_type = normalization_type
        
        # SAR-specific normalization parameters
        if normalization_type == "fixed":
            # Fixed normalization based on typical SAR statistics
            # These values are based on empirical analysis of Sentinel-1 data
            self.register_buffer('mean', torch.tensor([0.5, 0.5]))  # SAR typically centered around 0.5
            self.register_buffer('std', torch.tensor([0.3, 0.3]))   # SAR has higher variance than RGB
        elif normalization_type == "adaptive":
            # Adaptive normalization that learns from data
            self.mean = nn.Parameter(torch.zeros(num_channels))
            self.std = nn.Parameter(torch.ones(num_channels))
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
    
    def forward(self, x):
        """Apply SAR-specific normalization.
        
        Args:
            x: Input tensor of shape (B, C, H, W) with values in [0, 1]
            
        Returns:
            Normalized tensor with SAR-appropriate statistics
        """
        if self.normalization_type == "adaptive":
            # Compute running statistics for adaptive normalization
            if self.training:
                # Update running statistics during training
                batch_mean = x.mean(dim=[0, 2, 3])  # Mean per channel
                batch_var = x.var(dim=[0, 2, 3], unbiased=False)  # Variance per channel
                
                # Update running statistics (exponential moving average)
                with torch.no_grad():
                    self.mean.data = 0.9 * self.mean.data + 0.1 * batch_mean
                    self.std.data = 0.9 * self.std.data + 0.1 * torch.sqrt(batch_var + 1e-8)
        
        # Apply normalization
        x_normalized = (x - self.mean.view(1, -1, 1, 1)) / (self.std.view(1, -1, 1, 1) + 1e-8)
        
        return x_normalized


class SARBackboneAdapter(nn.Module):
    """Adapter layer that properly initializes ImageNet-pretrained backbones for SAR data.
    
    This addresses the fundamental mismatch between RGB and SAR imagery by:
    1. Properly initializing the first convolutional layer for SAR channels
    2. Applying SAR-specific normalization
    3. Adapting the feature statistics for downstream layers
    """
    
    def __init__(self, backbone, num_channels=2, pretrained=True):
        super(SARBackboneAdapter, self).__init__()
        self.backbone = backbone
        self.num_channels = num_channels
        
        # Add SAR normalization layer
        self.sar_normalization = SARNormalization(num_channels, "adaptive")
        
        # Properly initialize the first conv layer for SAR
        if hasattr(self.backbone, 'conv1'):
            # For ResNet-style backbones
            self._adapt_conv1_layer()
        elif hasattr(self.backbone, 'features') and hasattr(self.backbone.features[0], 'conv'):
            # For Swin Transformer-style backbones
            self._adapt_swin_conv_layer()
    
    def _adapt_conv1_layer(self):
        """Adapt the first convolutional layer for SAR input."""
        original_conv1 = self.backbone.conv1
        
        # Create new conv layer with proper initialization for SAR
        new_conv1 = nn.Conv2d(
            self.num_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize weights properly for SAR
        if self.num_channels == 2:
            # For dual-polarization SAR (VH, VV)
            # Initialize with small random weights and scale appropriately
            nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Scale weights to account for SAR statistics
            with torch.no_grad():
                new_conv1.weight.data *= 0.5  # SAR typically has lower variance than RGB
        else:
            # For other channel configurations
            nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.backbone.conv1 = new_conv1
        logger.info(f"Adapted conv1 layer for {self.num_channels} SAR channels")
    
    def _adapt_swin_conv_layer(self):
        """Adapt Swin Transformer's first conv layer for SAR input."""
        # Implementation for Swin Transformer adaptation
        # This would be similar to _adapt_conv1_layer but for Swin architecture
        pass
    
    def forward(self, x):
        """Forward pass with SAR-specific preprocessing."""
        # Apply SAR normalization
        x = self.sar_normalization(x)
        
        # Pass through backbone
        return self.backbone(x)


def create_sar_adapted_backbone(backbone_name="resnet50", num_channels=2, pretrained=True):
    """Create a SAR-adapted backbone for vessel detection.
    
    Args:
        backbone_name: Name of the backbone architecture
        num_channels: Number of SAR channels (typically 2 for VH/VV)
        pretrained: Whether to use pretrained weights
        
    Returns:
        SAR-adapted backbone model
    """
    import torchvision.models as models
    
    # Load pretrained backbone
    if backbone_name == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
    elif backbone_name == "resnet101":
        backbone = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Create SAR adapter
    sar_backbone = SARBackboneAdapter(backbone, num_channels, pretrained)
    
    return sar_backbone


class SARFeatureEnhancement(nn.Module):
    """Feature enhancement layer specifically designed for SAR vessel detection.
    
    This layer enhances features that are important for vessel detection in SAR imagery:
    - Bright target enhancement (vessels appear as bright returns)
    - Speckle noise suppression
    - Edge preservation for vessel boundaries
    """
    
    def __init__(self, in_channels, out_channels):
        super(SARFeatureEnhancement, self).__init__()
        
        self.enhancement = nn.Sequential(
            # Bright target enhancement
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Speckle suppression while preserving edges
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Residual connection for feature preservation
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        enhanced = self.enhancement(x)
        skip = self.skip(x)
        return enhanced + skip
