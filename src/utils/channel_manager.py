"""
Channel Management System for Vessel Detection

This module provides intelligent channel management for Sentinel-1 and Sentinel-2 imagery,
including automatic creation of synthetic overlap channels when historical data is unavailable.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)


class ChannelManager:
    """Intelligent channel management system for vessel detection models."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize ChannelManager with model configuration.
        
        Args:
            model_config: Model configuration dictionary containing channel definitions
        """
        self.required_channels = model_config.get("Channels", [])
        self.base_channels = [ch for ch in self.required_channels if "overlap" not in ch["Name"]]
        self.overlap_channels = [ch for ch in self.required_channels if "overlap" in ch["Name"]]
        
        logger.info(f"ChannelManager initialized with {len(self.base_channels)} base channels and {len(self.overlap_channels)} overlap channels")
    
    def create_channels(self, base_data: List[Union[np.ndarray, torch.Tensor]], 
                       historical_data: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
                       catalog: str = "sentinel1") -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Create all required channels with intelligent fallbacks.
        
        Args:
            base_data: List of base channel data (VH, VV for Sentinel-1)
            historical_data: Optional historical data for temporal overlap channels
            catalog: Imagery catalog ("sentinel1" or "sentinel2")
            
        Returns:
            List of all required channels including synthetic overlaps if needed
        """
        channels = []
        
        # Add base channels
        logger.info(f"Adding {len(self.base_channels)} base channels")
        for i, ch in enumerate(self.base_channels):
            if i < len(base_data):
                channels.append(base_data[i])
            else:
                logger.warning(f"Base channel {ch['Name']} requested but no data available")
                # Create zero-filled channel as fallback
                if base_data:
                    fallback_channel = self._create_zero_channel(base_data[0])
                    channels.append(fallback_channel)
        
        # Add overlap channels
        if historical_data and len(historical_data) >= len(self.base_channels):
            logger.info("Creating historical overlap channels from temporal data")
            channels.extend(self._create_historical_overlaps(historical_data))
        else:
            logger.info("No historical data available - creating synthetic overlap channels")
            channels.extend(self._create_synthetic_overlaps(base_data, catalog))
        
        logger.info(f"Total channels created: {len(channels)}")
        return channels
    
    def _create_synthetic_overlaps(self, base_data: List[Union[np.ndarray, torch.Tensor]], 
                                  catalog: str) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Create realistic overlap channels when historical data unavailable.
        
        Args:
            base_data: Base channel data
            catalog: Imagery catalog type
            
        Returns:
            List of synthetic overlap channels
        """
        overlaps = []
        
        for ch in self.overlap_channels:
            if "vh_overlap" in ch["Name"] and len(base_data) > 0:
                overlap_data = self._create_vh_overlap(base_data[0], catalog)
            elif "vv_overlap" in ch["Name"] and len(base_data) > 1:
                overlap_data = self._create_vv_overlap(base_data[1], catalog)
            else:
                # Fallback: create synthetic channel from first available base channel
                fallback_data = base_data[0] if base_data else None
                overlap_data = self._create_generic_overlap(fallback_data, ch["Name"], catalog)
            
            overlaps.append(overlap_data)
        
        return overlaps
    
    def _create_vh_overlap(self, vh_data: Union[np.ndarray, torch.Tensor], 
                           catalog: str) -> Union[np.ndarray, torch.Tensor]:
        """Create VH overlap channel using spatial averaging and noise injection."""
        if vh_data is None:
            return self._create_zero_channel(vh_data)
        
        # Convert to numpy for processing
        if isinstance(vh_data, torch.Tensor):
            data_np = vh_data.detach().cpu().numpy()
            return_tensor = True
        else:
            data_np = vh_data
            return_tensor = False
        
        # Apply spatial averaging to simulate temporal change
        from scipy import ndimage
        try:
            # Gaussian smoothing to simulate temporal averaging
            smoothed = ndimage.gaussian_filter(data_np, sigma=1.0)
            
            # Add realistic noise (5% of signal range)
            noise_level = 0.05 * (np.max(data_np) - np.min(data_np))
            noise = np.random.normal(0, noise_level, data_np.shape)
            
            # Combine smoothed data with noise
            overlap_data = smoothed + noise
            
            # Ensure values stay in valid range
            overlap_data = np.clip(overlap_data, np.min(data_np), np.max(data_np))
            
        except ImportError:
            # Fallback if scipy not available
            logger.warning("SciPy not available, using simple averaging for overlap channels")
            overlap_data = data_np * 0.95 + np.random.normal(0, 0.02, data_np.shape)
        
        # Return in same format as input
        if return_tensor:
            return torch.from_numpy(overlap_data)
        return overlap_data
    
    def _create_vv_overlap(self, vv_data: Union[np.ndarray, torch.Tensor], 
                           catalog: str) -> Union[np.ndarray, torch.Tensor]:
        """Create VV overlap channel using spatial averaging and noise injection."""
        if vv_data is None:
            return self._create_zero_channel(vv_data)
        
        # Convert to numpy for processing
        if isinstance(vv_data, torch.Tensor):
            data_np = vv_data.detach().cpu().numpy()
            return_tensor = True
        else:
            data_np = vv_data
            return_tensor = False
        
        # Apply different processing for VV channel (typically different characteristics)
        try:
            from scipy import ndimage
            # Median filtering for VV (more robust to outliers)
            overlap_data = ndimage.median_filter(data_np, size=3)
            
            # Add smaller noise for VV (typically more stable)
            noise_level = 0.03 * (np.max(data_np) - np.min(data_np))
            noise = np.random.normal(0, noise_level, data_np.shape)
            overlap_data = overlap_data + noise
            
        except ImportError:
            # Fallback if scipy not available
            logger.warning("SciPy not available, using simple processing for VV overlap")
            overlap_data = data_np * 0.97 + np.random.normal(0, 0.01, data_np.shape)
        
        # Ensure values stay in valid range
        overlap_data = np.clip(overlap_data, np.min(data_np), np.max(data_np))
        
        # Return in same format as input
        if return_tensor:
            return torch.from_numpy(overlap_data)
        return overlap_data
    
    def _create_generic_overlap(self, base_data: Union[np.ndarray, torch.Tensor], 
                               channel_name: str, catalog: str) -> Union[np.ndarray, torch.Tensor]:
        """Create generic overlap channel when specific channel not available."""
        if base_data is None:
            return self._create_zero_channel(base_data)
        
        # Convert to numpy for processing
        if isinstance(base_data, torch.Tensor):
            data_np = base_data.detach().cpu().numpy()
            return_tensor = True
        else:
            data_np = base_data
            return_tensor = False
        
        # Simple synthetic channel creation
        overlap_data = data_np * 0.9 + np.random.normal(0, 0.05, data_np.shape)
        overlap_data = np.clip(overlap_data, np.min(data_np), np.max(data_np))
        
        # Return in same format as input
        if return_tensor:
            return torch.from_numpy(overlap_data)
        return overlap_data
    
    def _create_historical_overlaps(self, historical_data: List[Union[np.ndarray, torch.Tensor]]) -> List[Union[np.ndarray, torch.Tensor]]:
        """Create overlap channels from historical data."""
        overlaps = []
        
        for ch in self.overlap_channels:
            if "vh_overlap" in ch["Name"] and len(historical_data) > 0:
                overlaps.append(historical_data[0])
            elif "vv_overlap" in ch["Name"] and len(historical_data) > 1:
                overlaps.append(historical_data[1])
            else:
                # Use first available historical channel as fallback
                fallback = historical_data[0] if historical_data else None
                overlaps.append(fallback)
        
        return overlaps
    
    def _create_zero_channel(self, reference_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Create zero-filled channel with same shape as reference."""
        if reference_data is None:
            # Default shape if no reference available
            shape = (1, 800, 800)  # Default Sentinel-1 size
        else:
            if isinstance(reference_data, torch.Tensor):
                shape = reference_data.shape
                return torch.zeros(shape, dtype=reference_data.dtype, device=reference_data.device)
            else:
                shape = reference_data.shape
                return np.zeros(shape, dtype=reference_data.dtype)
    
    def validate_channel_count(self, channels: List[Union[np.ndarray, torch.Tensor]]) -> bool:
        """Validate that the correct number of channels was created."""
        expected_count = len(self.required_channels)
        actual_count = len(channels)
        
        if actual_count != expected_count:
            logger.error(f"Channel count mismatch: expected {expected_count}, got {actual_count}")
            return False
        
        logger.info(f"Channel count validation passed: {actual_count} channels")
        return True
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about channel configuration."""
        return {
            "total_channels": len(self.required_channels),
            "base_channels": [ch["Name"] for ch in self.base_channels],
            "overlap_channels": [ch["Name"] for ch in self.overlap_channels],
            "channel_config": self.required_channels
        }
