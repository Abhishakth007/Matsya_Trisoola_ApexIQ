# vessel-detection-sentinels/professor/masking.py

import numpy as np
import logging
from typing import Optional, Dict
from pipeline_clean import CONFIG
import cv2

logger = logging.getLogger("Professor.Masking")

class Masking:
    """
    A dedicated module for all land-sea masking logic.
    Centralizes mask creation, interpretation, and application.
    """
    
    def __init__(self):
        # SIMPLIFIED: No land-only mode needed - we want to detect vessels on water
        logger.info("Masking module initialized: using SNAP mask directly")

    def create_water_mask_from_image(self, image_array: np.ndarray, band_map: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        SIMPLIFIED: SNAP mask is already perfect - no need to recreate it.
        This function is kept for compatibility but should not be used.
        """
        logger.warning("⚠️ create_water_mask_from_image() called - SNAP mask should be used directly!")
        logger.warning("   └── This function is redundant since SNAP mask is perfect")
        
        # Emergency fallback - simple extraction from SNAP data
        if band_map and ('vv' in band_map and 'vh' in band_map):
            vv = image_array[band_map['vv']]
            vh = image_array[band_map['vh']]
        else:
            vv = image_array[0]
            vh = image_array[1]
        
        # SNAP mask: land=0, sea=original (non-zero)
        mask = (vv > 0) | (vh > 0)  # Any non-zero pixel is sea
        logger.info(f"✅ Emergency SNAP mask extraction: {np.sum(mask):,} sea pixels")
        return mask.astype(bool)

    def get_min_water_fraction_per_window(self) -> float:
        # SIMPLIFIED: No water fraction needed since SNAP mask is perfect
        return 0.0  # All non-zero pixels are sea

    def get_force_land_only_mode(self) -> bool:
        # SIMPLIFIED: No land-only mode - we want to detect vessels on water
        return False
