#!/usr/bin/env python3
"""
Robust Vessel Detection Pipeline
Production-grade pipeline for vessel detection on SAR imagery using windowed GDAL processing.

Key Features:
- Single clean processing path with fail-fast error handling
- Windowed/tiled processing to avoid memory crashes
- Structured logging and monitoring
- Preserves model interface and output format
- Production-grade error handling and validation
"""

import os
import sys
import logging
import time
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Import SAFE coordinate system
from safe_coordinate_system import SAFECoordinateSystem, create_safe_coordinate_system

# GDAL/Rasterio for efficient raster handling
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import rasterio.features
from rasterio.crs import CRS

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('RobustVesselDetection')


@dataclass
class ProcessingConfig:
    """Configuration for vessel detection processing."""
    window_size: int = 800  # Match training size exactly
    overlap: int = 200  # 25% of 800
    confidence_threshold: float = 0.85
    nms_threshold: float = 0.3  # IoU threshold to prevent duplications
    max_memory_gb: float = 8.0
    output_format: str = 'json'
    preserve_crs: bool = True


@dataclass
class DetectionResult:
    """Single vessel detection result."""
    x: float
    y: float
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_data: Optional[np.ndarray] = None
    detect_id: Optional[int] = None
    preprocess_row: Optional[float] = None
    preprocess_column: Optional[float] = None
    
    # Vessel attributes from postprocessor model
    vessel_length_m: Optional[float] = None
    vessel_width_m: Optional[float] = None
    vessel_speed_k: Optional[float] = None
    is_fishing_vessel: Optional[bool] = None
    vessel_type: Optional[str] = None
    heading_degrees: Optional[float] = None
    heading_bucket_probs: Optional[List[float]] = None


@dataclass
class ProcessingStats:
    """Processing statistics and monitoring data."""
    total_windows: int = 0
    processed_windows: int = 0
    total_detections: int = 0
    processing_time: float = 0.0
    memory_peak_gb: float = 0.0
    input_file_size_gb: float = 0.0


class RobustVesselDetectionPipeline:
    """
    Production-grade vessel detection pipeline with windowed processing.
    
    Features:
    - Single clean processing path
    - Windowed GDAL processing to avoid memory crashes
    - Structured logging and monitoring
    - Fail-fast error handling
    - Preserves model interface and output format
    """
    
    def __init__(self, 
                 detector_model_path: str,
                 postprocess_model_path: str,
                 device: str = 'auto',
                 config: Optional[ProcessingConfig] = None,
                 safe_file_path: Optional[str] = None):
        """
        Initialize the robust vessel detection pipeline.
        
        Args:
            detector_model_path: Path to detector model
            postprocess_model_path: Path to postprocessor model
            device: Device to use ('auto', 'cuda', 'cpu')
            config: Processing configuration
            safe_file_path: Path to SAFE file for coordinate system
        """
        self.detector_model_path = detector_model_path
        self.postprocess_model_path = postprocess_model_path
        self.config = config or ProcessingConfig()
        self.safe_file_path = safe_file_path
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"🔧 Pipeline initialized on device: {self.device}")
        
        # Initialize SAFE coordinate system
        self.safe_coord_system = None
        if safe_file_path and os.path.exists(safe_file_path):
            logger.info("🌍 Initializing SAFE coordinate system...")
            self.safe_coord_system = create_safe_coordinate_system(safe_file_path)
            if self.safe_coord_system:
                logger.info("✅ SAFE coordinate system initialized")
            else:
                logger.warning("⚠️ Failed to initialize SAFE coordinate system - using fallback")
        
        # Initialize crop extraction variables
        self._input_geotiff_path = None
        self.image_width = None
        self.image_height = None
        logger.info(f"   ├── Detector model: {os.path.basename(detector_model_path)}")
        logger.info(f"   ├── Postprocessor model: {os.path.basename(postprocess_model_path)}")
        logger.info(f"   └── Window size: {self.config.window_size}x{self.config.window_size}")
        
        # Load models
        self._load_models()
        
        # Initialize processing stats
        self.stats = ProcessingStats()
    
    def _load_models(self) -> None:
        """Load detector and postprocessor models with validation."""
        logger.info("📦 Loading models...")
        
        # Validate model files/directories exist
        if not os.path.exists(self.detector_model_path):
            raise FileNotFoundError(f"Detector model not found: {self.detector_model_path}")
        if not os.path.exists(self.postprocess_model_path):
            raise FileNotFoundError(f"Postprocessor model not found: {self.postprocess_model_path}")
        
        # Check if paths are directories and contain best.pth
        detector_path = self.detector_model_path
        if os.path.isdir(detector_path):
            detector_path = os.path.join(detector_path, 'best.pth')
            if not os.path.exists(detector_path):
                raise FileNotFoundError(f"Detector model file not found: {detector_path}")
        
        postprocess_path = self.postprocess_model_path
        if os.path.isdir(postprocess_path):
            postprocess_path = os.path.join(postprocess_path, 'best.pth')
            if not os.path.exists(postprocess_path):
                raise FileNotFoundError(f"Postprocessor model file not found: {postprocess_path}")
        
        try:
            # Use the same model loading approach as the previous pipeline
            from pipeline_clean import load_model
            
            # Create example tensor for model loading
            example = [torch.randn(6, 64, 64, device=self.device), None]
            
            # Load detector model
            self.detector_model = load_model(self.detector_model_path, example, self.device)
            if self.detector_model is None:
                raise RuntimeError("Failed to load detector model")
            logger.info("✅ Detector model loaded successfully")
            
            # Load postprocessor model
            self.postprocess_model = load_model(self.postprocess_model_path, example, self.device)
            if self.postprocess_model is None:
                raise RuntimeError("Failed to load postprocessor model")
            logger.info("✅ Postprocessor model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
    
    def process_scene(self, 
                     input_geotiff: str,
                     output_dir: str,
                     water_mask_geotiff: Optional[str] = None) -> Dict[str, Any]:
        """
        Process entire scene with windowed vessel detection.
        
        Args:
            input_geotiff: Path to input SAR GeoTIFF
            output_dir: Output directory for results
            water_mask_geotiff: Path to water mask GeoTIFF (optional)
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        logger.info("🚀 Starting robust vessel detection pipeline")
        logger.info("=" * 80)
        
        try:
            # Validate inputs
            self._validate_inputs(input_geotiff, output_dir, water_mask_geotiff)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Store input GeoTIFF path and dimensions for crop extraction
            self._input_geotiff_path = input_geotiff
            
            # Load and validate input data
            input_metadata = self._load_input_metadata(input_geotiff)
            water_mask = self._load_water_mask(water_mask_geotiff, input_metadata)
            
            # Store image dimensions for crop extraction
            self.image_width = input_metadata['width']
            self.image_height = input_metadata['height']
            
            # Generate processing windows
            windows = self._generate_processing_windows(input_metadata, water_mask)
            self.stats.total_windows = len(windows)
            
            logger.info(f"📊 Processing plan:")
            logger.info(f"   ├── Input dimensions: {input_metadata['width']}x{input_metadata['height']}")
            logger.info(f"   ├── Total windows: {len(windows)}")
            logger.info(f"   ├── Window size: {self.config.window_size}x{self.config.window_size}")
            logger.info(f"   └── Overlap: {self.config.overlap}px")
            
            # Process windows
            all_detections = []
            for i, window in enumerate(windows):
                logger.info(f"🔄 Processing window {i+1}/{len(windows)}: {window}")
                
                try:
                    detections = self._process_window(
                        input_geotiff, window, water_mask, input_metadata
                    )
                    all_detections.extend(detections)
                    self.stats.processed_windows += 1
                    self.stats.total_detections += len(detections)
                    
                    if detections:
                        logger.info(f"   └── Found {len(detections)} detections")
                    
                except Exception as e:
                    logger.error(f"❌ Window {i+1} failed: {e}")
                    raise  # Fail fast - no silent skips
            
            # Post-process and save results
            final_results = self._postprocess_and_save(
                all_detections, input_metadata, output_dir
            )
            
            # Update final statistics
            self.stats.processing_time = time.time() - start_time
            
            logger.info("✅ Pipeline completed successfully")
            logger.info(f"📈 Final statistics:")
            logger.info(f"   ├── Total detections: {self.stats.total_detections}")
            logger.info(f"   ├── Processing time: {self.stats.processing_time:.2f}s")
            logger.info(f"   └── Windows processed: {self.stats.processed_windows}/{self.stats.total_windows}")
            
            return final_results
            
        except Exception as e:
            logger.error("❌ Pipeline failed")
            logger.error(f"Error: {e}")
            raise
    
    def _validate_inputs(self, 
                        input_geotiff: str, 
                        output_dir: str, 
                        water_mask_geotiff: Optional[str]) -> None:
        """Validate all input parameters."""
        logger.info("🔍 Validating inputs...")
        
        # Check input GeoTIFF
        if not os.path.exists(input_geotiff):
            raise FileNotFoundError(f"Input GeoTIFF not found: {input_geotiff}")
        
        # Check water mask if provided
        if water_mask_geotiff and not os.path.exists(water_mask_geotiff):
            raise FileNotFoundError(f"Water mask GeoTIFF not found: {water_mask_geotiff}")
        
        # Check output directory is writable
        try:
            os.makedirs(output_dir, exist_ok=True)
            test_file = os.path.join(output_dir, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"Cannot write to output directory {output_dir}: {e}")
        
        logger.info("✅ Input validation passed")
    
    def _load_input_metadata(self, input_geotiff: str) -> Dict[str, Any]:
        """Load and validate input GeoTIFF metadata."""
        logger.info("📂 Loading input metadata...")
        
        try:
            with rasterio.open(input_geotiff) as src:
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'crs': src.crs,
                    'transform': src.transform,
                    'bands': src.count,
                    'dtype': src.dtypes[0],
                    'nodata': src.nodata,
                    'bounds': src.bounds
                }
                
                # Validate required bands
                if metadata['bands'] < 2:
                    raise ValueError(f"Insufficient bands: {metadata['bands']} (need at least 2 for VV/VH)")
                
                # Get file size
                file_size_bytes = os.path.getsize(input_geotiff)
                self.stats.input_file_size_gb = file_size_bytes / (1024**3)
                
                logger.info(f"✅ Input metadata loaded:")
                logger.info(f"   ├── Dimensions: {metadata['width']}x{metadata['height']}")
                logger.info(f"   ├── Bands: {metadata['bands']}")
                logger.info(f"   ├── CRS: {metadata['crs']}")
                logger.info(f"   ├── Data type: {metadata['dtype']}")
                logger.info(f"   └── File size: {self.stats.input_file_size_gb:.2f} GB")
                
                return metadata
                
        except Exception as e:
            raise RuntimeError(f"Failed to load input metadata: {e}")
    
    def _load_water_mask(self, 
                        water_mask_geotiff: Optional[str], 
                        input_metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load water mask if provided."""
        if not water_mask_geotiff:
            logger.info("🌊 No water mask provided - processing entire image")
            return None
        
        logger.info("🌊 Loading water mask...")
        
        try:
            with rasterio.open(water_mask_geotiff) as src:
                # Validate mask dimensions match input
                if src.width != input_metadata['width'] or src.height != input_metadata['height']:
                    raise ValueError(
                        f"Water mask dimensions {src.width}x{src.height} "
                        f"don't match input {input_metadata['width']}x{input_metadata['height']}"
                    )
                
                # Read mask (assuming single band)
                mask = src.read(1)
                
                # Validate mask values
                unique_values = np.unique(mask)
                logger.info(f"🔍 Mask value range: [{mask.min():.3f}, {mask.max():.3f}]")
                logger.info(f"🔍 Unique mask values (first 10): {unique_values[:10]}")
                
                # Handle continuous mask values - SNAP mask can have continuous values
                # Land pixels are typically 0, sea pixels have various intensities
                # Convert to binary: 0 = land, 1 = sea
                mask = (mask > 0).astype(np.uint8)
                
                sea_pixels = np.sum(mask)
                total_pixels = mask.size
                sea_ratio = sea_pixels / total_pixels * 100
                
                logger.info(f"✅ Water mask loaded:")
                logger.info(f"   ├── Sea pixels: {sea_pixels:,}")
                logger.info(f"   ├── Land pixels: {total_pixels - sea_pixels:,}")
                logger.info(f"   └── Sea ratio: {sea_ratio:.1f}%")
                
                return mask
                
        except Exception as e:
            raise RuntimeError(f"Failed to load water mask: {e}")
    
    def _generate_processing_windows(self, 
                                   input_metadata: Dict[str, Any], 
                                   water_mask: Optional[np.ndarray]) -> List[Window]:
        """Generate processing windows with overlap."""
        logger.info("🪟 Generating processing windows...")
        
        width = input_metadata['width']
        height = input_metadata['height']
        window_size = self.config.window_size
        overlap = self.config.overlap
        step = window_size - overlap
        
        windows = []
        
        for y in range(0, height - window_size + 1, step):
            for x in range(0, width - window_size + 1, step):
                # Create window
                window = Window(x, y, window_size, window_size)
                
                # If water mask provided, check if window has sufficient sea pixels
                if water_mask is not None:
                    window_mask = water_mask[y:y+window_size, x:x+window_size]
                    sea_pixels = np.sum(window_mask)
                    sea_ratio = sea_pixels / (window_size * window_size)
                    
                    # Skip windows with < 10% sea pixels
                    if sea_ratio < 0.1:
                        continue
                
                windows.append(window)
        
        logger.info(f"✅ Generated {len(windows)} processing windows")
        return windows
    
    def _process_window(self, 
                       input_geotiff: str, 
                       window: Window, 
                       water_mask: Optional[np.ndarray],
                       input_metadata: Dict[str, Any]) -> List[DetectionResult]:
        """Process a single window for vessel detection."""
        
        try:
            with rasterio.open(input_geotiff) as src:
                # Read window data
                vv_data = src.read(1, window=window)
                vh_data = src.read(2, window=window)
                
                # Apply water mask if provided
                if water_mask is not None:
                    window_mask = water_mask[window.row_off:window.row_off+window.height,
                                           window.col_off:window.col_off+window.width]
                    vv_data = np.where(window_mask == 1, vv_data, 0)
                    vh_data = np.where(window_mask == 1, vh_data, 0)
                
                # Skip if no valid data
                if np.all(vv_data == 0) and np.all(vh_data == 0):
                    return []
                
                # Prepare model input (6-channel format)
                model_input = self._prepare_model_input(vv_data, vh_data)
                
                # Run detection
                detections = self._run_detection(model_input, window, input_metadata)
                
                return detections
                
        except Exception as e:
            raise RuntimeError(f"Failed to process window {window}: {e}")
    
    def _prepare_model_input(self, vv_data: np.ndarray, vh_data: np.ndarray) -> torch.Tensor:
        """Prepare 6-channel model input from VV/VH data."""
        # Normalize to [0, 1] range
        vv_norm = (vv_data - vv_data.min()) / (vv_data.max() - vv_data.min() + 1e-8)
        vh_norm = (vh_data - vh_data.min()) / (vh_data.max() - vh_data.min() + 1e-8)
        
        # Create 6-channel input: [VH, VV, VH, VV, VH, VV]
        channels = np.stack([vh_norm, vv_norm, vh_norm, vv_norm, vh_norm, vv_norm], axis=0)
        
        # Convert to tensor
        tensor = torch.from_numpy(channels).float().unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def _run_detection(self, 
                      model_input: torch.Tensor, 
                      window: Window, 
                      input_metadata: Dict[str, Any]) -> List[DetectionResult]:
        """Run vessel detection on model input."""
        
        with torch.no_grad():
            # Run detector - expects 4D tensor [batch, channels, height, width]
            detections, _ = self.detector_model(model_input)
            
            # Extract detection results
            detection_results = self._extract_detections(detections, window, input_metadata)
            
            # Run postprocessor to get vessel attributes if we have detections
            if self.postprocess_model and len(detection_results) > 0:
                try:
                    logger.info(f"🔍 Running postprocessor to collect vessel attributes for {len(detection_results)} detections")
                    
                    # Create 2-channel tensor for postprocessor (VV + VH only)
                    # Take only first 2 channels from the 6-channel input
                    postprocess_tensor = model_input[:, :2, :, :]  # Shape: [batch, 2, height, width]
                    postprocess_tensor_list = [postprocess_tensor[0]]  # Convert to list format for postprocessor
                    
                    logger.debug(f"🔍 Postprocessor input tensor shape: {postprocess_tensor.shape}")
                    
                    # Run postprocessor model to get vessel attributes
                    postprocessed_output, _ = self.postprocess_model(postprocess_tensor_list)
                    logger.info(f"🔍 Postprocessor output shape: {postprocessed_output.shape}")
                    
                    # Merge postprocessor results with detections
                    detection_results = self._merge_detections(detection_results, postprocessed_output)
                    logger.info(f"🔍 Combined detector + postprocessor results: {len(detection_results)} detections")
                    
                except Exception as postprocess_error:
                    logger.error(f"❌ Postprocessing failed for window: {postprocess_error}")
                    import traceback
                    logger.error(f"❌ Postprocessing traceback: {traceback.format_exc()}")
                    logger.warning("⚠️ Continuing with detector results only (missing vessel attributes)")
            elif not self.postprocess_model:
                logger.warning("⚠️ No postprocessor model available - vessel attributes will be missing")
            elif len(detection_results) == 0:
                logger.debug("🔍 No detections to postprocess")
            
            return detection_results
    
    def _merge_detections(self, detector_results: List[DetectionResult], postprocess_results: torch.Tensor) -> List[DetectionResult]:
        """Merge detector and postprocessor results to add vessel attributes."""
        try:
            logger.debug(f"🔍 Merging detector results with postprocessor attributes")
            logger.debug(f"🔍 Detector results type: {type(detector_results)}")
            logger.debug(f"🔍 Postprocessor results shape: {postprocess_results.shape}")
            
            # Postprocessor provides: [length, width, heading_classes(16), speed, vessel_type(2)]
            # Shape: [batch_size, 21] where 21 = 1 + 1 + 16 + 1 + 2
            
            if len(detector_results) > 0 and postprocess_results is not None:
                # Extract vessel attributes from postprocessor output
                batch_size = postprocess_results.shape[0]
                num_detections = len(detector_results)
                
                logger.debug(f"🔍 Batch size: {batch_size}, Detections: {num_detections}")
                
                # For each detection, add vessel attributes
                for i, detection in enumerate(detector_results):
                    if i < batch_size:
                        # Extract attributes from postprocessor output
                        length = float(postprocess_results[i, 0])  # Length (normalized 0-1)
                        width = float(postprocess_results[i, 1])   # Width  (normalized 0-1)
                        
                        # Heading: get the class with highest probability
                        heading_logits = postprocess_results[i, 2:18]  # 16 heading classes
                        # Apply softmax to obtain probabilities
                        heading_probs = torch.nn.functional.softmax(heading_logits, dim=0)
                        heading_class = int(torch.argmax(heading_probs).item())
                        heading_degrees = heading_class * 22.5  # Convert class to degrees (16 classes = 22.5° each)
                        
                        speed = float(postprocess_results[i, 18])  # Speed
                        
                        # Vessel type: get the class with highest probability
                        vessel_type_logits = postprocess_results[i, 19:21]  # 2 vessel type classes
                        vessel_type_probs = torch.nn.functional.softmax(vessel_type_logits, dim=0)
                        vessel_type_class = int(torch.argmax(vessel_type_probs).item())
                        is_fishing_vessel = vessel_type_class == 1  # 0 = cargo, 1 = fishing
                        
                        # Add attributes to detection
                        # Scale length/width to meters per docs (x100)
                        detection.vessel_length_m = 100.0 * length
                        detection.vessel_width_m = 100.0 * width
                        detection.vessel_speed_k = speed
                        detection.is_fishing_vessel = is_fishing_vessel
                        detection.vessel_type = "fishing" if is_fishing_vessel else "cargo"
                        detection.heading_degrees = heading_degrees
                        
                        # Store heading probabilities (softmax) for CSV output
                        detection.heading_bucket_probs = [float(heading_probs[j]) for j in range(16)]
                        
                        logger.debug(f"🔍 Added attributes to detection {i}: length={length:.1f}, width={width:.1f}, heading={heading_degrees:.1f}°, speed={speed:.1f}, type={'fishing' if is_fishing_vessel else 'cargo'}")
            
            logger.debug(f"🔍 Successfully merged {len(detector_results)} detections with vessel attributes")
            return detector_results
            
        except Exception as e:
            logger.error(f"❌ Error merging detections with postprocessor results: {e}")
            import traceback
            logger.error(f"❌ Merge traceback: {traceback.format_exc()}")
            return detector_results
    
    def _extract_detections(self, 
                           detections, 
                           window: Window, 
                           input_metadata: Dict[str, Any]) -> List[DetectionResult]:
        """Extract detection results from detector model output."""
        detection_results = []
        
        if not detections:
            return detection_results
        
        # Safe extraction of model output (debug pipeline approach)
        try:
            if isinstance(detections, (list, tuple)) and len(detections) > 0:
                output = detections[0]  # Extract from batch/list
            elif isinstance(detections, dict):
                output = detections  # Already a dict
            else:
                return detection_results
        except (IndexError, TypeError):
            return detection_results
        
        # Check if output is valid and has required keys
        if not isinstance(output, dict) or "boxes" not in output or "scores" not in output:
            return detection_results
        
        # Check if we have valid detections
        try:
            if len(output["boxes"]) == 0 or len(output["scores"]) == 0:
                return detection_results
        except (TypeError, KeyError):
            return detection_results
        
        # Extract boxes and scores
        boxes = output["boxes"]
        scores = output["scores"]
        
        # Convert to geographic coordinates
        transform = input_metadata['transform']
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # Apply confidence threshold
            if score < self.config.confidence_threshold:
                continue
            
            # Extract box coordinates (assuming format: [x1, y1, x2, y2])
            x1, y1, x2, y2 = box
            
            # Convert window-relative coordinates to image coordinates
            img_x1 = window.col_off + x1
            img_y1 = window.row_off + y1
            img_x2 = window.col_off + x2
            img_y2 = window.row_off + y2
            
            # Calculate center point
            center_x = (img_x1 + img_x2) / 2
            center_y = (img_y1 + img_y2) / 2
            
            # Convert to geographic coordinates using SAFE coordinate system
            if self.safe_coord_system:
                try:
                    geo_x, geo_y = self.safe_coord_system.pixel_to_geo(center_x, center_y)
                    logger.debug(f"🌍 SAFE coordinates: ({center_x:.1f}, {center_y:.1f}) -> ({geo_x:.6f}, {geo_y:.6f})")
                except Exception as e:
                    logger.warning(f"⚠️ SAFE coordinate transformation failed: {e}, using fallback")
                    geo_x, geo_y = rasterio.transform.xy(transform, center_y, center_x)
            else:
                # Fallback to rasterio transform (may be inaccurate)
                geo_x, geo_y = rasterio.transform.xy(transform, center_y, center_x)
                logger.warning(f"⚠️ Using fallback coordinates (may be inaccurate): ({geo_x:.6f}, {geo_y:.6f})")
            
            # Create detection result
            detection = DetectionResult(
                x=float(geo_x),
                y=float(geo_y),
                confidence=float(score),
                bbox=(int(img_x1), int(img_y1), int(img_x2), int(img_y2)),
                preprocess_row=float(center_y),
                preprocess_column=float(center_x)
            )
            
            detection_results.append(detection)
        
        return detection_results
    
    def _postprocess_and_save(self, 
                             detections: List[DetectionResult], 
                             input_metadata: Dict[str, Any],
                             output_dir: str) -> Dict[str, Any]:
        """Post-process detections and save results in multiple formats."""
        logger.info("💾 Saving results...")
        
        # Apply NMS if needed
        logger.info(f"🔍 Total detections before NMS: {len(detections)}")
        if len(detections) > 1:
            detections = self._apply_nms(detections)
            logger.info(f"🔧 Applied NMS: {len(detections)} detections after filtering")
        
        # Add detection IDs
        for i, det in enumerate(detections):
            det.detect_id = i
        
        # Save JSON results (original format)
        json_data = {
            'detections': [
                {
                    'x': det.x,
                    'y': det.y,
                    'confidence': det.confidence,
                    'bbox': det.bbox
                }
                for det in detections
            ],
            'metadata': {
                'input_crs': str(input_metadata['crs']),
                'input_bounds': list(input_metadata['bounds']),
                'total_detections': len(detections),
                'processing_stats': {
                    'total_windows': self.stats.total_windows,
                    'processed_windows': self.stats.processed_windows,
                    'processing_time': self.stats.processing_time,
                    'input_file_size_gb': self.stats.input_file_size_gb
                }
            }
        }
        
        json_file = os.path.join(output_dir, 'vessel_detections.json')
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"✅ JSON results saved to: {json_file}")
        
        # Save CSV results (compatible with original pipeline + vessel attributes)
        csv_data = []
        for det in detections:
            # Base detection data
            row_data = {
                'detect_id': det.detect_id,
                'lat': det.y,  # Note: y is latitude in geographic coordinates
                'lon': det.x,  # Note: x is longitude in geographic coordinates
                'confidence': det.confidence,
                'preprocess_row': det.preprocess_row,
                'preprocess_column': det.preprocess_column,
                'bbox_x1': det.bbox[0],
                'bbox_y1': det.bbox[1],
                'bbox_x2': det.bbox[2],
                'bbox_y2': det.bbox[3],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add vessel attributes if available
            if det.vessel_length_m is not None:
                row_data['vessel_length_m'] = det.vessel_length_m
            if det.vessel_width_m is not None:
                row_data['vessel_width_m'] = det.vessel_width_m
            if det.vessel_speed_k is not None:
                row_data['vessel_speed_k'] = det.vessel_speed_k
            if det.is_fishing_vessel is not None:
                row_data['is_fishing_vessel'] = det.is_fishing_vessel
            if det.vessel_type is not None:
                row_data['vessel_type'] = det.vessel_type
            if det.heading_degrees is not None:
                row_data['heading_degrees'] = det.heading_degrees
            
            # Add heading bucket probabilities if available
            if det.heading_bucket_probs is not None:
                for i, prob in enumerate(det.heading_bucket_probs):
                    row_data[f'heading_bucket_{i}'] = prob
            
            csv_data.append(row_data)
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, 'predictions.csv')
        csv_df.to_csv(csv_file, index=False)
        logger.info(f"✅ CSV results saved to: {csv_file}")
        
        # Save vessel crops
        if detections:
            crops_dir = os.path.join(output_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            
            crops_saved = 0
            for det in detections:
                if self._save_detection_crop(det, crops_dir):
                    crops_saved += 1
            
            logger.info(f"✅ Vessel crops saved to: {crops_dir}")
            logger.info(f"   └── {crops_saved}/{len(detections)} crops saved")
        
        logger.info(f"📊 Final results: {len(detections)} detections saved in multiple formats")
        
        return json_data
    
    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) <= 1:
            return detections
        
        # Convert to format for NMS
        boxes_xyxy = np.array([det.bbox for det in detections])
        scores = np.array([det.confidence for det in detections])
        
        # OpenCV dnn.NMSBoxes expects (x, y, w, h). Convert from (x1, y1, x2, y2)
        if boxes_xyxy.size == 0:
            return detections
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        w = (x2 - x1).clip(min=1)
        h = (y2 - y1).clip(min=1)
        boxes_xywh = np.stack([x1, y1, w, h], axis=1).astype(int)
        
        logger.info(f"🔧 NMS input: {len(detections)} boxes | IoU threshold={self.config.nms_threshold}")
        
        # Apply NMS
        # Note: cv2.dnn.NMSBoxes expects (boxes, scores, score_threshold, nms_threshold)
        # score_threshold: minimum confidence to consider (already filtered by confidence_threshold)
        # nms_threshold: IoU threshold for overlap (0.0-1.0, where 0.0 = no overlap allowed, 1.0 = any overlap allowed)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), 
            scores.tolist(), 
            0.0,  # score_threshold: include all detections (confidence filtering done elsewhere)
            self.config.nms_threshold  # nms_threshold: IoU threshold from config
        )
        
        if len(indices) > 0:
            indices = indices.flatten().tolist()
            kept = [detections[i] for i in indices]
            logger.info(f"🔧 NMS kept {len(kept)} / {len(detections)} detections")
            return kept
        else:
            logger.info("🔧 NMS removed all overlapping boxes (0 kept)")
            return []
    
    def _save_detection_crop(self, detection: DetectionResult, crops_dir: str, crop_size: int = 128) -> bool:
        """Save a single detection crop as PNG image by extracting from the original GeoTIFF."""
        try:
            # Extract crop from the original GeoTIFF using the bbox coordinates
            bbox = detection.bbox
            x1, y1, x2, y2 = bbox
            
            # Calculate crop dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Add padding around the detection
            padding = crop_size // 4
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(self.image_width, x2 + padding)
            crop_y2 = min(self.image_height, y2 + padding)
            
            # Read the crop from the original GeoTIFF
            crop_window = Window(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)
            
            # We need to read from the input GeoTIFF - store the path for crop extraction
            if not hasattr(self, '_input_geotiff_path'):
                logger.warning("⚠️ Input GeoTIFF path not available for crop extraction")
                return False
                
            with rasterio.open(self._input_geotiff_path) as src:
                # Read VV band (band 1) for the crop
                crop_data = src.read(1, window=crop_window)
                
                # Normalize to 0-255 range
                if crop_data.max() > crop_data.min():
                    crop_data = ((crop_data - crop_data.min()) / (crop_data.max() - crop_data.min()) * 255).astype(np.uint8)
                else:
                    crop_data = np.zeros_like(crop_data, dtype=np.uint8)
            
            # Save crop as PNG
            crop_filename = f"detection_{detection.detect_id:04d}.png"
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Save as PNG using OpenCV for better control
            cv2.imwrite(crop_path, crop_data)
            
            logger.debug(f"💾 Saved crop: {crop_filename} ({crop_data.shape[1]}x{crop_data.shape[0]})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save crop for detection {detection.detect_id}: {e}")
            return False


def main():
    """Main execution function."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'config.yml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)["main"]
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detector_model = cfg["sentinel1_detector"].replace("${PROJECT_ROOT:-.}", project_root)
    postprocess_model = cfg["sentinel1_postprocessor"].replace("${PROJECT_ROOT:-.}", project_root)
    
    # Input files
    input_geotiff = os.path.join(project_root, 'snap_output', 'step4_final_output.tif')
    water_mask_geotiff = input_geotiff  # Use same file for mask (land=0, sea=non-zero)
    output_dir = os.path.join(project_root, 'professor', 'outputs')
    
    # Build single-source processing config
    proc_cfg = ProcessingConfig(
        window_size=cfg.get("default_window_size", 512),
        overlap=cfg.get("default_overlap", 200),
        confidence_threshold=cfg.get("thresholds", {}).get("conf_threshold", 0.2),
        nms_threshold=cfg.get("thresholds", {}).get("nms_threshold", 0.3),
        max_memory_gb=8.0,
        output_format='json',
        preserve_crs=True
    )

    # Create pipeline with explicit config (no permutations, no overrides)
    pipeline = RobustVesselDetectionPipeline(
        detector_model_path=detector_model,
        postprocess_model_path=postprocess_model,
        device='auto',
        config=proc_cfg
    )
    
    # Process scene
    results = pipeline.process_scene(
        input_geotiff=input_geotiff,
        output_dir=output_dir,
        water_mask_geotiff=water_mask_geotiff
    )
    
    print(f"✅ Pipeline completed successfully!")
    print(f"   └── {results['metadata']['total_detections']} vessels detected")


if __name__ == '__main__':
    main()
