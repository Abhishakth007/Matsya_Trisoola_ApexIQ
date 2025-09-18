#!/usr/bin/env python3
"""
Robust Vessel Detection Pipeline - Main Execution Script
Production-grade execution with comprehensive monitoring and error handling.
"""

import os
import sys
import logging
import time
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'professor'))

from robust_vessel_detection_pipeline import RobustVesselDetectionPipeline, ProcessingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('RobustPipelineMain')


def load_configuration() -> dict:
    """Load and validate configuration."""
    logger.info("📋 Loading configuration...")
    
    config_path = os.path.join(PROJECT_ROOT, 'src', 'config', 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'main' not in config:
        raise ValueError("Configuration missing 'main' section")
    
    logger.info("✅ Configuration loaded successfully")
    return config['main']


def validate_input_files(config: dict) -> tuple:
    """Validate all required input files exist."""
    logger.info("🔍 Validating input files...")
    
    # Get model paths
    detector_model = config["sentinel1_detector"].replace("${PROJECT_ROOT:-.}", PROJECT_ROOT)
    postprocess_model = config["sentinel1_postprocessor"].replace("${PROJECT_ROOT:-.}", PROJECT_ROOT)
    
    # Validate model files
    if not os.path.exists(detector_model):
        raise FileNotFoundError(f"Detector model not found: {detector_model}")
    if not os.path.exists(postprocess_model):
        raise FileNotFoundError(f"Postprocessor model not found: {postprocess_model}")
    
    # Get input data paths
    input_geotiff = os.path.join(PROJECT_ROOT, 'snap_output', 'step4_final_output.tif')
    if not os.path.exists(input_geotiff):
        raise FileNotFoundError(f"Input GeoTIFF not found: {input_geotiff}")
    
    # SAFE file path for coordinate system
    safe_file_path = os.path.join(PROJECT_ROOT, 'data', 'S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE')
    if not os.path.exists(safe_file_path):
        logger.warning(f"⚠️ SAFE file not found: {safe_file_path} - coordinate accuracy may be reduced")
        safe_file_path = None
    
    # Output directory
    output_dir = os.path.join(PROJECT_ROOT, 'professor', 'outputs')
    
    logger.info("✅ Input file validation passed")
    logger.info(f"   ├── Detector model: {os.path.basename(detector_model)}")
    logger.info(f"   ├── Postprocessor model: {os.path.basename(postprocess_model)}")
    logger.info(f"   ├── Input GeoTIFF: {os.path.basename(input_geotiff)}")
    logger.info(f"   ├── SAFE file: {os.path.basename(safe_file_path) if safe_file_path else 'Not found'}")
    logger.info(f"   └── Output directory: {output_dir}")
    
    return detector_model, postprocess_model, input_geotiff, output_dir, safe_file_path


def create_processing_config(config: dict) -> ProcessingConfig:
    """Create processing configuration from main config."""
    return ProcessingConfig(
        window_size=config.get("default_window_size", 800),
        overlap=config.get("default_overlap", 200),
        confidence_threshold=config.get("thresholds", {}).get("conf_threshold", 0.85),
        nms_threshold=config.get("thresholds", {}).get("nms_threshold", 0.3),
        max_memory_gb=config.get("max_memory_gb", 8.0),
        output_format=config.get("output_format", "json"),
        preserve_crs=config.get("preserve_crs", True)
    )


def main():
    """Main execution function with comprehensive error handling."""
    start_time = time.time()
    
    try:
        logger.info("🚀 ROBUST VESSEL DETECTION PIPELINE - STARTING")
        logger.info("=" * 80)
        
        # Load configuration
        config = load_configuration()
        
        # Validate input files
        detector_model, postprocess_model, input_geotiff, output_dir, safe_file_path = validate_input_files(config)
        
        # Create processing configuration
        processing_config = create_processing_config(config)
        
        # Initialize pipeline
        logger.info("🔧 Initializing robust vessel detection pipeline...")
        pipeline = RobustVesselDetectionPipeline(
            detector_model_path=detector_model,
            postprocess_model_path=postprocess_model,
            device='auto',
            config=processing_config,
            safe_file_path=safe_file_path
        )
        
        # Process scene
        logger.info("🌊 Starting scene processing...")
        results = pipeline.process_scene(
            input_geotiff=input_geotiff,
            output_dir=output_dir,
            water_mask_geotiff=input_geotiff  # Use same file for mask (land=0, sea=non-zero)
        )
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 ROBUST PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📊 Final Results:")
        logger.info(f"   ├── Total detections: {results['metadata']['total_detections']}")
        logger.info(f"   ├── Processing time: {total_time:.2f}s")
        logger.info(f"   ├── Windows processed: {results['metadata']['processing_stats']['processed_windows']}")
        logger.info(f"   └── Output file: {os.path.join(output_dir, 'vessel_detections.json')}")
        
        return results
        
    except FileNotFoundError as e:
        logger.error("❌ FILE NOT FOUND ERROR")
        logger.error(f"Missing file: {e}")
        logger.error("Please check that all required files exist and paths are correct.")
        raise
        
    except ValueError as e:
        logger.error("❌ CONFIGURATION ERROR")
        logger.error(f"Configuration issue: {e}")
        logger.error("Please check your configuration file.")
        raise
        
    except RuntimeError as e:
        logger.error("❌ RUNTIME ERROR")
        logger.error(f"Processing failed: {e}")
        logger.error("Check input data and system resources.")
        raise
        
    except Exception as e:
        logger.error("❌ UNEXPECTED ERROR")
        logger.error(f"Unexpected error: {e}")
        logger.error("Full traceback:", exc_info=True)
        raise


if __name__ == '__main__':
    main()
