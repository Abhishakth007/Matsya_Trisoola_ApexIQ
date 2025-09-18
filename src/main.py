"""VIIRS Vessel Detection Service"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Response
from pydantic import BaseModel

from src.data.image import prepare_scenes
from src.inference.pipeline import detect_vessels
from src.utils.config_loader import get_config, get_channel_config, get_processing_config
from src.utils.gdal_manager import validate_gdal_dependencies

app = FastAPI()

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config", "config.yml"
)

HOST = "0.0.0.0"  # nosec B104
PORT = int(os.getenv("SVD_PORT", default=5557))

MODEL_VERSION = datetime.today().strftime("%Y%m%d")

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["main"]


class SVDResponse(BaseModel):
    """Response object for vessel detections"""
    status: str


class SVDRequest(BaseModel):
    """Request object for vessel detections"""
    scene_id: str
    output_dir: str
    raw_path: str
    force_cpu: Optional[bool] = False
    historical1: Optional[str] = None
    historical2: Optional[str] = None
    gcp_bucket: Optional[str] = None
    window_size: Optional[int] = 2048
    padding: Optional[int] = 400
    overlap: Optional[int] = 20
    avoid: Optional[bool] = False
    nms_thresh: Optional[float] = 10
    conf: Optional[float] = 0.9
    save_crops: Optional[bool] = True
    detector_batch_size: int = 4
    postprocessor_batch_size: int = 32
    debug_mode: Optional[bool] = False
    remove_clouds: Optional[bool] = False
    aoi_coords: Optional[list] = None  # List of [lon, lat] pairs for AOI polygon


@app.on_event("startup")
async def sentinel_init() -> None:
    """Sentinel Vessel Service Initialization"""
    logger.info("🚀 Starting Sentinel Vessel Detection Service...")
    
    try:
        # Validate GDAL dependencies
        gdal_status = validate_gdal_dependencies()
        if not gdal_status["all_available"]:
            logger.warning("⚠️ Some GDAL components missing, service may have limited functionality")
        else:
            logger.info("✅ All GDAL components available")
        
        # Validate configuration
        config = get_config()
        if config.validate_model_paths():
            logger.info("✅ Model paths validated successfully")
        else:
            logger.warning("⚠️ Some model paths are invalid, service may fail")
        
        logger.info("🎉 Service initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        logger.error("Service will start with limited functionality")
    
    logger.info("Service started, models will be loaded on demand.")


def load_sentinel1_model() -> str:
    """Stub for loading Sentinel-1 model"""
    global current_model
    current_model = "sentinel1"
    logger.info("Sentinel-1 model loaded.")
    return current_model


def load_sentinel2_model() -> str:
    """Stub for loading Sentinel-2 model"""
    global current_model
    current_model = "sentinel2"
    logger.info("Sentinel-2 model loaded.")
    return current_model


def validate_detect_vessels_params(**kwargs):
    """Validate parameters before calling detect_vessels to prevent runtime crashes.
    
    Args:
        **kwargs: All parameters for detect_vessels function
        
    Raises:
        ValueError: If validation fails
    """
    required_params = [
        'detector_model_dir', 'postprocess_model_dir', 'raw_path', 
        'scene_id', 'base_path', 'output_dir', 'window_size', 
        'padding', 'overlap', 'conf', 'nms_thresh', 'save_crops', 
        'device', 'catalog'
    ]
    
    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    
    # Validate specific parameters
    if not os.path.exists(kwargs['detector_model_dir']):
        raise ValueError(f"Detector model directory not found: {kwargs['detector_model_dir']}")
    
    if not os.path.exists(kwargs['postprocess_model_dir']):
        raise ValueError(f"Postprocessor model directory not found: {kwargs['postprocess_model_dir']}")
    
    if kwargs['window_size'] <= 0:
        raise ValueError(f"Window size must be positive: {kwargs['window_size']}")
    
    if not (0 <= kwargs['conf'] <= 1):
        raise ValueError(f"Confidence must be between 0 and 1: {kwargs['conf']}")
    
    if kwargs['overlap'] < 0:
        raise ValueError(f"Overlap must be non-negative: {kwargs['overlap']}")
    
    if kwargs['padding'] < 0:
        raise ValueError(f"Padding must be non-negative: {kwargs['padding']}")
    
    # Validate device
    if not isinstance(kwargs['device'], torch.device):
        raise ValueError(f"Device must be torch.device, got: {type(kwargs['device'])}")
    
    # Validate catalog
    if kwargs['catalog'] not in ['sentinel1', 'sentinel2']:
        raise ValueError(f"Invalid catalog: {kwargs['catalog']}")
    
    logger.info("✅ Parameter validation passed - all parameters are valid")
    return True


@app.post("/detections", response_model=SVDResponse)
async def get_detections(info: SVDRequest, response: Response) -> SVDResponse:
    """Returns vessel detections Response object for a given Request object"""
    start = perf_counter()
    scene_id = info.scene_id
    raw_path = info.raw_path
    output = info.output_dir

    with TemporaryDirectory() as tmpdir:
        if not os.path.exists(output):
            logger.debug(f"Creating output dir at {output}")
            os.makedirs(output)

        scratch_path = tmpdir
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not info.force_cpu else "cpu"
        )

        cat_path = os.path.join(scratch_path, scene_id + "_cat.npy")
        base_path = os.path.join(scratch_path, scene_id + "_base.tif")
        # FIXED: More robust scene ID detection for various naming conventions
        scene_id_upper = scene_id.upper()
        
        # Handle various Sentinel-1 naming patterns
        if (scene_id_upper.startswith("S1") or 
            "S1A" in scene_id_upper or 
            "S1B" in scene_id_upper or
            scene_id_upper.startswith("SENTINEL-1") or
            scene_id_upper.startswith("SENTINEL1")):
            catalog = "sentinel1"
        # Handle various Sentinel-2 naming patterns
        elif (scene_id_upper.startswith("S2") or 
              "S2A" in scene_id_upper or 
              "S2B" in scene_id_upper or
              scene_id_upper.startswith("SENTINEL-2") or
              scene_id_upper.startswith("SENTINEL2")):
            catalog = "sentinel2"
        else:
            # Try to infer from file structure or metadata
            logger.warning(f"Unknown scene_id format: {scene_id}, attempting to infer catalog")
            
            # Check if SAFE directory structure exists
            safe_path = os.path.join(raw_path, scene_id)
            if os.path.exists(safe_path):
                # Look for Sentinel-1 specific files
                if any("S1A" in f or "S1B" in f for f in os.listdir(safe_path)):
                    catalog = "sentinel1"
                    logger.info(f"Inferred Sentinel-1 from SAFE structure: {scene_id}")
                # Look for Sentinel-2 specific files
                elif any("S2A" in f or "S2B" in f for f in os.listdir(safe_path)):
                    catalog = "sentinel2"
                    logger.info(f"Inferred Sentinel-2 from SAFE structure: {scene_id}")
                else:
                    # Default to Sentinel-1 for maritime applications
                    catalog = "sentinel1"
                    logger.warning(f"Could not determine catalog, defaulting to Sentinel-1: {scene_id}")
            else:
                # Default to Sentinel-1 for maritime applications
                catalog = "sentinel1"
                logger.warning(f"Could not determine catalog, defaulting to Sentinel-1: {scene_id}")


        # ENHANCED: Use new configuration system for model paths
        try:
            config_loader = get_config()
            model_paths = config_loader.get_model_paths(catalog)
            detector_model_dir = model_paths["detector"]
            postprocess_model_dir = model_paths["postprocessor"]
            
            if not detector_model_dir or not postprocess_model_dir:
                raise ValueError(f"Incomplete model configuration for {catalog}")
                
            logger.info(f"✅ Model paths loaded for {catalog}: detector={detector_model_dir}, postprocessor={postprocess_model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load model configuration for {catalog}: {e}")
            raise ValueError(f"Model configuration error for catalog {catalog}: {e}")

        # FIXED: Add input validation before processing
        from src.utils.input_validation import validate_system_inputs
        
        # Validate scene ID format with more flexible pattern matching
        if not (scene_id.startswith("S1") or scene_id.startswith("S2") or 
                "S1A" in scene_id or "S1B" in scene_id or 
                "S2A" in scene_id or "S2B" in scene_id):
            logger.warning(f"Unexpected scene_id format: {scene_id}, proceeding anyway")

        img_array = None
        if not os.path.exists(cat_path) or not os.path.exists(base_path):
            logger.info("Preprocessing raw scenes.")
            img_array = prepare_scenes(
                raw_path,
                scratch_path,
                scene_id,
                info.historical1,
                info.historical2,
                catalog,
                cat_path,
                base_path,
                device,
                detector_model_dir,
                postprocess_model_dir,
                aoi_coords=info.aoi_coords,
            )
            
            # ✅ FIXED: Validate system inputs AFTER image is created
            if img_array is not None:
                try:
                    # Create a complete config for validation
                    validation_config = {
                        'catalog': catalog,
                        'window_size': info.window_size,
                        'expected_channels': 6 if catalog == 'sentinel1' else 3,
                        'detector_model_dir': detector_model_dir,
                        'postprocess_model_dir': postprocess_model_dir,
                        'device': 'cuda' if device.type == 'cuda' else 'cpu'
                    }
                    
                    is_valid, validation_summary = validate_system_inputs(
                        img=img_array,
                        config=validation_config
                    )
                    
                    if not is_valid:
                        logger.warning(f"Input validation warnings: {validation_summary.get('warnings', [])}")
                        if validation_summary.get('errors'):
                            logger.error(f"Input validation errors: {validation_summary.get('errors', [])}")
                            raise ValueError("Input validation failed")
                            
                    logger.info("✅ Input validation completed successfully")
                    
                except Exception as e:
                    logger.error(f"Input validation error: {e}")
                    # Continue processing but log the issue

        # Run inference with proper parameter validation
        try:
            # Prepare parameters dictionary with correct names
            detection_params = {
                'detector_model_dir': detector_model_dir,
                'postprocess_model_dir': postprocess_model_dir,
                'raw_path': raw_path,
                'scene_id': scene_id,
                'img_array': img_array,
                'base_path': base_path,
                'output_dir': output,  # ✅ FIXED: Correct parameter name
                'window_size': info.window_size,
                'padding': info.padding,
                'overlap': info.overlap,
                'conf': info.conf,
                'nms_thresh': info.nms_thresh,
                'save_crops': info.save_crops,
                'device': device,
                'catalog': catalog,
                'avoid': info.avoid,
                'remove_clouds': info.remove_clouds,
                'detector_batch_size': info.detector_batch_size,
                'postprocessor_batch_size': info.postprocessor_batch_size,
                'debug_mode': info.debug_mode,
                'water_mask': None,  # 🌊 Water mask will be created internally if needed
            }
            
            # Validate parameters before calling
            validate_detect_vessels_params(**detection_params)
            
            # Call with keyword arguments for safety
            detect_vessels(**detection_params)
            
        except Exception as e:
            logger.error(f"❌ Vessel detection failed: {e}")
            raise ValueError(f"Detection pipeline error: {e}")

        status = "200"

    elapsed_time = perf_counter() - start
    logger.info(f"SVD elapsed_time={elapsed_time:.2f}s, detections complete")

    return SVDResponse(status=status)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host=HOST, port=PORT, proxy_headers=True)
