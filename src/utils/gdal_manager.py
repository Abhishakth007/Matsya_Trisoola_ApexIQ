"""
GDAL Dependency Management System

This module provides robust GDAL component management with validation,
graceful degradation, and comprehensive error handling.
"""

import logging
import os
from typing import Optional, Any, Dict, List

logger = logging.getLogger(__name__)

# Global GDAL component availability flags
GDAL_AVAILABLE = False
OSR_AVAILABLE = False
OGR_AVAILABLE = False

# Import GDAL components with comprehensive error handling
try:
    from osgeo import gdal, ogr, osr
    GDAL_AVAILABLE = True
    OSR_AVAILABLE = True
    OGR_AVAILABLE = True
    logger.info("✅ All GDAL components successfully imported")
except ImportError as e:
    logger.error(f"❌ GDAL components missing: {e}")
    logger.error("Install GDAL with: pip install gdal pyproj")
    
    # Try partial imports
    try:
        from osgeo import gdal
        GDAL_AVAILABLE = True
        logger.warning("⚠️ Only GDAL core available, OSR/OGR missing")
    except ImportError:
        GDAL_AVAILABLE = False
        logger.error("❌ GDAL core also unavailable")
    
    try:
        from osgeo import ogr
        OGR_AVAILABLE = True
    except ImportError:
        OGR_AVAILABLE = False
    
    try:
        from osgeo import osr
        OSR_AVAILABLE = True
    except ImportError:
        OSR_AVAILABLE = False


def validate_gdal_dependencies() -> Dict[str, bool]:
    """
    Validate all required GDAL components are available.
    
    Returns:
        Dictionary with component availability status
    """
    status = {
        "gdal": GDAL_AVAILABLE,
        "ogr": OGR_AVAILABLE,
        "osr": OSR_AVAILABLE,
        "all_available": all([GDAL_AVAILABLE, OSR_AVAILABLE, OGR_AVAILABLE])
    }
    
    logger.info(f"GDAL dependency status: {status}")
    
    if not status["all_available"]:
        missing = []
        if not GDAL_AVAILABLE:
            missing.append("GDAL core")
        if not OSR_AVAILABLE:
            missing.append("OSR (coordinate systems)")
        if not OGR_AVAILABLE:
            missing.append("OGR (vector operations)")
        
        logger.warning(f"Missing GDAL components: {', '.join(missing)}")
        
        # Provide installation guidance
        if os.name == 'nt':  # Windows
            logger.error("On Windows, install GDAL from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal")
        else:  # Unix/Linux/macOS
            logger.error("Install GDAL with: sudo apt-get install python3-gdal (Ubuntu/Debian)")
            logger.error("Or: brew install gdal (macOS)")
            logger.error("Or: pip install gdal pyproj")
    
    return status


def require_gdal_component(component: str) -> None:
    """
    Require a specific GDAL component to be available.
    
    Args:
        component: Component name ("gdal", "ogr", "osr")
        
    Raises:
        RuntimeError: If required component is not available
    """
    component_map = {
        "gdal": GDAL_AVAILABLE,
        "ogr": OGR_AVAILABLE,
        "osr": OSR_AVAILABLE
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown GDAL component: {component}")
    
    if not component_map[component]:
        raise RuntimeError(
            f"Required GDAL component '{component}' is not available. "
            f"Install with: pip install gdal pyproj"
        )


class GDALResourceManager:
    """Context manager for GDAL resources to ensure proper cleanup."""
    
    def __init__(self):
        self.resources: List[Any] = []
    
    def add_resource(self, resource: Any) -> None:
        """Add a GDAL resource for automatic cleanup."""
        self.resources.append(resource)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up all GDAL resources."""
        self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up all managed GDAL resources."""
        for resource in self.resources:
            try:
                if hasattr(resource, 'Close'):
                    resource.Close()
                    logger.debug("Closed GDAL resource")
                elif hasattr(resource, 'None'):
                    resource = None
                    logger.debug("Set GDAL resource to None")
                elif hasattr(resource, 'close'):
                    resource.close()
                    logger.debug("Closed GDAL resource (close method)")
            except Exception as e:
                logger.warning(f"Failed to clean up GDAL resource: {e}")
        
        self.resources.clear()


def robust_gdal_operation(func):
    """
    Decorator for GDAL operations with proper resource cleanup.
    
    Usage:
        @robust_gdal_operation
        def warp_image(in_path, out_path):
            ds = gdal.Open(in_path)
            gdal_resources.append(ds)
            # ... processing ...
            return result
    """
    def wrapper(*args, **kwargs):
        gdal_resources = []
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"GDAL operation failed: {e}")
            
            # Clean up any GDAL resources
            for resource in gdal_resources:
                try:
                    if hasattr(resource, 'Close'):
                        resource.Close()
                    elif hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'None'):
                        resource = None
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up resource during error: {cleanup_error}")
            
            raise
        finally:
            # Ensure cleanup happens even on success
            for resource in gdal_resources:
                try:
                    if hasattr(resource, 'Close'):
                        resource.Close()
                    elif hasattr(resource, 'close'):
                        resource.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up resource in finally: {cleanup_error}")
    
    return wrapper


# Convenience functions for common GDAL operations
def create_spatial_reference(epsg_code: Optional[int] = None) -> Any:
    """
    Create a spatial reference object with error handling.
    
    Args:
        epsg_code: EPSG code for the coordinate system
        
    Returns:
        SpatialReference object
        
    Raises:
        RuntimeError: If OSR is not available
    """
    require_gdal_component("osr")
    
    srs = osr.SpatialReference()
    if epsg_code:
        srs.ImportFromEPSG(epsg_code)
    
    return srs


def create_coordinate_transformer(src_srs: Any, dst_srs: Any) -> Any:
    """
    Create a coordinate transformer with error handling.
    
    Args:
        src_srs: Source spatial reference
        dst_srs: Destination spatial reference
        
    Returns:
        CoordinateTransformation object
        
    Raises:
        RuntimeError: If OSR is not available
    """
    require_gdal_component("osr")
    
    transformer = osr.CoordinateTransformation(src_srs, dst_srs)
    if transformer is None:
        raise RuntimeError("Failed to create coordinate transformer")
    
    return transformer


def open_raster_dataset(path: str, mode: str = "r") -> Any:
    """
    Open a raster dataset with error handling.
    
    Args:
        path: Path to the raster file
        mode: Open mode ("r" for read, "w" for write)
        
    Returns:
        GDAL dataset object
        
    Raises:
        RuntimeError: If GDAL is not available or file cannot be opened
    """
    require_gdal_component("gdal")
    
    dataset = gdal.Open(path, gdal.GA_ReadOnly if mode == "r" else gdal.GA_Update)
    if dataset is None:
        raise RuntimeError(f"Cannot open raster dataset: {path}")
    
    return dataset


def get_raster_info(dataset: Any) -> Dict[str, Any]:
    """
    Get comprehensive information about a raster dataset.
    
    Args:
        dataset: GDAL dataset object
        
    Returns:
        Dictionary with raster information
    """
    require_gdal_component("gdal")
    
    info = {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "band_count": dataset.RasterCount,
        "projection": dataset.GetProjection(),
        "geotransform": dataset.GetGeoTransform(),
        "metadata": dataset.GetMetadata()
    }
    
    # Get band information
    bands = []
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        band_info = {
            "index": i,
            "data_type": band.DataType,
            "no_data_value": band.GetNoDataValue(),
            "scale": band.GetScale(),
            "offset": band.GetOffset(),
            "color_interpretation": band.GetColorInterpretation()
        }
        bands.append(band_info)
    
    info["bands"] = bands
    return info


# Initialize GDAL configuration
if GDAL_AVAILABLE:
    try:
        # Fix GDAL warnings by explicitly setting exception handling
        gdal.UseExceptions()
        
        # Set GDAL configuration options for better performance
        gdal.SetConfigOption("GDAL_CACHEMAX", "512")  # 512MB cache
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")
        
        logger.info("✅ GDAL configuration initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ GDAL configuration failed: {e}")


# Export availability flags for other modules
__all__ = [
    'GDAL_AVAILABLE', 'OSR_AVAILABLE', 'OGR_AVAILABLE',
    'validate_gdal_dependencies', 'require_gdal_component',
    'GDALResourceManager', 'robust_gdal_operation',
    'create_spatial_reference', 'create_coordinate_transformer',
    'open_raster_dataset', 'get_raster_info'
]
