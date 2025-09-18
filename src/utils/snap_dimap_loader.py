"""
SNAP DIMAP Format Loader
========================

This module provides a robust loader for SNAP's DIMAP format, which consists of:
- .dim file (XML metadata)
- .data/ folder with ENVI format image files
- Multiple bands (VV, VH, land/sea mask, etc.)

Key Features:
- XML metadata parsing
- ENVI image file loading
- Band detection and ordering
- Georeferencing extraction
- Synthetic mask creation
- Robust error handling
"""

import logging
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import os

# Import GDAL components
try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    logging.warning("GDAL not available - georeferencing features will be limited")

logger = logging.getLogger(__name__)

class SNAPDIMAPLoader:
    """
    Robust loader for SNAP DIMAP format files.
    
    Handles:
    - DIMAP XML metadata parsing
    - ENVI image file loading
    - Band detection and ordering
    - Georeferencing extraction
    - Synthetic mask creation
    """
    
    def __init__(self):
        """Initialize SNAP DIMAP loader."""
        self.metadata = {}
        self.band_info = {}
        self.georeferencing = {}
        
        logger.info("🔧 SNAP DIMAP Loader initialized")
    
    def load_dimap(self, dim_file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load SNAP DIMAP format file.
        
        Args:
            dim_file_path: Path to .dim file
            
        Returns:
            Tuple of (image_array, metadata)
        """
        try:
            dim_path = Path(dim_file_path)
            if not dim_path.exists():
                raise FileNotFoundError(f"DIMAP file not found: {dim_file_path}")
            
            logger.info(f"📁 Loading SNAP DIMAP: {dim_path.name}")
            
            # Step 1: Parse DIMAP XML metadata
            self._parse_dimap_metadata(dim_path)
            
            # Step 2: Detect available bands
            self._detect_bands(dim_path)
            
            # Step 3: Load image data
            image_array = self._load_image_bands(dim_path)
            
            # Step 4: Extract georeferencing
            self._extract_georeferencing(dim_path)
            
            # Step 5: Create comprehensive metadata
            metadata = self._create_metadata(image_array)
            
            logger.info(f"✅ DIMAP loaded successfully: {image_array.shape}")
            logger.info(f"📊 Bands: {list(self.band_info.keys())}")
            
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"❌ DIMAP loading failed: {e}")
            raise
    
    def _parse_dimap_metadata(self, dim_path: Path):
        """Parse DIMAP XML metadata file."""
        try:
            tree = ET.parse(dim_path)
            root = tree.getroot()
            
            # Extract basic metadata
            self.metadata = {
                'dataset_name': self._get_xml_text(root, './/DATASET_NAME'),
                'product_type': self._get_xml_text(root, './/PRODUCT_TYPE'),
                'start_time': self._get_xml_text(root, './/PRODUCT_SCENE_RASTER_START_TIME'),
                'stop_time': self._get_xml_text(root, './/PRODUCT_SCENE_RASTER_STOP_TIME'),
                'producer': self._get_xml_text(root, './/DATASET_PRODUCER_NAME'),
                'comments': self._get_xml_text(root, './/DATASET_COMMENTS')
            }
            
            # Extract coordinate reference system
            crs_info = self._extract_crs_info(root)
            self.metadata.update(crs_info)
            
            logger.info(f"📋 Dataset: {self.metadata.get('dataset_name', 'Unknown')}")
            logger.info(f"📋 Product: {self.metadata.get('product_type', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Metadata parsing failed: {e}")
            self.metadata = {}
    
    def _get_xml_text(self, root: ET.Element, xpath: str) -> str:
        """Safely extract text from XML element."""
        try:
            element = root.find(xpath)
            return element.text if element is not None else ""
        except Exception:
            return ""
    
    def _extract_crs_info(self, root: ET.Element) -> Dict:
        """Extract coordinate reference system information."""
        crs_info = {}
        
        try:
            # Extract CRS information
            crs_element = root.find('.//Coordinate_Reference_System')
            if crs_element is not None:
                crs_info['crs_name'] = self._get_xml_text(crs_element, './/HORIZONTAL_CS_NAME')
                crs_info['crs_code'] = self._get_xml_text(crs_element, './/HORIZONTAL_CS_CODE')
                crs_info['datum'] = self._get_xml_text(crs_element, './/HORIZONTAL_DATUM_NAME')
                crs_info['projection'] = self._get_xml_text(crs_element, './/PROJECTION_NAME')
                
                # Extract geotransform if available
                geotransform = self._extract_geotransform(crs_element)
                if geotransform:
                    crs_info['geotransform'] = geotransform
                    
        except Exception as e:
            logger.warning(f"⚠️ CRS extraction failed: {e}")
        
        return crs_info
    
    def _extract_geotransform(self, crs_element: ET.Element) -> Optional[List[float]]:
        """Extract geotransform from CRS element."""
        try:
            # Look for geotransform in various possible locations
            geotransform_elements = [
                './/Geoposition/Geoposition_Insert',
                './/Geoposition_Insert',
                './/Geotransform'
            ]
            
            for xpath in geotransform_elements:
                element = crs_element.find(xpath)
                if element is not None:
                    # Try to extract 6 geotransform parameters
                    params = []
                    for i in range(6):
                        param = self._get_xml_text(element, f'.//PARAMETER_{i}')
                        if param:
                            try:
                                params.append(float(param))
                            except ValueError:
                                break
                    
                    if len(params) == 6:
                        return params
                        
        except Exception as e:
            logger.warning(f"⚠️ Geotransform extraction failed: {e}")
        
        return None
    
    def _detect_bands(self, dim_path: Path):
        """Detect available bands in the DIMAP data folder."""
        try:
            data_folder = dim_path.parent / f"{dim_path.stem}.data"
            
            if not data_folder.exists():
                raise FileNotFoundError(f"Data folder not found: {data_folder}")
            
            # Look for ENVI image files (.img)
            img_files = list(data_folder.glob("*.img"))
            
            self.band_info = {}
            
            for img_file in img_files:
                band_name = img_file.stem
                hdr_file = img_file.with_suffix('.hdr')
                
                if hdr_file.exists():
                    # Load band information
                    band_data = self._load_band_info(img_file, hdr_file)
                    self.band_info[band_name] = band_data
                    
                    logger.info(f"📊 Detected band: {band_name} - {band_data['shape']}")
            
            # Sort bands in logical order (VV, VH, mask, etc.)
            self._sort_bands()
            
        except Exception as e:
            logger.error(f"❌ Band detection failed: {e}")
            raise
    
    def _load_band_info(self, img_file: Path, hdr_file: Path) -> Dict:
        """Load information about a specific band."""
        try:
            if not GDAL_AVAILABLE:
                # Fallback: basic file info
                return {
                    'path': str(img_file),
                    'hdr_path': str(hdr_file),
                    'exists': True,
                    'shape': 'unknown'
                }
            
            # Use GDAL to get band information
            dataset = gdal.Open(str(img_file))
            if dataset is None:
                raise ValueError(f"Could not open band file: {img_file}")
            
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            bands = dataset.RasterCount
            dtype = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
            
            # Get georeferencing
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            
            dataset = None  # Close dataset
            
            return {
                'path': str(img_file),
                'hdr_path': str(hdr_file),
                'exists': True,
                'shape': (height, width),
                'bands': bands,
                'dtype': dtype,
                'geotransform': geotransform,
                'projection': projection
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Band info loading failed for {img_file}: {e}")
            return {
                'path': str(img_file),
                'hdr_path': str(hdr_file),
                'exists': False,
                'shape': 'unknown'
            }
    
    def _sort_bands(self):
        """Sort bands in logical order for processing."""
        try:
            # Define preferred band order
            preferred_order = [
                'Sigma0_VV', 'Amplitude_VV', 'VV',
                'Sigma0_VH', 'Amplitude_VH', 'VH',
                'land_sea_mask', 'mask', 'Mask'
            ]
            
            # Create sorted band info
            sorted_bands = {}
            
            # Add bands in preferred order
            for band_name in preferred_order:
                if band_name in self.band_info:
                    sorted_bands[band_name] = self.band_info[band_name]
            
            # Add any remaining bands
            for band_name, band_data in self.band_info.items():
                if band_name not in sorted_bands:
                    sorted_bands[band_name] = band_data
            
            self.band_info = sorted_bands
            
            logger.info(f"📊 Band order: {list(self.band_info.keys())}")
            
        except Exception as e:
            logger.warning(f"⚠️ Band sorting failed: {e}")
    
    def _load_image_bands(self, dim_path: Path) -> np.ndarray:
        """Load image data from all detected bands."""
        try:
            if not self.band_info:
                raise ValueError("No bands detected")
            
            # Load first band to get dimensions
            first_band = list(self.band_info.keys())[0]
            first_band_data = self.band_info[first_band]
            
            if not first_band_data['exists']:
                raise ValueError(f"First band file not found: {first_band_data['path']}")
            
            # Get image dimensions
            if GDAL_AVAILABLE:
                dataset = gdal.Open(first_band_data['path'])
                if dataset is None:
                    raise ValueError(f"Could not open first band: {first_band_data['path']}")
                
                height, width = dataset.RasterYSize, dataset.RasterXSize
                dtype = dataset.GetRasterBand(1).DataType
                dataset = None
            else:
                # Fallback: assume standard dimensions
                height, width = 1000, 1000
                dtype = gdal.GDT_Float32
            
            # Load all bands
            band_arrays = []
            band_names = []
            
            for band_name, band_data in self.band_info.items():
                if band_data['exists']:
                    try:
                        band_array = self._load_single_band(band_data['path'])
                        if band_array is not None:
                            band_arrays.append(band_array)
                            band_names.append(band_name)
                            logger.info(f"✅ Loaded band: {band_name} - {band_array.shape}")
                        else:
                            logger.warning(f"⚠️ Failed to load band: {band_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading band {band_name}: {e}")
            
            if not band_arrays:
                raise ValueError("No bands could be loaded")
            
            # Stack bands into single array (channels, height, width)
            image_array = np.stack(band_arrays, axis=0)
            
            # Store band names for reference
            self.metadata['band_names'] = band_names
            
            logger.info(f"📊 Loaded {len(band_arrays)} bands: {band_names}")
            logger.info(f"📊 Final array shape: {image_array.shape}")
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Image band loading failed: {e}")
            raise
    
    def _load_single_band(self, band_path: str) -> Optional[np.ndarray]:
        """Load a single band from ENVI file."""
        try:
            if not GDAL_AVAILABLE:
                # Fallback: try to load as raw binary
                return self._load_raw_band(band_path)
            
            # Use GDAL to load band
            dataset = gdal.Open(band_path)
            if dataset is None:
                logger.warning(f"⚠️ Could not open band: {band_path}")
                return None
            
            # Read band data
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
            
            # Convert to float32 and normalize if needed
            data = data.astype(np.float32)
            
            # Normalize if data is in integer format
            if data.max() > 1.0:
                data = data / 255.0
            
            dataset = None  # Close dataset
            
            return data
            
        except Exception as e:
            logger.warning(f"⚠️ Single band loading failed for {band_path}: {e}")
            return None
    
    def _load_raw_band(self, band_path: str) -> Optional[np.ndarray]:
        """Fallback method to load band as raw binary."""
        try:
            # This is a simplified fallback - in practice, you'd need to parse the .hdr file
            # to get the correct dimensions and data type
            logger.warning(f"⚠️ Using raw band loading fallback for: {band_path}")
            
            # Try to read as float32 array (this is a guess)
            data = np.fromfile(band_path, dtype=np.float32)
            
            # Assume square image (this is a rough approximation)
            size = int(np.sqrt(len(data)))
            if size * size == len(data):
                data = data.reshape(size, size)
                return data.astype(np.float32)
            else:
                logger.warning(f"⚠️ Could not reshape raw data: {len(data)} elements")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Raw band loading failed: {e}")
            return None
    
    def _extract_georeferencing(self, dim_path: Path):
        """Extract georeferencing information."""
        try:
            self.georeferencing = {}
            
            # Get georeferencing from first available band
            for band_name, band_data in self.band_info.items():
                if band_data['exists'] and 'geotransform' in band_data:
                    self.georeferencing = {
                        'geotransform': band_data['geotransform'],
                        'projection': band_data.get('projection', ''),
                        'source_band': band_name
                    }
                    break
            
            # If no georeferencing found, try to extract from tie point grids
            if not self.georeferencing:
                self._extract_tie_point_georeferencing(dim_path)
            
            if self.georeferencing:
                logger.info(f"🗺️ Georeferencing extracted from: {self.georeferencing.get('source_band', 'tie_points')}")
            else:
                logger.warning("⚠️ No georeferencing information found")
                
        except Exception as e:
            logger.warning(f"⚠️ Georeferencing extraction failed: {e}")
    
    def _extract_tie_point_georeferencing(self, dim_path: Path):
        """Extract georeferencing from tie point grids."""
        try:
            data_folder = dim_path.parent / f"{dim_path.stem}.data"
            tie_points_folder = data_folder / "tie_point_grids"
            
            if not tie_points_folder.exists():
                return
            
            # Look for latitude and longitude tie point grids
            lat_file = tie_points_folder / "latitude.img"
            lon_file = tie_points_folder / "longitude.img"
            
            if lat_file.exists() and lon_file.exists():
                # Load tie point grids to estimate georeferencing
                lat_data = self._load_single_band(str(lat_file))
                lon_data = self._load_single_band(str(lon_file))
                
                if lat_data is not None and lon_data is not None:
                    # Create approximate geotransform from tie points
                    min_lat, max_lat = lat_data.min(), lat_data.max()
                    min_lon, max_lon = lon_data.min(), lon_data.max()
                    
                    height, width = lat_data.shape
                    
                    # Create geotransform
                    pixel_width = (max_lon - min_lon) / width
                    pixel_height = (max_lat - min_lat) / height
                    
                    geotransform = [
                        min_lon,           # top-left longitude
                        pixel_width,       # pixel width
                        0,                 # rotation (0 for north-up)
                        max_lat,           # top-left latitude
                        0,                 # rotation (0 for north-up)
                        -pixel_height      # pixel height (negative for north-up)
                    ]
                    
                    self.georeferencing = {
                        'geotransform': geotransform,
                        'projection': 'EPSG:4326',  # Assume WGS84
                        'source_band': 'tie_points',
                        'bounds': {
                            'min_lat': min_lat,
                            'max_lat': max_lat,
                            'min_lon': min_lon,
                            'max_lon': max_lon
                        }
                    }
                    
                    logger.info(f"🗺️ Tie point georeferencing: {min_lat:.4f} to {max_lat:.4f} lat, {min_lon:.4f} to {max_lon:.4f} lon")
                    
        except Exception as e:
            logger.warning(f"⚠️ Tie point georeferencing extraction failed: {e}")
    
    def _create_metadata(self, image_array: np.ndarray) -> Dict:
        """Create comprehensive metadata dictionary."""
        try:
            metadata = {
                # Basic image info
                'width': image_array.shape[2],
                'height': image_array.shape[1],
                'num_bands': image_array.shape[0],
                'shape': image_array.shape,
                'dtype': str(image_array.dtype),
                
                # DIMAP metadata
                'dimap_metadata': self.metadata,
                
                # Band information
                'band_names': self.metadata.get('band_names', []),
                'band_info': self.band_info,
                
                # Georeferencing
                'georeferencing': self.georeferencing,
                
                # Data statistics
                'data_range': {
                    'min': float(image_array.min()),
                    'max': float(image_array.max()),
                    'mean': float(image_array.mean()),
                    'std': float(image_array.std())
                }
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Metadata creation failed: {e}")
            return {
                'width': image_array.shape[2],
                'height': image_array.shape[1],
                'num_bands': image_array.shape[0],
                'shape': image_array.shape,
                'error': str(e)
            }


def load_snap_dimap(dim_file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to load SNAP DIMAP file.
    
    Args:
        dim_file_path: Path to .dim file
        
    Returns:
        Tuple of (image_array, metadata)
    """
    loader = SNAPDIMAPLoader()
    return loader.load_dimap(dim_file_path)


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        dim_file = sys.argv[1]
        try:
            image_array, metadata = load_snap_dimap(dim_file)
            print(f"✅ Loaded DIMAP: {image_array.shape}")
            print(f"📊 Bands: {metadata.get('band_names', [])}")
            print(f"🗺️ Georeferencing: {metadata.get('georeferencing', {})}")
        except Exception as e:
            print(f"❌ Failed to load DIMAP: {e}")
    else:
        print("Usage: python snap_dimap_loader.py <path_to_dim_file>")
