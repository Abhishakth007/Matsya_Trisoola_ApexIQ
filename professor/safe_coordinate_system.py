#!/usr/bin/env python3
"""
SAFE Coordinate System - Extract and map coordinates from raw SAFE files
Provides accurate georeferencing by using the original SAR coordinate grid points.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import griddata
import rasterio

logger = logging.getLogger(__name__)


class SAFECoordinateSystem:
    """
    Coordinate system based on SAFE file geolocation grid points.
    Maps pixel coordinates from processed GeoTIFF to real geographic coordinates.
    """
    
    def __init__(self, safe_file_path: str):
        """
        Initialize with SAFE file path.
        
        Args:
            safe_file_path: Path to the SAFE directory
        """
        self.safe_file_path = safe_file_path
        self.grid_points = []  # List of (pixel, line, lon, lat)
        self.interpolator = None  # Flag to indicate if interpolator is built
        self.image_bounds = None
        
    def extract_grid_points(self) -> bool:
        """
        Extract geolocation grid points from SAFE annotation files.
        Uses the proven approach from functions_post_snap.py.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"🔍 Extracting geolocation grid points from SAFE file: {self.safe_file_path}")
        
        # Use the proven approach from functions_post_snap.py
        return self._extract_from_annotation_xml()
    
    def _extract_from_kml(self, kml_path: str) -> bool:
        """Extract coordinates from KML file (primary method)."""
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            # Find coordinates element
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            coords_elem = root.find('.//kml:coordinates', ns)
            
            if coords_elem is None:
                logger.error("❌ No coordinates found in KML file")
                return False
            
            # Parse coordinates
            coords_text = coords_elem.text.strip()
            coordinates = []
            
            for coord_pair in coords_text.split():
                lon, lat = map(float, coord_pair.split(','))
                coordinates.append((lat, lon))  # Note: lat, lon order
            
            if len(coordinates) < 4:
                logger.error("❌ Insufficient coordinates in KML file")
                return False
            
            # Create a simple grid from the footprint
            # This is a simplified approach - we'll create grid points from the bounds
            lats = [coord[0] for coord in coordinates]
            lons = [coord[1] for coord in coordinates]
            
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Create a simple 10x10 grid across the footprint
            # Note: This is a simplified approach. For production, you'd want to use
            # the actual geolocation grid points from annotation XML files
            grid_points = []
            for i in range(10):
                for j in range(10):
                    lat = min_lat + (max_lat - min_lat) * i / 9
                    lon = min_lon + (max_lon - min_lon) * j / 9
                    # Map to pixel coordinates (simplified - assumes image covers full footprint)
                    pixel = j * 1000  # Simplified mapping
                    line = i * 1000   # Simplified mapping
                    grid_points.append((pixel, line, lon, lat))
            
            self.grid_points = grid_points
            logger.info(f"✅ Created {len(grid_points)} coordinate points from KML footprint")
            
            logger.info(f"📊 Coordinate bounds:")
            logger.info(f"   ├── Longitude range: {min_lon:.6f} - {max_lon:.6f}")
            logger.info(f"   └── Latitude range: {min_lat:.6f} - {max_lat:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error extracting from KML: {e}")
            return False
    
    def _extract_from_annotation_xml(self) -> bool:
        """Extract geolocation grid points from SAFE annotation files using proven approach."""
        annotation_dir = os.path.join(self.safe_file_path, "annotation")
        if not os.path.exists(annotation_dir):
            logger.error(f"❌ Annotation directory not found: {annotation_dir}")
            return False
            
        # Look for VV annotation file (primary) - same logic as functions_post_snap.py
        vv_xml = [f for f in os.listdir(annotation_dir) if 'vv' in f.lower() and f.endswith('.xml')]
        if not vv_xml:
            logger.error(f"❌ No VV annotation XML found in: {annotation_dir}")
            logger.error(f"   ├── Available files: {os.listdir(annotation_dir) if os.path.exists(annotation_dir) else 'Directory does not exist'}")
            logger.error(f"   └── Expected pattern: *vv*.xml")
            return False
        
        logger.info(f"✅ Found VV annotation file: {vv_xml[0]}")
        annotation_file = os.path.join(annotation_dir, vv_xml[0])
        logger.info(f"📄 Processing annotation file: {os.path.basename(annotation_file)}")
        
        try:
            # Parse XML
            tree = ET.parse(annotation_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"❌ XML parsing error in annotation file: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error reading annotation file: {e}")
            return False
        
        # Extract geolocation grid points - exact same logic as functions_post_snap.py
        pts = []
        for elem in root.iter():
            if elem.tag.endswith('geolocationGridPoint'):
                line = pixel = lat = lon = None
                try:
                    for child in elem:
                        if child.tag.endswith('line') and child.text: 
                            line = int(child.text)
                        elif child.tag.endswith('pixel') and child.text: 
                            pixel = int(child.text)
                        elif child.tag.endswith('latitude') and child.text: 
                            lat = float(child.text)
                        elif child.tag.endswith('longitude') and child.text: 
                            lon = float(child.text)
                    if line is not None and pixel is not None and lat is not None and lon is not None:
                        pts.append((pixel, line, lon, lat))
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Skipping invalid geolocation point: {e}")
                    continue
        
        logger.info(f"📊 Found {len(pts)} geolocation grid points in raw SAR data")
        
        if len(pts) < 2:
            logger.error("❌ Insufficient geolocation grid points found")
            return False
            
        # Sort points and log bounds - same as functions_post_snap.py
        pts.sort(key=lambda x: (x[0], x[1]))
        tl, br = pts[0], pts[-1]
        
        logger.info(f"📍 Coordinate bounds:")
        logger.info(f"   ├── Top-left: pixel({tl[0]}, {tl[1]}) → lat/lon({tl[3]:.6f}, {tl[2]:.6f})")
        logger.info(f"   └── Bottom-right: pixel({br[0]}, {br[1]}) → lat/lon({br[3]:.6f}, {br[2]:.6f})")
        
        self.grid_points = pts
        logger.info(f"✅ Extracted {len(pts)} geolocation grid points from annotation XML")
        
        return True
    
    def build_interpolator(self) -> bool:
        """
        Build interpolation function for coordinate transformation.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.grid_points:
            logger.error("❌ No grid points available. Call extract_grid_points() first.")
            return False
            
        logger.info("🔧 Building coordinate interpolation system...")
        
        try:
            # Prepare data for interpolation
            pixels = np.array([p[0] for p in self.grid_points])
            lines = np.array([p[1] for p in self.grid_points])
            lons = np.array([p[2] for p in self.grid_points])
            lats = np.array([p[3] for p in self.grid_points])
            
            # Create interpolation points (pixel, line) -> (lon, lat)
            points = np.column_stack((pixels, lines))
            
            # Store bounds for validation
            self.image_bounds = {
                'min_pixel': pixels.min(),
                'max_pixel': pixels.max(),
                'min_line': lines.min(),
                'max_line': lines.max(),
                'min_lon': lons.min(),
                'max_lon': lons.max(),
                'min_lat': lats.min(),
                'max_lat': lats.max()
            }
            
            # Create interpolation functions for lon and lat separately
            self.lon_interpolator = lambda p, l: griddata(
                points, lons, (p, l), method='linear', fill_value=np.nan
            )
            self.lat_interpolator = lambda p, l: griddata(
                points, lats, (p, l), method='linear', fill_value=np.nan
            )
            
            # Mark interpolator as built
            self.interpolator = True
            
            logger.info("✅ Coordinate interpolation system built successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building interpolator: {e}")
            return False
    
    def pixel_to_geo(self, pixel: float, line: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel: Pixel coordinate (x)
            line: Line coordinate (y)
            
        Returns:
            Tuple of (longitude, latitude)
        """
        if self.interpolator is None:
            raise RuntimeError("Interpolator not built. Call build_interpolator() first.")
            
        # Validate input coordinates
        if (pixel < self.image_bounds['min_pixel'] or 
            pixel > self.image_bounds['max_pixel'] or
            line < self.image_bounds['min_line'] or 
            line > self.image_bounds['max_line']):
            logger.warning(f"⚠️ Coordinates ({pixel}, {line}) outside SAFE bounds")
            
        # Interpolate coordinates
        lon = self.lon_interpolator(pixel, line)
        lat = self.lat_interpolator(pixel, line)
        
        # Check for NaN values (outside interpolation range)
        if np.isnan(lon) or np.isnan(lat):
            logger.warning(f"⚠️ Interpolation failed for ({pixel}, {line}) - using nearest neighbor")
            # Fallback to nearest neighbor
            lon = griddata(
                np.column_stack(([p[0] for p in self.grid_points], [p[1] for p in self.grid_points])),
                [p[2] for p in self.grid_points],
                (pixel, line), method='nearest'
            )
            lat = griddata(
                np.column_stack(([p[0] for p in self.grid_points], [p[1] for p in self.grid_points])),
                [p[3] for p in self.grid_points],
                (pixel, line), method='nearest'
            )
            
        return float(lon), float(lat)
    
    def validate_coordinates(self, test_points: List[Tuple[float, float]]) -> Dict:
        """
        Validate coordinate transformation with test points.
        
        Args:
            test_points: List of (pixel, line) test points
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"🧪 Validating coordinate transformation with {len(test_points)} test points...")
        
        results = {
            'valid_points': 0,
            'invalid_points': 0,
            'coordinate_ranges': {
                'lon_range': [float('inf'), float('-inf')],
                'lat_range': [float('inf'), float('-inf')]
            },
            'test_results': []
        }
        
        for pixel, line in test_points:
            try:
                lon, lat = self.pixel_to_geo(pixel, line)
                
                # Update ranges
                results['coordinate_ranges']['lon_range'][0] = min(results['coordinate_ranges']['lon_range'][0], lon)
                results['coordinate_ranges']['lon_range'][1] = max(results['coordinate_ranges']['lon_range'][1], lon)
                results['coordinate_ranges']['lat_range'][0] = min(results['coordinate_ranges']['lat_range'][0], lat)
                results['coordinate_ranges']['lat_range'][1] = max(results['coordinate_ranges']['lat_range'][1], lat)
                
                results['test_results'].append({
                    'pixel': pixel,
                    'line': line,
                    'lon': lon,
                    'lat': lat,
                    'valid': True
                })
                results['valid_points'] += 1
                
            except Exception as e:
                results['test_results'].append({
                    'pixel': pixel,
                    'line': line,
                    'error': str(e),
                    'valid': False
                })
                results['invalid_points'] += 1
                
        logger.info(f"✅ Validation complete: {results['valid_points']} valid, {results['invalid_points']} invalid")
        logger.info(f"📊 Coordinate ranges:")
        logger.info(f"   ├── Longitude: {results['coordinate_ranges']['lon_range'][0]:.6f} - {results['coordinate_ranges']['lon_range'][1]:.6f}")
        logger.info(f"   └── Latitude: {results['coordinate_ranges']['lat_range'][0]:.6f} - {results['coordinate_ranges']['lat_range'][1]:.6f}")
        
        return results


def create_safe_coordinate_system(safe_file_path: str) -> Optional[SAFECoordinateSystem]:
    """
    Create and initialize a SAFE coordinate system.
    
    Args:
        safe_file_path: Path to the SAFE directory
        
    Returns:
        SAFECoordinateSystem instance or None if failed
    """
    try:
        coord_system = SAFECoordinateSystem(safe_file_path)
        
        # Extract grid points
        if not coord_system.extract_grid_points():
            logger.error("❌ Failed to extract grid points from SAFE file")
            return None
            
        # Build interpolator
        if not coord_system.build_interpolator():
            logger.error("❌ Failed to build coordinate interpolator")
            return None
            
        logger.info("✅ SAFE coordinate system initialized successfully")
        return coord_system
        
    except Exception as e:
        logger.error(f"❌ Error creating SAFE coordinate system: {e}")
        return None


if __name__ == "__main__":
    # Test the coordinate system
    logging.basicConfig(level=logging.INFO)
    
    safe_path = "../data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    
    coord_system = create_safe_coordinate_system(safe_path)
    if coord_system:
        # Test with some sample points
        test_points = [
            (1000, 1000),
            (5000, 5000),
            (10000, 10000)
        ]
        
        validation = coord_system.validate_coordinates(test_points)
        print(f"Validation results: {validation}")
