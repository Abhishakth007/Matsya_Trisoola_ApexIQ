"""
Utility functions for exporting vessel detections in various formats.
"""
import os
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import tempfile
import shutil

# For shapefile export
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    SHAPEFILE_SUPPORT = True
except ImportError:
    SHAPEFILE_SUPPORT = False
    logging.warning("Shapefile export requires geopandas and shapely. Install with: pip install geopandas shapely")

logger = logging.getLogger(__name__)

def create_bounding_box(lat: float, lon: float, length: Optional[float] = None, 
                        heading: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """
    Create a bounding box around a vessel detection point.
    
    Args:
        lat: Latitude of the vessel center point
        lon: Longitude of the vessel center point
        length: Length of the vessel in meters (default: 50m if not provided)
        heading: Heading of the vessel in degrees (0 = North, 90 = East)
                 If None, creates a square box
    
    Returns:
        Tuple of (lons, lats) for the bounding box corners
    """
    # Use default length if not provided or invalid
    if length is None or not isinstance(length, (int, float)) or length <= 0:
        length = 50.0  # Default length in meters
    
    # FIXED: Use proper coordinate projection instead of crude approximation
    try:
        import pyproj
        from pyproj import Transformer
        
        # Create transformer from WGS84 to a local projected coordinate system
        # This provides accurate distance calculations
        transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (lat/lon)
            "EPSG:3857",  # Web Mercator (good for distance calculations)
            always_xy=True
        )
        
        # Transform center point to projected coordinates
        center_x, center_y = transformer.transform(lon, lat)
        
        # Calculate offsets in meters in projected space
        width = length * 0.6  # Assuming width is 60% of length
    
        # Create bounding box in projected coordinates
        half_length = length / 2
        half_width = width / 2
        
        if heading is None:
            # Create a simple box without rotation
            min_x = center_x - half_length
            max_x = center_x + half_length
            min_y = center_y - half_width
            max_y = center_y + half_width
            
            # Transform corners back to WGS84
            corners_lon, corners_lat = transformer.transform(
                [min_x, max_x, max_x, min_x, min_x],
                [min_y, min_y, max_y, max_y, min_y],
                direction="INVERSE"
            )
            
            lons = list(corners_lon)
            lats = list(corners_lat)
        else:
            # Create rotated box - this is more complex and requires rotation in projected space
            # For now, fall back to the original method but with better validation
            logger.warning("Rotated bounding boxes not yet implemented with proper projection")
            raise NotImplementedError("Rotated bounding boxes require additional implementation")
            
    except (ImportError, Exception) as e:
        # Fallback to improved approximation if pyproj is not available
        logger.warning(f"Using fallback coordinate conversion: {e}")
        
        # Improved approximation with better constants
        # Earth's radius varies by latitude, so we use a more accurate approximation
        earth_radius_m = 6371000  # Earth's radius in meters
        
        # Convert to radians for more accurate calculations
        lat_rad = np.radians(lat)
        
        # Calculate offsets in degrees with better precision
        lat_offset = length / (earth_radius_m * np.pi / 180)  # More accurate than 111000
        lon_offset = length / (earth_radius_m * np.cos(lat_rad) * np.pi / 180)
        
        width = length * 0.6
        width_lat_offset = width / (earth_radius_m * np.pi / 180)
        width_lon_offset = width / (earth_radius_m * np.cos(lat_rad) * np.pi / 180)
        
        if heading is None:
            # Create a simple box without rotation
            lats = [lat - lat_offset/2, lat - lat_offset/2, lat + lat_offset/2, lat + lat_offset/2, lat - lat_offset/2]
            lons = [lon - lon_offset/2, lon + lon_offset/2, lon + lon_offset/2, lon - lon_offset/2, lon - lon_offset/2]
        else:
            # Create a rotated box based on heading
            # Convert heading to radians (0 = North, 90 = East)
            heading_rad = np.radians(90 - heading)  # Adjust for GIS convention
            
            # Calculate offsets for each corner
            dx = np.array([width_lon_offset/2, length/2])
            dy = np.array([width_lat_offset/2, length/2])
            
            # Rotation matrix
            rot = np.array([
                [np.cos(heading_rad), -np.sin(heading_rad)],
                [np.sin(heading_rad), np.cos(heading_rad)]
            ])
            
            # Calculate corner offsets
            corners = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    offset = rot @ np.array([i * dx[0], j * dy[1]])
                    corners.append((lon + offset[0], lat + offset[1]))
            
            # Add the first corner again to close the polygon
            corners.append(corners[0])
            
            # Extract lons and lats
            lons, lats = zip(*corners)
    
    return list(lons), list(lats)

def predictions_to_geojson(predictions_df: pd.DataFrame, output_path: str) -> None:
    """
    Convert vessel detection predictions to GeoJSON format.
    
    Args:
        predictions_df: DataFrame containing vessel detections
        output_path: Path to save the GeoJSON file
    """
    features = []
    
    for _, row in predictions_df.iterrows():
        try:
            # Check if required columns exist
            if 'lat' not in row or 'lon' not in row:
                logger.warning(f"Skipping row without lat/lon: {row.name}")
                continue
                
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                logger.warning(f"Skipping row with NaN lat/lon: {row.name}")
                continue
            
            # FIXED: Standardized column name mapping for consistent exports
            # Map pipeline output columns to export format expectations
            column_mapping = {
                # Confidence/Score mapping
                'score': 'confidence',
                'confidence': 'confidence',
                
                # Length mapping
                'vessel_length_m': 'length',
                'length': 'length',
                
                # Width mapping
                'vessel_width_m': 'width',
                'width': 'width',
                
                # Speed mapping
                'vessel_speed_k': 'speed',
                'speed': 'speed',
                
                # Heading mapping
                'heading': 'heading',
                'orientation': 'heading',
                
                # Vessel type mapping
                'is_fishing_vessel': 'vessel_class',
                'vessel_class': 'vessel_class',
                'vessel_type': 'vessel_class'
            }
            
            # Get vessel length with proper column mapping
            length = None
            for length_col in ['vessel_length_m', 'length']:
                if length_col in row and pd.notna(row[length_col]):
                    try:
                        length = float(row[length_col])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get heading with proper column mapping
            heading = None
            for heading_col in ['heading', 'orientation']:
                if heading_col in row and pd.notna(row[heading_col]):
                    try:
                        heading = float(row[heading_col])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get vessel class with proper mapping
            vessel_class = None
            for class_col in ['is_fishing_vessel', 'vessel_class', 'vessel_type']:
                if class_col in row and pd.notna(row[class_col]):
                    try:
                        vessel_class = str(row[class_col])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Create bounding box
            lons, lats = create_bounding_box(
                float(row['lat']), 
                float(row['lon']), 
                length,
                heading
            )
            
            # Create properties dictionary with available fields
            properties = {}
            
            # Required fields with fallbacks
            properties["detect_id"] = str(row.get('detect_id', row.name))
            properties["scene_id"] = str(row.get('scene_id', ''))
            
            # Optional fields with type conversion
            if 'confidence' in row and pd.notna(row['confidence']):
                properties["confidence"] = float(row['confidence'])
            elif 'score' in row and pd.notna(row['score']):
                properties["confidence"] = float(row['score'])
            else:
                properties["confidence"] = 0.0
                
            # Add length if available
            if length is not None:
                properties["length"] = float(length)
                
            # Add other optional fields
            if 'vessel_class' in row and pd.notna(row['vessel_class']):
                properties["vessel_class"] = str(row['vessel_class'])
            else:
                properties["vessel_class"] = 'unknown'
                
            # Add vessel type indicators
            if 'is_vessel' in row and pd.notna(row['is_vessel']):
                properties["is_vessel"] = int(row['is_vessel'])
            else:
                properties["is_vessel"] = 1
                
            if 'is_fishing' in row and pd.notna(row['is_fishing']):
                properties["is_fishing"] = int(row['is_fishing'])
            elif 'is_fishing_vessel' in row and pd.notna(row['is_fishing_vessel']):
                try:
                    properties["is_fishing"] = 1 if float(row['is_fishing_vessel']) > 0.5 else 0
                except (ValueError, TypeError):
                    properties["is_fishing"] = 0
            else:
                properties["is_fishing"] = 0
                
            # Add timestamp if available
            if 'timestamp' in row and pd.notna(row['timestamp']):
                properties["timestamp"] = str(row['timestamp'])
            else:
                properties["timestamp"] = ''
                
            # Add distance from shore if available
            if 'distance_from_shore_km' in row and pd.notna(row['distance_from_shore_km']):
                try:
                    properties["distance_from_shore_km"] = float(row['distance_from_shore_km'])
                except (ValueError, TypeError):
                    properties["distance_from_shore_km"] = 0.0
            else:
                properties["distance_from_shore_km"] = 0.0
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(zip(lons, lats))]
                },
                "properties": properties
            }
            
            features.append(feature)
        except Exception as e:
            logger.error(f"Error creating GeoJSON feature for row {row.name}: {e}")
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    logger.info(f"Saved GeoJSON with {len(features)} features to {output_path}")

def predictions_to_shapefile(predictions_df: pd.DataFrame, output_path: str) -> None:
    """
    Convert vessel detection predictions to Shapefile format.
    
    Args:
        predictions_df: DataFrame containing vessel detections
        output_path: Path to save the Shapefile (without extension)
    """
    if not SHAPEFILE_SUPPORT:
        logger.error("Shapefile export requires geopandas and shapely. Install with: pip install geopandas shapely")
        return
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = predictions_df.copy()
    
    # Ensure required columns exist
    required_columns = ['lat', 'lon']
    for col in required_columns:
        if col not in df_copy.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return
    
    # Filter out rows with missing lat/lon
    df_copy = df_copy.dropna(subset=['lat', 'lon'])
    
    if len(df_copy) == 0:
        logger.error("No valid rows with lat/lon found in DataFrame")
        return
    
    # Add default values for missing columns
    if 'detect_id' not in df_copy.columns:
        df_copy['detect_id'] = [str(i) for i in range(len(df_copy))]
    
    if 'scene_id' not in df_copy.columns:
        df_copy['scene_id'] = ''
    
    if 'confidence' not in df_copy.columns and 'score' in df_copy.columns:
        df_copy['confidence'] = df_copy['score']
    elif 'confidence' not in df_copy.columns:
        df_copy['confidence'] = 0.5
    
    # Handle length column
    if 'length' not in df_copy.columns and 'vessel_length_m' in df_copy.columns:
        df_copy['length'] = df_copy['vessel_length_m']
    elif 'length' not in df_copy.columns:
        df_copy['length'] = 50.0  # Default length
    
    # Handle vessel class
    if 'vessel_class' not in df_copy.columns:
        df_copy['vessel_class'] = 'unknown'
    
    # Handle vessel type indicators
    if 'is_vessel' not in df_copy.columns:
        df_copy['is_vessel'] = 1
    
    if 'is_fishing' not in df_copy.columns and 'is_fishing_vessel' in df_copy.columns:
        try:
            df_copy['is_fishing'] = df_copy['is_fishing_vessel'].apply(
                lambda x: 1 if pd.notna(x) and float(x) > 0.5 else 0
            )
        except Exception as e:
            logger.error(f"Error converting is_fishing_vessel: {e}")
            df_copy['is_fishing'] = 0
    elif 'is_fishing' not in df_copy.columns:
        df_copy['is_fishing'] = 0
    
    # Rename columns to fit shapefile limitations (10 characters max)
    column_mapping = {
        'detect_id': 'det_id',
        'scene_id': 'scene_id',
        'detect_scene_id': 'det_sc_id',
        'detect_scene_row': 'det_sc_row',
        'detect_scene_column': 'det_sc_col',
        'row': 'row',
        'column': 'column',
        'confidence': 'conf',
        'length': 'length',
        'vessel_class': 'vess_class',
        'is_vessel': 'is_vessel',
        'is_fishing': 'is_fishing',
        'lon': 'lon',
        'lat': 'lat',
        'distance_from_shore_km': 'dist_shore',
        'timestamp': 'timestamp',
        'vessel_length_m': 'vess_len',
        'vessel_width_m': 'vess_width',
        'vessel_speed_k': 'vess_speed',
        'orientation': 'orient',
        'heading': 'heading',
        'preprocess_row': 'prep_row',
        'preprocess_column': 'prep_col'
    }
    
    # Apply column renaming
    df_copy = df_copy.rename(columns={col: column_mapping.get(col, col[:10]) for col in df_copy.columns})
    
    # Create a GeoDataFrame
    geometries = []
    valid_indices = []
    
    for idx, row in df_copy.iterrows():
        try:
            # Get vessel length - try different possible column names
            length = None
            for length_col in ['length', 'vess_len']:
                if length_col in row and pd.notna(row[length_col]):
                    try:
                        length = float(row[length_col])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get heading if available
            heading = None
            for heading_col in ['heading', 'orient']:
                if heading_col in row and pd.notna(row[heading_col]):
                    try:
                        heading = float(row[heading_col])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Create bounding box
            lons, lats = create_bounding_box(
                float(row['lat']), 
                float(row['lon']), 
                length,
                heading
            )
            
            # Create polygon from coordinates
            polygon = Polygon(list(zip(lons, lats)))
            geometries.append(polygon)
            valid_indices.append(idx)
        except Exception as e:
            logger.error(f"Error creating geometry for row {idx}: {e}")
    
    # Keep only rows with valid geometries
    df_copy = df_copy.loc[valid_indices]
    
    # Check for duplicate column names
    if len(df_copy.columns) != len(set(df_copy.columns)):
        # Find duplicated columns
        duplicated_cols = [col for col in df_copy.columns if list(df_copy.columns).count(col) > 1]
        logger.warning(f"Found duplicated column names: {duplicated_cols}")
        
        # Create a new DataFrame with unique column names
        new_columns = []
        seen_columns = set()
        for col in df_copy.columns:
            if col in seen_columns:
                count = 1
                new_col = f"{col}_{count}"
                while new_col in seen_columns:
                    count += 1
                    new_col = f"{col}_{count}"
                new_columns.append(new_col)
                seen_columns.add(new_col)
            else:
                new_columns.append(col)
                seen_columns.add(col)
        
        # Rename columns
        df_copy.columns = new_columns
    
    # Select essential columns only to avoid issues with too many columns
    essential_columns = ['det_id', 'scene_id', 'conf', 'length', 'vess_class', 
                         'is_vessel', 'is_fishing', 'lon', 'lat']
    
    # Keep only columns that exist in the DataFrame
    columns_to_keep = [col for col in essential_columns if col in df_copy.columns]
    
    # Add any additional columns up to a reasonable limit (20 columns max for shapefiles)
    max_columns = 20
    additional_columns = [col for col in df_copy.columns 
                          if col not in columns_to_keep 
                          and col not in ['geometry']]
    
    # Add additional columns up to the limit
    remaining_slots = max_columns - len(columns_to_keep)
    if remaining_slots > 0 and additional_columns:
        columns_to_keep.extend(additional_columns[:remaining_slots])
    
    # Create GeoDataFrame with selected columns
    try:
        gdf = gpd.GeoDataFrame(
            df_copy[columns_to_keep],
            geometry=geometries,
            crs="EPSG:4326"  # WGS84
        )
        
        # Save to shapefile
        gdf.to_file(output_path)
        
        logger.info(f"Saved Shapefile with {len(gdf)} features to {output_path}")
    except Exception as e:
        logger.error(f"Error creating or saving shapefile: {e}")
        raise

def export_detections(predictions_df: pd.DataFrame, output_dir: str, base_filename: str = "vessel_detections") -> None:
    """
    Export vessel detections in multiple formats.
    
    Args:
        predictions_df: DataFrame containing vessel detections
        output_dir: Directory to save the output files
        base_filename: Base name for the output files (without extension)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export as CSV (already done in the pipeline, but included for completeness)
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    predictions_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # Export as GeoJSON
    geojson_path = os.path.join(output_dir, f"{base_filename}.geojson")
    try:
        predictions_to_geojson(predictions_df, geojson_path)
        logger.info(f"Saved GeoJSON to {geojson_path}")
    except Exception as e:
        logger.error(f"Failed to create GeoJSON: {e}")
    
    # Export as Shapefile
    try:
        # Create the shapefile directly in the output directory
        shapefile_base = os.path.join(output_dir, base_filename)
        
        # Create the shapefile
        predictions_to_shapefile(predictions_df, shapefile_base)
        
        # Check if the shapefile directory was created successfully
        shapefile_path = os.path.join(shapefile_base, os.path.basename(shapefile_base) + '.shp')
        if os.path.exists(shapefile_path):
            logger.info(f"Shapefile created successfully at {shapefile_base}")
        else:
            logger.error(f"Failed to create shapefile at {shapefile_base}")
    except Exception as e:
        logger.error(f"Failed to create shapefile: {e}")
        logger.error("Make sure geopandas and shapely are installed: pip install geopandas shapely")