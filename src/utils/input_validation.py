"""
Input validation framework for vessel detection system.
Provides comprehensive validation for all system inputs and data consistency.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation for vessel detection system."""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_image_data(self, img: np.ndarray, expected_channels: int = 6) -> bool:
        """Validate image data consistency.
        
        Args:
            img: Input image array
            expected_channels: Expected number of channels
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check basic array properties
            if not isinstance(img, np.ndarray):
                self.validation_errors.append("Image must be numpy array")
                return False
            
            if len(img.shape) != 3:
                self.validation_errors.append(f"Image must be 3D (C,H,W), got shape {img.shape}")
                return False
            
            # Check channel count
            actual_channels = img.shape[0]
            if actual_channels != expected_channels:
                self.validation_warnings.append(
                    f"Expected {expected_channels} channels, got {actual_channels}"
                )
            
            # Check data type and range
            if img.dtype != np.uint8:
                self.validation_warnings.append(f"Image dtype is {img.dtype}, expected uint8")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(img)):
                self.validation_errors.append("Image contains NaN values")
                return False
            
            if np.any(np.isinf(img)):
                self.validation_errors.append("Image contains infinite values")
                return False
            
            # Check value range
            min_val = np.min(img)
            max_val = np.max(img)
            if min_val < 0 or max_val > 255:
                self.validation_warnings.append(f"Image values outside [0,255] range: [{min_val}, {max_val}]")
            
            logger.info(f"Image validation passed: shape={img.shape}, dtype={img.dtype}, range=[{min_val}, {max_val}]")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Image validation error: {e}")
            return False
    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate geographic coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for NaN or infinite values
            if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
                self.validation_errors.append("Coordinates contain NaN or infinite values")
                return False
            
            # Check latitude range (-90 to 90)
            if lat < -90 or lat > 90:
                self.validation_errors.append(f"Latitude out of range [-90, 90]: {lat}")
                return False
            
            # Check longitude range (-180 to 180)
            if lon < -180 or lon > 180:
                self.validation_errors.append(f"Longitude out of range [-180, 180]: {lon}")
                return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Coordinate validation error: {e}")
            return False
    
    def validate_detection_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate detection DataFrame structure and content.
        
        Args:
            df: Detection DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(df, pd.DataFrame):
                self.validation_errors.append("Detections must be pandas DataFrame")
                return False
            
            # Check required columns
            required_columns = ['lat', 'lon', 'score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.validation_errors.append(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for empty DataFrame
            if len(df) == 0:
                self.validation_warnings.append("Detection DataFrame is empty")
                return True  # Empty is valid
            
            # Validate coordinates for each detection
            invalid_coords = 0
            for idx, row in df.iterrows():
                if not self.validate_coordinates(row['lat'], row['lon']):
                    invalid_coords += 1
            
            if invalid_coords > 0:
                self.validation_errors.append(f"{invalid_coords} detections have invalid coordinates")
                return False
            
            # Validate score values
            if 'score' in df.columns:
                scores = df['score']
                if np.any(scores < 0) or np.any(scores > 1):
                    self.validation_warnings.append("Some scores outside [0,1] range")
            
            logger.info(f"Detection DataFrame validation passed: {len(df)} detections")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Detection DataFrame validation error: {e}")
            return False
    
    def validate_file_paths(self, paths: Dict[str, str]) -> bool:
        """Validate file paths exist and are accessible.
        
        Args:
            paths: Dictionary of path names to file paths
            
        Returns:
            True if all paths valid, False otherwise
        """
        try:
            for name, path in paths.items():
                if not path:
                    self.validation_errors.append(f"{name} path is empty")
                    continue
                
                if not os.path.exists(path):
                    self.validation_errors.append(f"{name} path does not exist: {path}")
                    continue
                
                if not os.access(path, os.R_OK):
                    self.validation_errors.append(f"{name} path not readable: {path}")
                    continue
                
                logger.debug(f"Path validation passed: {name} = {path}")
            
            return len(self.validation_errors) == 0
            
        except Exception as e:
            self.validation_errors.append(f"File path validation error: {e}")
            return False
    
    def validate_model_configuration(self, config: Dict) -> bool:
        """Validate model configuration parameters.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_keys = ['detector_model_dir', 'postprocess_model_dir']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                self.validation_errors.append(f"Missing required config keys: {missing_keys}")
                return False
            
            # Validate model directories exist
            model_paths = {
                'detector': config['detector_model_dir'],
                'postprocessor': config.get('postprocess_model_dir')
            }
            
            if not self.validate_file_paths(model_paths):
                return False
            
            # Validate window size
            window_size = config.get('window_size', 1024)
            if not isinstance(window_size, int) or window_size <= 0:
                self.validation_errors.append(f"Invalid window_size: {window_size}")
                return False
            
            # Validate device
            device = config.get('device', 'cpu')
            if device not in ['cpu', 'cuda', 'auto']:
                self.validation_errors.append(f"Invalid device: {device}")
                return False
            
            logger.info("Model configuration validation passed")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Model configuration validation error: {e}")
            return False
    
    def validate_database_schema(self, conn) -> bool:
        """Validate database schema consistency.
        
        Args:
            conn: Database connection
            
        Returns:
            True if valid, False otherwise
        """
        try:
            cursor = conn.cursor()
            
            # Check required tables exist
            required_tables = ['datasets', 'images', 'windows', 'labels', 'detections']
            for table in required_tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if not cursor.fetchone():
                    self.validation_errors.append(f"Required table missing: {table}")
                    return False
            
            # Check table schemas
            table_schemas = {
                'datasets': ['id', 'collection_id', 'name', 'task', 'categories'],
                'images': ['id', 'uuid', 'name', 'format', 'channels', 'width', 'height'],
                'windows': ['id', 'dataset_id', 'image_id', 'row', 'column', 'height', 'width'],
                'labels': ['id', 'window_id', 'row', 'column', 'height', 'width'],
                'detections': ['id', 'scene_id', 'detect_id', 'lat', 'lon', 'score']
            }
            
            for table, required_columns in table_schemas.items():
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                missing_columns = [col for col in required_columns if col not in columns]
                if missing_columns:
                    self.validation_errors.append(f"Table {table} missing columns: {missing_columns}")
                    return False
            
            logger.info("Database schema validation passed")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Database schema validation error: {e}")
            return False
    
    def get_validation_summary(self) -> Dict[str, List[str]]:
        """Get validation summary with errors and warnings.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        return {
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy()
        }
    
    def clear_validation_state(self):
        """Clear validation errors and warnings."""
        self.validation_errors.clear()
        self.validation_warnings.clear()
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors).
        
        Returns:
            True if no validation errors, False otherwise
        """
        return len(self.validation_errors) == 0


def validate_system_inputs(
    img: np.ndarray,
    config: Dict,
    db_conn=None,
    detection_df: Optional[pd.DataFrame] = None
) -> Tuple[bool, Dict[str, List[str]]]:
    """Comprehensive system input validation.
    
    Args:
        img: Input image array
        config: System configuration
        db_conn: Database connection (optional)
        detection_df: Detection DataFrame (optional)
        
    Returns:
        Tuple of (is_valid, validation_summary)
    """
    validator = InputValidator()
    
    # Validate image data
    validator.validate_image_data(img)
    
    # Validate model configuration
    validator.validate_model_configuration(config)
    
    # Validate database if provided
    if db_conn:
        validator.validate_database_schema(db_conn)
    
    # Validate detection DataFrame if provided
    if detection_df is not None:
        validator.validate_detection_dataframe(detection_df)
    
    # Log validation results
    summary = validator.get_validation_summary()
    if summary['errors']:
        logger.error(f"Validation errors: {summary['errors']}")
    if summary['warnings']:
        logger.warning(f"Validation warnings: {summary['warnings']}")
    
    return validator.is_valid(), summary


def validate_export_data(df: pd.DataFrame, export_format: str) -> Tuple[bool, List[str]]:
    """Validate data for export formats.
    
    Args:
        df: DataFrame to export
        export_format: Export format ('geojson', 'shapefile', 'csv')
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        if not isinstance(df, pd.DataFrame):
            errors.append("Data must be pandas DataFrame")
            return False, errors
        
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns based on format
        if export_format == 'geojson':
            required_columns = ['lat', 'lon']
        elif export_format == 'shapefile':
            required_columns = ['lat', 'lon']
        elif export_format == 'csv':
            required_columns = ['lat', 'lon', 'score']
        else:
            errors.append(f"Unsupported export format: {export_format}")
            return False, errors
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns for {export_format}: {missing_columns}")
            return False, errors
        
        # Validate coordinates
        invalid_coords = 0
        for idx, row in df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                invalid_coords += 1
            elif row['lat'] < -90 or row['lat'] > 90 or row['lon'] < -180 or row['lon'] > 180:
                invalid_coords += 1
        
        if invalid_coords > 0:
            errors.append(f"{invalid_coords} rows have invalid coordinates")
            return False, errors
        
        return True, errors
        
    except Exception as e:
        errors.append(f"Export validation error: {e}")
        return False, errors
