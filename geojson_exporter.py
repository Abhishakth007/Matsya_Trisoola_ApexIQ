#!/usr/bin/env python3
"""
GeoJSON Export Module

Converts vessel detection predictions to GeoJSON format for
geographic visualization and analysis tools.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class GeoJSONExporter:
    """
    Exports vessel detections to GeoJSON format with full attribute support.
    """
    
    def __init__(self):
        """Initialize the GeoJSON exporter."""
        self.feature_collection = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {}
        }
    
    def export_predictions_to_geojson(self, 
                                    predictions_df: pd.DataFrame,
                                    output_path: str,
                                    include_bboxes: bool = True,
                                    scene_metadata: Optional[Dict] = None) -> str:
        """
        Export predictions DataFrame to GeoJSON format.
        
        Args:
            predictions_df: Predictions DataFrame
            output_path: Output GeoJSON file path
            include_bboxes: Whether to include bounding box geometries
            scene_metadata: Optional scene metadata to include
            
        Returns:
            Path to created GeoJSON file
        """
        logger.info(f"📄 Exporting {len(predictions_df)} predictions to GeoJSON...")
        
        # Initialize feature collection
        features = []
        
        # Process each detection
        for _, detection in predictions_df.iterrows():
            feature = self._create_detection_feature(detection, include_bboxes)
            features.append(feature)
        
        # Create feature collection
        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": self._create_metadata(predictions_df, scene_metadata)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"✅ GeoJSON exported to: {output_path}")
        return output_path
    
    def _create_detection_feature(self, detection: pd.Series, include_bboxes: bool) -> Dict:
        """Create a GeoJSON feature for a single detection."""
        
        # Create point geometry for vessel center
        point_geometry = {
            "type": "Point",
            "coordinates": [float(detection['lon']), float(detection['lat'])]
        }
        
        # Create properties with all available attributes
        properties = {
            "detect_id": detection['detect_id'],
            "confidence": float(detection['confidence']),
            "timestamp": detection.get('timestamp', datetime.now().isoformat()),
        }
        
        # Add vessel attributes if available
        vessel_attrs = {
            'vessel_length_m': 'length_meters',
            'vessel_width_m': 'width_meters',
            'vessel_speed_k': 'speed_knots',
            'vessel_type': 'vessel_type',
            'heading_degrees': 'heading_degrees',
            'is_fishing_vessel': 'is_fishing_vessel'
        }
        
        for csv_col, geojson_prop in vessel_attrs.items():
            if csv_col in detection and pd.notna(detection[csv_col]):
                properties[geojson_prop] = self._convert_value(detection[csv_col])
        
        # Add pixel coordinates
        if 'preprocess_row' in detection and 'preprocess_column' in detection:
            properties['pixel_coordinates'] = {
                'x': float(detection['preprocess_column']),
                'y': float(detection['preprocess_row'])
            }
        
        # Add bounding box if requested and available
        if include_bboxes and all(col in detection for col in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']):
            properties['bounding_box'] = {
                'x1': int(detection['bbox_x1']),
                'y1': int(detection['bbox_y1']),
                'x2': int(detection['bbox_x2']),
                'y2': int(detection['bbox_y2']),
                'width': int(detection['bbox_x2'] - detection['bbox_x1']),
                'height': int(detection['bbox_y2'] - detection['bbox_y1'])
            }
        
        # Add heading bucket probabilities if available
        heading_buckets = {}
        for i in range(16):
            bucket_col = f'heading_bucket_{i}'
            if bucket_col in detection and pd.notna(detection[bucket_col]):
                heading_buckets[f'bucket_{i}'] = float(detection[bucket_col])
        
        if heading_buckets:
            properties['heading_probabilities'] = heading_buckets
        
        # Create feature
        feature = {
            "type": "Feature",
            "geometry": point_geometry,
            "properties": properties
        }
        
        return feature
    
    def _create_metadata(self, predictions_df: pd.DataFrame, scene_metadata: Optional[Dict]) -> Dict:
        """Create metadata for the GeoJSON file."""
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "total_detections": len(predictions_df),
            "confidence_range": {
                "min": float(predictions_df['confidence'].min()),
                "max": float(predictions_df['confidence'].max()),
                "mean": float(predictions_df['confidence'].mean())
            },
            "spatial_bounds": {
                "min_lat": float(predictions_df['lat'].min()),
                "max_lat": float(predictions_df['lat'].max()),
                "min_lon": float(predictions_df['lon'].min()),
                "max_lon": float(predictions_df['lon'].max())
            }
        }
        
        # Add vessel statistics if available
        if 'vessel_length_m' in predictions_df.columns:
            lengths = predictions_df['vessel_length_m'].dropna()
            if len(lengths) > 0:
                metadata['vessel_statistics'] = {
                    "length_range_m": {
                        "min": float(lengths.min()),
                        "max": float(lengths.max()),
                        "mean": float(lengths.mean())
                    }
                }
        
        # Add vessel type distribution
        if 'vessel_type' in predictions_df.columns:
            type_counts = predictions_df['vessel_type'].value_counts().to_dict()
            metadata['vessel_type_distribution'] = type_counts
        
        # Include scene metadata if provided
        if scene_metadata:
            metadata['scene_info'] = scene_metadata
        
        return metadata
    
    def _convert_value(self, value: Any) -> Any:
        """Convert pandas values to JSON-serializable types."""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return value
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def export_predictions_to_geojson(predictions_csv_path: str,
                                output_geojson_path: str,
                                scene_metadata: Optional[Dict] = None) -> str:
    """
    Standalone function to export predictions CSV to GeoJSON.
    
    Args:
        predictions_csv_path: Input predictions CSV file
        output_geojson_path: Output GeoJSON file path
        scene_metadata: Optional scene metadata
        
    Returns:
        Path to created GeoJSON file
    """
    logger.info(f"📄 Converting {predictions_csv_path} to GeoJSON...")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_csv_path)
    
    # Create exporter
    exporter = GeoJSONExporter()
    
    # Export to GeoJSON
    output_path = exporter.export_predictions_to_geojson(
        predictions_df, 
        output_geojson_path,
        include_bboxes=True,
        scene_metadata=scene_metadata
    )
    
    return output_path

def create_correlation_geojson(predictions_df: pd.DataFrame,
                             correlation_results_df: pd.DataFrame,
                             output_path: str) -> str:
    """
    Create GeoJSON with correlation status included.
    
    Args:
        predictions_df: Original predictions
        correlation_results_df: AIS correlation results
        output_path: Output GeoJSON path
        
    Returns:
        Path to created GeoJSON file
    """
    logger.info("📄 Creating correlation-enhanced GeoJSON...")
    
    # Merge predictions with correlation results
    merged_df = predictions_df.merge(
        correlation_results_df[['detect_id', 'matched_mmsi', 'match_confidence', 'is_dark_ship']],
        on='detect_id',
        how='left'
    )
    
    # Create exporter
    exporter = GeoJSONExporter()
    
    # Modify the feature creation to include correlation info
    features = []
    for _, detection in merged_df.iterrows():
        feature = exporter._create_detection_feature(detection, include_bboxes=True)
        
        # Add correlation properties
        if pd.notna(detection.get('matched_mmsi')):
            feature['properties']['correlation'] = {
                'matched_mmsi': str(detection['matched_mmsi']),
                'match_confidence': float(detection['match_confidence']),
                'is_dark_ship': bool(detection.get('is_dark_ship', False)),
                'status': 'correlated'
            }
        else:
            feature['properties']['correlation'] = {
                'matched_mmsi': None,
                'match_confidence': 0.0,
                'is_dark_ship': True,
                'status': 'dark_ship'
            }
        
        features.append(feature)
    
    # Create feature collection
    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": exporter._create_metadata(merged_df, {
            "correlation_summary": {
                "total_detections": len(merged_df),
                "correlated_vessels": len(merged_df[merged_df['matched_mmsi'].notna()]),
                "dark_ships": len(merged_df[merged_df['matched_mmsi'].isna()])
            }
        })
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2, default=exporter._json_serializer)
    
    logger.info(f"✅ Correlation GeoJSON saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test the GeoJSON exporter
    logging.basicConfig(level=logging.INFO)
    
    # Test basic export
    export_predictions_to_geojson(
        "professor/outputs/predictions.csv",
        "professor/outputs/predictions.geojson"
    )
