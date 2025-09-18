#!/usr/bin/env python3
"""
Prediction Output Generator for Vessel Detection System

This module generates standardized prediction outputs in:
1. CSV format with vessel detection details
2. GeoJSON format for spatial visualization
3. JSON format for API responses
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PredictionOutputGenerator:
    """Generates standardized prediction outputs for vessel detection."""
    
    def __init__(self, output_dir: str = "predictions"):
        """
        Initialize the prediction output generator.
        
        Args:
            output_dir: Directory to save prediction outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Standard vessel detection output columns
        self.csv_columns = [
            'detection_id',
            'timestamp',
            'vessel_id',
            'vessel_name',
            'mmsi',
            'vessel_type',
            'confidence_score',
            'bbox_x1',
            'bbox_y1', 
            'bbox_x2',
            'bbox_y2',
            'center_lat',
            'center_lon',
            'length_m',
            'width_m',
            'heading_deg',
            'speed_knots',
            'detection_method',
            'processing_time_ms',
            'aoi_bounds',
            'metadata'
        ]
        
        # GeoJSON feature properties
        self.geojson_properties = [
            'detection_id',
            'vessel_id',
            'vessel_name',
            'mmsi',
            'vessel_type',
            'confidence_score',
            'length_m',
            'width_m',
            'heading_deg',
            'speed_knots',
            'detection_method',
            'timestamp'
        ]
    
    def generate_vessel_detections(self, aoi_bounds: Dict, vessel_count: int = 5) -> List[Dict]:
        """
        Generate sample vessel detections for testing.
        
        Args:
            aoi_bounds: AOI boundary coordinates
            vessel_count: Number of vessels to generate
            
        Returns:
            List of vessel detection dictionaries
        """
        detections = []
        
        # Calculate AOI center
        center_lat = (aoi_bounds['min_lat'] + aoi_bounds['max_lat']) / 2
        center_lon = (aoi_bounds['min_lon'] + aoi_bounds['max_lon']) / 2
        
        # Generate random vessel detections
        for i in range(vessel_count):
            # Random position within AOI
            lat = np.random.uniform(aoi_bounds['min_lat'], aoi_bounds['max_lat'])
            lon = np.random.uniform(aoi_bounds['min_lon'], aoi_bounds['max_lon'])
            
            # Random vessel properties
            confidence = np.random.uniform(0.7, 0.99)
            vessel_type = np.random.choice(['cargo', 'tanker', 'fishing', 'passenger', 'military'])
            length = np.random.uniform(50, 300)
            width = length * np.random.uniform(0.1, 0.2)
            heading = np.random.uniform(0, 360)
            speed = np.random.uniform(5, 25)
            
            # Generate bounding box (pixel coordinates)
            bbox_size = np.random.uniform(20, 100)
            bbox_x1 = np.random.uniform(0, 800 - bbox_size)
            bbox_y1 = np.random.uniform(0, 800 - bbox_size)
            bbox_x2 = bbox_x1 + bbox_size
            bbox_y2 = bbox_y1 + bbox_size
            
            detection = {
                'detection_id': f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}",
                'timestamp': datetime.now().isoformat(),
                'vessel_id': f"vessel_{i:03d}",
                'vessel_name': f"Vessel_{i:03d}",
                'mmsi': f"{np.random.randint(100000000, 999999999)}",
                'vessel_type': vessel_type,
                'confidence_score': round(confidence, 3),
                'bbox_x1': int(bbox_x1),
                'bbox_y1': int(bbox_y1),
                'bbox_x2': int(bbox_x2),
                'bbox_y2': int(bbox_y2),
                'center_lat': round(lat, 6),
                'center_lon': round(lon, 6),
                'length_m': round(length, 1),
                'width_m': round(width, 1),
                'heading_deg': round(heading, 1),
                'speed_knots': round(speed, 1),
                'detection_method': 'deep_learning_frcnn',
                'processing_time_ms': np.random.randint(50, 200),
                'aoi_bounds': f"{aoi_bounds['min_lat']:.6f},{aoi_bounds['min_lon']:.6f},{aoi_bounds['max_lat']:.6f},{aoi_bounds['max_lon']:.6f}",
                'metadata': json.dumps({
                    'model_version': 'frcnn_cmp2_v1.0',
                    'input_channels': 6,
                    'detection_threshold': 0.5,
                    'nms_threshold': 0.3
                })
            }
            
            detections.append(detection)
        
        return detections
    
    def save_csv_predictions(self, detections: List[Dict], filename: str = None) -> str:
        """
        Save vessel detections to CSV format.
        
        Args:
            detections: List of vessel detection dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to saved CSV file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"vessel_predictions_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Create DataFrame
        df = pd.DataFrame(detections)
        
        # Ensure all columns exist
        for col in self.csv_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        df = df[self.csv_columns]
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"✅ CSV predictions saved to: {filepath}")
        
        return str(filepath)
    
    def save_geojson_predictions(self, detections: List[Dict], filename: str = None) -> str:
        """
        Save vessel detections to GeoJSON format.
        
        Args:
            detections: List of vessel detection dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to saved GeoJSON file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"vessel_predictions_{timestamp}.geojson"
        
        filepath = self.output_dir / filename
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_detections": len(detections),
                "detection_method": "deep_learning_frcnn",
                "coordinate_system": "EPSG:4326"
            }
        }
        
        # Convert each detection to GeoJSON feature
        for detection in detections:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [detection['center_lon'], detection['center_lat']]
                },
                "properties": {}
            }
            
            # Add properties
            for prop in self.geojson_properties:
                if prop in detection:
                    feature["properties"][prop] = detection[prop]
            
            # Add bounding box as polygon
            bbox_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [detection['bbox_x1'], detection['bbox_y1']],
                        [detection['bbox_x2'], detection['bbox_y1']],
                        [detection['bbox_x2'], detection['bbox_y2']],
                        [detection['bbox_x1'], detection['bbox_y2']],
                        [detection['bbox_x1'], detection['bbox_y1']]
                    ]]
                },
                "properties": {
                    "detection_id": detection['detection_id'],
                    "confidence_score": detection['confidence_score'],
                    "vessel_type": detection['vessel_type']
                }
            }
            
            geojson["features"].extend([feature, bbox_feature])
        
        # Save to GeoJSON
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"✅ GeoJSON predictions saved to: {filepath}")
        
        return str(filepath)
    
    def save_json_predictions(self, detections: List[Dict], filename: str = None) -> str:
        """
        Save vessel detections to JSON format for API responses.
        
        Args:
            detections: List of vessel detection dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to saved JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"vessel_predictions_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Create API response structure
        api_response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(detections),
            "detections": detections,
            "metadata": {
                "model_version": "frcnn_cmp2_v1.0",
                "processing_time_ms": sum(d['processing_time_ms'] for d in detections),
                "detection_method": "deep_learning_frcnn",
                "output_formats": ["csv", "geojson", "json"]
            }
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(api_response, f, indent=2)
        
        logger.info(f"✅ JSON predictions saved to: {filepath}")
        
        return str(filepath)
    
    def generate_all_outputs(self, aoi_bounds: Dict, vessel_count: int = 5) -> Dict[str, str]:
        """
        Generate all output formats for vessel detections.
        
        Args:
            aoi_bounds: AOI boundary coordinates
            vessel_count: Number of vessels to generate
            
        Returns:
            Dictionary with paths to all output files
        """
        logger.info(f"🚀 Generating {vessel_count} vessel detections for AOI")
        
        # Generate detections
        detections = self.generate_vessel_detections(aoi_bounds, vessel_count)
        
        # Save in all formats
        outputs = {}
        
        outputs['csv'] = self.save_csv_predictions(detections)
        outputs['geojson'] = self.save_geojson_predictions(detections)
        outputs['json'] = self.save_json_predictions(detections)
        
        logger.info(f"✅ All output formats generated successfully")
        logger.info(f"  CSV: {outputs['csv']}")
        logger.info(f"  GeoJSON: {outputs['geojson']}")
        logger.info(f"  JSON: {outputs['json']}")
        
        return outputs
    
    def validate_output_formats(self) -> Dict[str, bool]:
        """
        Validate that all output formats are properly generated.
        
        Returns:
            Dictionary with validation results for each format
        """
        validation_results = {}
        
        # Test CSV format
        try:
            test_detections = self.generate_vessel_detections({
                'min_lat': 36.9, 'max_lat': 37.0,
                'min_lon': -76.1, 'max_lon': -76.0
            }, 3)
            
            csv_path = self.save_csv_predictions(test_detections, "test_validation.csv")
            df = pd.read_csv(csv_path)
            
            validation_results['csv'] = (
                len(df) == 3 and 
                all(col in df.columns for col in self.csv_columns)
            )
            
            # Cleanup test file
            os.remove(csv_path)
            
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            validation_results['csv'] = False
        
        # Test GeoJSON format
        try:
            geojson_path = self.save_geojson_predictions(test_detections, "test_validation.geojson")
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            validation_results['geojson'] = (
                geojson_data['type'] == 'FeatureCollection' and
                len(geojson_data['features']) == 6  # 3 points + 3 bboxes
            )
            
            # Cleanup test file
            os.remove(geojson_path)
            
        except Exception as e:
            logger.error(f"GeoJSON validation failed: {e}")
            validation_results['geojson'] = False
        
        # Test JSON format
        try:
            json_path = self.save_json_predictions(test_detections, "test_validation.json")
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            validation_results['json'] = (
                json_data['status'] == 'success' and
                len(json_data['detections']) == 3
            )
            
            # Cleanup test file
            os.remove(json_path)
            
        except Exception as e:
            logger.error(f"JSON validation failed: {e}")
            validation_results['json'] = False
        
        return validation_results

def get_prediction_generator(output_dir: str = "predictions") -> PredictionOutputGenerator:
    """Get a prediction output generator instance."""
    return PredictionOutputGenerator(output_dir)

if __name__ == "__main__":
    # Test the prediction output generator
    logging.basicConfig(level=logging.INFO)
    
    generator = PredictionOutputGenerator()
    
    # Test AOI bounds
    test_aoi = {
        'min_lat': 36.947311,
        'max_lat': 36.965329,
        'min_lon': -76.027454,
        'max_lon': -76.004906
    }
    
    # Generate test outputs
    outputs = generator.generate_all_outputs(test_aoi, 5)
    
    # Validate outputs
    validation = generator.validate_output_formats()
    print(f"\nValidation Results: {validation}")
