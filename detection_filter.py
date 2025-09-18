#!/usr/bin/env python3
"""
Detection Attribute Filter Module

Filters vessel detections based on attribute completeness to ensure
high-quality predictions for correlation and analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DetectionAttributeFilter:
    """
    Filters vessel detections based on attribute completeness and quality.
    
    Ensures only high-quality detections with sufficient attributes
    proceed to correlation and final output.
    """
    
    def __init__(self, min_attributes: int = 3, min_confidence: float = 0.3):
        """
        Initialize the attribute filter.
        
        Args:
            min_attributes: Minimum number of vessel attributes required
            min_confidence: Minimum confidence score required
        """
        self.min_attributes = min_attributes
        self.min_confidence = min_confidence
        
        # Define vessel attribute columns
        self.vessel_attributes = [
            'vessel_length_m',
            'vessel_width_m', 
            'vessel_speed_k',
            'vessel_type',
            'heading_degrees',
            'is_fishing_vessel'
        ]
        
        logger.info(f"🔧 Filter initialized: min_attributes={min_attributes}, min_confidence={min_confidence}")
    
    def filter_predictions(self, predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter predictions based on attribute completeness.
        
        Args:
            predictions_df: Input predictions DataFrame
            
        Returns:
            Tuple of (filtered_df, filter_stats)
        """
        logger.info(f"🔍 Filtering {len(predictions_df)} predictions...")
        
        if len(predictions_df) == 0:
            return predictions_df, {'filtered_count': 0, 'removed_count': 0}
        
        # Create copy for filtering
        filtered_df = predictions_df.copy()
        initial_count = len(filtered_df)
        
        # 1. Filter by confidence threshold
        confidence_mask = filtered_df['confidence'] >= self.min_confidence
        filtered_df = filtered_df[confidence_mask].reset_index(drop=True)
        after_confidence = len(filtered_df)
        
        logger.info(f"📊 After confidence filter (>={self.min_confidence}): {after_confidence}/{initial_count}")
        
        # 2. Filter by attribute completeness
        if len(filtered_df) > 0:
            attribute_counts = self._count_vessel_attributes(filtered_df)
            attribute_mask = attribute_counts >= self.min_attributes
            
            # Store attribute counts for analysis
            filtered_df['attribute_count'] = attribute_counts
            
            # Apply attribute filter
            before_attributes = len(filtered_df)
            filtered_df = filtered_df[attribute_mask].reset_index(drop=True)
            after_attributes = len(filtered_df)
            
            logger.info(f"📊 After attribute filter (>={self.min_attributes} attrs): {after_attributes}/{before_attributes}")
        else:
            after_attributes = 0
        
        # 3. Additional quality filters
        if len(filtered_df) > 0:
            filtered_df = self._apply_quality_filters(filtered_df)
            final_count = len(filtered_df)
            
            logger.info(f"📊 After quality filters: {final_count}/{after_attributes}")
        else:
            final_count = 0
        
        # Generate filter statistics
        filter_stats = {
            'initial_count': initial_count,
            'after_confidence': after_confidence,
            'after_attributes': after_attributes,
            'final_count': final_count,
            'removed_count': initial_count - final_count,
            'removal_rate': (initial_count - final_count) / initial_count * 100 if initial_count > 0 else 0,
            'confidence_removed': initial_count - after_confidence,
            'attribute_removed': after_confidence - after_attributes,
            'quality_removed': after_attributes - final_count
        }
        
        logger.info(f"✅ Filtering complete: {final_count}/{initial_count} predictions kept ({filter_stats['removal_rate']:.1f}% removed)")
        
        return filtered_df, filter_stats
    
    def _count_vessel_attributes(self, df: pd.DataFrame) -> np.ndarray:
        """Count non-null vessel attributes for each detection."""
        attribute_counts = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            count = 0
            for attr in self.vessel_attributes:
                if attr in row and pd.notna(row[attr]):
                    # Additional validation for specific attributes
                    if attr == 'vessel_length_m' and row[attr] > 0:
                        count += 1
                    elif attr == 'vessel_width_m' and row[attr] > 0:
                        count += 1
                    elif attr == 'vessel_speed_k' and row[attr] >= 0:
                        count += 1
                    elif attr == 'vessel_type' and row[attr] in ['fishing', 'cargo', 'other']:
                        count += 1
                    elif attr == 'heading_degrees' and 0 <= row[attr] <= 360:
                        count += 1
                    elif attr == 'is_fishing_vessel' and isinstance(row[attr], (bool, int, float)):
                        count += 1
            
            attribute_counts[i] = count
        
        return attribute_counts
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional quality filters to detections."""
        logger.info("🔧 Applying quality filters...")
        
        # Filter 1: Remove detections with unrealistic vessel dimensions
        if 'vessel_length_m' in df.columns:
            # Remove vessels < 5m or > 500m (unrealistic for maritime detection)
            size_mask = (df['vessel_length_m'] >= 5) & (df['vessel_length_m'] <= 500)
            df = df[size_mask].reset_index(drop=True)
            logger.info(f"📊 After size filter (5-500m): {len(df)} detections")
        
        # Filter 2: Remove detections with unrealistic speeds
        if 'vessel_speed_k' in df.columns:
            # Remove vessels > 50 knots (unrealistic for most vessels)
            speed_mask = df['vessel_speed_k'] <= 50
            df = df[speed_mask].reset_index(drop=True)
            logger.info(f"📊 After speed filter (≤50kn): {len(df)} detections")
        
        # Filter 3: Validate geographic coordinates
        coord_mask = (
            (df['lat'] >= -90) & (df['lat'] <= 90) &
            (df['lon'] >= -180) & (df['lon'] <= 180)
        )
        df = df[coord_mask].reset_index(drop=True)
        logger.info(f"📊 After coordinate validation: {len(df)} detections")
        
        return df
    
    def generate_filter_report(self, filter_stats: Dict) -> str:
        """Generate a detailed filter report."""
        report = f"""
🔍 DETECTION FILTER REPORT
========================

📊 Filter Statistics:
   • Initial detections: {filter_stats['initial_count']}
   • After confidence filter: {filter_stats['after_confidence']} (-{filter_stats['confidence_removed']})
   • After attribute filter: {filter_stats['after_attributes']} (-{filter_stats['attribute_removed']})
   • Final detections: {filter_stats['final_count']} (-{filter_stats['quality_removed']})
   
📈 Overall Performance:
   • Detections kept: {filter_stats['final_count']}/{filter_stats['initial_count']} ({100-filter_stats['removal_rate']:.1f}%)
   • Detections removed: {filter_stats['removed_count']} ({filter_stats['removal_rate']:.1f}%)
   
🎯 Filter Breakdown:
   • Confidence filter: {filter_stats['confidence_removed']} removed
   • Attribute filter: {filter_stats['attribute_removed']} removed  
   • Quality filter: {filter_stats['quality_removed']} removed
        """
        
        return report.strip()

def filter_detections_by_attributes(predictions_csv_path: str, 
                                  output_csv_path: str,
                                  min_attributes: int = 3,
                                  min_confidence: float = 0.3) -> Dict:
    """
    Standalone function to filter detections by attributes.
    
    Args:
        predictions_csv_path: Input predictions CSV
        output_csv_path: Output filtered CSV
        min_attributes: Minimum vessel attributes required
        min_confidence: Minimum confidence required
        
    Returns:
        Dictionary with filter statistics
    """
    logger.info(f"🔧 Filtering detections: {predictions_csv_path}")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_csv_path)
    
    # Create filter
    filter_module = DetectionAttributeFilter(min_attributes, min_confidence)
    
    # Apply filtering
    filtered_df, filter_stats = filter_module.filter_predictions(predictions_df)
    
    # Save filtered results
    filtered_df.to_csv(output_csv_path, index=False)
    logger.info(f"✅ Filtered predictions saved to: {output_csv_path}")
    
    # Generate and print report
    report = filter_module.generate_filter_report(filter_stats)
    print(report)
    
    return filter_stats

if __name__ == "__main__":
    # Test the filter module
    import sys
    
    if len(sys.argv) >= 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        min_attrs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        
        filter_detections_by_attributes(input_csv, output_csv, min_attrs)
    else:
        # Default test
        filter_detections_by_attributes(
            "professor/outputs/predictions.csv",
            "professor/outputs/predictions_filtered.csv",
            min_attributes=3
        )
