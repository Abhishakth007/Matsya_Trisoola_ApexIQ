#!/usr/bin/env python3
"""
Correlation Visualizer Module

Creates visualizations showing AIS correlation status:
- Green boxes: AIS-correlated vessels (with MMSI)
- Red boxes: Dark vessels (no AIS match)
- Enhanced labeling with correlation information
"""

import cv2
import numpy as np
import pandas as pd
import rasterio
import os
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class CorrelationVisualizer:
    """
    Creates visualizations showing vessel detection correlation status with AIS data.
    """
    
    def __init__(self, sar_image_path: str):
        """
        Initialize the correlation visualizer.
        
        Args:
            sar_image_path: Path to the SAR GeoTIFF image
        """
        self.sar_image_path = sar_image_path
        self.sar_image = None
        self.image_height = None
        self.image_width = None
        
        self._load_sar_image()
    
    def _load_sar_image(self):
        """Load and normalize SAR image."""
        logger.info(f"🖼️ Loading SAR image: {self.sar_image_path}")
        
        try:
            with rasterio.open(self.sar_image_path) as src:
                # Read first band for visualization
                self.sar_image = src.read(1)
                self.image_height, self.image_width = self.sar_image.shape
                
                # Normalize to 0-255 range
                p2, p98 = np.percentile(self.sar_image, (2, 98))
                self.sar_image = np.clip(self.sar_image, p2, p98)
                
                if p98 > p2:
                    self.sar_image = ((self.sar_image - p2) / (p98 - p2) * 255).astype(np.uint8)
                else:
                    self.sar_image = np.zeros_like(self.sar_image, dtype=np.uint8)
                
                logger.info(f"✅ SAR image loaded: {self.image_width} × {self.image_height}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load SAR image: {e}")
            raise
    
    def create_correlation_visualization(self,
                                       predictions_df: pd.DataFrame,
                                       correlation_results_df: pd.DataFrame,
                                       output_path: str,
                                       region_bounds: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Create visualization showing correlation status.
        
        Args:
            predictions_df: Original predictions DataFrame
            correlation_results_df: Correlation results DataFrame
            output_path: Output image path
            region_bounds: Optional region bounds (x_min, y_min, x_max, y_max)
            
        Returns:
            Path to created visualization
        """
        logger.info("🎨 Creating correlation visualization...")
        
        # Merge predictions with correlation results
        merged_df = predictions_df.merge(
            correlation_results_df[['detect_id', 'matched_mmsi', 'match_confidence', 'is_dark_ship']],
            on='detect_id',
            how='left'
        )
        
        # Prepare image
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            display_image = self.sar_image[y_min:y_max, x_min:x_max].copy()
            
            # Filter detections to region
            center_x = (merged_df['bbox_x1'] + merged_df['bbox_x2']) / 2
            center_y = (merged_df['bbox_y1'] + merged_df['bbox_y2']) / 2
            
            region_mask = (
                (center_x >= x_min) & (center_x <= x_max) &
                (center_y >= y_min) & (center_y <= y_max)
            )
            
            region_detections = merged_df[region_mask].copy()
            offset_x, offset_y = x_min, y_min
        else:
            # Downsample for full scene to prevent memory issues
            max_size = 4000
            if max(self.sar_image.shape) > max_size:
                scale_factor = max_size / max(self.sar_image.shape)
                new_height = int(self.sar_image.shape[0] * scale_factor)
                new_width = int(self.sar_image.shape[1] * scale_factor)
                
                display_image = cv2.resize(self.sar_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Scale bounding boxes
                merged_df = merged_df.copy()
                for col in ['bbox_x1', 'bbox_x2']:
                    merged_df[col] = merged_df[col] * scale_factor
                for col in ['bbox_y1', 'bbox_y2']:
                    merged_df[col] = merged_df[col] * scale_factor
                
                logger.info(f"🔧 Downsampled image: {self.sar_image.shape} → {display_image.shape}")
            else:
                display_image = self.sar_image.copy()
            
            region_detections = merged_df
            offset_x, offset_y = 0, 0
        
        # Convert to BGR for OpenCV
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes with correlation status
        correlated_count = 0
        dark_ship_count = 0
        
        for _, detection in region_detections.iterrows():
            # Get bounding box coordinates
            x1 = int(detection['bbox_x1'] - offset_x)
            y1 = int(detection['bbox_y1'] - offset_y)
            x2 = int(detection['bbox_x2'] - offset_x)
            y2 = int(detection['bbox_y2'] - offset_y)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, display_image.shape[1]))
            y1 = max(0, min(y1, display_image.shape[0]))
            x2 = max(0, min(x2, display_image.shape[1]))
            y2 = max(0, min(y2, display_image.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Determine correlation status and color
            is_dark = detection.get('is_dark_ship', True)
            matched_mmsi = detection.get('matched_mmsi')
            
            if not is_dark and pd.notna(matched_mmsi) and matched_mmsi != 'UNKNOWN':
                # Correlated vessel - GREEN
                color = (0, 255, 0)
                status = "CORRELATED"
                label = f"MMSI:{matched_mmsi}"
                correlated_count += 1
            else:
                # Dark ship - RED
                color = (0, 0, 255)
                status = "DARK SHIP"
                label = "NO AIS"
                dark_ship_count += 1
            
            # Draw bounding box with thicker lines for visibility
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 3)
            
            # Create detection label
            detect_id = detection['detect_id']
            confidence = detection['confidence']
            full_label = f"ID:{detect_id} ({confidence:.2f})"
            
            if status == "CORRELATED":
                full_label += f"\n{label}"
            else:
                full_label += f"\n{label}"
            
            # Add vessel attributes if available
            if pd.notna(detection.get('vessel_length_m')):
                length = detection['vessel_length_m']
                full_label += f"\n{length:.0f}m"
            
            # Draw label background
            lines = full_label.split('\n')
            line_height = 25
            max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for line in lines])
            
            # Background rectangle
            cv2.rectangle(display_image, (x1, y1-len(lines)*line_height-10), 
                         (x1+max_width+10, y1-5), (0, 0, 0), -1)
            
            # Draw text lines
            for i, line in enumerate(lines):
                text_y = y1 - (len(lines)-i) * line_height + 5
                cv2.putText(display_image, line, (x1+5, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add title and statistics
        title = f"Vessel Correlation Status: {len(region_detections)} total detections"
        stats = f"Correlated: {correlated_count} | Dark Ships: {dark_ship_count}"
        
        # Title background
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        stats_size = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        
        cv2.rectangle(display_image, (10, 10), (max(title_size[0], stats_size[0])+20, 100), (0, 0, 0), -1)
        
        # Title text
        cv2.putText(display_image, title, (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(display_image, stats, (15, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add legend
        legend_y = display_image.shape[0] - 80
        cv2.rectangle(display_image, (10, legend_y-10), (300, display_image.shape[0]-10), (0, 0, 0), -1)
        
        # Correlated legend
        cv2.rectangle(display_image, (20, legend_y), (40, legend_y+20), (0, 255, 0), -1)
        cv2.putText(display_image, "AIS Correlated", (50, legend_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dark ship legend
        cv2.rectangle(display_image, (20, legend_y+30), (40, legend_y+50), (0, 0, 255), -1)
        cv2.putText(display_image, "Dark Ships", (50, legend_y+45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save image
        cv2.imwrite(output_path, display_image)
        
        logger.info(f"✅ Correlation visualization saved: {output_path}")
        logger.info(f"📊 Correlation summary: {correlated_count} correlated, {dark_ship_count} dark ships")
        
        return output_path
    
    def create_split_visualizations(self,
                                  predictions_df: pd.DataFrame,
                                  correlation_results_df: pd.DataFrame,
                                  output_dir: str,
                                  region_bounds: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, str]:
        """
        Create separate visualizations for correlated and dark vessels.
        
        Args:
            predictions_df: Original predictions
            correlation_results_df: Correlation results
            output_dir: Output directory
            region_bounds: Optional region bounds
            
        Returns:
            Dictionary with paths to created visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Merge data
        merged_df = predictions_df.merge(
            correlation_results_df[['detect_id', 'matched_mmsi', 'match_confidence', 'is_dark_ship']],
            on='detect_id',
            how='left'
        )
        
        # Split into correlated and dark vessels
        correlated_vessels = merged_df[
            merged_df['matched_mmsi'].notna() & 
            (merged_df['matched_mmsi'] != 'UNKNOWN') &
            (~merged_df.get('is_dark_ship', True))
        ]
        
        dark_vessels = merged_df[
            merged_df['matched_mmsi'].isna() | 
            (merged_df['matched_mmsi'] == 'UNKNOWN') |
            merged_df.get('is_dark_ship', True)
        ]
        
        created_files = {}
        
        # Create correlated vessels visualization
        if len(correlated_vessels) > 0:
            correlated_path = os.path.join(output_dir, "correlated_vessels.png")
            self._create_single_status_visualization(
                correlated_vessels, correlated_path, "CORRELATED VESSELS", 
                (0, 255, 0), region_bounds
            )
            created_files['correlated'] = correlated_path
        
        # Create dark vessels visualization
        if len(dark_vessels) > 0:
            dark_path = os.path.join(output_dir, "dark_vessels.png")
            self._create_single_status_visualization(
                dark_vessels, dark_path, "DARK SHIPS", 
                (0, 0, 255), region_bounds
            )
            created_files['dark'] = dark_path
        
        # Create combined visualization
        combined_path = os.path.join(output_dir, "correlation_combined.png")
        combined_file = self.create_correlation_visualization(
            predictions_df, correlation_results_df, combined_path, region_bounds
        )
        created_files['combined'] = combined_file
        
        logger.info(f"✅ Created {len(created_files)} correlation visualizations")
        return created_files
    
    def _create_single_status_visualization(self,
                                          detections_df: pd.DataFrame,
                                          output_path: str,
                                          title: str,
                                          color: Tuple[int, int, int],
                                          region_bounds: Optional[Tuple[int, int, int, int]]) -> str:
        """Create visualization for single correlation status (correlated or dark)."""
        
        # Prepare image
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            display_image = self.sar_image[y_min:y_max, x_min:x_max].copy()
            offset_x, offset_y = x_min, y_min
        else:
            display_image = self.sar_image.copy()
            offset_x, offset_y = 0, 0
        
        # Convert to BGR
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        
        # Draw detections
        for _, detection in detections_df.iterrows():
            x1 = int(detection['bbox_x1'] - offset_x)
            y1 = int(detection['bbox_y1'] - offset_y)
            x2 = int(detection['bbox_x2'] - offset_x)
            y2 = int(detection['bbox_y2'] - offset_y)
            
            # Bounds check
            x1 = max(0, min(x1, display_image.shape[1]))
            y1 = max(0, min(y1, display_image.shape[0]))
            x2 = max(0, min(x2, display_image.shape[1]))
            y2 = max(0, min(y2, display_image.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            if 'matched_mmsi' in detection and pd.notna(detection['matched_mmsi']) and detection['matched_mmsi'] != 'UNKNOWN':
                label = f"ID:{detection['detect_id']}\nMMSI:{detection['matched_mmsi']}"
            else:
                label = f"ID:{detection['detect_id']}\nDARK SHIP"
            
            # Add confidence
            label += f"\nConf:{detection['confidence']:.2f}"
            
            # Draw label
            lines = label.split('\n')
            line_height = 20
            max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines])
            
            # Background
            cv2.rectangle(display_image, (x1, y1-len(lines)*line_height-10), 
                         (x1+max_width+10, y1-5), (0, 0, 0), -1)
            
            # Text
            for i, line in enumerate(lines):
                text_y = y1 - (len(lines)-i) * line_height + 5
                cv2.putText(display_image, line, (x1+5, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(display_image, f"{title}: {len(detections_df)} vessels", (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Save
        cv2.imwrite(output_path, display_image)
        logger.info(f"✅ {title} visualization saved: {output_path}")
        
        return output_path

def create_correlation_visualizations(sar_image_path: str,
                                    predictions_csv_path: str,
                                    correlation_results_csv_path: str,
                                    output_dir: str,
                                    region_bounds: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, str]:
    """
    Standalone function to create all correlation visualizations.
    
    Args:
        sar_image_path: Path to SAR image
        predictions_csv_path: Path to predictions CSV
        correlation_results_csv_path: Path to correlation results CSV
        output_dir: Output directory
        region_bounds: Optional region bounds
        
    Returns:
        Dictionary with paths to created files
    """
    logger.info("🎨 Creating correlation visualizations...")
    
    # Load data
    predictions_df = pd.read_csv(predictions_csv_path)
    correlation_results_df = pd.read_csv(correlation_results_csv_path)
    
    # Create visualizer
    visualizer = CorrelationVisualizer(sar_image_path)
    
    # Create visualizations
    created_files = visualizer.create_split_visualizations(
        predictions_df, correlation_results_df, output_dir, region_bounds
    )
    
    return created_files

if __name__ == "__main__":
    # Test the correlation visualizer
    logging.basicConfig(level=logging.INFO)
    
    # Test with existing data
    create_correlation_visualizations(
        sar_image_path="snap_output/step4_final_output.tif",
        predictions_csv_path="professor/outputs/predictions.csv",
        correlation_results_csv_path="ais_correlation/ais_correlation_results.csv",
        output_dir="correlation_visualizations"
    )
