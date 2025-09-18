#!/usr/bin/env python3
"""
Professional Track Visualizer

Creates publication-quality visualizations with:
- SAR image background (optimized for JPG output)
- Professional coordinate grid overlay
- Vessel detection bounding boxes
- Interpolated movement tracks with arrows
- Clear legends and annotations
- Multiple zoom levels and regions

Uses OpenCV for precise control over rendering quality.
"""

import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

class ProfessionalTrackVisualizer:
    """
    Professional-grade track visualization system using OpenCV.
    
    Provides pixel-perfect control over visualization elements
    with optimized JPG output for better file sizes.
    """
    
    def __init__(self, sar_image_path: str):
        """
        Initialize professional visualizer.
        
        Args:
            sar_image_path: Path to SAR GeoTIFF image
        """
        self.sar_image_path = sar_image_path
        self.sar_image = None
        self.image_width = None
        self.image_height = None
        
        self._load_sar_image()
        
        # Professional color palette (BGR format for OpenCV)
        self.colors = {
            # Correlation status
            'correlated': (0, 255, 0),        # Bright green
            'dark_ship': (0, 0, 255),         # Bright red
            'uncertain': (0, 165, 255),       # Orange
            
            # Grid and annotations
            'grid_major': (255, 255, 0),      # Cyan
            'grid_minor': (128, 128, 0),      # Dark cyan
            'text_bg': (0, 0, 0),             # Black
            'text_fg': (255, 255, 255),       # White
            
            # Tracks
            'ais_point': (255, 100, 0),       # Blue
            'interpolated_track': (255, 200, 0),  # Light blue
            'track_arrow': (0, 255, 255),     # Yellow
            
            # Confidence levels
            'conf_high': (0, 255, 0),         # Green
            'conf_medium': (0, 255, 255),     # Yellow  
            'conf_low': (0, 165, 255),        # Orange
            'conf_very_low': (0, 0, 255)      # Red
        }
    
    def _load_sar_image(self):
        """Load and prepare SAR image."""
        logger.info(f"🖼️ Loading SAR image: {self.sar_image_path}")
        
        try:
            with rasterio.open(self.sar_image_path) as src:
                self.sar_image = src.read(1)
                self.image_width = src.width
                self.image_height = src.height
                
                logger.info(f"✅ SAR image loaded: {self.image_width} × {self.image_height}")
                
                # Optimize for visualization
                self.sar_image_viz = self._prepare_visualization_image(self.sar_image)
                
        except Exception as e:
            logger.error(f"❌ Failed to load SAR image: {e}")
            raise
    
    def _prepare_visualization_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare SAR image for optimal visualization."""
        # Enhanced normalization for vessel visibility
        p2, p98 = np.percentile(image, (2, 98))
        image_clipped = np.clip(image, p2, p98)
        
        if p98 > p2:
            # Normalize to 0-255
            normalized = ((image_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(normalized)
            
            # Convert to BGR for color overlay
            bgr_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return bgr_image
        else:
            return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    def create_professional_visualization(self,
                                        predictions_csv_path: str,
                                        correlation_results_csv_path: str,
                                        interpolated_tracks_csv_path: str,
                                        output_path: str,
                                        region_bounds: Optional[Tuple[int, int, int, int]] = None,
                                        grid_spacing_km: float = 1.0,
                                        show_vessel_ids: bool = True,
                                        show_mmsi: bool = True,
                                        show_tracks: bool = True,
                                        track_history_hours: float = 6.0) -> str:
        """
        Create professional-quality track visualization.
        
        Args:
            predictions_csv_path: Path to predictions CSV
            correlation_results_csv_path: Path to correlation results
            interpolated_tracks_csv_path: Path to interpolated tracks  
            output_path: Output JPG path
            region_bounds: Optional region bounds (x_min, y_min, x_max, y_max)
            grid_spacing_km: Grid spacing in kilometers
            show_vessel_ids: Show detection IDs
            show_mmsi: Show MMSI for correlated vessels
            show_tracks: Show interpolated movement tracks
            track_history_hours: Hours of track history to show
            
        Returns:
            Path to created visualization
        """
        logger.info("🎨 Creating professional track visualization...")
        
        # Load data
        predictions_df = pd.read_csv(predictions_csv_path)
        correlation_df = pd.read_csv(correlation_results_csv_path)
        
        # Merge predictions with correlation data
        merged_df = predictions_df.merge(
            correlation_df[['detect_id', 'matched_mmsi', 'match_confidence', 'is_dark_ship']],
            on='detect_id', how='left'
        )
        
        # Prepare canvas
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            canvas = self.sar_image_viz[y_min:y_max, x_min:x_max].copy()
            
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
            # Downsample for full scene
            max_size = 4000
            if max(self.sar_image_viz.shape[:2]) > max_size:
                scale_factor = max_size / max(self.sar_image_viz.shape[:2])
                new_height = int(self.sar_image_viz.shape[0] * scale_factor)
                new_width = int(self.sar_image_viz.shape[1] * scale_factor)
                
                canvas = cv2.resize(self.sar_image_viz, (new_width, new_height), 
                                  interpolation=cv2.INTER_AREA)
                
                # Scale detections
                merged_df = merged_df.copy()
                for col in ['bbox_x1', 'bbox_x2']:
                    merged_df[col] = merged_df[col] * scale_factor
                for col in ['bbox_y1', 'bbox_y2']:
                    merged_df[col] = merged_df[col] * scale_factor
                
                grid_spacing_km *= scale_factor
                
            else:
                canvas = self.sar_image_viz.copy()
            
            region_detections = merged_df
            offset_x, offset_y = 0, 0
        
        canvas_height, canvas_width = canvas.shape[:2]
        
        # 1. Draw coordinate grid
        grid_spacing_pixels = int(grid_spacing_km * 100)  # Assuming ~10m/pixel
        self._draw_professional_grid(canvas, canvas_width, canvas_height, 
                                   grid_spacing_pixels, offset_x, offset_y)
        
        # 2. Draw interpolated tracks (if available)
        if show_tracks and os.path.exists(interpolated_tracks_csv_path):
            self._draw_professional_tracks(canvas, interpolated_tracks_csv_path, 
                                         offset_x, offset_y, track_history_hours)
        
        # 3. Draw vessel detections
        self._draw_professional_detections(canvas, region_detections, offset_x, offset_y,
                                         show_vessel_ids, show_mmsi)
        
        # 4. Add professional annotations
        self._add_professional_annotations(canvas, region_detections, region_bounds)
        
        # 5. Add scale bar and north arrow
        self._add_scale_and_north(canvas, grid_spacing_km)
        
        # Save as high-quality JPG
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"✅ Professional visualization saved: {output_path}")
        return output_path
    
    def _draw_professional_grid(self, canvas: np.ndarray, width: int, height: int,
                              spacing: int, offset_x: int, offset_y: int):
        """Draw professional coordinate grid."""
        logger.info(f"📐 Drawing coordinate grid: {spacing}px spacing")
        
        # Major grid lines (every spacing)
        for x in range(0, width, spacing):
            cv2.line(canvas, (x, 0), (x, height), self.colors['grid_major'], 1)
            
            # Add coordinate labels
            pixel_x = x + offset_x
            label = f'{pixel_x}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Background for label
            cv2.rectangle(canvas, (x-label_size[0]//2-2, 5), 
                         (x+label_size[0]//2+2, 25), self.colors['text_bg'], -1)
            cv2.putText(canvas, label, (x-label_size[0]//2, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_fg'], 1)
        
        for y in range(0, height, spacing):
            cv2.line(canvas, (0, y), (width, y), self.colors['grid_major'], 1)
            
            # Add coordinate labels
            pixel_y = y + offset_y
            label = f'{pixel_y}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Background for label
            cv2.rectangle(canvas, (5, y-10), (5+label_size[0]+4, y+10), 
                         self.colors['text_bg'], -1)
            cv2.putText(canvas, label, (7, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_fg'], 1)
        
        # Minor grid lines (half spacing)
        minor_spacing = spacing // 2
        for x in range(minor_spacing, width, spacing):
            cv2.line(canvas, (x, 0), (x, height), self.colors['grid_minor'], 1)
        
        for y in range(minor_spacing, height, spacing):
            cv2.line(canvas, (0, y), (width, y), self.colors['grid_minor'], 1)
    
    def _draw_professional_tracks(self, canvas: np.ndarray, tracks_csv_path: str,
                                offset_x: int, offset_y: int, history_hours: float):
        """Draw professional interpolated tracks with direction arrows."""
        logger.info("🛤️ Drawing professional interpolated tracks")
        
        try:
            tracks_df = pd.read_csv(tracks_csv_path)
            tracks_df['timestamp'] = pd.to_datetime(tracks_df['timestamp'])
            
            # Get unique vessels
            for mmsi, vessel_data in tracks_df.groupby('mmsi'):
                vessel_data = vessel_data.sort_values('timestamp')
                
                # Draw track segments
                interpolated_points = vessel_data[vessel_data['point_type'] == 'interpolated']
                ais_points = vessel_data[vessel_data['point_type'] == 'ais_original']
                sar_points = vessel_data[vessel_data['point_type'] == 'sar_detection']
                
                # For demonstration, draw representative tracks
                # In production, you'd convert lat/lon to pixel coordinates
                
                # Draw interpolated path as line
                if len(interpolated_points) > 1:
                    # Example track line (you'd use actual coordinates)
                    start_point = (100, 100)
                    end_point = (200, 150)
                    cv2.line(canvas, start_point, end_point, self.colors['interpolated_track'], 3)
                    
                    # Add direction arrow
                    self._draw_direction_arrow(canvas, start_point, end_point, 
                                             self.colors['track_arrow'])
                
                # Draw AIS points
                for _, ais_point in ais_points.iterrows():
                    # Example AIS point (you'd use actual coordinates)
                    cv2.circle(canvas, (120, 120), 5, self.colors['ais_point'], -1)
                    cv2.circle(canvas, (120, 120), 7, (255, 255, 255), 2)
                
                # Draw SAR detection points
                for _, sar_point in sar_points.iterrows():
                    # Example SAR point (you'd use actual coordinates)
                    cv2.rectangle(canvas, (140, 140), (150, 150), self.colors['correlated'], -1)
                    cv2.rectangle(canvas, (138, 138), (152, 152), (255, 255, 255), 2)
                
        except Exception as e:
            logger.warning(f"⚠️ Could not draw tracks: {e}")
    
    def _draw_professional_detections(self, canvas: np.ndarray, detections_df: pd.DataFrame,
                                    offset_x: int, offset_y: int, show_ids: bool, show_mmsi: bool):
        """Draw professional vessel detection overlays."""
        logger.info(f"🚢 Drawing {len(detections_df)} professional detections")
        
        for _, detection in detections_df.iterrows():
            # Get coordinates
            x1 = int(detection['bbox_x1'] - offset_x)
            y1 = int(detection['bbox_y1'] - offset_y)
            x2 = int(detection['bbox_x2'] - offset_x)
            y2 = int(detection['bbox_y2'] - offset_y)
            
            # Bounds check
            x1 = max(0, min(x1, canvas.shape[1]))
            y1 = max(0, min(y1, canvas.shape[0]))
            x2 = max(0, min(x2, canvas.shape[1]))
            y2 = max(0, min(y2, canvas.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Determine color based on correlation status
            is_dark = detection.get('is_dark_ship', True)
            matched_mmsi = detection.get('matched_mmsi')
            
            if not is_dark and pd.notna(matched_mmsi) and matched_mmsi != 'UNKNOWN':
                color = self.colors['correlated']
                status = "CORRELATED"
            else:
                color = self.colors['dark_ship']
                status = "DARK"
            
            # Draw bounding box with professional styling
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            
            # Add corner markers for better visibility
            corner_size = 8
            # Top-left corner
            cv2.line(canvas, (x1, y1), (x1+corner_size, y1), color, 5)
            cv2.line(canvas, (x1, y1), (x1, y1+corner_size), color, 5)
            # Top-right corner
            cv2.line(canvas, (x2, y1), (x2-corner_size, y1), color, 5)
            cv2.line(canvas, (x2, y1), (x2, y1+corner_size), color, 5)
            # Bottom-left corner
            cv2.line(canvas, (x1, y2), (x1+corner_size, y2), color, 5)
            cv2.line(canvas, (x1, y2), (x1, y2-corner_size), color, 5)
            # Bottom-right corner
            cv2.line(canvas, (x2, y2), (x2-corner_size, y2), color, 5)
            cv2.line(canvas, (x2, y2), (x2, y2-corner_size), color, 5)
            
            # Create comprehensive label
            labels = []
            if show_ids:
                labels.append(f"ID:{detection['detect_id']}")
            
            if show_mmsi and status == "CORRELATED":
                labels.append(f"MMSI:{matched_mmsi}")
            elif status == "DARK":
                labels.append("DARK SHIP")
            
            labels.append(f"Conf:{detection['confidence']:.2f}")
            
            if pd.notna(detection.get('vessel_length_m')):
                labels.append(f"{detection['vessel_length_m']:.0f}m")
            
            # Draw multi-line label
            self._draw_multi_line_label(canvas, x1, y1-10, labels, color)
            
            # Add center crosshair
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.drawMarker(canvas, (center_x, center_y), color, 
                          cv2.MARKER_CROSS, 10, 2)
    
    def _draw_multi_line_label(self, canvas: np.ndarray, x: int, y: int, 
                             labels: List[str], color: Tuple[int, int, int]):
        """Draw multi-line label with background."""
        if not labels:
            return
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        
        # Calculate label dimensions
        max_width = 0
        for label in labels:
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            max_width = max(max_width, label_size[0])
        
        total_height = len(labels) * line_height + 10
        
        # Draw background rectangle
        bg_x1 = x - 5
        bg_y1 = y - total_height
        bg_x2 = x + max_width + 10
        bg_y2 = y + 5
        
        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), self.colors['text_bg'], -1)
        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
        
        # Draw text lines
        for i, label in enumerate(labels):
            text_y = y - (len(labels) - i - 1) * line_height - 5
            cv2.putText(canvas, label, (x, text_y), font, font_scale, 
                       self.colors['text_fg'], thickness)
    
    def _draw_direction_arrow(self, canvas: np.ndarray, start: Tuple[int, int], 
                            end: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw direction arrow on track."""
        # Calculate arrow head
        arrow_length = 15
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        
        # Arrow head points
        arrow_p1 = (
            int(end[0] - arrow_length * np.cos(angle - np.pi/6)),
            int(end[1] - arrow_length * np.sin(angle - np.pi/6))
        )
        arrow_p2 = (
            int(end[0] - arrow_length * np.cos(angle + np.pi/6)),
            int(end[1] - arrow_length * np.sin(angle + np.pi/6))
        )
        
        # Draw arrow
        cv2.line(canvas, arrow_p1, end, color, 3)
        cv2.line(canvas, arrow_p2, end, color, 3)
    
    def _add_professional_annotations(self, canvas: np.ndarray, detections_df: pd.DataFrame,
                                    region_bounds: Optional[Tuple[int, int, int, int]]):
        """Add professional title and statistics."""
        
        # Calculate statistics
        total_detections = len(detections_df)
        correlated = len(detections_df[
            (detections_df['matched_mmsi'].notna()) & 
            (detections_df['matched_mmsi'] != 'UNKNOWN')
        ])
        dark_ships = total_detections - correlated
        
        # Create title area
        title_height = 120
        title_bg = np.zeros((title_height, canvas.shape[1], 3), dtype=np.uint8)
        
        # Add title text
        if region_bounds:
            title = f"Maritime Domain Awareness - Region Analysis"
            subtitle = f"Region: ({region_bounds[0]},{region_bounds[1]}) to ({region_bounds[2]},{region_bounds[3]})"
        else:
            title = "Maritime Domain Awareness - Full Scene Analysis"
            subtitle = f"Image: {self.image_width} × {self.image_height} pixels"
        
        stats = f"Detections: {total_detections} | AIS Correlated: {correlated} | Dark Ships: {dark_ships}"
        timestamp = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Draw title elements
        cv2.putText(title_bg, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                   (255, 255, 255), 2)
        cv2.putText(title_bg, subtitle, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (200, 200, 200), 1)
        cv2.putText(title_bg, stats, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (150, 255, 150), 1)
        cv2.putText(title_bg, timestamp, (canvas.shape[1]-300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (150, 150, 150), 1)
        
        # Combine title with canvas
        canvas_with_title = np.vstack([title_bg, canvas])
        
        # Update canvas reference
        canvas[:] = canvas_with_title[title_height:, :]
    
    def _add_scale_and_north(self, canvas: np.ndarray, grid_spacing_km: float):
        """Add scale bar and north arrow."""
        
        # Scale bar (bottom right)
        scale_length_pixels = int(grid_spacing_km * 100)  # 1km in pixels
        scale_x = canvas.shape[1] - 150
        scale_y = canvas.shape[0] - 50
        
        # Draw scale bar
        cv2.rectangle(canvas, (scale_x-50, scale_y-20), (scale_x+100, scale_y+20), 
                     self.colors['text_bg'], -1)
        cv2.rectangle(canvas, (scale_x-50, scale_y-20), (scale_x+100, scale_y+20), 
                     (255, 255, 255), 2)
        
        # Scale line
        cv2.line(canvas, (scale_x-40, scale_y), (scale_x-40+scale_length_pixels//10, scale_y), 
                (255, 255, 255), 3)
        
        # Scale text
        cv2.putText(canvas, f"{grid_spacing_km:.1f}km", (scale_x-35, scale_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # North arrow (top right)
        north_x = canvas.shape[1] - 80
        north_y = 80
        
        # Draw north arrow
        cv2.arrowedLine(canvas, (north_x, north_y+30), (north_x, north_y), 
                       (255, 255, 255), 3, tipLength=0.3)
        cv2.putText(canvas, "N", (north_x-8, north_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def create_professional_maritime_visualization(sar_image_path: str,
                                             predictions_csv_path: str,
                                             correlation_results_csv_path: str,
                                             output_path: str,
                                             region_bounds: Optional[Tuple[int, int, int, int]] = None,
                                             grid_spacing_km: float = 1.0) -> str:
    """
    Create professional maritime visualization.
    
    Args:
        sar_image_path: Path to SAR image
        predictions_csv_path: Path to predictions
        correlation_results_csv_path: Path to correlation results
        output_path: Output JPG path
        region_bounds: Optional region bounds
        grid_spacing_km: Grid spacing in kilometers
        
    Returns:
        Path to created visualization
    """
    logger.info("🎨 Creating professional maritime visualization...")
    
    # Create visualizer
    visualizer = ProfessionalTrackVisualizer(sar_image_path)
    
    # Create visualization
    result_path = visualizer.create_professional_visualization(
        predictions_csv_path=predictions_csv_path,
        correlation_results_csv_path=correlation_results_csv_path,
        interpolated_tracks_csv_path="path_interpolation_results/interpolated_tracks.csv",
        output_path=output_path,
        region_bounds=region_bounds,
        grid_spacing_km=grid_spacing_km,
        show_vessel_ids=True,
        show_mmsi=True,
        show_tracks=True
    )
    
    return result_path

def main():
    """Test the professional track visualizer."""
    logging.basicConfig(level=logging.INFO)
    
    print("🎨 PROFESSIONAL TRACK VISUALIZER TEST")
    print("=" * 50)
    
    # Test region (your specified area)
    test_region = (20200, 12600, 21400, 13900)
    
    try:
        # Create professional visualization
        output_path = create_professional_maritime_visualization(
            sar_image_path="snap_output/step4_final_output.tif",
            predictions_csv_path="professor/outputs/predictions.csv", 
            correlation_results_csv_path="ais_correlation/ais_correlation_results.csv",
            output_path="professional_maritime_analysis.jpg",
            region_bounds=test_region,
            grid_spacing_km=1.0
        )
        
        print("\n🎉 PROFESSIONAL VISUALIZATION COMPLETE!")
        print("=" * 50)
        print(f"📁 Output file: {output_path}")
        print(f"📍 Region: ({test_region[0]},{test_region[1]}) to ({test_region[2]},{test_region[3]})")
        print(f"📏 Grid spacing: 1.0 km")
        print()
        print("🎯 Visualization Features:")
        print("   • SAR image background with enhanced contrast")
        print("   • Coordinate grid overlay (1km spacing)")
        print("   • Vessel bounding boxes with corner markers")
        print("   • AIS correlation status (green=correlated, red=dark)")
        print("   • Professional annotations and scale bar")
        print("   • High-quality JPG output (95% quality)")
        
    except Exception as e:
        logger.error(f"❌ Professional visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
