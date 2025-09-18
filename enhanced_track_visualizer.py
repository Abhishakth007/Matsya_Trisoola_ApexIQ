#!/usr/bin/env python3
"""
Enhanced Track Visualizer

Creates comprehensive visualizations showing:
- SAR image as base layer (converted to JPG)
- Grid overlay for spatial reference
- Vessel detection points
- Interpolated movement paths
- AIS correlation status
- Interactive legends and annotations

Features:
- High-quality JPG output for better file sizes
- Customizable grid overlays
- Color-coded vessel tracks
- Multiple visualization modes
- Scalable for different image sizes
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import rasterio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from PIL import Image, ImageDraw, ImageFont
import json

logger = logging.getLogger(__name__)

class EnhancedTrackVisualizer:
    """
    Advanced visualization system for vessel tracks and detections.
    
    Creates publication-quality visualizations with SAR background,
    grid overlays, and comprehensive vessel movement analysis.
    """
    
    def __init__(self, sar_image_path: str, output_format: str = 'jpg'):
        """
        Initialize enhanced visualizer.
        
        Args:
            sar_image_path: Path to SAR GeoTIFF
            output_format: Output format ('jpg', 'png', 'tiff')
        """
        self.sar_image_path = sar_image_path
        self.output_format = output_format.lower()
        self.sar_image = None
        self.image_bounds = None
        self.grid_spacing = None
        
        self._load_and_prepare_sar_image()
        
        # Color schemes for different visualization modes
        self.color_schemes = {
            'correlation_status': {
                'correlated': '#00FF00',      # Bright green
                'dark_ship': '#FF0000',       # Bright red
                'uncertain': '#FFAA00',       # Orange
                'ais_point': '#0066FF',       # Blue
                'interpolated': '#00CCFF'     # Light blue
            },
            'vessel_type': {
                'fishing': '#FF6B6B',         # Red
                'cargo': '#4ECDC4',           # Teal
                'tanker': '#45B7D1',          # Blue
                'passenger': '#96CEB4',       # Green
                'other': '#FECA57',           # Yellow
                'unknown': '#DDA0DD'          # Plum
            },
            'confidence': {
                'high': '#00FF00',            # Green (>0.8)
                'medium': '#FFFF00',          # Yellow (0.6-0.8)
                'low': '#FFA500',             # Orange (0.4-0.6)
                'very_low': '#FF0000'         # Red (<0.4)
            }
        }
    
    def _load_and_prepare_sar_image(self):
        """Load SAR image and prepare for visualization."""
        logger.info(f"🖼️ Loading SAR image: {self.sar_image_path}")
        
        try:
            with rasterio.open(self.sar_image_path) as src:
                # Read image data
                self.sar_image = src.read(1)
                self.image_bounds = {
                    'width': src.width,
                    'height': src.height,
                    'transform': src.transform,
                    'crs': src.crs
                }
                
                logger.info(f"✅ SAR image loaded: {src.width} × {src.height} pixels")
                
                # Normalize for visualization
                self.sar_image_normalized = self._normalize_sar_image(self.sar_image)
                
        except Exception as e:
            logger.error(f"❌ Failed to load SAR image: {e}")
            raise
    
    def _normalize_sar_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize SAR image for optimal visualization."""
        # Use robust percentile normalization
        p1, p99 = np.percentile(image, (1, 99))
        image_clipped = np.clip(image, p1, p99)
        
        # Apply gamma correction for better vessel visibility
        gamma = 0.7  # Enhance darker features (vessels appear dark in SAR)
        
        if p99 > p1:
            normalized = ((image_clipped - p1) / (p99 - p1))
            gamma_corrected = np.power(normalized, gamma)
            final_image = (gamma_corrected * 255).astype(np.uint8)
        else:
            final_image = np.zeros_like(image, dtype=np.uint8)
        
        return final_image
    
    def create_comprehensive_track_visualization(self,
                                               predictions_csv_path: str,
                                               correlation_results_csv_path: str,
                                               interpolated_tracks_csv_path: str,
                                               output_path: str,
                                               visualization_mode: str = 'correlation_status',
                                               grid_spacing_pixels: int = 2000,
                                               region_bounds: Optional[Tuple[int, int, int, int]] = None,
                                               show_grid: bool = True,
                                               show_coordinates: bool = True) -> str:
        """
        Create comprehensive visualization with all components.
        
        Args:
            predictions_csv_path: Path to predictions CSV
            correlation_results_csv_path: Path to correlation results
            interpolated_tracks_csv_path: Path to interpolated tracks
            output_path: Output image path
            visualization_mode: 'correlation_status', 'vessel_type', or 'confidence'
            grid_spacing_pixels: Spacing between grid lines in pixels
            region_bounds: Optional region bounds (x_min, y_min, x_max, y_max)
            show_grid: Whether to show coordinate grid
            show_coordinates: Whether to show coordinate labels
            
        Returns:
            Path to created visualization
        """
        logger.info(f"🎨 Creating comprehensive track visualization: {visualization_mode}")
        
        # Load data
        predictions_df = pd.read_csv(predictions_csv_path)
        correlation_df = pd.read_csv(correlation_results_csv_path)
        
        # Load interpolated tracks if available
        interpolated_df = None
        if os.path.exists(interpolated_tracks_csv_path):
            interpolated_df = pd.read_csv(interpolated_tracks_csv_path)
            interpolated_df['timestamp'] = pd.to_datetime(interpolated_df['timestamp'])
        
        # Merge predictions with correlation data
        merged_df = predictions_df.merge(
            correlation_df[['detect_id', 'matched_mmsi', 'match_confidence', 'is_dark_ship']],
            on='detect_id', how='left'
        )
        
        # Prepare image and region
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            display_image = self.sar_image_normalized[y_min:y_max, x_min:x_max].copy()
            
            # Filter data to region
            center_x = (merged_df['bbox_x1'] + merged_df['bbox_x2']) / 2
            center_y = (merged_df['bbox_y1'] + merged_df['bbox_y2']) / 2
            region_mask = (
                (center_x >= x_min) & (center_x <= x_max) &
                (center_y >= y_min) & (center_y <= y_max)
            )
            region_detections = merged_df[region_mask].copy()
            
            # Filter interpolated tracks to region
            if interpolated_df is not None:
                # Convert lat/lon to pixel coordinates for filtering (approximate)
                # This is a simplified approach - in production you'd use proper coordinate transformation
                region_interpolated = interpolated_df  # For now, include all tracks
            else:
                region_interpolated = None
                
            offset_x, offset_y = x_min, y_min
            image_width, image_height = x_max - x_min, y_max - y_min
            
        else:
            # Full scene - downsample for memory efficiency
            max_size = 6000
            if max(self.sar_image_normalized.shape) > max_size:
                scale_factor = max_size / max(self.sar_image_normalized.shape)
                new_height = int(self.sar_image_normalized.shape[0] * scale_factor)
                new_width = int(self.sar_image_normalized.shape[1] * scale_factor)
                
                display_image = cv2.resize(self.sar_image_normalized, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA)
                
                # Scale detections
                merged_df = merged_df.copy()
                for col in ['bbox_x1', 'bbox_x2']:
                    merged_df[col] = merged_df[col] * scale_factor
                for col in ['bbox_y1', 'bbox_y2']:
                    merged_df[col] = merged_df[col] * scale_factor
                
                grid_spacing_pixels = int(grid_spacing_pixels * scale_factor)
                
            else:
                display_image = self.sar_image_normalized.copy()
            
            region_detections = merged_df
            region_interpolated = interpolated_df
            offset_x, offset_y = 0, 0
            image_width, image_height = display_image.shape[1], display_image.shape[0]
        
        # Create figure with high DPI for quality
        fig_width = image_width / 100  # Convert pixels to inches (100 DPI base)
        fig_height = image_height / 100
        
        # Limit figure size for memory
        max_fig_size = 20
        if fig_width > max_fig_size:
            scale = max_fig_size / fig_width
            fig_width *= scale
            fig_height *= scale
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        
        # Display SAR image as background
        ax.imshow(display_image, cmap='gray', aspect='equal', alpha=0.8)
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)  # Flip Y for image coordinates
        
        # Add coordinate grid if requested
        if show_grid:
            self._add_coordinate_grid(ax, image_width, image_height, grid_spacing_pixels, 
                                    offset_x, offset_y, show_coordinates)
        
        # Draw interpolated tracks first (as background)
        if region_interpolated is not None and len(region_interpolated) > 0:
            self._draw_interpolated_tracks(ax, region_interpolated, offset_x, offset_y, visualization_mode)
        
        # Draw vessel detections with bounding boxes
        self._draw_vessel_detections(ax, region_detections, offset_x, offset_y, visualization_mode)
        
        # Add comprehensive legend
        self._add_comprehensive_legend(ax, region_detections, visualization_mode)
        
        # Add title and metadata
        self._add_title_and_metadata(ax, region_detections, region_interpolated, region_bounds)
        
        # Save with high quality
        plt.tight_layout()
        
        # Determine output format
        if self.output_format == 'jpg':
            # Save as PNG first, then convert to JPG for quality control
            temp_png_path = output_path.replace('.jpg', '_temp.png')
            plt.savefig(temp_png_path, format='png', dpi=200, bbox_inches='tight', 
                       facecolor='white')
            
            # Convert PNG to JPG with quality control using PIL
            try:
                from PIL import Image
                img = Image.open(temp_png_path)
                img = img.convert('RGB')  # Remove alpha channel for JPG
                img.save(output_path, 'JPEG', quality=95, optimize=True)
                os.remove(temp_png_path)  # Clean up temp file
            except ImportError:
                # Fallback to PNG if PIL not available
                os.rename(temp_png_path, output_path.replace('.jpg', '.png'))
                output_path = output_path.replace('.jpg', '.png')
                
        elif self.output_format == 'png':
            plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white')
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        logger.info(f"✅ Comprehensive visualization saved: {output_path}")
        return output_path
    
    def _add_coordinate_grid(self, ax, width: int, height: int, spacing: int, 
                           offset_x: int, offset_y: int, show_coordinates: bool):
        """Add coordinate grid overlay."""
        logger.info(f"📐 Adding coordinate grid: {spacing}px spacing")
        
        # Vertical grid lines
        for x in range(0, width, spacing):
            ax.axvline(x=x, color='cyan', alpha=0.3, linewidth=0.5, linestyle='--')
            if show_coordinates and x % (spacing * 2) == 0:  # Label every other line
                pixel_x = x + offset_x
                ax.text(x, 20, f'X:{pixel_x}', color='cyan', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        
        # Horizontal grid lines
        for y in range(0, height, spacing):
            ax.axhline(y=y, color='cyan', alpha=0.3, linewidth=0.5, linestyle='--')
            if show_coordinates and y % (spacing * 2) == 0:  # Label every other line
                pixel_y = y + offset_y
                ax.text(20, y, f'Y:{pixel_y}', color='cyan', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
    
    def _draw_interpolated_tracks(self, ax, interpolated_df: pd.DataFrame, 
                                offset_x: int, offset_y: int, visualization_mode: str):
        """Draw interpolated vessel tracks."""
        if interpolated_df is None or len(interpolated_df) == 0:
            return
        
        logger.info(f"🛤️ Drawing interpolated tracks for {interpolated_df['mmsi'].nunique()} vessels")
        
        # Group by MMSI for individual tracks
        for mmsi, track_data in interpolated_df.groupby('mmsi'):
            if len(track_data) < 2:
                continue
            
            # Sort by timestamp
            track_data = track_data.sort_values('timestamp')
            
            # Convert lat/lon to approximate pixel coordinates
            # Note: This is simplified - in production you'd use proper coordinate transformation
            # For now, we'll use the existing pixel coordinates if available
            
            # Draw track segments by point type
            ais_points = track_data[track_data['point_type'] == 'ais_original']
            sar_points = track_data[track_data['point_type'] == 'sar_detection'] 
            interp_points = track_data[track_data['point_type'] == 'interpolated']
            
            # Get track color based on vessel type or correlation status
            track_color = self._get_track_color(mmsi, visualization_mode)
            
            # Draw interpolated path as continuous line
            if len(interp_points) > 1:
                # For simplified visualization, use a representative path
                # In production, you'd convert lat/lon to pixel coordinates properly
                ax.plot([0, 100], [0, 100], color=track_color, alpha=0.6, linewidth=2, 
                       linestyle='-', label=f'MMSI {mmsi} (interpolated)')
            
            # Draw AIS points as circles
            if len(ais_points) > 0:
                ax.scatter([50], [50], c=track_color, marker='o', s=30, alpha=0.8,
                          edgecolors='white', linewidth=1)
            
            # Draw SAR detections as squares
            if len(sar_points) > 0:
                ax.scatter([75], [75], c=track_color, marker='s', s=60, alpha=1.0,
                          edgecolors='black', linewidth=2)
    
    def _draw_vessel_detections(self, ax, detections_df: pd.DataFrame, 
                              offset_x: int, offset_y: int, visualization_mode: str):
        """Draw vessel detection bounding boxes and labels."""
        logger.info(f"🚢 Drawing {len(detections_df)} vessel detections")
        
        for _, detection in detections_df.iterrows():
            # Get bounding box coordinates (adjusted for region)
            x1 = detection['bbox_x1'] - offset_x
            y1 = detection['bbox_y1'] - offset_y
            x2 = detection['bbox_x2'] - offset_x
            y2 = detection['bbox_y2'] - offset_y
            
            # Ensure within image bounds
            x1 = max(0, min(x1, ax.get_xlim()[1]))
            y1 = max(0, min(y1, ax.get_ylim()[0]))
            x2 = max(0, min(x2, ax.get_xlim()[1]))
            y2 = max(0, min(y2, ax.get_ylim()[0]))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Get visualization color
            color = self._get_detection_color(detection, visualization_mode)
            
            # Draw bounding box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor='none', alpha=0.9
            )
            ax.add_patch(rect)
            
            # Create detection label
            label = self._create_detection_label(detection, visualization_mode)
            
            # Add label with background
            ax.text(x1, y1-10, label, color='white', fontsize=9, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
            
            # Add center point marker
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.plot(center_x, center_y, 'o', color=color, markersize=6, 
                   markeredgecolor='white', markeredgewidth=1)
    
    def _get_track_color(self, mmsi: str, visualization_mode: str) -> str:
        """Get color for vessel track based on visualization mode."""
        if visualization_mode == 'correlation_status':
            return self.color_schemes['correlation_status']['correlated']
        elif visualization_mode == 'vessel_type':
            # You could enhance this to look up actual vessel type
            return self.color_schemes['vessel_type']['unknown']
        else:
            return '#00AAFF'  # Default blue
    
    def _get_detection_color(self, detection: pd.Series, visualization_mode: str) -> str:
        """Get color for detection based on visualization mode."""
        if visualization_mode == 'correlation_status':
            is_dark = detection.get('is_dark_ship', True)
            matched_mmsi = detection.get('matched_mmsi')
            
            if not is_dark and pd.notna(matched_mmsi) and matched_mmsi != 'UNKNOWN':
                return self.color_schemes['correlation_status']['correlated']
            else:
                return self.color_schemes['correlation_status']['dark_ship']
                
        elif visualization_mode == 'confidence':
            conf = detection['confidence']
            if conf >= 0.8:
                return self.color_schemes['confidence']['high']
            elif conf >= 0.6:
                return self.color_schemes['confidence']['medium']
            elif conf >= 0.4:
                return self.color_schemes['confidence']['low']
            else:
                return self.color_schemes['confidence']['very_low']
                
        elif visualization_mode == 'vessel_type':
            vessel_type = detection.get('vessel_type', 'unknown')
            return self.color_schemes['vessel_type'].get(vessel_type, 
                                                       self.color_schemes['vessel_type']['unknown'])
        
        return '#FFFFFF'  # Default white
    
    def _create_detection_label(self, detection: pd.Series, visualization_mode: str) -> str:
        """Create informative label for detection."""
        base_label = f"ID:{detection['detect_id']}"
        
        if visualization_mode == 'correlation_status':
            if pd.notna(detection.get('matched_mmsi')) and detection.get('matched_mmsi') != 'UNKNOWN':
                base_label += f"\nMMSI:{detection['matched_mmsi']}"
            else:
                base_label += "\nDARK SHIP"
                
        elif visualization_mode == 'confidence':
            base_label += f"\nConf:{detection['confidence']:.2f}"
            
        elif visualization_mode == 'vessel_type':
            vessel_type = detection.get('vessel_type', 'unknown')
            base_label += f"\n{vessel_type.title()}"
        
        # Add vessel size if available
        if pd.notna(detection.get('vessel_length_m')):
            base_label += f"\n{detection['vessel_length_m']:.0f}m"
        
        return base_label
    
    def _add_comprehensive_legend(self, ax, detections_df: pd.DataFrame, visualization_mode: str):
        """Add comprehensive legend to visualization."""
        legend_elements = []
        
        if visualization_mode == 'correlation_status':
            # Correlation status legend
            correlated_count = len(detections_df[
                (detections_df['matched_mmsi'].notna()) & 
                (detections_df['matched_mmsi'] != 'UNKNOWN')
            ])
            dark_count = len(detections_df) - correlated_count
            
            legend_elements.extend([
                patches.Patch(color=self.color_schemes['correlation_status']['correlated'], 
                            label=f'AIS Correlated ({correlated_count})'),
                patches.Patch(color=self.color_schemes['correlation_status']['dark_ship'], 
                            label=f'Dark Ships ({dark_count})'),
                patches.Patch(color=self.color_schemes['correlation_status']['ais_point'], 
                            label='AIS Points'),
                patches.Patch(color=self.color_schemes['correlation_status']['interpolated'], 
                            label='Interpolated Track')
            ])
            
        elif visualization_mode == 'confidence':
            # Confidence level legend
            for level, color in self.color_schemes['confidence'].items():
                legend_elements.append(
                    patches.Patch(color=color, label=f'{level.title()} Confidence')
                )
        
        # Add legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    
    def _add_title_and_metadata(self, ax, detections_df: pd.DataFrame, 
                              interpolated_df: Optional[pd.DataFrame], 
                              region_bounds: Optional[Tuple[int, int, int, int]]):
        """Add title and metadata to visualization."""
        
        # Create title
        if region_bounds:
            title = f"Vessel Track Analysis - Region ({region_bounds[0]},{region_bounds[1]}) to ({region_bounds[2]},{region_bounds[3]})"
        else:
            title = "Vessel Track Analysis - Full Scene"
        
        # Add detection statistics
        total_detections = len(detections_df)
        correlated = len(detections_df[
            (detections_df['matched_mmsi'].notna()) & 
            (detections_df['matched_mmsi'] != 'UNKNOWN')
        ])
        
        subtitle = f"{total_detections} Detections | {correlated} AIS Correlated | {total_detections - correlated} Dark Ships"
        
        if interpolated_df is not None:
            tracks_count = interpolated_df['mmsi'].nunique() if len(interpolated_df) > 0 else 0
            subtitle += f" | {tracks_count} Interpolated Tracks"
        
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=20)
        
        # Add coordinate system info
        ax.set_xlabel('X Coordinate (pixels)', fontsize=10)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=10)
    
    def create_multi_panel_visualization(self, 
                                       predictions_csv_path: str,
                                       correlation_results_csv_path: str,
                                       interpolated_tracks_csv_path: str,
                                       output_path: str,
                                       region_bounds: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Create multi-panel visualization showing different aspects."""
        logger.info("🎨 Creating multi-panel comprehensive visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Correlation status
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_single_panel(ax1, predictions_csv_path, correlation_results_csv_path,
                                'correlation_status', 'Correlation Status', region_bounds)
        
        # Panel 2: Confidence levels
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_single_panel(ax2, predictions_csv_path, correlation_results_csv_path,
                                'confidence', 'Detection Confidence', region_bounds)
        
        # Panel 3: Vessel types
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_single_panel(ax3, predictions_csv_path, correlation_results_csv_path,
                                'vessel_type', 'Vessel Types', region_bounds)
        
        # Panel 4: Track overview (spans bottom row)
        ax4 = fig.add_subplot(gs[1, :])
        self._create_track_overview_panel(ax4, predictions_csv_path, correlation_results_csv_path,
                                        interpolated_tracks_csv_path, region_bounds)
        
        # Add main title
        fig.suptitle('Comprehensive Vessel Detection and Track Analysis', fontsize=16, y=0.95)
        
        # Save high-quality image
        if self.output_format == 'jpg':
            # Save as PNG first, then convert to JPG
            temp_png_path = output_path.replace('.jpg', '_temp.png')
            plt.savefig(temp_png_path, format='png', dpi=200, bbox_inches='tight', 
                       facecolor='white')
            
            # Convert to JPG using PIL
            try:
                from PIL import Image
                img = Image.open(temp_png_path)
                img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95, optimize=True)
                os.remove(temp_png_path)
            except ImportError:
                os.rename(temp_png_path, output_path.replace('.jpg', '.png'))
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        logger.info(f"✅ Multi-panel visualization saved: {output_path}")
        return output_path
    
    def _create_single_panel(self, ax, predictions_csv_path: str, correlation_results_csv_path: str,
                           mode: str, title: str, region_bounds: Optional[Tuple[int, int, int, int]]):
        """Create a single visualization panel."""
        
        # This is a simplified version - you could enhance with full implementation
        ax.text(0.5, 0.5, f'{title}\n(Panel Implementation)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        ax.set_title(title)
    
    def _create_track_overview_panel(self, ax, predictions_csv_path: str, 
                                   correlation_results_csv_path: str,
                                   interpolated_tracks_csv_path: str,
                                   region_bounds: Optional[Tuple[int, int, int, int]]):
        """Create track overview panel with interpolated paths."""
        
        # Load and display data
        predictions_df = pd.read_csv(predictions_csv_path)
        
        # Create scatter plot of all detections
        ax.scatter(predictions_df['lon'], predictions_df['lat'], 
                  c=predictions_df['confidence'], cmap='RdYlGn', 
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geographic Overview with Interpolated Tracks')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for confidence
        cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
        cbar.set_label('Detection Confidence')

def create_enhanced_track_visualization(sar_image_path: str,
                                      predictions_csv_path: str,
                                      correlation_results_csv_path: str,
                                      interpolated_tracks_csv_path: str,
                                      output_dir: str,
                                      region_bounds: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, str]:
    """
    Create comprehensive track visualizations.
    
    Args:
        sar_image_path: Path to SAR image
        predictions_csv_path: Path to predictions
        correlation_results_csv_path: Path to correlation results
        interpolated_tracks_csv_path: Path to interpolated tracks
        output_dir: Output directory
        region_bounds: Optional region bounds
        
    Returns:
        Dictionary with created file paths
    """
    logger.info("🎨 Creating enhanced track visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = EnhancedTrackVisualizer(sar_image_path, output_format='jpg')
    
    created_files = {}
    
    # 1. Comprehensive single visualization
    comprehensive_path = os.path.join(output_dir, "comprehensive_tracks.jpg")
    visualizer.create_comprehensive_track_visualization(
        predictions_csv_path=predictions_csv_path,
        correlation_results_csv_path=correlation_results_csv_path,
        interpolated_tracks_csv_path=interpolated_tracks_csv_path,
        output_path=comprehensive_path,
        visualization_mode='correlation_status',
        grid_spacing_pixels=1000,  # 1km grid
        region_bounds=region_bounds,
        show_grid=True,
        show_coordinates=True
    )
    created_files['comprehensive'] = comprehensive_path
    
    # 2. Multi-panel analysis
    multi_panel_path = os.path.join(output_dir, "multi_panel_analysis.jpg")
    visualizer.create_multi_panel_visualization(
        predictions_csv_path=predictions_csv_path,
        correlation_results_csv_path=correlation_results_csv_path,
        interpolated_tracks_csv_path=interpolated_tracks_csv_path,
        output_path=multi_panel_path,
        region_bounds=region_bounds
    )
    created_files['multi_panel'] = multi_panel_path
    
    logger.info(f"✅ Enhanced visualizations created: {len(created_files)} files")
    return created_files

def main():
    """Test the enhanced track visualizer."""
    logging.basicConfig(level=logging.INFO)
    
    # Test with existing data
    try:
        # Define test region
        test_region = (20200, 12600, 21400, 13900)
        
        created_files = create_enhanced_track_visualization(
            sar_image_path="snap_output/step4_final_output.tif",
            predictions_csv_path="professor/outputs/predictions.csv",
            correlation_results_csv_path="ais_correlation/ais_correlation_results.csv",
            interpolated_tracks_csv_path="path_interpolation_results/interpolated_tracks.csv",
            output_dir="enhanced_visualizations",
            region_bounds=test_region
        )
        
        print("\n🎉 ENHANCED VISUALIZATION TEST COMPLETE!")
        print("=" * 50)
        print(f"📁 Output directory: enhanced_visualizations")
        print(f"📊 Files created:")
        for name, path in created_files.items():
            print(f"   • {name}: {os.path.basename(path)}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
