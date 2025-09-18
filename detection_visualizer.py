#!/usr/bin/env python3
"""
Vessel Detection Visualizer Module

Creates visualization images with bounding boxes overlaid on SAR imagery.
Supports multiple visualization modes and output formats for Stage 1 requirements.

Features:
- Draws bounding boxes from predictions.csv on original SAR images
- Multiple visualization modes (confidence-based colors, vessel types, etc.)
- Supports both full scene and region-specific visualization
- Outputs high-quality images for analysis and reporting
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import rasterio
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VesselDetectionVisualizer:
    """
    Comprehensive vessel detection visualization module.
    
    Handles drawing bounding boxes on SAR imagery with various visualization options.
    """
    
    def __init__(self, sar_image_path: str, predictions_csv_path: str):
        """
        Initialize the visualizer.
        
        Args:
            sar_image_path: Path to the SAR GeoTIFF image
            predictions_csv_path: Path to predictions.csv file
        """
        self.sar_image_path = sar_image_path
        self.predictions_csv_path = predictions_csv_path
        self.predictions_df = None
        self.sar_image = None
        self.image_height = None
        self.image_width = None
        
        # Load data
        self._load_predictions()
        self._load_sar_image()
    
    def _load_predictions(self):
        """Load predictions from CSV file."""
        logger.info(f"📂 Loading predictions from: {self.predictions_csv_path}")
        
        if not os.path.exists(self.predictions_csv_path):
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_csv_path}")
        
        self.predictions_df = pd.read_csv(self.predictions_csv_path)
        logger.info(f"✅ Loaded {len(self.predictions_df)} detections")
        
        # Validate required columns
        required_cols = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
        missing_cols = [col for col in required_cols if col not in self.predictions_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in predictions: {missing_cols}")
        
        logger.info(f"📊 Confidence range: {self.predictions_df['confidence'].min():.3f} - {self.predictions_df['confidence'].max():.3f}")
    
    def _load_sar_image(self):
        """Load SAR image for visualization."""
        logger.info(f"🖼️ Loading SAR image from: {self.sar_image_path}")
        
        if not os.path.exists(self.sar_image_path):
            raise FileNotFoundError(f"SAR image file not found: {self.sar_image_path}")
        
        try:
            # Load using rasterio for GeoTIFF support
            with rasterio.open(self.sar_image_path) as src:
                # Read first band (VV) for visualization
                self.sar_image = src.read(1)
                self.image_height, self.image_width = self.sar_image.shape
                
                logger.info(f"✅ SAR image loaded: {self.image_width} × {self.image_height} pixels")
                logger.info(f"📊 Image value range: {self.sar_image.min()} - {self.sar_image.max()}")
                
                # Normalize image for visualization (0-255)
                self.sar_image_normalized = self._normalize_image(self.sar_image)
                
        except Exception as e:
            logger.error(f"❌ Failed to load SAR image: {e}")
            raise
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range for visualization."""
        # Use percentile-based normalization to handle outliers
        p2, p98 = np.percentile(image, (2, 98))
        image_clipped = np.clip(image, p2, p98)
        
        # Normalize to 0-255
        if p98 > p2:
            image_norm = ((image_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
        else:
            image_norm = np.zeros_like(image, dtype=np.uint8)
        
        return image_norm
    
    def create_detection_visualization(self, 
                                     output_path: str,
                                     visualization_mode: str = 'confidence',
                                     region_bounds: Optional[Tuple[int, int, int, int]] = None,
                                     min_confidence: float = 0.0,
                                     show_labels: bool = True,
                                     image_scale: float = 1.0,
                                     max_image_size: int = 8000) -> str:
        """
        Create visualization with bounding boxes overlaid on SAR image.
        
        Args:
            output_path: Path for output image
            visualization_mode: 'confidence', 'vessel_type', 'vessel_size', or 'uniform'
            region_bounds: Optional (x_min, y_min, x_max, y_max) for region-specific view
            min_confidence: Minimum confidence to display
            show_labels: Whether to show detection labels
            image_scale: Scale factor for output image
            
        Returns:
            str: Path to created visualization
        """
        logger.info(f"🎨 Creating detection visualization: {visualization_mode} mode")
        
        # Filter predictions by confidence
        filtered_preds = self.predictions_df[self.predictions_df['confidence'] >= min_confidence].copy()
        logger.info(f"📊 Showing {len(filtered_preds)} detections (min_conf={min_confidence})")
        
        # Prepare image with memory optimization
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            display_image = self.sar_image_normalized[y_min:y_max, x_min:x_max]
            # Adjust bounding boxes for region
            filtered_preds = self._filter_region_detections(filtered_preds, region_bounds)
            offset_x, offset_y = x_min, y_min
        else:
            # For full scene, downsample if too large to prevent memory issues
            display_image = self.sar_image_normalized.copy()
            offset_x, offset_y = 0, 0
            
            # Downsample large images to prevent memory crashes
            if max(display_image.shape) > max_image_size:
                scale_factor = max_image_size / max(display_image.shape)
                new_height = int(display_image.shape[0] * scale_factor)
                new_width = int(display_image.shape[1] * scale_factor)
                
                logger.info(f"🔧 Downsampling image: {display_image.shape} → ({new_height}, {new_width})")
                display_image = cv2.resize(display_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Scale bounding boxes accordingly
                filtered_preds = filtered_preds.copy()
                for col in ['bbox_x1', 'bbox_x2']:
                    filtered_preds[col] = filtered_preds[col] * scale_factor
                for col in ['bbox_y1', 'bbox_y2']:
                    filtered_preds[col] = filtered_preds[col] * scale_factor
                
                logger.info(f"📊 Scaled {len(filtered_preds)} bounding boxes by factor {scale_factor:.3f}")
        
        # Create figure with memory-safe sizing
        max_fig_size = 20  # Maximum figure size in inches
        fig_height = min(display_image.shape[0] * image_scale / 100, max_fig_size)
        fig_width = min(display_image.shape[1] * image_scale / 100, max_fig_size)
        
        # Ensure reasonable figure size
        if fig_width < 8:
            fig_width = 8
        if fig_height < 6:
            fig_height = 6
            
        logger.info(f"📐 Creating figure: {fig_width:.1f} × {fig_height:.1f} inches")
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(display_image, cmap='gray', aspect='equal')
        ax.set_xlim(0, display_image.shape[1])
        ax.set_ylim(display_image.shape[0], 0)  # Flip Y axis for image coordinates
        
        # Draw bounding boxes
        colors = self._get_visualization_colors(filtered_preds, visualization_mode)
        
        for idx, (_, detection) in enumerate(filtered_preds.iterrows()):
            # Get bounding box coordinates (adjusted for region if needed)
            x1 = detection['bbox_x1'] - offset_x
            y1 = detection['bbox_y1'] - offset_y
            x2 = detection['bbox_x2'] - offset_x
            y2 = detection['bbox_y2'] - offset_y
            
            # Ensure bbox is within image bounds
            x1 = max(0, min(x1, display_image.shape[1]))
            y1 = max(0, min(y1, display_image.shape[0]))
            x2 = max(0, min(x2, display_image.shape[1]))
            y2 = max(0, min(y2, display_image.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
            
            # Draw bounding box
            width = x2 - x1
            height = y2 - y1
            color = colors[idx]
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label if requested
            if show_labels:
                label = self._create_detection_label(detection, visualization_mode)
                ax.text(x1, y1-5, label, color=color, fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Set title and labels
        title = f"Vessel Detections ({visualization_mode.title()} Mode) - {len(filtered_preds)} vessels"
        if region_bounds:
            title += f" in region ({region_bounds[0]},{region_bounds[1]}) to ({region_bounds[2]},{region_bounds[3]})"
        
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        
        # Add legend for visualization mode
        self._add_legend(ax, visualization_mode, filtered_preds)
        
        # Save figure with memory optimization
        plt.tight_layout()
        
        # Use lower DPI for large images to save memory
        dpi = 150 if max(display_image.shape) > 4000 else 300
        logger.info(f"💾 Saving with DPI: {dpi}")
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"✅ Visualization saved to: {output_path}")
        return output_path
    
    def _filter_region_detections(self, df: pd.DataFrame, region_bounds: Tuple[int, int, int, int]) -> pd.DataFrame:
        """Filter detections to those within the specified region."""
        x_min, y_min, x_max, y_max = region_bounds
        
        # Filter based on bounding box centers
        center_x = (df['bbox_x1'] + df['bbox_x2']) / 2
        center_y = (df['bbox_y1'] + df['bbox_y2']) / 2
        
        region_mask = (
            (center_x >= x_min) & (center_x <= x_max) &
            (center_y >= y_min) & (center_y <= y_max)
        )
        
        return df[region_mask].copy()
    
    def _get_visualization_colors(self, df: pd.DataFrame, mode: str) -> List[str]:
        """Get colors for bounding boxes based on visualization mode."""
        if mode == 'confidence':
            # Color by confidence: red (low) to green (high)
            confidences = df['confidence'].values
            normalized_conf = (confidences - confidences.min()) / (confidences.max() - confidences.min() + 1e-8)
            colors = [plt.cm.RdYlGn(conf) for conf in normalized_conf]
            
        elif mode == 'vessel_type':
            # Color by vessel type
            type_colors = {'fishing': 'blue', 'cargo': 'red', 'other': 'yellow'}
            colors = [type_colors.get(vtype, 'white') for vtype in df.get('vessel_type', 'other')]
            
        elif mode == 'vessel_size':
            # Color by vessel length
            if 'vessel_length_m' in df.columns:
                lengths = df['vessel_length_m'].fillna(0).values
                normalized_length = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-8)
                colors = [plt.cm.viridis(length) for length in normalized_length]
            else:
                colors = ['cyan'] * len(df)
                
        else:  # uniform
            colors = ['lime'] * len(df)
        
        return colors
    
    def _create_detection_label(self, detection: pd.Series, mode: str) -> str:
        """Create label text for detection based on visualization mode."""
        base_label = f"ID:{detection['detect_id']}"
        
        if mode == 'confidence':
            return f"{base_label}\nConf:{detection['confidence']:.2f}"
        elif mode == 'vessel_type':
            vessel_type = detection.get('vessel_type', 'unknown')
            return f"{base_label}\n{vessel_type.title()}"
        elif mode == 'vessel_size':
            if pd.notna(detection.get('vessel_length_m')):
                length = detection['vessel_length_m']
                return f"{base_label}\n{length:.0f}m"
            else:
                return base_label
        else:
            return base_label
    
    def _add_legend(self, ax, mode: str, df: pd.DataFrame):
        """Add legend to the visualization."""
        if mode == 'confidence':
            # Add colorbar for confidence
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                     norm=plt.Normalize(vmin=df['confidence'].min(), 
                                                       vmax=df['confidence'].max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Confidence Score', rotation=270, labelpad=20)
            
        elif mode == 'vessel_type':
            # Add legend for vessel types
            unique_types = df['vessel_type'].dropna().unique() if 'vessel_type' in df.columns else ['unknown']
            type_colors = {'fishing': 'blue', 'cargo': 'red', 'other': 'yellow', 'unknown': 'white'}
            
            legend_elements = [patches.Patch(color=type_colors.get(vtype, 'white'), label=vtype.title()) 
                             for vtype in unique_types]
            ax.legend(handles=legend_elements, loc='upper right')
            
        elif mode == 'vessel_size':
            if 'vessel_length_m' in df.columns and df['vessel_length_m'].notna().any():
                # Add colorbar for vessel size
                lengths = df['vessel_length_m'].dropna()
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                         norm=plt.Normalize(vmin=lengths.min(), 
                                                           vmax=lengths.max()))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
                cbar.set_label('Vessel Length (m)', rotation=270, labelpad=20)
    
    def create_full_scene_visualization(self, output_dir: str, modes: List[str] = None) -> List[str]:
        """
        Create full scene visualizations in multiple modes.
        
        Args:
            output_dir: Directory for output images
            modes: List of visualization modes to create
            
        Returns:
            List of created file paths
        """
        if modes is None:
            modes = ['confidence', 'vessel_type', 'vessel_size', 'uniform']
        
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        for mode in modes:
            output_file = os.path.join(output_dir, f"full_scene_{mode}.png")
            created_file = self.create_detection_visualization(
                output_path=output_file,
                visualization_mode=mode,
                show_labels=False,  # Disable labels for full scene to save memory
                image_scale=0.3,    # Further scale down for full scene
                max_image_size=4000  # Limit image size to prevent memory issues
            )
            created_files.append(created_file)
        
        return created_files
    
    def create_region_visualization(self, 
                                  region_bounds: Tuple[int, int, int, int],
                                  output_dir: str,
                                  region_name: str = "region") -> str:
        """
        Create high-resolution visualization of a specific region.
        
        Args:
            region_bounds: (x_min, y_min, x_max, y_max) pixel coordinates
            output_dir: Directory for output images
            region_name: Name for the region (used in filename)
            
        Returns:
            Path to created visualization
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{region_name}_detailed.png")
        
        return self.create_detection_visualization(
            output_path=output_file,
            visualization_mode='confidence',
            region_bounds=region_bounds,
            show_labels=True,
            image_scale=1.0  # Full resolution for region
        )
    
    def create_detection_summary(self, output_dir: str) -> str:
        """Create a summary visualization with statistics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confidence distribution
        ax1.hist(self.predictions_df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detection Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Vessel size distribution (if available)
        if 'vessel_length_m' in self.predictions_df.columns:
            lengths = self.predictions_df['vessel_length_m'].dropna()
            if len(lengths) > 0:
                ax2.hist(lengths, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_xlabel('Vessel Length (m)')
                ax2.set_ylabel('Number of Vessels')
                ax2.set_title('Vessel Size Distribution')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No vessel size data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Vessel Size Distribution (No Data)')
        else:
            ax2.text(0.5, 0.5, 'No vessel size data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Vessel Size Distribution (No Data)')
        
        # 3. Spatial distribution
        ax3.scatter(self.predictions_df['lon'], self.predictions_df['lat'], 
                   c=self.predictions_df['confidence'], cmap='RdYlGn', alpha=0.7, s=50)
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title('Spatial Distribution of Detections')
        ax3.grid(True, alpha=0.3)
        
        # 4. Vessel type distribution (if available)
        if 'vessel_type' in self.predictions_df.columns:
            type_counts = self.predictions_df['vessel_type'].value_counts()
            if len(type_counts) > 0:
                ax4.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Vessel Type Distribution')
            else:
                ax4.text(0.5, 0.5, 'No vessel type data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Vessel Type Distribution (No Data)')
        else:
            ax4.text(0.5, 0.5, 'No vessel type data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Vessel Type Distribution (No Data)')
        
        plt.tight_layout()
        
        # Add overall statistics
        stats_text = f"""
Detection Summary:
• Total Detections: {len(self.predictions_df)}
• Confidence Range: {self.predictions_df['confidence'].min():.3f} - {self.predictions_df['confidence'].max():.3f}
• Average Confidence: {self.predictions_df['confidence'].mean():.3f}
• Image Dimensions: {self.image_width} × {self.image_height} pixels
        """
        
        fig.suptitle('Vessel Detection Analysis Summary', fontsize=16, y=0.98)
        fig.text(0.02, 0.02, stats_text.strip(), fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        summary_path = os.path.join(output_dir, 'detection_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ Summary visualization saved to: {summary_path}")
        return summary_path
    
    def create_opencv_visualization(self, output_path: str, 
                                  region_bounds: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Create visualization using OpenCV for better performance with large images.
        
        Args:
            output_path: Path for output image
            region_bounds: Optional region bounds for cropping
            
        Returns:
            Path to created visualization
        """
        logger.info("🎨 Creating OpenCV-based visualization")
        
        # Prepare image
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            display_image = self.sar_image_normalized[y_min:y_max, x_min:x_max].copy()
            filtered_preds = self._filter_region_detections(self.predictions_df, region_bounds)
            offset_x, offset_y = x_min, y_min
        else:
            display_image = self.sar_image_normalized.copy()
            filtered_preds = self.predictions_df.copy()
            offset_x, offset_y = 0, 0
        
        # Convert to BGR for OpenCV
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes
        for _, detection in filtered_preds.iterrows():
            # Get bounding box coordinates
            x1 = int(detection['bbox_x1'] - offset_x)
            y1 = int(detection['bbox_y1'] - offset_y)
            x2 = int(detection['bbox_x2'] - offset_x)
            y2 = int(detection['bbox_y2'] - offset_y)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, display_image.shape[1]))
            y1 = max(0, min(y1, display_image.shape[0]))
            x2 = max(0, min(x2, display_image.shape[1]))
            y2 = max(0, min(y2, display_image.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Color based on confidence (green for high, red for low)
            confidence = detection['confidence']
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow
            elif confidence >= 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"ID:{detection['detect_id']} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(display_image, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(display_image, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save image
        cv2.imwrite(output_path, display_image)
        logger.info(f"✅ OpenCV visualization saved to: {output_path}")
        return output_path

def main():
    """Example usage of the visualizer."""
    # Define paths
    sar_image_path = "snap_output/step4_final_output.tif"
    predictions_csv_path = "professor/outputs/predictions.csv"
    output_dir = "visualization_outputs"
    
    try:
        # Create visualizer
        visualizer = VesselDetectionVisualizer(sar_image_path, predictions_csv_path)
        
        # Create full scene visualizations
        logger.info("🎨 Creating full scene visualizations...")
        full_scene_files = visualizer.create_full_scene_visualization(output_dir)
        
        # Create region-specific visualization for your test area
        test_region = (20200, 12600, 21400, 13900)  # Your specified area
        logger.info("🎯 Creating test region visualization...")
        region_file = visualizer.create_region_visualization(
            region_bounds=test_region,
            output_dir=output_dir,
            region_name="test_area"
        )
        
        # Create summary
        logger.info("📊 Creating detection summary...")
        summary_file = visualizer.create_detection_summary(output_dir)
        
        # Create OpenCV version for large images
        logger.info("🖼️ Creating OpenCV visualization...")
        opencv_file = visualizer.create_opencv_visualization(
            os.path.join(output_dir, "opencv_full_scene.png")
        )
        
        print("\n🎉 Visualization Complete!")
        print("=" * 50)
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Files created:")
        for file_path in full_scene_files + [region_file, summary_file, opencv_file]:
            print(f"   • {os.path.basename(file_path)}")
        
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        raise

if __name__ == "__main__":
    main()
