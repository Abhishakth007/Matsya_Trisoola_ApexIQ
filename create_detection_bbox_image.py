#!/usr/bin/env python3
"""
Create Detection Bounding Box Image

Creates a visual image showing all predicted vessels with bounding boxes
on the SAR image for manual inspection and verification.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import rasterio
import logging
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_sar_image(image_path):
    """Load and prepare SAR image for visualization."""
    logger.info(f"📖 Loading SAR image: {image_path}")
    
    try:
        with rasterio.open(image_path) as src:
            # Read the image data
            image_data = src.read()
            
            # Handle different band configurations
            if image_data.shape[0] == 1:
                # Single band - use as grayscale
                image = image_data[0]
            elif image_data.shape[0] >= 3:
                # Multi-band - use first 3 bands as RGB
                image = np.transpose(image_data[:3], (1, 2, 0))
            else:
                # Two bands - use first band
                image = image_data[0]
            
            logger.info(f"✅ Image loaded: shape {image.shape}, dtype {image.dtype}")
            return image, src.bounds, src.transform
            
    except Exception as e:
        logger.error(f"❌ Error loading SAR image: {e}")
        return None, None, None

def normalize_image_for_display(image):
    """Normalize image data for display."""
    if image is None:
        return None
    
    # Handle different image shapes
    if len(image.shape) == 3:
        # RGB image
        normalized = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[2]):
            band = image[:, :, i]
            # Normalize each band separately
            band_min, band_max = np.percentile(band, [2, 98])
            band_norm = np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255)
            normalized[:, :, i] = band_norm.astype(np.uint8)
        return normalized
    else:
        # Grayscale image
        # Use percentile normalization to handle outliers
        img_min, img_max = np.percentile(image, [2, 98])
        normalized = np.clip((image - img_min) / (img_max - img_min) * 255, 0, 255)
        return normalized.astype(np.uint8)

def load_predictions(predictions_path):
    """Load vessel predictions."""
    logger.info(f"📊 Loading predictions: {predictions_path}")
    
    try:
        predictions_df = pd.read_csv(predictions_path)
        logger.info(f"✅ Loaded {len(predictions_df)} predictions")
        
        # Display prediction info
        logger.info(f"📈 Prediction confidence range: {predictions_df['confidence'].min():.3f} - {predictions_df['confidence'].max():.3f}")
        logger.info(f"📍 Coordinate ranges:")
        logger.info(f"   Bbox X: {predictions_df['bbox_x1'].min():.0f} - {predictions_df['bbox_x2'].max():.0f}")
        logger.info(f"   Bbox Y: {predictions_df['bbox_y1'].min():.0f} - {predictions_df['bbox_y2'].max():.0f}")
        logger.info(f"   Lat: {predictions_df['lat'].min():.6f} - {predictions_df['lat'].max():.6f}")
        logger.info(f"   Lon: {predictions_df['lon'].min():.6f} - {predictions_df['lon'].max():.6f}")
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"❌ Error loading predictions: {e}")
        return None

def create_bbox_visualization(image, predictions_df, output_path, max_size=8000):
    """Create bounding box visualization."""
    logger.info(f"🎨 Creating bounding box visualization...")
    
    if image is None or predictions_df is None:
        logger.error("❌ Cannot create visualization - missing image or predictions")
        return False
    
    # Normalize image for display
    display_image = normalize_image_for_display(image)
    
    # Handle image size for memory efficiency
    height, width = display_image.shape[:2]
    scale_factor = 1.0
    
    if max(height, width) > max_size:
        scale_factor = max_size / max(height, width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        if len(display_image.shape) == 3:
            display_image = cv2.resize(display_image, (new_width, new_height))
        else:
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        logger.info(f"📏 Resized image: {width}x{height} → {new_width}x{new_height} (scale: {scale_factor:.3f})")
    
    # Create figure
    fig_width = min(20, max(12, new_width / 500)) if 'new_width' in locals() else 16
    fig_height = min(20, max(8, new_height / 500)) if 'new_height' in locals() else 12
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Display image
    if len(display_image.shape) == 3:
        ax.imshow(display_image)
    else:
        ax.imshow(display_image, cmap='gray')
    
    # Add bounding boxes
    colors = ['red', 'yellow', 'cyan', 'magenta', 'green', 'orange', 'purple', 'pink']
    
    # Sort predictions by confidence (highest first)
    sorted_predictions = predictions_df.sort_values('confidence', ascending=False)
    
    logger.info(f"🔲 Adding {len(sorted_predictions)} bounding boxes...")
    
    for idx, (_, pred) in enumerate(sorted_predictions.iterrows()):
        # Get bounding box coordinates (they're already in pixel coordinates)
        x1 = pred['bbox_x1'] * scale_factor
        y1 = pred['bbox_y1'] * scale_factor  
        x2 = pred['bbox_x2'] * scale_factor
        y2 = pred['bbox_y2'] * scale_factor
        confidence = pred['confidence']
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Choose color based on confidence
        if confidence >= 0.8:
            color = 'red'
            alpha = 0.8
        elif confidence >= 0.6:
            color = 'yellow'
            alpha = 0.7
        elif confidence >= 0.4:
            color = 'cyan'
            alpha = 0.6
        else:
            color = 'magenta'
            alpha = 0.5
        
        # Create rectangle
        rect = Rectangle((x1, y1), box_width, box_height, 
                        linewidth=2, edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(rect)
        
        # Add a small dot at the center of the detection
        center_x = x1 + box_width / 2
        center_y = y1 + box_height / 2
        ax.plot(center_x, center_y, 'o', color=color, markersize=4, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    
    # Customize plot
    ax.set_title(f'Vessel Detections: {len(predictions_df)} vessels detected\n'
                f'Confidence range: {predictions_df["confidence"].min():.3f} - {predictions_df["confidence"].max():.3f}',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='High confidence (≥0.8)'),
        plt.Line2D([0], [0], color='yellow', lw=2, label='Good confidence (0.6-0.8)'),
        plt.Line2D([0], [0], color='cyan', lw=2, label='Medium confidence (0.4-0.6)'),
        plt.Line2D([0], [0], color='magenta', lw=2, label='Low confidence (<0.4)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Save image
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Visualization saved: {output_path}")
    return True

def main():
    """Main function."""
    logger.info("🚀 Creating Detection Bounding Box Visualization")
    
    # File paths
    sar_image_path = "data/aoi_test_vv_vh.tif"  # Try the test image first
    predictions_path = "maritime_analysis_results/maritime_analysis_20250917_162029/predictions_raw.csv"
    output_path = "detection_bbox_visualization.png"
    
    # Alternative SAR image paths to try
    alternative_paths = [
        "snap_output/step4_final_output.tif",
        "data/aoi_test_vv_vh.tif",
        "data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE/measurement/s1a-iw-grd-vv-20230620t230642-20230620t230707-049076-05e6cc-001.tiff"
    ]
    
    # Try to find a working SAR image
    image, bounds, transform = None, None, None
    for path in alternative_paths:
        if os.path.exists(path):
            logger.info(f"📁 Trying SAR image: {path}")
            image, bounds, transform = load_sar_image(path)
            if image is not None:
                sar_image_path = path
                break
        else:
            logger.warning(f"⚠️ SAR image not found: {path}")
    
    if image is None:
        logger.error("❌ No SAR image could be loaded")
        return False
    
    # Load predictions
    predictions_df = load_predictions(predictions_path)
    if predictions_df is None:
        logger.error("❌ No predictions could be loaded")
        return False
    
    # Create visualization
    success = create_bbox_visualization(image, predictions_df, output_path)
    
    if success:
        logger.info(f"🎉 Success! Detection visualization created: {output_path}")
        logger.info(f"📊 Summary:")
        logger.info(f"   Total detections: {len(predictions_df)}")
        logger.info(f"   High confidence (≥0.8): {len(predictions_df[predictions_df['confidence'] >= 0.8])}")
        logger.info(f"   Good confidence (0.6-0.8): {len(predictions_df[(predictions_df['confidence'] >= 0.6) & (predictions_df['confidence'] < 0.8)])}")
        logger.info(f"   Medium confidence (0.4-0.6): {len(predictions_df[(predictions_df['confidence'] >= 0.4) & (predictions_df['confidence'] < 0.6)])}")
        logger.info(f"   Low confidence (<0.4): {len(predictions_df[predictions_df['confidence'] < 0.4])}")
        
        return True
    else:
        logger.error("❌ Failed to create visualization")
        return False

if __name__ == "__main__":
    main()
