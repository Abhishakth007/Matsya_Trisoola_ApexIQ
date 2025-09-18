#!/usr/bin/env python3
"""
Simple Bounding Box Visualizer
Memory-efficient visualization for specific regions
"""

import cv2
import numpy as np
import pandas as pd
import rasterio
import os

def create_region_visualization(sar_image_path, predictions_csv_path, region_bounds, output_path):
    """
    Create visualization for a specific region with bounding boxes.
    
    Args:
        sar_image_path: Path to SAR GeoTIFF
        predictions_csv_path: Path to predictions.csv
        region_bounds: (x_min, y_min, x_max, y_max) tuple
        output_path: Output image path
    """
    print(f"🎨 Creating region visualization...")
    print(f"📍 Region: {region_bounds}")
    
    # Load predictions
    df = pd.read_csv(predictions_csv_path)
    print(f"📊 Total detections: {len(df)}")
    
    # Filter detections in region
    x_min, y_min, x_max, y_max = region_bounds
    
    # Filter by bounding box centers
    center_x = (df['bbox_x1'] + df['bbox_x2']) / 2
    center_y = (df['bbox_y1'] + df['bbox_y2']) / 2
    
    region_detections = df[
        (center_x >= x_min) & (center_x <= x_max) &
        (center_y >= y_min) & (center_y <= y_max)
    ].copy()
    
    print(f"🎯 Detections in region: {len(region_detections)}")
    
    # Load SAR image region
    with rasterio.open(sar_image_path) as src:
        # Read the specific region
        window = rasterio.windows.Window(x_min, y_min, x_max - x_min, y_max - y_min)
        region_image = src.read(1, window=window)
    
    print(f"🖼️ Region image shape: {region_image.shape}")
    
    # Normalize image to 0-255
    p2, p98 = np.percentile(region_image, (2, 98))
    region_image = np.clip(region_image, p2, p98)
    if p98 > p2:
        region_image = ((region_image - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        region_image = np.zeros_like(region_image, dtype=np.uint8)
    
    # Convert to BGR for OpenCV
    region_image_bgr = cv2.cvtColor(region_image, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding boxes
    for _, detection in region_detections.iterrows():
        # Adjust coordinates relative to region
        x1 = int(detection['bbox_x1'] - x_min)
        y1 = int(detection['bbox_y1'] - y_min)
        x2 = int(detection['bbox_x2'] - x_min)
        y2 = int(detection['bbox_y2'] - y_min)
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, region_image.shape[1]))
        y1 = max(0, min(y1, region_image.shape[0]))
        x2 = max(0, min(x2, region_image.shape[1]))
        y2 = max(0, min(y2, region_image.shape[0]))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Color based on confidence
        confidence = detection['confidence']
        if confidence >= 0.8:
            color = (0, 255, 0)      # Green - high confidence
        elif confidence >= 0.6:
            color = (0, 255, 255)    # Yellow - medium confidence
        elif confidence >= 0.4:
            color = (0, 165, 255)    # Orange - low-medium confidence
        else:
            color = (0, 0, 255)      # Red - low confidence
        
        # Draw bounding box
        cv2.rectangle(region_image_bgr, (x1, y1), (x2, y2), color, 3)
        
        # Add detection label
        label = f"ID:{detection['detect_id']} ({confidence:.2f})"
        if pd.notna(detection.get('vessel_length_m')):
            label += f" {detection['vessel_length_m']:.0f}m"
        
        # Background for text
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(region_image_bgr, (x1, y1-label_size[1]-15), 
                     (x1+label_size[0]+10, y1-5), (0, 0, 0), -1)
        
        # Text
        cv2.putText(region_image_bgr, label, (x1+5, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add title and region info
    title = f"Vessel Detections: {len(region_detections)} vessels in region"
    region_info = f"Region: ({x_min},{y_min}) to ({x_max},{y_max})"
    
    # Add title to image
    cv2.putText(region_image_bgr, title, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(region_image_bgr, region_info, (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite(output_path, region_image_bgr)
    print(f"✅ Visualization saved to: {output_path}")
    
    return output_path, len(region_detections)

def create_detection_overview(sar_image_path, predictions_csv_path, output_path, downsample_factor=10):
    """
    Create a heavily downsampled overview of all detections.
    
    Args:
        sar_image_path: Path to SAR GeoTIFF
        predictions_csv_path: Path to predictions.csv
        output_path: Output image path
        downsample_factor: Factor to downsample image (default 10x)
    """
    print(f"🌍 Creating detection overview (downsampled {downsample_factor}x)...")
    
    # Load predictions
    df = pd.read_csv(predictions_csv_path)
    print(f"📊 Total detections: {len(df)}")
    
    # Load and downsample SAR image
    with rasterio.open(sar_image_path) as src:
        # Read full image
        full_image = src.read(1)
        original_shape = full_image.shape
        
        # Downsample
        new_height = original_shape[0] // downsample_factor
        new_width = original_shape[1] // downsample_factor
        
        downsampled = cv2.resize(full_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    print(f"🖼️ Downsampled: {original_shape} → {downsampled.shape}")
    
    # Normalize image
    p2, p98 = np.percentile(downsampled, (2, 98))
    downsampled = np.clip(downsampled, p2, p98)
    if p98 > p2:
        downsampled = ((downsampled - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        downsampled = np.zeros_like(downsampled, dtype=np.uint8)
    
    # Convert to BGR
    overview_image = cv2.cvtColor(downsampled, cv2.COLOR_GRAY2BGR)
    
    # Draw detection points (scaled down)
    for _, detection in df.iterrows():
        # Scale coordinates
        center_x = int((detection['bbox_x1'] + detection['bbox_x2']) / 2 / downsample_factor)
        center_y = int((detection['bbox_y1'] + detection['bbox_y2']) / 2 / downsample_factor)
        
        # Ensure within bounds
        if 0 <= center_x < new_width and 0 <= center_y < new_height:
            # Color by confidence
            confidence = detection['confidence']
            if confidence >= 0.8:
                color = (0, 255, 0)      # Green
            elif confidence >= 0.6:
                color = (0, 255, 255)    # Yellow
            else:
                color = (0, 0, 255)      # Red
            
            # Draw circle for detection
            cv2.circle(overview_image, (center_x, center_y), 3, color, -1)
            cv2.circle(overview_image, (center_x, center_y), 5, color, 2)
    
    # Add title
    title = f"Detection Overview: {len(df)} vessels detected"
    cv2.putText(overview_image, title, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(output_path, overview_image)
    print(f"✅ Overview saved to: {output_path}")
    
    return output_path

def main():
    """Main function to create visualizations."""
    print("🎨 VESSEL DETECTION VISUALIZER")
    print("=" * 50)
    
    # Paths
    sar_image_path = "snap_output/step4_final_output.tif"
    predictions_csv_path = "professor/outputs/predictions.csv"
    output_dir = "visualization_outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Create overview of all detections
        overview_path = os.path.join(output_dir, "detection_overview.png")
        create_detection_overview(sar_image_path, predictions_csv_path, overview_path)
        
        # 2. Create detailed view of your test region
        test_region = (20200, 12600, 21400, 13900)
        region_path = os.path.join(output_dir, "test_region_detections.png")
        region_file, detection_count = create_region_visualization(
            sar_image_path, predictions_csv_path, test_region, region_path
        )
        
        print("\n🎉 VISUALIZATION COMPLETE!")
        print("=" * 50)
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Files created:")
        print(f"   • detection_overview.png - Full scene overview")
        print(f"   • test_region_detections.png - Your test area ({detection_count} vessels)")
        print()
        print(f"🎯 Test Area Results:")
        print(f"   📍 Region: (20200,12600) to (21400,13900)")
        print(f"   🚢 Detected vessels: {detection_count}")
        print(f"   📊 Expected vessels: ~25")
        print(f"   📈 Detection rate: {detection_count/25*100:.1f}%")
        
        if detection_count < 20:
            print()
            print("⚠️ UNDER-DETECTION CONFIRMED:")
            print("   • Visual evidence of missing vessels")
            print("   • Confidence threshold may still be too high")
            print("   • Consider lowering conf_threshold to 0.10-0.15")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
