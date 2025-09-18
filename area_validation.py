#!/usr/bin/env python3
"""
Area-specific Vessel Detection Validation
Checks detections within a specific rectangular area defined by x,y coordinates
"""

import pandas as pd
import numpy as np

def main():
    # Define the area boundaries (pixel coordinates)
    # Top left - 20200, 12600
    # Bottom Left - 20200, 13900  
    # Bottom Right - 21400, 13900
    # Top Right - 21400, 12600
    
    area_bounds = {
        'x_min': 20200,  # Left boundary
        'x_max': 21400,  # Right boundary  
        'y_min': 12600,  # Top boundary
        'y_max': 13900   # Bottom boundary
    }
    
    print('🔍 AREA-SPECIFIC VESSEL DETECTION VALIDATION')
    print('=' * 60)
    print('📍 Target Area Boundaries (pixel coordinates):')
    print(f'   🔲 Top Left: ({area_bounds["x_min"]}, {area_bounds["y_min"]})')
    print(f'   🔲 Bottom Left: ({area_bounds["x_min"]}, {area_bounds["y_max"]})')
    print(f'   🔲 Bottom Right: ({area_bounds["x_max"]}, {area_bounds["y_max"]})')
    print(f'   🔲 Top Right: ({area_bounds["x_max"]}, {area_bounds["y_min"]})')
    print(f'   📏 Area size: {area_bounds["x_max"] - area_bounds["x_min"]} × {area_bounds["y_max"] - area_bounds["y_min"]} pixels')
    print()
    
    # Load predictions
    df = pd.read_csv('professor/outputs/predictions.csv')
    print(f'📊 Total detections in scene: {len(df)}')
    print()
    
    # Filter detections within the specified area using preprocessing coordinates
    print('🎯 FILTERING DETECTIONS BY AREA:')
    print('Using preprocess_column (x) and preprocess_row (y) coordinates...')
    
    area_detections = df[
        (df['preprocess_column'] >= area_bounds['x_min']) &
        (df['preprocess_column'] <= area_bounds['x_max']) &
        (df['preprocess_row'] >= area_bounds['y_min']) &
        (df['preprocess_row'] <= area_bounds['y_max'])
    ]
    
    print(f'✅ Found {len(area_detections)} detections in target area')
    print()
    
    if len(area_detections) > 0:
        print('📋 DETECTIONS IN TARGET AREA:')
        print('=' * 60)
        
        # Sort by confidence (highest first)
        area_detections = area_detections.sort_values('confidence', ascending=False)
        
        for idx, (_, detection) in enumerate(area_detections.iterrows(), 1):
            print(f'{idx:2d}. Detection ID: {detection["detect_id"]}')
            print(f'    📍 Pixel coords: ({detection["preprocess_column"]:.1f}, {detection["preprocess_row"]:.1f})')
            print(f'    🌍 Geo coords: ({detection["lat"]:.6f}, {detection["lon"]:.6f})')
            print(f'    🎯 Confidence: {detection["confidence"]:.3f}')
            
            # Show vessel attributes if available
            if pd.notna(detection.get('vessel_length_m')):
                print(f'    🚢 Length: {detection["vessel_length_m"]:.1f}m, Width: {detection["vessel_width_m"]:.1f}m')
                print(f'    ⚡ Speed: {detection["vessel_speed_k"]:.1f}kn, Type: {detection["vessel_type"]}')
                print(f'    🧭 Heading: {detection["heading_degrees"]:.1f}°')
            else:
                print(f'    ⚠️ Missing vessel attributes')
            
            # Show bounding box
            if pd.notna(detection.get('bbox_x1')):
                bbox_width = detection['bbox_x2'] - detection['bbox_x1']
                bbox_height = detection['bbox_y2'] - detection['bbox_y1']
                print(f'    📦 Bbox: ({detection["bbox_x1"]:.0f},{detection["bbox_y1"]:.0f}) to ({detection["bbox_x2"]:.0f},{detection["bbox_y2"]:.0f}) - {bbox_width:.0f}×{bbox_height:.0f}px')
            print()
        
        # Statistics for area detections
        print('📊 AREA DETECTION STATISTICS:')
        print('=' * 40)
        print(f'🎯 Total detections in area: {len(area_detections)}')
        print(f'🎯 Confidence range: {area_detections["confidence"].min():.3f} - {area_detections["confidence"].max():.3f}')
        print(f'🎯 Average confidence: {area_detections["confidence"].mean():.3f}')
        
        # Vessel attributes statistics
        area_with_attrs = area_detections.dropna(subset=['vessel_length_m'])
        if len(area_with_attrs) > 0:
            print(f'🚢 Vessels with attributes: {len(area_with_attrs)}/{len(area_detections)} ({len(area_with_attrs)/len(area_detections)*100:.1f}%)')
            print(f'🚢 Length range: {area_with_attrs["vessel_length_m"].min():.1f}m - {area_with_attrs["vessel_length_m"].max():.1f}m')
            print(f'🚢 Average length: {area_with_attrs["vessel_length_m"].mean():.1f}m')
            print(f'⚡ Speed range: {area_with_attrs["vessel_speed_k"].min():.1f}kn - {area_with_attrs["vessel_speed_k"].max():.1f}kn')
            
            # Vessel type distribution
            if 'vessel_type' in area_with_attrs.columns:
                type_counts = area_with_attrs['vessel_type'].value_counts()
                print(f'🏷️ Vessel types: {dict(type_counts)}')
        
        print()
        
        # Calculate detection density
        area_pixels = (area_bounds['x_max'] - area_bounds['x_min']) * (area_bounds['y_max'] - area_bounds['y_min'])
        area_km2 = area_pixels * (10 * 10) / (1000 * 1000)  # Assuming 10m/pixel
        detection_density = len(area_detections) / area_km2
        
        print('📏 AREA METRICS:')
        print(f'   📐 Area: {area_pixels:,} pixels ({area_km2:.1f} km²)')
        print(f'   🎯 Detection density: {detection_density:.2f} vessels/km²')
        
        # Check spacing between detections in the area
        if len(area_detections) > 1:
            min_distance = float('inf')
            max_distance = 0
            distances = []
            
            for i in range(len(area_detections)):
                for j in range(i+1, len(area_detections)):
                    row1 = area_detections.iloc[i]
                    row2 = area_detections.iloc[j]
                    
                    # Calculate pixel distance
                    dx = row1['preprocess_column'] - row2['preprocess_column']
                    dy = row1['preprocess_row'] - row2['preprocess_row']
                    pixel_distance = np.sqrt(dx**2 + dy**2)
                    meter_distance = pixel_distance * 10  # Assuming 10m/pixel
                    
                    distances.append(meter_distance)
                    min_distance = min(min_distance, meter_distance)
                    max_distance = max(max_distance, meter_distance)
            
            print(f'   📊 Closest vessels in area: {min_distance:.1f}m apart')
            print(f'   📊 Farthest vessels in area: {max_distance:.1f}m apart')
            print(f'   📊 Average separation in area: {np.mean(distances):.1f}m')
    
    else:
        print('❌ NO DETECTIONS FOUND IN TARGET AREA')
        print()
        print('💡 TROUBLESHOOTING SUGGESTIONS:')
        print('   1. Check if coordinates are in correct format (pixel coordinates)')
        print('   2. Verify area boundaries are within image bounds')
        print('   3. Consider expanding the search area')
        print('   4. Check if detections exist nearby')
        
        # Show nearest detections to the area center
        area_center_x = (area_bounds['x_min'] + area_bounds['x_max']) / 2
        area_center_y = (area_bounds['y_min'] + area_bounds['y_max']) / 2
        
        print(f'   📍 Area center: ({area_center_x}, {area_center_y})')
        
        # Find closest detections to area center
        df['distance_to_center'] = np.sqrt(
            (df['preprocess_column'] - area_center_x)**2 + 
            (df['preprocess_row'] - area_center_y)**2
        )
        
        nearest_5 = df.nsmallest(5, 'distance_to_center')
        print('   🔍 5 nearest detections to area center:')
        for _, det in nearest_5.iterrows():
            distance_pixels = det['distance_to_center']
            distance_meters = distance_pixels * 10
            print(f'      ID {det["detect_id"]}: ({det["preprocess_column"]:.0f}, {det["preprocess_row"]:.0f}) - {distance_meters:.0f}m away')

    print()
    print('✅ Area validation analysis complete!')

if __name__ == "__main__":
    main()
