#!/usr/bin/env python3
"""
Duplicate Detection Analysis for Vessel Predictions
Analyzes the predictions.csv file for various types of duplicates
"""

import pandas as pd
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def main():
    # Load predictions
    df = pd.read_csv('professor/outputs/predictions.csv')
    print('🔍 DUPLICATE DETECTION ANALYSIS')
    print('=' * 50)
    print(f'📊 Total detections: {len(df)}')
    print()

    # 1. Check for exact coordinate duplicates
    print('1️⃣ EXACT COORDINATE DUPLICATES:')
    coord_duplicates = df.duplicated(subset=['lat', 'lon'], keep=False)
    if coord_duplicates.any():
        print(f'   ❌ Found {coord_duplicates.sum()} exact coordinate duplicates')
        dup_coords = df[coord_duplicates][['detect_id', 'lat', 'lon', 'confidence']].sort_values('confidence', ascending=False)
        print(dup_coords.to_string(index=False))
    else:
        print('   ✅ No exact coordinate duplicates found')
    print()

    # 2. Check for near-duplicate coordinates (within ~10 meters)
    print('2️⃣ NEAR-DUPLICATE COORDINATES (< 0.0001° ≈ 10m):')
    close_detections = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            lat_diff = abs(df.iloc[i]['lat'] - df.iloc[j]['lat'])
            lon_diff = abs(df.iloc[i]['lon'] - df.iloc[j]['lon'])
            if lat_diff < 0.0001 and lon_diff < 0.0001:
                distance_approx = np.sqrt(lat_diff**2 + lon_diff**2) * 111000  # rough meters
                close_detections.append({
                    'id1': df.iloc[i]['detect_id'],
                    'id2': df.iloc[j]['detect_id'],
                    'lat1': df.iloc[i]['lat'],
                    'lon1': df.iloc[i]['lon'],
                    'lat2': df.iloc[j]['lat'],
                    'lon2': df.iloc[j]['lon'],
                    'conf1': df.iloc[i]['confidence'],
                    'conf2': df.iloc[j]['confidence'],
                    'distance_m': distance_approx
                })

    if close_detections:
        print(f'   ⚠️ Found {len(close_detections)} pairs of close detections:')
        for det in close_detections:
            print(f'   ID {det["id1"]} (conf={det["conf1"]:.3f}) & ID {det["id2"]} (conf={det["conf2"]:.3f}) - {det["distance_m"]:.1f}m apart')
    else:
        print('   ✅ No close coordinate duplicates found')
    print()

    # 3. Check bbox overlaps
    print('3️⃣ BOUNDING BOX OVERLAPS:')
    high_overlaps = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if pd.notna(df.iloc[i]['bbox_x1']) and pd.notna(df.iloc[j]['bbox_x1']):
                box1 = [df.iloc[i]['bbox_x1'], df.iloc[i]['bbox_y1'], df.iloc[i]['bbox_x2'], df.iloc[i]['bbox_y2']]
                box2 = [df.iloc[j]['bbox_x1'], df.iloc[j]['bbox_y1'], df.iloc[j]['bbox_x2'], df.iloc[j]['bbox_y2']]
                iou = calculate_iou(box1, box2)
                if iou > 0.3:  # High overlap threshold
                    high_overlaps.append({
                        'id1': df.iloc[i]['detect_id'],
                        'id2': df.iloc[j]['detect_id'],
                        'conf1': df.iloc[i]['confidence'],
                        'conf2': df.iloc[j]['confidence'],
                        'iou': iou
                    })

    if high_overlaps:
        print(f'   ⚠️ Found {len(high_overlaps)} pairs with high bbox overlap (IoU > 0.3):')
        for overlap in high_overlaps:
            print(f'   ID {overlap["id1"]} (conf={overlap["conf1"]:.3f}) & ID {overlap["id2"]} (conf={overlap["conf2"]:.3f}) - IoU={overlap["iou"]:.3f}')
    else:
        print('   ✅ No high bbox overlaps found (IoU > 0.3)')
    print()

    # 4. Check preprocessing coordinate duplicates
    print('4️⃣ PREPROCESSING COORDINATE DUPLICATES:')
    preproc_duplicates = df.duplicated(subset=['preprocess_row', 'preprocess_column'], keep=False)
    if preproc_duplicates.any():
        print(f'   ❌ Found {preproc_duplicates.sum()} preprocessing coordinate duplicates')
        dup_preproc = df[preproc_duplicates][['detect_id', 'preprocess_row', 'preprocess_column', 'confidence']].sort_values('confidence', ascending=False)
        print(dup_preproc.to_string(index=False))
    else:
        print('   ✅ No preprocessing coordinate duplicates found')
    print()

    # 5. Check vessel attribute similarity (for vessels with attributes)
    print('5️⃣ SIMILAR VESSEL ATTRIBUTES:')
    df_with_attrs = df.dropna(subset=['vessel_length_m', 'vessel_width_m'])
    if len(df_with_attrs) > 1:
        similar_vessels = []
        for i in range(len(df_with_attrs)):
            for j in range(i+1, len(df_with_attrs)):
                row1 = df_with_attrs.iloc[i]
                row2 = df_with_attrs.iloc[j]
                
                # Check if vessels are very similar in size and close in location
                length_diff = abs(row1['vessel_length_m'] - row2['vessel_length_m'])
                width_diff = abs(row1['vessel_width_m'] - row2['vessel_width_m'])
                lat_diff = abs(row1['lat'] - row2['lat'])
                lon_diff = abs(row1['lon'] - row2['lon'])
                
                if (length_diff < 2.0 and width_diff < 1.0 and 
                    lat_diff < 0.001 and lon_diff < 0.001):  # Very similar vessels close together
                    similar_vessels.append({
                        'id1': row1['detect_id'],
                        'id2': row2['detect_id'],
                        'conf1': row1['confidence'],
                        'conf2': row2['confidence'],
                        'length1': row1['vessel_length_m'],
                        'length2': row2['vessel_length_m'],
                        'distance_deg': np.sqrt(lat_diff**2 + lon_diff**2)
                    })
        
        if similar_vessels:
            print(f'   ⚠️ Found {len(similar_vessels)} pairs of similar vessels close together:')
            for sim in similar_vessels:
                print(f'   ID {sim["id1"]} (L={sim["length1"]:.1f}m, conf={sim["conf1"]:.3f}) & ID {sim["id2"]} (L={sim["length2"]:.1f}m, conf={sim["conf2"]:.3f}) - {sim["distance_deg"]*111000:.1f}m apart')
        else:
            print('   ✅ No similar vessels found close together')
    else:
        print('   ℹ️ Insufficient vessels with attributes for comparison')
    print()

    # 6. Check for missing vessel attributes (could indicate failed postprocessing)
    print('6️⃣ MISSING VESSEL ATTRIBUTES:')
    missing_attrs = df[df['vessel_length_m'].isna()]
    if len(missing_attrs) > 0:
        print(f'   ⚠️ Found {len(missing_attrs)} detections without vessel attributes:')
        print(f'   Detection IDs: {missing_attrs["detect_id"].tolist()}')
        print(f'   Confidence range: {missing_attrs["confidence"].min():.3f} - {missing_attrs["confidence"].max():.3f}')
    else:
        print('   ✅ All detections have vessel attributes')
    print()

    # 7. Summary statistics
    print('📊 SUMMARY STATISTICS:')
    print(f'   🎯 Total detections: {len(df)}')
    print(f'   🎯 Detections with vessel attributes: {len(df_with_attrs)}')
    print(f'   🎯 Confidence range: {df["confidence"].min():.3f} - {df["confidence"].max():.3f}')
    print(f'   🎯 Average confidence: {df["confidence"].mean():.3f}')

    # Check confidence distribution to see if low-confidence detections might be duplicates
    low_conf = df[df['confidence'] < 0.7]
    high_conf = df[df['confidence'] >= 0.7]
    print(f'   🎯 Low confidence (<0.7): {len(low_conf)} detections')
    print(f'   🎯 High confidence (≥0.7): {len(high_conf)} detections')

    print()
    print('🔧 NMS EFFECTIVENESS ASSESSMENT:')
    total_issues = len(close_detections) + len(high_overlaps) + coord_duplicates.sum()
    
    if total_issues == 0:
        print('   ✅ EXCELLENT: NMS system is working very effectively')
        print('   ✅ No significant duplicates detected')
        print('   💡 Current NMS threshold (0.5 IoU) is optimal')
    elif total_issues <= 2:
        print('   ✅ GOOD: NMS system is working well with minimal duplicates')
        print(f'   ℹ️ Found {total_issues} potential duplicate(s) - within acceptable range')
    else:
        print('   ⚠️ NEEDS ATTENTION: Multiple potential duplicates detected')
        print(f'   ⚠️ Found {total_issues} potential duplicates')
        print('   💡 Consider lowering NMS threshold (e.g., 0.3 or 0.4)')

    # Distance analysis between all detections
    print()
    print('📏 DETECTION SPACING ANALYSIS:')
    distances = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            lat_diff = abs(df.iloc[i]['lat'] - df.iloc[j]['lat'])
            lon_diff = abs(df.iloc[i]['lon'] - df.iloc[j]['lon'])
            distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)
            distance_m = distance_deg * 111000  # rough conversion to meters
            distances.append(distance_m)
    
    if distances:
        distances = np.array(distances)
        print(f'   📊 Minimum separation: {distances.min():.1f} meters')
        print(f'   📊 Average separation: {distances.mean():.1f} meters')
        print(f'   📊 Median separation: {np.median(distances):.1f} meters')
        
        # Count very close detections
        very_close = np.sum(distances < 100)  # Less than 100m
        close = np.sum(distances < 500)       # Less than 500m
        print(f'   📊 Detections < 100m apart: {very_close}')
        print(f'   📊 Detections < 500m apart: {close}')

if __name__ == "__main__":
    main()
