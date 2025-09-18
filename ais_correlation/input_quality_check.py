#!/usr/bin/env python3
"""
Input Quality Check: Compare expected vessels vs detected vessels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_sar_image_bounds():
    """Get SAR image bounds from predictions data."""
    predictions = pd.read_csv('../professor/outputs/predictions.csv')
    
    # Add some buffer around the detections to define the scene
    lat_buffer = 0.01  # ~1km buffer
    lon_buffer = 0.01  # ~1km buffer
    
    bounds = {
        'min_lat': predictions['lat'].min() - lat_buffer,
        'max_lat': predictions['lat'].max() + lat_buffer,
        'min_lon': predictions['lon'].min() - lon_buffer,
        'max_lon': predictions['lon'].max() + lon_buffer
    }
    
    return bounds, predictions

def find_ais_vessels_in_scene(ais_data, sar_timestamp, scene_bounds, time_tolerance_minutes=5):
    """Find AIS vessels that should be visible in the SAR scene."""
    
    # Convert SAR timestamp to datetime
    if isinstance(sar_timestamp, str):
        sar_timestamp = datetime.fromisoformat(sar_timestamp.replace('Z', '+00:00'))
    
    # Filter AIS data by time window
    time_tolerance = timedelta(minutes=time_tolerance_minutes)
    time_start = sar_timestamp - time_tolerance
    time_end = sar_timestamp + time_tolerance
    
    ais_data['BaseDateTime'] = pd.to_datetime(ais_data['BaseDateTime'])
    time_filtered = ais_data[
        (ais_data['BaseDateTime'] >= time_start) & 
        (ais_data['BaseDateTime'] <= time_end)
    ]
    
    print(f"📊 AIS data within ±{time_tolerance_minutes} minutes of SAR timestamp: {len(time_filtered)} records")
    
    # Filter by spatial bounds
    spatial_filtered = time_filtered[
        (time_filtered['LAT'] >= scene_bounds['min_lat']) &
        (time_filtered['LAT'] <= scene_bounds['max_lat']) &
        (time_filtered['LON'] >= scene_bounds['min_lon']) &
        (time_filtered['LON'] <= scene_bounds['max_lon'])
    ]
    
    print(f"📊 AIS data within spatial bounds: {len(spatial_filtered)} records")
    
    # Get unique vessels (MMSI) in the scene
    unique_vessels = spatial_filtered['MMSI'].unique()
    print(f"📊 Unique vessels in scene: {len(unique_vessels)}")
    
    # For each unique vessel, get the closest record to SAR timestamp
    vessel_records = []
    for mmsi in unique_vessels:
        vessel_data = spatial_filtered[spatial_filtered['MMSI'] == mmsi]
        
        # Find closest record to SAR timestamp
        time_diffs = abs((vessel_data['BaseDateTime'] - sar_timestamp).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        closest_record = vessel_data.loc[closest_idx]
        
        vessel_records.append({
            'MMSI': mmsi,
            'LAT': closest_record['LAT'],
            'LON': closest_record['LON'],
            'SOG': closest_record['SOG'],
            'COG': closest_record['COG'],
            'Heading': closest_record['Heading'],
            'Length': closest_record['Length'],
            'Width': closest_record['Width'],
            'VesselType': closest_record['VesselType'],
            'BaseDateTime': closest_record['BaseDateTime'],
            'time_gap_minutes': time_diffs[closest_idx] / 60.0
        })
    
    return pd.DataFrame(vessel_records)

def analyze_detection_coverage(predictions, ais_vessels):
    """Analyze how well our detections cover the expected vessels."""
    
    print(f"\n🔍 DETECTION COVERAGE ANALYSIS")
    print("=" * 50)
    
    # Load correlation results
    try:
        correlation_results = pd.read_csv('ais_correlation_results.csv')
        matched_detections = correlation_results[correlation_results['matched_mmsi'] != 'UNKNOWN']
        dark_ships = correlation_results[correlation_results['matched_mmsi'] == 'UNKNOWN']
        
        print(f"📊 Our detections: {len(predictions)}")
        print(f"📊 Expected vessels: {len(ais_vessels)}")
        print(f"📊 Matched detections: {len(matched_detections)}")
        print(f"📊 Dark ships: {len(dark_ships)}")
        print(f"📊 Detection rate: {len(predictions)/len(ais_vessels)*100:.1f}%")
        print(f"📊 Match rate: {len(matched_detections)/len(predictions)*100:.1f}%")
        
        # Check which expected vessels we found
        matched_mmsis = set(matched_detections['matched_mmsi'].astype(str))
        expected_mmsis = set(ais_vessels['MMSI'].astype(str))
        
        found_vessels = matched_mmsis.intersection(expected_mmsis)
        missed_vessels = expected_mmsis - matched_mmsis
        
        print(f"\n📊 Vessel Coverage:")
        print(f"  Found expected vessels: {len(found_vessels)}/{len(expected_mmsis)} ({len(found_vessels)/len(expected_mmsis)*100:.1f}%)")
        print(f"  Missed expected vessels: {len(missed_vessels)}")
        print(f"  Extra detections (dark ships): {len(dark_ships)}")
        
        if missed_vessels:
            print(f"\n❌ Missed vessels: {list(missed_vessels)}")
            
            # Analyze missed vessels
            missed_data = ais_vessels[ais_vessels['MMSI'].astype(str).isin(missed_vessels)]
            print(f"\n📊 Missed vessel characteristics:")
            print(f"  Average SOG: {missed_data['SOG'].mean():.1f} knots")
            print(f"  Average Length: {missed_data['Length'].mean():.1f} meters")
            print(f"  Average Width: {missed_data['Width'].mean():.1f} meters")
            print(f"  Vessel types: {missed_data['VesselType'].value_counts().to_dict()}")
        
        if found_vessels:
            print(f"\n✅ Found vessels: {list(found_vessels)}")
            
            # Analyze found vessels
            found_data = ais_vessels[ais_vessels['MMSI'].astype(str).isin(found_vessels)]
            print(f"\n📊 Found vessel characteristics:")
            print(f"  Average SOG: {found_data['SOG'].mean():.1f} knots")
            print(f"  Average Length: {found_data['Length'].mean():.1f} meters")
            print(f"  Average Width: {found_data['Width'].mean():.1f} meters")
            print(f"  Vessel types: {found_data['VesselType'].value_counts().to_dict()}")
        
    except FileNotFoundError:
        print("❌ Correlation results not found. Run correlation first.")

def main():
    print('🔍 INPUT QUALITY CHECK')
    print('=' * 50)
    
    # SAR timestamp
    sar_timestamp = "2023-06-20T23:06:42"
    print(f"SAR Timestamp: {sar_timestamp}")
    
    # Get SAR image bounds and predictions
    scene_bounds, predictions = get_sar_image_bounds()
    print(f"\n📊 SAR Scene Bounds:")
    print(f"  LAT: {scene_bounds['min_lat']:.6f} to {scene_bounds['max_lat']:.6f}")
    print(f"  LON: {scene_bounds['min_lon']:.6f} to {scene_bounds['max_lon']:.6f}")
    print(f"  Scene size: ~{calculate_distance(scene_bounds['min_lat'], scene_bounds['min_lon'], scene_bounds['max_lat'], scene_bounds['max_lon'])/1000:.1f} km")
    
    # Load AIS data
    ais_data = pd.read_csv('../ais_data/AIS_175664700472271242_3396-1756647005869.csv')
    print(f"\n📊 Total AIS records: {len(ais_data)}")
    print(f"📊 Unique AIS vessels: {ais_data['MMSI'].nunique()}")
    
    # Find expected vessels in scene
    print(f"\n🔍 Finding expected vessels in SAR scene...")
    ais_vessels = find_ais_vessels_in_scene(ais_data, sar_timestamp, scene_bounds, time_tolerance_minutes=5)
    
    if len(ais_vessels) > 0:
        print(f"\n📊 Expected vessels in scene:")
        print(ais_vessels[['MMSI', 'LAT', 'LON', 'SOG', 'Length', 'Width', 'VesselType', 'time_gap_minutes']].to_string(index=False))
        
        # Analyze detection coverage
        analyze_detection_coverage(predictions, ais_vessels)
        
        # Save results
        ais_vessels.to_csv('expected_vessels_in_scene.csv', index=False)
        print(f"\n💾 Expected vessels saved to: expected_vessels_in_scene.csv")
        
    else:
        print("❌ No AIS vessels found in the SAR scene!")

if __name__ == '__main__':
    main()

