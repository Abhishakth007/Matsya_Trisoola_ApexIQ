#!/usr/bin/env python3
"""
Debug motion prediction to see if it's working correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime
from motion_models import predict_ais_position

def main():
    print('🔍 DEBUGGING MOTION PREDICTION')
    print('=' * 50)

    # Load AIS data
    ais_data = pd.read_csv('../ais_data/AIS_175664700472271242_3396-1756647005869.csv')
    
    # Load results to see what we matched
    results = pd.read_csv('ais_correlation_results.csv')
    matched = results[results['matched_mmsi'] != 'UNKNOWN']
    
    # Load predictions to get detection positions
    predictions = pd.read_csv('../professor/outputs/predictions.csv')
    
    # SAR timestamp (from our system)
    sar_timestamp = datetime(2023, 6, 20, 23, 6, 42)
    
    print(f'SAR Timestamp: {sar_timestamp}')
    print()
    
    # Test a few specific cases
    test_cases = [
        {'detect_id': 5, 'mmsi': 636021288},
        {'detect_id': 11, 'mmsi': 353693000},
        {'detect_id': 12, 'mmsi': 538004260}
    ]
    
    for case in test_cases:
        detect_id = case['detect_id']
        mmsi = case['mmsi']
        
        print(f'🔍 TESTING DETECTION {detect_id} (MMSI {mmsi})')
        print('-' * 40)
        
        # Get detection position
        detection = predictions[predictions['detect_id'] == detect_id].iloc[0]
        print(f'Detection position: LAT {detection["lat"]:.6f}, LON {detection["lon"]:.6f}')
        
        # Get AIS data for this MMSI
        mmsi_data = ais_data[ais_data['MMSI'] == int(mmsi)]
        print(f'AIS records for MMSI {mmsi}: {len(mmsi_data)}')
        
        # Find the closest AIS record to SAR timestamp
        mmsi_data['BaseDateTime'] = pd.to_datetime(mmsi_data['BaseDateTime'])
        time_diffs = abs((mmsi_data['BaseDateTime'] - sar_timestamp).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        closest_ais = mmsi_data.loc[closest_idx]
        
        print(f'Closest AIS record:')
        print(f'  Time: {closest_ais["BaseDateTime"]}')
        print(f'  Position: LAT {closest_ais["LAT"]:.6f}, LON {closest_ais["LON"]:.6f}')
        print(f'  SOG: {closest_ais["SOG"]} knots, COG: {closest_ais["COG"]}°')
        print(f'  Heading: {closest_ais["Heading"]}°')
        
        # Calculate time gap
        dt_seconds = (sar_timestamp - closest_ais['BaseDateTime']).total_seconds()
        print(f'Time gap: {dt_seconds/60:.1f} minutes')
        
        # Predict position
        ais_dict = closest_ais.to_dict()
        pred_lat, pred_lon, uncertainty = predict_ais_position(ais_dict, dt_seconds)
        
        print(f'Predicted position: LAT {pred_lat:.6f}, LON {pred_lon:.6f}')
        print(f'Uncertainty: {uncertainty:.1f} meters')
        
        # Calculate distance between detection and predicted position (Haversine formula)
        def calculate_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            delta_lat = np.radians(lat2 - lat1)
            delta_lon = np.radians(lon2 - lon1)
            a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        distance = calculate_distance(detection['lat'], detection['lon'], pred_lat, pred_lon)
        
        print(f'Distance to detection: {distance:.1f} meters')
        print()
        
        # Check if this matches our results
        result = matched[matched['detect_id'] == detect_id].iloc[0]
        print(f'Our result distance: {result["distance_meters"]:.1f} meters')
        print(f'Our result time gap: {result["time_gap_minutes"]:.1f} minutes')
        print()

if __name__ == '__main__':
    main()
