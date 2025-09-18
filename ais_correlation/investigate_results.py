#!/usr/bin/env python3
"""
Investigate suspicious AIS correlation results
"""

import pandas as pd
import numpy as np

def main():
    print('🔍 INVESTIGATING SUSPICIOUS RESULTS')
    print('=' * 50)

    # Load the results
    results = pd.read_csv('ais_correlation_results.csv')
    matched = results[results['matched_mmsi'] != 'UNKNOWN']

    print('Matched Results Analysis:')
    print('  Total matched:', len(matched))
    print('  Distance range:', matched['distance_meters'].min(), '-', matched['distance_meters'].max(), 'meters')
    print('  Time gap range:', matched['time_gap_minutes'].min(), '-', matched['time_gap_minutes'].max(), 'minutes')
    print('  Confidence range:', matched['match_confidence'].min(), '-', matched['match_confidence'].max())
    print()

    print('Position Scores:')
    print('  Range:', matched['position_score'].min(), '-', matched['position_score'].max())
    print('  All > 0.9?', (matched['position_score'] > 0.9).all())
    print()

    # Check matched MMSIs
    matched_mmsis = matched['matched_mmsi'].unique()
    print('Matched MMSIs:', matched_mmsis)
    print()

    # Load AIS data
    ais_data = pd.read_csv('../ais_data/AIS_175664700472271242_3396-1756647005869.csv')
    
    print('🔍 CHECKING IF MMSIs EXIST IN AIS DATA')
    print('=' * 50)

    for mmsi in matched_mmsis:
        mmsi_data = ais_data[ais_data['MMSI'] == int(mmsi)]
        if len(mmsi_data) > 0:
            print(f'MMSI {mmsi}: {len(mmsi_data)} records')
            print(f'  Time range: {mmsi_data["BaseDateTime"].min()} to {mmsi_data["BaseDateTime"].max()}')
            print(f'  Position range: LAT {mmsi_data["LAT"].min():.6f}-{mmsi_data["LAT"].max():.6f}')
            print(f'  Position range: LON {mmsi_data["LON"].min():.6f}-{mmsi_data["LON"].max():.6f}')
        else:
            print(f'MMSI {mmsi}: NOT FOUND in AIS data!')
        print()

    # Check if we're using raw AIS positions
    print('🔍 CHECKING IF WE ARE USING RAW AIS POSITIONS')
    print('=' * 50)
    
    # Load predictions
    predictions = pd.read_csv('../professor/outputs/predictions.csv')
    print('Predictions Analysis:')
    print('  Total detections:', len(predictions))
    print('  Position range: LAT {:.6f}-{:.6f}'.format(predictions['lat'].min(), predictions['lat'].max()))
    print('  Position range: LON {:.6f}-{:.6f}'.format(predictions['lon'].min(), predictions['lon'].max()))
    print()

    # Check if any AIS positions are very close to detection positions
    print('🔍 CHECKING FOR SUSPICIOUSLY CLOSE MATCHES')
    print('=' * 50)
    
    for _, match in matched.iterrows():
        detect_id = match['detect_id']
        detection = predictions[predictions['detect_id'] == detect_id].iloc[0]
        
        print(f'Detection {detect_id}:')
        print(f'  Detection position: LAT {detection["lat"]:.6f}, LON {detection["lon"]:.6f}')
        print(f'  Matched MMSI: {match["matched_mmsi"]}')
        print(f'  Distance: {match["distance_meters"]:.1f} meters')
        print(f'  Time gap: {match["time_gap_minutes"]:.1f} minutes')
        print()

if __name__ == '__main__':
    main()
