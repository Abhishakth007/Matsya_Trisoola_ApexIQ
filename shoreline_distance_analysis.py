#!/usr/bin/env python3
"""
Shoreline Distance Analysis

Check how many of the 119 AIS vessels are within 50m of shoreline
vs how many of our 28 correlations are in open water.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def estimate_distance_to_shore(lat, lon):
    """
    Rough estimation of distance to shore for Chesapeake Bay.
    This is a simplified approach - in reality you'd use coastline data.
    """
    # Chesapeake Bay is roughly between these coordinates
    # Main bay runs north-south, with eastern and western shores
    
    # Rough eastern shore longitude (approximate)
    eastern_shore_lon = -75.8
    # Rough western shore longitude (approximate) 
    western_shore_lon = -76.8
    
    # Distance to nearest shore (very rough approximation)
    dist_to_east = abs(lon - eastern_shore_lon) * 111000 * np.cos(np.radians(lat))  # meters
    dist_to_west = abs(lon - western_shore_lon) * 111000 * np.cos(np.radians(lat))  # meters
    
    # Return distance to nearest shore
    return min(dist_to_east, dist_to_west)

def main():
    logger.info("🚀 Analyzing vessel distances to shoreline")
    
    # Real SAR coordinates
    real_bounds = {
        'min_lat': 35.713299,
        'max_lat': 37.623062,
        'min_lon': -78.501892,
        'max_lon': -75.307404
    }
    
    sar_timestamp = datetime(2023, 6, 20, 23, 6, 42)
    
    # Load AIS data
    ais_df = pd.read_csv("ais_data/AIS_175664700472271242_3396-1756647005869.csv")
    ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
    
    # Filter for ground truth vessels (±10min in SAR coverage)
    time_delta = timedelta(minutes=10)
    time_filter = (
        (ais_df['BaseDateTime'] >= sar_timestamp - time_delta) &
        (ais_df['BaseDateTime'] <= sar_timestamp + time_delta)
    )
    spatial_filter = (
        (ais_df['LAT'] >= real_bounds['min_lat']) &
        (ais_df['LAT'] <= real_bounds['max_lat']) &
        (ais_df['LON'] >= real_bounds['min_lon']) &
        (ais_df['LON'] <= real_bounds['max_lon'])
    )
    
    ground_truth_ais = ais_df[time_filter & spatial_filter]
    
    # Get unique vessels (closest record per MMSI)
    unique_vessels = []
    for mmsi in ground_truth_ais['MMSI'].unique():
        vessel_records = ground_truth_ais[ground_truth_ais['MMSI'] == mmsi]
        # Get record closest to SAR time
        time_diffs = abs((vessel_records['BaseDateTime'] - sar_timestamp).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        unique_vessels.append(vessel_records.loc[closest_idx])
    
    vessels_df = pd.DataFrame(unique_vessels)
    
    logger.info(f"📊 Analyzing {len(vessels_df)} unique ground truth vessels")
    
    # Estimate distances to shore
    vessels_df['shore_distance_m'] = vessels_df.apply(
        lambda row: estimate_distance_to_shore(row['LAT'], row['LON']), axis=1
    )
    
    # Analyze distribution
    within_50m = vessels_df[vessels_df['shore_distance_m'] <= 50]
    within_100m = vessels_df[vessels_df['shore_distance_m'] <= 100]
    within_500m = vessels_df[vessels_df['shore_distance_m'] <= 500]
    open_sea = vessels_df[vessels_df['shore_distance_m'] > 500]
    
    logger.info(f"🏖️ DISTANCE TO SHORE ANALYSIS:")
    logger.info(f"   Within 50m of shore: {len(within_50m)} vessels ({len(within_50m)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Within 100m of shore: {len(within_100m)} vessels ({len(within_100m)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Within 500m of shore: {len(within_500m)} vessels ({len(within_500m)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Open sea (>500m): {len(open_sea)} vessels ({len(open_sea)/len(vessels_df)*100:.1f}%)")
    
    # Load correlation results
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    correlated_mmsis = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']['matched_mmsi'].unique()
    
    # Check which distance categories our correlations fall into
    correlated_vessels = vessels_df[vessels_df['MMSI'].astype(str).isin([str(mmsi) for mmsi in correlated_mmsis])]
    
    if len(correlated_vessels) > 0:
        logger.info(f"🎯 CORRELATED VESSELS DISTANCE ANALYSIS:")
        logger.info(f"   Total correlated: {len(correlated_vessels)}")
        
        corr_within_50m = correlated_vessels[correlated_vessels['shore_distance_m'] <= 50]
        corr_within_100m = correlated_vessels[correlated_vessels['shore_distance_m'] <= 100]
        corr_within_500m = correlated_vessels[correlated_vessels['shore_distance_m'] <= 500]
        corr_open_sea = correlated_vessels[correlated_vessels['shore_distance_m'] > 500]
        
        logger.info(f"   Within 50m: {len(corr_within_50m)} ({len(corr_within_50m)/len(correlated_vessels)*100:.1f}%)")
        logger.info(f"   Within 100m: {len(corr_within_100m)} ({len(corr_within_100m)/len(correlated_vessels)*100:.1f}%)")
        logger.info(f"   Within 500m: {len(corr_within_500m)} ({len(corr_within_500m)/len(correlated_vessels)*100:.1f}%)")
        logger.info(f"   Open sea: {len(corr_open_sea)} ({len(corr_open_sea)/len(correlated_vessels)*100:.1f}%)")
    
    # Summary
    logger.info(f"\n🎯 SUMMARY:")
    logger.info(f"   Ground truth vessels: {len(vessels_df)}")
    logger.info(f"   Successfully correlated: {len(correlated_vessels)}")
    logger.info(f"   Correlation rate: {len(correlated_vessels)/len(vessels_df)*100:.1f}%")
    logger.info(f"   Land mask extension: 50m (from SNAP processing)")
    
    if len(within_50m) > 0:
        logger.info(f"\n⚠️ POTENTIAL ISSUE:")
        logger.info(f"   {len(within_50m)} vessels within 50m of shore (would be masked out)")
        logger.info(f"   This could explain the gap between {len(vessels_df)} total and {len(correlated_vessels)} detected")

if __name__ == "__main__":
    main()
