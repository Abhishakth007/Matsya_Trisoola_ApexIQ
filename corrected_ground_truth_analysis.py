#!/usr/bin/env python3
"""
Corrected Ground Truth Analysis

Using the REAL SAR coordinates from the SAFE KML file:
Lat: 35.713299 to 37.623062
Lon: -78.501892 to -75.307404
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Corrected Ground Truth Analysis with REAL SAR coordinates")
    
    # REAL SAR coordinates from SAFE KML
    real_bounds = {
        'min_lat': 35.713299,
        'max_lat': 37.623062,
        'min_lon': -78.501892,
        'max_lon': -75.307404
    }
    
    sar_timestamp = datetime(2023, 6, 20, 23, 6, 42)
    
    logger.info(f"📍 Using REAL SAR bounds from SAFE KML:")
    logger.info(f"   Lat: {real_bounds['min_lat']:.6f} to {real_bounds['max_lat']:.6f}")
    logger.info(f"   Lon: {real_bounds['min_lon']:.6f} to {real_bounds['max_lon']:.6f}")
    
    # Load AIS data
    ais_df = pd.read_csv("ais_data/AIS_175664700472271242_3396-1756647005869.csv")
    ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
    logger.info(f"✅ Loaded {len(ais_df)} AIS records")
    
    # Filter by REAL coverage and ±10min
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
    
    # Apply filters
    spatial_only = ais_df[spatial_filter]
    both_filters = ais_df[time_filter & spatial_filter]
    
    logger.info(f"📊 CORRECTED RESULTS:")
    logger.info(f"   Total AIS records: {len(ais_df)}")
    logger.info(f"   In REAL SAR coverage (any time): {len(spatial_only)} records")
    logger.info(f"   Unique vessels in coverage (any time): {spatial_only['MMSI'].nunique()}")
    logger.info(f"   In REAL coverage + ±10min: {len(both_filters)} records")
    logger.info(f"   Unique vessels in coverage + ±10min: {both_filters['MMSI'].nunique()}")
    
    # Get the vessel list
    ground_truth_mmsis = both_filters['MMSI'].unique()
    logger.info(f"🚢 Ground truth vessels (±10min in REAL coverage): {len(ground_truth_mmsis)}")
    
    # Load correlation results
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    correlated_mmsis = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']['matched_mmsi'].unique()
    
    # Check overlap
    gt_set = set([str(mmsi) for mmsi in ground_truth_mmsis])
    corr_set = set([str(mmsi) for mmsi in correlated_mmsis])
    overlap = gt_set.intersection(corr_set)
    
    logger.info(f"🔍 MMSI COMPARISON:")
    logger.info(f"   Ground truth MMSIs: {len(gt_set)}")
    logger.info(f"   Correlated MMSIs: {len(corr_set)}")
    logger.info(f"   Overlap: {len(overlap)}")
    
    if len(overlap) > 0:
        logger.info(f"   ✅ Overlapping MMSIs: {list(overlap)}")
    
    # Show some examples
    if len(ground_truth_mmsis) > 0:
        logger.info(f"📋 Sample ground truth MMSIs: {ground_truth_mmsis[:10]}")
    
    logger.info(f"📋 Sample correlation MMSIs: {list(correlated_mmsis)[:10]}")

if __name__ == "__main__":
    main()
