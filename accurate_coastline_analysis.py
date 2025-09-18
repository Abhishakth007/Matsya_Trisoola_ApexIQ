#!/usr/bin/env python3
"""
Accurate Coastline Analysis

Uses multiple methods to accurately determine which AIS vessels are close to land:
1. Speed-based analysis (stationary vessels often near ports)
2. Density clustering to identify port areas
3. Navigational status analysis
4. Approximate coastline distance using detailed Chesapeake Bay geography
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def get_chesapeake_bay_coastline_points():
    """
    Define key coastline points for Chesapeake Bay for more accurate distance calculation.
    These are major geographic features that define the bay's shoreline.
    """
    # Major coastline points around Chesapeake Bay (approximate)
    coastline_points = [
        # Western Shore (Maryland)
        (39.2, -76.6),  # Baltimore area
        (38.9, -76.5),  # Annapolis area
        (38.3, -76.4),  # Calvert County
        (37.8, -76.3),  # St. Mary's County
        (37.5, -76.2),  # Point Lookout
        
        # Eastern Shore (Maryland/Virginia)
        (39.1, -76.0),  # Kent Island area
        (38.8, -76.1),  # Eastern Shore
        (38.5, -75.9),  # Dorchester County
        (38.0, -75.8),  # Somerset County
        (37.5, -75.7),  # Virginia Eastern Shore
        
        # Virginia Western Shore
        (37.4, -76.3),  # Norfolk/Hampton Roads area
        (37.2, -76.4),  # Newport News
        (37.0, -76.5),  # Hampton
        (36.9, -76.3),  # Norfolk
        
        # Bay mouth
        (37.0, -76.0),  # Bay mouth eastern
        (36.9, -76.2),  # Bay mouth center
        (36.8, -76.1),  # Cape Henry area
    ]
    
    return coastline_points

def calculate_distance_to_coastline(lat, lon, coastline_points):
    """
    Calculate minimum distance from a point to the coastline.
    Uses Haversine formula for accuracy.
    """
    min_distance = float('inf')
    
    for coast_lat, coast_lon in coastline_points:
        # Haversine formula for distance calculation
        R = 6371000  # Earth radius in meters
        
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(coast_lat), np.radians(coast_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        min_distance = min(min_distance, distance)
    
    return min_distance

def analyze_vessel_speeds(vessels_df):
    """
    Analyze vessel speeds to identify stationary vessels (likely near ports).
    """
    logger.info("🚢 Analyzing vessel speeds...")
    
    # Speed categories
    stationary = vessels_df[vessels_df['SOG'] <= 0.5]  # <= 0.5 knots
    very_slow = vessels_df[(vessels_df['SOG'] > 0.5) & (vessels_df['SOG'] <= 2.0)]  # 0.5-2 knots
    slow = vessels_df[(vessels_df['SOG'] > 2.0) & (vessels_df['SOG'] <= 5.0)]  # 2-5 knots
    normal = vessels_df[vessels_df['SOG'] > 5.0]  # > 5 knots
    
    logger.info(f"📊 Speed Analysis:")
    logger.info(f"   Stationary (≤0.5 knots): {len(stationary)} vessels ({len(stationary)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Very slow (0.5-2 knots): {len(very_slow)} vessels ({len(very_slow)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Slow (2-5 knots): {len(slow)} vessels ({len(slow)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Normal (>5 knots): {len(normal)} vessels ({len(normal)/len(vessels_df)*100:.1f}%)")
    
    return stationary, very_slow, slow, normal

def analyze_navigation_status(vessels_df):
    """
    Analyze navigational status to identify anchored/moored vessels.
    """
    logger.info("⚓ Analyzing navigation status...")
    
    if 'NavigationStatus' in vessels_df.columns:
        status_counts = vessels_df['NavigationStatus'].value_counts()
        logger.info(f"📊 Navigation Status Distribution:")
        for status, count in status_counts.head(10).items():
            logger.info(f"   {status}: {count} vessels ({count/len(vessels_df)*100:.1f}%)")
    else:
        logger.warning("⚠️ NavigationStatus column not found in data")

def cluster_vessel_positions(vessels_df):
    """
    Use DBSCAN clustering to identify port areas and vessel concentrations.
    """
    logger.info("🎯 Clustering vessel positions to identify port areas...")
    
    # Prepare coordinates for clustering
    coords = vessels_df[['LAT', 'LON']].values
    
    # DBSCAN clustering (eps in degrees, min_samples for cluster)
    # eps=0.01 degrees ≈ ~1km at this latitude
    dbscan = DBSCAN(eps=0.01, min_samples=3)
    clusters = dbscan.fit_predict(coords)
    
    vessels_df['cluster'] = clusters
    
    # Analyze clusters
    unique_clusters = set(clusters)
    unique_clusters.discard(-1)  # Remove noise points
    
    logger.info(f"📊 Clustering Results:")
    logger.info(f"   Number of clusters (potential ports): {len(unique_clusters)}")
    logger.info(f"   Noise points (isolated vessels): {sum(clusters == -1)}")
    
    # Analyze each cluster
    for cluster_id in sorted(unique_clusters):
        cluster_vessels = vessels_df[vessels_df['cluster'] == cluster_id]
        center_lat = cluster_vessels['LAT'].mean()
        center_lon = cluster_vessels['LON'].mean()
        
        logger.info(f"   Cluster {cluster_id}: {len(cluster_vessels)} vessels at ({center_lat:.4f}, {center_lon:.4f})")
    
    return vessels_df

def main():
    logger.info("🚀 Starting Accurate Coastline Analysis")
    
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
    
    # Get Chesapeake Bay coastline points
    coastline_points = get_chesapeake_bay_coastline_points()
    logger.info(f"🗺️ Using {len(coastline_points)} coastline reference points")
    
    # Calculate accurate distances to coastline
    logger.info("📏 Calculating distances to coastline...")
    vessels_df['coastline_distance_m'] = vessels_df.apply(
        lambda row: calculate_distance_to_coastline(row['LAT'], row['LON'], coastline_points), 
        axis=1
    )
    
    # Analyze distance distribution
    within_100m = vessels_df[vessels_df['coastline_distance_m'] <= 100]
    within_500m = vessels_df[vessels_df['coastline_distance_m'] <= 500]
    within_1km = vessels_df[vessels_df['coastline_distance_m'] <= 1000]
    within_2km = vessels_df[vessels_df['coastline_distance_m'] <= 2000]
    within_5km = vessels_df[vessels_df['coastline_distance_m'] <= 5000]
    open_sea = vessels_df[vessels_df['coastline_distance_m'] > 5000]
    
    logger.info(f"\n🏖️ ACCURATE DISTANCE TO COASTLINE ANALYSIS:")
    logger.info(f"   Within 100m: {len(within_100m)} vessels ({len(within_100m)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Within 500m: {len(within_500m)} vessels ({len(within_500m)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Within 1km: {len(within_1km)} vessels ({len(within_1km)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Within 2km: {len(within_2km)} vessels ({len(within_2km)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Within 5km: {len(within_5km)} vessels ({len(within_5km)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Open sea (>5km): {len(open_sea)} vessels ({len(open_sea)/len(vessels_df)*100:.1f}%)")
    
    # Speed analysis
    stationary, very_slow, slow, normal = analyze_vessel_speeds(vessels_df)
    
    # Navigation status analysis
    analyze_navigation_status(vessels_df)
    
    # Clustering analysis
    vessels_df = cluster_vessel_positions(vessels_df)
    
    # Cross-analysis: Speed vs Distance
    logger.info(f"\n🔍 CROSS ANALYSIS - Speed vs Distance to Shore:")
    
    stationary_near_shore = stationary[stationary['coastline_distance_m'] <= 1000]
    slow_near_shore = very_slow[very_slow['coastline_distance_m'] <= 1000]
    
    logger.info(f"   Stationary vessels near shore (<1km): {len(stationary_near_shore)}")
    logger.info(f"   Very slow vessels near shore (<1km): {len(slow_near_shore)}")
    
    # Compare with our correlations
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    correlated_mmsis = set([str(mmsi) for mmsi in corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']['matched_mmsi'].unique()])
    
    # Check which vessels we successfully correlated
    vessels_df['correlated'] = vessels_df['MMSI'].astype(str).isin(correlated_mmsis)
    
    correlated_vessels = vessels_df[vessels_df['correlated']]
    missed_vessels = vessels_df[~vessels_df['correlated']]
    
    logger.info(f"\n📊 CORRELATION ANALYSIS BY DISTANCE:")
    logger.info(f"   Correlated vessels: {len(correlated_vessels)}")
    logger.info(f"   Missed vessels: {len(missed_vessels)}")
    
    if len(missed_vessels) > 0:
        missed_near_shore = missed_vessels[missed_vessels['coastline_distance_m'] <= 1000]
        missed_stationary = missed_vessels[missed_vessels['SOG'] <= 0.5]
        
        logger.info(f"   Missed vessels near shore (<1km): {len(missed_near_shore)} ({len(missed_near_shore)/len(missed_vessels)*100:.1f}%)")
        logger.info(f"   Missed stationary vessels: {len(missed_stationary)} ({len(missed_stationary)/len(missed_vessels)*100:.1f}%)")
    
    # Final summary
    logger.info(f"\n🎯 FINAL SUMMARY:")
    logger.info(f"   Total vessels in ±10min window: {len(vessels_df)}")
    logger.info(f"   Vessels within 1km of shore: {len(within_1km)} ({len(within_1km)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Stationary vessels (≤0.5 knots): {len(stationary)} ({len(stationary)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Successfully correlated: {len(correlated_vessels)} ({len(correlated_vessels)/len(vessels_df)*100:.1f}%)")
    logger.info(f"   Correlation rate for open sea vessels (>1km): {len(correlated_vessels[correlated_vessels['coastline_distance_m'] > 1000])/len(vessels_df[vessels_df['coastline_distance_m'] > 1000])*100:.1f}%")

if __name__ == "__main__":
    main()
