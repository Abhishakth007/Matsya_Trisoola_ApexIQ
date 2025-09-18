#!/usr/bin/env python3
"""
AIS-SAR Ground Truth Analysis

Analyzes how many unique AIS vessels are actually present within:
1. SAR image coverage area
2. ±10 minute window of SAR acquisition
3. Compares to detection and correlation performance

This gives us the ground truth baseline for performance evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'professor'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ais_correlation'))

from safe_coordinate_system import SAFECoordinateSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def extract_sar_timestamp_from_safe(safe_folder_path):
    """Extract SAR acquisition timestamp from SAFE folder name."""
    try:
        # Extract from folder name: S1A_IW_GRDH_1SDV_20230620T230642_...
        folder_name = os.path.basename(safe_folder_path)
        if 'T' in folder_name:
            # Find the timestamp part
            parts = folder_name.split('_')
            for part in parts:
                if 'T' in part and len(part) == 15:  # Format: 20230620T230642
                    date_part = part[:8]  # 20230620
                    time_part = part[9:]   # 230642
                    
                    year = int(date_part[:4])
                    month = int(date_part[4:6])
                    day = int(date_part[6:8])
                    hour = int(time_part[:2])
                    minute = int(time_part[2:4])
                    second = int(time_part[4:6])
                    
                    timestamp = datetime(year, month, day, hour, minute, second)
                    logger.info(f"📅 Extracted SAR timestamp: {timestamp}")
                    return timestamp
        
        # Fallback
        logger.warning("⚠️ Could not extract timestamp from SAFE folder, using default")
        return datetime(2023, 6, 20, 23, 6, 42)
        
    except Exception as e:
        logger.error(f"❌ Error extracting SAR timestamp: {e}")
        return datetime(2023, 6, 20, 23, 6, 42)

def get_sar_coverage_bounds(safe_folder_path, sar_image_path):
    """Get SAR image geographic coverage bounds."""
    try:
        logger.info(f"🗺️ Extracting SAR coverage bounds...")
        
        # Try using SAFECoordinateSystem
        coord_system = SAFECoordinateSystem(safe_folder_path)
        
        # Get image dimensions (approximate)
        # For a typical Sentinel-1 IW GRDH product
        approx_width = 25000  # pixels
        approx_height = 16500  # pixels
        
        # Get corner coordinates
        corners = [
            coord_system.pixel_to_geo(0, 0),
            coord_system.pixel_to_geo(approx_width, 0),
            coord_system.pixel_to_geo(0, approx_height),
            coord_system.pixel_to_geo(approx_width, approx_height)
        ]
        
        # Extract lat/lon bounds
        lats = [corner[0] for corner in corners]
        lons = [corner[1] for corner in corners]
        
        bounds = {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
        
        logger.info(f"📍 SAR coverage bounds:")
        logger.info(f"   Latitude: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
        logger.info(f"   Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")
        
        return bounds
        
    except Exception as e:
        logger.warning(f"⚠️ Could not extract precise bounds: {e}")
        # Use approximate Chesapeake Bay bounds as fallback
        logger.info("📍 Using approximate Chesapeake Bay bounds as fallback")
        return {
            'min_lat': 35.9,
            'max_lat': 37.6,
            'min_lon': -77.3,
            'max_lon': -75.5
        }

def load_ais_data(ais_data_path):
    """Load and prepare AIS data."""
    logger.info(f"📊 Loading AIS data from: {ais_data_path}")
    
    try:
        ais_df = pd.read_csv(ais_data_path)
        logger.info(f"✅ Loaded {len(ais_df)} AIS records")
        
        # Convert timestamp
        ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
        
        # Clean data
        ais_df = ais_df.dropna(subset=['LAT', 'LON', 'MMSI'])
        logger.info(f"✅ After cleaning: {len(ais_df)} valid records")
        
        return ais_df
        
    except Exception as e:
        logger.error(f"❌ Error loading AIS data: {e}")
        return None

def filter_ais_by_coverage_and_time(ais_df, bounds, sar_timestamp, time_window_minutes=10):
    """Filter AIS data by geographic coverage and time window."""
    logger.info(f"🔍 Filtering AIS data...")
    logger.info(f"   Time window: ±{time_window_minutes} minutes around {sar_timestamp}")
    
    # Spatial filtering
    spatial_filter = (
        (ais_df['LAT'] >= bounds['min_lat']) &
        (ais_df['LAT'] <= bounds['max_lat']) &
        (ais_df['LON'] >= bounds['min_lon']) &
        (ais_df['LON'] <= bounds['max_lon'])
    )
    
    spatial_filtered = ais_df[spatial_filter].copy()
    logger.info(f"📍 Spatial filtering: {len(ais_df)} -> {len(spatial_filtered)} records in SAR coverage")
    
    # Temporal filtering
    time_delta = timedelta(minutes=time_window_minutes)
    time_filter = (
        (spatial_filtered['BaseDateTime'] >= sar_timestamp - time_delta) &
        (spatial_filtered['BaseDateTime'] <= sar_timestamp + time_delta)
    )
    
    final_filtered = spatial_filtered[time_filter].copy()
    logger.info(f"⏰ Temporal filtering: {len(spatial_filtered)} -> {len(final_filtered)} records in ±{time_window_minutes}min window")
    
    return spatial_filtered, final_filtered

def analyze_unique_vessels(spatial_df, temporal_df, sar_timestamp):
    """Analyze unique vessels in different datasets."""
    logger.info(f"🚢 Analyzing unique vessels...")
    
    # Unique vessels in spatial coverage (any time)
    spatial_vessels = spatial_df['MMSI'].nunique()
    logger.info(f"📊 Unique vessels in SAR coverage (any time): {spatial_vessels}")
    
    # Unique vessels in temporal window within coverage
    temporal_vessels = temporal_df['MMSI'].nunique()
    logger.info(f"📊 Unique vessels in SAR coverage + ±10min window: {temporal_vessels}")
    
    # Get the actual vessel list for temporal window
    temporal_vessel_list = temporal_df['MMSI'].unique()
    
    # Calculate time gaps for temporal vessels
    time_gaps = []
    for mmsi in temporal_vessel_list:
        vessel_data = temporal_df[temporal_df['MMSI'] == mmsi]
        if len(vessel_data) > 0:
            # Get closest record to SAR time
            time_diffs = abs((vessel_data['BaseDateTime'] - sar_timestamp).dt.total_seconds())
            min_gap = time_diffs.min()
            time_gaps.append(min_gap / 60.0)  # Convert to minutes
    
    avg_time_gap = np.mean(time_gaps) if time_gaps else 0
    logger.info(f"📊 Average time gap to SAR: {avg_time_gap:.2f} minutes")
    
    return spatial_vessels, temporal_vessels, temporal_vessel_list, time_gaps

def load_detection_and_correlation_results(results_dir):
    """Load our detection and correlation results."""
    logger.info(f"📈 Loading detection and correlation results...")
    
    try:
        # Load detection results
        detection_file = os.path.join(results_dir, "predictions_raw.csv")
        if os.path.exists(detection_file):
            detections_df = pd.read_csv(detection_file)
            total_detections = len(detections_df)
            logger.info(f"🎯 Total detections: {total_detections}")
        else:
            logger.warning(f"⚠️ Detection file not found: {detection_file}")
            total_detections = 0
            detections_df = None
        
        # Load correlation results
        correlation_file = os.path.join(results_dir, "correlation_results", "ais_correlation_results.csv")
        if os.path.exists(correlation_file):
            correlation_df = pd.read_csv(correlation_file)
            
            # Count correlated vessels (non-UNKNOWN MMSI)
            correlated = correlation_df[correlation_df['matched_mmsi'] != 'UNKNOWN']
            correlated_count = len(correlated)
            
            # Get unique correlated MMSIs
            correlated_mmsis = correlated['matched_mmsi'].unique()
            unique_correlated = len(correlated_mmsis)
            
            logger.info(f"🔗 Total correlations: {correlated_count}")
            logger.info(f"🔗 Unique vessels correlated: {unique_correlated}")
            
            return total_detections, correlated_count, unique_correlated, correlated_mmsis
        else:
            logger.warning(f"⚠️ Correlation file not found: {correlation_file}")
            return total_detections, 0, 0, []
            
    except Exception as e:
        logger.error(f"❌ Error loading results: {e}")
        return 0, 0, 0, []

def compare_vessel_lists(ais_vessels, correlated_mmsis):
    """Compare AIS ground truth vessels with correlated vessels."""
    logger.info(f"🔍 Comparing vessel lists...")
    
    # Convert to sets for comparison
    ais_set = set(ais_vessels)
    corr_set = set(correlated_mmsis)
    
    # Find matches and misses
    matched_vessels = ais_set.intersection(corr_set)
    missed_vessels = ais_set.difference(corr_set)
    extra_vessels = corr_set.difference(ais_set)
    
    logger.info(f"✅ Matched vessels: {len(matched_vessels)}")
    logger.info(f"❌ Missed vessels: {len(missed_vessels)}")
    logger.info(f"❓ Extra correlations: {len(extra_vessels)}")
    
    if matched_vessels:
        logger.info(f"🎯 Matched MMSIs: {sorted(matched_vessels)}")
    
    if missed_vessels:
        logger.info(f"😢 Missed MMSIs: {sorted(missed_vessels)}")
        
    if extra_vessels:
        logger.info(f"🤔 Extra MMSIs: {sorted(extra_vessels)}")
    
    return matched_vessels, missed_vessels, extra_vessels

def main():
    """Main analysis function."""
    logger.info("🚀 Starting AIS-SAR Ground Truth Analysis")
    
    # File paths
    safe_folder = "data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    sar_image = "snap_output/step4_final_output.tif"
    ais_data = "ais_data/AIS_175664700472271242_3396-1756647005869.csv"
    results_dir = "maritime_analysis_results/maritime_analysis_20250917_162029"
    
    # Extract SAR timestamp
    sar_timestamp = extract_sar_timestamp_from_safe(safe_folder)
    
    # Get SAR coverage bounds
    bounds = get_sar_coverage_bounds(safe_folder, sar_image)
    
    # Load AIS data
    ais_df = load_ais_data(ais_data)
    if ais_df is None:
        logger.error("❌ Failed to load AIS data")
        return
    
    # Filter AIS data
    spatial_df, temporal_df = filter_ais_by_coverage_and_time(ais_df, bounds, sar_timestamp, 10)
    
    # Analyze unique vessels
    spatial_vessels, temporal_vessels, temporal_vessel_list, time_gaps = analyze_unique_vessels(spatial_df, temporal_df, sar_timestamp)
    
    # Load our results
    total_detections, correlated_count, unique_correlated, correlated_mmsis = load_detection_and_correlation_results(results_dir)
    
    # Compare vessel lists
    matched, missed, extra = compare_vessel_lists(temporal_vessel_list, correlated_mmsis)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 GROUND TRUTH ANALYSIS SUMMARY")
    logger.info("="*80)
    logger.info(f"🌊 AIS GROUND TRUTH (±10min window in SAR coverage):")
    logger.info(f"   Unique vessels present: {temporal_vessels}")
    logger.info(f"   Average time gap to SAR: {np.mean(time_gaps):.2f} minutes")
    logger.info(f"")
    logger.info(f"🎯 OUR DETECTION PERFORMANCE:")
    logger.info(f"   Total detections: {total_detections}")
    logger.info(f"   Total correlations: {correlated_count}")
    logger.info(f"   Unique vessels correlated: {unique_correlated}")
    logger.info(f"")
    logger.info(f"📈 PERFORMANCE METRICS:")
    if temporal_vessels > 0:
        detection_rate = (total_detections / temporal_vessels) * 100
        correlation_rate = (unique_correlated / temporal_vessels) * 100
        logger.info(f"   Detection Rate: {detection_rate:.1f}% ({total_detections}/{temporal_vessels})")
        logger.info(f"   Correlation Rate: {correlation_rate:.1f}% ({unique_correlated}/{temporal_vessels})")
        logger.info(f"   Vessel Capture Rate: {len(matched)}/{temporal_vessels} = {(len(matched)/temporal_vessels)*100:.1f}%")
    else:
        logger.info("   Cannot calculate rates - no ground truth vessels found")
    logger.info(f"")
    logger.info(f"🎯 VESSEL MATCHING:")
    logger.info(f"   Successfully matched: {len(matched)} vessels")
    logger.info(f"   Missed: {len(missed)} vessels")
    logger.info(f"   Extra correlations: {len(extra)} vessels")
    logger.info("="*80)

if __name__ == "__main__":
    main()
