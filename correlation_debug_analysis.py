#!/usr/bin/env python3
"""
Correlation Debug Analysis

Investigates the critical correlation issues:
1. Are we checking the right geographic area?
2. What are the actual timestamps of correlated vessels?
3. Are there MMSI format mismatches?
4. Are we using fake/incorrect correlation results?
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import rasterio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def get_actual_sar_bounds_from_geotiff(sar_image_path):
    """Get actual SAR image bounds from the GeoTIFF file."""
    logger.info(f"📍 Reading actual SAR bounds from: {sar_image_path}")
    
    try:
        with rasterio.open(sar_image_path) as src:
            # Get the bounds
            bounds = src.bounds
            crs = src.crs
            transform = src.transform
            shape = src.shape
            
            logger.info(f"✅ SAR Image Properties:")
            logger.info(f"   CRS: {crs}")
            logger.info(f"   Shape: {shape} (height, width)")
            logger.info(f"   Transform: {transform}")
            logger.info(f"   Bounds: {bounds}")
            logger.info(f"   Min X (West): {bounds.left}")
            logger.info(f"   Max X (East): {bounds.right}")  
            logger.info(f"   Min Y (South): {bounds.bottom}")
            logger.info(f"   Max Y (North): {bounds.top}")
            
            # Convert to lat/lon if needed
            if crs.to_string() == 'EPSG:4326':
                actual_bounds = {
                    'min_lat': bounds.bottom,
                    'max_lat': bounds.top,
                    'min_lon': bounds.left,
                    'max_lon': bounds.right
                }
                logger.info(f"📍 Actual SAR Coverage (Lat/Lon):")
                logger.info(f"   Latitude: {actual_bounds['min_lat']:.6f} to {actual_bounds['max_lat']:.6f}")
                logger.info(f"   Longitude: {actual_bounds['min_lon']:.6f} to {actual_bounds['max_lon']:.6f}")
            else:
                logger.warning(f"⚠️ SAR image is not in EPSG:4326. CRS: {crs}")
                # Use approximate bounds for now
                actual_bounds = {
                    'min_lat': 35.9,
                    'max_lat': 37.6,
                    'min_lon': -77.3,
                    'max_lon': -75.5
                }
                logger.info(f"📍 Using approximate bounds for analysis")
            
            return actual_bounds, crs, shape
            
    except Exception as e:
        logger.error(f"❌ Error reading SAR bounds: {e}")
        # Fallback bounds
        return {
            'min_lat': 35.9,
            'max_lat': 37.6,
            'min_lon': -77.3,
            'max_lon': -75.5
        }, None, None

def compare_coverage_areas(ground_truth_bounds, correlation_bounds):
    """Compare the coverage areas used in different analyses."""
    logger.info(f"🔍 Comparing coverage areas...")
    
    logger.info(f"📊 Ground Truth Analysis Bounds:")
    logger.info(f"   Lat: {ground_truth_bounds['min_lat']:.6f} to {ground_truth_bounds['max_lat']:.6f}")
    logger.info(f"   Lon: {ground_truth_bounds['min_lon']:.6f} to {ground_truth_bounds['max_lon']:.6f}")
    
    logger.info(f"📊 Correlation System Bounds:")
    logger.info(f"   Lat: {correlation_bounds['min_lat']:.6f} to {correlation_bounds['max_lat']:.6f}")
    logger.info(f"   Lon: {correlation_bounds['min_lon']:.6f} to {correlation_bounds['max_lon']:.6f}")
    
    # Check if they match
    lat_match = (abs(ground_truth_bounds['min_lat'] - correlation_bounds['min_lat']) < 0.01 and
                abs(ground_truth_bounds['max_lat'] - correlation_bounds['max_lat']) < 0.01)
    lon_match = (abs(ground_truth_bounds['min_lon'] - correlation_bounds['min_lon']) < 0.01 and
                abs(ground_truth_bounds['max_lon'] - correlation_bounds['max_lon']) < 0.01)
    
    if lat_match and lon_match:
        logger.info(f"✅ Coverage areas MATCH - Same geographic region")
    else:
        logger.warning(f"⚠️ Coverage areas DIFFER - Different geographic regions!")
        logger.warning(f"   Lat difference: {abs(ground_truth_bounds['min_lat'] - correlation_bounds['min_lat']):.6f}")
        logger.warning(f"   Lon difference: {abs(ground_truth_bounds['min_lon'] - correlation_bounds['min_lon']):.6f}")

def analyze_correlation_results_in_detail(correlation_file, ais_file, sar_timestamp):
    """Analyze the correlation results in detail."""
    logger.info(f"🔍 Analyzing correlation results in detail...")
    
    try:
        # Load correlation results
        corr_df = pd.read_csv(correlation_file)
        logger.info(f"✅ Loaded {len(corr_df)} correlation results")
        
        # Load full AIS data
        ais_df = pd.read_csv(ais_file)
        ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
        logger.info(f"✅ Loaded {len(ais_df)} AIS records")
        
        # Analyze correlated vessels
        correlated = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']
        logger.info(f"📊 Correlated vessels: {len(correlated)}")
        
        if len(correlated) > 0:
            logger.info(f"🔍 Analyzing each correlated vessel:")
            
            for idx, row in correlated.iterrows():
                mmsi = row['matched_mmsi']
                time_gap = row['time_gap_minutes']
                
                # Find this MMSI in AIS data
                mmsi_data = ais_df[ais_df['MMSI'] == mmsi]
                
                if len(mmsi_data) > 0:
                    # Find closest record to SAR time
                    time_diffs = abs((mmsi_data['BaseDateTime'] - sar_timestamp).dt.total_seconds() / 60.0)
                    closest_idx = time_diffs.idxmin()
                    closest_record = mmsi_data.loc[closest_idx]
                    actual_time_gap = time_diffs.loc[closest_idx]
                    
                    logger.info(f"   MMSI {mmsi}:")
                    logger.info(f"     Correlation time gap: {time_gap:.2f} min")
                    logger.info(f"     Actual closest gap: {actual_time_gap:.2f} min")
                    logger.info(f"     Position: {closest_record['LAT']:.6f}, {closest_record['LON']:.6f}")
                    logger.info(f"     Timestamp: {closest_record['BaseDateTime']}")
                    
                    # Check if within ±10min window
                    if actual_time_gap <= 10.0:
                        logger.info(f"     ✅ Within ±10min window")
                    else:
                        logger.warning(f"     ⚠️ OUTSIDE ±10min window!")
                else:
                    logger.error(f"   ❌ MMSI {mmsi} NOT FOUND in AIS data!")
        
        return correlated
        
    except Exception as e:
        logger.error(f"❌ Error analyzing correlation results: {e}")
        return None

def check_mmsi_format_consistency(ground_truth_mmsis, correlation_mmsis):
    """Check for MMSI format inconsistencies."""
    logger.info(f"🔍 Checking MMSI format consistency...")
    
    logger.info(f"📊 Ground Truth MMSIs:")
    logger.info(f"   Count: {len(ground_truth_mmsis)}")
    logger.info(f"   Type: {type(ground_truth_mmsis[0]) if len(ground_truth_mmsis) > 0 else 'None'}")
    logger.info(f"   Sample: {ground_truth_mmsis[:5] if len(ground_truth_mmsis) > 0 else 'None'}")
    
    logger.info(f"📊 Correlation MMSIs:")
    logger.info(f"   Count: {len(correlation_mmsis)}")
    logger.info(f"   Type: {type(correlation_mmsis[0]) if len(correlation_mmsis) > 0 else 'None'}")
    logger.info(f"   Sample: {correlation_mmsis[:5] if len(correlation_mmsis) > 0 else 'None'}")
    
    # Convert both to same format for comparison
    try:
        gt_set = set([str(mmsi) for mmsi in ground_truth_mmsis])
        corr_set = set([str(mmsi) for mmsi in correlation_mmsis])
        
        overlap = gt_set.intersection(corr_set)
        logger.info(f"🔍 After format normalization:")
        logger.info(f"   Overlap: {len(overlap)} MMSIs")
        
        if len(overlap) > 0:
            logger.info(f"   ✅ Found overlapping MMSIs: {list(overlap)[:10]}")
        else:
            logger.warning(f"   ⚠️ Still NO overlap after format normalization")
            
        return overlap
        
    except Exception as e:
        logger.error(f"❌ Error checking MMSI formats: {e}")
        return set()

def investigate_detection_positions(predictions_file, ais_file, sar_timestamp, bounds):
    """Investigate if our detections are in reasonable positions."""
    logger.info(f"🔍 Investigating detection positions...")
    
    try:
        # Load predictions
        pred_df = pd.read_csv(predictions_file)
        logger.info(f"✅ Loaded {len(pred_df)} predictions")
        
        # Load AIS data for ±10min window
        ais_df = pd.read_csv(ais_file)
        ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
        
        # Filter AIS to ±10min window and coverage area
        time_delta = timedelta(minutes=10)
        time_filter = (
            (ais_df['BaseDateTime'] >= sar_timestamp - time_delta) &
            (ais_df['BaseDateTime'] <= sar_timestamp + time_delta)
        )
        spatial_filter = (
            (ais_df['LAT'] >= bounds['min_lat']) &
            (ais_df['LAT'] <= bounds['max_lat']) &
            (ais_df['LON'] >= bounds['min_lon']) &
            (ais_df['LON'] <= bounds['max_lon'])
        )
        
        ground_truth_ais = ais_df[time_filter & spatial_filter]
        logger.info(f"✅ Ground truth: {len(ground_truth_ais)} AIS records in ±10min window")
        
        # Check detection positions
        det_in_bounds = pred_df[
            (pred_df['lat'] >= bounds['min_lat']) &
            (pred_df['lat'] <= bounds['max_lat']) &
            (pred_df['lon'] >= bounds['min_lon']) &
            (pred_df['lon'] <= bounds['max_lon'])
        ]
        
        logger.info(f"📊 Detection Analysis:")
        logger.info(f"   Total detections: {len(pred_df)}")
        logger.info(f"   Detections in bounds: {len(det_in_bounds)}")
        logger.info(f"   Detection lat range: {pred_df['lat'].min():.6f} to {pred_df['lat'].max():.6f}")
        logger.info(f"   Detection lon range: {pred_df['lon'].min():.6f} to {pred_df['lon'].max():.6f}")
        logger.info(f"   AIS lat range: {ground_truth_ais['LAT'].min():.6f} to {ground_truth_ais['LAT'].max():.6f}")
        logger.info(f"   AIS lon range: {ground_truth_ais['LON'].min():.6f} to {ground_truth_ais['LON'].max():.6f}")
        
        # Calculate distances between detections and AIS positions
        if len(ground_truth_ais) > 0 and len(pred_df) > 0:
            logger.info(f"🔍 Calculating closest detection-AIS distances...")
            
            min_distances = []
            for _, detection in pred_df.head(10).iterrows():  # Check first 10 detections
                det_lat, det_lon = detection['lat'], detection['lon']
                
                # Calculate distances to all AIS positions
                distances = []
                for _, ais_record in ground_truth_ais.iterrows():
                    ais_lat, ais_lon = ais_record['LAT'], ais_record['LON']
                    
                    # Haversine distance approximation
                    lat_diff = (det_lat - ais_lat) * 111000  # ~111km per degree
                    lon_diff = (det_lon - ais_lon) * 111000 * np.cos(np.radians(det_lat))
                    distance = np.sqrt(lat_diff**2 + lon_diff**2)
                    distances.append(distance)
                
                if distances:
                    min_dist = min(distances)
                    min_distances.append(min_dist)
                    logger.info(f"   Detection ({det_lat:.6f}, {det_lon:.6f}): closest AIS {min_dist:.0f}m")
            
            if min_distances:
                avg_min_dist = np.mean(min_distances)
                logger.info(f"📊 Average minimum distance: {avg_min_dist:.0f}m")
        
        return det_in_bounds, ground_truth_ais
        
    except Exception as e:
        logger.error(f"❌ Error investigating detection positions: {e}")
        return None, None

def main():
    """Main debugging function."""
    logger.info("🚨 Starting Correlation Debug Analysis")
    
    # File paths
    sar_image = "snap_output/step4_final_output.tif"
    ais_file = "ais_data/AIS_175664700472271242_3396-1756647005869.csv"
    results_dir = "maritime_analysis_results/maritime_analysis_20250917_162029"
    correlation_file = os.path.join(results_dir, "correlation_results", "ais_correlation_results.csv")
    predictions_file = os.path.join(results_dir, "predictions_raw.csv")
    
    # SAR timestamp
    sar_timestamp = datetime(2023, 6, 20, 23, 6, 42)
    logger.info(f"📅 SAR timestamp: {sar_timestamp}")
    
    # 1. GET ACTUAL SAR COVERAGE BOUNDS
    logger.info("\n" + "="*80)
    logger.info("🗺️ STEP 1: VERIFY SAR COVERAGE BOUNDS")
    logger.info("="*80)
    
    actual_bounds, crs, shape = get_actual_sar_bounds_from_geotiff(sar_image)
    
    # Ground truth analysis used these bounds (from previous run)
    ground_truth_bounds = {
        'min_lat': 35.9,
        'max_lat': 37.6,
        'min_lon': -77.3,
        'max_lon': -75.5
    }
    
    compare_coverage_areas(ground_truth_bounds, actual_bounds)
    
    # 2. ANALYZE CORRELATION RESULTS IN DETAIL
    logger.info("\n" + "="*80)
    logger.info("🔍 STEP 2: ANALYZE CORRELATION RESULTS")
    logger.info("="*80)
    
    correlated_vessels = analyze_correlation_results_in_detail(correlation_file, ais_file, sar_timestamp)
    
    # 3. CHECK MMSI FORMAT CONSISTENCY
    logger.info("\n" + "="*80)
    logger.info("🔍 STEP 3: CHECK MMSI FORMAT CONSISTENCY")
    logger.info("="*80)
    
    # Load ground truth MMSIs (±10min window)
    ais_df = pd.read_csv(ais_file)
    ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
    
    time_delta = timedelta(minutes=10)
    time_filter = (
        (ais_df['BaseDateTime'] >= sar_timestamp - time_delta) &
        (ais_df['BaseDateTime'] <= sar_timestamp + time_delta)
    )
    spatial_filter = (
        (ais_df['LAT'] >= actual_bounds['min_lat']) &
        (ais_df['LAT'] <= actual_bounds['max_lat']) &
        (ais_df['LON'] >= actual_bounds['min_lon']) &
        (ais_df['LON'] <= actual_bounds['max_lon'])
    )
    
    ground_truth_ais = ais_df[time_filter & spatial_filter]
    ground_truth_mmsis = ground_truth_ais['MMSI'].unique()
    
    # Load correlation MMSIs
    corr_df = pd.read_csv(correlation_file)
    correlation_mmsis = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']['matched_mmsi'].unique()
    
    overlap = check_mmsi_format_consistency(ground_truth_mmsis, correlation_mmsis)
    
    # 4. INVESTIGATE DETECTION POSITIONS
    logger.info("\n" + "="*80)
    logger.info("🎯 STEP 4: INVESTIGATE DETECTION POSITIONS")
    logger.info("="*80)
    
    det_in_bounds, ground_truth_ais_final = investigate_detection_positions(
        predictions_file, ais_file, sar_timestamp, actual_bounds
    )
    
    # 5. FINAL SUMMARY
    logger.info("\n" + "="*80)
    logger.info("🚨 CORRELATION DEBUG SUMMARY")
    logger.info("="*80)
    
    logger.info(f"🗺️ COVERAGE ANALYSIS:")
    logger.info(f"   SAR bounds verified: {'✅' if actual_bounds else '❌'}")
    logger.info(f"   Coverage area consistency: Check logs above")
    
    logger.info(f"📊 CORRELATION ANALYSIS:")
    logger.info(f"   Correlated vessels analyzed: {len(correlated_vessels) if correlated_vessels is not None else 0}")
    logger.info(f"   Vessels within ±10min: Check individual analysis above")
    
    logger.info(f"🔍 FORMAT ANALYSIS:")
    logger.info(f"   MMSI format overlap found: {len(overlap) if overlap else 0}")
    
    logger.info(f"🎯 DETECTION ANALYSIS:")
    logger.info(f"   Detections in coverage area: {len(det_in_bounds) if det_in_bounds is not None else 'Unknown'}")
    logger.info(f"   Ground truth vessels: {len(ground_truth_ais_final) if ground_truth_ais_final is not None else 'Unknown'}")
    
    logger.info("="*80)
    logger.info("🔍 Check the detailed logs above for specific issues!")

if __name__ == "__main__":
    main()
