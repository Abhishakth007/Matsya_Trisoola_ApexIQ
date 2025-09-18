#!/usr/bin/env python3
"""
Correlation Pattern Analysis

Deep dive into the 28 successful correlations vs 91 missed correlations.
Identify patterns in:
1. Vessel characteristics (size, speed, type)
2. Spatial distribution 
3. Temporal gaps
4. Detection confidence scores
5. AIS data quality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load all necessary data files."""
    logger.info("📊 Loading data files...")
    
    # Load AIS data
    ais_df = pd.read_csv("ais_data/AIS_175664700472271242_3396-1756647005869.csv")
    ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
    logger.info(f"✅ AIS data: {len(ais_df)} records")
    
    # Load predictions
    pred_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/predictions_raw.csv")
    logger.info(f"✅ Predictions: {len(pred_df)} detections")
    
    # Load correlation results
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    logger.info(f"✅ Correlation results: {len(corr_df)} records")
    
    return ais_df, pred_df, corr_df

def get_ground_truth_vessels(ais_df, sar_timestamp):
    """Get ground truth vessels in ±10min window."""
    real_bounds = {
        'min_lat': 35.713299,
        'max_lat': 37.623062,
        'min_lon': -78.501892,
        'max_lon': -75.307404
    }
    
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
        time_diffs = abs((vessel_records['BaseDateTime'] - sar_timestamp).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        unique_vessels.append(vessel_records.loc[closest_idx])
    
    return pd.DataFrame(unique_vessels)

def analyze_correlation_patterns(ground_truth_df, corr_df):
    """Analyze patterns in successful vs failed correlations."""
    logger.info("🔍 Analyzing correlation patterns...")
    
    # Identify successful correlations
    successful_mmsis = set([str(mmsi) for mmsi in corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']['matched_mmsi'].unique()])
    
    # Split ground truth into correlated vs missed
    ground_truth_df['mmsi_str'] = ground_truth_df['MMSI'].astype(str)
    correlated_vessels = ground_truth_df[ground_truth_df['mmsi_str'].isin(successful_mmsis)]
    missed_vessels = ground_truth_df[~ground_truth_df['mmsi_str'].isin(successful_mmsis)]
    
    logger.info(f"📊 Correlation Breakdown:")
    logger.info(f"   Successfully correlated: {len(correlated_vessels)} vessels")
    logger.info(f"   Missed correlations: {len(missed_vessels)} vessels")
    
    return correlated_vessels, missed_vessels

def analyze_vessel_characteristics(correlated_vessels, missed_vessels):
    """Analyze vessel characteristics patterns."""
    logger.info("\n🚢 VESSEL CHARACTERISTICS ANALYSIS:")
    
    # Speed analysis
    logger.info(f"\n📈 SPEED PATTERNS:")
    
    corr_speeds = correlated_vessels['SOG'].describe()
    missed_speeds = missed_vessels['SOG'].describe()
    
    logger.info(f"   Correlated vessels speed stats:")
    logger.info(f"     Mean: {corr_speeds['mean']:.2f} knots")
    logger.info(f"     Median: {corr_speeds['50%']:.2f} knots") 
    logger.info(f"     Std: {corr_speeds['std']:.2f} knots")
    logger.info(f"     Range: {corr_speeds['min']:.2f} - {corr_speeds['max']:.2f} knots")
    
    logger.info(f"   Missed vessels speed stats:")
    logger.info(f"     Mean: {missed_speeds['mean']:.2f} knots")
    logger.info(f"     Median: {missed_speeds['50%']:.2f} knots")
    logger.info(f"     Std: {missed_speeds['std']:.2f} knots") 
    logger.info(f"     Range: {missed_speeds['min']:.2f} - {missed_speeds['max']:.2f} knots")
    
    # Speed categories
    corr_stationary = len(correlated_vessels[correlated_vessels['SOG'] <= 0.5])
    corr_moving = len(correlated_vessels[correlated_vessels['SOG'] > 0.5])
    missed_stationary = len(missed_vessels[missed_vessels['SOG'] <= 0.5])
    missed_moving = len(missed_vessels[missed_vessels['SOG'] > 0.5])
    
    logger.info(f"\n📊 SPEED CATEGORY BREAKDOWN:")
    logger.info(f"   Correlated vessels:")
    logger.info(f"     Stationary (≤0.5 knots): {corr_stationary} ({corr_stationary/len(correlated_vessels)*100:.1f}%)")
    logger.info(f"     Moving (>0.5 knots): {corr_moving} ({corr_moving/len(correlated_vessels)*100:.1f}%)")
    logger.info(f"   Missed vessels:")
    logger.info(f"     Stationary (≤0.5 knots): {missed_stationary} ({missed_stationary/len(missed_vessels)*100:.1f}%)")
    logger.info(f"     Moving (>0.5 knots): {missed_moving} ({missed_moving/len(missed_vessels)*100:.1f}%)")
    
    # Vessel size analysis (if available)
    if 'Length' in correlated_vessels.columns:
        logger.info(f"\n📏 VESSEL SIZE PATTERNS:")
        
        corr_sizes = correlated_vessels['Length'].dropna()
        missed_sizes = missed_vessels['Length'].dropna()
        
        if len(corr_sizes) > 0:
            logger.info(f"   Correlated vessels size: {corr_sizes.mean():.1f}m avg (n={len(corr_sizes)})")
        if len(missed_sizes) > 0:
            logger.info(f"   Missed vessels size: {missed_sizes.mean():.1f}m avg (n={len(missed_sizes)})")
    
    # Vessel type analysis
    if 'VesselType' in correlated_vessels.columns:
        logger.info(f"\n🏷️ VESSEL TYPE PATTERNS:")
        
        corr_types = correlated_vessels['VesselType'].value_counts().head(5)
        missed_types = missed_vessels['VesselType'].value_counts().head(5)
        
        logger.info(f"   Top correlated vessel types:")
        for vtype, count in corr_types.items():
            logger.info(f"     Type {vtype}: {count} vessels ({count/len(correlated_vessels)*100:.1f}%)")
        
        logger.info(f"   Top missed vessel types:")
        for vtype, count in missed_types.items():
            logger.info(f"     Type {vtype}: {count} vessels ({count/len(missed_vessels)*100:.1f}%)")

def analyze_correlation_scores(corr_df):
    """Analyze correlation scoring patterns."""
    logger.info(f"\n🎯 CORRELATION SCORING ANALYSIS:")
    
    successful_corr = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']
    
    if len(successful_corr) > 0:
        logger.info(f"   Successful correlations: {len(successful_corr)}")
        logger.info(f"   Match confidence range: {successful_corr['match_confidence'].min():.3f} - {successful_corr['match_confidence'].max():.3f}")
        logger.info(f"   Average match confidence: {successful_corr['match_confidence'].mean():.3f}")
        
        # Time gap analysis
        logger.info(f"   Time gap range: {successful_corr['time_gap_minutes'].min():.2f} - {successful_corr['time_gap_minutes'].max():.2f} minutes")
        logger.info(f"   Average time gap: {successful_corr['time_gap_minutes'].mean():.2f} minutes")
        
        # Distance analysis
        logger.info(f"   Distance range: {successful_corr['distance_meters'].min():.1f} - {successful_corr['distance_meters'].max():.1f} meters")
        logger.info(f"   Average distance: {successful_corr['distance_meters'].mean():.1f} meters")
        
        # Score component analysis
        score_components = ['position_score', 'heading_score', 'size_score', 'speed_score', 'type_score', 'temporal_score']
        
        logger.info(f"\n📊 SCORE COMPONENT ANALYSIS:")
        for component in score_components:
            if component in successful_corr.columns:
                avg_score = successful_corr[component].mean()
                logger.info(f"   Average {component}: {avg_score:.3f}")

def analyze_spatial_patterns(correlated_vessels, missed_vessels):
    """Analyze spatial distribution patterns."""
    logger.info(f"\n🗺️ SPATIAL DISTRIBUTION ANALYSIS:")
    
    # Geographic spread
    logger.info(f"   Correlated vessels geographic range:")
    logger.info(f"     Lat: {correlated_vessels['LAT'].min():.6f} - {correlated_vessels['LAT'].max():.6f}")
    logger.info(f"     Lon: {correlated_vessels['LON'].min():.6f} - {correlated_vessels['LON'].max():.6f}")
    
    logger.info(f"   Missed vessels geographic range:")
    logger.info(f"     Lat: {missed_vessels['LAT'].min():.6f} - {missed_vessels['LAT'].max():.6f}")
    logger.info(f"     Lon: {missed_vessels['LON'].min():.6f} - {missed_vessels['LON'].max():.6f}")
    
    # Calculate centroids
    corr_center_lat = correlated_vessels['LAT'].mean()
    corr_center_lon = correlated_vessels['LON'].mean()
    missed_center_lat = missed_vessels['LAT'].mean()
    missed_center_lon = missed_vessels['LON'].mean()
    
    logger.info(f"   Correlated vessels centroid: ({corr_center_lat:.6f}, {corr_center_lon:.6f})")
    logger.info(f"   Missed vessels centroid: ({missed_center_lat:.6f}, {missed_center_lon:.6f})")

def analyze_detection_confidence_patterns(pred_df, corr_df):
    """Analyze if detection confidence affects correlation success."""
    logger.info(f"\n🎯 DETECTION CONFIDENCE vs CORRELATION ANALYSIS:")
    
    # Get detection IDs that were successfully correlated
    successful_corr = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']
    correlated_detect_ids = successful_corr['detect_id'].unique()
    
    # Split predictions by correlation success
    correlated_detections = pred_df[pred_df['detect_id'].isin(correlated_detect_ids)]
    missed_detections = pred_df[~pred_df['detect_id'].isin(correlated_detect_ids)]
    
    logger.info(f"   Correlated detections: {len(correlated_detections)}")
    logger.info(f"   Missed detections: {len(missed_detections)}")
    
    # Confidence analysis
    logger.info(f"   Correlated detection confidence:")
    logger.info(f"     Mean: {correlated_detections['confidence'].mean():.3f}")
    logger.info(f"     Range: {correlated_detections['confidence'].min():.3f} - {correlated_detections['confidence'].max():.3f}")
    
    logger.info(f"   Missed detection confidence:")
    logger.info(f"     Mean: {missed_detections['confidence'].mean():.3f}")
    logger.info(f"     Range: {missed_detections['confidence'].min():.3f} - {missed_detections['confidence'].max():.3f}")
    
    # Confidence categories
    high_conf_corr = len(correlated_detections[correlated_detections['confidence'] >= 0.8])
    high_conf_missed = len(missed_detections[missed_detections['confidence'] >= 0.8])
    
    logger.info(f"   High confidence (≥0.8) correlations: {high_conf_corr}/{len(correlated_detections)} ({high_conf_corr/len(correlated_detections)*100:.1f}%)")
    logger.info(f"   High confidence (≥0.8) missed: {high_conf_missed}/{len(missed_detections)} ({high_conf_missed/len(missed_detections)*100:.1f}%)")

def main():
    """Main analysis function."""
    logger.info("🚀 Starting Correlation Pattern Analysis")
    
    sar_timestamp = datetime(2023, 6, 20, 23, 6, 42)
    
    # Load data
    ais_df, pred_df, corr_df = load_data()
    
    # Get ground truth vessels
    ground_truth_df = get_ground_truth_vessels(ais_df, sar_timestamp)
    logger.info(f"📊 Ground truth vessels: {len(ground_truth_df)}")
    
    # Analyze correlation patterns
    correlated_vessels, missed_vessels = analyze_correlation_patterns(ground_truth_df, corr_df)
    
    # Detailed analyses
    analyze_vessel_characteristics(correlated_vessels, missed_vessels)
    analyze_correlation_scores(corr_df)
    analyze_spatial_patterns(correlated_vessels, missed_vessels)
    analyze_detection_confidence_patterns(pred_df, corr_df)
    
    # Summary insights
    logger.info(f"\n🎯 KEY INSIGHTS SUMMARY:")
    logger.info(f"   Total ground truth vessels: {len(ground_truth_df)}")
    logger.info(f"   Successfully correlated: {len(correlated_vessels)} ({len(correlated_vessels)/len(ground_truth_df)*100:.1f}%)")
    logger.info(f"   Missed correlations: {len(missed_vessels)} ({len(missed_vessels)/len(ground_truth_df)*100:.1f}%)")
    
    # Check for obvious patterns
    if len(correlated_vessels) > 0 and len(missed_vessels) > 0:
        corr_avg_speed = correlated_vessels['SOG'].mean()
        missed_avg_speed = missed_vessels['SOG'].mean()
        
        logger.info(f"\n💡 PATTERN OBSERVATIONS:")
        logger.info(f"   Correlated vessels avg speed: {corr_avg_speed:.2f} knots")
        logger.info(f"   Missed vessels avg speed: {missed_avg_speed:.2f} knots")
        
        if corr_avg_speed > missed_avg_speed * 2:
            logger.info(f"   🚨 PATTERN FOUND: Correlated vessels are significantly faster!")
        elif missed_avg_speed > corr_avg_speed * 2:
            logger.info(f"   🚨 PATTERN FOUND: Missed vessels are significantly faster!")
        else:
            logger.info(f"   ℹ️ No obvious speed pattern difference")

if __name__ == "__main__":
    main()
