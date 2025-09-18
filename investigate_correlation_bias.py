#!/usr/bin/env python3
"""
Investigate Correlation Bias

Deep dive into WHY the correlation system is biased towards large vessels.
Check every step of the correlation pipeline for systematic bias:

1. Detection bias - Are we only detecting large vessels?
2. Spatial gating bias - Do large vessels get preferential spatial treatment?
3. Scoring bias - Do large vessels get higher correlation scores?
4. Temporal bias - Do large vessels get better temporal matching?
5. Multi-cue scoring bias - Are scoring weights favoring large vessels?
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_all_data():
    """Load all pipeline data."""
    logger.info("📊 Loading all pipeline data...")
    
    # AIS ground truth
    ais_df = pd.read_csv("ais_data/AIS_175664700472271242_3396-1756647005869.csv")
    ais_df['BaseDateTime'] = pd.to_datetime(ais_df['BaseDateTime'])
    
    # Predictions (detections)
    pred_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/predictions_raw.csv")
    
    # Correlation results
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    
    logger.info(f"✅ AIS records: {len(ais_df)}")
    logger.info(f"✅ Detections: {len(pred_df)}")
    logger.info(f"✅ Correlation records: {len(corr_df)}")
    
    return ais_df, pred_df, corr_df

def get_ground_truth_with_sizes(ais_df, sar_timestamp):
    """Get ground truth vessels with their sizes."""
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
    
    filtered_ais = ais_df[time_filter & spatial_filter]
    
    # Get unique vessels with all their data
    unique_vessels = []
    for mmsi in filtered_ais['MMSI'].unique():
        vessel_records = filtered_ais[filtered_ais['MMSI'] == mmsi]
        time_diffs = abs((vessel_records['BaseDateTime'] - sar_timestamp).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        unique_vessels.append(vessel_records.loc[closest_idx])
    
    ground_truth_df = pd.DataFrame(unique_vessels)
    
    # Add size categories
    ground_truth_df['size_category'] = ground_truth_df['Length'].apply(lambda x: 
        'Large (>100m)' if pd.notna(x) and x > 100 else
        'Medium (50-100m)' if pd.notna(x) and x >= 50 else
        'Small (<50m)' if pd.notna(x) and x < 50 else
        'Unknown size'
    )
    
    return ground_truth_df

def analyze_detection_bias(pred_df, ground_truth_df):
    """Check if detection system is biased towards large vessels."""
    logger.info("\n🔍 STEP 1: DETECTION BIAS ANALYSIS")
    
    # Check if we have vessel size info in predictions
    size_columns = [col for col in pred_df.columns if 'length' in col.lower() or 'size' in col.lower()]
    logger.info(f"📏 Size-related columns in predictions: {size_columns}")
    
    if 'vessel_length_m' in pred_df.columns:
        logger.info(f"\n📊 PREDICTED VESSEL SIZES:")
        pred_sizes = pred_df['vessel_length_m'].dropna()
        logger.info(f"   Detected vessel size range: {pred_sizes.min():.1f} - {pred_sizes.max():.1f}m")
        logger.info(f"   Average detected vessel size: {pred_sizes.mean():.1f}m")
        logger.info(f"   Median detected vessel size: {pred_sizes.median():.1f}m")
        
        # Size distribution
        large_detected = len(pred_df[pred_df['vessel_length_m'] > 100])
        medium_detected = len(pred_df[(pred_df['vessel_length_m'] >= 50) & (pred_df['vessel_length_m'] <= 100)])
        small_detected = len(pred_df[pred_df['vessel_length_m'] < 50])
        
        logger.info(f"\n📈 DETECTION SIZE DISTRIBUTION:")
        logger.info(f"   Large vessels (>100m) detected: {large_detected} ({large_detected/len(pred_df)*100:.1f}%)")
        logger.info(f"   Medium vessels (50-100m) detected: {medium_detected} ({medium_detected/len(pred_df)*100:.1f}%)")
        logger.info(f"   Small vessels (<50m) detected: {small_detected} ({small_detected/len(pred_df)*100:.1f}%)")
    
    # Compare with ground truth
    logger.info(f"\n📊 GROUND TRUTH SIZE DISTRIBUTION:")
    gt_size_dist = ground_truth_df['size_category'].value_counts()
    for category, count in gt_size_dist.items():
        logger.info(f"   {category}: {count} vessels ({count/len(ground_truth_df)*100:.1f}%)")
    
    # Check detection confidence by size
    if 'vessel_length_m' in pred_df.columns:
        logger.info(f"\n🎯 DETECTION CONFIDENCE BY VESSEL SIZE:")
        large_conf = pred_df[pred_df['vessel_length_m'] > 100]['confidence'].mean()
        medium_conf = pred_df[(pred_df['vessel_length_m'] >= 50) & (pred_df['vessel_length_m'] <= 100)]['confidence'].mean()
        small_conf = pred_df[pred_df['vessel_length_m'] < 50]['confidence'].mean()
        
        logger.info(f"   Large vessels (>100m) avg confidence: {large_conf:.3f}")
        logger.info(f"   Medium vessels (50-100m) avg confidence: {medium_conf:.3f}")
        logger.info(f"   Small vessels (<50m) avg confidence: {small_conf:.3f}")
        
        if large_conf > small_conf + 0.1:
            logger.warning(f"   🚨 BIAS DETECTED: Large vessels have {large_conf - small_conf:.3f} higher confidence!")

def analyze_correlation_scoring_bias(corr_df, ground_truth_df):
    """Analyze if correlation scoring is biased towards large vessels."""
    logger.info("\n🔍 STEP 2: CORRELATION SCORING BIAS ANALYSIS")
    
    # Get successful correlations with vessel info
    successful_corr = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN'].copy()
    
    # Merge with ground truth to get vessel sizes
    successful_corr['matched_mmsi_int'] = successful_corr['matched_mmsi'].astype(str)
    ground_truth_df['MMSI_str'] = ground_truth_df['MMSI'].astype(str)
    
    merged_corr = successful_corr.merge(
        ground_truth_df[['MMSI_str', 'Length', 'VesselType', 'size_category']], 
        left_on='matched_mmsi_int', 
        right_on='MMSI_str', 
        how='left'
    )
    
    logger.info(f"📊 CORRELATION SCORES BY VESSEL SIZE:")
    
    # Analyze scores by size category
    if 'Length' in merged_corr.columns:
        size_categories = ['Large (>100m)', 'Medium (50-100m)', 'Small (<50m)']
        
        for category in size_categories:
            cat_data = merged_corr[merged_corr['size_category'] == category]
            if len(cat_data) > 0:
                logger.info(f"\n   {category} vessels ({len(cat_data)} correlations):")
                logger.info(f"     Match confidence: {cat_data['match_confidence'].mean():.3f}")
                logger.info(f"     Position score: {cat_data['position_score'].mean():.3f}")
                logger.info(f"     Size score: {cat_data['size_score'].mean():.3f}")
                logger.info(f"     Speed score: {cat_data['speed_score'].mean():.3f}")
                logger.info(f"     Type score: {cat_data['type_score'].mean():.3f}")
                logger.info(f"     Average distance: {cat_data['distance_meters'].mean():.1f}m")
            else:
                logger.info(f"\n   {category} vessels: 0 correlations")

def analyze_spatial_gating_bias(corr_df):
    """Check if spatial gating is biased."""
    logger.info("\n🔍 STEP 3: SPATIAL GATING BIAS ANALYSIS")
    
    # Check distance patterns
    successful_corr = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']
    failed_corr = corr_df[corr_df['matched_mmsi'] == 'UNKNOWN']
    
    logger.info(f"📊 SPATIAL DISTANCE PATTERNS:")
    logger.info(f"   Successful correlations:")
    logger.info(f"     Distance range: {successful_corr['distance_meters'].min():.1f} - {successful_corr['distance_meters'].max():.1f}m")
    logger.info(f"     Average distance: {successful_corr['distance_meters'].mean():.1f}m")
    logger.info(f"     Median distance: {successful_corr['distance_meters'].median():.1f}m")
    
    # Check if there are failed correlations with distance info
    if 'distance_meters' in failed_corr.columns and not failed_corr['distance_meters'].isna().all():
        logger.info(f"   Failed correlations:")
        logger.info(f"     Distance range: {failed_corr['distance_meters'].min():.1f} - {failed_corr['distance_meters'].max():.1f}m")
        logger.info(f"     Average distance: {failed_corr['distance_meters'].mean():.1f}m")
    else:
        logger.info(f"   Failed correlations: No distance data (filtered out by spatial gating)")

def check_correlation_algorithm_bias():
    """Check the correlation algorithm configuration for bias."""
    logger.info("\n🔍 STEP 4: CORRELATION ALGORITHM BIAS CHECK")
    
    # Read correlation engine source to check for bias
    try:
        with open("ais_correlation/correlation_engine.py", "r") as f:
            content = f.read()
        
        # Check scoring weights
        if "weights" in content:
            logger.info("📊 Checking correlation scoring weights...")
            
            # Look for weight definitions
            import re
            weight_pattern = r"weights.*?=.*?\{([^}]+)\}"
            matches = re.search(weight_pattern, content, re.DOTALL)
            if matches:
                logger.info(f"   Found scoring weights in code")
            
        # Check for size-based filtering
        size_keywords = ['length', 'size', 'Length', 'large', 'small']
        for keyword in size_keywords:
            if keyword in content:
                logger.info(f"   Found '{keyword}' references in correlation code")
        
        # Check for vessel type filtering
        type_keywords = ['VesselType', 'type', 'cargo', 'fishing']
        for keyword in type_keywords:
            if keyword in content:
                logger.info(f"   Found '{keyword}' references in correlation code")
                
    except FileNotFoundError:
        logger.warning("⚠️ Could not read correlation engine source code")

def analyze_missing_correlations_by_size(ground_truth_df, corr_df):
    """Analyze which vessels are missing correlations by size."""
    logger.info("\n🔍 STEP 5: MISSING CORRELATIONS BY SIZE")
    
    # Get successful MMSIs
    successful_mmsis = set([str(mmsi) for mmsi in corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']['matched_mmsi'].unique()])
    
    # Categorize vessels
    ground_truth_df['MMSI_str'] = ground_truth_df['MMSI'].astype(str)
    ground_truth_df['correlated'] = ground_truth_df['MMSI_str'].isin(successful_mmsis)
    
    logger.info(f"📊 CORRELATION SUCCESS RATE BY VESSEL SIZE:")
    
    for category in ['Large (>100m)', 'Medium (50-100m)', 'Small (<50m)', 'Unknown size']:
        cat_vessels = ground_truth_df[ground_truth_df['size_category'] == category]
        if len(cat_vessels) > 0:
            correlated_count = len(cat_vessels[cat_vessels['correlated']])
            success_rate = correlated_count / len(cat_vessels) * 100
            
            logger.info(f"   {category}:")
            logger.info(f"     Total vessels: {len(cat_vessels)}")
            logger.info(f"     Successfully correlated: {correlated_count}")
            logger.info(f"     Success rate: {success_rate:.1f}%")
            
            # Show average characteristics for this category
            if len(cat_vessels) > 0:
                avg_speed = cat_vessels['SOG'].mean()
                avg_length = cat_vessels['Length'].mean() if 'Length' in cat_vessels.columns else 0
                logger.info(f"     Average speed: {avg_speed:.2f} knots")
                if avg_length > 0:
                    logger.info(f"     Average length: {avg_length:.1f}m")

def main():
    """Main bias investigation."""
    logger.info("🚀 Starting Correlation Bias Investigation")
    
    sar_timestamp = datetime(2023, 6, 20, 23, 6, 42)
    
    # Load data
    ais_df, pred_df, corr_df = load_all_data()
    ground_truth_df = get_ground_truth_with_sizes(ais_df, sar_timestamp)
    
    logger.info(f"📊 Ground truth vessels: {len(ground_truth_df)}")
    
    # Run bias analyses
    analyze_detection_bias(pred_df, ground_truth_df)
    analyze_correlation_scoring_bias(corr_df, ground_truth_df)
    analyze_spatial_gating_bias(corr_df)
    check_correlation_algorithm_bias()
    analyze_missing_correlations_by_size(ground_truth_df, corr_df)
    
    # Final summary
    logger.info(f"\n🎯 BIAS INVESTIGATION SUMMARY:")
    logger.info(f"   This analysis checked for systematic bias in:")
    logger.info(f"   1. Detection system (confidence by size)")
    logger.info(f"   2. Correlation scoring (score components by size)")
    logger.info(f"   3. Spatial gating (distance filtering)")
    logger.info(f"   4. Algorithm configuration (code review)")
    logger.info(f"   5. Success rates by vessel category")
    logger.info(f"\n   Review the detailed results above to identify bias sources.")

if __name__ == "__main__":
    main()
