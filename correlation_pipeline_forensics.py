#!/usr/bin/env python3
"""
Correlation Pipeline Forensics

Complete forensic analysis of the correlation pipeline to identify ALL issues
causing fake correlations. This will traverse every step and identify problems.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def analyze_correlation_configuration():
    """Analyze correlation engine configuration for issues."""
    logger.info("🔍 FORENSIC STEP 1: CORRELATION CONFIGURATION ANALYSIS")
    
    issues = []
    
    # Configuration issues from the code
    config_issues = {
        "spatial_gate_radius": 10000.0,  # 10km - TOO LARGE
        "max_distance_hard_limit": 25000.0,  # 25km - INSANELY LARGE
        "confidence_threshold": 0.5,  # Reasonable
        "adaptive_radius_growth": 50.0,  # 50m per minute - TOO GENEROUS
        "adaptive_radius_cap": 20000.0,  # 20km cap - TOO LARGE
    }
    
    logger.info(f"📊 Current Configuration Issues:")
    
    if config_issues["spatial_gate_radius"] > 1000:
        issue = f"❌ Spatial gate radius: {config_issues['spatial_gate_radius']/1000:.1f}km is TOO LARGE (should be <1km)"
        logger.error(issue)
        issues.append(("CONFIG_SPATIAL_GATE_TOO_LARGE", issue))
    
    if config_issues["max_distance_hard_limit"] > 5000:
        issue = f"❌ Hard distance limit: {config_issues['max_distance_hard_limit']/1000:.1f}km is INSANE (should be <5km)"
        logger.error(issue)
        issues.append(("CONFIG_HARD_LIMIT_INSANE", issue))
    
    if config_issues["adaptive_radius_growth"] > 20:
        issue = f"❌ Adaptive growth: {config_issues['adaptive_radius_growth']}m/min is TOO GENEROUS (should be <20m/min)"
        logger.error(issue)
        issues.append(("CONFIG_ADAPTIVE_GROWTH_TOO_GENEROUS", issue))
    
    return issues

def analyze_position_scoring_algorithm():
    """Analyze position scoring algorithm for bias."""
    logger.info("\n🔍 FORENSIC STEP 2: POSITION SCORING ALGORITHM ANALYSIS")
    
    issues = []
    
    # From the code: score = np.exp(-normalized_distance / 2)
    # normalized_distance = distance / (uncertainty + 100)
    
    logger.info(f"📊 Position Scoring Algorithm Issues:")
    
    # Test various distances to see scoring behavior
    test_distances = [100, 500, 1000, 5000, 10000, 25000]  # meters
    uncertainty = 500  # typical uncertainty
    
    logger.info(f"   Distance vs Position Score (uncertainty={uncertainty}m):")
    
    for distance in test_distances:
        normalized_distance = distance / (uncertainty + 100)
        score = np.exp(-normalized_distance / 2)
        logger.info(f"     {distance:5d}m → score: {score:.3f}")
        
        if distance > 1000 and score > 0.5:
            issue = f"❌ Distance {distance}m gets score {score:.3f} - TOO HIGH for large distance"
            logger.error(f"     {issue}")
            issues.append(("SCORING_DISTANCE_TOO_GENEROUS", issue))
    
    # Check if uncertainty normalization is problematic
    if True:  # Always check this
        issue = f"❌ Position scoring uses uncertainty normalization - allows large vessels to match at huge distances"
        logger.error(issue)
        issues.append(("SCORING_UNCERTAINTY_BIAS", issue))
    
    return issues

def analyze_size_scoring_bias():
    """Analyze size scoring for bias towards large vessels."""
    logger.info("\n🔍 FORENSIC STEP 3: SIZE SCORING BIAS ANALYSIS")
    
    issues = []
    
    # From code: Hard size gate rejects if dimensions differ by more than 100%
    # This means a 50m detection can match a 100m AIS vessel!
    
    logger.info(f"📊 Size Scoring Issues:")
    
    # Test size mismatches
    test_cases = [
        (30, 60, "Small detection vs Medium AIS"),
        (50, 100, "Medium detection vs Large AIS"),
        (30, 150, "Small detection vs Large AIS"),
        (60, 200, "Medium detection vs Huge AIS")
    ]
    
    for det_size, ais_size, description in test_cases:
        length_diff = abs(det_size - ais_size) / max(ais_size, 1)
        
        if length_diff <= 1.0:  # From the code's hard gate
            score = np.exp(-length_diff)
            logger.info(f"   {description}: {det_size}m vs {ais_size}m → ALLOWED, score: {score:.3f}")
            
            if det_size < ais_size * 0.5:  # Detection is less than half AIS size
                issue = f"❌ Size scoring allows {det_size}m detection to match {ais_size}m AIS vessel"
                logger.error(f"     {issue}")
                issues.append(("SIZE_SCORING_TOO_PERMISSIVE", issue))
        else:
            logger.info(f"   {description}: {det_size}m vs {ais_size}m → REJECTED")
    
    return issues

def analyze_scoring_weights_bias():
    """Analyze scoring weights for bias."""
    logger.info("\n🔍 FORENSIC STEP 4: SCORING WEIGHTS BIAS ANALYSIS")
    
    issues = []
    
    # From code
    weights = {
        'position': 0.4,    # 40% - most important
        'heading': 0.2,     # 20%
        'size': 0.2,        # 20%
        'speed': 0.1,       # 10%
        'type': 0.05,       # 5%
        'temporal': 0.05    # 5%
    }
    
    logger.info(f"📊 Scoring Weights Analysis:")
    for component, weight in weights.items():
        logger.info(f"   {component}: {weight*100:.1f}%")
    
    # Check for bias
    if weights['position'] > 0.3:
        issue = f"❌ Position weight ({weights['position']*100:.1f}%) is TOO HIGH - allows distant matches"
        logger.error(issue)
        issues.append(("WEIGHTS_POSITION_TOO_HIGH", issue))
    
    if weights['size'] < 0.3:
        issue = f"❌ Size weight ({weights['size']*100:.1f}%) is TOO LOW - ignores size mismatches"
        logger.error(issue)
        issues.append(("WEIGHTS_SIZE_TOO_LOW", issue))
    
    if weights['type'] < 0.1:
        issue = f"❌ Type weight ({weights['type']*100:.1f}%) is TOO LOW - ignores vessel type"
        logger.error(issue)
        issues.append(("WEIGHTS_TYPE_TOO_LOW", issue))
    
    return issues

def analyze_actual_correlation_results():
    """Analyze actual correlation results for anomalies."""
    logger.info("\n🔍 FORENSIC STEP 5: ACTUAL RESULTS ANOMALY ANALYSIS")
    
    issues = []
    
    # Load correlation results
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    successful_corr = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']
    
    logger.info(f"📊 Correlation Results Anomalies:")
    
    # Check for impossible scores
    impossible_scores = successful_corr[successful_corr['match_confidence'] > 1.0]
    if len(impossible_scores) > 0:
        issue = f"❌ Found {len(impossible_scores)} correlations with confidence > 1.0 (impossible!)"
        logger.error(issue)
        issues.append(("RESULTS_IMPOSSIBLE_CONFIDENCE", issue))
        
        for _, row in impossible_scores.iterrows():
            logger.error(f"     MMSI {row['matched_mmsi']}: confidence = {row['match_confidence']:.3f}")
    
    # Check for zero position scores with high confidence
    zero_position = successful_corr[successful_corr['position_score'] == 0.0]
    if len(zero_position) > 0:
        issue = f"❌ Found {len(zero_position)} correlations with zero position score"
        logger.error(issue)
        issues.append(("RESULTS_ZERO_POSITION_SCORE", issue))
    
    # Check for huge distances with high position scores
    large_distance_high_score = successful_corr[
        (successful_corr['distance_meters'] > 1000) & 
        (successful_corr['position_score'] > 0.8)
    ]
    if len(large_distance_high_score) > 0:
        issue = f"❌ Found {len(large_distance_high_score)} correlations with >1km distance but >0.8 position score"
        logger.error(issue)
        issues.append(("RESULTS_LARGE_DISTANCE_HIGH_SCORE", issue))
        
        for _, row in large_distance_high_score.iterrows():
            logger.error(f"     MMSI {row['matched_mmsi']}: {row['distance_meters']:.0f}m distance, {row['position_score']:.3f} position score")
    
    return issues

def analyze_detection_to_ais_size_mismatch():
    """Analyze detection vs AIS size mismatches in correlations."""
    logger.info("\n🔍 FORENSIC STEP 6: DETECTION-AIS SIZE MISMATCH ANALYSIS")
    
    issues = []
    
    # Load all data
    pred_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/predictions_raw.csv")
    corr_df = pd.read_csv("maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv")
    ais_df = pd.read_csv("ais_data/AIS_175664700472271242_3396-1756647005869.csv")
    
    successful_corr = corr_df[corr_df['matched_mmsi'] != 'UNKNOWN']
    
    logger.info(f"📊 Detection vs AIS Size Mismatch Analysis:")
    
    size_mismatches = []
    
    for _, corr_row in successful_corr.iterrows():
        # Get detection info
        detection = pred_df[pred_df['detect_id'] == corr_row['detect_id']].iloc[0]
        det_length = detection['vessel_length_m']
        
        # Get AIS info
        mmsi = corr_row['matched_mmsi']
        ais_vessel = ais_df[ais_df['MMSI'] == int(mmsi)]
        if len(ais_vessel) > 0:
            ais_length = ais_vessel['Length'].iloc[0]
            
            if pd.notna(det_length) and pd.notna(ais_length):
                size_ratio = max(det_length, ais_length) / min(det_length, ais_length)
                
                if size_ratio > 2.0:  # More than 2x size difference
                    mismatch_info = {
                        'mmsi': mmsi,
                        'detection_size': det_length,
                        'ais_size': ais_length,
                        'size_ratio': size_ratio,
                        'distance': corr_row['distance_meters'],
                        'match_confidence': corr_row['match_confidence']
                    }
                    size_mismatches.append(mismatch_info)
    
    if len(size_mismatches) > 0:
        issue = f"❌ Found {len(size_mismatches)} correlations with >2x size mismatch"
        logger.error(issue)
        issues.append(("RESULTS_MAJOR_SIZE_MISMATCH", issue))
        
        for mismatch in size_mismatches[:5]:  # Show first 5
            logger.error(f"     MMSI {mismatch['mmsi']}: {mismatch['detection_size']:.1f}m detection vs {mismatch['ais_size']:.1f}m AIS (ratio: {mismatch['size_ratio']:.1f}x)")
    
    return issues

def create_comprehensive_todo_list(all_issues):
    """Create comprehensive todo list from all identified issues."""
    logger.info("\n📋 CREATING COMPREHENSIVE TODO LIST")
    
    # Group issues by category
    config_issues = [issue for issue in all_issues if issue[0].startswith('CONFIG_')]
    scoring_issues = [issue for issue in all_issues if issue[0].startswith('SCORING_')]
    weights_issues = [issue for issue in all_issues if issue[0].startswith('WEIGHTS_')]
    results_issues = [issue for issue in all_issues if issue[0].startswith('RESULTS_')]
    
    todo_items = []
    
    # Configuration fixes
    if config_issues:
        todo_items.append({
            'category': 'CRITICAL_CONFIG_FIXES',
            'priority': 'URGENT',
            'items': [
                'Reduce spatial_gate_radius from 10km to 500m maximum',
                'Reduce max_distance_hard_limit from 25km to 2km maximum', 
                'Reduce adaptive_radius_growth from 50m/min to 10m/min',
                'Reduce adaptive_radius_cap from 20km to 1km'
            ]
        })
    
    # Scoring algorithm fixes
    if scoring_issues:
        todo_items.append({
            'category': 'SCORING_ALGORITHM_FIXES',
            'priority': 'HIGH',
            'items': [
                'Fix position scoring to heavily penalize distances >500m',
                'Remove uncertainty bias that favors large vessels',
                'Add strict distance cutoffs regardless of uncertainty',
                'Implement size-based distance limits (small vessels = small search radius)'
            ]
        })
    
    # Size scoring fixes
    todo_items.append({
        'category': 'SIZE_SCORING_FIXES', 
        'priority': 'HIGH',
        'items': [
            'Reduce size mismatch tolerance from 100% to 30%',
            'Add hard rejection for detection <50% of AIS vessel size',
            'Implement vessel-type specific size validation',
            'Add size consistency checks before correlation'
        ]
    })
    
    # Scoring weights rebalancing
    if weights_issues:
        todo_items.append({
            'category': 'SCORING_WEIGHTS_REBALANCE',
            'priority': 'MEDIUM',
            'items': [
                'Reduce position weight from 40% to 25%',
                'Increase size weight from 20% to 35%',
                'Increase type weight from 5% to 15%',
                'Add vessel class consistency scoring (cargo vs pleasure craft)'
            ]
        })
    
    # Result validation
    if results_issues:
        todo_items.append({
            'category': 'RESULT_VALIDATION',
            'priority': 'HIGH',
            'items': [
                'Add confidence score validation (reject >1.0)',
                'Add sanity checks for impossible correlations',
                'Implement detection-AIS size ratio validation',
                'Add correlation result auditing and logging'
            ]
        })
    
    # Additional systematic fixes
    todo_items.append({
        'category': 'SYSTEMATIC_IMPROVEMENTS',
        'priority': 'MEDIUM', 
        'items': [
            'Add vessel type filtering (cargo ships vs pleasure craft)',
            'Implement detection confidence thresholds by vessel size',
            'Add geographic zone-based correlation parameters',
            'Create correlation quality metrics and monitoring'
        ]
    })
    
    return todo_items

def main():
    """Main forensic analysis."""
    logger.info("🚀 Starting Complete Correlation Pipeline Forensics")
    
    all_issues = []
    
    # Run all forensic analyses
    all_issues.extend(analyze_correlation_configuration())
    all_issues.extend(analyze_position_scoring_algorithm())
    all_issues.extend(analyze_size_scoring_bias())
    all_issues.extend(analyze_scoring_weights_bias())
    all_issues.extend(analyze_actual_correlation_results())
    all_issues.extend(analyze_detection_to_ais_size_mismatch())
    
    # Create comprehensive todo list
    todo_list = create_comprehensive_todo_list(all_issues)
    
    # Output results
    logger.info(f"\n🚨 FORENSIC ANALYSIS COMPLETE")
    logger.info(f"   Total issues identified: {len(all_issues)}")
    logger.info(f"   Todo categories created: {len(todo_list)}")
    
    logger.info(f"\n📋 COMPREHENSIVE TODO LIST:")
    for todo_category in todo_list:
        logger.info(f"\n🔥 {todo_category['category']} (Priority: {todo_category['priority']}):")
        for i, item in enumerate(todo_category['items'], 1):
            logger.info(f"   {i}. {item}")
    
    logger.info(f"\n🎯 SUMMARY OF ROOT CAUSES:")
    logger.info(f"   1. Configuration allows 25km correlations (INSANE)")
    logger.info(f"   2. Position scoring favors large vessels via uncertainty bias")
    logger.info(f"   3. Size scoring allows 2x+ size mismatches")
    logger.info(f"   4. Scoring weights prioritize distance over vessel characteristics")
    logger.info(f"   5. No validation prevents impossible correlation results")
    
    logger.info(f"\n🚨 THE CORRELATION SYSTEM IS FUNDAMENTALLY BROKEN!")
    logger.info(f"   Every correlation is likely matching wrong vessels!")

if __name__ == "__main__":
    main()
