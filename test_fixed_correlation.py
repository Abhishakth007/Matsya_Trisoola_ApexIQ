#!/usr/bin/env python3
"""
Test Fixed Correlation Engine

Quick test script to validate the correlation fixes without running the entire pipeline.
Uses existing detection results and tests only the correlation component.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ais_correlation'))

from ais_correlation.correlation_engine import CorrelationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

def load_test_data():
    """Load existing detection and AIS data for testing."""
    logger.info("📊 Loading test data...")
    
    # Use existing detection results
    predictions_file = "maritime_analysis_results/maritime_analysis_20250917_162029/predictions_raw.csv"
    ais_file = "ais_data/AIS_175664700472271242_3396-1756647005869.csv"
    safe_folder = "data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    
    if not os.path.exists(predictions_file):
        logger.error(f"❌ Predictions file not found: {predictions_file}")
        return None, None, None
    
    if not os.path.exists(ais_file):
        logger.error(f"❌ AIS file not found: {ais_file}")
        return None, None, None
    
    logger.info(f"✅ Using detection results: {predictions_file}")
    logger.info(f"✅ Using AIS data: {ais_file}")
    logger.info(f"✅ Using SAFE folder: {safe_folder}")
    
    return predictions_file, ais_file, safe_folder

def compare_old_vs_new_results(old_results_file, new_results):
    """Compare old (broken) vs new (fixed) correlation results."""
    logger.info("\n🔍 COMPARING OLD vs NEW CORRELATION RESULTS")
    
    try:
        # Load old results
        old_df = pd.read_csv(old_results_file)
        old_successful = old_df[old_df['matched_mmsi'] != 'UNKNOWN']
        
        logger.info(f"📊 OLD SYSTEM (Broken):")
        logger.info(f"   Total correlations: {len(old_successful)}")
        logger.info(f"   Average distance: {old_successful['distance_meters'].mean():.1f}m")
        logger.info(f"   Distance range: {old_successful['distance_meters'].min():.1f} - {old_successful['distance_meters'].max():.1f}m")
        logger.info(f"   Average confidence: {old_successful['match_confidence'].mean():.3f}")
        logger.info(f"   Confidence range: {old_successful['match_confidence'].min():.3f} - {old_successful['match_confidence'].max():.3f}")
        
        # Count impossible results
        impossible_conf = len(old_successful[old_successful['match_confidence'] > 1.0])
        large_distances = len(old_successful[old_successful['distance_meters'] > 1000])
        
        logger.info(f"   Impossible confidence (>1.0): {impossible_conf}")
        logger.info(f"   Large distances (>1km): {large_distances}")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load old results: {e}")
    
    # Analyze new results
    if len(new_results) > 0:
        new_df = pd.DataFrame([{
            'matched_mmsi': r.matched_mmsi,
            'match_confidence': r.match_confidence,
            'distance_meters': r.distance_meters,
            'position_score': r.position_score,
            'size_score': r.size_score,
            'type_score': r.type_score,
            'is_dark_ship': r.is_dark_ship
        } for r in new_results])
        
        new_successful = new_df[new_df['matched_mmsi'].notna()]
        
        logger.info(f"\n📊 NEW SYSTEM (Fixed):")
        logger.info(f"   Total correlations: {len(new_successful)}")
        
        if len(new_successful) > 0:
            logger.info(f"   Average distance: {new_successful['distance_meters'].mean():.1f}m")
            logger.info(f"   Distance range: {new_successful['distance_meters'].min():.1f} - {new_successful['distance_meters'].max():.1f}m")
            logger.info(f"   Average confidence: {new_successful['match_confidence'].mean():.3f}")
            logger.info(f"   Confidence range: {new_successful['match_confidence'].min():.3f} - {new_successful['match_confidence'].max():.3f}")
            
            # Check for improvements
            impossible_conf_new = len(new_successful[new_successful['match_confidence'] > 1.0])
            large_distances_new = len(new_successful[new_successful['distance_meters'] > 1000])
            
            logger.info(f"   Impossible confidence (>1.0): {impossible_conf_new}")
            logger.info(f"   Large distances (>1km): {large_distances_new}")
            
            # Show sample correlations
            logger.info(f"\n🎯 SAMPLE NEW CORRELATIONS:")
            for i, (_, row) in enumerate(new_successful.head(5).iterrows()):
                logger.info(f"   {i+1}. MMSI {row['matched_mmsi']}: {row['distance_meters']:.1f}m, conf {row['match_confidence']:.3f}")
        else:
            logger.info(f"   No successful correlations (all rejected by new validation)")
    else:
        logger.info(f"\n📊 NEW SYSTEM (Fixed): No results to analyze")

def run_correlation_test():
    """Run the correlation test with fixed engine."""
    logger.info("🚀 Testing Fixed Correlation Engine")
    
    # Load test data
    predictions_file, ais_file, safe_folder = load_test_data()
    if not all([predictions_file, ais_file, safe_folder]):
        logger.error("❌ Cannot run test - missing data files")
        return False
    
    try:
        # Initialize FIXED correlation engine with new parameters
        logger.info("🔧 Initializing FIXED correlation engine...")
        engine = CorrelationEngine(
            spatial_gate_radius=500.0,      # FIXED: Was 10000.0
            confidence_threshold=0.5,       # Unchanged
            max_temporal_gap_hours=2.0,     # Unchanged
            dt_search_step_minutes=5.0,     # Unchanged  
            max_distance_hard_limit=2000.0  # FIXED: Was 25000.0
        )
        
        logger.info(f"✅ Engine configured with FIXED parameters:")
        logger.info(f"   Spatial gate: {engine.spatial_gate_radius}m (was 10000m)")
        logger.info(f"   Hard limit: {engine.max_distance_hard_limit}m (was 25000m)")
        logger.info(f"   Position weight: {engine.weights['position']*100:.1f}% (was 40%)")
        logger.info(f"   Size weight: {engine.weights['size']*100:.1f}% (was 20%)")
        logger.info(f"   Type weight: {engine.weights['type']*100:.1f}% (was 5%)")
        
        # Load data
        logger.info("📊 Loading data into fixed engine...")
        success = engine.load_data(predictions_file, ais_file, safe_folder)
        
        if not success:
            logger.error("❌ Failed to load data into correlation engine")
            return False
        
        # Run correlation
        logger.info("🔗 Running FIXED correlation...")
        results = engine.correlate()
        
        if not results:
            logger.error("❌ Correlation failed to produce results")
            return False
        
        logger.info(f"✅ Correlation completed: {len(results)} results")
        
        # Compare with old results
        old_results_file = "maritime_analysis_results/maritime_analysis_20250917_162029/correlation_results/ais_correlation_results.csv"
        compare_old_vs_new_results(old_results_file, results)
        
        # Detailed analysis of new results
        logger.info(f"\n🎯 DETAILED ANALYSIS OF FIXED RESULTS:")
        
        successful_new = [r for r in results if r.matched_mmsi is not None]
        dark_ships_new = [r for r in results if r.is_dark_ship]
        
        logger.info(f"   Successful correlations: {len(successful_new)}")
        logger.info(f"   Dark ships: {len(dark_ships_new)}")
        logger.info(f"   Total processed: {len(results)}")
        
        if len(successful_new) > 0:
            # Quality metrics
            distances = [r.distance_meters for r in successful_new]
            confidences = [r.match_confidence for r in successful_new]
            
            logger.info(f"\n📊 QUALITY METRICS:")
            logger.info(f"   Max distance: {max(distances):.1f}m (should be ≤500m)")
            logger.info(f"   Max confidence: {max(confidences):.3f} (should be ≤1.0)")
            logger.info(f"   Average distance: {np.mean(distances):.1f}m")
            logger.info(f"   Average confidence: {np.mean(confidences):.3f}")
            
            # Validation checks
            valid_distances = all(d <= 500 for d in distances)
            valid_confidences = all(c <= 1.0 for c in confidences)
            
            logger.info(f"\n✅ VALIDATION RESULTS:")
            logger.info(f"   All distances ≤500m: {'✅ PASS' if valid_distances else '❌ FAIL'}")
            logger.info(f"   All confidences ≤1.0: {'✅ PASS' if valid_confidences else '❌ FAIL'}")
            
            if valid_distances and valid_confidences:
                logger.info(f"🎉 CORRELATION FIXES SUCCESSFUL! All results are now realistic!")
            else:
                logger.warning(f"⚠️ Some issues remain - additional fixes needed")
        else:
            logger.info(f"📊 No successful correlations with fixed parameters")
            logger.info(f"   This might be expected if all previous correlations were fake")
            logger.info(f"   The system is now rejecting impossible matches (good!)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running correlation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🎯 TESTING FIXED CORRELATION SYSTEM")
    logger.info("="*80)
    
    success = run_correlation_test()
    
    logger.info("\n" + "="*80)
    if success:
        logger.info("🎉 CORRELATION TEST COMPLETED")
        logger.info("   Check the detailed logs above to verify fixes worked")
        logger.info("   The system should now produce realistic, accurate correlations")
    else:
        logger.error("❌ CORRELATION TEST FAILED")
        logger.error("   Check error logs above for issues")
    
    logger.info("="*80)

if __name__ == "__main__":
    main()

