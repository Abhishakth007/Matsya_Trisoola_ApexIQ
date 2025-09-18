#!/usr/bin/env python3
"""
Test AIS Correlation

Simple test script to run AIS correlation on the available data.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ais_correlation.correlation_engine import CorrelationEngine

def main():
    """Main test function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting AIS Correlation Test")
    
    # Initialize correlation engine with ADAPTIVE GROWTH parameters
    engine = CorrelationEngine(
        spatial_gate_radius=10000.0,  # 10km base gate (research-based)
        confidence_threshold=0.5,     # Lower threshold for more matches
        max_temporal_gap_hours=2.0,   # ±2h window (total 4h) - matches AIS filtering
        dt_search_step_minutes=5.0,
        max_distance_hard_limit=25000.0  # 25km hard limit
    )
    
    # Define paths
    predictions_csv = "../professor/outputs/predictions.csv"
    ais_csv = "../ais_data/AIS_175664700472271242_3396-1756647005869.csv"
    safe_folder = "../data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    
    # Check if files exist
    if not os.path.exists(predictions_csv):
        logger.error(f"❌ Predictions file not found: {predictions_csv}")
        return False
    
    if not os.path.exists(ais_csv):
        logger.error(f"❌ AIS file not found: {ais_csv}")
        return False
    
    if not os.path.exists(safe_folder):
        logger.error(f"❌ SAFE folder not found: {safe_folder}")
        return False
    
    logger.info("✅ All input files found")
    
    # Load data
    logger.info("📂 Loading data...")
    success = engine.load_data(predictions_csv, ais_csv, safe_folder)
    
    if not success:
        logger.error("❌ Failed to load data")
        return False
    
    logger.info("✅ Data loaded successfully")
    
    # Perform correlation
    logger.info("🔍 Performing AIS correlation...")
    try:
        results = engine.correlate()
        logger.info("✅ Correlation completed successfully")
    except Exception as e:
        logger.error(f"❌ Correlation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Save results
    output_path = "ais_correlation_results.csv"
    logger.info(f"💾 Saving results to: {output_path}")
    try:
        engine.save_results(results, output_path)
        logger.info("✅ Results saved successfully")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        return False
    
    # Print summary
    print("\n" + "="*60)
    print("📊 AIS CORRELATION RESULTS SUMMARY")
    print("="*60)
    
    total_detections = len(results)
    matched_detections = sum(1 for r in results if not r.is_dark_ship)
    dark_ships = sum(1 for r in results if r.is_dark_ship)
    ambiguous_matches = sum(1 for r in results if r.is_ambiguous)
    
    print(f"Total detections: {total_detections}")
    print(f"Matched to AIS: {matched_detections}")
    print(f"Dark ships: {dark_ships}")
    print(f"Ambiguous matches: {ambiguous_matches}")
    print(f"Match rate: {matched_detections/total_detections*100:.1f}%")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    print("-" * 60)
    for result in results:
        status = "🌍 MATCHED" if not result.is_dark_ship else "🚫 DARK SHIP"
        if result.is_ambiguous:
            status += " (AMBIGUOUS)"
        
        print(f"Detection {result.detect_id}: {status}")
        if not result.is_dark_ship:
            print(f"  MMSI: {result.matched_mmsi}")
            print(f"  Match confidence: {result.match_confidence:.3f}")
            print(f"  Time gap: {result.time_gap_minutes:.1f} minutes")
            print(f"  Distance: {result.distance_meters:.0f} meters")
            print(f"  Scores: pos={result.position_score:.3f}, "
                  f"heading={result.heading_score:.3f}, "
                  f"size={result.size_score:.3f}, "
                  f"speed={result.speed_score:.3f}, "
                  f"type={result.type_score:.3f}")
        print()
    
    print("="*60)
    print("✅ AIS Correlation Test Completed Successfully!")
    print(f"📄 Results saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
