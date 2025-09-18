#!/usr/bin/env python3
"""
Integrated Maritime Pipeline Runner

Simple script to execute the complete integrated pipeline with
optimized settings for your specific use case.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from integrated_maritime_pipeline import IntegratedMaritimePipeline

def main():
    """Run the integrated maritime pipeline with optimized settings."""
    
    print("🌊 INTEGRATED MARITIME PIPELINE")
    print("=" * 50)
    print("🎯 Optimized for open sea vessel detection and AIS correlation")
    print()
    
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/integrated_pipeline_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('PipelineRunner')
    logger.info(f"📝 Logging to: {log_file}")
    
    # Define input paths
    input_paths = {
        'sar_image': "snap_output/step4_final_output.tif",
        'ais_data': "ais_data/AIS_175664700472271242_3396-1756647005869.csv", 
        'safe_folder': "data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE",
        'output_base': "maritime_analysis_results"
    }
    
    # Define test region (your specified area)
    test_region = (20200, 12600, 21400, 13900)
    
    # Validate input files (except output directory which will be created)
    logger.info("🔍 Validating input files...")
    for name, path in input_paths.items():
        if name == 'output_base':
            # Create output base directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ {name}: {path} (created)")
        elif not os.path.exists(path):
            logger.error(f"❌ {name} not found: {path}")
            return False
        else:
            logger.info(f"✅ {name}: {path}")
    
    print(f"📍 Test region: ({test_region[0]}, {test_region[1]}) to ({test_region[2]}, {test_region[3]})")
    print(f"📏 Test area size: {test_region[2]-test_region[0]} × {test_region[3]-test_region[1]} pixels")
    print()
    
    try:
        # Initialize pipeline
        logger.info("🚀 Initializing integrated pipeline...")
        pipeline = IntegratedMaritimePipeline(config_path="src/config/config.yml")
        
        # Run complete analysis
        logger.info("🔄 Starting complete maritime analysis...")
        results = pipeline.run_complete_analysis(
            sar_image_path=input_paths['sar_image'],
            ais_data_path=input_paths['ais_data'],
            safe_folder_path=input_paths['safe_folder'],
            output_base_dir=input_paths['output_base'],
            test_region_bounds=test_region
        )
        
        print("\n🎉 PIPELINE EXECUTION SUCCESSFUL!")
        print(f"📁 Complete results package: {results['output_directory']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Run completed successfully!")
    else:
        print("\n❌ Run failed - check logs for details")
        sys.exit(1)
