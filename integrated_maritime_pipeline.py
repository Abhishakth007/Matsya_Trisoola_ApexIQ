#!/usr/bin/env python3
"""
Integrated Maritime Pipeline

Complete end-to-end vessel detection and AIS correlation system.
Integrates existing detection and correlation systems with enhanced outputs.

Pipeline Flow:
1. Vessel Detection → 2. Attribute Filtering → 3. Multi-format Output → 
4. AIS Correlation → 5. Correlation Visualization → 6. Final Results Package
"""

import os
import sys
import yaml
import json
import logging
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'professor'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ais_correlation'))

# Import pipeline components
from detection_filter import DetectionAttributeFilter, filter_detections_by_attributes
from geojson_exporter import export_predictions_to_geojson, create_correlation_geojson
from correlation_visualizer import CorrelationVisualizer, create_correlation_visualizations
from simple_bbox_visualizer import create_region_visualization as create_bbox_visualization

# Import existing systems
from professor.robust_vessel_detection_pipeline import RobustVesselDetectionPipeline, ProcessingConfig
from ais_correlation.correlation_engine import CorrelationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('IntegratedMaritimePipeline')

class IntegratedMaritimePipeline:
    """
    Master controller for the integrated maritime awareness pipeline.
    
    Orchestrates vessel detection, attribute filtering, AIS correlation,
    and comprehensive output generation.
    """
    
    def __init__(self, config_path: str = "src/config/config.yml"):
        """
        Initialize the integrated pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        
        logger.info("🚀 Integrated Maritime Pipeline initialized")
        logger.info(f"📋 Configuration loaded from: {config_path}")
    
    def _load_config(self) -> Dict:
        """Load pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('main', {})
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def run_complete_analysis(self,
                            sar_image_path: str,
                            ais_data_path: str,
                            safe_folder_path: str,
                            output_base_dir: str,
                            test_region_bounds: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Run complete maritime analysis pipeline.
        
        Args:
            sar_image_path: Path to SAR GeoTIFF image
            ais_data_path: Path to AIS CSV data
            safe_folder_path: Path to SAFE folder
            output_base_dir: Base output directory
            test_region_bounds: Optional test region bounds
            
        Returns:
            Dictionary with complete results and file paths
        """
        start_time = datetime.now()
        logger.info("🌊 STARTING INTEGRATED MARITIME ANALYSIS")
        logger.info("=" * 80)
        
        try:
            # Create timestamped output directory
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_base_dir, f"maritime_analysis_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            self.results['output_directory'] = output_dir
            self.results['start_time'] = start_time.isoformat()
            
            # PHASE 1: VESSEL DETECTION
            logger.info("🎯 PHASE 1: VESSEL DETECTION")
            detection_results = self._run_vessel_detection(sar_image_path, safe_folder_path, output_dir)
            
            # PHASE 2: ATTRIBUTE FILTERING  
            logger.info("🔍 PHASE 2: ATTRIBUTE FILTERING")
            filtering_results = self._run_attribute_filtering(detection_results, output_dir)
            
            # PHASE 3: MULTI-FORMAT OUTPUT GENERATION
            logger.info("📊 PHASE 3: OUTPUT GENERATION")
            output_results = self._generate_detection_outputs(filtering_results, output_dir, test_region_bounds)
            
            # PHASE 4: AIS CORRELATION
            logger.info("🛰️ PHASE 4: AIS CORRELATION")
            correlation_results = self._run_ais_correlation(
                filtering_results['filtered_predictions_path'], 
                ais_data_path, 
                safe_folder_path, 
                output_dir
            )
            
            # PHASE 5: CORRELATION VISUALIZATION
            logger.info("🎨 PHASE 5: CORRELATION VISUALIZATION")
            visualization_results = self._create_correlation_visualizations(
                sar_image_path, filtering_results, correlation_results, output_dir, test_region_bounds
            )
            
            # PHASE 6: FINAL RESULTS PACKAGE
            logger.info("📦 PHASE 6: FINAL RESULTS PACKAGE")
            final_results = self._create_final_package(
                output_dir, detection_results, filtering_results, 
                correlation_results, visualization_results
            )
            
            # Update results
            self.results.update({
                'detection': detection_results,
                'filtering': filtering_results,
                'outputs': output_results,
                'correlation': correlation_results,
                'visualization': visualization_results,
                'final': final_results,
                'end_time': datetime.now().isoformat(),
                'total_time_seconds': (datetime.now() - start_time).total_seconds()
            })
            
            logger.info("✅ INTEGRATED MARITIME ANALYSIS COMPLETE")
            self._print_final_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def _run_vessel_detection(self, sar_image_path: str, safe_folder_path: str, output_dir: str) -> Dict:
        """Run vessel detection phase."""
        logger.info("🔄 Running vessel detection pipeline...")
        
        # Create detection output directory
        detection_dir = os.path.join(output_dir, "detection_results")
        os.makedirs(detection_dir, exist_ok=True)
        
        # Setup detection pipeline
        detector_model = self.config["sentinel1_detector"].replace("${PROJECT_ROOT:-.}", PROJECT_ROOT)
        postprocess_model = self.config["sentinel1_postprocessor"].replace("${PROJECT_ROOT:-.}", PROJECT_ROOT)
        
        # Create processing config
        proc_config = ProcessingConfig(
            window_size=self.config.get("default_window_size", 200),
            overlap=self.config.get("default_overlap", 100),
            confidence_threshold=self.config.get("thresholds", {}).get("conf_threshold", 0.15),
            nms_threshold=self.config.get("thresholds", {}).get("nms_threshold", 0.5),
            preserve_crs=True
        )
        
        # Initialize pipeline
        pipeline = RobustVesselDetectionPipeline(
            detector_model_path=detector_model,
            postprocess_model_path=postprocess_model,
            device='auto',
            config=proc_config,
            safe_file_path=safe_folder_path
        )
        
        # Process scene
        detection_results = pipeline.process_scene(
            input_geotiff=sar_image_path,
            output_dir=detection_dir,
            water_mask_geotiff=sar_image_path
        )
        
        # Copy predictions to main results
        predictions_path = os.path.join(detection_dir, "predictions.csv")
        main_predictions_path = os.path.join(output_dir, "predictions_raw.csv")
        shutil.copy2(predictions_path, main_predictions_path)
        
        results = {
            'predictions_path': main_predictions_path,
            'detection_directory': detection_dir,
            'total_detections': detection_results['metadata']['total_detections'],
            'processing_stats': detection_results['metadata']['processing_stats']
        }
        
        logger.info(f"✅ Detection complete: {results['total_detections']} vessels detected")
        return results
    
    def _run_attribute_filtering(self, detection_results: Dict, output_dir: str) -> Dict:
        """Run attribute filtering phase."""
        logger.info("🔧 Running attribute filtering...")
        
        # Setup filter parameters
        min_attributes = 3  # Require at least 3 vessel attributes
        min_confidence = self.config.get("thresholds", {}).get("conf_threshold", 0.15)
        
        # Filter predictions
        filtered_path = os.path.join(output_dir, "predictions_filtered.csv")
        filter_stats = filter_detections_by_attributes(
            detection_results['predictions_path'],
            filtered_path,
            min_attributes=min_attributes,
            min_confidence=min_confidence
        )
        
        results = {
            'filtered_predictions_path': filtered_path,
            'filter_statistics': filter_stats,
            'min_attributes_required': min_attributes,
            'quality_detections': filter_stats['final_count']
        }
        
        logger.info(f"✅ Filtering complete: {results['quality_detections']} quality detections")
        return results
    
    def _generate_detection_outputs(self, filtering_results: Dict, output_dir: str, 
                                  test_region_bounds: Optional[Tuple[int, int, int, int]]) -> Dict:
        """Generate multi-format detection outputs."""
        logger.info("📊 Generating detection outputs...")
        
        # 1. Export to GeoJSON
        geojson_path = os.path.join(output_dir, "predictions.geojson")
        export_predictions_to_geojson(
            filtering_results['filtered_predictions_path'],
            geojson_path
        )
        
        # 2. Create bounding box visualization
        bbox_viz_path = os.path.join(output_dir, "detections_with_bboxes.png")
        create_bbox_visualization(
            "snap_output/step4_final_output.tif",
            filtering_results['filtered_predictions_path'],
            test_region_bounds if test_region_bounds else (0, 0, 25970, 16732),
            bbox_viz_path
        )
        
        # 3. Copy vessel crops from detection results
        crops_source = os.path.join(os.path.dirname(filtering_results['filtered_predictions_path']), "..", "detection_results", "crops")
        crops_dest = os.path.join(output_dir, "vessel_crops")
        
        if os.path.exists(crops_source):
            shutil.copytree(crops_source, crops_dest, dirs_exist_ok=True)
            logger.info(f"📸 Vessel crops copied to: {crops_dest}")
        
        results = {
            'geojson_path': geojson_path,
            'bbox_visualization_path': bbox_viz_path,
            'vessel_crops_directory': crops_dest if os.path.exists(crops_source) else None
        }
        
        logger.info("✅ Detection outputs generated")
        return results
    
    def _run_ais_correlation(self, predictions_path: str, ais_data_path: str, 
                           safe_folder_path: str, output_dir: str) -> Dict:
        """Run AIS correlation phase."""
        logger.info("🔗 Running AIS correlation...")
        
        # Create correlation output directory
        correlation_dir = os.path.join(output_dir, "correlation_results")
        os.makedirs(correlation_dir, exist_ok=True)
        
        # Initialize correlation engine
        engine = CorrelationEngine(
            spatial_gate_radius=10000.0,  # 10km
            confidence_threshold=0.5,
            max_temporal_gap_hours=2.0
        )
        
        # Load data
        success = engine.load_data(predictions_path, ais_data_path, safe_folder_path)
        
        if not success:
            raise RuntimeError("Failed to load correlation data")
        
        # Run correlation
        correlation_results = engine.correlate()
        
        # Save results
        correlation_csv_path = os.path.join(correlation_dir, "ais_correlation_results.csv")
        engine.save_results(correlation_results, correlation_csv_path)
        
        # Calculate statistics
        total_detections = len(correlation_results)
        correlated_count = sum(1 for r in correlation_results if not r.is_dark_ship)
        dark_ship_count = sum(1 for r in correlation_results if r.is_dark_ship)
        match_rate = (correlated_count / total_detections * 100) if total_detections > 0 else 0
        
        results = {
            'correlation_results_path': correlation_csv_path,
            'correlation_directory': correlation_dir,
            'total_detections': total_detections,
            'correlated_vessels': correlated_count,
            'dark_ships': dark_ship_count,
            'match_rate_percent': match_rate
        }
        
        logger.info(f"✅ Correlation complete: {correlated_count}/{total_detections} vessels correlated ({match_rate:.1f}%)")
        return results
    
    def _create_correlation_visualizations(self, sar_image_path: str, filtering_results: Dict,
                                         correlation_results: Dict, output_dir: str,
                                         test_region_bounds: Optional[Tuple[int, int, int, int]]) -> Dict:
        """Create correlation status visualizations."""
        logger.info("🎨 Creating correlation visualizations...")
        
        # Create visualization output directory
        viz_dir = os.path.join(output_dir, "correlation_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create correlation visualizations
        viz_files = create_correlation_visualizations(
            sar_image_path=sar_image_path,
            predictions_csv_path=filtering_results['filtered_predictions_path'],
            correlation_results_csv_path=correlation_results['correlation_results_path'],
            output_dir=viz_dir,
            region_bounds=test_region_bounds
        )
        
        # Create correlation-enhanced GeoJSON
        correlation_geojson_path = os.path.join(output_dir, "predictions_with_correlation.geojson")
        
        # Load data for correlation GeoJSON
        predictions_df = pd.read_csv(filtering_results['filtered_predictions_path'])
        correlation_df = pd.read_csv(correlation_results['correlation_results_path'])
        
        create_correlation_geojson(predictions_df, correlation_df, correlation_geojson_path)
        
        results = {
            'visualization_directory': viz_dir,
            'correlation_geojson_path': correlation_geojson_path,
            'visualization_files': viz_files
        }
        
        logger.info("✅ Correlation visualizations created")
        return results
    
    def _create_final_package(self, output_dir: str, detection_results: Dict,
                            filtering_results: Dict, correlation_results: Dict,
                            visualization_results: Dict) -> Dict:
        """Create final results package with summary."""
        logger.info("📦 Creating final results package...")
        
        # Create summary JSON
        summary = {
            "pipeline_info": {
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "configuration": self.config
            },
            "detection_summary": {
                "total_raw_detections": detection_results['total_detections'],
                "quality_detections": filtering_results['quality_detections'],
                "filter_statistics": filtering_results['filter_statistics']
            },
            "correlation_summary": {
                "total_detections": correlation_results['total_detections'],
                "correlated_vessels": correlation_results['correlated_vessels'],
                "dark_ships": correlation_results['dark_ships'],
                "match_rate_percent": correlation_results['match_rate_percent']
            },
            "output_files": {
                "raw_predictions": "predictions_raw.csv",
                "filtered_predictions": "predictions_filtered.csv",
                "predictions_geojson": "predictions.geojson",
                "correlation_results": "correlation_results/ais_correlation_results.csv",
                "correlation_geojson": "predictions_with_correlation.geojson",
                "detection_visualization": "detections_with_bboxes.png",
                "correlation_visualizations": "correlation_visualizations/"
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create README
        readme_path = os.path.join(output_dir, "README.md")
        self._create_results_readme(readme_path, summary)
        
        results = {
            'summary_path': summary_path,
            'readme_path': readme_path,
            'complete_package_directory': output_dir
        }
        
        logger.info("✅ Final package created")
        return results
    
    def _create_results_readme(self, readme_path: str, summary: Dict):
        """Create README file for results package."""
        
        readme_content = f"""# Maritime Analysis Results

Generated: {summary['pipeline_info']['timestamp']}

## 📊 Summary Statistics

### Detection Results
- **Raw Detections**: {summary['detection_summary']['total_raw_detections']}
- **Quality Detections**: {summary['detection_summary']['quality_detections']}
- **Filter Removal Rate**: {summary['detection_summary']['filter_statistics']['removal_rate']:.1f}%

### AIS Correlation Results  
- **Total Analyzed**: {summary['correlation_summary']['total_detections']}
- **AIS Correlated**: {summary['correlation_summary']['correlated_vessels']}
- **Dark Ships**: {summary['correlation_summary']['dark_ships']}
- **Match Rate**: {summary['correlation_summary']['match_rate_percent']:.1f}%

## 📁 Output Files

### Core Results
- `predictions_raw.csv` - All raw detections from vessel detection
- `predictions_filtered.csv` - Quality-filtered detections  
- `predictions.geojson` - GeoJSON format for GIS tools
- `predictions_with_correlation.geojson` - GeoJSON with AIS correlation status

### AIS Correlation
- `correlation_results/ais_correlation_results.csv` - Detailed correlation results
- `correlation_visualizations/` - Visual correlation status images

### Visualizations
- `detections_with_bboxes.png` - SAR image with detection bounding boxes
- `correlation_visualizations/correlation_combined.png` - Combined correlation status
- `correlation_visualizations/correlated_vessels.png` - AIS-matched vessels only
- `correlation_visualizations/dark_vessels.png` - Unmatched vessels only

### Supporting Data
- `vessel_crops/` - Individual vessel detection crops
- `pipeline_summary.json` - Complete processing statistics
- `README.md` - This file

## 🎯 Usage

### For GIS Analysis
Use the GeoJSON files with QGIS, ArcGIS, or other GIS software:
- `predictions.geojson` - Basic vessel detections
- `predictions_with_correlation.geojson` - Enhanced with AIS correlation

### For Visual Analysis
Review the PNG visualizations to understand:
- Detection coverage and quality
- AIS correlation success rate
- Spatial distribution patterns

### For Further Processing
Use the CSV files for additional analysis:
- `predictions_filtered.csv` - High-quality detections
- `correlation_results/ais_correlation_results.csv` - Correlation details

## 🔧 Configuration Used

Window Size: {summary['pipeline_info']['configuration'].get('default_window_size', 'N/A')}
Overlap: {summary['pipeline_info']['configuration'].get('default_overlap', 'N/A')}
Confidence Threshold: {summary['pipeline_info']['configuration'].get('thresholds', {}).get('conf_threshold', 'N/A')}
NMS Threshold: {summary['pipeline_info']['configuration'].get('thresholds', {}).get('nms_threshold', 'N/A')}
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _print_final_summary(self):
        """Print final pipeline summary."""
        print("\n" + "="*80)
        print("🎉 INTEGRATED MARITIME PIPELINE COMPLETE")
        print("="*80)
        print(f"📁 Results directory: {self.results['output_directory']}")
        print(f"⏱️ Total processing time: {self.results['total_time_seconds']:.1f} seconds")
        print()
        print("📊 DETECTION SUMMARY:")
        print(f"   • Raw detections: {self.results['detection']['total_detections']}")
        print(f"   • Quality detections: {self.results['filtering']['quality_detections']}")
        print(f"   • Filter removal rate: {self.results['filtering']['filter_statistics']['removal_rate']:.1f}%")
        print()
        print("🛰️ CORRELATION SUMMARY:")
        print(f"   • AIS correlated: {self.results['correlation']['correlated_vessels']}")
        print(f"   • Dark ships: {self.results['correlation']['dark_ships']}")
        print(f"   • Match rate: {self.results['correlation']['match_rate_percent']:.1f}%")
        print()
        print("📦 OUTPUT FILES:")
        print(f"   • Filtered predictions: predictions_filtered.csv")
        print(f"   • GeoJSON format: predictions.geojson")
        print(f"   • Correlation results: correlation_results/ais_correlation_results.csv")
        print(f"   • Bbox visualization: detections_with_bboxes.png")
        print(f"   • Correlation visualizations: correlation_visualizations/")
        print("="*80)

def main():
    """Main execution function."""
    # Define input paths
    sar_image_path = "snap_output/step4_final_output.tif"
    ais_data_path = "ais_data/AIS_175664700472271242_3396-1756647005869.csv"
    safe_folder_path = "data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    output_base_dir = "maritime_analysis_results"
    
    # Define test region (your specified area)
    test_region = (20200, 12600, 21400, 13900)
    
    # Create and run pipeline
    pipeline = IntegratedMaritimePipeline()
    
    results = pipeline.run_complete_analysis(
        sar_image_path=sar_image_path,
        ais_data_path=ais_data_path,
        safe_folder_path=safe_folder_path,
        output_base_dir=output_base_dir,
        test_region_bounds=test_region
    )
    
    return results

if __name__ == "__main__":
    main()
