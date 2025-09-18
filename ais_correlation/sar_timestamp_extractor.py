#!/usr/bin/env python3
"""
SAR Timestamp Extractor

Extracts the actual SAR image acquisition timestamp from SAFE folder structure.
The timestamp in predictions.csv is NOT the actual SAR acquisition time.
"""

import os
import re
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def extract_sar_timestamp(safe_folder_path: str) -> Optional[datetime]:
    """
    Extract the actual SAR image acquisition timestamp from SAFE folder.
    
    Args:
        safe_folder_path: Path to the SAFE folder
        
    Returns:
        datetime: SAR acquisition timestamp, or None if extraction fails
    """
    try:
        # Method 1: Extract from folder name (most reliable)
        folder_name = os.path.basename(safe_folder_path.rstrip('/'))
        timestamp = _extract_timestamp_from_filename(folder_name)
        if timestamp:
            logger.info(f"✅ Extracted SAR timestamp from folder name: {timestamp}")
            return timestamp
        
        # Method 2: Extract from manifest.safe file
        manifest_path = os.path.join(safe_folder_path, "manifest.safe")
        if os.path.exists(manifest_path):
            timestamp = _extract_timestamp_from_manifest(manifest_path)
            if timestamp:
                logger.info(f"✅ Extracted SAR timestamp from manifest: {timestamp}")
                return timestamp
        
        # Method 3: Extract from annotation XML files
        annotation_dir = os.path.join(safe_folder_path, "annotation")
        if os.path.exists(annotation_dir):
            timestamp = _extract_timestamp_from_annotation(annotation_dir)
            if timestamp:
                logger.info(f"✅ Extracted SAR timestamp from annotation: {timestamp}")
                return timestamp
        
        logger.error(f"❌ Failed to extract SAR timestamp from: {safe_folder_path}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error extracting SAR timestamp: {e}")
        return None

def _extract_timestamp_from_filename(folder_name: str) -> Optional[datetime]:
    """
    Extract timestamp from SAFE folder name.
    
    Format: S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE
    Timestamp: 20230620T230642 (start time)
    """
    try:
        # Pattern: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_...
        pattern = r'S1[AB]_IW_GRDH_1SDV_(\d{8}T\d{6})_'
        match = re.search(pattern, folder_name)
        
        if match:
            timestamp_str = match.group(1)
            # Parse: 20230620T230642 -> 2023-06-20 23:06:42
            timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
            return timestamp
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Error parsing timestamp from filename: {e}")
        return None

def _extract_timestamp_from_manifest(manifest_path: str) -> Optional[datetime]:
    """
    Extract timestamp from manifest.safe file.
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for timestamp patterns in the manifest
        patterns = [
            r'(\d{8}T\d{6})',  # YYYYMMDDTHHMMSS
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',  # YYYY-MM-DDTHH:MM:SS
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Use the first match (usually the acquisition time)
                timestamp_str = matches[0]
                try:
                    if 'T' in timestamp_str and len(timestamp_str) == 15:
                        # Format: 20230620T230642
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
                    else:
                        # Format: 2023-06-20T23:06:42
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                    return timestamp
                except ValueError:
                    continue
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Error extracting timestamp from manifest: {e}")
        return None

def _extract_timestamp_from_annotation(annotation_dir: str) -> Optional[datetime]:
    """
    Extract timestamp from annotation XML files.
    """
    try:
        # Look for VV annotation file
        vv_files = [f for f in os.listdir(annotation_dir) if 'vv' in f.lower() and f.endswith('.xml')]
        
        if not vv_files:
            return None
            
        annotation_file = os.path.join(annotation_dir, vv_files[0])
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for timestamp patterns
        patterns = [
            r'<startTime>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})</startTime>',
            r'<startTime>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})</startTime>',
            r'(\d{8}T\d{6})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                timestamp_str = matches[0]
                try:
                    if '.' in timestamp_str:
                        # Format: 2023-06-20T23:06:42.123456
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
                    elif 'T' in timestamp_str and len(timestamp_str) == 15:
                        # Format: 20230620T230642
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
                    else:
                        # Format: 2023-06-20T23:06:42
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                    return timestamp
                except ValueError:
                    continue
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Error extracting timestamp from annotation: {e}")
        return None

def get_sar_timestamp_from_predictions(predictions_csv_path: str, safe_folder_path: str) -> Optional[datetime]:
    """
    Get the actual SAR timestamp, overriding the timestamp in predictions.csv.
    
    Args:
        predictions_csv_path: Path to predictions.csv
        safe_folder_path: Path to SAFE folder
        
    Returns:
        datetime: Actual SAR acquisition timestamp
    """
    # Extract actual SAR timestamp from SAFE folder
    sar_timestamp = extract_sar_timestamp(safe_folder_path)
    
    if sar_timestamp:
        logger.info(f"🌍 Using actual SAR timestamp: {sar_timestamp}")
        return sar_timestamp
    else:
        logger.warning("⚠️ Failed to extract SAR timestamp, using predictions.csv timestamp as fallback")
        # Fallback to predictions.csv timestamp (not recommended)
        try:
            import pandas as pd
            df = pd.read_csv(predictions_csv_path)
            if len(df) > 0:
                fallback_timestamp = pd.to_datetime(df['timestamp'].iloc[0])
                logger.warning(f"⚠️ Using fallback timestamp: {fallback_timestamp}")
                return fallback_timestamp
        except Exception as e:
            logger.error(f"❌ Error reading fallback timestamp: {e}")
    
    return None

if __name__ == "__main__":
    # Test the timestamp extraction
    logging.basicConfig(level=logging.INFO)
    
    safe_path = "../data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    timestamp = extract_sar_timestamp(safe_path)
    
    if timestamp:
        print(f"✅ SAR Timestamp: {timestamp}")
        print(f"📅 Formatted: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        print("❌ Failed to extract SAR timestamp")
