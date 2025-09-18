#!/usr/bin/env python3
"""
Compute unique AIS vessels in the SAFE scene at the exact SAR timestamp and
compare with detections. Uses SAFE footprint bounds and timestamp from the SAFE.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

from sar_timestamp_extractor import extract_sar_timestamp


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def get_safe_bounds(safe_path: str) -> Optional[Dict[str, float]]:
    """Extract footprint bounds (min/max lat/lon) from SAFE KML or annotation via SAFECoordinateSystem image_bounds."""
    try:
        # Prefer using SAFECoordinateSystem if available to get bounds computed during interpolator build
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'professor'))
        from safe_coordinate_system import SAFECoordinateSystem  # type: ignore

        cs = SAFECoordinateSystem(safe_path)
        # Try KML first; if not, fall back to annotation XML
        if not cs.extract_grid_points():
            return None
        if not cs.build_interpolator():
            return None
        if not cs.image_bounds:
            return None
        return {
            'min_lat': float(cs.image_bounds['min_lat']),
            'max_lat': float(cs.image_bounds['max_lat']),
            'min_lon': float(cs.image_bounds['min_lon']),
            'max_lon': float(cs.image_bounds['max_lon']),
        }
    except Exception:
        return None


def main():
    # Inputs
    safe_folder_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE'))
    ais_csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'ais_data', 'AIS_175664700472271242_3396-1756647005869.csv'))
    results_csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'ais_correlation_results.csv'))

    print('🔍 Scene vs AIS check at exact SAR timestamp')
    print('===========================================')
    print(f'Using SAFE: {safe_folder_path}')

    # 1) Extract SAR timestamp
    sar_ts = extract_sar_timestamp(safe_folder_path)
    if sar_ts is None:
        print('❌ Could not extract SAR timestamp from SAFE')
        return
    print(f'🕒 SAR timestamp: {sar_ts} (UTC)')

    # 2) Extract SAFE footprint bounds
    bounds = get_safe_bounds(safe_folder_path)
    if not bounds:
        print('❌ Could not derive SAFE scene bounds')
        return
    print(f"🗺️  Scene bounds: LAT {bounds['min_lat']:.6f}..{bounds['max_lat']:.6f}, LON {bounds['min_lon']:.6f}..{bounds['max_lon']:.6f}")

    # 3) Load AIS and filter to exact timestamp (±1 minute tolerance)
    ais = pd.read_csv(ais_csv_path)
    ais['BaseDateTime'] = pd.to_datetime(ais['BaseDateTime'])
    tol = timedelta(minutes=1)
    t0, t1 = sar_ts - tol, sar_ts + tol
    ais_t = ais[(ais['BaseDateTime'] >= t0) & (ais['BaseDateTime'] <= t1)]
    print(f'⏱️  AIS within ±1 min: {len(ais_t)} records')

    # Spatial filter by SAFE bounds
    in_scene = ais_t[(ais_t['LAT'] >= bounds['min_lat']) & (ais_t['LAT'] <= bounds['max_lat']) &
                     (ais_t['LON'] >= bounds['min_lon']) & (ais_t['LON'] <= bounds['max_lon'])]
    unique_in_scene = sorted(in_scene['MMSI'].unique().tolist())
    print(f'🛰️  Unique AIS vessels in scene @ SAR time: {len(unique_in_scene)}')

    # 4) Compare with detections matched
    if os.path.exists(results_csv_path):
        res = pd.read_csv(results_csv_path)
        matched = res[res['matched_mmsi'] != 'UNKNOWN']
        matched_mmsi = sorted(matched['matched_mmsi'].astype(str).unique().tolist())
        print(f'📌 Detections matched to AIS: {len(matched_mmsi)}')
        overlap = sorted(list(set(map(str, unique_in_scene)).intersection(set(matched_mmsi))))
        missing = sorted(list(set(map(str, unique_in_scene)) - set(matched_mmsi)))
        extras = sorted(list(set(matched_mmsi) - set(map(str, unique_in_scene))))
        print(f'✅ Overlap (expected & matched): {len(overlap)} -> {overlap}')
        print(f'❌ Expected but not matched: {len(missing)} -> {missing}')
        print(f'❓ Matched but not in-scene@exact time: {len(extras)} -> {extras}')
    else:
        print('ℹ️ No correlation results file found to compare matches.')

    # Save filtered AIS for inspection
    out_csv = os.path.join(os.path.dirname(results_csv_path), 'ais_in_scene_exact_time.csv')
    in_scene.to_csv(out_csv, index=False)
    print(f'💾 Saved AIS in-scene@exact time to: {out_csv}')


if __name__ == '__main__':
    main()



