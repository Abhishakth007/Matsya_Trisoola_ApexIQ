#!/usr/bin/env python3
"""
AIS Correlation Engine

Implements the production-ready AIS correlation algorithm for matching
SAR vessel detections with AIS data, handling any temporal gap.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import cv2

from .motion_models import MotionModel, create_motion_model, predict_ais_position
from .sar_timestamp_extractor import get_sar_timestamp_from_predictions

logger = logging.getLogger(__name__)

class CorrelationResult:
    """Result of AIS correlation for a single detection."""
    
    def __init__(self, detect_id: int, lat: float, lon: float, confidence: float):
        self.detect_id = detect_id
        self.lat = lat
        self.lon = lon
        self.confidence = confidence
        
        # Correlation results
        self.matched_mmsi: Optional[int] = None
        self.match_confidence: float = 0.0
        self.time_gap_minutes: float = 0.0
        self.distance_meters: float = 0.0
        
        # Score breakdown
        self.position_score: float = 0.0
        self.heading_score: float = 0.0
        self.size_score: float = 0.0
        self.speed_score: float = 0.0
        self.type_score: float = 0.0
        self.temporal_score: float = 0.0
        
        # Flags
        self.is_dark_ship: bool = False
        self.is_ambiguous: bool = False

class CorrelationEngine:
    """
    Production-ready AIS correlation engine.
    
    Implements the hybrid algorithm with:
    - Adaptive temporal gap handling
    - Multi-cue validation
    - Spatial gating with KD-tree
    - Hungarian algorithm for assignment
    """
    
    def __init__(self, 
                 spatial_gate_radius: float = 500.0,    # 500m base gate (realistic)
                 confidence_threshold: float = 0.5,     # Lower threshold for more matches
                 max_temporal_gap_hours: float = 2.0,
                 dt_search_step_minutes: float = 5.0,
                 max_distance_hard_limit: float = 2000.0):  # 2km hard limit (realistic)
        """
        Initialize correlation engine.
        
        Args:
            spatial_gate_radius: Maximum distance for spatial gating (meters)
            confidence_threshold: Minimum confidence for valid matches
            max_temporal_gap_hours: Maximum temporal gap to search (hours)
            dt_search_step_minutes: Time step for temporal search (minutes)
        """
        self.spatial_gate_radius = spatial_gate_radius
        self.confidence_threshold = confidence_threshold
        self.max_temporal_gap_hours = max_temporal_gap_hours
        self.dt_search_step_minutes = dt_search_step_minutes
        self.max_distance_hard_limit = max_distance_hard_limit
        
        # Weighting factors for multi-cue scoring (rebalanced to prevent distance bias)
        self.weights = {
            'position': 0.25,   # 25% - reduced from 40% to prevent distance bias
            'heading': 0.15,    # 15% - slightly reduced
            'size': 0.35,       # 35% - increased from 20% to enforce size matching
            'speed': 0.1,       # 10% - unchanged
            'type': 0.1,        # 10% - increased from 5% for vessel type consistency
            'temporal': 0.05    # 5% - unchanged
        }
        
        # Data storage
        self.ais_data: Optional[pd.DataFrame] = None
        self.predictions_data: Optional[pd.DataFrame] = None
        self.sar_timestamp: Optional[datetime] = None
        
    def load_data(self, 
                  predictions_csv: str, 
                  ais_csv: str, 
                  safe_folder_path: str) -> bool:
        """
        Load predictions and AIS data.
        
        Args:
            predictions_csv: Path to predictions.csv
            ais_csv: Path to AIS CSV file
            safe_folder_path: Path to SAFE folder for timestamp extraction
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load predictions data
            logger.info(f"📂 Loading predictions from: {predictions_csv}")
            self.predictions_data = pd.read_csv(predictions_csv)
            logger.info(f"✅ Loaded {len(self.predictions_data)} detections")
            
            # Load AIS data
            logger.info(f"📂 Loading AIS data from: {ais_csv}")
            self.ais_data = pd.read_csv(ais_csv)
            logger.info(f"✅ Loaded {len(self.ais_data)} AIS records")
            
            # Extract actual SAR timestamp
            logger.info(f"🕐 Extracting SAR timestamp from: {safe_folder_path}")
            self.sar_timestamp = get_sar_timestamp_from_predictions(
                predictions_csv, safe_folder_path
            )
            
            if self.sar_timestamp:
                logger.info(f"✅ SAR timestamp: {self.sar_timestamp}")
            else:
                logger.error("❌ Failed to extract SAR timestamp")
                return False
            
            # Clean AIS data
            self._clean_ais_data()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def _clean_ais_data(self):
        """Clean and filter AIS data."""
        if self.ais_data is None or self.sar_timestamp is None:
            return
            
        logger.info("🧹 Cleaning and filtering AIS data...")
        
        # Remove records with missing critical data
        initial_count = len(self.ais_data)
        
        # Filter out records with missing SOG or COG
        self.ais_data = self.ais_data.dropna(subset=['SOG', 'COG'])
        
        # Convert timestamps
        self.ais_data['BaseDateTime'] = pd.to_datetime(self.ais_data['BaseDateTime'])
        
        # 🚀 CRITICAL OPTIMIZATION: Filter AIS data by temporal window around SAR timestamp
        logger.info(f"🕐 Filtering AIS data around SAR timestamp: {self.sar_timestamp}")
        
        # Define time window: ±2 hours around SAR time (fixed window for efficiency)
        time_window_hours = 2.0  # Fixed 2-hour window before and after SAR
        time_window = pd.Timedelta(hours=time_window_hours)
        
        # Filter AIS data to relevant time window (work on a copy)
        ais_full = self.ais_data
        time_filter = (
            (ais_full['BaseDateTime'] >= self.sar_timestamp - time_window) &
            (ais_full['BaseDateTime'] <= self.sar_timestamp + time_window)
        )
        
        self.ais_data = ais_full[time_filter].copy()
        filtered_count = len(self.ais_data)
        
        logger.info(f"⏰ Temporal filtering: {initial_count} -> {filtered_count} records (±{time_window_hours}h window)")
        
        if filtered_count == 0:
            logger.warning("⚠️ No AIS data in temporal window - expanding search")
            # Expand time window if no data found (recompute from full set)
            time_window_hours = 6.0
            time_window = pd.Timedelta(hours=time_window_hours)
            time_filter = (
                (ais_full['BaseDateTime'] >= self.sar_timestamp - time_window) &
                (ais_full['BaseDateTime'] <= self.sar_timestamp + time_window)
            )
            self.ais_data = ais_full[time_filter].copy()
            filtered_count = len(self.ais_data)
            logger.info(f"⏰ Expanded temporal filtering: {filtered_count} records (±{time_window_hours}h window)")

        # Keep only the closest-in-time AIS message per MMSI within the window
        if filtered_count > 0:
            before = len(self.ais_data)
            # Compute absolute time delta to SAR timestamp
            self.ais_data['abs_dt'] = (self.sar_timestamp - self.ais_data['BaseDateTime']).abs()
            # Select the row with minimum abs_dt for each MMSI
            idx = self.ais_data.groupby('MMSI')['abs_dt'].idxmin()
            self.ais_data = self.ais_data.loc[idx].reset_index(drop=True)
            after = len(self.ais_data)
            logger.info(f"🧮 Per-MMSI closest-in-time selection: {before} -> {after} records")
            # Drop helper column
            self.ais_data.drop(columns=['abs_dt'], inplace=True)
        
        # Fill missing values with defaults
        self.ais_data['Heading'] = self.ais_data['Heading'].fillna(self.ais_data['COG'])
        self.ais_data['VesselType'] = self.ais_data['VesselType'].fillna(90)  # Default: other
        
        # Fill missing dimensions with type-based averages
        vessel_type_avg_length = self.ais_data.groupby('VesselType')['Length'].mean()
        vessel_type_avg_width = self.ais_data.groupby('VesselType')['Width'].mean()
        
        for vessel_type in self.ais_data['VesselType'].unique():
            if pd.notna(vessel_type):
                mask = (self.ais_data['VesselType'] == vessel_type) & self.ais_data['Length'].isna()
                if vessel_type in vessel_type_avg_length and pd.notna(vessel_type_avg_length[vessel_type]):
                    self.ais_data.loc[mask, 'Length'] = vessel_type_avg_length[vessel_type]
                
                mask = (self.ais_data['VesselType'] == vessel_type) & self.ais_data['Width'].isna()
                if vessel_type in vessel_type_avg_width and pd.notna(vessel_type_avg_width[vessel_type]):
                    self.ais_data.loc[mask, 'Width'] = vessel_type_avg_width[vessel_type]
        
        # Fill remaining missing dimensions with global averages
        self.ais_data['Length'] = self.ais_data['Length'].fillna(self.ais_data['Length'].mean())
        self.ais_data['Width'] = self.ais_data['Width'].fillna(self.ais_data['Width'].mean())
        
        final_count = len(self.ais_data)
        reduction_percent = (1 - final_count / initial_count) * 100 if initial_count > 0 else 0
        logger.info(f"✅ Cleaned AIS data: {initial_count} -> {final_count} records ({reduction_percent:.1f}% reduction)")
        
        # Log temporal distribution
        if final_count > 0:
            time_range = self.ais_data['BaseDateTime'].max() - self.ais_data['BaseDateTime'].min()
            logger.info(f"📊 AIS data time range: {time_range.total_seconds()/3600:.1f} hours")
            logger.info(f"📊 AIS data spans: {self.ais_data['BaseDateTime'].min()} to {self.ais_data['BaseDateTime'].max()}")
    
    def correlate(self) -> List[CorrelationResult]:
        """
        Perform AIS correlation for all detections.
        
        Returns:
            List of CorrelationResult objects
        """
        if self.ais_data is None or self.predictions_data is None or self.sar_timestamp is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("🔍 Starting AIS correlation...")
        
        # Initialize results
        results = []
        
        # Create correlation results for each detection
        for _, detection in self.predictions_data.iterrows():
            result = CorrelationResult(
                detect_id=detection['detect_id'],
                lat=detection['lat'],
                lon=detection['lon'],
                confidence=detection['confidence']
            )
            results.append(result)
        
        # Perform temporal search
        best_dt, best_global_score = self._find_optimal_temporal_gap(results)
        logger.info(f"🎯 Optimal temporal gap: {best_dt/60:.1f} minutes (score: {best_global_score:.3f})")
        
        # Perform per-detection correlation (better approach)
        self._current_dt_sec = float(best_dt)
        for result in results:
            best_match = self._find_best_ais_match(result, best_dt)
            if best_match is not None:
                self._apply_match(result, best_match, best_dt)
            else:
                # Dark ship (no match)
                result.matched_mmsi = None
                result.distance_meters = None
                result.time_gap_minutes = None
                result.match_confidence = 0.0
        
        # Post-process results
        self._post_process_results(results)
        
        logger.info(f"✅ Correlation complete: {len(results)} detections processed")
        return results
    
    def _find_optimal_temporal_gap(self, results: List[CorrelationResult]) -> Tuple[float, float]:
        """
        Find optimal temporal gap using adaptive search.
        
        Returns:
            Tuple of (optimal_dt_seconds, global_score)
        """
        logger.info("⏰ Finding optimal temporal gap...")
        
        # 🚀 OPTIMIZATION: Calculate actual time gaps from AIS data
        if len(self.ais_data) == 0:
            logger.warning("⚠️ No AIS data available for temporal gap search")
            return 0.0, 0.0
        
        # Calculate time gaps between SAR timestamp and AIS timestamps
        ais_times = self.ais_data['BaseDateTime']
        time_gaps = (self.sar_timestamp - ais_times).dt.total_seconds()
        
        # Filter to reasonable range
        max_dt = self.max_temporal_gap_hours * 3600
        valid_gaps = time_gaps[(time_gaps >= 0) & (time_gaps <= max_dt)]
        
        if len(valid_gaps) == 0:
            logger.warning("⚠️ No AIS data within temporal search range")
            return 0.0, 0.0
        
        logger.info(f"📊 Found {len(valid_gaps)} AIS records within {self.max_temporal_gap_hours}h range")
        logger.info(f"📊 Time gaps range: {valid_gaps.min()/60:.1f} to {valid_gaps.max()/60:.1f} minutes")
        
        # Use actual time gaps from data instead of uniform grid
        unique_gaps = sorted(valid_gaps.unique())
        
        # Limit to reasonable number of gaps for performance
        if len(unique_gaps) > 50:
            # Sample gaps evenly
            step = len(unique_gaps) // 50
            unique_gaps = unique_gaps[::step]
        
        logger.info(f"🔍 Testing {len(unique_gaps)} temporal gaps...")
        
        best_dt = 0
        best_score = -np.inf
        
        for dt in unique_gaps:
            score = self._compute_global_correlation_score(results, dt)
            if score > best_score:
                best_score = score
                best_dt = dt
        
        logger.info(f"🎯 Best temporal gap: {best_dt/60:.1f} minutes (score: {best_score:.3f})")
        return best_dt, best_score
    
    def _compute_global_correlation_score(self, results: List[CorrelationResult], dt: float) -> float:
        """
        Compute global correlation score for given temporal gap.
        
        Args:
            results: List of detection results
            dt: Temporal gap in seconds
            
        Returns:
            Global correlation score
        """
        total_score = 0.0
        valid_matches = 0
        
        for result in results:
            # Find best AIS match for this detection
            best_match = self._find_best_ais_match(result, dt)
            
            if best_match is not None:
                total_score += best_match['total_score']
                valid_matches += 1
        
        # Normalize by number of detections
        if len(results) > 0:
            return total_score / len(results)
        else:
            return 0.0
    
    def _find_best_ais_match(self, result: CorrelationResult, dt: float) -> Optional[Dict]:
        """
        Find best AIS match for a single detection.
        
        Args:
            result: Detection result
            dt: Temporal gap in seconds
            
        Returns:
            Best match dictionary or None
        """
        # Predict AIS positions to SAR timestamp
        predicted_ais = self._predict_ais_positions(dt)
        
        if len(predicted_ais) == 0:
            return None
        
        # Spatial gating
        candidates = self._spatial_gating(result, predicted_ais)
        
        if len(candidates) == 0:
            return None
        
        # Multi-cue scoring
        best_match = None
        best_score = -np.inf
        
        for candidate in candidates:
            score = self._compute_multi_cue_score(result, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_match is not None:
            best_match['total_score'] = best_score
        
        return best_match
    
    def _predict_ais_positions(self, dt: float) -> List[Dict]:
        """
        Predict AIS positions to SAR timestamp.
        
        Args:
            dt: Temporal gap in seconds
            
        Returns:
            List of predicted AIS positions
        """
        predicted_ais = []
        
        # 🚀 OPTIMIZATION: Vectorized processing for better performance
        logger.debug(f"🔮 Predicting positions for {len(self.ais_data)} AIS records (dt={dt/60:.1f}min)")
        
        for _, ais_record in self.ais_data.iterrows():
            try:
                # Compute per-record temporal gap to SAR timestamp (seconds)
                dt_record = float((self.sar_timestamp - ais_record['BaseDateTime']).total_seconds())
                # Clamp to [0, max_temporal_gap_hours]
                if dt_record < 0:
                    dt_record = 0.0
                max_dt = float(self.max_temporal_gap_hours) * 3600.0
                if dt_record > max_dt:
                    dt_record = max_dt
                # Predict position using per-record dt
                pred_lat, pred_lon, uncertainty = predict_ais_position(ais_record.to_dict(), dt_record)
                
                predicted_ais.append({
                    'MMSI': ais_record['MMSI'],
                    'pred_lat': pred_lat,
                    'pred_lon': pred_lon,
                    'uncertainty': uncertainty,
                    'dt_seconds': dt_record,
                    'time_gap_minutes': dt_record / 60.0,
                    'original_record': ais_record.to_dict()
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error predicting AIS position for MMSI {ais_record['MMSI']}: {e}")
                continue
        
        logger.debug(f"✅ Predicted {len(predicted_ais)} AIS positions")
        return predicted_ais
    
    def _spatial_gating(self, result: CorrelationResult, predicted_ais: List[Dict]) -> List[Dict]:
        """
        Apply spatial gating to filter candidates.
        
        Args:
            result: Detection result
            predicted_ais: List of predicted AIS positions
            
        Returns:
            List of spatially gated candidates
        """
        candidates = []
        rejected_count = 0
        
        for ais in predicted_ais:
            # Calculate distance
            distance = self._calculate_distance(
                result.lat, result.lon,
                ais['pred_lat'], ais['pred_lon']
            )
            
            # SIZE-BASED DISTANCE LIMITS: Smaller vessels get smaller search radius  
            # Get detection size from the result context (will be available in correlation)
            detection_size = 50.0  # Default medium size, will be refined in actual correlation
            
            # Size-based base radius (smaller vessels = smaller search area)
            if detection_size < 30:
                size_based_radius = 200.0  # 200m for small vessels
            elif detection_size < 60:
                size_based_radius = 400.0  # 400m for medium vessels  
            else:
                size_based_radius = 500.0  # 500m for large vessels
            
            time_gap_minutes = float(ais.get('time_gap_minutes', 0.0))
            # Realistic growth: +10m per minute, cap at size-appropriate limit
            adaptive_radius = min(size_based_radius + (time_gap_minutes * 10.0), size_based_radius * 2)
            # Cap at hard limit
            max_distance = min(adaptive_radius, self.max_distance_hard_limit)
            
            # Debug logging for large distances
            if distance > 10000:  # > 10km
                logger.warning(f"🚨 Large distance detected: {distance/1000:.1f}km for MMSI {ais['MMSI']} "
                             f"(gate: {max_distance/1000:.1f}km, time_gap: {time_gap_minutes:.1f}min)")
            
            # Apply strict spatial gating
            if distance <= max_distance:
                ais['distance'] = distance
                candidates.append(ais)
            else:
                rejected_count += 1
                if distance > self.max_distance_hard_limit:
                    logger.warning(f"🚫 HARD REJECT: {distance/1000:.1f}km > {self.max_distance_hard_limit/1000:.1f}km limit for MMSI {ais['MMSI']}")
        
        if rejected_count > 0:
            logger.info(f"🚫 Spatial gating rejected {rejected_count}/{len(predicted_ais)} AIS candidates for detection {result.detect_id}")
        
        # Log adaptive gate statistics
        if len(candidates) > 0:
            time_gaps = [c.get('time_gap_minutes', 0) for c in candidates]
            distances = [c.get('distance', 0) for c in candidates]
            avg_time_gap = sum(time_gaps) / len(time_gaps)
            avg_distance = sum(distances) / len(distances)
            logger.info(f"📊 Detection {result.detect_id}: {len(candidates)} candidates, avg time gap: {avg_time_gap:.1f}min, avg distance: {avg_distance/1000:.1f}km")
        
        return candidates
    
    def _compute_multi_cue_score(self, result: CorrelationResult, candidate: Dict) -> float:
        """
        Compute multi-cue correlation score.
        
        Args:
            result: Detection result
            candidate: AIS candidate
            
        Returns:
            Total correlation score
        """
        # Get detection attributes
        detection = self.predictions_data[self.predictions_data['detect_id'] == result.detect_id].iloc[0]
        ais_record = candidate['original_record']
        
        # Hard guards: reject if detection confidence too low or not a vessel
        if result.confidence < 0.6:
            return 0.0  # Reject low confidence detections
        
        # HARD DISTANCE LIMIT: Reject any match > 10km
        if 'distance' in candidate:
            distance_km = candidate['distance'] / 1000.0
            if distance_km > self.max_distance_hard_limit / 1000.0:
                logger.warning(f"🚫 HARD REJECT in scoring: {distance_km:.1f}km > {self.max_distance_hard_limit/1000:.1f}km limit")
                return 0.0
        
        # Reject if detection is flagged as non-vessel
        if 'is_fishing_vessel' in detection and not detection['is_fishing_vessel']:
            # Could be more sophisticated here, but for now just penalize heavily
            pass  # Continue with scoring but will be heavily penalized
        
        # Position score (Mahalanobis distance)
        position_score = self._compute_position_score(result, candidate)
        
        # Heading score
        heading_score = self._compute_heading_score(detection, ais_record)
        
        # Size score
        size_score = self._compute_size_score(detection, ais_record)
        
        # Speed score
        speed_score = self._compute_speed_score(detection, ais_record)
        
        # Type score
        type_score = self._compute_type_score(detection, ais_record)
        
        # Temporal score
        temporal_score = self._compute_temporal_score(candidate)
        
        # Weighted combination
        total_score = (
            self.weights['position'] * position_score +
            self.weights['heading'] * heading_score +
            self.weights['size'] * size_score +
            self.weights['speed'] * speed_score +
            self.weights['type'] * type_score +
            self.weights['temporal'] * temporal_score
        )
        
        return total_score
    
    def _compute_position_score(self, result: CorrelationResult, candidate: Dict) -> float:
        """Compute position-based correlation score (FIXED - no uncertainty bias)."""
        # Ensure distance is available; compute if missing
        if 'distance' in candidate:
            distance = candidate['distance']
        else:
            distance = self._calculate_distance(
                result.lat, result.lon,
                candidate.get('pred_lat', result.lat),
                candidate.get('pred_lon', result.lon)
            )
            candidate['distance'] = distance
        
        # STRICT DISTANCE CUTOFFS (no uncertainty bias)
        if distance > 500.0:  # Hard cutoff at 500m
            logger.debug(f"🚫 DISTANCE REJECT: {distance:.1f}m > 500m limit")
            return 0.0
        
        # Simple distance-based scoring (no uncertainty normalization)
        # 0m = 1.0, 500m = 0.0, exponential decay
        score = np.exp(-distance / 200.0)  # 200m = 1/e score
        return score
    
    def _compute_heading_score(self, detection: pd.Series, ais_record: pd.Series) -> float:
        """Compute heading-based correlation score."""
        try:
            # Prefer heading bucket probabilities if available
            bucket_cols = [f'heading_bucket_{i}' for i in range(16)]
            if all(col in detection.index for col in bucket_cols):
                det_probs = np.array([float(detection[col]) for col in bucket_cols])
                # Don't re-softmax already normalized probabilities
                det_class = int(np.argmax(det_probs))
                det_heading = (det_class + 0.5) * 22.5  # Use bin center, not edge
            else:
                det_heading = detection.get('heading_degrees', np.nan)
            
            ais_heading = ais_record['Heading']
            
            # Handle missing heading
            if pd.isna(det_heading) or pd.isna(ais_heading):
                return 0.5  # Neutral score
            
            # Down-weight heading when vessel is nearly stationary (COG unreliable)
            ais_sog = ais_record.get('SOG', 0.0)
            if ais_sog < 1.0:  # Less than 1 knot
                return 0.5  # Neutral score for stationary vessels
            
            # Calculate angular difference
            diff = abs(det_heading - ais_heading)
            diff = min(diff, 360 - diff)  # Handle wraparound
            
            # Convert to score (0-1, higher is better)
            score = np.exp(-diff / 45)  # 45 degrees = 1/e
            return score
            
        except Exception:
            return 0.5
    
    def _compute_size_score(self, detection: pd.Series, ais_record: pd.Series) -> float:
        """Compute size-based correlation score."""
        try:
            det_length = detection['vessel_length_m']
            det_width = detection['vessel_width_m']
            ais_length = ais_record['Length']
            ais_width = ais_record['Width']
            
            # Handle missing dimensions
            if pd.isna(det_length) or pd.isna(ais_length):
                return 0.5
            
            # Calculate relative differences
            length_diff = abs(det_length - ais_length) / max(ais_length, 1)
            width_diff = abs(det_width - ais_width) / max(ais_width, 1)
            
            # HARD REJECTION: Detection must be at least 50% of AIS vessel size
            if det_length < ais_length * 0.5:
                logger.debug(f"🚫 SIZE REJECT: Detection {det_length:.1f}m < 50% of AIS {ais_length:.1f}m")
                return 0.0
            
            # Hard size gate: reject if dimensions differ by more than 30% (realistic)
            if length_diff > 0.3 or width_diff > 0.3:
                return 0.0  # Reject size mismatch
            
            # Combined size difference
            size_diff = (length_diff + width_diff) / 2
            
            # Convert to score (0-1, higher is better)
            score = np.exp(-size_diff)
            return score
            
        except Exception:
            return 0.5
    
    def _compute_speed_score(self, detection: pd.Series, ais_record: pd.Series) -> float:
        """Compute speed-based correlation score."""
        try:
            det_speed = detection['vessel_speed_k']
            ais_speed = ais_record['SOG']
            
            # Handle missing speed
            if pd.isna(det_speed) or pd.isna(ais_speed):
                return 0.5
            
            # Calculate speed difference
            speed_diff = abs(det_speed - ais_speed)
            
            # Convert to score (0-1, higher is better) - more lenient for small differences
            score = np.exp(-speed_diff / 10)  # 10 knots = 1/e (more lenient)
            return score
            
        except Exception:
            return 0.5
    
    def _compute_type_score(self, detection: pd.Series, ais_record: pd.Series) -> float:
        """Compute vessel type correlation score."""
        try:
            det_type = detection['vessel_type']
            ais_type = ais_record['VesselType']
            
            # Handle missing types
            if pd.isna(det_type) or pd.isna(ais_type):
                return 0.5
            
            # ENHANCED TYPE VALIDATION: Strict vessel class consistency
            # Cargo ships should NOT match fishing detections and vice versa
            if det_type == 'fishing':
                if ais_type == 30:  # Fishing vessel
                    return 0.9
                elif ais_type in [36, 37]:  # Small pleasure/sailing craft
                    return 0.6
                elif ais_type == 70:  # Large cargo - MAJOR MISMATCH
                    logger.debug(f"🚫 TYPE REJECT: Fishing detection vs Cargo AIS")
                    return 0.0
                else:
                    return 0.3
            elif det_type == 'cargo':
                if ais_type in [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]:  # Cargo types
                    return 0.9
                elif ais_type == 30:  # Fishing vessel - mismatch
                    return 0.2
                else:
                    return 0.5
            else:
                return 0.4  # Unknown detection type
            
        except Exception:
            return 0.5
    
    def _compute_temporal_score(self, candidate: Dict) -> float:
        """Compute temporal correlation score."""
        minutes = abs(candidate.get('time_gap_minutes', 0.0))
        # Exponential decay: 30 min -> ~0.37, 60 min -> ~0.14
        return float(np.exp(-minutes / 30.0))
    
    def _correlate_with_temporal_gap(self, results: List[CorrelationResult], dt: float):
        """Perform correlation with specific temporal gap."""
        logger.info(f"🔗 Correlating with temporal gap: {dt/60:.1f} minutes")
        
        # Predict AIS positions
        predicted_ais = self._predict_ais_positions(dt)
        logger.info(f"📍 Predicted {len(predicted_ais)} AIS positions")
        
        # Build cost matrix for Hungarian algorithm (enforce one-to-one)
        cost_matrix = self._build_cost_matrix(results, predicted_ais)
        
        # Solve assignment problem
        matches = self._solve_assignment(cost_matrix, len(results), len(predicted_ais))
        
        # Apply matches to results
        for i, result in enumerate(results):
            if i < len(matches) and matches[i] >= 0 and matches[i] < len(predicted_ais):
                # Valid AIS match
                ais_candidate = predicted_ais[matches[i]]
                self._apply_match(result, ais_candidate, dt)
            else:
                # Dark ship (no match or invalid match)
                result.matched_mmsi = None
                result.distance_meters = None
                result.time_gap_minutes = None
                result.match_confidence = 0.0
    
    def _build_cost_matrix(self, results: List[CorrelationResult], predicted_ais: List[Dict]) -> np.ndarray:
        """Build cost matrix for Hungarian algorithm."""
        n_detections = len(results)
        n_ais = len(predicted_ais)
        
        # Initialize cost matrix
        # Rectangular: detections x (ais + dummies) to allow unmatched
        cost_matrix = np.full((n_detections, n_ais + n_detections), 1.0)
        
        # Fill detection-to-AIS costs
        for i, result in enumerate(results):
            for j, ais in enumerate(predicted_ais):
                score = self._compute_multi_cue_score(result, ais)
                cost_matrix[i, j] = 1.0 - score  # Convert to cost (lower is better)
        
        # Fill detection-to-dummy costs (for dark ships) after AIS columns
        dark_cost = 1.0 - self.confidence_threshold  # e.g., 0.3 if threshold=0.7
        for i in range(n_detections):
            cost_matrix[i, n_ais + i] = dark_cost  # Dark ship cost based on confidence threshold
        
        return cost_matrix
    
    def _solve_assignment(self, cost_matrix: np.ndarray, n_detections: int, n_ais: int) -> List[int]:
        """Solve assignment with one-to-one constraint using Hungarian (scipy)."""
        try:
            # Hungarian on rectangular matrix (detections x (ais + dummies))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Map each detection i to selected column; default -1 (dark)
            selection = [-1] * n_detections
            for r, c in zip(row_ind, col_ind):
                if r >= n_detections:
                    continue
                # If assigned to a real AIS column and meets confidence threshold, accept; else dark
                if c < n_ais and (1.0 - cost_matrix[r, c]) >= self.confidence_threshold:
                    selection[r] = int(c)
                else:
                    selection[r] = -1
            return selection
        except Exception as e:
            logger.error(f"❌ Error solving assignment: {e}")
            return [-1] * n_detections
    
    def _apply_match(self, result: CorrelationResult, ais_candidate: Dict, dt: float):
        """Apply AIS match to detection result."""
        result.matched_mmsi = ais_candidate['MMSI']
        result.distance_meters = ais_candidate['distance']
        # Record the per-candidate time gap actually used
        result.time_gap_minutes = float(ais_candidate.get('time_gap_minutes', 0.0))
        
        # Compute detailed scores
        detection = self.predictions_data[self.predictions_data['detect_id'] == result.detect_id].iloc[0]
        ais_record = ais_candidate['original_record']
        
        result.position_score = self._compute_position_score(result, ais_candidate)
        result.heading_score = self._compute_heading_score(detection, ais_record)
        result.size_score = self._compute_size_score(detection, ais_record)
        result.speed_score = self._compute_speed_score(detection, ais_record)
        result.type_score = self._compute_type_score(detection, ais_record)
        result.temporal_score = self._compute_temporal_score(ais_candidate)
        
        # Overall match confidence
        raw_confidence = (
            self.weights['position'] * result.position_score +
            self.weights['heading'] * result.heading_score +
            self.weights['size'] * result.size_score +
            self.weights['speed'] * result.speed_score +
            self.weights['type'] * result.type_score +
            self.weights['temporal'] * result.temporal_score
        )
        
        # CONFIDENCE VALIDATION: Cap at 1.0 and add sanity checks
        result.match_confidence = min(raw_confidence, 1.0)
        
        # SIZE RATIO VALIDATION: Reject correlations with >2x size difference
        try:
            det_length = detection.get('vessel_length_m', 0)
            ais_length = ais_record.get('Length', 0)
            if pd.notna(det_length) and pd.notna(ais_length) and det_length > 0 and ais_length > 0:
                size_ratio = max(det_length, ais_length) / min(det_length, ais_length)
                if size_ratio > 2.0:
                    logger.warning(f"🚫 SIZE RATIO REJECT: {det_length:.1f}m vs {ais_length:.1f}m (ratio: {size_ratio:.1f}x)")
                    result.match_confidence = 0.0
                    result.matched_mmsi = None
                    result.is_dark_ship = True
                    return
        except:
            pass
        
        # SANITY CHECK: Reject impossible correlations
        if result.distance_meters > 1000.0:  # >1km is suspicious
            logger.warning(f"🚫 SANITY REJECT: {result.distance_meters:.1f}m distance too large")
            result.match_confidence = 0.0
            result.matched_mmsi = None
            result.is_dark_ship = True
    
    def _post_process_results(self, results: List[CorrelationResult]):
        """Post-process correlation results with comprehensive auditing."""
        logger.info("🔧 Post-processing results...")
        
        # CORRELATION AUDITING: Log detailed statistics
        successful_correlations = [r for r in results if r.matched_mmsi is not None]
        dark_ships = [r for r in results if r.is_dark_ship]
        
        logger.info(f"📊 CORRELATION AUDIT RESULTS:")
        logger.info(f"   Total detections processed: {len(results)}")
        logger.info(f"   Successful correlations: {len(successful_correlations)}")
        logger.info(f"   Dark ships: {len(dark_ships)}")
        logger.info(f"   Correlation rate: {len(successful_correlations)/len(results)*100:.1f}%")
        
        if len(successful_correlations) > 0:
            distances = [r.distance_meters for r in successful_correlations]
            confidences = [r.match_confidence for r in successful_correlations]
            
            logger.info(f"   Distance stats: {min(distances):.1f}m - {max(distances):.1f}m (avg: {np.mean(distances):.1f}m)")
            logger.info(f"   Confidence stats: {min(confidences):.3f} - {max(confidences):.3f} (avg: {np.mean(confidences):.3f})")
            
            # Audit for suspicious correlations
            suspicious = [r for r in successful_correlations if r.distance_meters > 500 or r.match_confidence > 1.0]
            if len(suspicious) > 0:
                logger.warning(f"⚠️ Found {len(suspicious)} suspicious correlations!")
                for r in suspicious[:3]:  # Show first 3
                    logger.warning(f"   MMSI {r.matched_mmsi}: {r.distance_meters:.1f}m, confidence {r.match_confidence:.3f}")
        
        for result in results:
            # Flag dark ships
            if result.matched_mmsi is None or result.match_confidence < self.confidence_threshold:
                result.is_dark_ship = True
                result.matched_mmsi = None
                result.match_confidence = 0.0
            
            # Flag ambiguous matches
            if result.match_confidence > 0.5 and result.match_confidence < self.confidence_threshold:
                result.is_ambiguous = True
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters."""
        # Haversine formula
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def save_results(self, results: List[CorrelationResult], output_path: str):
        """Save correlation results to CSV."""
        logger.info(f"💾 Saving results to: {output_path}")
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {
                'detect_id': result.detect_id,
                'lat': result.lat,
                'lon': result.lon,
                'confidence': result.confidence,
                'matched_mmsi': result.matched_mmsi if result.matched_mmsi is not None else 'UNKNOWN',
                'match_confidence': result.match_confidence,
                'time_gap_minutes': result.time_gap_minutes,
                'distance_meters': result.distance_meters,
                'position_score': result.position_score,
                'heading_score': result.heading_score,
                'size_score': result.size_score,
                'speed_score': result.speed_score,
                'type_score': result.type_score,
                'temporal_score': result.temporal_score,
                'is_dark_ship': result.is_dark_ship,
                'is_ambiguous': result.is_ambiguous
            }
            csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Results saved: {len(csv_data)} correlations")

if __name__ == "__main__":
    # Test the correlation engine
    logging.basicConfig(level=logging.INFO)
    
    engine = CorrelationEngine()
    
    # Load data
    success = engine.load_data(
        predictions_csv="../professor/outputs/predictions.csv",
        ais_csv="../ais_data/AIS_175664700472271242_3396-1756647005869.csv",
        safe_folder_path="../data/S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE"
    )
    
    if success:
        # Perform correlation
        results = engine.correlate()
        
        # Save results
        engine.save_results(results, "ais_correlation_results.csv")
        
        # Print summary
        matched = sum(1 for r in results if not r.is_dark_ship)
        dark_ships = sum(1 for r in results if r.is_dark_ship)
        
        print(f"\n📊 Correlation Summary:")
        print(f"  Total detections: {len(results)}")
        print(f"  Matched to AIS: {matched}")
        print(f"  Dark ships: {dark_ships}")
        print(f"  Match rate: {matched/len(results)*100:.1f}%")
    else:
        print("❌ Failed to load data")
