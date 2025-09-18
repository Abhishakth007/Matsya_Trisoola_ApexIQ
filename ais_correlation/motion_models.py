#!/usr/bin/env python3
"""
Motion Models for AIS Track Prediction

Implements motion models for predicting vessel positions over time gaps.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MotionModel(ABC):
    """Abstract base class for motion models."""
    
    @abstractmethod
    def predict(self, state: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state and covariance.
        
        Args:
            state: Current state vector [x, y, vx, vy]
            dt: Time step in seconds
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        pass
    
    @abstractmethod
    def get_process_noise(self, dt: float) -> np.ndarray:
        """
        Get process noise matrix for given time step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Process noise matrix
        """
        pass

class ConstantVelocityModel(MotionModel):
    """
    Constant Velocity (CV) motion model.
    
    State vector: [x, y, vx, vy] (position and velocity)
    Assumes constant velocity with process noise.
    """
    
    def __init__(self, process_noise_std: float = 1.0):
        """
        Initialize CV model.
        
        Args:
            process_noise_std: Standard deviation of process noise (m/s²)
        """
        self.process_noise_std = process_noise_std
        
    def predict(self, state: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using CV model.
        
        State transition matrix F:
        [1  0  dt  0 ]
        [0  1  0   dt]
        [0  0  1   0 ]
        [0  0  0   1 ]
        """
        if len(state) != 4:
            raise ValueError(f"CV model expects 4D state, got {len(state)}D")
            
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Predict state
        predicted_state = F @ state
        
        # Process noise matrix
        Q = self.get_process_noise(dt)
        
        return predicted_state, Q
    
    def get_process_noise(self, dt: float) -> np.ndarray:
        """
        Get process noise matrix for CV model.
        
        Q = σ² * [dt⁴/4  0     dt³/2  0   ]
                 [0      dt⁴/4  0     dt³/2]
                 [dt³/2  0     dt²    0   ]
                 [0      dt³/2  0     dt²  ]
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        sigma2 = self.process_noise_std ** 2
        
        Q = sigma2 * np.array([
            [dt4/4, 0, dt3/2, 0],
            [0, dt4/4, 0, dt3/2],
            [dt3/2, 0, dt2, 0],
            [0, dt3/2, 0, dt2]
        ])
        
        return Q

class ConstantTurnRateVelocityModel(MotionModel):
    """
    Constant Turn Rate and Velocity (CTRV) motion model.
    
    State vector: [x, y, v, heading, turn_rate]
    Assumes constant velocity and turn rate with process noise.
    """
    
    def __init__(self, process_noise_std: float = 1.0, turn_rate_noise_std: float = 0.1):
        """
        Initialize CTRV model.
        
        Args:
            process_noise_std: Standard deviation of velocity process noise (m/s²)
            turn_rate_noise_std: Standard deviation of turn rate process noise (rad/s²)
        """
        self.process_noise_std = process_noise_std
        self.turn_rate_noise_std = turn_rate_noise_std
        
    def predict(self, state: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using CTRV model.
        
        State: [x, y, v, heading, turn_rate]
        """
        if len(state) != 5:
            raise ValueError(f"CTRV model expects 5D state, got {len(state)}D")
            
        x, y, v, heading, turn_rate = state
        
        # Handle zero turn rate case
        if abs(turn_rate) < 1e-6:
            # Linear motion
            new_x = x + v * np.cos(heading) * dt
            new_y = y + v * np.sin(heading) * dt
            new_v = v
            new_heading = heading
            new_turn_rate = turn_rate
        else:
            # Curved motion
            new_x = x + (v / turn_rate) * (np.sin(heading + turn_rate * dt) - np.sin(heading))
            new_y = y + (v / turn_rate) * (np.cos(heading) - np.cos(heading + turn_rate * dt))
            new_v = v
            new_heading = heading + turn_rate * dt
            new_turn_rate = turn_rate
        
        predicted_state = np.array([new_x, new_y, new_v, new_heading, new_turn_rate])
        
        # Process noise matrix (simplified)
        Q = self.get_process_noise(dt)
        
        return predicted_state, Q
    
    def get_process_noise(self, dt: float) -> np.ndarray:
        """
        Get process noise matrix for CTRV model.
        """
        dt2 = dt * dt
        
        # Simplified process noise matrix
        Q = np.diag([
            self.process_noise_std**2 * dt2,  # x position noise
            self.process_noise_std**2 * dt2,  # y position noise
            self.process_noise_std**2 * dt2,  # velocity noise
            self.turn_rate_noise_std**2 * dt2,  # heading noise
            self.turn_rate_noise_std**2 * dt2   # turn rate noise
        ])
        
        return Q

def create_motion_model(vessel_type: int, speed: float) -> MotionModel:
    """
    Create appropriate motion model based on vessel type and speed.
    
    Args:
        vessel_type: AIS vessel type code
        speed: Current speed in knots
        
    Returns:
        MotionModel: Appropriate motion model
    """
    # Vessel type categories (simplified)
    if vessel_type in [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]:  # Cargo vessels
        # Large cargo vessels: low process noise, use CV model
        return ConstantVelocityModel(process_noise_std=0.5)
    
    elif vessel_type in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:  # Fishing vessels
        # Fishing vessels: high process noise, use CV model
        return ConstantVelocityModel(process_noise_std=2.0)
    
    elif vessel_type in [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]:  # Pilot vessels, tugs
        # Small vessels: high process noise, use CV model
        return ConstantVelocityModel(process_noise_std=1.5)
    
    elif vessel_type in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:  # Other vessels
        # Unknown vessels: medium process noise
        return ConstantVelocityModel(process_noise_std=1.0)
    
    else:
        # Default: medium process noise
        return ConstantVelocityModel(process_noise_std=1.0)

def predict_ais_position(ais_record: dict, dt_seconds: float, motion_model: Optional[MotionModel] = None) -> Tuple[float, float, float]:
    """
    Predict AIS vessel position after time gap.
    
    Args:
        ais_record: AIS record with LAT, LON, SOG, COG, Heading
        dt_seconds: Time gap in seconds
        motion_model: Motion model to use (optional)
        
    Returns:
        Tuple of (predicted_lat, predicted_lon, uncertainty_radius_m)
    """
    try:
        # Extract AIS data
        lat = ais_record['LAT']
        lon = ais_record['LON']
        sog = ais_record['SOG']  # Speed over ground (knots)
        cog = ais_record['COG']  # Course over ground (degrees)
        heading = ais_record.get('Heading', cog)  # Use COG as fallback for heading
        
        # Convert speed from knots to m/s
        speed_ms = sog * 0.514444  # knots to m/s
        
        # Convert heading to radians
        heading_rad = np.radians(heading)
        
        # Distance traveled over dt on great-circle (spherical earth approximation)
        distance_m = speed_ms * float(dt_seconds)  # can be negative for backward propagation

        R = 6371000.0  # Earth radius (m)
        
        # Initial lat/lon in radians
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        brg = heading_rad
        
        # Destination point given distance and bearing (allow negative distance for backward)
        if abs(distance_m) < 1.0:  # Less than 1 meter movement
            pred_lat = lat
            pred_lon = lon
        else:
            ang_dist = abs(distance_m) / R
            sin_lat1 = np.sin(lat1)
            cos_lat1 = np.cos(lat1)
            sin_ad = np.sin(ang_dist)
            cos_ad = np.cos(ang_dist)
            sin_brg = np.sin(brg)
            cos_brg = np.cos(brg)
            
            sin_lat2 = sin_lat1 * cos_ad + cos_lat1 * sin_ad * cos_brg
            lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
            y_term = sin_brg * sin_ad * cos_lat1
            x_term = cos_ad - sin_lat1 * sin_lat2
            lon2 = lon1 + np.arctan2(y_term, x_term)
            
            # Normalize lon to [-180, 180]
            lon2 = (lon2 + np.pi) % (2 * np.pi) - np.pi
            
            pred_lat = np.degrees(lat2)
            pred_lon = np.degrees(lon2)
        
        # Calculate uncertainty using linear drift model (physically plausible)
        base_uncertainty = 300.0  # base geolocation error in meters
        
        # Drift per minute by vessel type (m/min)
        drift_per_min_by_type = {
            'cargo': 10.0,    # VesselType 70-79
            'fishing': 30.0,  # VesselType 30-39  
            'small': 20.0,    # VesselType 50-59
            'other': 15.0     # default
        }
        
        tmin = abs(dt_seconds) / 60.0
        vessel_type = int(ais_record.get('VesselType', 90))
        
        if 70 <= vessel_type <= 79:
            drift = drift_per_min_by_type['cargo'] * tmin
        elif 30 <= vessel_type <= 39:
            drift = drift_per_min_by_type['fishing'] * tmin
        elif 50 <= vessel_type <= 59:
            drift = drift_per_min_by_type['small'] * tmin
        else:
            drift = drift_per_min_by_type['other'] * tmin
        
        # Cap uncertainty at 5km to prevent "galactic" gates
        uncertainty_radius = base_uncertainty + min(drift, 5000.0)
        
        return float(pred_lat), float(pred_lon), uncertainty_radius
        
    except Exception as e:
        logger.error(f"❌ Error predicting AIS position: {e}")
        # Return original position with high uncertainty
        return ais_record['LAT'], ais_record['LON'], 10000.0  # 10km uncertainty

if __name__ == "__main__":
    # Test motion models
    logging.basicConfig(level=logging.INFO)
    
    # Test CV model
    cv_model = ConstantVelocityModel(process_noise_std=1.0)
    state = np.array([0, 0, 10, 5])  # x, y, vx, vy
    dt = 3600  # 1 hour
    
    predicted_state, Q = cv_model.predict(state, dt)
    print(f"CV Model - Initial: {state}")
    print(f"CV Model - Predicted: {predicted_state}")
    print(f"CV Model - Process noise: {Q}")
    
    # Test AIS prediction
    ais_record = {
        'LAT': 37.0,
        'LON': -76.0,
        'SOG': 10.0,  # 10 knots
        'COG': 45.0,  # 45 degrees
        'Heading': 45.0,
        'VesselType': 70  # Cargo vessel
    }
    
    pred_lat, pred_lon, uncertainty = predict_ais_position(ais_record, dt)
    print(f"\nAIS Prediction:")
    print(f"  Original: ({ais_record['LAT']:.6f}, {ais_record['LON']:.6f})")
    print(f"  Predicted: ({pred_lat:.6f}, {pred_lon:.6f})")
    print(f"  Uncertainty: {uncertainty:.1f} meters")
