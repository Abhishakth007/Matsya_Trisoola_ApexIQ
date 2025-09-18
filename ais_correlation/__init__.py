"""
AIS Correlation Module

This module implements the production-ready AIS correlation algorithm
for matching SAR vessel detections with AIS data, handling any temporal gap.
"""

from .sar_timestamp_extractor import extract_sar_timestamp
from .motion_models import MotionModel, ConstantVelocityModel
from .correlation_engine import CorrelationEngine

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "extract_sar_timestamp",
    "MotionModel",
    "ConstantVelocityModel",
    "CorrelationEngine"
]
