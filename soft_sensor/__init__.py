"""
soft_sensor/__init__.py - Public API for the soft_sensor package.
"""

from soft_sensor.model import (
    SoftSensorModel,
    apply_soft_sensor,
    build_feature_matrix,
    PROXY_COLS,
    TARGET_COLS,
)

__all__ = [
    "SoftSensorModel",
    "apply_soft_sensor",
    "build_feature_matrix",
    "PROXY_COLS",
    "TARGET_COLS",
]
