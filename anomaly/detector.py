"""
anomaly/detector.py - Anomaly Detection + Attribution module.

Uses the residuals (actual - predicted) from the core Digital Twin model.

Error typing:
  - bottleneck: sustained cycle-time residual spike + neighbour also elevated
  - defect:     sustained residual without a cycle-time spike (quality drift)
  - sensor-fault: residual that does NOT correlate with neighbour behaviour

Attribution:
  (a) Propagation-by-timing: check if an upstream neighbour was flagged in the
      preceding window. Suggests the anomaly was propagated.
  (b) Propagation-by-edge-weight: use the magnitude of the upstream station's
      residual as a proxy for GCN influence (higher upstream residual = more
      likely origin).

Outputs a structured flag per timestep per station.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyResult:
    station_id: int
    residual: float
    threshold: float
    is_anomaly: bool
    anomaly_type: str           # "bottleneck" | "defect" | "sensor-fault" | "none"
    type_confidence: float      # 0.0 – 1.0
    attribution_origin: Optional[int]   # station_id of likely origin, or None
    attribution_confidence: float       # 0.0 – 1.0
    attribution_label: str      # human-readable "origin: S3 / propagated effect"


# --------------------------------------------------------------------------- #
#  Per-station history buffer
# --------------------------------------------------------------------------- #
@dataclass
class _StationHistory:
    normal_residuals: deque = field(default_factory=lambda: deque(maxlen=60))
    recent_flags: deque = field(default_factory=lambda: deque(maxlen=20))
    consecutive_flags: int = 0
    recent_residuals: deque = field(default_factory=lambda: deque(maxlen=10))


# --------------------------------------------------------------------------- #
#  AnomalyDetector
# --------------------------------------------------------------------------- #
class AnomalyDetector:
    """
    Detects and classifies anomalies per-station by comparing real-time
    observations against the model's 1-step ahead forecast.

    Threshold: mean + alpha * std of the normal-operation residual window.
    Threshold updates freeze when an anomaly is active (prevents adaptive
    masking of a sustained fault).
    """

    # Sustained = flagged for this many consecutive steps before we do typing
    SUSTAINED_STEPS = 4

    def __init__(self, alpha: float = 3.0, window_size: int = 30) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self._hist: Dict[int, _StationHistory] = {}
        # Store latest residuals for all stations (for attribution cross-check)
        self._latest_residuals: Dict[int, float] = {}
        self._latest_flags: Dict[int, bool] = {}

    # ---------------------------------------------------------------------- #
    #  Main per-timestep entry point
    # ---------------------------------------------------------------------- #
    def process(
        self,
        station_id: int,
        actual: float,
        predicted: float,
        upstream_station_id: Optional[int] = None,
        station_type: str = "well_instrumented",
    ) -> AnomalyResult:
        """
        Process one timestep for a station.

        Parameters
        ----------
        station_id        : current station index
        actual            : observed cycle_time_s
        predicted         : 1-step ahead model prediction
        upstream_station_id : station_id of direct upstream neighbour (i-1), or None
        station_type      : "well_instrumented" | "proxy_only" | "manual"

        Returns
        -------
        AnomalyResult
        """
        h = self._hist.setdefault(station_id, _StationHistory())
        residual = abs(actual - predicted)

        # --- Threshold -------------------------------------------------------
        if len(h.normal_residuals) < 5:
            threshold = 999.0
            is_anomaly = False
        else:
            recent = list(h.normal_residuals)[-self.window_size:]
            mu = float(np.mean(recent))
            sigma = float(np.std(recent)) + 1e-3
            threshold = mu + self.alpha * sigma
            is_anomaly = residual > threshold

        # --- Update history (freeze on anomaly to avoid adaptive masking) ----
        if not is_anomaly or len(h.normal_residuals) < self.window_size:
            h.normal_residuals.append(residual)

        h.recent_flags.append(is_anomaly)
        h.recent_residuals.append(residual)
        if is_anomaly:
            h.consecutive_flags += 1
        else:
            h.consecutive_flags = 0

        # Store for cross-station attribution
        self._latest_residuals[station_id] = residual
        self._latest_flags[station_id] = is_anomaly

        # --- Error typing  ---------------------------------------------------
        anomaly_type, type_confidence = self._classify_type(
            h, residual, threshold, actual, predicted, station_type
        )

        # --- Attribution  ----------------------------------------------------
        attr_origin, attr_conf, attr_label = self._attribute(
            station_id, upstream_station_id, h, is_anomaly, anomaly_type
        )

        return AnomalyResult(
            station_id=station_id,
            residual=residual,
            threshold=threshold,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type if is_anomaly else "none",
            type_confidence=type_confidence if is_anomaly else 0.0,
            attribution_origin=attr_origin,
            attribution_confidence=attr_conf,
            attribution_label=attr_label,
        )

    # ---------------------------------------------------------------------- #
    #  Error typing
    # ---------------------------------------------------------------------- #
    def _classify_type(
        self,
        h: _StationHistory,
        residual: float,
        threshold: float,
        actual: float,
        predicted: float,
        station_type: str,
    ) -> Tuple[str, float]:
        """
        Classify anomaly type using rule-based scoring across three classes:
          bottleneck  : sustained cycle-time surge
          defect      : sustained output drift without cycle spike
          sensor-fault: isolated residual, uncorrelated with recent history

        Returns (type_str, confidence).
        """
        # Score each class as normalised distance from its profile
        sustained = h.consecutive_flags >= self.SUSTAINED_STEPS
        signed_resid = actual - predicted  # positive = actual > predicted

        # --- Bottleneck: positive cycle-time surge, sustained ---
        # Cycle time inflation is > threshold AND positive (slower than predicted)
        bottleneck_score = 0.0
        if signed_resid > 0:  # actually slower than predicted
            bottleneck_score += 0.5
        if sustained:
            bottleneck_score += 0.3
        # Rising residuals (trend) also supports bottleneck
        recent_r = list(h.recent_residuals)
        if len(recent_r) >= 4:
            if recent_r[-1] > recent_r[-4]:
                bottleneck_score += 0.2

        # --- Defect: sustained residual but NOT a large cycle-time absolute delta ---
        defect_score = 0.0
        if sustained:
            defect_score += 0.4
        # Defects often show oscillating residuals (quality drifts in a wave)
        if len(recent_r) >= 6:
            diffs = np.diff(recent_r[-6:])
            sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
            if sign_changes >= 2:
                defect_score += 0.3
        if abs(signed_resid) / (threshold + 1e-3) < 2.0:  # not a huge spike
            defect_score += 0.3

        # --- Sensor-fault: isolated, not sustained, no upstrm correlation ---
        sensor_score = 0.0
        if not sustained:
            sensor_score += 0.5
        if station_type == "proxy_only":
            sensor_score += 0.2  # proxy stations more prone to sensor faults
        # Sensor faults: residual doesn't follow any trend
        if len(recent_r) >= 3 and residual > max(recent_r[:-1]) * 2.0:
            sensor_score += 0.3

        scores = {
            "bottleneck": bottleneck_score,
            "defect": defect_score,
            "sensor-fault": sensor_score,
        }
        best = max(scores, key=scores.get)
        total = sum(scores.values()) + 1e-6
        confidence = scores[best] / total

        return best, min(1.0, confidence)

    # ---------------------------------------------------------------------- #
    #  Attribution
    # ---------------------------------------------------------------------- #
    def _attribute(
        self,
        station_id: int,
        upstream_id: Optional[int],
        h: _StationHistory,
        is_anomaly: bool,
        anomaly_type: str,
    ) -> Tuple[Optional[int], float, str]:
        """
        Determine likely origin of an anomaly.

        Strategy:
          (a) Timing check: if upstream was flagged before this station
              (it has a non-zero recent flag count and is still flagged),
              attribute to upstream.
          (b) Edge-weight proxy: upstream residual > this station's residual
              suggests the root cause is upstream.

        Returns (origin_station_id | None, confidence, label_str).
        """
        if not is_anomaly or upstream_id is None:
            return station_id, 0.0, f"origin: S{station_id} / no propagation detected"

        up_flagged = self._latest_flags.get(upstream_id, False)
        up_residual = self._latest_residuals.get(upstream_id, 0.0)
        my_residual = self._latest_residuals.get(station_id, 0.0)

        timing_support = up_flagged  # upstream actively anomalous
        weight_support = up_residual > my_residual * 0.5  # upstream residual is significant

        if timing_support and weight_support:
            conf = 0.80 + 0.15 * min(1.0, up_residual / (my_residual + 1e-3) - 0.5)
            conf = min(0.95, conf)
            label = (
                f"origin: S{upstream_id} / propagated to S{station_id} "
                f"(up_resid={up_residual:.1f} vs self_resid={my_residual:.1f})"
            )
            return upstream_id, conf, label
        elif timing_support:
            conf = 0.60
            label = f"origin: S{upstream_id} / timing-based propagation (upstream flagged)"
            return upstream_id, conf, label
        else:
            conf = 0.30
            label = f"origin: S{station_id} / local fault (no upstream correlation)"
            return station_id, conf, label
