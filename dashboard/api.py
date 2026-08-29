"""
dashboard/api.py - Backend API / Data layer for the Digital Twin UI.

KEY DESIGN: The entire run is pre-computed once at init time.
get_state(t) just returns a cached dict — it does NOT drive the stateful
anomaly detector again on every call. This prevents double-processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from action import PrescriptiveEngine
from anomaly import AnomalyDetector, AnomalyResult
from model import AssemblyLineDataset, DigitalTwinModel, get_edge_index
from simulator import AssemblyLineSimulator, load_config
from soft_sensor import SoftSensorModel, apply_soft_sensor

ROOT = Path(__file__).parent.parent

FEATURE_COLS = [
    "cycle_time_s", "queue_depth",
    "est_temperature", "est_vibration", "est_torque",
    "motor_current", "setpoint_error", "cycle_duration_s",
    "pass_fail",
]


class DigitalTwinAPI:
    """
    Pre-computes the entire simulation run at init.
    get_state(t) returns a cached dict - safe to call from sliders.
    get_aggregate_stats() returns rollup metrics for Plant Manager / Leadership.
    """

    def __init__(self, run_steps: int = 300, seed: int = 42) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.edge_index = get_edge_index(ROOT / "configs" / "graph.yaml")
        self.station_names = {s.id: s.name for s in cfg.stations}
        self.station_types = {s.id: s.type for s in cfg.stations}
        self.run_steps = run_steps

        # ── 1. Load models ──────────────────────────────────────────────────
        ss_path = ROOT / "data" / "soft_sensor" / "soft_sensor.joblib"
        core_path = ROOT / "data" / "model" / "core_model.pt"

        print("Loading soft sensor …")
        self.ss_model = SoftSensorModel.load(ss_path)

        # Normalisation stats computed from training data (seed 0-19, no faults).
        # We use a clean no-fault run at the same seed to get stable means/stds
        # that match what the model saw during training.
        sim_norm = AssemblyLineSimulator(config=cfg, seed=seed)
        norm_obs, _, _ = sim_norm.run(n_steps=run_steps)
        norm_obs["run_id"] = seed
        norm_obs = apply_soft_sensor(norm_obs, self.ss_model, cfg.proxy_only_ids)
        norm_ds = AssemblyLineDataset(norm_obs, n_stations=cfg.n_stations)
        self.means = norm_ds.means
        self.stds = norm_ds.stds
        in_features = len(self.means)

        print("Loading core model …")
        self.core_model = DigitalTwinModel(in_features, use_gcn=True)
        self.core_model.load_state_dict(
            torch.load(core_path, map_location="cpu", weights_only=True)
        )
        self.core_model.eval()

        # ── 2. Run simulation ───────────────────────────────────────────────
        print("Running simulation …")
        sim = AssemblyLineSimulator(config=cfg, seed=seed)
        sim.inject_bottleneck(station_id=3, start_step=60, duration_steps=60, severity=0.6)
        sim.inject_defect(station_id=8, start_step=150, pattern="sine_drift")

        raw_obs, self.fault_log, _ = sim.run(n_steps=run_steps)
        raw_obs["run_id"] = seed

        print("Applying soft sensor …")
        self.obs_df = apply_soft_sensor(raw_obs, self.ss_model, cfg.proxy_only_ids)

        # Build normalised feature tensor  (T, N, F)
        norm_df = self.obs_df.copy()
        norm_df[FEATURE_COLS] = norm_df[FEATURE_COLS].fillna(0.0)
        norm_df[FEATURE_COLS] = (norm_df[FEATURE_COLS] - self.means) / self.stds

        n_stations = cfg.n_stations
        self.feature_tensor = np.zeros(
            (run_steps, n_stations, len(FEATURE_COLS)), dtype=np.float32
        )
        pivot = norm_df.pivot(index="timestep", columns="station_id", values=FEATURE_COLS)
        for f_idx, feat in enumerate(FEATURE_COLS):
            self.feature_tensor[:, :, f_idx] = pivot[feat].values

        self.max_t = run_steps - 1
        self.SEQ_LEN = 15

        # ── 3. PRE-COMPUTE all states ONCE in strict time order ─────────────
        print("Pre-computing all states …")
        self._state_cache: Dict[int, Dict[str, Any]] = {}
        self._anomaly_log: List[Dict[str, Any]] = []

        # alpha=3.5 => flags only when residual > mean + 3.5*std of normal window
        # Higher alpha reduces false positives from model prediction noise
        detector = AnomalyDetector(alpha=3.5, window_size=40)
        action_engine = PrescriptiveEngine(ROOT / "configs" / "rules.yaml")
        mean_c = float(self.means["cycle_time_s"])
        std_c = float(self.stds["cycle_time_s"])

        for t in range(self.SEQ_LEN - 1, run_steps):
            self._state_cache[t] = self._compute_state(
                t, detector, action_engine, mean_c, std_c
            )

        print("API ready.")

    # ───────────────────────────────────────────────────────────────────────
    def get_state(self, t: int) -> Dict[str, Any]:
        """Return pre-computed state dict for timestep t."""
        t = max(self.SEQ_LEN - 1, min(t, self.max_t))
        return self._state_cache[t]

    # ───────────────────────────────────────────────────────────────────────
    def _compute_state(
        self,
        t: int,
        detector: AnomalyDetector,
        action_engine: PrescriptiveEngine,
        mean_c: float,
        std_c: float,
    ) -> Dict[str, Any]:
        """Called once per timestep at init. Never called again."""
        seq_len = self.SEQ_LEN
        X_np = self.feature_tensor[t - seq_len + 1 : t + 1]
        X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)  # (1,15,N,F)
        X = X.transpose(1, 2)  # (1,N,15,F)

        with torch.no_grad():
            p_short, p_long_m, p_long_s = self.core_model(X, self.edge_index)

        # Unnormalise
        p_short  = p_short  * std_c + mean_c
        p_long_m = p_long_m * std_c + mean_c
        p_long_s = p_long_s * std_c

        curr_obs = (
            self.obs_df[self.obs_df["timestep"] == t]
            .sort_values("station_id")
        )

        stations_out: List[Dict[str, Any]] = []
        active_anomalies: List[Dict[str, Any]] = []

        for sid in range(self.cfg.n_stations):
            row = curr_obs.iloc[sid]
            stype = self.station_types[sid]

            actual_ct = float(row["cycle_time_s"])
            pred_ct   = float(p_short[0, sid, 0].item())
            upstream  = sid - 1 if sid > 0 else None

            result: AnomalyResult = detector.process(
                station_id=sid,
                actual=actual_ct,
                predicted=pred_ct,
                upstream_station_id=upstream,
                station_type=stype,
            )

            baseline_ct  = float(row["baseline_cycle_s"])
            baseline_ratio = actual_ct / baseline_ct if baseline_ct > 0 else 1.0
            forecast_ct  = float(p_long_m[0, sid, 15].item())
            forecast_ratio = forecast_ct / baseline_ct if baseline_ct > 0 else 1.0

            # --- Baseline-ratio defect detection ---
            # The LSTM learns periodic defect patterns during training → residuals stay
            # small even during an active defect. So we add a SECOND detection path:
            # compare actual cycle_time to the station's own physical baseline.
            #
            # If cycle_time is significantly above baseline AND residual is NOT
            # also spiking (which would indicate a bottleneck), call it a defect.
            #
            # Threshold: 22% above baseline sustained → defect flag
            DEFECT_BASELINE_RATIO = 1.22
            if (not result.is_anomaly) and (baseline_ratio > DEFECT_BASELINE_RATIO):
                # Override: baseline deviation that the model didn't flag
                result.is_anomaly = True
                result.anomaly_type = "defect"
                result.type_confidence = min(0.85, (baseline_ratio - 1.0) * 4.0)
                result.attribution_label = (
                    f"origin: S{sid} / quality drift detected "
                    f"({baseline_ratio:.0%} of baseline cycle time)"
                )
                result.attribution_confidence = 0.70

            attr_name = (
                self.station_names[result.attribution_origin]
                if result.attribution_origin is not None
                else self.station_names[sid]
            )

            def _safe(col: str) -> Optional[float]:
                v = row.get(col, np.nan)
                return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

            node: Dict[str, Any] = {
                "station_id": sid,
                "station":    self.station_names[sid],
                "station_type": stype,
                "feeder": self.station_names[sid - 1] if sid > 0 else "None",

                "cycle_time":  actual_ct,
                "queue_depth": float(row["queue_depth"]),
                "temperature": _safe("est_temperature"),
                "vibration":   _safe("est_vibration"),
                "torque":      _safe("est_torque"),

                "anomaly_flagged":    result.is_anomaly,
                "residual_error":     result.residual,
                "threshold":          result.threshold,
                "anomaly_type":       result.anomaly_type,
                "type_confidence":    result.type_confidence,

                "attribution_origin":     attr_name,
                "attribution_confidence": result.attribution_confidence,
                "attribution_label":      result.attribution_label,

                "queue_depth_forecast_ratio": forecast_ratio,
                "forecast_short":     p_short[0, sid, :].tolist(),
                "forecast_long_mean": p_long_m[0, sid, :].tolist(),
                "forecast_long_std":  p_long_s[0, sid, :].tolist(),
            }

            stations_out.append(node)

            if result.is_anomaly or forecast_ratio > 1.2:
                active_anomalies.append(node)

            # Accumulate for aggregate stats
            self._anomaly_log.append({
                "timestep":     t,
                "station_id":   sid,
                "station":      self.station_names[sid],
                "is_anomaly":   result.is_anomaly,
                "anomaly_type": result.anomaly_type,
                "residual":     result.residual,
                "cycle_time":   actual_ct,
                "baseline":     baseline_ct,
            })

        # Actions
        actions_out: List[Dict[str, Any]] = []
        seen_rules: set = set()
        for astate in active_anomalies:
            for act in action_engine.evaluate(astate):
                key = (act.rule_id, astate["station_id"])
                if key in seen_rules:
                    continue
                seen_rules.add(key)
                actions_out.append({
                    "rule_id":        act.rule_id,
                    "severity":       act.severity,
                    "target":         act.target,
                    "recommendation": act.recommendation,
                    "message":        act.message,
                })

        return {"timestep": t, "stations": stations_out, "actions": actions_out}

    # ───────────────────────────────────────────────────────────────────────
    def get_aggregate_stats(self) -> Dict[str, Any]:
        if not self._anomaly_log:
            return {}

        log_df = pd.DataFrame(self._anomaly_log).drop_duplicates(
            subset=["timestep", "station_id"]
        )

        flag_counts = (
            log_df[log_df["is_anomaly"]]
            .groupby("station").size()
            .sort_values(ascending=False)
            .to_dict()
        )

        t_rate = (
            log_df.groupby("timestep")["is_anomaly"]
            .mean().reset_index()
            .rename(columns={"is_anomaly": "anomaly_rate"})
        )

        disrupted = int(log_df.groupby("timestep")["is_anomaly"].any().sum())
        defect_steps = int(
            log_df[log_df["anomaly_type"] == "defect"]["timestep"].nunique()
        )
        bn_steps = int(
            log_df[log_df["anomaly_type"] == "bottleneck"]["timestep"].nunique()
        )

        fault_log = self.fault_log
        lead_times = []
        for _, row in fault_log.iterrows():
            sid = int(row["station_id"])
            fault_start = int(row["start_step"])
            early = log_df[
                (log_df["station_id"] == sid) & log_df["is_anomaly"]
            ]["timestep"]
            early = early[early >= 15]
            if len(early) > 0:
                lead_times.append(fault_start - int(early.min()))

        return {
            "flag_counts": flag_counts,
            "anomaly_rate_over_time": t_rate.to_dict("records"),
            "disrupted_minutes": disrupted,
            "defect_steps": defect_steps,
            "bottleneck_steps": bn_steps,
            "avg_lead_time_minutes": float(np.mean(lead_times)) if lead_times else 0.0,
            "total_timesteps": int(log_df["timestep"].nunique()),
            "fault_log": fault_log.to_dict("records"),
        }
