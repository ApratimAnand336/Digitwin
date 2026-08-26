"""
dashboard/api.py - Backend API / Data layer for the Digital Twin UI.

Wraps the Simulator, Soft Sensor, Core Model, Anomaly Detector, and Action Engine
into a single stateful interface that the Streamlit dashboard can query.
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
from soft_sensor import SoftSensorModel, apply_soft_sensor, TARGET_COLS

ROOT = Path(__file__).parent.parent


class DigitalTwinAPI:
    """
    Provides a unified API for the dashboard to step through the simulation
    and retrieve integrated insights (predictions, anomalies, actions).
    """

    def __init__(self, run_steps: int = 300, seed: int = 42) -> None:
        self.cfg = load_config()
        self.edge_index = get_edge_index(ROOT / "configs" / "graph.yaml")

        # 1. Load Models
        ss_path = ROOT / "data" / "soft_sensor" / "soft_sensor.joblib"
        core_path = ROOT / "data" / "model" / "core_model.pt"

        self.ss_model = SoftSensorModel.load(ss_path)

        # 2. Recompute normalisation stats from a dummy run (must match training)
        sim_dummy = AssemblyLineSimulator(config=self.cfg, seed=0)
        dummy_obs, _, _ = sim_dummy.run(n_steps=100)
        dummy_obs["run_id"] = 0
        dummy_obs = apply_soft_sensor(dummy_obs, self.ss_model, self.cfg.proxy_only_ids)
        dummy_ds = AssemblyLineDataset(dummy_obs, n_stations=self.cfg.n_stations)

        self.means = dummy_ds.means
        self.stds = dummy_ds.stds
        in_features = len(self.means)

        self.core_model = DigitalTwinModel(in_features, use_gcn=True)
        self.core_model.load_state_dict(
            torch.load(core_path, map_location="cpu", weights_only=True)
        )
        self.core_model.eval()

        # 3. Initialise logic engines
        self.detector = AnomalyDetector(alpha=2.5, window_size=30)
        self.action_engine = PrescriptiveEngine(ROOT / "configs" / "rules.yaml")

        # 4. Generate the main simulation run (pre-computed for the prototype)
        self.sim = AssemblyLineSimulator(config=self.cfg, seed=seed)

        # Inject demonstration faults
        # Bottleneck at S3 (Underbody Framing) from t=60 to 120
        self.sim.inject_bottleneck(station_id=3, start_step=60, duration_steps=60, severity=0.6)
        # Defect at S8 (Brake Line Routing) from t=150 onwards
        self.sim.inject_defect(station_id=8, start_step=150, pattern="sine_drift")

        print("Generating simulation run...")
        raw_obs, self.fault_log, _ = self.sim.run(n_steps=run_steps)
        raw_obs["run_id"] = seed

        print("Applying soft sensor...")
        self.obs_df = apply_soft_sensor(raw_obs, self.ss_model, self.cfg.proxy_only_ids)

        # Pre-format the features array for fast slicing
        feature_cols = [
            "cycle_time_s", "queue_depth",
            "est_temperature", "est_vibration", "est_torque",
            "motor_current", "setpoint_error", "cycle_duration_s",
            "pass_fail"
        ]

        # Fill NaNs and normalise
        norm_df = self.obs_df.copy()
        norm_df[feature_cols] = norm_df[feature_cols].fillna(0.0)
        norm_df[feature_cols] = (norm_df[feature_cols] - self.means) / self.stds

        # Pivot to (Timesteps, Stations, Features)
        n_stations = self.cfg.n_stations
        self.feature_tensor = np.zeros(
            (run_steps, n_stations, len(feature_cols)), dtype=np.float32
        )

        pivot = norm_df.pivot(index="timestep", columns="station_id", values=feature_cols)
        for f_idx, feat in enumerate(feature_cols):
            self.feature_tensor[:, :, f_idx] = pivot[feat].values

        self.max_t = run_steps - 1

        # Accumulate anomaly results per timestep for aggregate views
        self._anomaly_log: List[Dict[str, Any]] = []

        self.station_names = {s.id: s.name for s in self.cfg.stations}
        self.station_types = {s.id: s.type for s in self.cfg.stations}

    def get_state(self, t: int) -> Dict[str, Any]:
        """
        Retrieves the full system state at timestep t, formatted for the UI.
        Runs inference through the core model and prescriptive engine.
        """
        seq_len = 15
        if t < seq_len - 1:
            t = seq_len - 1
        if t > self.max_t:
            t = self.max_t

        # 1. Slice current window
        X_np = self.feature_tensor[t - seq_len + 1 : t + 1]
        X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)  # (1,15,N,F)
        X = X.transpose(1, 2)  # (1,N,15,F)

        # 2. Forward pass
        with torch.no_grad():
            p_short, p_long_m, p_long_s = self.core_model(X, self.edge_index)

        # Unnormalise cycle_time_s predictions
        mean_c = float(self.means["cycle_time_s"])
        std_c = float(self.stds["cycle_time_s"])

        p_short = p_short * std_c + mean_c
        p_long_m = p_long_m * std_c + mean_c
        p_long_s = p_long_s * std_c

        # Get actual current values (at time t)
        curr_obs = self.obs_df[self.obs_df["timestep"] == t].sort_values("station_id")

        state_response: Dict[str, Any] = {"timestep": t, "stations": [], "actions": []}
        active_anomalies: List[Dict[str, Any]] = []

        # 3. Process each station
        for sid in range(self.cfg.n_stations):
            row = curr_obs.iloc[sid]
            stype = self.station_types[sid]

            actual_cycle_time = float(row["cycle_time_s"])
            pred_1step_ahead = float(p_short[0, sid, 0].item())

            upstream_id = sid - 1 if sid > 0 else None

            # Anomaly Detection with real classification + attribution
            result: AnomalyResult = self.detector.process(
                station_id=sid,
                actual=actual_cycle_time,
                predicted=pred_1step_ahead,
                upstream_station_id=upstream_id,
                station_type=stype,
            )

            # Long-horizon queue forecast ratio
            baseline_ct = float(row["baseline_cycle_s"])
            forecast_ct = float(p_long_m[0, sid, 15].item())
            forecast_ratio = forecast_ct / baseline_ct if baseline_ct > 0 else 1.0

            # Attribution label uses real station names
            attr_origin_name = (
                self.station_names[result.attribution_origin]
                if result.attribution_origin is not None
                else self.station_names[sid]
            )

            node_state: Dict[str, Any] = {
                "station_id": sid,
                "station": self.station_names[sid],
                "station_type": stype,
                "feeder": self.station_names[sid - 1] if sid > 0 else "None",

                "cycle_time": actual_cycle_time,
                "queue_depth": float(row["queue_depth"]),
                "temperature": (
                    float(row["est_temperature"])
                    if not np.isnan(row["est_temperature"])
                    else None
                ),
                "vibration": (
                    float(row["est_vibration"])
                    if not np.isnan(row["est_vibration"])
                    else None
                ),
                "torque": (
                    float(row["est_torque"])
                    if not np.isnan(row["est_torque"])
                    else None
                ),

                # --- Anomaly outputs (now from real detector) ---
                "anomaly_flagged": result.is_anomaly,
                "residual_error": result.residual,
                "threshold": result.threshold,
                "anomaly_type": result.anomaly_type,
                "type_confidence": result.type_confidence,

                # Attribution
                "attribution_origin": attr_origin_name,
                "attribution_confidence": result.attribution_confidence,
                "attribution_label": result.attribution_label,

                "queue_depth_forecast_ratio": forecast_ratio,

                # Full forecasts for UI plotting
                "forecast_short": p_short[0, sid, :].tolist(),
                "forecast_long_mean": p_long_m[0, sid, :].tolist(),
                "forecast_long_std": p_long_s[0, sid, :].tolist(),
            }

            state_response["stations"].append(node_state)

            if result.is_anomaly or forecast_ratio > 1.2:
                active_anomalies.append(node_state)

            # Log for aggregate/Plant Manager view
            self._anomaly_log.append({
                "timestep": t,
                "station_id": sid,
                "station": self.station_names[sid],
                "is_anomaly": result.is_anomaly,
                "anomaly_type": result.anomaly_type,
                "residual": result.residual,
                "cycle_time": actual_cycle_time,
                "baseline": baseline_ct,
            })

        # 4. Generate Prescriptive Actions via rules engine
        for state_dict in active_anomalies:
            actions = self.action_engine.evaluate(state_dict)
            for act in actions:
                state_response["actions"].append({
                    "rule_id": act.rule_id,
                    "severity": act.severity,
                    "target": act.target,
                    "recommendation": act.recommendation,
                    "message": act.message,
                })

        return state_response

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Returns aggregate metrics for the Plant Manager and Leadership views.
        Built from the accumulated anomaly log across all get_state() calls.
        """
        if not self._anomaly_log:
            return {}

        log_df = pd.DataFrame(self._anomaly_log).drop_duplicates(
            subset=["timestep", "station_id"]
        )

        # Flag counts per station
        flag_counts = (
            log_df[log_df["is_anomaly"]]
            .groupby("station")
            .size()
            .sort_values(ascending=False)
            .to_dict()
        )

        # Anomaly rate over time (fraction of stations flagged per timestep)
        t_rate = (
            log_df.groupby("timestep")["is_anomaly"]
            .mean()
            .reset_index()
            .rename(columns={"is_anomaly": "anomaly_rate"})
        )

        # Estimated disrupted timesteps (any station flagged)
        disrupted_steps = log_df.groupby("timestep")["is_anomaly"].any().sum()
        disrupted_minutes = int(disrupted_steps)  # 1 step = 1 min

        # Estimated defect exposure: steps where a defect was active
        defect_steps = int(
            log_df[log_df["anomaly_type"] == "defect"]["timestep"].nunique()
        )

        # Bottleneck steps
        bn_steps = int(
            log_df[log_df["anomaly_type"] == "bottleneck"]["timestep"].nunique()
        )

        # Lead time: estimated minutes of early warning
        # (simple heuristic: model flags before cycle-time explodes by comparing
        #  residual-cross time vs fault_log start times in the active run)
        fault_log = self.fault_log
        lead_times = []
        for _, row in fault_log.iterrows():
            sid = int(row["station_id"])
            fault_start = int(row["start_step"])
            # Find first flag at this station after model had enough history
            station_flags = log_df[
                (log_df["station_id"] == sid) & log_df["is_anomaly"]
            ]["timestep"]
            early_flags = station_flags[station_flags >= 15]  # model needs 15 steps warmup
            if len(early_flags) > 0:
                first_flag = int(early_flags.min())
                lead = fault_start - first_flag  # positive = flagged BEFORE fault label start
                lead_times.append(lead)

        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

        return {
            "flag_counts": flag_counts,
            "anomaly_rate_over_time": t_rate.to_dict("records"),
            "disrupted_minutes": disrupted_minutes,
            "defect_steps": defect_steps,
            "bottleneck_steps": bn_steps,
            "avg_lead_time_minutes": avg_lead_time,
            "total_timesteps": int(log_df["timestep"].nunique()),
            "fault_log": fault_log.to_dict("records"),
        }
