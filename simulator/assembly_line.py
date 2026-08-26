"""
simulator/assembly_line.py - Synthetic assembly-line data generator.

Design notes
------------
* One timestep = 1 minute (configurable in line_config.yaml).
* Station order is sequential (0 → 1 → ... → N-1); queue ripple flows forward.
* True sensor values (temperature, vibration, torque) evolve as slow AR(1)
  drift + cycle-time-correlated load effect + Gaussian noise.
* Proxy signals are a nonlinear (tanh/product) function of true sensors + extra
  noise -- NOT a simple linear copy.  This makes Stage 2 (soft sensor) meaningful.
* Queue ripple: upstream cycle-time excess is propagated downstream via EMA.
  A backed-up queue adds wait time to the downstream station's effective cycle.
* Fault injection is scripted and logged separately; fault labels are NEVER
  written into the observations DataFrame that the model sees.

Output
------
run() -> (observations_df, fault_log_df, ground_truth_df)
  observations_df : tidy table one row per (station, timestep).
                    True-sensor columns are NaN for proxy-only and manual stations.
                    Proxy columns are NaN for manual stations.
  fault_log_df    : ground-truth fault labels (station, type, start, end).
                    Must NOT be used as model input.
  ground_truth_df : hidden true-sensor values for ALL stations (proxy-only
                    stations included).  Used ONLY at eval time to measure
                    soft-sensor quality.  Must NOT be used as model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from simulator.line_config import LineConfig, load_config

# --------------------------------------------------------------------------- #
#  Per-station physical baselines (stable across a simulator instance)
# --------------------------------------------------------------------------- #
# These are determined at __init__ time from the RNG so each seed gives a
# consistent but unique set of station characteristics.

# Temperature range
_TEMP_BASE_RANGE = (74.0, 76.0)
# Vibration range
_VIB_BASE_RANGE = (2.4, 2.6)
# Torque range
_TORQUE_BASE_RANGE = (54.0, 56.0)
# Motor current range
_CURRENT_BASE_RANGE = (19.0, 21.0)

# How much upstream wait bleeds into effective cycle time.
# 0.5 = a 20-second queue buildup adds ~10s wait at downstream station.
_WAIT_BLEED_FACTOR = 0.5


# --------------------------------------------------------------------------- #
#  Fault event container
# --------------------------------------------------------------------------- #
@dataclass
class FaultEvent:
    station_id: int
    fault_type: str   # "bottleneck" | "defect"
    start_step: int
    end_step: int     # exclusive
    severity: float = 0.5
    pattern: str = "sine_drift"


# --------------------------------------------------------------------------- #
#  Main simulator
# --------------------------------------------------------------------------- #
class AssemblyLineSimulator:
    """
    Generates synthetic per-station, per-timestep data for the full assembly line.
    """

    def __init__(
        self,
        config: Optional[LineConfig] = None,
        seed: int = 42,
    ) -> None:
        self.cfg = config or load_config()
        self.rng = np.random.default_rng(seed)
        self._fault_events: List[FaultEvent] = []

        n = self.cfg.n_stations
        # Per-station physical baselines -- seeded so they vary station-to-station
        self._temp_base    = self.rng.uniform(*_TEMP_BASE_RANGE, n)
        self._vib_base     = self.rng.uniform(*_VIB_BASE_RANGE, n)
        self._torque_base  = self.rng.uniform(*_TORQUE_BASE_RANGE, n)
        self._current_base = self.rng.uniform(*_CURRENT_BASE_RANGE, n)

    # ---------------------------------------------------------------------- #
    #  Fault injection API
    # ---------------------------------------------------------------------- #
    def inject_bottleneck(
        self,
        station_id: int,
        start_step: int,
        duration_steps: int,
        severity: float = 0.5,
    ) -> None:
        self._fault_events.append(
            FaultEvent(
                station_id=station_id,
                fault_type="bottleneck",
                start_step=start_step,
                end_step=start_step + duration_steps,
                severity=severity,
            )
        )

    def inject_defect(
        self,
        station_id: int,
        start_step: int,
        pattern: str = "sine_drift",
    ) -> None:
        self._fault_events.append(
            FaultEvent(
                station_id=station_id,
                fault_type="defect",
                start_step=start_step,
                end_step=10_000_000,
                severity=0.35,
                pattern=pattern,
            )
        )

    def reset_faults(self) -> None:
        self._fault_events = []

    # ---------------------------------------------------------------------- #
    #  Internal helpers
    # ---------------------------------------------------------------------- #
    def _build_fault_masks(
        self, n_steps: int, n_stations: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bn_active   = np.zeros((n_steps, n_stations), dtype=bool)
        bn_severity = np.zeros((n_steps, n_stations), dtype=float)
        def_active  = np.zeros((n_steps, n_stations), dtype=bool)
        def_start   = np.full(n_stations, -1, dtype=int)

        for ev in self._fault_events:
            s  = ev.station_id
            t0 = ev.start_step
            t1 = min(ev.end_step, n_steps)
            if ev.fault_type == "bottleneck":
                bn_active[t0:t1, s]   = True
                bn_severity[t0:t1, s] = ev.severity
            elif ev.fault_type == "defect":
                def_active[t0:t1, s] = True
                if def_start[s] < 0:
                    def_start[s] = t0

        return bn_active, bn_severity, def_active, def_start

    def _proxy_from_true(
        self,
        temperature: float,
        vibration: float,
        torque: float,
        station_idx: int,
        cycle_time: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        """
        Compute proxy signals from true sensor values.
        """
        cfg = self.cfg
        np_frac = cfg.noise_proxy_sigma_frac

        # Universal mapping so the soft sensor can generalize to unseen proxy-only stations
        
        # motor_current mostly driven by vibration
        motor_current = 20.0 * (1.0 + 0.8 * (vibration - 2.5)) + rng.normal(0.0, np_frac * 10.0)
        motor_current = max(0.5, motor_current)

        # setpoint_error driven by torque (twist) and temperature (thermal expansion)
        setpoint_error = (
            0.08 * np.tanh((torque - 55.0) / 10.0) +
            0.04 * np.tanh((temperature - 75.0) / 10.0) +
            rng.normal(0.0, 0.015)
        )

        # cycle_duration is noisy gate-sensor reading
        cycle_duration_s = cycle_time + rng.normal(0.0, np_frac * 3.0)

        return float(motor_current), float(setpoint_error), float(cycle_duration_s)

    # ---------------------------------------------------------------------- #
    #  Main simulation loop
    # ---------------------------------------------------------------------- #
    def run(
        self,
        n_steps: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the simulator for n_steps timesteps.

        Returns
        -------
        observations_df : pd.DataFrame
            Tidy, one row per (station_id, timestep).  True-sensor columns
            are NaN where the station's coverage mask excludes them.
        fault_log_df : pd.DataFrame
            Ground-truth fault labels.  MUST NOT be given to the model.
        ground_truth_df : pd.DataFrame
            Hidden true-sensor values for ALL stations (including proxy-only).
            Used only at eval time to score the soft sensor.
        """
        cfg = self.cfg
        if n_steps is None:
            n_steps = cfg.default_run_hours * 60 // cfg.timestep_minutes

        n_st   = cfg.n_stations
        alpha  = cfg.queue_ema_alpha
        nc     = cfg.noise_cycle_sigma_frac
        ns     = cfg.noise_sensor_sigma_frac
        rng    = self.rng

        # Precompute fault masks
        bn_active, bn_sev, def_active, def_start = self._build_fault_masks(
            n_steps, n_st
        )

        # ------------------------------------------------------------------ #
        #  Per-station state (carries across timesteps)
        # ------------------------------------------------------------------ #
        temp_state   = self._temp_base.copy()
        vib_state    = self._vib_base.copy()
        torque_state = self._torque_base.copy()
        quality      = np.ones(n_st)          # 1.0 = perfect output quality
        queue        = np.zeros(n_st)         # seconds of upstream backlog
        prev_cycle   = np.array(
            [s.baseline_cycle_s for s in cfg.stations], dtype=float
        )

        obs_rows = []
        gt_rows  = []   # ground truth (all true sensor values, incl. proxy-only)

        for t in range(n_steps):
            cur_cycle = np.zeros(n_st)

            for i, st in enumerate(cfg.stations):
                base_c = st.baseline_cycle_s

                # ---------- Queue ripple from upstream ----------
                if i > 0:
                    upstream_excess = max(0.0, prev_cycle[i - 1] - cfg.stations[i - 1].baseline_cycle_s)
                    queue[i] = (1.0 - alpha) * queue[i] + alpha * upstream_excess
                else:
                    queue[i] = 0.0

                # ---------- Cycle time ----------
                bn_factor  = 1.0 + bn_sev[t, i] if bn_active[t, i] else 1.0
                noise_c    = rng.normal(0.0, nc * base_c)
                processing = base_c * bn_factor + noise_c
                wait_time  = queue[i] * _WAIT_BLEED_FACTOR
                cycle_time = max(processing + wait_time, base_c * 0.4)
                cur_cycle[i] = cycle_time

                load_rel = (cycle_time / base_c) - 1.0  # fractional load vs baseline

                # ---------- True sensor evolution (AR1 + load correlation) ----------
                # Temperature
                temp_state[i] = (
                    self._temp_base[i]
                    + 0.92 * (temp_state[i] - self._temp_base[i])   # mean-revert
                    + load_rel * 14.0                                # load heat
                    + bn_active[t, i] * bn_sev[t, i] * 10.0        # bottleneck extra heat
                )
                temperature = temp_state[i] + rng.normal(0.0, ns * self._temp_base[i])

                # Vibration
                vib_state[i] = (
                    self._vib_base[i]
                    + 0.92 * (vib_state[i] - self._vib_base[i])
                    + load_rel * 0.6
                    + bn_active[t, i] * bn_sev[t, i] * 0.9
                )
                vibration = max(0.05, vib_state[i] + rng.normal(0.0, ns * self._vib_base[i]))

                # Torque
                torque_state[i] = (
                    self._torque_base[i]
                    + 0.92 * (torque_state[i] - self._torque_base[i])
                    + load_rel * 9.0
                    + bn_active[t, i] * bn_sev[t, i] * 6.0
                )
                # Defect: subtle torque oscillation (fastening quality drifts)
                if def_active[t, i]:
                    steps_in = t - def_start[i]
                    torque_state[i] += bn_sev[t, i] * 4.0 * np.sin(2 * np.pi * steps_in / 25)
                torque = max(1.0, torque_state[i] + rng.normal(0.0, ns * self._torque_base[i]))

                # ---------- Output quality ----------
                if def_active[t, i]:
                    steps_in = t - def_start[i]
                    # Ramp-in sinusoidal drift: quality degrades in a repeating wave
                    ramp      = min(1.0, steps_in / 20.0)
                    quality[i] = 1.0 - ramp * 0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * steps_in / 30))
                else:
                    quality[i] = min(1.0, quality[i] + 0.03)  # slow recovery
                out_quality = float(np.clip(quality[i] + rng.normal(0.0, 0.02), 0.0, 1.0))

                # ---------- Proxy signals (nonlinear of true sensors) ----------
                motor_c, setpt_err, cycle_dur = self._proxy_from_true(
                    temperature, vibration, torque, i, cycle_time, rng
                )

                # ---------- Pass/fail (manual stations) ----------
                pass_fail = int(out_quality > 0.5) if st.type == "manual" else None

                # ---------- Apply sensor coverage mask ----------
                wi = st.type == "well_instrumented"
                po = st.type == "proxy_only"
                mn = st.type == "manual"

                nan = float("nan")

                obs_rows.append({
                    "station_id":      i,
                    "timestep":        t,
                    "station_name":    st.name,
                    "station_type":    st.type,
                    "cycle_time_s":    round(cycle_time, 3),
                    "baseline_cycle_s": base_c,
                    "queue_depth":     round(float(queue[i]), 3),
                    # True sensors -- NaN for proxy-only and manual
                    "temperature":     round(temperature, 3) if wi else nan,
                    "vibration":       round(vibration, 4)   if wi else nan,
                    "torque":          round(torque, 3)      if wi else nan,
                    # Proxy signals -- NaN for manual
                    "motor_current":   round(motor_c, 4)     if not mn else nan,
                    "setpoint_error":  round(setpt_err, 5)   if not mn else nan,
                    "cycle_duration_s": round(cycle_dur, 3)  if not mn else nan,
                    # Manual only
                    "pass_fail":       pass_fail,
                })

                # Ground truth row -- full internal state for ALL stations
                gt_rows.append({
                    "station_id":     i,
                    "timestep":       t,
                    "station_type":   st.type,
                    "_gt_temperature": round(temperature, 3),
                    "_gt_vibration":   round(vibration, 4),
                    "_gt_torque":      round(torque, 3),
                    "_gt_quality":     round(out_quality, 4),
                })

            prev_cycle = cur_cycle.copy()

        # ------------------------------------------------------------------ #
        #  Build DataFrames
        # ------------------------------------------------------------------ #
        obs_df = pd.DataFrame(obs_rows)
        gt_df  = pd.DataFrame(gt_rows)

        # Fault log
        fault_rows = [
            {
                "station_id": ev.station_id,
                "fault_type": ev.fault_type,
                "start_step": ev.start_step,
                "end_step":   min(ev.end_step, n_steps),
                "severity":   ev.severity,
            }
            for ev in self._fault_events
        ]
        fault_log = (
            pd.DataFrame(fault_rows)
            if fault_rows
            else pd.DataFrame(
                columns=["station_id", "fault_type", "start_step", "end_step", "severity"]
            )
        )

        return obs_df, fault_log, gt_df

    # ---------------------------------------------------------------------- #
    #  Convenience: save outputs
    # ---------------------------------------------------------------------- #
    def save(
        self,
        obs_df: pd.DataFrame,
        fault_log: pd.DataFrame,
        gt_df: pd.DataFrame,
        out_dir: Path | str,
        run_id: int = 0,
    ) -> None:
        """Save simulation outputs to parquet/csv files."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        obs_df.to_parquet(out / f"run_{run_id:04d}_obs.parquet", index=False)
        fault_log.to_csv(out / f"run_{run_id:04d}_faults.csv", index=False)
        gt_df.to_parquet(out / f"run_{run_id:04d}_gt.parquet", index=False)
