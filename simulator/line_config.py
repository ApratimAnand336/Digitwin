"""
simulator/line_config.py - Load and parse the line YAML config into dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "line_config.yaml"


@dataclass
class StationConfig:
    id: int
    name: str
    type: str  # "well_instrumented" | "proxy_only" | "manual"
    baseline_cycle_s: float


@dataclass
class LineConfig:
    name: str
    n_stations: int
    timestep_minutes: int
    default_run_hours: int
    stations: List[StationConfig]
    # Sensor column names
    true_signal_names: List[str]   # e.g. ["temperature", "vibration", "torque"]
    proxy_signal_names: List[str]  # e.g. ["motor_current", "setpoint_error", "cycle_duration_s"]
    # Noise fractions (fraction of baseline value)
    noise_cycle_sigma_frac: float
    noise_sensor_sigma_frac: float
    noise_proxy_sigma_frac: float
    # Queue ripple
    queue_ema_alpha: float

    @property
    def well_instrumented_ids(self) -> List[int]:
        return [s.id for s in self.stations if s.type == "well_instrumented"]

    @property
    def proxy_only_ids(self) -> List[int]:
        return [s.id for s in self.stations if s.type == "proxy_only"]

    @property
    def manual_ids(self) -> List[int]:
        return [s.id for s in self.stations if s.type == "manual"]


def load_config(path: Path | str | None = None) -> LineConfig:
    """Load line configuration from a YAML file."""
    if path is None:
        path = _DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    stations = [
        StationConfig(
            id=s["id"],
            name=s["name"],
            type=s["type"],
            baseline_cycle_s=float(s["baseline_cycle_s"]),
        )
        for s in raw["stations"]
    ]
    return LineConfig(
        name=raw["line"]["name"],
        n_stations=raw["line"]["n_stations"],
        timestep_minutes=raw["line"]["timestep_minutes"],
        default_run_hours=raw["line"]["default_run_hours"],
        stations=stations,
        true_signal_names=raw["sensors"]["true_signals"],
        proxy_signal_names=raw["sensors"]["proxy_signals"],
        noise_cycle_sigma_frac=raw["noise"]["cycle_time_sigma_frac"],
        noise_sensor_sigma_frac=raw["noise"]["sensor_sigma_frac"],
        noise_proxy_sigma_frac=raw["noise"]["proxy_sigma_frac"],
        queue_ema_alpha=raw["queue"]["ema_alpha"],
    )
