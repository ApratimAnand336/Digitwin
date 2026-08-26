"""
model/dataset.py - Dataset generation and sequence batching.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Features used for the LSTM
FEATURE_COLS = [
    "cycle_time_s", "queue_depth", 
    "est_temperature", "est_vibration", "est_torque",
    "motor_current", "setpoint_error", "cycle_duration_s",
    "pass_fail"
]


class AssemblyLineDataset(Dataset):
    """
    Rolling window dataset over the simulated observations.
    """
    def __init__(
        self,
        obs_df: pd.DataFrame,
        n_stations: int,
        seq_len: int = 15,
        short_horizon: int = 5,
        long_horizon: int = 30,
    ) -> None:
        self.seq_len = seq_len
        self.short_horizon = short_horizon
        self.long_horizon = long_horizon
        self.n_stations = n_stations
        
        # Ensure data is sorted by run_id, timestep, station_id
        df = obs_df.sort_values(["run_id", "timestep", "station_id"]).copy()
        
        # Fill NaNs (manual stations, or missing proxies) with 0 for the network
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
        
        # Normalise features roughly to [-2, 2] range for stability
        self.means = df[FEATURE_COLS].mean()
        self.stds = df[FEATURE_COLS].std().replace(0, 1.0)
        df[FEATURE_COLS] = (df[FEATURE_COLS] - self.means) / self.stds
        
        self.samples = []
        
        # Group by run_id
        for run_id, run_group in df.groupby("run_id"):
            n_steps = run_group["timestep"].nunique()
            # Extract features as (Timesteps, Stations, Features)
            # pivot so rows are timesteps, columns are (station, feature)
            pivot = run_group.pivot(index="timestep", columns="station_id", values=FEATURE_COLS)
            
            # Reorder columns to (Station, Feature) instead of (Feature, Station)
            # We want shape: (Timesteps, Stations, Features)
            data_3d = np.zeros((n_steps, n_stations, len(FEATURE_COLS)), dtype=np.float32)
            for f_idx, feat in enumerate(FEATURE_COLS):
                data_3d[:, :, f_idx] = pivot[feat].values
                
            # cycle_time_s is feature index 0
            # Target is the UNNORMALIZED cycle_time_s
            target_3d = run_group.pivot(index="timestep", columns="station_id", values="cycle_time_s").values
            
            # Create rolling windows
            total_window = seq_len + long_horizon
            for t in range(n_steps - total_window + 1):
                X = data_3d[t : t + seq_len]
                Y_short = target_3d[t + seq_len : t + seq_len + short_horizon]
                Y_long = target_3d[t + seq_len : t + seq_len + long_horizon]
                
                # Reshape targets to (Stations, Horizon)
                Y_short = Y_short.transpose(1, 0)
                Y_long = Y_long.transpose(1, 0)
                
                # Transpose X to (Stations, SeqLen, Features)
                X = X.transpose(1, 0, 2)
                
                self.samples.append((
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(Y_short, dtype=torch.float32),
                    torch.tensor(Y_long, dtype=torch.float32)
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]
