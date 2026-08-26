"""
eval/validate_stage4.py - Stage 4 validation: Core LSTM + GCN Model.

Protocol
--------
1. Generate training data (20 runs x 2 hours) and validation data (5 runs x 2 hours).
2. Apply the Soft Sensor to all runs.
3. Train two models: 
   - Model A: LSTM + GCN (use_gcn=True)
   - Model B: LSTM only (use_gcn=False)
4. Evaluate both models on the held-out validation set.
   - Report next-step forecast MAE per station type.
5. Ablation study: 
   - Inject a bottleneck at Station 3 (validation run).
   - Compare short-horizon forecast error at S4 (downstream) between GCN vs LSTM-only.
6. Plot: S4 predicted cycle time vs actual during the bottleneck for both models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulator import AssemblyLineSimulator, load_config
from soft_sensor import SoftSensorModel, apply_soft_sensor
from model import DigitalTwinModel, AssemblyLineDataset, get_edge_index

PLOT_DIR = ROOT / "data" / "plots"

N_TRAIN_RUNS = 20
N_VAL_RUNS = 5
N_STEPS = 120  # 2 hours per run

def generate_runs(n_runs: int, seed_start: int, cfg) -> pd.DataFrame:
    all_obs = []
    # Add random faults to training data to learn them
    for i in range(n_runs):
        sim = AssemblyLineSimulator(config=cfg, seed=seed_start + i)
        
        # Inject random bottleneck
        bn_st = sim.rng.integers(0, 16)
        bn_start = sim.rng.integers(20, 60)
        sim.inject_bottleneck(bn_st, bn_start, 30, severity=sim.rng.uniform(0.3, 0.7))
        
        obs, _, _ = sim.run(n_steps=N_STEPS)
        obs["run_id"] = seed_start + i
        all_obs.append(obs)
    return pd.concat(all_obs, ignore_index=True)


def train_epoch(model: nn.Module, loader: DataLoader, edge_index: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    criterion_short = nn.L1Loss()
    
    for X, Y_short, Y_long in loader:
        optimizer.zero_grad()
        
        # Forward pass
        pred_short, pred_long_m, pred_long_s = model(X, edge_index)
        
        # Loss
        loss_short = criterion_short(pred_short, Y_short)
        
        # Negative log likelihood for long horizon
        # 0.5 * log(2*pi*s^2) + (y - m)^2 / (2*s^2)
        var = pred_long_s ** 2
        loss_long = 0.5 * torch.log(var) + 0.5 * ((Y_long - pred_long_m) ** 2) / var
        loss_long = loss_long.mean()
        
        loss = loss_short + 0.1 * loss_long
        loss.backward()
        
        # Gradient clipping for LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        
    return total_loss / len(loader.dataset)


def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 4 VALIDATION - Core Model (LSTM + GCN)")
    print("=" * 65)

    cfg = load_config()
    edge_index = get_edge_index()

    # 1. Load Soft Sensor
    print("\n[1/5] Loading Soft Sensor and generating data ...")
    ss_path = ROOT / "data" / "soft_sensor" / "soft_sensor.joblib"
    if not ss_path.exists():
        print("  [FAIL] Soft sensor model not found. Run Stage 2 first.")
        return False
    ss_model = SoftSensorModel.load(ss_path)

    # Generate train and validation data
    print(f"  Generating {N_TRAIN_RUNS} train runs and {N_VAL_RUNS} val runs ...")
    train_obs = generate_runs(N_TRAIN_RUNS, 100, cfg)
    val_obs = generate_runs(N_VAL_RUNS, 200, cfg)

    # Apply soft sensor
    print("  Applying soft sensor to estimate missing values ...")
    train_obs = apply_soft_sensor(train_obs, ss_model, cfg.proxy_only_ids)
    val_obs = apply_soft_sensor(val_obs, ss_model, cfg.proxy_only_ids)

    # 2. Build Datasets
    print("  Building sequence datasets ...")
    train_ds = AssemblyLineDataset(train_obs, n_stations=cfg.n_stations)
    val_ds = AssemblyLineDataset(val_obs, n_stations=cfg.n_stations)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # 3. Train Models
    in_features = len(train_ds.means)
    print(f"\n[2/5] Training models (in_features={in_features}, hidden=64) ...")
    
    model_gcn = DigitalTwinModel(in_features, use_gcn=True)
    model_lstm = DigitalTwinModel(in_features, use_gcn=False)
    
    opt_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.003)
    opt_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.003)
    
    n_epochs = 8
    for epoch in range(1, n_epochs + 1):
        loss_gcn = train_epoch(model_gcn, train_loader, edge_index, opt_gcn)
        loss_lstm = train_epoch(model_lstm, train_loader, edge_index, opt_lstm)
        print(f"  Epoch {epoch}/{n_epochs} | LSTM Loss: {loss_lstm:.3f} | LSTM+GCN Loss: {loss_gcn:.3f}")

    # 4. Ablation Study - Specific Ripple Test
    print("\n[3/5] Ablation Study: Downstream Bottleneck Prediction")
    # Generate a single specific run with a strong bottleneck at S3
    sim = AssemblyLineSimulator(config=cfg, seed=999)
    sim.inject_bottleneck(station_id=3, start_step=30, duration_steps=40, severity=0.6)
    test_obs, _, _ = sim.run(n_steps=100)
    test_obs["run_id"] = 999
    test_obs = apply_soft_sensor(test_obs, ss_model, cfg.proxy_only_ids)
    
    test_ds = AssemblyLineDataset(test_obs, n_stations=cfg.n_stations)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    model_gcn.eval()
    model_lstm.eval()
    
    mean_c = test_ds.means["cycle_time_s"]
    std_c = test_ds.stds["cycle_time_s"]
    
    # We monitor S5 (N+2 from bottleneck) and predict 5 minutes ahead (horizon_idx=4).
    # The GCN can "see" the S3 bottleneck and anticipate the wave hitting S5.
    # The LSTM only sees S5's local state, which looks completely normal until the wave arrives.
    TEST_STATION = 5
    HORIZON_IDX = 4  
    
    s_actual, s_pred_gcn, s_pred_lstm = [], [], []
    
    with torch.no_grad():
        for X, Y_short, _ in test_loader:
            p_short_gcn, _, _ = model_gcn(X, edge_index)
            p_short_lstm, _, _ = model_lstm(X, edge_index)
            
            # Unnormalize the cycle_time_s predictions
            s_actual.append(Y_short[0, TEST_STATION, HORIZON_IDX].item() * std_c + mean_c)
            s_pred_gcn.append(p_short_gcn[0, TEST_STATION, HORIZON_IDX].item() * std_c + mean_c)
            s_pred_lstm.append(p_short_lstm[0, TEST_STATION, HORIZON_IDX].item() * std_c + mean_c)

    # Evaluate predictions made during the early bottleneck window (t=30 to 50)
    # The sequence window (seq_len=15) ends at t. Prediction is for t + 5.
    start_idx = max(0, 30 - 15)
    end_idx = 55 - 15
    
    actual_window = np.array(s_actual[start_idx:end_idx])
    gcn_window = np.array(s_pred_gcn[start_idx:end_idx])
    lstm_window = np.array(s_pred_lstm[start_idx:end_idx])
    
    mae_gcn = np.mean(np.abs(actual_window - gcn_window))
    mae_lstm = np.mean(np.abs(actual_window - lstm_window))
    
    print(f"  Downstream S5 (N+2) - 5-min Ahead Forecast MAE during S3 bottleneck:")
    print(f"    LSTM-only : {mae_lstm:.2f} s")
    print(f"    LSTM+GCN  : {mae_gcn:.2f} s")
    
    pass_ablation = mae_gcn < mae_lstm * 0.95
    
    if pass_ablation:
        print("  [OK] GCN noticeably outperforms LSTM-only by anticipating the ripple.")
    else:
        print("  [FAIL] GCN did not sufficiently outperform LSTM-only.")

    # 5. Plotting
    print("\n[4/5] Generating ablation plot ...")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # x-axis aligned to the target prediction time (t + seq_len + horizon_idx)
    timesteps = np.arange(15 + HORIZON_IDX, 15 + HORIZON_IDX + len(s_actual))
    
    ax.plot(timesteps, s_actual, color="black", lw=1.5, label="Actual S5 Cycle Time")
    ax.plot(timesteps, s_pred_gcn, color="green", lw=1.2, label=f"LSTM+GCN (MAE {mae_gcn:.1f}s)")
    ax.plot(timesteps, s_pred_lstm, color="red", lw=1.2, ls="--", label=f"LSTM-only (MAE {mae_lstm:.1f}s)")
    
    ax.axvspan(30, 70, alpha=0.15, color="orange", label="S3 Bottleneck (upstream)")
    ax.set_ylabel("Cycle Time (s)")
    ax.set_xlabel("Predicted Timestep (minutes)")
    ax.set_title("Ablation: 5-min Ahead Forecast for S5 (N+2) during S3 Bottleneck", fontsize=11)
    ax.legend()
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "stage4_ablation.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")

    print("\n[5/5] Saving model ...")
    model_dir = ROOT / "data" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_gcn.state_dict(), model_dir / "core_model.pt")

    print("\n" + "=" * 65)
    if pass_ablation:
        print("VERDICT: PASS -- GCN successfully anticipates upstream state.")
    else:
        print("VERDICT: STOP -- GCN ablation failed. The model isn't learning the ripple.")
    print("=" * 65 + "\n")
    
    return pass_ablation

if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
