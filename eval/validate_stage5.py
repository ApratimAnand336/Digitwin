"""
eval/validate_stage5.py - Stage 5 validation: Anomaly Detection.

Protocol
--------
1. Load Soft Sensor and Core GCN Model.
2. Generate a 4-hour test run with two specific faults:
   - Bottleneck at S3 (starts t=60, lasts 40 mins)
   - Defect at S8 (starts t=130, persists)
3. Step through the dataset chronologically:
   - Get 1-step ahead prediction from Core Model.
   - Pass actual vs prediction into AnomalyDetector.
4. Record anomaly scores and flags.
5. Verify that S3 flags bottleneck and S8 flags defect during their windows.
6. Plot the scores and thresholds for both stations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulator import AssemblyLineSimulator, load_config
from soft_sensor import SoftSensorModel, apply_soft_sensor
from model import DigitalTwinModel, AssemblyLineDataset, get_edge_index
from anomaly import AnomalyDetector

PLOT_DIR = ROOT / "data" / "plots"


def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 5 VALIDATION - Anomaly Detection")
    print("=" * 65)

    cfg = load_config()
    edge_index = get_edge_index()

    print("\n[1/4] Loading models ...")
    ss_path = ROOT / "data" / "soft_sensor" / "soft_sensor.joblib"
    core_path = ROOT / "data" / "model" / "core_model.pt"
    
    if not ss_path.exists() or not core_path.exists():
        print("  [FAIL] Missing model files. Run Stages 2 and 4 first.")
        return False
        
    ss_model = SoftSensorModel.load(ss_path)
    
    # We need a dummy dataset to get the normalization constants and feature count
    sim_dummy = AssemblyLineSimulator(config=cfg, seed=0)
    dummy_obs, _, _ = sim_dummy.run(n_steps=100)
    dummy_obs["run_id"] = 0
    dummy_obs = apply_soft_sensor(dummy_obs, ss_model, cfg.proxy_only_ids)
    dummy_ds = AssemblyLineDataset(dummy_obs, n_stations=cfg.n_stations)
    
    in_features = len(dummy_ds.means)
    core_model = DigitalTwinModel(in_features, use_gcn=True)
    core_model.load_state_dict(torch.load(core_path))
    core_model.eval()

    print("\n[2/4] Generating test data with faults ...")
    sim = AssemblyLineSimulator(config=cfg, seed=404)
    # 1. Bottleneck at S3 (t=60 to 100)
    sim.inject_bottleneck(station_id=3, start_step=60, duration_steps=40, severity=0.6)
    # 2. Defect at S8 (t=140 onwards, sinusoidal drift)
    sim.inject_defect(station_id=8, start_step=140, pattern="sine_drift")
    
    test_obs, fault_log, _ = sim.run(n_steps=240)
    test_obs["run_id"] = 404
    test_obs = apply_soft_sensor(test_obs, ss_model, cfg.proxy_only_ids)
    
    test_ds = AssemblyLineDataset(test_obs, n_stations=cfg.n_stations)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print("\n[3/4] Running Anomaly Detector in real-time simulation ...")
    detector = AnomalyDetector(alpha=4.0, window_size=30)
    
    mean_c = test_ds.means["cycle_time_s"]
    std_c = test_ds.stds["cycle_time_s"]
    
    # Tracking for plotting (station 3 and 8)
    timesteps = []
    
    s3_scores, s3_thresh, s3_flags = [], [], []
    s8_scores, s8_thresh, s8_flags = [], [], []
    
    with torch.no_grad():
        for i, (X, Y_short, _) in enumerate(test_loader):
            # i is the timestep offset. X covers t to t+14. 
            # Prediction is for t+15 (which is horizon index 0 of Y_short)
            current_t = i + 15 
            timesteps.append(current_t)
            
            p_short, _, _ = core_model(X, edge_index)
            
            for sid in [3, 8]:
                # 1-step ahead prediction (horizon index 0)
                pred = p_short[0, sid, 0].item() * std_c + mean_c
                actual = Y_short[0, sid, 0].item() * std_c + mean_c
                
                result = detector.process(sid, actual, pred,
                                         upstream_station_id=sid-1 if sid > 0 else None)
                
                if sid == 3:
                    s3_scores.append(result.residual)
                    s3_thresh.append(result.threshold)
                    s3_flags.append(result.is_anomaly)
                else:
                    s8_scores.append(result.residual)
                    s8_thresh.append(result.threshold)
                    s8_flags.append(result.is_anomaly)

    # Check detections
    s3_flags = np.array(s3_flags)
    s8_flags = np.array(s8_flags)
    
    t_array = np.array(timesteps)
    s3_fault_mask = (t_array >= 60) & (t_array < 100)
    s8_fault_mask = (t_array >= 140)
    
    s3_detected = s3_flags[s3_fault_mask].sum() > 0
    s8_detected = s8_flags[s8_fault_mask].sum() > 0
    
    print(f"  Station 3 (Bottleneck, t=60-100) -> Detected: {s3_detected} ({s3_flags[s3_fault_mask].sum()} flags)")
    print(f"  Station 8 (Defect, t=140+)       -> Detected: {s8_detected} ({s8_flags[s8_fault_mask].sum()} flags)")
    
    all_ok = s3_detected and s8_detected
    
    print("\n[4/4] Generating plots ...")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # S3 Plot
    ax = axes[0]
    ax.plot(timesteps, s3_scores, color="blue", label="Residual Error")
    ax.plot(timesteps, s3_thresh, color="red", linestyle="--", label="Dynamic Threshold")
    ax.fill_between(timesteps, 0, max(s3_scores)*1.1, where=s3_fault_mask, color="orange", alpha=0.2, label="True Fault Window")
    
    # Mark anomalies
    anom_t = t_array[s3_flags]
    anom_s = np.array(s3_scores)[s3_flags]
    ax.scatter(anom_t, anom_s, color="red", marker="x", s=50, label="Anomaly Flagged")
    
    ax.set_title("Station 3 - Bottleneck Detection (Cycle Time Surge)")
    ax.set_ylabel("Error (s)")
    ax.legend(loc="upper right")
    
    # S8 Plot
    ax = axes[1]
    ax.plot(timesteps, s8_scores, color="green", label="Residual Error")
    ax.plot(timesteps, s8_thresh, color="red", linestyle="--", label="Dynamic Threshold")
    ax.fill_between(timesteps, 0, max(s8_scores)*1.1, where=s8_fault_mask, color="orange", alpha=0.2, label="True Fault Window")
    
    anom_t8 = t_array[s8_flags]
    anom_s8 = np.array(s8_scores)[s8_flags]
    ax.scatter(anom_t8, anom_s8, color="red", marker="x", s=50, label="Anomaly Flagged")
    
    ax.set_title("Station 8 - Defect Detection (Sinusoidal Drift)")
    ax.set_ylabel("Error (s)")
    ax.set_xlabel("Timestep (minutes)")
    ax.legend(loc="upper right")
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "stage5_anomaly.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")

    print("\n" + "=" * 65)
    if all_ok:
        print("VERDICT: PASS -- Anomalies successfully detected dynamically.")
    else:
        print("VERDICT: STOP -- Failed to detect faults.")
    print("=" * 65 + "\n")

    return all_ok

if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
