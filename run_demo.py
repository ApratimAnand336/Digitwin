"""
run_demo.py - End-to-end DigitalTwin.ai pipeline demo.

Runs the entire pipeline on a fresh simulated run and saves outputs:
  1. Generate simulation data with one bottleneck + one defect
  2. Train / load soft sensor
  3. Train / load core GCN model
  4. Run anomaly detection + action engine
  5. Print summary report + save plots

Run: python run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import AssemblyLineSimulator, load_config
from soft_sensor import SoftSensorModel, apply_soft_sensor
from model import DigitalTwinModel, AssemblyLineDataset, get_edge_index
from anomaly import AnomalyDetector
from action import PrescriptiveEngine

PLOT_DIR = ROOT / "data" / "plots"
MODEL_DIR = ROOT / "data" / "model"
SS_DIR = ROOT / "data" / "soft_sensor"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 65


def banner(msg: str) -> None:
    print(f"\n{SEP}\n  {msg}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1: Generate demo run
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 1 — Generating demo simulation run")
cfg = load_config()
sim = AssemblyLineSimulator(config=cfg, seed=8888)
sim.inject_bottleneck(station_id=3, start_step=40, duration_steps=50, severity=0.65)
sim.inject_defect(station_id=8, start_step=120, pattern="sine_drift")

obs_df, fault_log, gt_df = sim.run(n_steps=200)
obs_df["run_id"] = 0
print(f"  Generated {len(obs_df)} rows ({cfg.n_stations} stations × 200 timesteps)")
print(f"  Fault log:\n{fault_log.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2: Soft sensor
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 2 — Soft Sensor")
ss_path = SS_DIR / "soft_sensor.joblib"
if ss_path.exists():
    print("  Loading existing soft sensor model …")
    ss_model = SoftSensorModel.load(ss_path)
else:
    print("  Training soft sensor (this will take ~30 s) …")
    # Quick demo training on one run (validate_stage2 does proper train/val split)
    ss_model = SoftSensorModel(n_estimators=100)
    ss_model.fit(obs_df, verbose=True)
    ss_model.save(ss_path)

obs_df = apply_soft_sensor(obs_df, ss_model, cfg.proxy_only_ids)
print("  Soft sensor applied to proxy-only stations.")

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4: Core model
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 4 — Loading Core LSTM+GCN Model")
edge_index = get_edge_index()

# Normalisation from this run
ds = AssemblyLineDataset(obs_df, n_stations=cfg.n_stations)
in_features = len(ds.means)

core_path = MODEL_DIR / "core_model.pt"
if core_path.exists():
    print("  Loading pretrained core model …")
    core_model = DigitalTwinModel(in_features, use_gcn=True)
    core_model.load_state_dict(torch.load(core_path, map_location="cpu", weights_only=True))
    core_model.eval()
else:
    print("  Core model not found. Run validate_stage4.py first to train it.")
    print("  Skipping model-dependent stages.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 5: Anomaly detection + attribution
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 5 — Anomaly Detection + Attribution")
from torch.utils.data import DataLoader

detector = AnomalyDetector(alpha=3.0, window_size=30)
engine = PrescriptiveEngine()

mean_c = float(ds.means["cycle_time_s"])
std_c = float(ds.stds["cycle_time_s"])

loader = DataLoader(ds, batch_size=1, shuffle=False)

anomaly_records = []
action_records = []

station_names = {s.id: s.name for s in cfg.stations}
station_types = {s.id: s.type for s in cfg.stations}

core_model.eval()
with torch.no_grad():
    for i, (X, Y_short, _) in enumerate(loader):
        current_t = i + 15

        p_short, p_long_m, p_long_s = core_model(X, edge_index)

        for sid in range(cfg.n_stations):
            pred = p_short[0, sid, 0].item() * std_c + mean_c
            actual = Y_short[0, sid, 0].item() * std_c + mean_c

            result = detector.process(
                station_id=sid,
                actual=actual,
                predicted=pred,
                upstream_station_id=sid - 1 if sid > 0 else None,
                station_type=station_types[sid],
            )

            anomaly_records.append({
                "timestep": current_t,
                "station_id": sid,
                "station": station_names[sid],
                "is_anomaly": result.is_anomaly,
                "anomaly_type": result.anomaly_type,
                "type_confidence": result.type_confidence,
                "attribution_origin_id": result.attribution_origin,
                "attribution_confidence": result.attribution_confidence,
                "residual": result.residual,
                "threshold": result.threshold,
                "actual": actual,
                "predicted": pred,
            })

            # Actions for flagged stations
            if result.is_anomaly:
                state = {
                    "station": station_names[sid],
                    "feeder": station_names[sid - 1] if sid > 0 else "None",
                    "anomaly_flagged": result.is_anomaly,
                    "anomaly_type": result.anomaly_type,
                    "attribution_confidence": result.attribution_confidence,
                    "attribution_origin": (
                        station_names[result.attribution_origin]
                        if result.attribution_origin is not None else station_names[sid]
                    ),
                    "queue_depth_forecast_ratio": 1.0,
                }
                actions = engine.evaluate(state)
                for act in actions:
                    action_records.append({
                        "timestep": current_t,
                        "station": station_names[sid],
                        "rule_id": act.rule_id,
                        "severity": act.severity,
                        "recommendation": act.recommendation,
                        "message": act.message,
                    })

anom_df = pd.DataFrame(anomaly_records)
act_df = pd.DataFrame(action_records)

# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 5 — Detection Metrics")

t_array = anom_df["timestep"].unique()
s3_fault_mask = (anom_df["station_id"] == 3) & (anom_df["timestep"] >= 40) & (anom_df["timestep"] < 90)
s8_fault_mask = (anom_df["station_id"] == 8) & (anom_df["timestep"] >= 120)

s3_detected = anom_df[s3_fault_mask]["is_anomaly"].sum()
s8_detected = anom_df[s8_fault_mask]["is_anomaly"].sum()

print(f"  Station 3 (Bottleneck, t=40–90): {s3_detected} flags (total window = {s3_fault_mask.sum()} steps)")
print(f"  Station 8 (Defect,     t=120+) : {s8_detected} flags (total window = {s8_fault_mask.sum()} steps)")

# Precision / recall
true_positive_mask = (
    s3_fault_mask & anom_df["is_anomaly"]
) | (
    s8_fault_mask & anom_df["is_anomaly"]
)
all_flags = anom_df["is_anomaly"].sum()
all_true = s3_fault_mask.sum() + s8_fault_mask.sum()
true_positives = true_positive_mask.sum()

prec = true_positives / all_flags if all_flags > 0 else 0.0
recall = true_positives / all_true if all_true > 0 else 0.0

print(f"\n  Precision : {prec:.2f}")
print(f"  Recall    : {recall:.2f}")
print(f"  F1        : {2*prec*recall/(prec+recall+1e-6):.2f}")

# Attribution accuracy for downstream S4
s4_flags_during_bn = anom_df[
    (anom_df["station_id"] == 4) & (anom_df["timestep"] >= 40) & (anom_df["timestep"] < 90)
    & anom_df["is_anomaly"]
]
if len(s4_flags_during_bn) > 0:
    attr_correct = (s4_flags_during_bn["attribution_origin_id"] == 3).mean()
    print(f"\n  Attribution accuracy (S4→S3 during bottleneck): {attr_correct:.0%}")
else:
    print("\n  No downstream S4 flags during bottleneck window.")

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 6 — Actions summary
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 6 — Sample Prescriptive Actions")
if act_df.empty:
    print("  No actions generated (no anomalies detected).")
else:
    # Deduplicate – show first occurrence of each unique rule+station
    unique_acts = act_df.drop_duplicates(subset=["station", "rule_id"])
    print(f"  {len(unique_acts)} unique action types generated:")
    for _, row in unique_acts.iterrows():
        print(f"  [{row['severity']}] t={row['timestep']:3d} | {row['station']}")
        print(f"       Rule: {row['rule_id']}")
        print(f"       {row['message']}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────
banner("Stage 5 — Generating Demo Plot")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for idx, (sid, title, fault_start, fault_end) in enumerate([
    (3, "Station 3 — Bottleneck (cycle-time surge)", 40, 90),
    (8, "Station 8 — Defect (sinusoidal quality drift)", 120, 200),
]):
    ax = axes[idx]
    grp = anom_df[anom_df["station_id"] == sid]
    ax.plot(grp["timestep"], grp["actual"], color="black", lw=1.5, label="Actual", zorder=3)
    ax.plot(grp["timestep"], grp["predicted"], color="steelblue", lw=1.2,
            ls="--", label="Predicted (1-step)", zorder=3)
    ax.fill_between(grp["timestep"], 0,
                    grp["actual"].max() * 1.3,
                    where=((grp["timestep"] >= fault_start) & (grp["timestep"] < fault_end)),
                    alpha=0.15, color="orange", label="True Fault Window")
    flags = grp[grp["is_anomaly"]]
    if len(flags) > 0:
        ax.scatter(flags["timestep"], flags["actual"],
                   color="red", marker="x", s=60, zorder=4, label="Detected Anomaly")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Cycle Time (s)")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Timestep (minutes)")
fig.suptitle("DigitalTwin.ai — Demo Run Anomaly Detection", fontsize=12, fontweight="bold")
fig.tight_layout()
out = PLOT_DIR / "demo_anomaly_detection.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  Plot saved → {out}")

banner("DEMO COMPLETE")
print("  Run `streamlit run dashboard/app.py` to launch the full dashboard.\n")
