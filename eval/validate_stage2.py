"""
eval/validate_stage2.py - Stage 2 validation: Soft Sensor Module.

Protocol
--------
1. Generate training data: 10 simulation runs (seeds 0-9, no faults)
   using well-instrumented stations only.  Concatenate into one DataFrame.
2. Hold out station 9 (Fuel System Install, well-instrumented) -- pretend
   it's proxy-only by hiding its true sensor columns.
3. Train SoftSensorModel on all other well-instrumented stations.
4. Evaluate on:
   a) Held-out station 9 (the "pseudo proxy-only" test):
      Compare est_* vs true obs columns.
   b) Actual proxy-only stations (4, 8, 15):
      Compare est_* vs _gt_* columns from ground_truth.parquet.
5. Plot predicted vs actual temperature over time for station 9.
6. Stop condition: if MAE > 2 * true_std for any target at station 9,
   flag STOP and explain.

Note on manual stations
-----------------------
Stations 11 (HVAC Install) and 17 (Final QC Inspection) have NO proxy
signals (only timestamps + pass/fail).  The soft sensor is not attempted
for them.  Their estimated sensor values will remain NaN and will be
handled in Stage 4 exclusively via the GCN neighbor pathway.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulator import AssemblyLineSimulator, load_config
from soft_sensor import SoftSensorModel, apply_soft_sensor, TARGET_COLS

# ---- Paths ----------------------------------------------------------------
PLOT_DIR       = ROOT / "data" / "plots"
MODEL_SAVE_DIR = ROOT / "data" / "soft_sensor"
VAL_OBS        = ROOT / "data" / "runs" / "stage1_validation_obs.parquet"
VAL_GT         = ROOT / "data" / "runs" / "stage1_validation_gt.parquet"

HELD_OUT_STATION = 9    # Fuel System Install (well-instrumented, pretend proxy-only)
PROXY_ONLY_IDS   = [4, 8, 15]
MANUAL_IDS       = [11, 17]
N_TRAIN_RUNS     = 10   # runs for training data (no faults, varied seeds)
N_STEPS          = 240  # 4 hours per run

# Stop if MAE > STOP_MAE_STD_MULTIPLE × true_std
STOP_MAE_STD_MULTIPLE = 2.0


# --------------------------------------------------------------------------- #
#  Data generation
# --------------------------------------------------------------------------- #
def generate_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate N_TRAIN_RUNS simulated runs (no faults) and concatenate.
    Returns (obs_df, gt_df) for all runs.
    """
    all_obs = []
    all_gt  = []
    cfg = load_config()
    print(f"  Generating {N_TRAIN_RUNS} training runs × {N_STEPS} steps ...")
    for seed in range(N_TRAIN_RUNS):
        sim = AssemblyLineSimulator(config=cfg, seed=seed)
        obs, _, gt = sim.run(n_steps=N_STEPS)
        # Add a run_id column so lags don't bleed across runs
        obs["run_id"] = seed
        gt["run_id"]  = seed
        all_obs.append(obs)
        all_gt.append(gt)
    obs_df = pd.concat(all_obs, ignore_index=True)
    gt_df  = pd.concat(all_gt,  ignore_index=True)
    total_wi = (obs_df["station_type"] == "well_instrumented").sum()
    print(
        f"  Total rows: {len(obs_df):,}  "
        f"| Well-instrumented rows (training eligible): {total_wi:,}"
    )
    return obs_df, gt_df


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def train_model(obs_df: pd.DataFrame) -> SoftSensorModel:
    print(f"\n  Training SoftSensorModel (hold out station {HELD_OUT_STATION}) ...")
    model = SoftSensorModel(n_estimators=200, max_depth=4, learning_rate=0.07)
    model.fit(obs_df, held_out_station_id=HELD_OUT_STATION, verbose=True)
    save_path = MODEL_SAVE_DIR / "soft_sensor.joblib"
    model.save(save_path)
    print(f"  Model saved -> {save_path}")
    return model


# --------------------------------------------------------------------------- #
#  Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate_held_out(model: SoftSensorModel, val_obs: pd.DataFrame, val_gt: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Evaluate on held-out station 9 (pseudo proxy-only)."""
    metrics = model.evaluate(val_obs, val_gt, station_ids=[HELD_OUT_STATION])
    print(f"\n  --- Held-out Station {HELD_OUT_STATION} (Fuel System Install) ---")
    print(f"  {'Target':<14} {'MAE':>8} {'RMSE':>8} {'true_std':>10} {'rel_err':>9}  {'Status'}")
    print("  " + "-" * 65)

    all_pass = True
    for _, row in metrics.iterrows():
        rel_err = row["MAE"] / row["true_std"] if row["true_std"] > 0 else float("inf")
        threshold = STOP_MAE_STD_MULTIPLE * row["true_std"]
        status = "[OK]  " if row["MAE"] <= threshold else "[FAIL]"
        if row["MAE"] > threshold:
            all_pass = False
        print(
            f"  {row['target']:<14} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} "
            f"{row['true_std']:>10.3f} {rel_err:>9.3f}  {status}"
        )
    return metrics, all_pass


def evaluate_proxy_only(model: SoftSensorModel, val_obs: pd.DataFrame, val_gt: pd.DataFrame) -> pd.DataFrame:
    """Evaluate on actual proxy-only stations using hidden ground truth."""
    metrics = model.evaluate(val_obs, val_gt, station_ids=PROXY_ONLY_IDS)
    print(f"\n  --- Actual Proxy-Only Stations {PROXY_ONLY_IDS} ---")
    print(f"  {'SID':<5} {'Station type':<16} {'Target':<14} {'MAE':>8} {'RMSE':>8} {'true_std':>10} {'rel_err':>9}")
    print("  " + "-" * 75)
    for _, row in metrics.iterrows():
        rel_err = row["MAE"] / row["true_std"] if row["true_std"] > 0 else float("inf")
        sname = {4: "Engine Bay Prep", 8: "Brake Line Rtg", 15: "Seating Install"}.get(
            int(row["station_id"]), f"S{int(row['station_id'])}"
        )
        print(
            f"  {int(row['station_id']):<5} {row['station_type']:<16} {row['target']:<14} "
            f"{row['MAE']:>8.3f} {row['RMSE']:>8.3f} {row['true_std']:>10.3f} {rel_err:>9.3f}"
        )
    return metrics


# --------------------------------------------------------------------------- #
#  Plot
# --------------------------------------------------------------------------- #
def plot_predicted_vs_actual(
    model: SoftSensorModel,
    val_obs: pd.DataFrame,
    val_gt: pd.DataFrame,
) -> None:
    """Plot temperature predicted vs actual for held-out station 9."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sid = HELD_OUT_STATION
    grp = val_obs[val_obs["station_id"] == sid].copy().sort_values("timestep")
    pred = model.predict(grp)

    # True temperature (from ground truth -- station 9 is well-instrumented
    # in val_obs, so we can use the true column directly)
    true_temp = grp["temperature"].values
    est_temp  = pred["est_temperature"].values
    timesteps = grp["timestep"].values

    # Also get vibration and torque
    true_vib  = grp["vibration"].values
    est_vib   = pred["est_vibration"].values
    true_tor  = grp["torque"].values
    est_tor   = pred["est_torque"].values

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for ax, true_vals, est_vals, label, unit in zip(
        axes,
        [true_temp, true_vib, true_tor],
        [est_temp,  est_vib,  est_tor],
        ["Temperature", "Vibration", "Torque"],
        ["°C", "mm/s RMS", "Nm"],
    ):
        ax.plot(timesteps, true_vals, color="steelblue", lw=1.3, label=f"True {label}")
        ax.plot(timesteps, est_vals,  color="orangered", lw=1.1, ls="--", label=f"Soft-sensor estimate")
        mae = np.nanmean(np.abs(true_vals - est_vals))
        ax.set_ylabel(f"{label} ({unit})", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"S{sid} (Fuel System Install) -- {label}  |  MAE = {mae:.3f} {unit}",
            fontsize=9
        )

    axes[-1].set_xlabel("Timestep (minutes)")
    fig.suptitle(
        f"Soft Sensor: Predicted vs Actual  (Station {sid}, held-out from training)",
        fontsize=11,
    )
    fpath = PLOT_DIR / "stage2_soft_sensor.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved -> {fpath}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 2 VALIDATION - Soft Sensor Module")
    print("=" * 65)

    # ---- 1. Training data ----
    print("\n[1/5] Generating training data ...")
    train_obs, train_gt = generate_training_data()

    # ---- 2. Train ----
    print("\n[2/5] Training model ...")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model = train_model(train_obs)

    # ---- 3. Load validation data (Stage 1 run) ----
    print("\n[3/5] Loading validation data (Stage 1 run) ...")
    if not VAL_OBS.exists():
        print("  [FAIL] Stage 1 validation data not found. Run validate_stage1.py first.")
        return False
    val_obs = pd.read_parquet(VAL_OBS)
    val_gt  = pd.read_parquet(VAL_GT)
    print(f"  Loaded {len(val_obs)} rows from Stage 1 run.")

    # ---- 4a. Evaluate on held-out station ----
    print("\n[4/5] Evaluating ...")
    held_out_metrics, held_out_pass = evaluate_held_out(model, val_obs, val_gt)

    # ---- 4b. Evaluate on proxy-only stations ----
    proxy_metrics = evaluate_proxy_only(model, val_obs, val_gt)

    # ---- 4c. Manual stations ----
    print(f"\n  --- Manual Stations {MANUAL_IDS} ---")
    print("  Soft sensor NOT attempted: no proxy signals available.")
    print("  These stations will be handled by GCN neighbor estimation in Stage 4.")

    # ---- 5. Plot ----
    print("\n[5/5] Generating plot ...")
    plot_predicted_vs_actual(model, val_obs, val_gt)

    # ---- Verdict ----
    # Also apply soft sensor to augment val_obs and save for downstream stages
    cfg = load_config()
    augmented_obs = apply_soft_sensor(val_obs, model, cfg.proxy_only_ids)
    aug_path = ROOT / "data" / "runs" / "stage2_augmented_obs.parquet"
    augmented_obs.to_parquet(aug_path, index=False)
    print(f"  Augmented observations saved -> {aug_path}")

    print("\n" + "=" * 65)
    if held_out_pass:
        print("VERDICT: PASS -- Soft sensor error within acceptable bounds.")
        print(f"  Held-out S{HELD_OUT_STATION}: MAE < {STOP_MAE_STD_MULTIPLE}x true_std for all targets.")
        print("  Proxy-only station estimates computed and saved.")
        print("  Manual stations (11, 17) explicitly skipped -- handled by GCN in Stage 4.")
    else:
        print("VERDICT: STOP -- Soft sensor error too high.")
        print(f"  MAE exceeded {STOP_MAE_STD_MULTIPLE}x true_std at held-out station {HELD_OUT_STATION}.")
        print("  Investigate feature engineering or training data before Stage 3.")
    print("=" * 65 + "\n")
    return held_out_pass


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
