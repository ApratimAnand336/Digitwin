"""
eval/validate_stage1.py - Stage 1 validation: Assembly Line Simulator.

Checks:
  1. Generate 4 hours of data with one bottleneck + one defect.
  2. Plot cycle time heatmap (all stations x timesteps) - ripple must be visible.
  3. Plot cycle times for 4 consecutive stations around the bottleneck.
  4. Confirm proxy-only and manual stations have NaN for true-sensor columns.
  5. Print summary stats (mean/std per station for cycle_time).
  6. Verdict: PASS if ripple is present and NaN pattern is correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make sure the project root is on the path when run as a script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulator import AssemblyLineSimulator, load_config

# ---- Configuration --------------------------------------------------------
PLOT_DIR = ROOT / "data" / "plots"
DATA_DIR = ROOT / "data" / "runs"
FAULT_DIR = ROOT / "data" / "fault_logs"

N_HOURS = 4
N_STEPS = N_HOURS * 60     # 240 timesteps (1 min each)

# Bottleneck: station 3 (Underbody Framing), starts at step 60, lasts 50 steps
BN_STATION  = 3
BN_START    = 60
BN_DURATION = 50
BN_SEVERITY = 0.55   # 55% cycle time inflation

# Defect: station 8 (Brake Line Routing, proxy-only), starts at step 130
DEF_STATION = 8
DEF_START   = 130


def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = AssemblyLineSimulator(seed=42)
    sim.inject_bottleneck(BN_STATION, BN_START, BN_DURATION, BN_SEVERITY)
    sim.inject_defect(DEF_STATION, DEF_START)
    obs_df, fault_log, gt_df = sim.run(n_steps=N_STEPS)
    return obs_df, fault_log, gt_df


def save_data(obs_df, fault_log, gt_df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAULT_DIR.mkdir(parents=True, exist_ok=True)
    obs_df.to_parquet(DATA_DIR / "stage1_validation_obs.parquet", index=False)
    fault_log.to_csv(FAULT_DIR / "stage1_validation_faults.csv", index=False)
    gt_df.to_parquet(DATA_DIR / "stage1_validation_gt.parquet", index=False)
    print(f"  Saved observations  -> {DATA_DIR / 'stage1_validation_obs.parquet'}")
    print(f"  Saved fault log     -> {FAULT_DIR / 'stage1_validation_faults.csv'}")
    print(f"  Saved ground truth  -> {DATA_DIR / 'stage1_validation_gt.parquet'}")


def plot_cycle_heatmap(obs_df: pd.DataFrame, cfg) -> None:
    """Heatmap of cycle_time / baseline_cycle_s across all stations and timesteps."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pivot = obs_df.pivot(index="station_id", columns="timestep", values="cycle_time_s")
    baseline = obs_df.groupby("station_id")["baseline_cycle_s"].first()
    # Normalise each station's row by its baseline
    norm_pivot = pivot.div(baseline, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        norm_pivot.values,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0.9, vmax=2.0,
        origin="upper",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("cycle_time / baseline", fontsize=10)

    # Annotate fault windows
    ax.axvspan(BN_START, BN_START + BN_DURATION, alpha=0.15, color="red", label="bottleneck window")
    ax.axvspan(DEF_START, N_STEPS, alpha=0.10, color="purple", label="defect window")

    station_names = obs_df.groupby("station_id")["station_name"].first().values
    ax.set_yticks(range(len(station_names)))
    ax.set_yticklabels([f"S{i}: {n[:18]}" for i, n in enumerate(station_names)], fontsize=7)
    ax.set_xlabel("Timestep (minutes)")
    ax.set_title("Cycle Time / Baseline — All Stations (red=slow, green=fast)\nBottleneck S3 + Defect S8", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)

    fpath = PLOT_DIR / "stage1_heatmap.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap       -> {fpath}")


def plot_ripple(obs_df: pd.DataFrame, cfg) -> None:
    """Line plot of cycle times for 4 stations around the bottleneck."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Show bottleneck station + 3 downstream
    stations_to_plot = list(range(BN_STATION, min(BN_STATION + 5, cfg.n_stations)))

    fig, axes = plt.subplots(
        len(stations_to_plot), 1, figsize=(13, 2.5 * len(stations_to_plot)), sharex=True
    )
    if len(stations_to_plot) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, sid, color in zip(axes, stations_to_plot, colors):
        sub = obs_df[obs_df["station_id"] == sid].copy()
        base = sub["baseline_cycle_s"].iloc[0]
        name = sub["station_name"].iloc[0]
        stype = sub["station_type"].iloc[0]

        ax.plot(sub["timestep"], sub["cycle_time_s"], color=color, lw=1.2,
                label=f"S{sid} {name[:22]} ({stype[:2].upper()})")
        ax.axhline(base, color=color, lw=0.8, ls="--", alpha=0.6, label="baseline")
        ax.axvspan(BN_START, BN_START + BN_DURATION, alpha=0.18, color="red", label="bottleneck" if sid == BN_STATION else None)
        ax.set_ylabel("cycle_time (s)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(base * 0.5, base * 2.5)

    axes[-1].set_xlabel("Timestep (minutes)")
    fig.suptitle("Ripple Effect: Cycle Times at Bottleneck Station + Downstream", fontsize=11, y=1.01)
    fpath = PLOT_DIR / "stage1_ripple.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ripple plot   -> {fpath}")


def check_nan_pattern(obs_df: pd.DataFrame) -> bool:
    """
    Verify NaN pattern per station type:
      - well_instrumented: temperature/vibration/torque are NOT NaN
      - proxy_only:        temperature/vibration/torque ARE NaN;
                           motor_current/setpoint_error NOT NaN
      - manual:            true sensors NaN; proxy signals NaN;
                           pass_fail NOT NaN
    Returns True if all checks pass.
    """
    true_cols  = ["temperature", "vibration", "torque"]
    proxy_cols = ["motor_current", "setpoint_error", "cycle_duration_s"]
    all_ok = True

    for stype, group in obs_df.groupby("station_type"):
        nan_frac_true  = group[true_cols].isna().mean()
        nan_frac_proxy = group[proxy_cols].isna().mean()
        nan_frac_pf    = group["pass_fail"].isna().mean()

        if stype == "well_instrumented":
            for col in true_cols:
                if nan_frac_true[col] > 0.01:
                    print(f"  [FAIL] well_instrumented station has NaN in {col} ({nan_frac_true[col]:.0%})")
                    all_ok = False
                else:
                    print(f"  [OK]   well_instrumented '{col}' non-null (NaN={nan_frac_true[col]:.0%})")
            if nan_frac_pf < 0.99:
                print(f"  [FAIL] well_instrumented has unexpected pass_fail values")
                all_ok = False

        elif stype == "proxy_only":
            for col in true_cols:
                if nan_frac_true[col] < 0.99:
                    print(f"  [FAIL] proxy_only station '{col}' not NaN ({nan_frac_true[col]:.0%} NaN)")
                    all_ok = False
                else:
                    print(f"  [OK]   proxy_only '{col}' all NaN (correct)")
            for col in proxy_cols:
                if nan_frac_proxy[col] > 0.01:
                    print(f"  [FAIL] proxy_only '{col}' unexpectedly NaN ({nan_frac_proxy[col]:.0%})")
                    all_ok = False
                else:
                    print(f"  [OK]   proxy_only '{col}' non-null (NaN={nan_frac_proxy[col]:.0%})")

        elif stype == "manual":
            for col in true_cols + proxy_cols:
                nan_frac = group[col].isna().mean()
                if nan_frac < 0.99:
                    print(f"  [FAIL] manual station '{col}' not NaN ({nan_frac:.0%} NaN)")
                    all_ok = False
            if nan_frac_pf > 0.01:
                print(f"  [FAIL] manual station has NaN pass_fail ({nan_frac_pf:.0%})")
                all_ok = False
            else:
                print(f"  [OK]   manual pass_fail non-null (NaN={nan_frac_pf:.0%})")

    return all_ok


def check_ripple(obs_df: pd.DataFrame) -> bool:
    """
    Confirm the bottleneck ripple is visible downstream.
    Compare mean cycle time at BN_STATION+1 during vs outside the bottleneck window.
    The during-window mean should be noticeably higher.
    """
    ds_id = BN_STATION + 1  # first downstream station
    sub = obs_df[obs_df["station_id"] == ds_id].copy()

    # Use a window that gives the EMA time to build up (skip first 5 steps of fault)
    ripple_window = sub[
        (sub["timestep"] >= BN_START + 5) & (sub["timestep"] < BN_START + BN_DURATION)
    ]
    normal_window = sub[sub["timestep"] < BN_START]

    if ripple_window.empty or normal_window.empty:
        print("  [FAIL] Not enough data to check ripple")
        return False

    mean_during = ripple_window["cycle_time_s"].mean()
    mean_normal = normal_window["cycle_time_s"].mean()
    ratio = mean_during / mean_normal

    base = sub["baseline_cycle_s"].iloc[0]
    name = sub["station_name"].iloc[0]

    print(f"  Downstream S{ds_id} ({name[:25]}):")
    print(f"    baseline:           {base:.1f}s")
    print(f"    mean cycle (normal):    {mean_normal:.1f}s")
    print(f"    mean cycle (bottleneck window): {mean_during:.1f}s")
    print(f"    ratio (during/normal):  {ratio:.3f}")

    if ratio > 1.08:   # at least 8% increase downstream
        print(f"  [OK]   Ripple confirmed at downstream station (ratio={ratio:.3f} > 1.08)")
        return True
    else:
        print(f"  [FAIL] Ripple too weak at downstream station (ratio={ratio:.3f} <= 1.08)")
        return False


def print_summary_stats(obs_df: pd.DataFrame, cfg) -> None:
    """Print mean/std of cycle_time per station."""
    stats = obs_df.groupby(["station_id", "station_name", "station_type"])["cycle_time_s"].agg(
        ["mean", "std", "min", "max"]
    ).round(2)

    print("\n  Per-station cycle_time summary (seconds):")
    print(f"  {'ID':<4} {'Type':<18} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}  Name")
    print("  " + "-" * 80)
    for (sid, sname, stype), row in stats.iterrows():
        flag = " <<FAULT>>" if sid in (BN_STATION, DEF_STATION) else ""
        print(f"  {sid:<4} {stype:<18} {row['mean']:>7.1f} {row['std']:>7.2f} "
              f"{row['min']:>7.1f} {row['max']:>7.1f}  {sname[:30]}{flag}")


def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 1 VALIDATION - Assembly Line Simulator")
    print("=" * 65)

    # 1. Generate data
    print(f"\n[1/5] Running simulator ({N_HOURS}h = {N_STEPS} steps, 18 stations)...")
    obs_df, fault_log, gt_df = run_simulation()
    n_rows = len(obs_df)
    print(f"  Generated {n_rows} rows ({N_STEPS} timesteps x 18 stations)")
    print(f"  Columns: {list(obs_df.columns)}")
    print(f"\n  Fault log:")
    print(fault_log.to_string(index=False))
    save_data(obs_df, fault_log, gt_df)

    # 2. NaN pattern check
    print("\n[2/5] Checking sensor coverage NaN pattern ...")
    nan_ok = check_nan_pattern(obs_df)

    # 3. Ripple check
    print("\n[3/5] Checking ripple effect at downstream station ...")
    cfg = load_config()
    ripple_ok = check_ripple(obs_df)

    # 4. Summary stats
    print("\n[4/5] Summary statistics ...")
    print_summary_stats(obs_df, cfg)

    # 5. Plots
    print("\n[5/5] Generating plots ...")
    plot_cycle_heatmap(obs_df, cfg)
    plot_ripple(obs_df, cfg)

    # Verdict
    all_ok = nan_ok and ripple_ok
    print("\n" + "=" * 65)
    if all_ok:
        print("VERDICT: PASS -- Stage 1 simulator is working correctly.")
        print("  NaN pattern correct, ripple visible, data looks plausible.")
        print(f"  Plots saved to: {PLOT_DIR}")
    else:
        problems = []
        if not nan_ok:    problems.append("NaN pattern incorrect")
        if not ripple_ok: problems.append("Ripple not visible downstream")
        print(f"VERDICT: STOP -- {len(problems)} problem(s) found:")
        for p in problems:
            print(f"  * {p}")
        print("  Fix simulator before proceeding to Stage 2.")
    print("=" * 65 + "\n")
    return all_ok


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
