"""
soft_sensor/model.py - Gradient-boosted regression models for predicting
true sensor values from proxy signals.

Design
------
* One GradientBoostingRegressor per target signal (temperature, vibration, torque).
* Features: proxy signals (motor_current, setpoint_error, cycle_duration_s)
  + lag-1/2/3 of each proxy + cycle_time_s + queue_depth + baseline_cycle_s.
* Trained on well-instrumented stations only (where both proxy AND true sensor
  values are available in observations_df).
* Applied to proxy-only stations where true sensor columns are NaN.
* For manual stations (no proxy signals at all): not attempted here -- they are
  handled in Stage 4 purely via GCN neighbor-based estimation.  This is
  documented explicitly and not treated as a gap.

The proxy→true relationship in the simulator is:
    motor_current  = I_base * (1 + 0.4*tanh(...temp...)) * (1 + 0.25*(vib/vib_base-1)) + noise
    setpoint_error = 0.12 * tanh((torque - torque_base) / torque_base) + noise
    cycle_duration = cycle_time + noise

This is a nonlinear (tanh/product) transform -- NOT a linear copy -- so the
GBM has real signal to learn, and simple linear regression would underperform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------- #
#  Column name constants
# --------------------------------------------------------------------------- #
PROXY_COLS   = ["motor_current", "setpoint_error", "cycle_duration_s"]
TARGET_COLS  = ["temperature", "vibration", "torque"]
EXTRA_FEATS  = ["cycle_time_s", "queue_depth", "baseline_cycle_s"]
N_LAGS       = 3


# --------------------------------------------------------------------------- #
#  Feature engineering
# --------------------------------------------------------------------------- #
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the feature matrix from an observations DataFrame.

    Lag features are computed within each station's time series (groupby
    station_id) so they don't bleed across station boundaries.

    Parameters
    ----------
    df : observations DataFrame (may span multiple stations / timesteps).
         Must have columns: PROXY_COLS + EXTRA_FEATS + 'station_id'.

    Returns
    -------
    feat : DataFrame with shape (len(df), n_features). Rows with NaN lag
           values (first N_LAGS rows per station) will have NaN in lag cols.
    """
    feat = df[EXTRA_FEATS + PROXY_COLS].copy().reset_index(drop=True)

    grouped = df.groupby("station_id", sort=False)
    for col in PROXY_COLS:
        for lag in range(1, N_LAGS + 1):
            lag_series = grouped[col].shift(lag)
            lag_series = lag_series.reset_index(drop=True)
            feat[f"{col}_lag{lag}"] = lag_series

    return feat


# --------------------------------------------------------------------------- #
#  Main model class
# --------------------------------------------------------------------------- #
class SoftSensorModel:
    """
    Wraps three GradientBoostingRegressor models (one per true sensor target)
    for predicting true sensor values from proxy signals.

    Usage
    -----
    model = SoftSensorModel()
    model.fit(obs_df, held_out_station_id=9)   # train, hold out S9 for val
    preds = model.predict(obs_df)              # DataFrame: est_temperature, ...
    model.save("data/soft_sensor/model.joblib")

    model2 = SoftSensorModel.load("data/soft_sensor/model.joblib")
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.07,
    ) -> None:
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.models:        Dict[str, GradientBoostingRegressor] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False

    # ---------------------------------------------------------------------- #
    #  Fit
    # ---------------------------------------------------------------------- #
    def fit(
        self,
        df: pd.DataFrame,
        held_out_station_id: Optional[int] = None,
        verbose: bool = True,
    ) -> "SoftSensorModel":
        """
        Train on well-instrumented stations.

        Parameters
        ----------
        df                   : Full observations DataFrame (or multiple runs concat'd).
        held_out_station_id  : If set, exclude this station from training so it can
                               be used as a validation "pseudo proxy-only" station.
        verbose              : Print training MAE per target.
        """
        # Filter: well-instrumented only, optionally excluding held-out station
        mask = df["station_type"] == "well_instrumented"
        if held_out_station_id is not None:
            mask &= df["station_id"] != held_out_station_id
        train_df = df[mask].copy()

        if train_df.empty:
            raise ValueError("No training data after filtering. Check station types.")

        feat = build_feature_matrix(train_df)

        # Drop rows with NaN (first N_LAGS rows per station, or missing proxy cols)
        valid_mask = feat.notna().all(axis=1)
        for col in TARGET_COLS:
            valid_mask &= train_df[col].reset_index(drop=True).notna()

        X = feat[valid_mask].values
        self.feature_names = list(feat.columns)

        for target in TARGET_COLS:
            y = train_df[target].reset_index(drop=True)[valid_mask].values

            gbr = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                min_samples_leaf=5,
                random_state=42,
            )
            gbr.fit(X, y)
            self.models[target] = gbr

            if verbose:
                train_mae  = mean_absolute_error(y, gbr.predict(X))
                true_std   = y.std()
                print(
                    f"  {target:<12}: train MAE={train_mae:.3f}  "
                    f"true_std={true_std:.3f}  "
                    f"rel_err={train_mae/true_std:.3f}"
                )

        self.is_fitted = True
        return self

    # ---------------------------------------------------------------------- #
    #  Predict
    # ---------------------------------------------------------------------- #
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict true sensor values for every row in df.

        Returns DataFrame with columns [est_temperature, est_vibration, est_torque]
        aligned to df's index.

        Missing lag values (first N_LAGS rows per station) are filled with
        the column median before prediction to avoid NaN output.
        """
        if not self.is_fitted:
            raise RuntimeError("SoftSensorModel is not fitted. Call fit() first.")

        feat = build_feature_matrix(df)
        # Fill lag NaNs: forward-fill within station then global median fallback
        for col in feat.columns:
            if feat[col].isna().any():
                feat[col] = feat[col].ffill().bfill()
                feat[col] = feat[col].fillna(feat[col].median())

        X = feat.values
        result = pd.DataFrame(index=df.index)
        for target in TARGET_COLS:
            result[f"est_{target}"] = self.models[target].predict(X)

        return result

    # ---------------------------------------------------------------------- #
    #  Evaluate (against ground truth)
    # ---------------------------------------------------------------------- #
    def evaluate(
        self,
        obs_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        station_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute MAE / RMSE per station, per target.

        For well-instrumented stations: true values come from obs_df['temperature'...].
        For proxy-only stations:        true values come from gt_df['_gt_temperature'...].

        Parameters
        ----------
        obs_df      : Observations DataFrame (proxy signals + possibly true sensors).
        gt_df       : Ground truth DataFrame (all internal simulator values).
        station_ids : Optional subset to evaluate.

        Returns
        -------
        metrics_df  : DataFrame with columns [station_id, station_type, target, MAE, RMSE, true_std, n_samples].
        """
        if station_ids is not None:
            obs_sub = obs_df[obs_df["station_id"].isin(station_ids)].copy()
            gt_sub  = gt_df[gt_df["station_id"].isin(station_ids)].copy()
        else:
            obs_sub = obs_df.copy()
            gt_sub  = gt_df.copy()

        preds = self.predict(obs_sub)

        rows = []
        for sid, grp in obs_sub.groupby("station_id"):
            stype = grp["station_type"].iloc[0]
            pred_grp = preds.loc[grp.index]
            gt_grp   = gt_sub[gt_sub["station_id"] == sid]

            for target in TARGET_COLS:
                # Pick ground truth source
                if stype == "well_instrumented":
                    true_vals = grp[target].values
                else:
                    gt_col = f"_gt_{target}"
                    if gt_col not in gt_grp.columns:
                        continue
                    # Align by timestep
                    merged = grp[["timestep"]].merge(
                        gt_grp[["timestep", gt_col]], on="timestep", how="left"
                    )
                    true_vals = merged[gt_col].values

                est_vals = pred_grp[f"est_{target}"].values

                valid = ~(np.isnan(true_vals) | np.isnan(est_vals))
                if valid.sum() < 5:
                    continue

                y_true = true_vals[valid]
                y_pred = est_vals[valid]

                rows.append({
                    "station_id":   sid,
                    "station_type": stype,
                    "target":       target,
                    "MAE":          float(mean_absolute_error(y_true, y_pred)),
                    "RMSE":         float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "true_std":     float(y_true.std()),
                    "n_samples":    int(valid.sum()),
                })

        return pd.DataFrame(rows)

    # ---------------------------------------------------------------------- #
    #  Save / Load
    # ---------------------------------------------------------------------- #
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models":        self.models,
                "feature_names": self.feature_names,
                "n_estimators":  self.n_estimators,
                "max_depth":     self.max_depth,
                "learning_rate": self.learning_rate,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "SoftSensorModel":
        data = joblib.load(path)
        obj = cls(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            learning_rate=data["learning_rate"],
        )
        obj.models        = data["models"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted     = True
        return obj


# --------------------------------------------------------------------------- #
#  Apply soft sensor to fill proxy-only stations
# --------------------------------------------------------------------------- #
def apply_soft_sensor(
    obs_df: pd.DataFrame,
    model: SoftSensorModel,
    proxy_only_ids: List[int],
) -> pd.DataFrame:
    """
    Return obs_df augmented with estimated true sensor columns.

    Rules:
    - well_instrumented stations: est_* = true observed sensor value
    - proxy_only stations:        est_* = model prediction from proxy signals
    - manual stations:            est_* = NaN (handled by GCN in Stage 4)

    New columns added: est_temperature, est_vibration, est_torque.
    """
    result = obs_df.copy()

    for target in TARGET_COLS:
        est_col = f"est_{target}"
        # Default: copy true sensor (well-instrumented have it; others stay NaN)
        result[est_col] = result[target]

    # Proxy-only: run model
    proxy_mask = result["station_id"].isin(proxy_only_ids)
    if proxy_mask.any():
        preds = model.predict(result[proxy_mask])
        for target in TARGET_COLS:
            result.loc[proxy_mask, f"est_{target}"] = preds[f"est_{target}"].values

    return result
