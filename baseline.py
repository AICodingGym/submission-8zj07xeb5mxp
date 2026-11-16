"""Baseline pipeline for Ventilator Pressure Prediction (DSC 102 PA3).

- Validation: hold out 20% of whole breaths (grouped by breath_id), fixed seed.
- Metric: MAE on inspiratory rows only (u_out == 0).
- Model: XGBoost (hist) on per-breath engineered features.
- Output: submission.csv (id, pressure) for every test row, plus a line appended
  to experiments.csv recording MAE and runtime.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 42
VAL_FRAC = 0.2
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent

t0 = time.time()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()
    g = df.groupby("breath_id")
    df["step"] = g.cumcount()
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["dt"] = g["time_step"].diff().fillna(0)
    df["u_in_area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    for k in (1, 2, 3):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0)
    df["u_in_breath_mean"] = g["u_in"].transform("mean")
    df["u_in_breath_max"] = g["u_in"].transform("max")
    df["R_C"] = df["R"] * 100 + df["C"]  # categorical lung setting as an int
    df["RC_prod"] = df["R"] * df["C"]
    return df


FEATURES = [
    "R", "C", "R_C", "RC_prod", "time_step", "step", "u_in", "u_out",
    "u_in_cumsum", "u_in_area", "dt",
    "u_in_lag1", "u_in_lag2", "u_in_lag3",
    "u_in_lead1", "u_in_lead2", "u_in_lead3",
    "u_in_diff1", "u_out_lag1", "u_in_breath_mean", "u_in_breath_max",
]

train = add_features(pd.read_csv(DATA / "train.csv"))
test = add_features(pd.read_csv(DATA / "test.csv"))
print(f"loaded + features: {time.time() - t0:.1f}s  train={train.shape} test={test.shape}")

# Breath-level holdout split (fixed seed -> same split for every experiment)
breaths = train["breath_id"].unique()
rng = np.random.default_rng(SEED)
val_breaths = set(rng.choice(breaths, size=int(len(breaths) * VAL_FRAC), replace=False))
is_val = train["breath_id"].isin(val_breaths)
tr, va = train[~is_val], train[is_val]
assert set(tr.breath_id).isdisjoint(va.breath_id)
print(f"train breaths={tr.breath_id.nunique()}  val breaths={va.breath_id.nunique()}")

# Train only on inspiratory rows: those are the only rows that are scored.
tr_insp = tr[tr.u_out == 0]
va_insp = va[va.u_out == 0]

params = dict(
    objective="reg:absoluteerror", tree_method="hist", max_depth=10,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, nthread=-1, seed=SEED,
)
dtr = xgb.DMatrix(tr_insp[FEATURES], label=tr_insp["pressure"])
dva = xgb.DMatrix(va_insp[FEATURES], label=va_insp["pressure"])
booster = xgb.train(
    params, dtr, num_boost_round=2000, evals=[(dva, "val")],
    early_stopping_rounds=50, verbose_eval=100,
)

pred_va = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
val_mae = float(np.mean(np.abs(va_insp["pressure"].to_numpy() - pred_va)))
print(f"\nVALIDATION MAE (inspiratory, u_out==0): {val_mae:.5f}")
print(f"MLEvolve benchmark: 0.14237  gap: {val_mae - 0.14237:+.5f}")

# Per lung-setting error breakdown for later error analysis
err = va_insp.assign(abs_err=np.abs(va_insp["pressure"].to_numpy() - pred_va))
print(err.groupby(["R", "C"])["abs_err"].mean().round(4).to_string())

# Predict every test row (expiratory rows get a finite prediction too; not scored)
pred_te = booster.predict(xgb.DMatrix(test[FEATURES]),
                          iteration_range=(0, booster.best_iteration + 1))
sub = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": pred_te}).sort_values("id")
sample = pd.read_csv(DATA / "sample_submission.csv")
assert len(sub) == len(sample) and (sub["id"].to_numpy() == sample["id"].to_numpy()).all()
assert np.isfinite(sub["pressure"]).all()
sub.to_csv(OUT / "submission.csv", index=False)

runtime = time.time() - t0
print(f"wrote submission.csv ({len(sub)} rows)  runtime={runtime:.1f}s")

log = OUT / "experiments.csv"
row = pd.DataFrame([{
    "experiment": "baseline_xgb_v1", "val_mae": round(val_mae, 5),
    "best_iter": booster.best_iteration, "runtime_s": round(runtime, 1),
    "notes": "XGB hist, L1 obj, insp-only training, 21 feats, 20% breath holdout seed 42",
}])
row.to_csv(log, mode="a", header=not log.exists(), index=False)
