"""
tasks/tutorial/mapie_native.py
----------------------------
Variant A – MAPIE trains the base model internally (standard MAPIE usage).

The base model is a LightGBM regressor; MAPIE wraps it with split conformal
prediction.  The sigma (nonconformity) model is the same ECFP-based architecture
used in the main qubounds pipeline, so results are directly comparable.

Both adaptive (sigma-normalised, ResidualNormalisedScore) and non-adaptive
(plain AbsoluteConformityScore) variants are run and saved, enabling a
within-run comparison of interval width / coverage.

Inputs  (ploomber params)
---------
  train_data   : str  – path to Excel produced by load_dataset.py (Training sheet)
  test_data    : str  – path to Excel produced by load_dataset.py (Test sheet)
  meta_path    : str  – path to JSON metadata produced by load_dataset.py
  alpha        : float – miscoverage level (default 0.1 → 90 % intervals)
  ncm          : str  – sigma-model key passed to make_sigma_model()
  cache_path   : str  – ECFP SQLite cache path
  product      : dict – {nb, data, ncmodel_adaptive, ncmodel_plain}

Outputs
-------
  product["data"]             : Excel with predictions + intervals (both variants)
  product["ncmodel_adaptive"] : pickle – conformal model with sigma normalisation
  product["ncmodel_plain"]    : pickle – conformal model without normalisation
"""

# + tags=["parameters"]
alpha       = 0.1
ncm         = "rlgbmecfp"
cache_path  = None
product     = None
upstream = None
dataset = None
# -

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore, AbsoluteConformityScore
from sklearn.model_selection import train_test_split

from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_diagnostic import (
    make_sigma_model, sigma_diagnostics,
    detect_residual_degeneracy, PositiveSigmaWrapper,
)

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)

tag = f"tutorial_load_{dataset}"
train_data = upstream["tutorial_load_*"][tag]["train"]
test_data = upstream["tutorial_load_*"][tag]["test"]
meta_path = upstream["tutorial_load_*"][tag]["meta"]


# ── metadata
with open(meta_path) as f:
    meta = json.load(f)

target_col = meta["target_col"]
print(f"Dataset: {meta['dataset']}  target: {target_col}")

# ── load data
train_df = pd.read_excel(train_data, sheet_name="Training")
test_df = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

# Split training further into fit + calibration (80 / 20)
fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

print(f"fit={len(fit_df)}  cal={len(cal_df)}  test={len(test_df)}")

# ── ECFP fingerprints 
init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])


X_fit = to_ecfp(fit_df)
X_cal = to_ecfp(cal_df)
X_test = to_ecfp(test_df)

y_fit = fit_df[target_col].values.astype(float)
y_cal = cal_df[target_col].values.astype(float)
y_test = test_df[target_col].values.astype(float)

# ── base model (LightGBM, trained by MAPIE internally) ───────────────────────
# This is the key difference from the "predefined model" variant:
# MAPIE receives an unfitted estimator and trains it on X_fit / y_fit,
# then conformises on X_cal / y_cal.
base_model = LGBMRegressor(objective="huber", random_state=42, verbose=-1)

# ── sigma model (same as main pipeline) ───────────────────────────────────────
# We need residuals on the fit set to train sigma.
# Strategy: fit base model on fit_df, predict on fit_df, compute residuals.
base_model.fit(X_fit, y_fit)
y_fit_pred = base_model.predict(X_fit)
residuals_fit = np.abs(y_fit - y_fit_pred)

use_eps, diag = detect_residual_degeneracy(residuals_fit, y_fit)
print(f"Residual diag: p90={diag['p90']:.4g}  frac_zero={diag['frac_zero']:.2f}  use_eps={use_eps}")

sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)

sigma_fit_pred = sigma_model.predict(X_fit)
diag_sigma = sigma_diagnostics(residuals_fit, sigma_fit_pred)
print(f"Sigma R²={diag_sigma['r2']:.3f}  RMSE={diag_sigma['rmse']:.4g}")


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT A1 – Adaptive (ResidualNormalisedScore)
# ─────────────────────────────────────────────────────────────────────────────
# Re-create an unfitted base model so MAPIE can own the fitting.
# MAPIE SplitConformalRegressor with prefit=False fits the estimator on the
# data passed to .fit(), then conformises on the data passed to .conformalize().
base_adaptive = LGBMRegressor(objective="huber", random_state=42, verbose=-1)

conformity_adaptive = ResidualNormalisedScore(
    residual_estimator=sigma_model,
    prefit=True,   # sigma already fitted above
    sym=True,
)

mapie_adaptive = SplitConformalRegressor(
    estimator=base_adaptive,
    conformity_score=conformity_adaptive,
    prefit=False,            # MAPIE will fit base_adaptive
    confidence_level=1 - alpha,
)

mapie_adaptive.fit(X_fit, y_fit)
mapie_adaptive.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
lower_a = y_pis_a[:, 0, 0]
upper_a = y_pis_a[:, 1, 0]
width_a = upper_a - lower_a
covered_a = (y_test >= lower_a) & (y_test <= upper_a)

print(f"\nAdaptive CP  |  coverage={covered_a.mean():.3f}  mean_width={width_a.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT A2 – Non-adaptive (AbsoluteConformityScore, plain split CP)
# ─────────────────────────────────────────────────────────────────────────────
base_plain = LGBMRegressor(objective="huber", random_state=42, verbose=-1)

mapie_plain = SplitConformalRegressor(
    estimator=base_plain,
    conformity_score=AbsoluteConformityScore(),
    prefit=False,
    confidence_level=1 - alpha,
)

mapie_plain.fit(X_fit, y_fit)
mapie_plain.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

y_pred_p, y_pis_p = mapie_plain.predict_interval(X_test)
lower_p = y_pis_p[:, 0, 0]
upper_p = y_pis_p[:, 1, 0]
width_p = upper_p - lower_p
covered_p = (y_test >= lower_p) & (y_test <= upper_p)

print(f"Non-adaptive CP  |  coverage={covered_p.mean():.3f}  mean_width={width_p.mean():.4f}")

# ── save models ───────────────────────────────────────────────────────────────
def _save(path, mapie_obj, sigma_obj, diag_s, variant):
    save_dict = {
        "mapie": mapie_obj,
        "sigma_model": sigma_obj,
        "ncm": ncm,
        "variant": variant,
        "alpha": alpha,
        "sigma_r2":   diag_s["r2"],
        "sigma_rmse": diag_s["rmse"],
        "sigma_mae":  diag_s["mae"],
        "meta": meta,
    }
    with open(path, "wb") as fh:
        pickle.dump(save_dict, fh)
    print(f"Saved {variant} model → {path}")

_save(product["ncmodel_adaptive"], mapie_adaptive, sigma_model, diag_sigma, "adaptive")
_save(product["ncmodel_plain"],    mapie_plain,    sigma_model, diag_sigma, "plain")

# ── collate results ───────────────────────────────────────────────────────────
result_df = pd.DataFrame({
    "Smiles":            test_df["Smiles"].values,
    "True":              y_test,
    # adaptive
    "Pred_adaptive":     y_pred_a,
    "Lower_adaptive":    lower_a,
    "Upper_adaptive":    upper_a,
    "Width_adaptive":    width_a,
    "Covered_adaptive":  covered_a.astype(int),
    # plain
    "Pred_plain":        y_pred_p,
    "Lower_plain":       lower_p,
    "Upper_plain":       upper_p,
    "Width_plain":       width_p,
    "Covered_plain":     covered_p.astype(int),
})

metrics = pd.DataFrame([
    {
        "variant": "adaptive",
        "alpha": alpha,
        "coverage": covered_a.mean(),
        "mean_width": width_a.mean(),
        "median_width": np.median(width_a),
        "sigma_r2": diag_sigma["r2"],
        "n_test": len(test_df),
    },
    {
        "variant": "plain",
        "alpha": alpha,
        "coverage": covered_p.mean(),
        "mean_width": width_p.mean(),
        "median_width": np.median(width_p),
        "sigma_r2": None,
        "n_test": len(test_df),
    },
])

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics.to_excel(w, sheet_name="Metrics", index=False)

print("\nResults saved →", product["data"])
print(metrics[["variant", "coverage", "mean_width", "sigma_r2"]].to_string(index=False))
