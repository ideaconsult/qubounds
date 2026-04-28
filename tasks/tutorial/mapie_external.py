import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore, AbsoluteConformityScore
from sklearn.model_selection import train_test_split

from qubounds.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from qubounds.mapie_diagnostic import (
    make_sigma_model, sigma_diagnostics, detect_residual_degeneracy,
)
# Re-use ExternalPredictor from the main regression module
from qubounds.mapie_regression import ExternalPredictor
from IPython.display import display, Markdown, HTML
%matplotlib inline

"""
tasks/tutorial/mapie_external.py
------------------------------
Variant B – Pre-fitted external model, same pattern as the VEGA pipeline.

The base model is trained independently , and only
its *predictions* are passed to MAPIE via the ExternalPredictor wrapper.
This mirrors exactly how qubounds wraps VEGA models, making results directly
comparable to the main pipeline.

Adaptive (sigma-normalised) and non-adaptive variants are both computed.

Inputs  (ploomber params)
---------
  train_data    : str  – Training Excel (Smiles, target_col)
  test_data     : str  – Test Excel     (Smiles, target_col)
  meta_path     : str  – JSON metadata from load_dataset.py
  alpha         : float
  ncm           : str  – sigma-model key
  cache_path    : str
  base_model_type : str – "lgbm" (default) | "rf" | "ridge"
                          The external model type to train and then treat as black-box.
  product       : dict – {nb, data, ncmodel_adaptive, ncmodel_plain}
"""


# + tags=["parameters"]
alpha           = 0.1
ncm             = "rlgbmecfp"
cache_path      = None
base_model_type = "lgbm"   # treat as "external" – could be anything
product         = None
dataset = None
upstream = None
# -


Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)

tag = f"tutorial_load_{dataset}"
train_data = upstream["tutorial_load_*"][tag]["train"]
test_data = upstream["tutorial_load_*"][tag]["test"]
meta_path = upstream["tutorial_load_*"][tag]["meta"]


# ── metadata 
with open(meta_path) as f:
    meta = json.load(f)

target_col = meta["target_col"]
display(Markdown(f"## Dataset: {meta['dataset']}  target: {target_col}"))

# ── load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

# Split training: fit-set (for base model + sigma model) + calibration set
fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"fit={len(fit_df)}  cal={len(cal_df)}  test={len(test_df)}"))

# ── ECFP fingerprints ─────────────────────────────────────────────────────────
init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df)
X_cal  = to_ecfp(cal_df)
X_test = to_ecfp(test_df)

y_fit  = fit_df[target_col].values.astype(float)
y_cal  = cal_df[target_col].values.astype(float)
y_test = test_df[target_col].values.astype(float)

# ── "external" base model: train on fit_df, then freeze ──────────────────────
_BASE_MODELS = {
    "lgbm":  LGBMRegressor(objective="huber", random_state=42, verbose=-1),
    "rf":    RandomForestRegressor(n_estimators=100, random_state=42),
    "ridge": Ridge(),
}
if base_model_type not in _BASE_MODELS:
    raise ValueError(f"Unknown base_model_type='{base_model_type}'. "
                     f"Choose from {list(_BASE_MODELS)}")

base_model = _BASE_MODELS[base_model_type]
base_model.fit(X_fit, y_fit)

# Compute external predictions for calibration and test sets
# From here on, the base model is treated as a black box:
# only its outputs are used, not the model object itself.
y_pred_cal  = base_model.predict(X_cal)
y_pred_fit  = base_model.predict(X_fit)
y_pred_test = base_model.predict(X_test)

display(Markdown(f"- Base model ({base_model_type}) trained.  "
      f"Cal R² proxy: {np.corrcoef(y_cal, y_pred_cal)[0,1]**2:.3f}"))

# ── sigma model ───────────────────────────────────────────────────────────────
residuals_fit = np.abs(y_fit - y_pred_fit)

use_eps, diag = detect_residual_degeneracy(residuals_fit, y_fit)
display(Markdown(f"- Residual diag: p90={diag['p90']:.4g}  frac_zero={diag['frac_zero']:.2f}  use_eps={use_eps}"))

sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)

sigma_pred_fit = sigma_model.predict(X_fit)
diag_sigma = sigma_diagnostics(residuals_fit, sigma_pred_fit)
display(Markdown(f"- Sigma R²={diag_sigma['r2']:.3f}  RMSE={diag_sigma['rmse']:.4g}"))

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT B1 – Adaptive (ExternalPredictor + ResidualNormalisedScore)
# ─────────────────────────────────────────────────────────────────────────────
estimator_a = ExternalPredictor(y_pred_cal)

conformity_adaptive = ResidualNormalisedScore(
    residual_estimator=sigma_model,
    prefit=True,
    sym=True,
)

mapie_adaptive = SplitConformalRegressor(
    estimator=estimator_a,
    conformity_score=conformity_adaptive,
    prefit=True,
    confidence_level=1 - alpha,
)
mapie_adaptive.estimator_ = estimator_a.fit(None, None)
mapie_adaptive.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

# Predict: swap in test predictions before calling predict_interval
mapie_adaptive.estimator_.y_pred = y_pred_test
y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
lower_a = y_pis_a[:, 0, 0]
upper_a = y_pis_a[:, 1, 0]
width_a = upper_a - lower_a
covered_a = (y_test >= lower_a) & (y_test <= upper_a)

display(Markdown(f"- Adaptive CP  |  coverage={covered_a.mean():.3f}  mean_width={width_a.mean():.4f}"))

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT B2 – Non-adaptive (AbsoluteConformityScore)
# ─────────────────────────────────────────────────────────────────────────────
estimator_p = ExternalPredictor(y_pred_cal)

mapie_plain = SplitConformalRegressor(
    estimator=estimator_p,
    conformity_score=AbsoluteConformityScore(),
    prefit=True,
    confidence_level=1 - alpha,
)
mapie_plain.estimator_ = estimator_p.fit(None, None)
mapie_plain.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

mapie_plain.estimator_.y_pred = y_pred_test
y_pred_p, y_pis_p = mapie_plain.predict_interval(X_test)
lower_p = y_pis_p[:, 0, 0]
upper_p = y_pis_p[:, 1, 0]
width_p = upper_p - lower_p
covered_p = (y_test >= lower_p) & (y_test <= upper_p)

display(Markdown(f"- Non-adaptive CP  |  coverage={covered_p.mean():.3f}  mean_width={width_p.mean():.4f}"))

# ── save models ───────────────────────────────────────────────────────────────
def _save(path, mapie_obj, sigma_obj, diag_s, variant):
    save_dict = {
        "mapie": mapie_obj,
        "sigma_model": sigma_obj,
        "ncm": ncm,
        "variant": variant,
        "alpha": alpha,
        "base_model_type": base_model_type,
        "sigma_r2":   diag_s["r2"],
        "sigma_rmse": diag_s["rmse"],
        "sigma_mae":  diag_s["mae"],
        "meta": meta,
    }
    with open(path, "wb") as fh:
        pickle.dump(save_dict, fh)
    display(Markdown(f"Saved {variant} model → {path}"))

_save(product["ncmodel_adaptive"], mapie_adaptive, sigma_model, diag_sigma, "adaptive_external")
_save(product["ncmodel_plain"],    mapie_plain,    sigma_model, diag_sigma, "plain_external")

# ── collate results ───────────────────────────────────────────────────────────
result_df = pd.DataFrame({
    "Smiles":             test_df["Smiles"].values,
    "True":               y_test,
    "Pred_external":      y_pred_test,
    # adaptive
    "Lower_adaptive":     lower_a,
    "Upper_adaptive":     upper_a,
    "Width_adaptive":     width_a,
    "Covered_adaptive":   covered_a.astype(int),
    # plain
    "Lower_plain":        lower_p,
    "Upper_plain":        upper_p,
    "Width_plain":        width_p,
    "Covered_plain":      covered_p.astype(int),
})

metrics = pd.DataFrame([
    {
        "variant": "adaptive_external",
        "base_model": base_model_type,
        "alpha": alpha,
        "coverage": covered_a.mean(),
        "mean_width": width_a.mean(),
        "median_width": np.median(width_a),
        "sigma_r2": diag_sigma["r2"],
        "n_test": len(test_df),
    },
    {
        "variant": "plain_external",
        "base_model": base_model_type,
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

display(Markdown(f"- Results saved → {product['data']}"))
display(metrics[["variant", "coverage", "mean_width", "sigma_r2"]])
