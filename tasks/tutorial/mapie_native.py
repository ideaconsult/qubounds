# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_native.py
--------------------------------
Tutorial: Conformal Prediction for QSAR - Regression (internal base model)
===========================================================================

Demonstrates split conformal prediction with an internally-trained base model
using the SAME production functions from the main qubounds pipeline.

Key concepts illustrated:
  alpha, sigma model, q_hat, coverage (per-prediction and per-model),
  efficiency (per-prediction and per-model), calibration objective,
  exchangeability (KS test), NCM quality comparison.

Inputs (ploomber params) - all sourced from pipeline.tutorial.yaml
---------
  dataset        : dataset key (matches grid and upstream load task)
  ncm            : sigma-model key
  alpha          : miscoverage level
  cache_path     : ECFP SQLite cache path
  dataset_config : full dict from env.tutorial.yaml {dataset: {path, target_col, ...}}
  product        : {nb, data, ncmodel_adaptive, ncmodel_plain}
"""

# + tags=["parameters"]
dataset        = None
ncm            = "rlgbmecfp"
alpha          = 0.1
cache_path     = None
dataset_config = None
product        = None
upstream       = None
base_model = "lgbm"
threshold = 0.5
# -

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp, spearmanr, mannwhitneyu
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore, AbsoluteConformityScore
from qubounds.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from qubounds.mapie_diagnostic import make_sigma_model, sigma_diagnostics, detect_residual_degeneracy
from qubounds.mapie_regression import ExternalPredictor
%matplotlib inline

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

colors = {
    "TRAINING": "#4CAF50",  
    "CALIBRATION":  "#FF9800",
    "TEST":   "#2196F3",   
}

# ==============================================================================
# S0  Resolve upstream and dataset config
# ==============================================================================
tag        = f"tutorial_load_regr_{dataset}"
data = upstream["tutorial_load_regr_*"][tag]["data"]
meta_path  = upstream["tutorial_load_regr_*"][tag]["meta"]

with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]
pred_col = meta.get("pred_col", None)
if pred_col is None and base_model == "file": 
   base_model = "catboost" 
ad_cols = meta.get("ad_cols", [])
ad_col_directions = meta.get("ad_col_directions",[])
n_quantile_bins = meta.get("n_quantile_bins", 5)

display(Markdown(f"# CONFORMAL PREDICTION TUTORIAL - REGRESSION ({'external' if base_model=='file' else 'internal'} model)"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))

# ==============================================================================
# S1  Formal definitions
# ==============================================================================
display(Markdown("## S1  Formal definitions"))
display(Markdown(r"""
### Formal definitions (per-prediction vs per-model)

**alpha** -- miscoverage level. *Scalar, set by the user.*
alpha=0.1 => 10% may miss => 90% coverage targeted.

**Nonconformity score** s(x_i, y_i) -- *per-prediction*, on calibration set.
  Adaptive: s = |y - y_hat| / sigma_hat(x)
  Plain:    s = |y - y_hat|
Higher score = more surprising = model less reliable at this molecule.

**Calibration quantile** q_hat -- *per-model*, computed once.
  q_hat = quantile({s_1,...,s_n}, level = ceil((n+1)*(1-alpha)) / n)
The inflated level (n+1)/n ensures marginal coverage >= 1-alpha (Vovk 2005).

**Prediction interval** C(x) -- *per-prediction*, at inference time.
  Adaptive: y_hat +/- q_hat * sigma_hat(x)   (molecule-specific width)
  Plain:    y_hat +/- q_hat                   (same width for all molecules)

**Coverage**:
  Per-prediction:  cov_i = 1[y_i in C(x_i)]        (binary per molecule)
  Per-model:       Cov = mean(cov_i) over test set   (validity criterion)
  CP guarantee:    Cov >= 1-alpha in expectation.

**Efficiency**:
  Per-prediction:  w_i = upper_i - lower_i          (interval width)
  Per-model:       W = mean(w_i)                     (primary efficiency metric)

**Calibration objective:**
  Minimise W = mean interval width,  subject to: Cov >= 1-alpha
The adaptive variant achieves lower W when sigma_hat(x) correctly ranks
molecules by local uncertainty.

**Conditional vs marginal coverage:**
CP guarantees *marginal* coverage (averaged over all test molecules).
Per-subgroup coverage is not guaranteed. The adaptive variant approximates
conditional coverage but the formal guarantee remains marginal.
"""))

# ==============================================================================
# S2  Data splits
# ==============================================================================
display(Markdown("## S2  Data splits"))
display(Markdown("""
Split CP requires THREE disjoint sets:
  Fit set         -> train base model + sigma model
  Calibration set -> compute nonconformity scores; derive q_hat
  Test set        -> evaluate coverage and efficiency
The calibration set is CONSUMED by the conformal procedure.
"""))

train_df = pd.read_excel(data, sheet_name="Training")
test_df  = pd.read_excel(data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

if base_model == "file": 
    df_test_ad = test_df.copy()
    _available_ad = []
    if ad_cols:
        for _ac in ad_cols:
            if _ac in test_df.columns:
                df_test_ad[_ac] = test_df["Smiles"].map(
                    test_df.set_index("Smiles")[_ac]).values
                _available_ad.append(_ac)
            else:
                display(Markdown(f"- WARNING: AD column `{_ac}` not found in external file."))
        if _available_ad:
            display(Markdown(f"- AD columns loaded: {_available_ad}"))
    fit_df = train_df.copy().reset_index(drop=True)
    cal_df, test_df = train_test_split(test_df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)
    cal_df = cal_df.reset_index(drop=True)                
else:   
    df_test_ad = None
    _available_ad = [] 
    fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
    fit_df = fit_df.reset_index(drop=True)
    cal_df = cal_df.reset_index(drop=True)
# ==============================================================================
# S3  Fingerprints and base model
# ==============================================================================
display(Markdown("## S3  Fingerprints (ECFP4) and base model (LightGBM)"))

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df);  y_fit  = fit_df[target_col].values.astype(float)
X_cal  = to_ecfp(cal_df);  y_cal  = cal_df[target_col].values.astype(float)
X_test = to_ecfp(test_df); y_test = test_df[target_col].values.astype(float)

if base_model == "lgbm":
    base_model_ = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
    base_model_.fit(X_fit, y_fit)
    y_fit_pred  = base_model_.predict(X_fit)
    y_cal_pred  = base_model_.predict(X_cal)
    y_test_pred = base_model_.predict(X_test)
elif base_model == "catboost":
    base_model_ = CatBoostRegressor(random_state=42)
    base_model_.fit(X_fit, y_fit)
    y_fit_pred  = base_model_.predict(X_fit)
    y_cal_pred  = base_model_.predict(X_cal)
    y_test_pred = base_model_.predict(X_test)    
elif base_model == "file":
    y_fit_pred  = fit_df[pred_col].values.astype(float)
    y_cal_pred  = cal_df[pred_col].values.astype(float)
    y_test_pred = test_df[pred_col].values.astype(float)
else:
    assert False,f"{base_model} not supported"

display(Markdown(f"- Base model R2 on train set: {r2_score(y_fit, y_fit_pred):.3f}"))
display(Markdown(f"- Base model R2 on cal set: {r2_score(y_cal, y_cal_pred):.3f}"))
display(Markdown(f"- Base model R2 on test set: {r2_score(y_test, y_test_pred):.3f}"))

# ==============================================================================
# S4  Sigma model
# ==============================================================================
display(Markdown(f"## S4  Sigma model ({ncm})"))
display(Markdown(f"""
Predicts |y - y_hat| from ECFP fingerprints. Trained on fit-set residuals.
Low R2 does NOT invalidate coverage -- q_hat provides the guarantee regardless.
Better sigma model => narrower intervals (better efficiency).
"""))

residuals_fit = np.abs(y_fit - y_fit_pred)
_, _ = detect_residual_degeneracy(residuals_fit, y_fit)
sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)

sigma_pred_fit  = sigma_model.predict(X_fit)
sigma_pred_cal  = sigma_model.predict(X_cal)
sigma_pred_test = sigma_model.predict(X_test)

diag_s     = sigma_diagnostics(residuals_fit,                   sigma_pred_fit)
diag_s_cal = sigma_diagnostics(np.abs(y_cal - y_cal_pred),      sigma_pred_cal)
diag_s_test = sigma_diagnostics(np.abs(y_test - y_test_pred),      sigma_pred_test)

display(Markdown(f"- Sigma R2 fit={diag_s['r2']:.3f}  cal={diag_s_cal['r2']:.3f}"))

fig_s, ax_s = plt.subplots(figsize=(5, 4))
ax_s.scatter(residuals_fit, sigma_pred_fit, alpha=0.3, s=8, c="#2196F3", label="Fit")
ax_s.scatter(np.abs(y_cal - y_cal_pred), sigma_pred_cal, alpha=0.3, s=8,
             c="#FF9800", label="Cal")
ax_s.scatter(np.abs(y_test - y_test_pred), sigma_pred_test, alpha=0.3, s=8,
             c="#00FF00", label="Test")
lim = max(residuals_fit.max(), sigma_pred_fit.max()) * 1.05
ax_s.plot([0, lim], [0, lim], "k--", lw=1)
ax_s.set_xlabel("|y - y_hat|"); ax_s.set_ylabel("sigma_hat(x)")
ax_s.set_title(f"Sigma model (R2_cal={diag_s_cal['r2']:.3f})")
ax_s.legend(fontsize=8); ax_s.grid(True, alpha=0.3)
plt.tight_layout()
fig_s.savefig(out_dir / "sigma_scatter.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_s)

# ==============================================================================
# S5  Calibration scores and q_hat
# ==============================================================================
display(Markdown("## S5  Calibration scores and q_hat"))

eps = 1e-6
sigma_cal_safe  = np.maximum(sigma_pred_cal, eps)
sigma_test_safe = np.maximum(sigma_pred_test, eps)
scores_adaptive = np.abs(y_cal - y_cal_pred) / sigma_cal_safe
scores_plain    = np.abs(y_cal - y_cal_pred)

n_cal   = len(scores_adaptive)
q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
q_adap  = np.quantile(scores_adaptive, q_level)
q_plain = np.quantile(scores_plain,    q_level)

display(Markdown(f"- n_cal={n_cal}  q_level={q_level:.4f}  "
                 f"q_hat_adaptive={q_adap:.4f}  q_hat_plain={q_plain:.4f}"))

fig_q, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(scores_adaptive, bins="auto", density=True, alpha=0.7, color="#2196F3")
ax1.axvline(q_adap, color="red", lw=2, linestyle="--",
            label=f"q_hat={q_adap:.3f}  ({1-alpha:.0%})")
ax1.set_xlabel("|y-y_hat|/sigma  (adaptive)"); ax1.set_ylabel("Density")
ax1.set_title("Adaptive nonconformity scores"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax2.hist(scores_plain, bins="auto", density=True, alpha=0.7, color="#FF9800")
ax2.axvline(q_plain, color="red", lw=2, linestyle="--",
            label=f"q_hat={q_plain:.3f}  ({1-alpha:.0%})")
ax2.set_xlabel("|y-y_hat|  (plain)"); ax2.set_ylabel("Density")
ax2.set_title("Plain nonconformity scores"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.suptitle(f"Calibration score distributions  (alpha={alpha})", fontsize=11)
plt.tight_layout()
fig_q.savefig(out_dir / "calibration_scores.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_q)

# ==============================================================================
# S6  MAPIE: adaptive and plain conformal regressors
# ==============================================================================
display(Markdown("## S6  MAPIE conformal regressors (adaptive and plain)"))

# Adaptive
estimator_a   = ExternalPredictor(y_cal_pred)
mapie_adaptive = SplitConformalRegressor(
    estimator=estimator_a,
    conformity_score=ResidualNormalisedScore(residual_estimator=sigma_model,
                                              prefit=True, sym=True),
    prefit=True, confidence_level=1 - alpha,
)
mapie_adaptive.estimator_ = estimator_a.fit(None, None)
mapie_adaptive.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

estimator_a.y_pred = y_test_pred
y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
lower_a  = y_pis_a[:, 0, 0]; upper_a = y_pis_a[:, 1, 0]
width_a  = upper_a - lower_a
covered_a = (y_test >= lower_a) & (y_test <= upper_a)

# Plain
estimator_p  = ExternalPredictor(y_cal_pred)
mapie_plain  = SplitConformalRegressor(
    estimator=estimator_p,
    conformity_score=AbsoluteConformityScore(),
    prefit=True, confidence_level=1 - alpha,
)
mapie_plain.estimator_ = estimator_p.fit(None, None)
mapie_plain.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

estimator_p.y_pred = y_test_pred
y_pred_p, y_pis_p = mapie_plain.predict_interval(X_test)
lower_p  = y_pis_p[:, 0, 0]; upper_p = y_pis_p[:, 1, 0]
width_p  = upper_p - lower_p
covered_p = (y_test >= lower_p) & (y_test <= upper_p)

display(Markdown(f"- Adaptive: coverage={covered_a.mean():.3f}  mean_width={width_a.mean():.4f}"))
display(Markdown(f"- Plain:    coverage={covered_p.mean():.3f}  mean_width={width_p.mean():.4f}"))

# ── SAVE MODELS to product paths ──────────────────────────────────────────────
def _save_model(path, mapie_obj, sigma_obj, diag_s, variant):
    d = {"mapie": mapie_obj, "sigma_model": sigma_obj, "ncm": ncm,
         "variant": variant, "alpha": alpha,
         "sigma_r2": diag_s["r2"], "sigma_rmse": diag_s["rmse"],
         "sigma_mae": diag_s["mae"], "meta": meta}
    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as fh:
        pickle.dump(d, fh)
    display(Markdown(f"- Model saved: `{path}`"))

_save_model(product["ncmodel_adaptive"], mapie_adaptive, sigma_model, diag_s_cal, "adaptive")
_save_model(product["ncmodel_plain"],    mapie_plain,    sigma_model, diag_s_cal, "plain")

# ==============================================================================
# S7  Coverage and efficiency
# ==============================================================================
display(Markdown("## S7  Coverage and efficiency"))
display(Markdown(f"""
Objective: minimise W = mean interval width,  subject to Cov >= {1-alpha:.2f}.
Both variants achieve the coverage constraint by construction.
"""))

eff_df = pd.DataFrame([
    {"Variant": "Adaptive", "Coverage": covered_a.mean(), f"Target": f">={1-alpha:.2f}",
     "Mean width": width_a.mean(), "Median width": np.median(width_a), "Std width": width_a.std()},
    {"Variant": "Plain",    "Coverage": covered_p.mean(), f"Target": f">={1-alpha:.2f}",
     "Mean width": width_p.mean(), "Median width": np.median(width_p), "Std width": width_p.std()},
])
display(eff_df)

# Width distribution
fig_eff, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(11, 4))
ax_e1.hist(width_a, bins="auto", alpha=0.6, density=True, color="#2196F3",
           label=f"Adaptive  mean={width_a.mean():.3f}")
try:
    ax_e1.hist(width_p, bins="auto", alpha=0.6, density=True, color="#FF9800",
           label=f"Plain     mean={width_p.mean():.3f}")
except:    
    try:
        ax_e1.hist(width_p, bins=5, alpha=0.6, density=True, color="#FF9800",
               label=f"Plain     mean={width_p.mean():.3f}")         
    except:
        pass
ax_e1.axvline(width_a.mean(), color="#2196F3", lw=2, linestyle="--")
ax_e1.axvline(width_p.mean(), color="#FF9800", lw=2, linestyle="--")
ax_e1.set_xlabel("Interval width"); ax_e1.set_ylabel("Density")
ax_e1.set_title("Width distribution: adaptive vs plain")
ax_e1.legend(fontsize=8); ax_e1.grid(True, alpha=0.3)
ax_e2.scatter(width_p - width_a, y_test, c=covered_a.astype(int),
              cmap="RdYlGn", alpha=0.5, s=12)
ax_e2.axvline(0, color="black", lw=1, linestyle="--")
ax_e2.set_xlabel("Width reduction (plain minus adaptive)")
ax_e2.set_ylabel(target_col)
ax_e2.set_title("Width reduction per molecule\n(positive = adaptive narrower)")
ax_e2.grid(True, alpha=0.3)
pct = np.mean(width_a < width_p) * 100
display(Markdown(f"Adaptive is narrower than plain for **{pct:.1f}%** of test molecules."))
plt.tight_layout()
fig_eff.savefig(out_dir / "efficiency_analysis.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_eff)

# ==============================================================================
# S8  Visualise prediction intervals
# ==============================================================================
display(Markdown("## S8  Prediction intervals on test set"))

n_show = min(100, len(test_df))
idx_s  = np.argsort(y_test)[:n_show]
fig_pi, axes_pi = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, lower, upper, pred, cov, title in [
    (axes_pi[0], lower_a[idx_s], upper_a[idx_s], y_pred_a[idx_s], covered_a[idx_s], "Adaptive"),
    (axes_pi[1], lower_p[idx_s], upper_p[idx_s], y_pred_p[idx_s], covered_p[idx_s], "Plain"),
]:
    xp = np.arange(n_show)
    ax.fill_between(xp, lower, upper, alpha=0.25, color="#2196F3", label="Interval")
    ax.scatter(xp[cov],  y_test[idx_s][cov],  c="green", s=15, zorder=3, label="Covered")
    ax.scatter(xp[~cov], y_test[idx_s][~cov], c="red",   s=25, zorder=4, marker="x", label="Missed")
    ax.plot(xp, pred, color="#FF9800", lw=1, alpha=0.7, label="y_hat")
    ax.set_ylabel(target_col)
    ax.set_title(f"{title}  cov={cov.mean():.3f}  width={np.mean(upper-lower):.3f}")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)
axes_pi[1].set_xlabel(f"Test molecules (sorted by true {target_col})")
plt.suptitle(f"Prediction intervals  (alpha={alpha}, n={n_show})", fontsize=11)
plt.tight_layout()
fig_pi.savefig(out_dir / "prediction_intervals.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_pi)

# ==============================================================================
# S9  Exchangeability (KS test)
# ==============================================================================
display(Markdown("## S9  Exchangeability check (KS test)"))
# Compute conformal p-values for each test point
# scores_adaptive is the calibration NCM scores array
scores_test = np.abs(y_test - y_test_pred) / sigma_test_safe
p_values = np.array([
    np.mean(scores_adaptive >= s_i)   # fraction of cal scores >= test score
    for s_i in scores_test
])

#- the two-sample KS test verifies if distributional shift between calibration and test nonconformity scores; 
#- however, the p-value uniformity test may reveal residual non-uniformity, consistent with finite calibration set size / sigma model bias
# and is the more sensitive diagnostic.

scores = {
    "CALIBRATION":   scores_adaptive,
    "TEST":  scores_test,
    "p-values TEST": p_values,
    "uniform" : "uniform"
}
pairs = [
    ("CALIBRATION",   "TEST"),
    ("p-values TEST", "uniform")
]
ks_p = {}
fig_ks, axes = plt.subplots(1,len(pairs), figsize=(12, 4))
for ax, (k1, k2) in zip(axes, pairs):
    s1, s2 = scores[k1], scores[k2]
    ks_stat, _ks_p = ks_2samp(s1, s2)
    ks_p[f"{k1}_{k2}"] = _ks_p
    msg = f"KS p={_ks_p:.4f} ({'OK' if _ks_p > 0.05 else 'WARNING: possible shift'})"    

    ax.hist(s1, bins="auto", density=True, alpha=0.5,
            label=k1, color=colors.get(k1, "gray"))
    if k2 == "uniform":
        ax.axhline(y=1, color=colors.get(k2, "red"), linestyle='--', 
               label=k2, linewidth=2)        
        ax.set_xlabel("p-values")
    else:
        ax.hist(s2, bins="auto", density=True, alpha=0.5,
                label=k2, color=colors.get(k2, "red"))
        ax.set_xlabel("Nonconformity score")

    ax.set_title(f"{k1} vs {k2}\n{msg}")
    
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    display(Markdown(f"- [{k1} - {k2}] {msg}"))
plt.tight_layout()
fig_ks.savefig(out_dir / "exchangeability_ks.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_ks)


# ==============================================================================
# S10  Coverage guarantee sweep across alpha
# ==============================================================================
display(Markdown("## S10  Coverage guarantee across alpha levels"))

alphas_sw = np.arange(0.05, 0.51, 0.05)
cal_scores_a = mapie_adaptive._mapie_regressor.conformity_scores_
cal_scores_a = cal_scores_a[~np.isnan(cal_scores_a)]
cal_scores_p = mapie_plain._mapie_regressor.conformity_scores_
cal_scores_p = cal_scores_p[~np.isnan(cal_scores_p)]

coverages_a, widths_a_sw = [], []
coverages_p, widths_p_sw = [], []
for a in alphas_sw:
    for cs, yp_arr, sig_safe, cov_list, wid_list in [
        (cal_scores_a, y_pred_a, sigma_test_safe, coverages_a, widths_a_sw),
        (cal_scores_p, y_pred_p, None,            coverages_p, widths_p_sw),
    ]:
        _n = len(cs)
        _ql = min(np.ceil((_n + 1) * (1 - a)) / _n, 1.0)
        _q  = np.quantile(cs, _ql)
        if sig_safe is not None:
            _lo = yp_arr - _q * sig_safe;  _hi = yp_arr + _q * sig_safe
        else:
            _lo = yp_arr - _q;             _hi = yp_arr + _q
        cov_list.append(np.mean((y_test >= _lo) & (y_test <= _hi)))
        wid_list.append(np.mean(_hi - _lo))

fig_sw, (ax_sw1, ax_sw2) = plt.subplots(1, 2, figsize=(10, 4))
ax_sw1.plot(1 - alphas_sw, coverages_a, "o-", color="#2196F3", label="Adaptive")
ax_sw1.plot(1 - alphas_sw, coverages_p, "s--", color="#FF9800", label="Plain")
ax_sw1.plot([0.5, 0.95], [0.5, 0.95], "k:", lw=1, label="y=1-alpha (ideal)")
ax_sw1.set_xlabel("Target coverage"); ax_sw1.set_ylabel("Empirical coverage")
ax_sw1.set_title("Coverage guarantee across alpha levels")
ax_sw1.legend(fontsize=8); ax_sw1.grid(True, alpha=0.3)
ax_sw2.plot(1 - alphas_sw, widths_a_sw, "o-", color="#2196F3", label="Adaptive")
ax_sw2.plot(1 - alphas_sw, widths_p_sw, "s--", color="#FF9800", label="Plain")
ax_sw2.set_xlabel("Target coverage"); ax_sw2.set_ylabel("Mean interval width")
ax_sw2.set_title("Efficiency-coverage trade-off"); ax_sw2.legend(fontsize=8)
ax_sw2.grid(True, alpha=0.3)
plt.suptitle("Marginal coverage and efficiency across alpha values", fontsize=11)
plt.tight_layout()
fig_sw.savefig(out_dir / "coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_sw)

# ==============================================================================
# S11  NCM quality comparison
# ==============================================================================
display(Markdown("## S11  NCM quality vs coverage and efficiency"))
display(Markdown(r"""
Coverage is guaranteed regardless of NCM quality.
Efficiency (interval width) depends on NCM quality.
We simulate three quality levels using the same real calibration residuals.
"""))

np.random.seed(42)
_true_res_cal  = np.abs(y_cal - y_cal_pred)
_true_res_test = np.abs(y_test - y_pred_a)

_configs = [
    ("Poor  (R2~0.1)",  1.5,  "#e74c3c"),
    ("Medium (R2~0.5)", 0.75, "#3498db"),
    ("Good  (R2~0.9)",  0.25, "#27ae60"),
]

fig_ncm, axes_ncm = plt.subplots(2, 3, figsize=(14, 8))
ncm_rows = []
for col_idx, (label, noise, color) in enumerate(_configs):
    _sc = np.maximum(_true_res_cal  + np.random.normal(0, noise * _true_res_cal.mean(),
                                                         len(_true_res_cal)),  1e-4)
    _st = np.maximum(_true_res_test + np.random.normal(0, noise * _true_res_test.mean(),
                                                         len(_true_res_test)), 1e-4)
    _r2 = r2_score(_true_res_cal, _sc)
    _ss = _true_res_cal / np.maximum(_sc, eps)
    _n  = len(_ss)
    _ql = min(np.ceil((_n + 1) * (1 - alpha)) / _n, 1.0)
    _q  = np.quantile(_ss, _ql)
    _lo = y_pred_a - _q * np.maximum(_st, eps)
    _hi = y_pred_a + _q * np.maximum(_st, eps)
    _cov = np.mean((y_test >= _lo) & (y_test <= _hi))
    _w   = np.mean(_hi - _lo)
    ncm_rows.append({"Model": label, "R2": round(_r2, 3), "q_hat": round(_q, 3),
                     "Coverage": round(_cov, 3), "Mean_width": round(_w, 3)})

    axes_ncm[0, col_idx].scatter(_true_res_cal, _sc, alpha=0.35, s=8, color=color)
    _lim = max(_true_res_cal.max(), _sc.max()) * 1.05
    axes_ncm[0, col_idx].plot([0, _lim], [0, _lim], "k--", lw=1)
    axes_ncm[0, col_idx].set_xlabel("|y-y_hat| (true)"); axes_ncm[0, col_idx].set_ylabel("sigma_hat")
    axes_ncm[0, col_idx].set_title(f"{label} R2={_r2:.2f}", fontsize=7); axes_ncm[0, col_idx].grid(True, alpha=0.3)

    axes_ncm[1, col_idx].hist(_ss, bins="auto", density=True, alpha=0.65, color=color)
    axes_ncm[1, col_idx].axvline(_q, color="red", lw=2, linestyle="--",
                                  label=f"q={_q:.2f}  cov={_cov:.3f}  W={_w:.3f}")
    axes_ncm[1, col_idx].set_xlabel("|y-y_hat|/sigma"); axes_ncm[1, col_idx].set_ylabel("Density")
    axes_ncm[1, col_idx].set_title(f"cov={_cov:.3f}  W={_w:.3f}", fontsize=7)
    axes_ncm[1, col_idx].legend(fontsize=7); axes_ncm[1, col_idx].grid(True, alpha=0.3)
plt.suptitle(f"NCM quality vs CP outcome  (alpha={alpha})\n"
             "Coverage stable; efficiency improves with better NCM.", fontsize=10)
plt.tight_layout()
fig_ncm.savefig(out_dir / "ncm_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ncm)
display(pd.DataFrame(ncm_rows))

# ==============================================================================
# S12  AD comparison
# ==============================================================================
display(Markdown("## S12  AD comparison: interval width vs applicability domain indices"))

def compare_AD_width(df_test_ad):
    ad_results = []
    if not _available_ad:
        display(Markdown(
            "### Skipped\n\n"
            "No AD columns available in the external file. To enable, add to "
            f"`env.tutorial.yaml` under `{dataset}`:\n\n"
            "```yaml\n  ad_cols: [\"ADI\"]\n  ad_col_directions: [\"similarity\"]\n```"
        ))
    else:
        display(Markdown(f"Comparing CP interval width against: {_available_ad}"))
        display(Markdown("""
    **Interpretation:**
    - rho < 0 (similarity direction): CP and AD agree -- wider intervals outside AD.
    - rho near 0: CP adds information AD misses (e.g. out-of-AD but mechanistically
    reliable predictions ).
    """))
        # now using full test 
        X_test = to_ecfp(df_test_ad); y_test = df_test_ad[target_col].values.astype(float)
        y_test_pred = df_test_ad[pred_col].values.astype(float)
        estimator_a.y_pred = y_test_pred
        y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
        lower_a  = y_pis_a[:, 0, 0]; upper_a = y_pis_a[:, 1, 0]
        width_a  = upper_a - lower_a
        covered_a = (y_test >= lower_a) & (y_test <= upper_a)

        df_test_ad["Width_adaptive"]  = width_a
        df_test_ad["Covered_adaptive"] = covered_a.astype(int)
        for ad_col, ad_dir in zip(_available_ad, ad_col_directions[:len(_available_ad)]):
            display(Markdown(f"### {ad_col}  (direction: {ad_dir})"))
            ad_raw = df_test_ad[ad_col].astype(float)
            #if ad_dir == "distance":
            #    _rng = ad_raw.max() - ad_raw.min()
            #    ad_norm = 1.0 - (ad_raw - ad_raw.min()) / _rng if _rng > 0 \
            #              else pd.Series(np.ones(len(ad_raw)), index=ad_raw.index)
            #else:
            ad_norm = ad_raw.copy()
            mask      = ad_norm.notna() & pd.Series(width_a).notna()
            ad_norm_c = ad_norm[mask].reset_index(drop=True)
            cp_w_c    = pd.Series(width_a)[mask].reset_index(drop=True)
            covered_c = pd.Series(covered_a.astype(int))[mask].reset_index(drop=True)
            rho, pval = spearmanr(ad_norm_c.values, cp_w_c.values)
            rng_b     = np.random.default_rng(42)
            boot_rhos = [spearmanr(
                ad_norm_c.values[_i := rng_b.integers(0, len(ad_norm_c), len(ad_norm_c))],
                cp_w_c.values[_i])[0] for _ in range(1000)]
            rho_lo = np.quantile(boot_rhos, 0.025)
            rho_hi = np.quantile(boot_rhos, 0.975)
            expected = "negative" if ad_dir == "similarity" else "positive"
            display(Markdown(f"- Spearman rho={rho:.3f}  p={pval:.4f}  "
                            f"95% CI [{rho_lo:.3f}, {rho_hi:.3f}]  (expected {expected})"))
            if ad_dir == "similarity":
                in_ad  = cp_w_c[ad_norm_c >= threshold]
                out_ad = cp_w_c[ad_norm_c <  threshold]
            else:  # distance
                in_ad  = cp_w_c[ad_norm_c <= threshold]
                out_ad = cp_w_c[ad_norm_c >  threshold]
            fig_ad, axes_ad = plt.subplots(2, 2, figsize=(13, 10))
            axes_ad[0, 0].scatter(ad_norm_c, cp_w_c, alpha=0.25, s=8, color="#2196F3", rasterized=True)
            try:
                z  = np.polyfit(ad_norm_c, cp_w_c, 1)
                xr = np.linspace(ad_norm_c.min(), ad_norm_c.max(), 100)
                axes_ad[0, 0].plot(xr, np.poly1d(z)(xr), "r-", lw=2, alpha=0.7, label="trend")
            except Exception:
                pass
            axes_ad[0, 0].set_xlabel(f"{ad_col} (normalised)")
            axes_ad[0, 0].set_ylabel("Adaptive interval width")
            axes_ad[0, 0].set_title(f"Spearman rho={rho:.3f}  p={pval:.4f}\n"
                                    f"95% CI [{rho_lo:.3f}, {rho_hi:.3f}]  (expected {expected})", fontsize=9)
            axes_ad[0, 0].legend(fontsize=7); axes_ad[0, 0].grid(True, alpha=0.3)
            if len(in_ad) > 1 and len(out_ad) > 1:
                bp = axes_ad[0, 1].boxplot([in_ad.values, out_ad.values], patch_artist=True,
                                            widths=0.5, medianprops=dict(color="black", lw=2))
                for patch, c_ in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
                    patch.set_facecolor(c_)
                u_stat, u_p = mannwhitneyu(in_ad, out_ad, alternative="two-sided")
                axes_ad[0, 1].text(0.98, 0.97, f"Mann-Whitney p={u_p:.4f}",
                                transform=axes_ad[0, 1].transAxes, ha="right", va="top",
                                fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            axes_ad[0, 1].set_xticklabels([f"In-AD (n={len(in_ad)})", f"Out-of-AD (n={len(out_ad)})"])
            axes_ad[0, 1].set_ylabel("Adaptive interval width")
            axes_ad[0, 1].set_title("In-AD vs Out-of-AD  (threshold=0.5)")
            axes_ad[0, 1].grid(True, alpha=0.3, axis="y")
            _tmp = pd.DataFrame({"AD_norm": ad_norm_c.values, "width": cp_w_c.values,
                                "covered": covered_c.values})
            _tmp["_bin"] = pd.qcut(ad_norm_c, q=n_quantile_bins, labels=False, duplicates="drop")
            strat_rows = []
            for b in sorted(_tmp["_bin"].dropna().unique()):
                _m = _tmp["_bin"] == b
                strat_rows.append({
                    "AD bin": f"Q{int(b)+1} [{ad_norm_c[_m].min():.2f}-{ad_norm_c[_m].max():.2f}]",
                    "n": int(_m.sum()),
                    "Mean width": _tmp.loc[_m, "width"].mean(),
                    "Coverage":   _tmp.loc[_m, "covered"].mean(),
                })
            strat_df = pd.DataFrame(strat_rows)
            display(strat_df)
            x = np.arange(len(strat_df))
            axes_ad[1, 0].bar(x, strat_df["Mean width"], color="#3498db", alpha=0.8, edgecolor="black")
            axes_ad[1, 0].set_xticks(x)
            axes_ad[1, 0].set_xticklabels(strat_df["AD bin"], rotation=30, ha="right", fontsize=8)
            axes_ad[1, 0].set_ylabel("Mean interval width"); axes_ad[1, 0].grid(True, alpha=0.3, axis="y")
            axes_ad[1, 0].set_title("Mean width by AD quintile  (Q1=lowest AD, Q5=highest AD)")
            ax2 = axes_ad[1, 0].twinx()
            ax2.plot(x, strat_df["Coverage"], "rs-", lw=2, ms=8, label="Coverage")
            ax2.axhline(1 - alpha, color="red", linestyle="--", lw=1.5, alpha=0.7,
                        label=f"Target {1-alpha:.0%}")
            ax2.set_ylim(0, 1.05); ax2.set_ylabel("Coverage"); ax2.legend(loc="upper left", fontsize=8)
            axes_ad[1, 1].hist(in_ad.values,  bins="auto", density=True, alpha=0.5,
                            color="#27ae60", label=f"In-AD (n={len(in_ad)})")
            axes_ad[1, 1].hist(out_ad.values, bins="auto", density=True, alpha=0.5,
                            color="#e74c3c", label=f"Out-of-AD (n={len(out_ad)})")
            if len(in_ad):
                axes_ad[1, 1].axvline(in_ad.mean(), color="#27ae60", lw=2, linestyle="--",
                                    label=f"Mean in={in_ad.mean():.3f}")
            if len(out_ad):
                axes_ad[1, 1].axvline(out_ad.mean(), color="#e74c3c", lw=2, linestyle="--",
                                    label=f"Mean out={out_ad.mean():.3f}")
            axes_ad[1, 1].set_xlabel("Adaptive interval width"); axes_ad[1, 1].set_ylabel("Density")
            axes_ad[1, 1].set_title("Width distribution: In-AD vs Out-of-AD")
            axes_ad[1, 1].legend(fontsize=8); axes_ad[1, 1].grid(True, alpha=0.3)
            if ad_dir == "similarity":
                interpretation = "Negative rho = wider intervals outside AD"
            else:
                interpretation = "Positive rho = wider intervals outside AD"        
            plt.suptitle(f"{dataset}: width vs {ad_col}  (rho={rho:.3f})\n {interpretation}", fontsize=10)
            plt.tight_layout()
            _plot_path = out_dir / f"ext_ad_{ad_col}.png"
            fig_ad.savefig(_plot_path, dpi=150, bbox_inches="tight")
            plt.show(); plt.close(fig_ad)
            display(Markdown(f"- Plot: `{_plot_path}`"))
            ad_results.append({
                "dataset": dataset, "ad_col": ad_col, "direction": ad_dir,
                "n": int(mask.sum()), "spearman_rho": round(rho, 4),
                "p_value": round(pval, 6), "rho_CI_lo": round(rho_lo, 4),
                "rho_CI_hi": round(rho_hi, 4),
                "mean_width_in_AD":  round(float(in_ad.mean()),  4) if len(in_ad)  else None,
                "mean_width_out_AD": round(float(out_ad.mean()), 4) if len(out_ad) else None,
            })
    return ad_results

ad_results = compare_AD_width(df_test_ad)

# ==============================================================================
# S13  Save results
# ==============================================================================
result_df = pd.DataFrame({
    "Smiles": test_df["Smiles"].values, "True": y_test,
    "Pred_adaptive": y_pred_a, "Lower_adaptive": lower_a,
    "Upper_adaptive": upper_a, "Width_adaptive": width_a,
    "Covered_adaptive": covered_a.astype(int),
    "Pred_plain": y_pred_p, "Lower_plain": lower_p,
    "Upper_plain": upper_p, "Width_plain": width_p,
    "Covered_plain": covered_p.astype(int),
    "Sigma_pred": sigma_pred_test,
    "Status" : "TEST"
})
metrics_df = pd.DataFrame([
    {"variant": "adaptive", "alpha": alpha, "coverage": covered_a.mean(),
     "mean_width": width_a.mean(), "sigma_r2": diag_s_cal["r2"], "n_test": len(test_df),
     "ks_pvalue": ks_p},
    {"variant": "plain",    "alpha": alpha, "coverage": covered_p.mean(),
     "mean_width": width_p.mean(), "sigma_r2": None, "n_test": len(test_df),
     "ks_pvalue": ks_p},
])
with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics_df.to_excel(w, sheet_name="Metrics",    index=False)
    pd.DataFrame(ncm_rows).to_excel(w, sheet_name="NCM_comparison", index=False)
    if ad_results:
        pd.DataFrame(ad_results).to_excel(w, sheet_name="AD_CP_correlations", index=False)    

display(Markdown(f"## [OK] Regression tutorial ({base_model}) complete."))
display(Markdown(f"- Results         : `{product['data']}`"))
display(Markdown(f"- Model adaptive  : `{product['ncmodel_adaptive']}`"))
display(Markdown(f"- Model plain     : `{product['ncmodel_plain']}`"))
display(Markdown(f"- Plots           : `{out_dir}`"))
