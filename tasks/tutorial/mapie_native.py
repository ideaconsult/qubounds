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
from scipy.stats import ks_2samp
from lightgbm import LGBMRegressor
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore, AbsoluteConformityScore

from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_diagnostic import make_sigma_model, sigma_diagnostics, detect_residual_degeneracy
from tasks.mapie_regression import ExternalPredictor
%matplotlib inline

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

# ==============================================================================
# S0  Resolve upstream and dataset config
# ==============================================================================
tag        = f"tutorial_load_{dataset}"
train_data = upstream["tutorial_load_*"][tag]["train"]
test_data  = upstream["tutorial_load_*"][tag]["test"]
meta_path  = upstream["tutorial_load_*"][tag]["meta"]

with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - REGRESSION (internal model)"))
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

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"- Fit set: {len(fit_df)}  Calibration: {len(cal_df)}  Test: {len(test_df)}"))

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

base_model = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
base_model.fit(X_fit, y_fit)
y_fit_pred  = base_model.predict(X_fit)
y_cal_pred  = base_model.predict(X_cal)
y_test_pred = base_model.predict(X_test)

display(Markdown(f"- Base model R2 on cal set: {r2_score(y_cal, y_cal_pred):.3f}  (honest)"))

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

display(Markdown(f"- Sigma R2 fit={diag_s['r2']:.3f}  cal={diag_s_cal['r2']:.3f}"))

fig_s, ax_s = plt.subplots(figsize=(5, 4))
ax_s.scatter(residuals_fit, sigma_pred_fit, alpha=0.3, s=8, c="#2196F3", label="Fit")
ax_s.scatter(np.abs(y_cal - y_cal_pred), sigma_pred_cal, alpha=0.3, s=8,
             c="#FF9800", label="Cal")
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

scores_test_adap = np.abs(y_test - y_pred_a) / sigma_test_safe
ks_stat, ks_p = ks_2samp(scores_adaptive, scores_test_adap)
fig_ks, ax_ks = plt.subplots(figsize=(6, 4))
ax_ks.hist(scores_adaptive,  bins="auto", density=True, alpha=0.5, label="Calibration", color="#2196F3")
ax_ks.hist(scores_test_adap, bins="auto", density=True, alpha=0.5, label="Test",        color="#FF9800")
ax_ks.set_xlabel("Nonconformity score"); ax_ks.set_ylabel("Density")
ax_ks.set_title(f"Exchangeability: cal vs test  (KS p={ks_p:.4f})")
ax_ks.legend(fontsize=8); ax_ks.grid(True, alpha=0.3)
plt.tight_layout()
fig_ks.savefig(out_dir / "exchangeability_ks.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ks)
display(Markdown(f"- KS p={ks_p:.4f}  "
                 f"({'OK' if ks_p > 0.05 else 'WARNING: possible shift'})"))

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
    axes_ncm[0, col_idx].set_title(f"{label}\nR2={_r2:.2f}"); axes_ncm[0, col_idx].grid(True, alpha=0.3)

    axes_ncm[1, col_idx].hist(_ss, bins="auto", density=True, alpha=0.65, color=color)
    axes_ncm[1, col_idx].axvline(_q, color="red", lw=2, linestyle="--",
                                  label=f"q={_q:.2f}  cov={_cov:.3f}  W={_w:.3f}")
    axes_ncm[1, col_idx].set_xlabel("|y-y_hat|/sigma"); axes_ncm[1, col_idx].set_ylabel("Density")
    axes_ncm[1, col_idx].set_title(f"cov={_cov:.3f}  W={_w:.3f}")
    axes_ncm[1, col_idx].legend(fontsize=7); axes_ncm[1, col_idx].grid(True, alpha=0.3)

plt.suptitle(f"NCM quality vs CP outcome  (alpha={alpha})\n"
             "Coverage stable; efficiency improves with better NCM.", fontsize=10)
plt.tight_layout()
fig_ncm.savefig(out_dir / "ncm_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ncm)
display(pd.DataFrame(ncm_rows))

# ==============================================================================
# S12  Save results
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

display(Markdown("## [OK] Regression tutorial (native) complete."))
display(Markdown(f"- Results         : `{product['data']}`"))
display(Markdown(f"- Model adaptive  : `{product['ncmodel_adaptive']}`"))
display(Markdown(f"- Model plain     : `{product['ncmodel_plain']}`"))
display(Markdown(f"- Plots           : `{out_dir}`"))
