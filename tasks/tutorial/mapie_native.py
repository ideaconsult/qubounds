# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_native.py
-------------------------------
Tutorial: Conformal Prediction for QSAR – Regression
======================================================

This script is designed as an educational walk-through of split conformal
prediction applied to molecular property prediction.  It introduces the core
concepts step-by-step, with print statements that explain what each stage
does and why.

Key concepts illustrated
------------------------
  alpha (α)      : the allowed miscoverage rate (e.g. α=0.1 → 90 % intervals)
  coverage       : the fraction of test samples whose true value falls inside
                   the prediction interval. CP guarantees this ≥ 1-α on average.
  efficiency     : how *narrow* the intervals are. A method that always outputs
                   an infinite interval has perfect coverage but zero efficiency.
  exchangeability: the assumption that calibration and test samples are drawn
                   from the same distribution. Formally required for coverage
                   guarantees; practically tested with a KS test.

Two CP variants are compared
-----------------------------
  Adaptive (sigma-normalised) : interval width is molecule-specific, informed by
                                a sigma model trained on ECFP fingerprints.
                                Wide intervals for uncertain predictions,
                                narrow for confident ones.
  Plain (non-adaptive)        : a single fixed-width interval applied to all
                                molecules regardless of local uncertainty.

Inputs  (ploomber params)
---------
  dataset      : dataset key (matches upstream load task)
  alpha        : miscoverage level, default 0.1
  ncm          : sigma-model key, default "rlgbmecfp"
  cache_path   : ECFP SQLite cache path
  product      : {nb, data, ncmodel_adaptive, ncmodel_plain}
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp

from lightgbm import LGBMRegressor
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore, AbsoluteConformityScore
from sklearn.model_selection import train_test_split

from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_diagnostic import (
    make_sigma_model, sigma_diagnostics, detect_residual_degeneracy,
)
from IPython.display import display, Markdown, HTML
%matplotlib inline


# + tags=["parameters"]
alpha      = 0.1
ncm        = "rlgbmecfp"
cache_path = None
product    = None
upstream   = None
dataset    = None
# -


Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

# ═══════════════════════════════════════════════════════════════════════════════
# §0  Resolve upstream paths
# ═══════════════════════════════════════════════════════════════════════════════
tag       = f"tutorial_load_{dataset}"
train_data = upstream["tutorial_load_*"][tag]["train"]
test_data  = upstream["tutorial_load_*"][tag]["test"]
meta_path  = upstream["tutorial_load_*"][tag]["meta"]

with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]

display(Markdown(f"#  CONFORMAL PREDICTION TUTORIAL – REGRESSION"))
display(Markdown(f"##  Dataset : {meta['dataset']}   Target : {target_col}"))

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Data splits
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §1  Data splits"))

display(Markdown("""
Split conformal prediction requires THREE disjoint sets:

  Fit set        → train the base (QSAR) model
  Calibration set→ compute nonconformity scores; determine the quantile
                   threshold that controls coverage.  This set is CONSUMED
                   by the conformal procedure and not used for test evaluation.
  Test set       → evaluate coverage and efficiency on fresh molecules.

The calibration set must be EXCHANGEABLE with the test set:
informally, both should look like independent draws from the same
distribution.  We verify this with a Kolmogorov-Smirnov test later.
"""))

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"- Fit set          : {len(fit_df):>5d} molecules"))
display(Markdown(f"- Calibration set  : {len(cal_df):>5d} molecules"))
display(Markdown(f"- Test set         : {len(test_df):>5d} molecules"))

# ═══════════════════════════════════════════════════════════════════════════════
# §2  Molecular fingerprints
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §2  Molecular fingerprints (ECFP4, 2048 bits)"))
display(Markdown("""
Extended Connectivity Fingerprints (ECFP) encode local chemical
neighbourhoods as bit vectors.  We use radius 4, 2048 bits (ECFP4).
Results are cached in SQLite to avoid recomputation.
"""))

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df)
display(Markdown(f"- X_fit  : {X_fit.shape}"))
X_cal  = to_ecfp(cal_df)
display(Markdown(f"- X_cal  : {X_cal.shape}"))
X_test = to_ecfp(test_df)
display(Markdown(f"- X_test : {X_test.shape}"))
y_fit = fit_df[target_col].values.astype(float)
y_cal = cal_df[target_col].values.astype(float)
y_test = test_df[target_col].values.astype(float)

# ═══════════════════════════════════════════════════════════════════════════════
# §3  Base QSAR model
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §3  Base QSAR model (LightGBM)"))
display(Markdown("""
The base model is a LightGBM regressor with a Huber loss (robust to outliers).
It is trained ONLY on the fit set; the calibration set is withheld.
"""))

base_model = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
base_model.fit(X_fit, y_fit)
y_fit_pred = base_model.predict(X_fit)
y_cal_pred = base_model.predict(X_cal)

from sklearn.metrics import r2_score
r2_fit = r2_score(y_fit, y_fit_pred)
r2_cal = r2_score(y_cal, y_cal_pred)
display(Markdown(f"- Base model R² on fit set  : {r2_fit:.3f}"))
display(Markdown(f"- Base model R² on cal set  : {r2_cal:.3f}  (honest estimate)"))

# ═══════════════════════════════════════════════════════════════════════════════
# §4  Sigma (nonconformity) model
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §4  Sigma model – predicting local error magnitude"))
display(Markdown(f"""
The sigma model predicts |y - ŷ| for each molecule from its ECFP fingerprint.
This gives an *expected residual* σ̂(x) that varies across chemical space.

  - σ̂ small → the QSAR model is typically accurate here  → narrow interval
  - σ̂ large → the QSAR model is typically uncertain here  → wide interval

σ̂ is trained on the fit-set residuals using: {ncm}
It is evaluated on the calibration set to assess generalisation.
"""))

residuals_fit = np.abs(y_fit - y_fit_pred)
use_eps, diag = detect_residual_degeneracy(residuals_fit, y_fit)

sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)

sigma_fit_pred = sigma_model.predict(X_fit)
sigma_cal_pred = sigma_model.predict(X_cal)
sigma_test_pred = sigma_model.predict(X_test)

diag_sigma = sigma_diagnostics(residuals_fit, sigma_fit_pred)
diag_sigma_cal = sigma_diagnostics(np.abs(y_cal - y_cal_pred), sigma_cal_pred)

display(Markdown(f"- Sigma R² on fit set  : {diag_sigma['r2']:.3f}"))
display(Markdown(f"- Sigma R² on cal set  : {diag_sigma_cal['r2']:.3f}"))
display(Markdown("""
  Note: a low R² for the sigma model does not invalidate CP coverage.
  Coverage is guaranteed by the calibration quantile regardless of sigma model
  quality.  However, a better sigma model → narrower intervals (better efficiency).
"""))

# Sigma model scatter plot
fig_s, ax_s = plt.subplots(figsize=(5, 4))
ax_s.scatter(residuals_fit, sigma_fit_pred, alpha=0.3, s=8, label="Fit set")
ax_s.scatter(np.abs(y_cal - y_cal_pred), sigma_cal_pred, alpha=0.3, s=8, c="orange", label="Cal set")
lim = max(residuals_fit.max(), sigma_fit_pred.max()) * 1.05
ax_s.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect σ")
ax_s.set_xlabel("Actual residual |y - ŷ|")
ax_s.set_ylabel("Predicted σ̂(x)")
ax_s.set_title(f"Sigma model: predicted vs actual residual\n({ncm}, R²_cal={diag_sigma_cal['r2']:.3f})")
ax_s.legend(fontsize=8)
ax_s.grid(True, alpha=0.3)
plt.tight_layout()
fig_s.savefig(out_dir / "sigma_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_s)

# ═══════════════════════════════════════════════════════════════════════════════
# §5  Nonconformity scores and the calibration quantile
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §5  Nonconformity scores and the calibration quantile"))
display(Markdown(f"""
For each calibration molecule i, the nonconformity score is:

  - ADAPTIVE  : s_i = |y_i - ŷ_i| / σ̂(x_i)   (normalized by local uncertainty)
  - PLAIN     : s_i = |y_i - ŷ_i|              (raw residual)

The (1-α) quantile of {{s_1, ..., s_n}} is computed.  During prediction,
the interval for a new molecule x is:

  - ADAPTIVE  : ŷ ± q̂ · σ̂(x)                 (molecule-specific width)
  - PLAIN     : ŷ ± q̂                          (fixed width for all molecules)

Here α = {alpha}, so we target {(1-alpha):.0%} coverage.
"""))

# Compute adaptive nonconformity scores on calibration set
eps = 1e-6
sigma_cal_safe = np.maximum(sigma_cal_pred, eps)
scores_adaptive = np.abs(y_cal - y_cal_pred) / sigma_cal_safe
scores_plain = np.abs(y_cal - y_cal_pred)

n_cal = len(scores_adaptive)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_level = min(q_level, 1.0)

q_adaptive = np.quantile(scores_adaptive, q_level)
q_plain = np.quantile(scores_plain,    q_level)

display(Markdown(f"- Calibration set size (n) : {n_cal}"))
display(Markdown(f"- Quantile level used      : {q_level:.4f}  [= ceil((n+1)(1-α)) / n]"))
display(Markdown(f"- Adaptive quantile q̂     : {q_adaptive:.4f}  (dimensionless, in σ units)"))
display(Markdown(f"- Plain quantile q̂        : {q_plain:.4f}  (in target units)"))

# Plot nonconformity score distribution
fig_q, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(scores_adaptive, bins=40, density=True, alpha=0.7, color="#2196F3", label="Cal scores")
ax1.axvline(q_adaptive, color="red", lw=2, linestyle="--",
            label=f"q̂={q_adaptive:.3f}  ({(1-alpha):.0%} level)")
ax1.set_xlabel("Nonconformity score  |y - ŷ| / σ̂(x)")
ax1.set_ylabel("Density")
ax1.set_title("Adaptive: normalised scores")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax2.hist(scores_plain, bins=40, density=True, alpha=0.7, color="#FF9800", label="Cal scores")
ax2.axvline(q_plain, color="red", lw=2, linestyle="--",
            label=f"q̂={q_plain:.3f}  ({(1-alpha):.0%} level)")
ax2.set_xlabel("Nonconformity score  |y - ŷ|")
ax2.set_ylabel("Density")
ax2.set_title("Plain: raw residual scores")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.suptitle(f"Calibration nonconformity score distributions  (α={alpha})", fontsize=11)
plt.tight_layout()
fig_q.savefig(out_dir / "calibration_scores.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_q)

# ═══════════════════════════════════════════════════════════════════════════════
# §6  MAPIE conformal predictors
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §6  Fitting MAPIE conformal predictors"))
display(Markdown("""
MAPIE (Model Agnostic Prediction Interval Estimator) implements the
conformal procedure.  We use SplitConformalRegressor with prefit=False,
meaning MAPIE trains the base model on X_fit and conformises on X_cal.
"""))

# ── Adaptive
base_adaptive = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
conformity_adaptive = ResidualNormalisedScore(
    residual_estimator=sigma_model, prefit=True, sym=True)
mapie_adaptive = SplitConformalRegressor(
    estimator=base_adaptive,
    conformity_score=conformity_adaptive,
    prefit=False,
    confidence_level=1 - alpha)
mapie_adaptive.fit(X_fit, y_fit)
mapie_adaptive.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

# ── Plain
base_plain = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
mapie_plain = SplitConformalRegressor(
    estimator=base_plain,
    conformity_score=AbsoluteConformityScore(),
    prefit=False,
    confidence_level=1 - alpha)
mapie_plain.fit(X_fit, y_fit)
mapie_plain.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

display(Markdown("- Both conformal predictors fitted and conformalized."))

# ═══════════════════════════════════════════════════════════════════════════════
# §7  Test set predictions and coverage
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §7  Test set predictions: coverage and efficiency"))
display(Markdown(f"""
- Coverage (validity) : fraction of test molecules whose true value falls
                      inside the predicted interval. CP guarantees ≥ {1-alpha:.0%}.

- Efficiency          : how narrow the intervals are. Measured as mean interval
                      width (smaller = more informative). The ideal method
                      achieves target coverage with the smallest possible width.
"""))

y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
lower_a = y_pis_a[:, 0, 0]
upper_a = y_pis_a[:, 1, 0]
width_a = upper_a - lower_a
covered_a = (y_test >= lower_a) & (y_test <= upper_a)

y_pred_p, y_pis_p = mapie_plain.predict_interval(X_test)
lower_p = y_pis_p[:, 0, 0]
upper_p = y_pis_p[:, 1, 0]
width_p = upper_p - lower_p
covered_p = (y_test >= lower_p) & (y_test <= upper_p)

display(Markdown(f"-  {'Variant':<22}  {'Coverage':>10}  {'Mean width':>12}  {'Median width':>13}"))
display(Markdown(f"-  {'Adaptive (sigma)':22}  {covered_a.mean():>10.3f}  {width_a.mean():>12.4f}  {np.median(width_a):>13.4f}"))
display(Markdown(f"-  {'Plain (fixed)':22}  {covered_p.mean():>10.3f}  {width_p.mean():>12.4f}  {np.median(width_p):>13.4f}"))
display(Markdown(f"-  Target coverage : {1-alpha:.3f}  (α = {alpha})"))

# ═══════════════════════════════════════════════════════════════════════════════
# §8  Exchangeability check (KS test)
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §8  Exchangeability check (Kolmogorov-Smirnov test)"))
display(Markdown("""
CP coverage guarantees require that calibration and test scores are exchangeable
(informally: look like independent samples from the same distribution).

We check this by computing nonconformity scores on the test set and comparing
their distribution to the calibration scores using the KS test.

  - KS p-value >> 0.05 → no evidence of distributional shift → guarantee holds
  - KS p-value << 0.05 → possible shift → coverage may deviate from guarantee
"""))

sigma_test_safe = np.maximum(sigma_test_pred, eps)
scores_test_adaptive = np.abs(y_test - y_pred_a) / sigma_test_safe
scores_test_plain = np.abs(y_test - y_pred_p)

ks_a_stat, ks_a_p = ks_2samp(scores_adaptive, scores_test_adaptive)
ks_p_stat, ks_p_p = ks_2samp(scores_plain,    scores_test_plain)

display(Markdown(f"- Adaptive  KS stat={ks_a_stat:.4f}  p={ks_a_p:.4f}  →  "
      f"{'OK exchangeable' if ks_a_p > 0.05 else 'WARNING possible shift'}"))
display(Markdown(f"- Plain     KS stat={ks_p_stat:.4f}  p={ks_p_p:.4f}  →  "
      f"{'OK exchangeable' if ks_p_p > 0.05 else 'WARNING possible shift'}"))

# KS distribution overlay
fig_ks, (ax_ka, ax_kp) = plt.subplots(1, 2, figsize=(10, 4))
for ax, sc_cal, sc_test, ks_p_val, title in [
    (ax_ka, scores_adaptive, scores_test_adaptive, ks_a_p, "Adaptive"),
    (ax_kp, scores_plain,    scores_test_plain,    ks_p_p, "Plain"),
]:
    ax.hist(sc_cal,  bins=40, density=True, alpha=0.5, label="Calibration", color="#2196F3")
    ax.hist(sc_test, bins=40, density=True, alpha=0.5, label="Test",        color="#FF9800")
    ax.set_xlabel("Nonconformity score")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}: Cal vs Test score distributions\n(KS p={ks_p_val:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Exchangeability check: calibration vs test nonconformity scores", fontsize=11)
plt.tight_layout()
fig_ks.savefig(out_dir / "exchangeability_ks.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_ks)

# ═══════════════════════════════════════════════════════════════════════════════
# §9  Visualise prediction intervals on test set
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §9  Visualising prediction intervals"))

n_show = min(100, len(test_df))
idx_sorted = np.argsort(y_test)[:n_show]

fig_pi, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, lower, upper, pred, covered, title in [
    (axes[0], lower_a[idx_sorted], upper_a[idx_sorted],
     y_pred_a[idx_sorted], covered_a[idx_sorted], "Adaptive (molecule-specific width)"),
    (axes[1], lower_p[idx_sorted], upper_p[idx_sorted],
     y_pred_p[idx_sorted], covered_p[idx_sorted], "Plain (fixed width)"),
]:
    x_pos = np.arange(n_show)
    ax.fill_between(x_pos, lower, upper, alpha=0.25, color="#2196F3",
                    label="Prediction interval")
    ax.scatter(x_pos[covered],  y_test[idx_sorted][covered],
               c="green", s=15, zorder=3, label="Covered")
    ax.scatter(x_pos[~covered], y_test[idx_sorted][~covered],
               c="red",   s=25, zorder=4, marker="x", label="Not covered")
    ax.plot(x_pos, pred, color="#FF9800", lw=1, alpha=0.7, label="ŷ")
    ax.set_ylabel(target_col)
    ax.set_title(f"{title}   coverage={covered.mean():.3f}  "
                 f"mean width={np.mean(upper-lower):.3f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
axes[1].set_xlabel(f"Test molecules (sorted by true {target_col})")
plt.suptitle(f"Prediction intervals on test set  (α={alpha}, n={n_show})", fontsize=11)
plt.tight_layout()
fig_pi.savefig(out_dir / "prediction_intervals.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_pi)

# Interval width vs sigma
fig_sw, ax_sw = plt.subplots(figsize=(5, 4))
ax_sw.scatter(sigma_test_pred[idx_sorted], width_a[idx_sorted],
              alpha=0.4, s=10, color="#2196F3")
ax_sw.set_xlabel("σ̂(x)  (sigma model prediction)")
ax_sw.set_ylabel("Adaptive interval width")
ax_sw.set_title("Interval width is proportional to σ̂(x)")
ax_sw.grid(True, alpha=0.3)
plt.tight_layout()
fig_sw.savefig(out_dir / "width_vs_sigma.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_sw)

# ═══════════════════════════════════════════════════════════════════════════════
# §10  Marginal coverage vs alpha sweep
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §10  Marginal coverage guarantee: sweeping α"))
display(Markdown("""
A key property of split CP is that empirical coverage ≥ 1-α holds for ANY α,
provided exchangeability holds.  We verify this by re-computing coverage across
a range of α values using the already-fitted conformity scores.
"""))

alphas = np.arange(0.05, 0.51, 0.05)
coverages_a = []
coverages_p = []
widths_a_mean = []
widths_p_mean = []

# Retrieve calibration conformity scores stored by MAPIE
scores_mapie_a = mapie_adaptive._mapie_regressor.conformity_scores_
scores_mapie_a = scores_mapie_a[~np.isnan(scores_mapie_a)]
scores_mapie_p = mapie_plain._mapie_regressor.conformity_scores_
scores_mapie_p = scores_mapie_p[~np.isnan(scores_mapie_p)]

for a in alphas:
    n = len(scores_mapie_a)
    ql = min(np.ceil((n + 1) * (1 - a)) / n, 1.0)

    qa = np.quantile(scores_mapie_a, ql)
    qp = np.quantile(scores_mapie_p, ql)

    lo_a = y_pred_a - qa * sigma_test_safe
    hi_a = y_pred_a + qa * sigma_test_safe
    coverages_a.append(np.mean((y_test >= lo_a) & (y_test <= hi_a)))
    widths_a_mean.append(np.mean(hi_a - lo_a))

    lo_p = y_pred_p - qp
    hi_p = y_pred_p + qp
    coverages_p.append(np.mean((y_test >= lo_p) & (y_test <= hi_p)))
    widths_p_mean.append(np.mean(hi_p - lo_p))

fig_alpha, (ax_cov, ax_wid) = plt.subplots(1, 2, figsize=(10, 4))
ax_cov.plot(1 - alphas, coverages_a, "o-", label="Adaptive", color="#2196F3")
ax_cov.plot(1 - alphas, coverages_p, "s--", label="Plain",    color="#FF9800")
ax_cov.plot([1 - alphas.max(), 1 - alphas.min()],
            [1 - alphas.max(), 1 - alphas.min()], "k:", lw=1, label="y = 1-α (ideal)")
ax_cov.set_xlabel("Target coverage  (1 - α)")
ax_cov.set_ylabel("Empirical coverage")
ax_cov.set_title("Coverage guarantee across α levels")
ax_cov.legend(fontsize=8)
ax_cov.grid(True, alpha=0.3)
ax_wid.plot(1 - alphas, widths_a_mean, "o-", label="Adaptive", color="#2196F3")
ax_wid.plot(1 - alphas, widths_p_mean, "s--", label="Plain",    color="#FF9800")
ax_wid.set_xlabel("Target coverage  (1 - α)")
ax_wid.set_ylabel("Mean interval width")
ax_wid.set_title("Interval width vs coverage level\n(efficiency trade-off)")
ax_wid.legend(fontsize=8)
ax_wid.grid(True, alpha=0.3)
plt.suptitle("Marginal coverage and efficiency across α values", fontsize=11)
plt.tight_layout()
fig_alpha.savefig(out_dir / "coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_alpha)

for a, ca, cp in zip(alphas, coverages_a, coverages_p):
    display(Markdown(f"- α={a:.2f}  target={1-a:.2f}  "
          f"adaptive={ca:.3f}  plain={cp:.3f}"))

# ═══════════════════════════════════════════════════════════════════════════════
# §11  Save models and results
# ═══════════════════════════════════════════════════════════════════════════════
def _save(path, mapie_obj, sigma_obj, diag_s, variant):
    d = {
        "mapie": mapie_obj, "sigma_model": sigma_obj,
        "ncm": ncm, "variant": variant, "alpha": alpha,
        "sigma_r2": diag_s["r2"], "sigma_rmse": diag_s["rmse"],
        "sigma_mae": diag_s["mae"], "meta": meta,
    }
    with open(path, "wb") as fh:
        pickle.dump(d, fh)


_save(product["ncmodel_adaptive"], mapie_adaptive, sigma_model, diag_sigma, "adaptive")
_save(product["ncmodel_plain"],    mapie_plain,    sigma_model, diag_sigma, "plain")
result_df = pd.DataFrame({
    "Smiles": test_df["Smiles"].values,
    "True": y_test,
    "Pred_adaptive": y_pred_a,
    "Lower_adaptive": lower_a, "Upper_adaptive": upper_a,
    "Width_adaptive": width_a, "Covered_adaptive": covered_a.astype(int),
    "Sigma_pred": sigma_test_pred,
    "Pred_plain": y_pred_p,
    "Lower_plain": lower_p, "Upper_plain": upper_p,
    "Width_plain": width_p, "Covered_plain": covered_p.astype(int),
})

metrics = pd.DataFrame([
    {"variant": "adaptive", "alpha": alpha,
     "coverage": covered_a.mean(), "mean_width": width_a.mean(),
     "median_width": np.median(width_a),
     "sigma_r2": diag_sigma_cal["r2"], "n_test": len(test_df),
     "ks_stat": ks_a_stat, "ks_pvalue": ks_a_p},
    {"variant": "plain", "alpha": alpha,
     "coverage": covered_p.mean(), "mean_width": width_p.mean(),
     "median_width": np.median(width_p),
     "sigma_r2": None, "n_test": len(test_df),
     "ks_stat": ks_p_stat, "ks_pvalue": ks_p_p},
])

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics.to_excel(w, sheet_name="Metrics", index=False)

display(Markdown("## [OK] Tutorial complete."))
display(Markdown(f"- Results → {product['data']}"))
display(Markdown(f"- Plots   → {out_dir}"))
