# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_external.py
----------------------------------
Tutorial: Conformal Prediction with External Predictions + AD Comparison
=========================================================================

Mirrors the VEGA/paper pipeline exactly:
  - Predictions are read from an external file (pred_file in dataset_config).
  - The conformal wrapper treats the model as a black box (ExternalPredictor).
  - The same sigma model architecture as mapie_native.py is used.
  - If the external file also contains AD index columns, the AD vs CP
    comparison is performed at the end of this task.

SKIP BEHAVIOUR:
  If pred_file is null or the file does not exist, the task writes empty
  product files and exits immediately with an informative message.
  No "fallback to internal model" -- the external task is meaningless
  without real external predictions.

TRAIN/TEST SPLIT:
  If split_col is configured (e.g. "Split"), rows where split_col == split_train_value
  are used as calibration set, rows where split_col == split_test_value as test set.
  If split_col is absent or not configured, the task falls back to matching
  calibration/test by Smiles against the internally-split sets.

Inputs (ploomber params) - all sourced from pipeline.tutorial.yaml
---------
  dataset        : dataset key
  ncm            : sigma-model key
  alpha          : miscoverage level
  cache_path     : ECFP SQLite cache path
  dataset_config : full dict from env.tutorial.yaml
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
threshold = 0.5
# -

import json
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
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

cfg               = dataset_config.get(dataset, {}) if isinstance(dataset_config, dict) else {}
pred_file         = cfg.get("pred_file",           None)
pred_col          = cfg.get("pred_col",            "Pred")
smiles_col_ext    = cfg.get("smiles_col",          "Smiles")
split_col         = cfg.get("split_col",           None)
split_train_value = cfg.get("split_train_value",   "train")
split_test_value  = cfg.get("split_test_value",    "test")
ad_cols           = cfg.get("ad_cols",             [])
ad_col_directions = cfg.get("ad_col_directions",   [])
n_quantile_bins   = int(cfg.get("n_quantile_bins", 5))

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - REGRESSION (external model)"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))
display(Markdown(f"""
Configuration from `env.tutorial.yaml[{dataset}]`:
- pred_file         : `{pred_file}`
- pred_col          : `{pred_col}`
- smiles_col        : `{smiles_col_ext}`
- split_col         : `{split_col}`  (train=`{split_train_value}` / test=`{split_test_value}`)
- ad_cols           : `{ad_cols}`
- ad_col_directions : `{ad_col_directions}`
"""))

# ==============================================================================
# EARLY EXIT when no pred_file is configured
# ==============================================================================
if not pred_file or not Path(pred_file).exists():
    _reason = "not configured" if not pred_file else f"not found: `{pred_file}`"
    display(Markdown(f"""
## Skipped: no external predictions available

pred_file is {_reason}.

To enable this task, add to `env.tutorial.yaml` under `{dataset}`:

```yaml
  pred_file: "path/to/predictions.xlsx"
  pred_col:  "Pred"
  split_col: "Split"           # optional: column indicating train/test rows
  split_train_value: "train"   # optional: value marking calibration rows
  split_test_value:  "test"    # optional: value marking test rows
  ad_cols: ["ADI"]             # optional: AD index column names
  ad_col_directions: ["similarity"]
```
"""))
    # Write empty product files so Ploomber sees completed products
    pd.DataFrame().to_excel(str(product["data"]))
    for _key in ("ncmodel_adaptive", "ncmodel_plain"):
        _path = Path(str(product[_key]))
        _path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(_path), "wb") as _fh:
            pickle.dump({"skipped": True, "reason": _reason}, _fh)
    sys.exit(0)

# ==============================================================================
# S1  Load external file and resolve train/test split
# ==============================================================================
display(Markdown("## S1  Load external predictions file"))

_p   = Path(pred_file)
_ext = pd.read_excel(_p) if _p.suffix in {".xlsx", ".xls"} else pd.read_csv(_p)

# Normalise SMILES column
if smiles_col_ext in _ext.columns and smiles_col_ext != "Smiles":
    _ext = _ext.rename(columns={smiles_col_ext: "Smiles"})

if pred_col not in _ext.columns:
    display(Markdown(
        f"ERROR: prediction column `{pred_col}` not found.\n"
        f"Available columns: {_ext.columns.tolist()}"))
    sys.exit(1)

display(Markdown(f"- Loaded {len(_ext)} rows  columns: {_ext.columns.tolist()}"))

if split_col and split_col in _ext.columns:
    display(Markdown(f"- Using `{split_col}` column to split calibration / test rows."))
    _ext_cal  = _ext[_ext[split_col].astype(str).str.lower() == split_train_value.lower()].copy()
    vals = split_test_value
    if isinstance(vals, str):
        vals = [vals]
    else:
        vals = list(vals)
    vals = [v.lower() for v in vals]
    mask = _ext[split_col].astype(str).str.lower().isin(vals)
    _ext_test = _ext[mask].copy()
    display(Markdown(f"  - Calibration rows : {len(_ext_cal)}"))
    display(Markdown(f"  - Test rows        : {len(_ext_test)}"))
else:
    display(Markdown(
        f"- No `split_col` configured. Using all rows as test set; "
        f"calibration will be matched against the internally-split cal set by Smiles."))
    _ext_cal  = _ext.copy()
    _ext_test = _ext.copy()

# ==============================================================================
# S2  Internal fingerprints + sigma model
#     Sigma model is always trained internally on LightGBM residuals.
#     External model is used ONLY for predictions at test/cal time.
# ==============================================================================
display(Markdown(f"## S2  Fingerprints and sigma model ({ncm})"))
display(Markdown("""
Split CP requires calibration residuals to train the sigma model.
Since the external model may not provide residuals, we train an internal
LightGBM model on the fit set to generate them.
The sigma model then estimates expected residual magnitudes for any molecule
using ECFP fingerprints -- this is independent of which model made the predictions.
"""))

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df);  y_fit  = fit_df[target_col].values.astype(float)
X_cal  = to_ecfp(cal_df);  y_cal  = cal_df[target_col].values.astype(float)
X_test = to_ecfp(test_df); y_test = test_df[target_col].values.astype(float)

_internal = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
_internal.fit(X_fit, y_fit)
residuals_fit = np.abs(y_fit - _internal.predict(X_fit))

_, _ = detect_residual_degeneracy(residuals_fit, y_fit)
sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)

sigma_pred_cal  = sigma_model.predict(X_cal)
sigma_pred_test = sigma_model.predict(X_test)
diag_s_fit = sigma_diagnostics(residuals_fit, sigma_model.predict(X_fit))
diag_s_cal = sigma_diagnostics(np.abs(y_cal - _internal.predict(X_cal)), sigma_pred_cal)
display(Markdown(f"- Sigma R2 fit={diag_s_fit['r2']:.3f}  cal={diag_s_cal['r2']:.3f}"))

eps             = 1e-6
sigma_cal_safe  = np.maximum(sigma_pred_cal,  eps)
sigma_test_safe = np.maximum(sigma_pred_test, eps)

# ==============================================================================
# S3  Align external predictions with calibration and test sets
# ==============================================================================
display(Markdown("## S3  Align external predictions with calibration / test sets"))

_ext_cal_idx  = _ext_cal.set_index("Smiles")[pred_col]
_ext_test_idx = _ext_test.set_index("Smiles")[pred_col]

y_cal_pred_ext  = cal_df["Smiles"].map(_ext_cal_idx).values.astype(float)
y_test_pred_ext = test_df["Smiles"].map(_ext_test_idx).values.astype(float)

n_cal_matched  = (~np.isnan(y_cal_pred_ext)).sum()
n_test_matched = (~np.isnan(y_test_pred_ext)).sum()
display(Markdown(f"- External predictions matched: cal={n_cal_matched}/{len(cal_df)}  "
                 f"test={n_test_matched}/{len(test_df)}"))

if n_cal_matched == 0 or n_test_matched == 0:
    display(Markdown(
        "ERROR: No molecules matched between external file and internal split sets.\n"
        "Check that `smiles_col` in env.tutorial.yaml matches the SMILES column in the file,\n"
        "and that `split_col` (if configured) correctly separates train/test rows."))
    sys.exit(1)

# Fill unmatched molecules with internal predictions as fallback
y_cal_fallback  = _internal.predict(X_cal)
y_test_fallback = _internal.predict(X_test)
y_cal_pred_ext  = np.where(np.isnan(y_cal_pred_ext),  y_cal_fallback,  y_cal_pred_ext)
y_test_pred_ext = np.where(np.isnan(y_test_pred_ext), y_test_fallback, y_test_pred_ext)

# Carry AD columns into test dataframe
df_test_ad = test_df.copy()
_available_ad = []
if ad_cols:
    for _ac in ad_cols:
        if _ac in _ext_test.columns:
            df_test_ad[_ac] = test_df["Smiles"].map(
                _ext_test.set_index("Smiles")[_ac]).values
            _available_ad.append(_ac)
        else:
            display(Markdown(f"- WARNING: AD column `{_ac}` not found in external file."))
    if _available_ad:
        display(Markdown(f"- AD columns loaded: {_available_ad}"))

# ==============================================================================
# S4  MAPIE conformal regressors
# ==============================================================================
display(Markdown("## S4  MAPIE conformal regressors (adaptive and plain)"))
display(Markdown("""
ExternalPredictor wraps the pre-computed predictions as a sklearn-compatible
estimator. MAPIE sees only the prediction values, not the model itself.
This is the exact mechanism used for VEGA models in the paper.
"""))

estimator_a    = ExternalPredictor(y_cal_pred_ext)
mapie_adaptive = SplitConformalRegressor(
    estimator=estimator_a,
    conformity_score=ResidualNormalisedScore(
        residual_estimator=sigma_model, prefit=True, sym=True),
    prefit=True, confidence_level=1 - alpha)
mapie_adaptive.estimator_ = estimator_a.fit(None, None)
mapie_adaptive.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)
estimator_a.y_pred = y_test_pred_ext
y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
lower_a   = y_pis_a[:, 0, 0]; upper_a = y_pis_a[:, 1, 0]
width_a   = upper_a - lower_a; covered_a = (y_test >= lower_a) & (y_test <= upper_a)

estimator_p  = ExternalPredictor(y_cal_pred_ext)
mapie_plain  = SplitConformalRegressor(
    estimator=estimator_p,
    conformity_score=AbsoluteConformityScore(),
    prefit=True, confidence_level=1 - alpha)
mapie_plain.estimator_ = estimator_p.fit(None, None)
mapie_plain.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)
estimator_p.y_pred = y_test_pred_ext
y_pred_p, y_pis_p = mapie_plain.predict_interval(X_test)
lower_p   = y_pis_p[:, 0, 0]; upper_p = y_pis_p[:, 1, 0]
width_p   = upper_p - lower_p; covered_p = (y_test >= lower_p) & (y_test <= upper_p)

# Save models
def _save_model(path, mapie_obj, sigma_obj, diag_s, variant):
    d = {"mapie": mapie_obj, "sigma_model": sigma_obj, "ncm": ncm,
         "variant": variant, "alpha": alpha,
         "sigma_r2": diag_s["r2"], "sigma_rmse": diag_s["rmse"],
         "sigma_mae": diag_s["mae"], "meta": meta}
    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as fh:
        pickle.dump(d, fh)
    display(Markdown(f"- Model saved: `{path}`"))

_save_model(product["ncmodel_adaptive"], mapie_adaptive, sigma_model, diag_s_cal, "adaptive_external")
_save_model(product["ncmodel_plain"],    mapie_plain,    sigma_model, diag_s_cal, "plain_external")

# ==============================================================================
# S5  Coverage and efficiency
# ==============================================================================
display(Markdown("## S5  Coverage and efficiency"))
display(Markdown(f"Objective: minimise W = mean interval width,  subject to Cov >= {1-alpha:.2f}."))

eff_df = pd.DataFrame([
    {"Variant": "Adaptive", "Coverage": covered_a.mean(), "Target": f">={1-alpha:.2f}",
     "Mean width": width_a.mean(), "Median width": np.median(width_a), "Std width": width_a.std()},
    {"Variant": "Plain",    "Coverage": covered_p.mean(), "Target": f">={1-alpha:.2f}",
     "Mean width": width_p.mean(), "Median width": np.median(width_p), "Std width": width_p.std()},
])
display(eff_df)

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
ax_e2.set_title("Width reduction per molecule")
ax_e2.grid(True, alpha=0.3)
pct = np.mean(width_a < width_p) * 100
display(Markdown(f"Adaptive narrower than plain for **{pct:.1f}%** of test molecules."))
plt.tight_layout()
fig_eff.savefig(out_dir / "ext_efficiency_analysis.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_eff)

# ==============================================================================
# S6  Prediction intervals
# ==============================================================================
display(Markdown("## S6  Prediction intervals on test set"))

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
    ax.plot(xp, pred, color="#FF9800", lw=1, alpha=0.7, label="y_hat (external)")
    ax.set_ylabel(target_col)
    ax.set_title(f"{title}  cov={cov.mean():.3f}  width={np.mean(upper-lower):.3f}")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)
axes_pi[1].set_xlabel(f"Test molecules (sorted by true {target_col})")
plt.suptitle(f"External model prediction intervals  (alpha={alpha}, n={n_show})", fontsize=11)
plt.tight_layout()
fig_pi.savefig(out_dir / "ext_prediction_intervals.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_pi)

# ==============================================================================
# S7  Exchangeability (KS test)
# ==============================================================================
display(Markdown("## S7  Exchangeability check (KS test)"))
display(Markdown("""
Calibration and test nonconformity scores should be exchangeable for the
coverage guarantee to hold. This is particularly important when using
external predictions: if the external model behaves differently on the
calibration vs test split, exchangeability may be violated.

KS p > 0.05 -> no evidence of shift -> guarantee holds
KS p < 0.05 -> possible shift -> coverage may deviate from guarantee
"""))

scores_cal_adap  = np.abs(y_cal - y_cal_pred_ext) / sigma_cal_safe
scores_test_adap = np.abs(y_test - y_pred_a)       / sigma_test_safe
ks_stat, ks_p = ks_2samp(scores_cal_adap, scores_test_adap)
display(Markdown(f"- KS statistic={ks_stat:.4f}  p={ks_p:.4f}  "
                 f"({'OK' if ks_p > 0.05 else 'WARNING: possible shift'})"))

fig_ks, ax_ks = plt.subplots(figsize=(6, 4))
ax_ks.hist(scores_cal_adap,  bins="auto", density=True, alpha=0.5,
           label="Calibration", color="#2196F3")
ax_ks.hist(scores_test_adap, bins="auto", density=True, alpha=0.5,
           label="Test", color="#FF9800")
ax_ks.set_xlabel("Nonconformity score  |y - y_hat| / sigma_hat")
ax_ks.set_ylabel("Density")
ax_ks.set_title(f"Exchangeability: cal vs test  (KS p={ks_p:.4f})")
ax_ks.legend(fontsize=8); ax_ks.grid(True, alpha=0.3)
plt.tight_layout()
fig_ks.savefig(out_dir / "ext_exchangeability_ks.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ks)

# ==============================================================================
# S8  AD comparison
# ==============================================================================
display(Markdown("## S8  AD comparison: interval width vs applicability domain indices"))

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
        rho, pval = stats.spearmanr(ad_norm_c.values, cp_w_c.values)
        rng_b     = np.random.default_rng(42)
        boot_rhos = [stats.spearmanr(
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
            u_stat, u_p = stats.mannwhitneyu(in_ad, out_ad, alternative="two-sided")
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

# ==============================================================================
# S9  Save all results
# ==============================================================================
result_df = pd.DataFrame({
    "Smiles":           test_df["Smiles"].values, "True": y_test,
    "Pred_external":    y_test_pred_ext,
    "Lower_adaptive":   lower_a, "Upper_adaptive": upper_a,
    "Width_adaptive":   width_a, "Covered_adaptive": covered_a.astype(int),
    "Lower_plain":      lower_p, "Upper_plain":  upper_p,
    "Width_plain":      width_p, "Covered_plain": covered_p.astype(int),
    "Sigma_pred":       sigma_pred_test,
    **{c: df_test_ad[c].values for c in _available_ad if c in df_test_ad.columns},
})
metrics_df = pd.DataFrame([
    {"variant": "adaptive_external", "alpha": alpha, "coverage": covered_a.mean(),
     "mean_width": width_a.mean(), "sigma_r2": diag_s_cal["r2"],
     "n_test": len(test_df), "ks_pvalue": ks_p},
    {"variant": "plain_external",    "alpha": alpha, "coverage": covered_p.mean(),
     "mean_width": width_p.mean(), "sigma_r2": None,
     "n_test": len(test_df), "ks_pvalue": ks_p},
])
with pd.ExcelWriter(str(product["data"]), engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics_df.to_excel(w, sheet_name="Metrics",    index=False)
    if ad_results:
        pd.DataFrame(ad_results).to_excel(w, sheet_name="AD_CP_correlations", index=False)

display(Markdown("## [OK] External regression tutorial complete."))
display(Markdown(f"- Results        : `{product['data']}`"))
display(Markdown(f"- Model adaptive : `{product['ncmodel_adaptive']}`"))
display(Markdown(f"- Model plain    : `{product['ncmodel_plain']}`"))
display(Markdown(f"- Plots          : `{out_dir}`"))
