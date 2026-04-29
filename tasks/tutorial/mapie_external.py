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

AD comparison is part of this task (not separate) because:
  - AD metrics come from the same external file as the predictions.
  - The comparison requires the CP interval widths computed here.

If pred_file is null or missing the task runs with simulated predictions
(internal LightGBM, same as native) to demonstrate the code path.

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
from scipy import stats
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

# All dataset-specific settings come from env.tutorial.yaml via dataset_config
cfg               = dataset_config.get(dataset, {}) if isinstance(dataset_config, dict) else {}
pred_file         = cfg.get("pred_file",          None)
pred_col          = cfg.get("pred_col",           "Pred")
smiles_col_ext    = cfg.get("smiles_col",         "Smiles")
ad_cols           = cfg.get("ad_cols",            [])
ad_col_directions = cfg.get("ad_col_directions",  [])
n_quantile_bins   = int(cfg.get("n_quantile_bins", 5))
interval_width_col = cfg.get("interval_width_col", "Width_adaptive")
covered_col_cfg   = cfg.get("covered_col",        "Covered_adaptive")

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - REGRESSION (external model)"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))
display(Markdown(f"""
Configuration from `env.tutorial.yaml[{dataset}]`:
- pred_file  : `{pred_file}`
- pred_col   : `{pred_col}`
- ad_cols    : `{ad_cols}`
- ad_directions: `{ad_col_directions}`
"""))

# ==============================================================================
# S1  Data splits and fingerprints
# ==============================================================================
train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"- Fit={len(fit_df)}  Cal={len(cal_df)}  Test={len(test_df)}"))

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df);  y_fit  = fit_df[target_col].values.astype(float)
X_cal  = to_ecfp(cal_df);  y_cal  = cal_df[target_col].values.astype(float)
X_test = to_ecfp(test_df); y_test = test_df[target_col].values.astype(float)

# ==============================================================================
# S2  Sigma model (trained on internal LightGBM fit-set residuals)
#     The sigma model is always trained internally -- it only needs residuals,
#     not the external model itself.
# ==============================================================================
display(Markdown(f"## S2  Sigma model ({ncm})"))

_internal_base = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
_internal_base.fit(X_fit, y_fit)
y_fit_pred_internal = _internal_base.predict(X_fit)
y_cal_pred_internal = _internal_base.predict(X_cal)

residuals_fit = np.abs(y_fit - y_fit_pred_internal)
_, _ = detect_residual_degeneracy(residuals_fit, y_fit)
sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)

sigma_pred_fit  = sigma_model.predict(X_fit)
sigma_pred_cal  = sigma_model.predict(X_cal)
sigma_pred_test = sigma_model.predict(X_test)

diag_s     = sigma_diagnostics(residuals_fit, sigma_pred_fit)
diag_s_cal = sigma_diagnostics(np.abs(y_cal - y_cal_pred_internal), sigma_pred_cal)
display(Markdown(f"- Sigma R2 fit={diag_s['r2']:.3f}  cal={diag_s_cal['r2']:.3f}"))
display(Markdown("""
Note: the sigma model is trained on internal LightGBM residuals.
This is consistent with the VEGA pipeline: sigma models are always
trained from the VEGA training set residuals, not the external model.
The external model is used ONLY for predictions at test time.
"""))

# ==============================================================================
# S3  External predictions
#     Load from pred_file if available; fall back to internal model otherwise.
# ==============================================================================
display(Markdown("## S3  External predictions"))

df_test_with_ad = test_df.copy()  # will accumulate AD columns if available

if pred_file and Path(pred_file).exists():
    display(Markdown(f"- Loading external predictions from: `{pred_file}`"))
    _p   = Path(pred_file)
    _ext = pd.read_excel(_p) if _p.suffix in {".xlsx", ".xls"} else pd.read_csv(_p)

    # Normalise SMILES column name
    if smiles_col_ext != "Smiles" and smiles_col_ext in _ext.columns:
        _ext = _ext.rename(columns={smiles_col_ext: "Smiles"})

    # Validate that pred_col exists
    if pred_col not in _ext.columns:
        display(Markdown(
            f"WARNING: column `{pred_col}` not found in `{pred_file}`.\n"
            f"Available: {_ext.columns.tolist()}.\n"
            f"Falling back to internal predictions."))
        y_cal_pred_ext  = y_cal_pred_internal
        y_test_pred_ext = _internal_base.predict(X_test)
    else:
        # Align external predictions with calibration and test sets by Smiles
        _ext_idx = _ext.set_index("Smiles")[pred_col]
        y_cal_pred_ext  = cal_df["Smiles"].map(_ext_idx).fillna(
            pd.Series(y_cal_pred_internal, index=cal_df.index)).astype(float).values
        y_test_pred_ext = test_df["Smiles"].map(_ext_idx).fillna(
            pd.Series(_internal_base.predict(X_test), index=test_df.index)).astype(float).values

        n_matched_cal  = cal_df["Smiles"].isin(_ext["Smiles"]).sum()
        n_matched_test = test_df["Smiles"].isin(_ext["Smiles"]).sum()
        display(Markdown(f"- Matched cal={n_matched_cal}/{len(cal_df)}  "
                         f"test={n_matched_test}/{len(test_df)}"))

        # Carry AD columns from external file into test dataframe
        if ad_cols:
            _available_ad = [c for c in ad_cols if c in _ext.columns]
            _missing_ad   = [c for c in ad_cols if c not in _ext.columns]
            if _missing_ad:
                display(Markdown(f"WARNING: AD columns not found in file: {_missing_ad}"))
            if _available_ad:
                _ad_data = _ext.set_index("Smiles")[_available_ad]
                for c in _available_ad:
                    df_test_with_ad[c] = test_df["Smiles"].map(_ad_data[c]).values
                display(Markdown(f"- AD columns loaded: {_available_ad}"))
else:
    if pred_file:
        display(Markdown(f"WARNING: pred_file `{pred_file}` not found."))
    display(Markdown("- Using internal LightGBM predictions to demonstrate the code path."))
    y_cal_pred_ext  = y_cal_pred_internal
    y_test_pred_ext = _internal_base.predict(X_test)

# ==============================================================================
# S4  MAPIE conformal regressors with external predictions
# ==============================================================================
display(Markdown("## S4  MAPIE conformal regressors (adaptive and plain)"))
display(Markdown("""
ExternalPredictor wraps the pre-computed predictions as a sklearn-compatible
estimator. MAPIE sees only the prediction values, not the model itself.
This is the exact mechanism used for VEGA models in the paper.
"""))

eps = 1e-6
sigma_cal_safe  = np.maximum(sigma_pred_cal,  eps)
sigma_test_safe = np.maximum(sigma_pred_test, eps)

# Adaptive
estimator_a   = ExternalPredictor(y_cal_pred_ext)
mapie_adaptive = SplitConformalRegressor(
    estimator=estimator_a,
    conformity_score=ResidualNormalisedScore(residual_estimator=sigma_model,
                                              prefit=True, sym=True),
    prefit=True, confidence_level=1 - alpha,
)
mapie_adaptive.estimator_ = estimator_a.fit(None, None)
mapie_adaptive.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)
estimator_a.y_pred = y_test_pred_ext
y_pred_a, y_pis_a = mapie_adaptive.predict_interval(X_test)
lower_a  = y_pis_a[:, 0, 0]; upper_a = y_pis_a[:, 1, 0]
width_a  = upper_a - lower_a; covered_a = (y_test >= lower_a) & (y_test <= upper_a)

# Plain
estimator_p  = ExternalPredictor(y_cal_pred_ext)
mapie_plain  = SplitConformalRegressor(
    estimator=estimator_p,
    conformity_score=AbsoluteConformityScore(),
    prefit=True, confidence_level=1 - alpha,
)
mapie_plain.estimator_ = estimator_p.fit(None, None)
mapie_plain.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)
estimator_p.y_pred = y_test_pred_ext
y_pred_p, y_pis_p = mapie_plain.predict_interval(X_test)
lower_p  = y_pis_p[:, 0, 0]; upper_p = y_pis_p[:, 1, 0]
width_p  = upper_p - lower_p; covered_p = (y_test >= lower_p) & (y_test <= upper_p)

display(Markdown(f"- Adaptive: coverage={covered_a.mean():.3f}  mean_width={width_a.mean():.4f}"))
display(Markdown(f"- Plain:    coverage={covered_p.mean():.3f}  mean_width={width_p.mean():.4f}"))

# Save models to product paths
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
# S5  AD comparison (runs only if ad_cols are configured and loaded)
# ==============================================================================
display(Markdown("## S5  AD comparison: interval width vs applicability domain indices"))

_available_ad = [c for c in ad_cols if c in df_test_with_ad.columns]

if not _available_ad:
    display(Markdown(
        "### Skipped\n\n"
        "No AD columns available. To enable, add to `env.tutorial.yaml` "
        f"under `{dataset}`:\n\n"
        "```yaml\n"
        "  pred_file: \"path/to/predictions.xlsx\"\n"
        "  ad_cols: [\"ADI\"]\n"
        "  ad_col_directions: [\"similarity\"]\n"
        "```"
    ))
else:
    display(Markdown(f"Comparing CP interval width against: {_available_ad}"))
    display(Markdown("""
**Interpretation:**
- If Spearman rho < 0 (similarity direction): CP and AD agree -- wider intervals
  for molecules outside the AD. CP adds the statistical coverage guarantee AD lacks.
- If rho near 0: CP finds uncertain in-AD and reliable out-of-AD cases that AD misses.
  This is the BCF/polymer example: out-of-AD but narrow CP interval because the model
  has captured a mechanistic trend (steric exclusion at high molecular weight).
"""))

    # Add CP columns to df_test_with_ad for plotting
    df_test_with_ad["Width_adaptive"] = width_a
    df_test_with_ad["Width_plain"]    = width_p
    df_test_with_ad["Covered_adaptive"] = covered_a.astype(int)
    df_test_with_ad["Covered_plain"]    = covered_p.astype(int)

    ad_results = []

    for ad_col, ad_dir in zip(_available_ad,
                               ad_col_directions[:len(_available_ad)]):
        display(Markdown(f"### {ad_col}  (direction: {ad_dir})"))

        ad_raw  = df_test_with_ad[ad_col].astype(float)
        if ad_dir == "distance":
            rng = ad_raw.max() - ad_raw.min()
            ad_norm = 1.0 - (ad_raw - ad_raw.min()) / rng if rng > 0 \
                      else pd.Series(np.ones(len(ad_raw)), index=ad_raw.index)
        else:
            ad_norm = ad_raw.copy()

        mask       = ad_norm.notna() & pd.Series(width_a).notna()
        ad_norm_c  = ad_norm[mask].reset_index(drop=True)
        cp_w_c     = pd.Series(width_a)[mask].reset_index(drop=True)
        covered_c  = pd.Series(covered_a.astype(int))[mask].reset_index(drop=True)

        # Spearman with bootstrap CI
        rho, pval  = stats.spearmanr(ad_norm_c.values, cp_w_c.values)
        rng_boot   = np.random.default_rng(42)
        boot_rhos  = [
            stats.spearmanr(
                ad_norm_c.values[i := rng_boot.integers(0, len(ad_norm_c), len(ad_norm_c))],
                cp_w_c.values[i]
            )[0] for _ in range(1000)
        ]
        rho_lo = np.quantile(boot_rhos, 0.025)
        rho_hi = np.quantile(boot_rhos, 0.975)
        expected = "negative" if ad_dir == "similarity" else "positive"

        display(Markdown(
            f"- Spearman rho={rho:.3f}  p={pval:.4f}  "
            f"95% CI [{rho_lo:.3f}, {rho_hi:.3f}]  (expected {expected})"))

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # Panel A: scatter + trend
        axes[0, 0].scatter(ad_norm_c, cp_w_c, alpha=0.25, s=8,
                           color="#2196F3", rasterized=True)
        try:
            z = np.polyfit(ad_norm_c, cp_w_c, 1)
            xr = np.linspace(ad_norm_c.min(), ad_norm_c.max(), 100)
            axes[0, 0].plot(xr, np.poly1d(z)(xr), "r-", lw=2, alpha=0.7, label="trend")
        except Exception:
            pass
        axes[0, 0].set_xlabel(f"{ad_col} (normalised, higher=more reliable)")
        axes[0, 0].set_ylabel("Adaptive interval width")
        axes[0, 0].set_title(
            f"Spearman rho={rho:.3f}  p={pval:.4f}\n"
            f"95% CI [{rho_lo:.3f}, {rho_hi:.3f}]  (expected {expected})", fontsize=9)
        axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)

        # Panel B: in-AD vs out-of-AD boxplot
        in_ad  = cp_w_c[ad_norm_c >= 0.5].dropna()
        out_ad = cp_w_c[ad_norm_c <  0.5].dropna()
        if len(in_ad) > 1 and len(out_ad) > 1:
            bp = axes[0, 1].boxplot(
                [in_ad.values, out_ad.values], patch_artist=True, widths=0.5,
                medianprops=dict(color="black", lw=2))
            for patch, c_ in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
                patch.set_facecolor(c_)
            u_stat, u_p = stats.mannwhitneyu(in_ad, out_ad, alternative="two-sided")
            axes[0, 1].text(0.98, 0.97, f"Mann-Whitney p={u_p:.4f}",
                            transform=axes[0, 1].transAxes, ha="right", va="top",
                            fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        axes[0, 1].set_xticklabels([f"In-AD (n={len(in_ad)})",
                                     f"Out-of-AD (n={len(out_ad)})"], fontsize=9)
        axes[0, 1].set_ylabel("Adaptive interval width")
        axes[0, 1].set_title("In-AD vs Out-of-AD  (threshold=0.5)")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # Panel C: stratified bar + coverage overlay
        _tmp = pd.DataFrame({"AD_norm": ad_norm_c.values, "width": cp_w_c.values,
                               "covered": covered_c.values})
        _tmp["_bin"] = pd.qcut(ad_norm_c, q=n_quantile_bins,
                               labels=False, duplicates="drop")
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
        axes[1, 0].bar(x, strat_df["Mean width"], color="#3498db", alpha=0.8, edgecolor="black")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(strat_df["AD bin"], rotation=30, ha="right", fontsize=8)
        axes[1, 0].set_ylabel("Mean interval width"); axes[1, 0].grid(True, alpha=0.3, axis="y")
        axes[1, 0].set_title("Mean interval width by AD quintile\n(Q1=lowest AD, Q5=highest AD)")
        ax2 = axes[1, 0].twinx()
        ax2.plot(x, strat_df["Coverage"], "rs-", lw=2, ms=8, label="Coverage")
        ax2.axhline(1 - alpha, color="red", linestyle="--", lw=1.5, alpha=0.7,
                    label=f"Target {1-alpha:.0%}")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("Coverage"); ax2.legend(loc="upper left", fontsize=8)

        # Panel D: width KDE in-AD vs out-of-AD
        axes[1, 1].hist(in_ad.values,  bins="auto", density=True, alpha=0.5,
                        color="#27ae60", label=f"In-AD (n={len(in_ad)})")
        axes[1, 1].hist(out_ad.values, bins="auto", density=True, alpha=0.5,
                        color="#e74c3c", label=f"Out-of-AD (n={len(out_ad)})")
        if len(in_ad):
            axes[1, 1].axvline(in_ad.mean(), color="#27ae60", lw=2, linestyle="--",
                               label=f"Mean in={in_ad.mean():.3f}")
        if len(out_ad):
            axes[1, 1].axvline(out_ad.mean(), color="#e74c3c", lw=2, linestyle="--",
                               label=f"Mean out={out_ad.mean():.3f}")
        axes[1, 1].set_xlabel("Adaptive interval width"); axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Width distribution: In-AD vs Out-of-AD")
        axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f"{dataset}: interval width vs {ad_col}  (rho={rho:.3f})\n"
            f"Negative rho = wider CP intervals outside AD (expected for similarity metric)",
            fontsize=10)
        plt.tight_layout()
        _plot_path = out_dir / f"ad_{ad_col}_comparison.png"
        fig.savefig(_plot_path, dpi=150, bbox_inches="tight")
        plt.show() 
        plt.close(fig)
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
# S6  Save all results
# ==============================================================================
result_df = pd.DataFrame({
    "Smiles": test_df["Smiles"].values, "True": y_test,
    "Pred_external": y_test_pred_ext,
    "Lower_adaptive": lower_a, "Upper_adaptive": upper_a,
    "Width_adaptive": width_a, "Covered_adaptive": covered_a.astype(int),
    "Lower_plain": lower_p,    "Upper_plain": upper_p,
    "Width_plain": width_p,    "Covered_plain": covered_p.astype(int),
    **{c: df_test_with_ad[c].values for c in _available_ad
       if c in df_test_with_ad.columns},
})

metrics_df = pd.DataFrame([
    {"variant": "adaptive_external", "alpha": alpha, "coverage": covered_a.mean(),
     "mean_width": width_a.mean(), "sigma_r2": diag_s_cal["r2"], "n_test": len(test_df)},
    {"variant": "plain_external",    "alpha": alpha, "coverage": covered_p.mean(),
     "mean_width": width_p.mean(), "sigma_r2": None,            "n_test": len(test_df)},
])

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics_df.to_excel(w, sheet_name="Metrics",    index=False)
    if _available_ad and ad_results:
        pd.DataFrame(ad_results).to_excel(w, sheet_name="AD_CP_correlations", index=False)

display(Markdown("## [OK] External regression tutorial complete."))
display(Markdown(f"- Results         : `{product['data']}`"))
display(Markdown(f"- Model adaptive  : `{product['ncmodel_adaptive']}`"))
display(Markdown(f"- Model plain     : `{product['ncmodel_plain']}`"))
display(Markdown(f"- Plots           : `{out_dir}`"))
