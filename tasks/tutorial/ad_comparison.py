# -*- coding: utf-8 -*-
"""
tasks/tutorial/ad_comparison.py
---------------------------------
Tutorial: Comparing Conformal Prediction metrics vs Applicability Domain indices.

Purpose
-------
Illustrates the relationship between CP uncertainty (interval width / singleton rate)
and applicability domain (AD) metrics read from an external file.  The tutorial does
NOT compute AD metrics -- they are assumed to be pre-computed by the external model
(e.g. VEGA ADI, OPERA AD index, or any 0-1 similarity/distance score) and stored in
the same predictions file as the model predictions.

For regression: CP metric is interval width (per-molecule).
For classification: CP metric is set size and singleton indicator (per-molecule).

Both are compared to one or more AD columns via:
  - Scatter plots (CP metric vs AD index)
  - Spearman / Pearson correlation with confidence interval
  - Stratified coverage / efficiency by AD quintile (the key regulatory question:
    "do predictions outside the AD have worse CP intervals?")
  - KDE overlay showing the distribution of CP widths for in-AD vs out-of-AD molecules

AD column interpretation
------------------------
The AD column can be either:
  - A SIMILARITY score (0=dissimilar, 1=similar): higher -> more reliable -> narrower CP
  - A DISTANCE score  (0=in-AD, large=out-of-AD): higher -> less reliable -> wider CP
Set ad_col_direction = "similarity" or "distance" accordingly.

Inputs (ploomber params)
------------------------
  regression_pred_file   : path to regression predictions Excel/CSV
                           must contain: smiles_col, true_col, pred_col,
                                         interval_width_col, [ad_col, ...]
  classification_pred_file : path to classification predictions Excel/CSV
                           must contain: smiles_col, true_col, pred_col,
                                         set_size_col, [ad_col, ...]
  smiles_col             : SMILES column name (default "Smiles")
  true_col_regr          : true value column for regression (default "True")
  interval_width_col     : interval width column (default "Width_adaptive")
  covered_col_regr       : covered indicator column for regression (default "Covered_adaptive")
  true_col_class         : true class column for classification (default "True")
  set_size_col_a         : set size column for Approach A (default "SetSize_A")
  set_size_col_b         : set size column for Approach B (default "SetSize_B")
  covered_col_a          : covered indicator for Approach A (default "Cov_A")
  covered_col_b          : covered indicator for Approach B (default "Cov_B")
  ad_cols                : list of AD column names to compare against
  ad_col_directions      : list of "similarity" or "distance" per ad_col
  alpha                  : miscoverage level (default 0.1)
  n_quantile_bins        : number of AD quantile bins for stratified analysis (default 5)
  product                : {nb, data, plots_dir}
"""

# + tags=["parameters"]
regression_pred_file     = None
classification_pred_file = None
smiles_col               = "Smiles"
true_col_regr            = "True"
interval_width_col       = "Width_adaptive"
covered_col_regr         = "Covered_adaptive"
true_col_class           = "True"
set_size_col_a           = "SetSize_A"
set_size_col_b           = "SetSize_B"
covered_col_a            = "Cov_A"
covered_col_b            = "Cov_B"
ad_cols                  = ["ADI"]           # list of AD column names
ad_col_directions        = ["similarity"]    # per-column: "similarity" or "distance"
alpha                    = 0.1
n_quantile_bins          = 5
product                  = None
upstream                 = None
# -

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from IPython.display import display, Markdown
%matplotlib inline

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
plots_dir = Path(product["plots_dir"])
plots_dir.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def _load(path, sheet=0):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_excel(p, sheet_name=sheet) if p.suffix in {".xlsx", ".xls"} else pd.read_csv(p)


def _normalise_ad(series, direction):
    """
    Return a series where higher value always means MORE RELIABLE (better AD).
    For similarity: return as-is.
    For distance: return 1 - normalised distance.
    """
    s = series.astype(float)
    if direction == "distance":
        rng = s.max() - s.min()
        s = 1.0 - (s - s.min()) / rng if rng > 0 else pd.Series(np.ones(len(s)), index=s.index)
    return s


def _spearman_ci(x, y, n_boot=1000, ci=0.95):
    """Spearman rho with bootstrap CI."""
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boot_rhos.append(stats.spearmanr(x[idx], y[idx])[0])
    lo = np.quantile(boot_rhos, (1 - ci) / 2)
    hi = np.quantile(boot_rhos, 1 - (1 - ci) / 2)
    return rho, p, lo, hi


def _stratified_table(df, ad_col_norm, cp_metric_col, covered_col, n_bins):
    """
    Bin molecules by AD quintile; compute mean CP metric and coverage per bin.
    Returns a DataFrame with columns: bin_label, n, mean_cp, coverage.
    """
    df = df.copy()
    df["_ad_bin"] = pd.qcut(ad_col_norm, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(df["_ad_bin"].dropna().unique()):
        mask = df["_ad_bin"] == b
        sub  = df[mask]
        lo   = ad_col_norm[mask].min()
        hi   = ad_col_norm[mask].max()
        rows.append({
            "AD bin": f"Q{int(b)+1} [{lo:.2f}-{hi:.2f}]",
            "n":      int(mask.sum()),
            f"Mean {cp_metric_col}": sub[cp_metric_col].mean(),
            "Coverage": sub[covered_col].mean() if covered_col in sub else float("nan"),
        })
    return pd.DataFrame(rows)


def _scatter_with_kde(ax, ad_norm, cp_metric, ad_label, cp_label, direction, color="#2196F3"):
    """Scatter + marginal KDE shading."""
    ax.scatter(ad_norm, cp_metric, alpha=0.25, s=8, color=color, rasterized=True)
    # Trend line
    try:
        z = np.polyfit(ad_norm, cp_metric, 1)
        xr = np.linspace(ad_norm.min(), ad_norm.max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), "r-", lw=2, alpha=0.7, label="trend")
    except Exception:
        pass
    rho, p, lo, hi = _spearman_ci(ad_norm.values, cp_metric.values)
    expected = "negative" if direction == "similarity" else "positive"
    ax.set_xlabel(f"{ad_label} (normalised, higher=more reliable)")
    ax.set_ylabel(cp_label)
    ax.set_title(f"Spearman rho={rho:.3f}  p={p:.4f}\n95% CI [{lo:.3f}, {hi:.3f}]  "
                 f"(expected {expected} correlation)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    return rho, p


def _in_out_ad_boxplot(ax, ad_norm, cp_metric, threshold, cp_label):
    """Boxplot of CP metric for in-AD vs out-of-AD molecules."""
    in_ad  = cp_metric[ad_norm >= threshold]
    out_ad = cp_metric[ad_norm <  threshold]
    data   = [in_ad.dropna().values, out_ad.dropna().values]
    labels = [f"In-AD (n={len(in_ad)})", f"Out-of-AD (n={len(out_ad)})"]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2),
                    boxprops=dict(edgecolor="black", lw=1.5))
    for patch, color in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
        patch.set_facecolor(color)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(cp_label)
    ax.set_title(f"In-AD vs Out-of-AD\n(threshold=0.5)")
    ax.grid(True, alpha=0.3, axis="y")
    # Mann-Whitney U test
    if len(in_ad) > 1 and len(out_ad) > 1:
        u_stat, u_p = stats.mannwhitneyu(in_ad.dropna(), out_ad.dropna(),
                                         alternative="two-sided")
        ax.text(0.98, 0.97, f"Mann-Whitney p={u_p:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))


def _stratified_bar(ax, strat_df, cp_col, covered=True, alpha_target=None):
    """Bar chart of CP metric by AD quintile, with coverage overlay."""
    x = np.arange(len(strat_df))
    bars = ax.bar(x, strat_df[cp_col], color="#3498db", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_df["AD bin"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(cp_col, fontsize=10)
    ax.set_title("Mean CP metric by AD quintile\n(Q1=lowest AD, Q5=highest AD)")
    ax.grid(True, alpha=0.3, axis="y")
    if "Coverage" in strat_df.columns and covered:
        ax2 = ax.twinx()
        ax2.plot(x, strat_df["Coverage"], "rs-", lw=2, ms=8, label="Coverage")
        if alpha_target is not None:
            ax2.axhline(1 - alpha_target, color="red", linestyle="--", lw=1.5,
                        alpha=0.7, label=f"Target {1-alpha_target:.0%}")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Coverage")
        ax2.legend(loc="upper left", fontsize=8)


# ==============================================================================
# MAIN
# ==============================================================================

display(Markdown("# AD vs CP Tutorial: Comparing Applicability Domain and Conformal Prediction"))
display(Markdown(f"""
This tutorial compares CP uncertainty metrics (interval width, set size) against
pre-computed AD indices read from external prediction files.

**Key question**: Do CP intervals encode the same information as AD metrics, and what
do they add beyond what AD already provides?

Expected finding (from the paper):
- CP interval width correlates *negatively* with AD similarity (wider intervals
  for out-of-AD molecules) -- confirming CP captures the same underlying signal.
- The correlation is imperfect: CP can assign narrow intervals to out-of-AD
  molecules when the model has captured a mechanistic trend (BCF example), and
  wide intervals to in-AD molecules when there is high local prediction error.
- CP adds the statistical coverage guarantee that AD cannot provide.

AD columns configured: {ad_cols}
Directions: {ad_col_directions}
"""))

results_summary = []

# ==============================================================================
# REGRESSION
# ==============================================================================
df_regr = _load(regression_pred_file)

if df_regr is not None:
    display(Markdown("## Regression: interval width vs AD index"))

    # Validate required columns
    _missing_r = [c for c in [interval_width_col, covered_col_regr]
                  if c not in df_regr.columns]
    if _missing_r:
        display(Markdown(f"WARNING: columns not found in regression file: {_missing_r}. "
                         f"Available: {df_regr.columns.tolist()}"))
    else:
        for ad_col, ad_dir in zip(ad_cols, ad_col_directions):
            if ad_col not in df_regr.columns:
                display(Markdown(f"- AD column '{ad_col}' not found in regression file, skipping."))
                continue

            display(Markdown(f"### AD column: {ad_col}  (direction: {ad_dir})"))

            ad_norm = _normalise_ad(df_regr[ad_col], ad_dir).rename("AD_norm")
            cp_w    = df_regr[interval_width_col].astype(float)
            covered = df_regr[covered_col_regr].astype(float)

            # Remove rows with NaN in either metric
            mask = ad_norm.notna() & cp_w.notna()
            ad_norm_c = ad_norm[mask]
            cp_w_c    = cp_w[mask]
            covered_c = covered[mask]

            fig, axes = plt.subplots(2, 2, figsize=(13, 10))

            # A: Scatter + trend
            rho, pval = _scatter_with_kde(
                axes[0, 0], ad_norm_c, cp_w_c,
                ad_label=ad_col, cp_label="Interval width",
                direction=ad_dir, color="#2196F3")

            # B: In-AD vs Out-of-AD boxplot
            _in_out_ad_boxplot(axes[0, 1], ad_norm_c, cp_w_c,
                               threshold=0.5, cp_label="Interval width")

            # C: Stratified by AD quintile
            strat = _stratified_table(
                pd.DataFrame({"AD_norm": ad_norm_c, interval_width_col: cp_w_c,
                               covered_col_regr: covered_c}),
                ad_col_norm=ad_norm_c,
                cp_metric_col=interval_width_col,
                covered_col=covered_col_regr,
                n_bins=n_quantile_bins)
            _stratified_bar(axes[1, 0], strat, cp_col=interval_width_col,
                            alpha_target=alpha)
            display(strat)

            # D: KDE of interval width for in-AD vs out-of-AD
            in_ad  = cp_w_c[ad_norm_c >= 0.5]
            out_ad = cp_w_c[ad_norm_c <  0.5]
            axes[1, 1].hist(in_ad.values,  bins=40, density=True, alpha=0.5,
                            color="#27ae60", label=f"In-AD (n={len(in_ad)})")
            axes[1, 1].hist(out_ad.values, bins=40, density=True, alpha=0.5,
                            color="#e74c3c", label=f"Out-of-AD (n={len(out_ad)})")
            axes[1, 1].axvline(in_ad.mean(),  color="#27ae60", lw=2, linestyle="--",
                               label=f"Mean in-AD={in_ad.mean():.3f}")
            axes[1, 1].axvline(out_ad.mean(), color="#e74c3c", lw=2, linestyle="--",
                               label=f"Mean out={out_ad.mean():.3f}")
            axes[1, 1].set_xlabel("Interval width")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title("Width distribution: In-AD vs Out-of-AD")
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(
                f"Regression CP interval width vs {ad_col}\n"
                f"Spearman rho={rho:.3f}  p={pval:.4f}  "
                f"(negative = wider intervals for out-of-AD)",
                fontsize=11)
            plt.tight_layout()
            _plot_path = plots_dir / f"regr_ad_{ad_col}.png"
            fig.savefig(_plot_path, dpi=150, bbox_inches="tight")
            plt.show(); plt.close(fig)
            display(Markdown(f"- Plot saved: {_plot_path}"))

            results_summary.append({
                "mode": "regression", "ad_col": ad_col, "direction": ad_dir,
                "n": int(mask.sum()), "spearman_rho": round(rho, 4),
                "p_value": round(pval, 6),
                "mean_width_in_AD":  round(in_ad.mean(),  4),
                "mean_width_out_AD": round(out_ad.mean(), 4),
            })

else:
    display(Markdown("- No regression prediction file provided. Set `regression_pred_file` "
                     "in env.tutorial.yaml to enable regression AD comparison."))

# ==============================================================================
# CLASSIFICATION
# ==============================================================================
df_class = _load(classification_pred_file)

if df_class is not None:
    display(Markdown("## Classification: set size / singleton rate vs AD index"))

    for ad_col, ad_dir in zip(ad_cols, ad_col_directions):
        if ad_col not in df_class.columns:
            display(Markdown(f"- AD column '{ad_col}' not found in classification file, skipping."))
            continue

        display(Markdown(f"### AD column: {ad_col}  (direction: {ad_dir})"))

        ad_norm = _normalise_ad(df_class[ad_col], ad_dir)
        mask    = ad_norm.notna()
        ad_norm_c = ad_norm[mask]

        for approach_label, sz_col, cov_col in [
            ("A: LAC (real proba)",    set_size_col_a, covered_col_a),
            ("B: NCM pseudo-proba",    set_size_col_b, covered_col_b),
        ]:
            if sz_col not in df_class.columns:
                display(Markdown(f"  - Column '{sz_col}' not found, skipping {approach_label}."))
                continue

            display(Markdown(f"#### Approach {approach_label}"))

            sz_c   = df_class.loc[mask, sz_col].astype(float)
            cov_c  = df_class.loc[mask, cov_col].astype(float) if cov_col in df_class.columns \
                     else pd.Series(np.nan, index=sz_c.index)
            sing_c = (sz_c == 1).astype(float)  # singleton indicator

            fig_c, axes_c = plt.subplots(2, 2, figsize=(13, 10))

            # A: Scatter set size vs AD
            rho_sz, pval_sz = _scatter_with_kde(
                axes_c[0, 0], ad_norm_c, sz_c,
                ad_label=ad_col, cp_label="Set size",
                direction=ad_dir, color="#9C27B0")

            # B: In-AD vs Out-of-AD boxplot for set size
            _in_out_ad_boxplot(axes_c[0, 1], ad_norm_c, sz_c,
                               threshold=0.5, cp_label="Set size")

            # C: Stratified by AD quintile
            strat_c = _stratified_table(
                pd.DataFrame({"AD_norm": ad_norm_c.values,
                               sz_col:   sz_c.values,
                               cov_col:  cov_c.values}),
                ad_col_norm=ad_norm_c.reset_index(drop=True),
                cp_metric_col=sz_col,
                covered_col=cov_col,
                n_bins=n_quantile_bins)
            _stratified_bar(axes_c[1, 0], strat_c, cp_col=sz_col, alpha_target=alpha)
            display(strat_c)

            # D: Singleton rate by AD quintile
            strat_sing = _stratified_table(
                pd.DataFrame({"AD_norm":  ad_norm_c.values,
                               "singleton": sing_c.values,
                               cov_col:    cov_c.values}),
                ad_col_norm=ad_norm_c.reset_index(drop=True),
                cp_metric_col="singleton",
                covered_col=cov_col,
                n_bins=n_quantile_bins)
            x_s = np.arange(len(strat_sing))
            axes_c[1, 1].bar(x_s, strat_sing["Mean singleton"],
                             color="#FF9800", alpha=0.8, edgecolor="black")
            axes_c[1, 1].set_xticks(x_s)
            axes_c[1, 1].set_xticklabels(strat_sing["AD bin"],
                                          rotation=30, ha="right", fontsize=8)
            axes_c[1, 1].set_ylabel("Singleton rate")
            axes_c[1, 1].set_title("Singleton rate by AD quintile\n"
                                   "(Q5=highest AD => most singletons expected)")
            axes_c[1, 1].grid(True, alpha=0.3, axis="y")
            rho_sing, _ = stats.spearmanr(ad_norm_c.values, sing_c.values)
            axes_c[1, 1].text(0.97, 0.97,
                              f"Spearman rho={rho_sing:.3f}",
                              transform=axes_c[1, 1].transAxes,
                              ha="right", va="top", fontsize=9,
                              bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            plt.suptitle(
                f"Classification CP ({approach_label}) vs {ad_col}\n"
                f"Set size Spearman rho={rho_sz:.3f}  p={pval_sz:.4f}",
                fontsize=11)
            plt.tight_layout()
            _plot_path_c = plots_dir / f"class_ad_{ad_col}_{sz_col}.png"
            fig_c.savefig(_plot_path_c, dpi=150, bbox_inches="tight")
            plt.show(); plt.close(fig_c)
            display(Markdown(f"- Plot saved: {_plot_path_c}"))

            results_summary.append({
                "mode": f"classification_{approach_label}", "ad_col": ad_col,
                "direction": ad_dir, "n": int(mask.sum()),
                "spearman_rho_setsize": round(rho_sz, 4),
                "p_value_setsize": round(pval_sz, 6),
                "spearman_rho_singleton": round(rho_sing, 4),
            })
else:
    display(Markdown("- No classification prediction file provided. Set "
                     "`classification_pred_file` in env.tutorial.yaml to enable."))

# ==============================================================================
# Summary table and interpretation
# ==============================================================================
if results_summary:
    display(Markdown("## Summary: CP vs AD correlations"))
    display(pd.DataFrame(results_summary))

    display(Markdown(f"""
### Interpretation guide

| Spearman rho (similarity direction) | Interpretation |
|---|---|
| rho < -0.3, p < 0.05 | CP captures same signal as AD: wider intervals outside AD |
| rho near 0 | CP and AD independent: CP adds new information |
| rho > +0.3 | Unexpected -- check AD column direction |

**Key regulatory message:**
- If rho is significantly negative: CP and AD agree on which molecules are uncertain.
  CP quantifies the uncertainty formally (coverage guarantee); AD only flags it.
- If rho is near 0 for some molecules: CP finds uncertain in-AD predictions and
  reliable out-of-AD predictions that AD would misclassify.
  This is the BCF/polymer example: out-of-AD but narrow CP interval.
- CP never *replaces* AD but provides the statistical guarantee AD was designed to
  give but could not: interval width is a *calibrated* uncertainty measure.
"""))

# Save summary
summary_df = pd.DataFrame(results_summary) if results_summary else pd.DataFrame()
with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    summary_df.to_excel(w, sheet_name="AD_CP_correlations", index=False)

display(Markdown(f"## [OK] AD comparison tutorial complete."))
display(Markdown(f"- Summary : {product['data']}"))
display(Markdown(f"- Plots   : {plots_dir}"))
