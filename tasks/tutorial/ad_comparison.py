
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from IPython.display import display, Markdown
%matplotlib inline

# -*- coding: utf-8 -*-
"""
tasks/tutorial/ad_comparison.py
---------------------------------
Tutorial: Comparing CP metrics vs Applicability Domain indices.

Runs once per dataset key (regression or classification grid).
All configuration -- pred_file, cp columns, ad_cols, ad_col_directions --
is read from dataset_config[dataset_key] as passed from env.tutorial.yaml.
"""

# + tags=["parameters"]
dataset_key    = None
dataset_config = None   # full dict from env: {dataset_key: {pred_file, ad_cols, ...}}
task_type      = "regression"   # "regression" or "classification"
alpha          = 0.1
product        = None
upstream       = None
# -


Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
plots_dir = Path(product["plots_dir"])
plots_dir.mkdir(parents=True, exist_ok=True)

# ── resolve dataset config ─────────────────────────────────────────────────────
cfg = dataset_config.get(dataset_key, {}) if isinstance(dataset_config, dict) else {}

pred_file         = cfg.get("pred_file",          None)
ad_cols           = cfg.get("ad_cols",            [])
ad_col_directions = cfg.get("ad_col_directions",  [])
smiles_col        = cfg.get("smiles_col",         "Smiles")
n_quantile_bins   = int(cfg.get("n_quantile_bins", 5))

# Regression-specific column names
interval_width_col = cfg.get("interval_width_col", "Width_adaptive")
covered_col_regr   = cfg.get("covered_col",        "Covered_adaptive")

# Classification-specific column names
set_size_col_a = cfg.get("set_size_col_a", "SetSize_A")
set_size_col_b = cfg.get("set_size_col_b", "SetSize_B")
covered_col_a  = cfg.get("covered_col_a",  "Cov_A")
covered_col_b  = cfg.get("covered_col_b",  "Cov_B")

display(Markdown(f"# AD vs CP Comparison: {dataset_key}  ({task_type})"))
display(Markdown(f"""
All configuration read from `env.tutorial.yaml` entry `{dataset_key}`:

- pred_file        : `{pred_file}`
- ad_cols          : `{ad_cols}`
- ad_col_directions: `{ad_col_directions}`
"""))

if not pred_file or not Path(pred_file).exists():
    display(Markdown(
        f"## Skipped\n\n"
        f"No `pred_file` configured for dataset `{dataset_key}`, or file not found.\n\n"
        f"To enable, add to `env.tutorial.yaml` under `{dataset_key}`:\n\n"
        f"```yaml\n"
        f"  pred_file: \"path/to/predictions.xlsx\"\n"
        f"  ad_cols: [\"ADI\"]\n"
        f"  ad_col_directions: [\"similarity\"]\n"
        f"```"
    ))
    # Write empty output so pipeline product exists
    pd.DataFrame().to_excel(product["data"])
    import sys; sys.exit(0)

if not ad_cols:
    display(Markdown(
        f"## Skipped\n\n"
        f"No `ad_cols` configured for dataset `{dataset_key}`.\n\n"
        f"Add `ad_cols` and `ad_col_directions` to this dataset entry in "
        f"`env.tutorial.yaml` to enable comparison."
    ))
    pd.DataFrame().to_excel(product["data"])
    import sys; sys.exit(0)

# ── load predictions file ──────────────────────────────────────────────────────
_p = Path(pred_file)
df = pd.read_excel(_p) if _p.suffix in {".xlsx", ".xls"} else pd.read_csv(_p)
display(Markdown(f"- Loaded {len(df)} rows from `{pred_file}`"))
display(Markdown(f"- Columns: {df.columns.tolist()}"))

# ── helpers ────────────────────────────────────────────────────────────────────

def _normalise_ad(series, direction):
    s = series.astype(float).copy()
    if direction == "distance":
        rng = s.max() - s.min()
        s = 1.0 - (s - s.min()) / rng if rng > 0 else pd.Series(
            np.ones(len(s)), index=s.index)
    return s


def _spearman_ci(x, y, n_boot=1000, ci=0.95):
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot_rhos = [stats.spearmanr(x[i], y[i])[0]
                 for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    boot_rhos = [stats.spearmanr(
        x[rng.integers(0, len(x), len(x))],
        y[rng.integers(0, len(x), len(x))])[0]
        for _ in range(n_boot)]
    lo = np.quantile(boot_rhos, (1 - ci) / 2)
    hi = np.quantile(boot_rhos, 1 - (1 - ci) / 2)
    return rho, p, lo, hi


def _stratified_table(df_, ad_norm, cp_col, cov_col, n_bins):
    df_ = df_.copy()
    df_["_bin"] = pd.qcut(ad_norm, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(df_["_bin"].dropna().unique()):
        mask = df_["_bin"] == b
        sub  = df_[mask]
        rows.append({
            "AD bin":          f"Q{int(b)+1} [{ad_norm[mask].min():.2f}-{ad_norm[mask].max():.2f}]",
            "n":               int(mask.sum()),
            f"Mean {cp_col}":  sub[cp_col].mean(),
            "Coverage":        sub[cov_col].mean() if cov_col in sub.columns else float("nan"),
        })
    return pd.DataFrame(rows)


def _make_2x2(ad_norm_c, cp_c, cov_c, ad_col, ad_dir, cp_label, color, alpha_target):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: scatter + trend
    axes[0, 0].scatter(ad_norm_c, cp_c, alpha=0.25, s=8, color=color, rasterized=True)
    try:
        z  = np.polyfit(ad_norm_c, cp_c, 1)
        xr = np.linspace(ad_norm_c.min(), ad_norm_c.max(), 100)
        axes[0, 0].plot(xr, np.poly1d(z)(xr), "r-", lw=2, alpha=0.7, label="trend")
    except Exception:
        pass
    rho, pval, lo_ci, hi_ci = _spearman_ci(
        np.asarray(ad_norm_c), np.asarray(cp_c))
    expected = "negative" if ad_dir == "similarity" else "positive"
    axes[0, 0].set_xlabel(f"{ad_col} (normalised, higher=more reliable)")
    axes[0, 0].set_ylabel(cp_label)
    axes[0, 0].set_title(
        f"Spearman rho={rho:.3f}  p={pval:.4f}\n"
        f"95% CI [{lo_ci:.3f}, {hi_ci:.3f}]  (expected {expected})", fontsize=9)
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend(fontsize=7)

    # Panel B: in-AD vs out-of-AD boxplot
    threshold = 0.5
    in_ad  = cp_c[ad_norm_c >= threshold].dropna()
    out_ad = cp_c[ad_norm_c <  threshold].dropna()
    if len(in_ad) > 0 and len(out_ad) > 0:
        bp = axes[0, 1].boxplot(
            [in_ad.values, out_ad.values], patch_artist=True, widths=0.5,
            medianprops=dict(color="black", lw=2),
            boxprops=dict(edgecolor="black", lw=1.5))
        for patch, c_ in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
            patch.set_facecolor(c_)
        axes[0, 1].set_xticklabels(
            [f"In-AD (n={len(in_ad)})", f"Out-of-AD (n={len(out_ad)})"], fontsize=9)
        u_stat, u_p = stats.mannwhitneyu(in_ad, out_ad, alternative="two-sided")
        axes[0, 1].text(0.98, 0.97, f"Mann-Whitney p={u_p:.4f}",
                        transform=axes[0, 1].transAxes, ha="right", va="top",
                        fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    axes[0, 1].set_ylabel(cp_label)
    axes[0, 1].set_title("In-AD vs Out-of-AD  (threshold=0.5)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Panel C: stratified bar + coverage line
    strat = _stratified_table(
        pd.DataFrame({"AD_norm": ad_norm_c.values,
                       cp_label: cp_c.values, "cov": cov_c.values}),
        ad_norm=pd.Series(ad_norm_c.values),
        cp_col=cp_label, cov_col="cov", n_bins=n_quantile_bins)
    x = np.arange(len(strat))
    axes[1, 0].bar(x, strat[f"Mean {cp_label}"],
                   color="#3498db", alpha=0.8, edgecolor="black")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strat["AD bin"], rotation=30, ha="right", fontsize=8)
    axes[1, 0].set_ylabel(cp_label); axes[1, 0].grid(True, alpha=0.3, axis="y")
    axes[1, 0].set_title("Mean CP metric by AD quintile\n(Q1=lowest AD, Q5=highest AD)")
    ax2 = axes[1, 0].twinx()
    ax2.plot(x, strat["Coverage"], "rs-", lw=2, ms=8, label="Coverage")
    ax2.axhline(1 - alpha_target, color="red", linestyle="--", lw=1.5,
                alpha=0.7, label=f"Target {1-alpha_target:.0%}")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Coverage"); ax2.legend(loc="upper left", fontsize=8)
    display(strat)

    # Panel D: width / set-size distribution in-AD vs out-of-AD
    axes[1, 1].hist(in_ad.values,  bins=40, density=True, alpha=0.5,
                    color="#27ae60", label=f"In-AD (n={len(in_ad)})")
    axes[1, 1].hist(out_ad.values, bins=40, density=True, alpha=0.5,
                    color="#e74c3c", label=f"Out-of-AD (n={len(out_ad)})")
    if len(in_ad):
        axes[1, 1].axvline(in_ad.mean(), color="#27ae60", lw=2, linestyle="--",
                           label=f"Mean in={in_ad.mean():.3f}")
    if len(out_ad):
        axes[1, 1].axvline(out_ad.mean(), color="#e74c3c", lw=2, linestyle="--",
                           label=f"Mean out={out_ad.mean():.3f}")
    axes[1, 1].set_xlabel(cp_label); axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title(f"{cp_label} distribution: In-AD vs Out-of-AD")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    return fig, rho, pval, strat


results_summary = []

# ==============================================================================
# REGRESSION
# ==============================================================================
if task_type == "regression":
    display(Markdown("## Regression: interval width vs AD index"))

    missing = [c for c in [interval_width_col, covered_col_regr] if c not in df.columns]
    if missing:
        display(Markdown(
            f"WARNING: columns not found: {missing}.\n"
            f"Available columns: {df.columns.tolist()}\n\n"
            f"Check `interval_width_col` and `covered_col` in `env.tutorial.yaml` "
            f"under dataset `{dataset_key}`."))
    else:
        cp_w    = df[interval_width_col].astype(float)
        covered = df[covered_col_regr].astype(float)

        for ad_col, ad_dir in zip(ad_cols, ad_col_directions):
            if ad_col not in df.columns:
                display(Markdown(f"- AD column `{ad_col}` not found, skipping."))
                continue

            display(Markdown(f"### {ad_col}  (direction: {ad_dir})"))
            ad_norm = _normalise_ad(df[ad_col], ad_dir)
            mask    = ad_norm.notna() & cp_w.notna()

            fig, rho, pval, strat = _make_2x2(
                ad_norm[mask].reset_index(drop=True),
                cp_w[mask].reset_index(drop=True),
                covered[mask].reset_index(drop=True),
                ad_col=ad_col, ad_dir=ad_dir,
                cp_label=interval_width_col,
                color="#2196F3", alpha_target=alpha)

            plt.suptitle(
                f"{dataset_key}: interval width vs {ad_col}\n"
                f"Spearman rho={rho:.3f}  p={pval:.4f}  "
                f"(negative = wider intervals for out-of-AD)", fontsize=11)
            plt.tight_layout()
            out_path = plots_dir / f"regr_{dataset_key}_{ad_col}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.show(); plt.close(fig)
            display(Markdown(f"Plot: `{out_path}`"))

            in_ad  = cp_w[mask][ad_norm[mask] >= 0.5]
            out_ad = cp_w[mask][ad_norm[mask] <  0.5]
            results_summary.append({
                "dataset": dataset_key, "task_type": "regression",
                "ad_col": ad_col, "direction": ad_dir,
                "n": int(mask.sum()),
                "spearman_rho": round(rho, 4), "p_value": round(pval, 6),
                "mean_width_in_AD":  round(float(in_ad.mean()),  4) if len(in_ad)  else None,
                "mean_width_out_AD": round(float(out_ad.mean()), 4) if len(out_ad) else None,
            })
elif task_type == "classification":
    display(Markdown("## Classification: set size / singleton rate vs AD index"))

    for ad_col, ad_dir in zip(ad_cols, ad_col_directions):
        if ad_col not in df.columns:
            display(Markdown(f"- AD column `{ad_col}` not found, skipping."))
            continue

        display(Markdown(f"### {ad_col}  (direction: {ad_dir})"))
        ad_norm   = _normalise_ad(df[ad_col], ad_dir)
        mask      = ad_norm.notna()
        ad_norm_c = ad_norm[mask].reset_index(drop=True)

        for approach_label, sz_col, cov_col in [
            ("A: LAC (real proba)", set_size_col_a, covered_col_a),
            ("B: NCM pseudo-proba", set_size_col_b, covered_col_b),
        ]:
            if sz_col not in df.columns:
                display(Markdown(f"  - Column `{sz_col}` not found, skipping {approach_label}."))
                continue

            display(Markdown(f"#### {approach_label}"))
            sz_c  = df.loc[mask, sz_col].astype(float).reset_index(drop=True)
            cov_c = df.loc[mask, cov_col].astype(float).reset_index(drop=True) \
                    if cov_col in df.columns \
                    else pd.Series(np.nan, index=sz_c.index)
            sing_c = (sz_c == 1).astype(float)

            fig_c, rho_sz, pval_sz, strat_c = _make_2x2(
                ad_norm_c, sz_c, cov_c,
                ad_col=ad_col, ad_dir=ad_dir,
                cp_label=sz_col,
                color="#9C27B0", alpha_target=alpha)

            # Replace Panel D (index 3) with singleton rate by quintile
            axes_list = fig_c.get_axes()
            ax_sing = axes_list[3]
            ax_sing.cla()
            strat_sing = _stratified_table(
                pd.DataFrame({"AD_norm": ad_norm_c.values,
                               "singleton": sing_c.values,
                               "cov": cov_c.values}),
                ad_norm=pd.Series(ad_norm_c.values),
                cp_col="singleton", cov_col="cov", n_bins=n_quantile_bins)
            xs = np.arange(len(strat_sing))
            ax_sing.bar(xs, strat_sing["Mean singleton"],
                        color="#FF9800", alpha=0.8, edgecolor="black")
            ax_sing.set_xticks(xs)
            ax_sing.set_xticklabels(strat_sing["AD bin"], rotation=30, ha="right", fontsize=8)
            ax_sing.set_ylabel("Singleton rate")
            ax_sing.set_title("Singleton rate by AD quintile\n"
                              "(Q5=highest AD => most singletons expected)")
            ax_sing.grid(True, alpha=0.3, axis="y")
            rho_sing, _ = stats.spearmanr(ad_norm_c.values, sing_c.values)
            ax_sing.text(0.97, 0.97, f"Spearman rho={rho_sing:.3f}",
                         transform=ax_sing.transAxes, ha="right", va="top",
                         fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            plt.suptitle(
                f"{dataset_key}: {approach_label} set size vs {ad_col}\n"
                f"Spearman rho={rho_sz:.3f}  p={pval_sz:.4f}", fontsize=11)
            plt.tight_layout()
            out_path_c = plots_dir / f"class_{dataset_key}_{ad_col}_{sz_col}.png"
            fig_c.savefig(out_path_c, dpi=150, bbox_inches="tight")
            plt.show(); plt.close(fig_c)
            display(Markdown(f"Plot: `{out_path_c}`"))

            results_summary.append({
                "dataset": dataset_key, "task_type": f"class_{approach_label}",
                "ad_col": ad_col, "direction": ad_dir,
                "n": int(mask.sum()),
                "spearman_rho_setsize":   round(rho_sz,   4),
                "p_value_setsize":        round(pval_sz,  6),
                "spearman_rho_singleton": round(rho_sing, 4),
            })

# ==============================================================================
# Summary + interpretation
# ==============================================================================
if results_summary:
    display(Markdown("## Summary: CP vs AD correlations"))
    summary_df = pd.DataFrame(results_summary)
    display(summary_df)
    display(Markdown(f"""
### Interpretation guide

| Spearman rho (similarity direction) | Interpretation |
|---|---|
| rho < -0.3, p < 0.05 | CP captures same signal as AD: wider/larger sets for out-of-AD |
| rho near 0            | CP and AD partly independent: CP adds information AD misses |
| rho > +0.3            | Unexpected -- check `ad_col_directions` in env.tutorial.yaml |

**Key regulatory message:**
CP interval width / set size correlates with AD metrics, confirming both respond to
the same underlying prediction uncertainty. CP adds what AD cannot: a *calibrated*,
statistically guaranteed bound on prediction error. The imperfect correlation reveals
cases where AD over-flags (out-of-AD but narrow CP -- e.g. BCF for large molecules)
and under-flags (in-AD but wide CP -- activity cliffs, noisy endpoints).
"""))
else:
    summary_df = pd.DataFrame()

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    summary_df.to_excel(w, sheet_name="AD_CP_correlations", index=False)

display(Markdown(f"## [OK] AD comparison complete: {dataset_key}"))
display(Markdown(f"- Summary : `{product['data']}`"))
display(Markdown(f"- Plots   : `{plots_dir}`"))
