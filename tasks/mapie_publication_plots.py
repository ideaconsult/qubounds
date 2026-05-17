# -*- coding: utf-8 -*-
"""
tasks/mapie_publication_plots.py
=================================
Publication-quality conformal prediction visualisation
=======================================================

Reads the Excel outputs already produced by mapie.py (regression) and
mapiec.py (classification) and produces a suite of alternative plots that
address the "smoothness" reviewer comment: the existing plot sorts molecules
by true endpoint value, which makes the CP band look jagged.  This task
re-plots the same data with molecules sorted by uncertainty, making the
adaptive character of the intervals immediately visible.

Regression plots
----------------
  R1  Sorted-by-width interval plot  (the primary fix)
      x = molecule rank (sorted by increasing interval width)
      y = predicted value  ±  CP interval
      Points coloured by covered/missed.  Smooth funnel by construction.

  R2  Predicted vs Observed coloured by interval width
      Traditional "pred vs obs" scatter, but colour encodes CP interval width.
      Immediately shows that wide intervals cluster in regions of poor model fit.

  R3  Interval-width distribution by AD bin  (when ADI column present)
      Violin/box plot: CP width in each ADI quintile.
      Direct visualisation of the paper's main claim about CP ↔ AD correlation.

  R4  Sigma vs interval width scatter
      σ̂(x) on x-axis, interval width on y-axis, coloured by |residual|.
      Shows that adaptive scaling works: width ∝ σ̂.

  R5  Two-model contrast panel  (optional, when two datasets are provided)
      Side-by-side R1 plots labelled "high-uncertainty model" vs
      "low-uncertainty model", echoing the TOC graphic.

Classification plots
--------------------
  C1  Prediction-set heatmap
      Rows = molecules (sorted by set size), columns = classes.
      Cells coloured by pseudo-probability; green outline = true class.
      Shows directly what "prediction set" means.

  C2  Set-size distribution by true class
      Stacked bar or violin: set-size counts per true class.
      Reveals per-class difficulty.

  C3  Singleton rate vs molecule rank
      x = molecule rank (sorted by singleton indicator or set size)
      y = cumulative singleton rate up to that rank (or just the binary flag).
      Connects to the TOC "efficiency reflects structural novelty" panel.

  C4  Confusion-matrix style heatmap with set-size overlay
      Rows = true class, columns = predicted class, cell = mean set size.
      Large set sizes on off-diagonal = genuine uncertainty; small sets on
      diagonal = confident correct predictions.

  C5  Set-size vs ADI (when ADI present)
      Analogous to R3: violin plot of set size per ADI quintile.

Parameters (Ploomber)
---------------------
  data     : path to mapie.py output Excel (or list of two)
  class_data    : path to classification mapiec.py output Excel
  alpha         : nominal miscoverage level (default 0.1)
  n_show        : max molecules to show in sorted-interval plot (default 200)
  product       : {nb, plots}
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm
from pathlib import Path
from IPython.display import display, Markdown
import pickle 
from scipy.stats import ks_2samp


# + tags=["parameters"]
data = None
alpha = 0.1
n_show = 200
model_type = None
ncm = None
cache_path = None   # path to ecfp4_cache.db  e.g. "{{CP_MODELS}}/mapie/ecfp4_cache.db"
product = None
upstream = None
# -

%matplotlib inline

# ── Output directory ──────────────────────────────────────────────────────────
out_dir  = Path(product["plots"])
out_dir.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
PALETTE = {
    "covered":    "#2196F3",   # blue
    "missed":     "#e74c3c",   # red
    "interval":   "#2196F3",
    "pred":       "#FF9800",   # orange
    "neutral":    "#607D8B",   # grey-blue
    "green":      "#27ae60",
    "background": "#F5F5F5",
    "in_set":    "#B0BEC5",   # grey  – other classes in prediction set
    "true_cov":  "#2196F3",   # blue  – true class, covered
    "true_miss": "#e74c3c",   # red   – true class, missed
    "divider":   "#78909C",   # set-size boundary lines    
}
_COLORS = {
    "CALIBRATION": "#FF9800",
    "TEST":        "#2196F3",
}

CMAP_WIDTH   = "YlOrRd"   # narrow=yellow → wide=red
CMAP_PROB    = "Blues"


def _savefig(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    display(Markdown(f"Saved: `{path}`"))
    plt.close(fig)


# =============================================================================
# Helper: load regression output
# Default sheet is the test set
# =============================================================================
def _load_regr(path, model, sheetname='Prediction Intervals'):
    """
    Load the 'Prediction Intervals' sheet written by mapie.py / predict_conformal().
    Column names follow the pattern  {model}_true, {model}_pred, {model}_lower,
    {model}_upper, {model}_sigma, Interval_Width, Smiles.
    Returns a tidy DataFrame with normalised column names.
    """
    meta = pd.read_excel(path, sheet_name="Cover sheet", header=None, index_col=0).transpose()
    df = pd.read_excel(path, sheet_name=sheetname)
    rename = {
        f"{model}_true":  "true",
        f"{model}_pred":  "pred",
        f"{model}_lower": "lower",
        f"{model}_upper": "upper",
        f"{model}_sigma": "sigma",
        "Interval_Width": "width",
    }
    df = df.rename(columns=rename)
    # Drop rows where key columns are missing
    df = df.dropna(subset=["pred", "lower", "upper", "width"]).reset_index(drop=True)
    df["has_true"]  = "true"  in df.columns and df["true"].notna().any()
    if df["has_true"].any():
        df["covered"] = (df["true"] >= df["lower"]) & (df["true"] <= df["upper"])
        df["residual"] = np.abs(df["true"] - df["pred"])
    else:
        df["covered"] = True
        df["residual"] = np.nan
    return df, meta


# =============================================================================
# R1 — Sorted-by-width interval plot
# =============================================================================
def plot_r1_sorted_by_width(df, meta, model, alpha, n_show, out_dir, suffix=""):
    """
    Molecules ranked by increasing CP interval width on x-axis.

    Visualization:
        - predicted value as point
        - CP interval as vertical error bars
        - true value as marker
    """

    import numpy as np
    import matplotlib.pyplot as plt

    df_s = df.sort_values("width").reset_index(drop=True)
    n = min(n_show, len(df_s))
    df_s = df_s.iloc[:n]

    idx = np.arange(n)

    has_true = "true" in df_s.columns and df_s["true"].notna().any()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor(PALETTE["background"])

    # ------------------------------------------------------------------
    # Error bar distances relative to prediction
    # ------------------------------------------------------------------
    yerr_lower = df_s["pred"] - df_s["lower"]
    yerr_upper = df_s["upper"] - df_s["pred"]

    # ------------------------------------------------------------------
    # Predicted values with CP intervals as error bars
    # ------------------------------------------------------------------
    ax.errorbar(
        idx,
        df_s["pred"],
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=3.5,
        color=PALETTE["pred"],
        ecolor="gray", # PALETTE["interval"],
        elinewidth=1.0,
        capsize=2,
        alpha=0.75,
        label="Prediction ± CP interval",
        zorder=3,
    )

    # ------------------------------------------------------------------
    # True values
    # ------------------------------------------------------------------
    if has_true:

        cov = df_s["covered"].values

        # Covered
        ax.scatter(
            idx[cov],
            df_s["true"].values[cov],
            s=18,
            color=PALETTE["covered"],
            alpha=0.8,
            marker="o",
            zorder=4,
            label=f"True (covered, n={cov.sum()})",
        )

        # Missed
        ax.scatter(
            idx[~cov],
            df_s["true"].values[~cov],
            s=28,
            color=PALETTE["missed"],
            alpha=0.95,
            marker="x",
            linewidths=1.5,
            zorder=5,
            label=f"True (missed, n={(~cov).sum()})",
        )

        cov_pct = cov.mean() * 100

        ax.set_title(
            f"{model} — Sorted by CP interval width "
            f"(α={alpha}, coverage={cov_pct:.1f}%)",
            fontsize=11,
        )

    else:

        ax.set_title(
            f"{model} — Sorted by CP interval width (α={alpha})",
            fontsize=11,
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    ax.set_xlabel(
        "Molecules ranked by increasing interval width →",
        fontsize=10,
    )

    units = meta["Property Units"].iloc[0]
    endpoint = meta["Property Name"].iloc[0].replace("Predicted", "")

    ax.set_ylabel(f"{endpoint}", fontsize=10)

    # ------------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------------
    ax.legend(fontsize=8, loc="upper left")

    ax.grid(
        True,
        alpha=0.25,
        linestyle="--",
    )

    # Mean width annotation
    mw = df_s["width"].mean()

    ax.annotate(
        f"Mean width = {mw:.3f}",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        bbox=dict(
            boxstyle="round",
            fc="white",
            alpha=0.8,
        ),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    path = out_dir / f"r1_sorted_width_{model}{suffix}.png"

    _savefig(fig, path)

    return path, fig


# =============================================================================
# R2 — Predicted vs Observed coloured by interval width
# =============================================================================
def plot_r2_pred_obs_colored(df, meta, model, alpha, out_dir):
    """
    Classic pred-vs-obs scatter.  Points coloured by CP interval width.
    Requires true values.
    """
    has_true = "true" in df.columns and df["true"].notna().any()
    if not has_true:
        display(Markdown(f"R2 skipped for {model}: no true values in this split."))
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df["true"], df["pred"],
                    c=df["width"], cmap=CMAP_WIDTH,
                    s=20, alpha=0.75, edgecolors="none",
                    norm=Normalize(vmin=df["width"].quantile(0.02),
                                   vmax=df["width"].quantile(0.98)))
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("CP interval width", fontsize=9)

    lo = min(df["true"].min(), df["pred"].min())
    hi = max(df["true"].max(), df["pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x (ideal)")

    cov_pct = df["covered"].mean() * 100
    ax.set_title(f"{model} — Pred vs Obs, coloured by CP width\n"
                 f"(α={alpha}, coverage={cov_pct:.1f}%)", fontsize=11)
    ax.set_xlabel("True value", fontsize=10)
    ax.set_ylabel("Predicted value", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")

    path = out_dir / f"r2_pred_obs_width_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# R3 — Interval-width distribution by ADI bin
# =============================================================================
def plot_r3_width_by_adi(df, meta, model, alpha, out_dir, adi_col="ADI", n_bins=5):
    """
    Violin / box: CP interval width across ADI quantile bins.
    Skipped when ADI column is absent.
    """
    if adi_col not in df.columns or df[adi_col].isna().all():
        display(Markdown(f"R3 skipped for {model}: no `{adi_col}` column."))
        return None

    df = df.dropna(subset=[adi_col, "width"]).copy()
    try:
        df["adi_bin"] = pd.qcut(df[adi_col], q=n_bins,
                                labels=[f"Q{i+1}" for i in range(n_bins)],
                                duplicates="drop")
    except ValueError:
        display(Markdown(f"R3 skipped for {model}: not enough unique ADI values."))
        return None

    bin_order = sorted(df["adi_bin"].dropna().unique())
    groups    = [df.loc[df["adi_bin"] == b, "width"].values for b in bin_order]
    sizes     = [len(g) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    vp = ax.violinplot(groups, positions=range(len(bin_order)),
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(PALETTE["interval"])
        body.set_alpha(0.55)
    vp["cmedians"].set_color(PALETTE["pred"])
    vp["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(
        [f"{b}\n(n={sizes[i]})" for i, b in enumerate(bin_order)],
        fontsize=9)
    ax.set_xlabel(f"ADI quintile  (Q1 = lowest similarity / most out-of-domain)",
                  fontsize=9)
    ax.set_ylabel("CP interval width", fontsize=10)
    ax.set_title(f"{model} — CP width by ADI quintile  (α={alpha})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Overlay mean coverage per bin if true available
    has_true = "true" in df.columns and df["true"].notna().any()
    if has_true:
        ax2 = ax.twinx()
        cov_by_bin = [df.loc[df["adi_bin"] == b, "covered"].mean()
                      for b in bin_order]
        ax2.plot(range(len(bin_order)), cov_by_bin,
                 "rs-", lw=2, ms=8, label="Coverage")
        ax2.axhline(1 - alpha, color="red", linestyle="--", lw=1.5,
                    label=f"Target {1-alpha:.0%}")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Coverage", fontsize=9)
        ax2.legend(loc="upper left", fontsize=8)

    path = out_dir / f"r3_width_by_adi_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# R4 — σ̂(x) vs interval width scatter
# =============================================================================
def plot_r4_sigma_vs_width(df, meta, model, alpha, out_dir):
    """
    σ̂(x) on x-axis, interval width on y-axis, coloured by |residual|.
    Shows that width ∝ σ̂ (adaptive scaling works).
    Requires sigma column.
    """
    if "sigma" not in df.columns or df["sigma"].isna().all():
        display(Markdown(f"R4 skipped for {model}: no sigma column."))
        return None

    df_s = df.dropna(subset=["sigma", "width"]).copy()
    has_resid = "residual" in df_s.columns and df_s["residual"].notna().any()

    fig, ax = plt.subplots(figsize=(6, 5))
    if has_resid:
        sc = ax.scatter(df_s["sigma"], df_s["width"],
                        c=df_s["residual"], cmap="plasma",
                        s=14, alpha=0.65, edgecolors="none",
                        norm=Normalize(vmin=0,
                                       vmax=df_s["residual"].quantile(0.95)))
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("|residual|", fontsize=9)
    else:
        ax.scatter(df_s["sigma"], df_s["width"],
                   s=14, alpha=0.55, color=PALETTE["covered"])

    # y = 2·q̂·σ̂ reference line  (width = 2·q̂·σ̂ for adaptive CP)
    sig_range = np.linspace(df_s["sigma"].min(), df_s["sigma"].max(), 100)
    # Back out q_hat from observed widths and sigmas
    valid = df_s["sigma"] > 0
    q_hat_est = np.median(df_s.loc[valid, "width"] /
                          (2 * df_s.loc[valid, "sigma"]))
    ax.plot(sig_range, 2 * q_hat_est * sig_range,
            "k--", lw=1.5, alpha=0.7,
            label=f"width = 2·q̂·σ̂  (q̂≈{q_hat_est:.3f})")

    ax.set_xlabel("σ̂(x)  — predicted uncertainty", fontsize=10)
    ax.set_ylabel("CP interval width", fontsize=10)
    ax.set_title(f"{model} — Adaptive scaling: σ̂ vs width  (α={alpha})",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")

    path = out_dir / f"r4_sigma_width_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# R5 — Two-model contrast panel
# =============================================================================
def plot_r5_contrast(df_a, meta, model_a, df_b, model_b, alpha, n_show, out_dir):
    """
    Side-by-side sorted-by-width plots for two models.
    Echoes the TOC graphic: narrow/confident left, wide/uncertain right.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, df, model, label in [
        (axes[0], df_a, model_a, "Low uncertainty"),
        (axes[1], df_b, model_b, "High uncertainty"),
    ]:
        df_s = df.sort_values("width").reset_index(drop=True)
        n    = min(n_show, len(df_s))
        df_s = df_s.iloc[:n]
        idx  = np.arange(n)

        ax.set_facecolor(PALETTE["background"])
        ax.fill_between(idx, df_s["lower"], df_s["upper"],
                        alpha=0.35, color=PALETTE["interval"],
                        label="CP interval")
        ax.plot(idx, df_s["pred"], color=PALETTE["pred"], lw=1.2, alpha=0.85,
                label="Predicted ŷ")

        has_true = "true" in df_s.columns and df_s["true"].notna().any()
        if has_true:
            cov  = df_s["covered"].values
            ax.scatter(idx[cov],  df_s["true"].values[cov],
                       s=10, color=PALETTE["covered"], alpha=0.7, zorder=4)
            ax.scatter(idx[~cov], df_s["true"].values[~cov],
                       s=20, color=PALETTE["missed"], alpha=0.9,
                       marker="x", lw=1.4, zorder=5)
            cov_pct = cov.mean() * 100
            cov_label = f"Coverage = {cov_pct:.1f}%"
        else:
            cov_label = ""

        mw = df_s["width"].mean()
        ax.set_title(f"{label}\n{model}  (mean width = {mw:.3f},  {cov_label})",
                     fontsize=10)
        ax.set_xlabel("Molecules ranked by increasing interval width →", fontsize=9)
        units = meta["Property Units"].iloc[0]
        endpoint = meta["Property Name"].iloc[0]    
        ax.set_ylabel(f"{endpoint} {units}", fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=7, loc="upper left")

    plt.suptitle(f"Adaptive CP: interval width reflects model uncertainty  (α={alpha})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"r5_contrast_{model_a}_vs_{model_b}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# Helper: load classification output
# =============================================================================
def _load_class(path, model, sheetname='Prediction Intervals'):
    """
    Load classification output from mapiec.py / predict_conformal_classifier_chunked().
    Expected columns (prefix = model name):
      {model}_true, {model}_pred, Set Size 
      p_{class}, pseudo_p_{class}  (one per class)
    Falls back gracefully if columns use alternative naming.
    """
    meta = pd.read_excel(path, sheet_name="Cover sheet", header=None, index_col=0).transpose()

    df = pd.read_excel(path, sheet_name=sheetname)

    # add singleton col based on set size
    # Standardise column names
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in (f"{model.lower()}_true", "experimental", "exp", "true"):
            rename[c] = "true"
        elif lc in (f"{model.lower()}_pred", "predicted", "pred", "hard_pred"):
            rename[c] = "pred"
        elif "set_size" in lc:
            rename[c] = "set_size"
        elif "singleton" in lc:
            rename[c] = "singleton"
        elif lc.startswith("pseudo_p_") or lc.startswith("p_pseudo"):
            rename[c] = c  # keep, extract later
        elif lc.startswith(("p_", "prob_", "probability_")):
            rename[c] = c  # keep

    df = df.rename(columns=rename)
    df = df.dropna(subset=["pred"]).reset_index(drop=True)

    has_true = "true" in df.columns and df["true"].notna().any()
    if has_true:
        df["true"] = df["true"].astype(int, errors="ignore")
        df["correct"] = (df["true"] == df["pred"]).astype(int)
    df["has_true"] = has_true

    # Detect class columns
    pseudo_cols = [c for c in df.columns
                   if c.lower().startswith("pseudo_p_") or "pseudo" in c.lower()]
    df.attrs["pseudo_cols"] = pseudo_cols

    return df, meta

# =============================================================================
# C1 — Classification dot-plot (analogue of regression R1)
# =============================================================================

def plot_c1_dotplot(
    df,                  # DataFrame from _load_class()  ← was `data` (dict)
    meta,
    model,
    alpha,
    out_dir,
    class_labels=None,   # override display labels, e.g. ["Non-mutagen","Mutagen"]
    n_show=1000,
    sheet=None,
):
    """
    Classification analogue of R1.  Works directly from the DataFrame
    returned by _load_class() — no intermediate dict required.
 
    x-axis  : molecule rank (sorted by set_size, then predicted class)
    y-axis  : class (discrete)
    Grey vertical bar  : prediction set span (min to max class in set)
    Grey dots          : other classes in prediction set
    Orange dot         : predicted class ŷ
    Blue dot           : true class (covered)
    Red X              : true class (missed)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if "set_size" not in df.columns:
        display(Markdown(f"C1 skipped for {model}: no set_size column."))
        return None
 
    # ── derive class list from in_set_class_* columns ─────────────────────────
    inset_cols = sorted([c for c in df.columns if c.startswith("in_set_class_")])
 
    def _parse_inset_cls(col):
        sfx = col.replace("in_set_class_", "")
        try:
            v = float(sfx)
            return int(v) if v == int(v) else v
        except ValueError:
            return sfx
 
    if inset_cols:
        classes_orig = [_parse_inset_cls(c) for c in inset_cols]
    else:
        vals = list(df["pred"].dropna().unique())
        if df["has_true"].any():
            vals += list(df["true"].dropna().unique())
        try:
            classes_orig = sorted(set(int(v) for v in vals))
        except Exception:
            classes_orig = sorted(set(str(v) for v in vals))
 
    n_classes  = len(classes_orig)
    cls_to_idx = {c: i for i, c in enumerate(classes_orig)}
 
    if class_labels is not None:
        assert len(class_labels) == n_classes, \
            f"class_labels length {len(class_labels)} != n_classes {n_classes}"
        disp_labels = class_labels
    else:
        disp_labels = [str(c) for c in classes_orig]
 
    classes_idx = np.arange(n_classes)
 
    # ── coerce pred / true to contiguous index ────────────────────────────────
    def _to_idx(v):
        try:
            return cls_to_idx.get(int(float(v)), cls_to_idx.get(v, -1))
        except (TypeError, ValueError):
            return cls_to_idx.get(v, -1)
 
    y_pred_idx = np.array([_to_idx(v) for v in df["pred"].values])
    valid_pred = y_pred_idx >= 0
    df_v       = df[valid_pred].reset_index(drop=True)
    y_pred_idx = y_pred_idx[valid_pred]
 
    has_true = df_v["has_true"].any() if "has_true" in df_v.columns else False
    y_true_idx = (
        np.array([_to_idx(v) for v in df_v["true"].values])
        if has_true else None
    )
 
    sz = df_v["set_size"].values.astype(int)
 
    # ── build in_set boolean matrix (n_raw, n_classes) ────────────────────────
    n_raw  = len(df_v)
    in_set = np.zeros((n_raw, n_classes), dtype=bool)
    if inset_cols:
        for col in inset_cols:
            j = cls_to_idx.get(_parse_inset_cls(col), -1)
            if j >= 0:
                in_set[:, j] = df_v[col].values.astype(bool)
    else:
        in_set[np.arange(n_raw), y_pred_idx] = True  # fallback
    in_set[np.arange(n_raw), y_pred_idx] = True      # always include predicted
 
    # ── coverage ──────────────────────────────────────────────────────────────
    covered_arr = (
        in_set[np.arange(n_raw), y_true_idx]
        if y_true_idx is not None else None
    )
 
    # ── sort: primary = set_size, secondary = predicted class ─────────────────
    sort_key = sz * 1000 + y_pred_idx
    order    = np.argsort(sort_key, kind="stable")
 
    y_pred_s  = y_pred_idx[order]
    y_true_s  = y_true_idx[order]  if y_true_idx  is not None else None
    covered_s = covered_arr[order] if covered_arr is not None else None
    in_set_s  = in_set[order]
    sz_s      = sz[order]
 
    n = min(n_show, n_raw)
    y_pred_s  = y_pred_s[:n]
    y_true_s  = y_true_s[:n]  if y_true_s  is not None else None
    covered_s = covered_s[:n] if covered_s is not None else None
    in_set_s  = in_set_s[:n]
    sz_s      = sz_s[:n]
 
    has_true = y_true_s is not None

    x      = np.arange(n)
    rng    = np.random.default_rng(0)
    jitter = rng.uniform(-0.07, 0.07, n)

    # ── figure ────────────────────────────────────────────────────────────────
    fig_h = max(3.0, 1.0 + n_classes * 0.7)
    fig, ax = plt.subplots(figsize=(min(16, max(10, n / 12)), fig_h))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    # ── grey vertical bar = prediction set span ───────────────────────────────
    for i in range(n):
        cls_in = [c for c in classes_idx if in_set_s[i, c]]
        if len(cls_in) > 1:
            ax.plot([x[i], x[i]], [min(cls_in), max(cls_in)],
                    color=PALETTE["in_set"], lw=2.5, zorder=1,
                    solid_capstyle="round")

    # ── grey dots: other classes in prediction set ────────────────────────────
    for cls in classes_idx:
        mask = in_set_s[:, cls] & (cls != y_pred_s)
        if has_true:
            # don't draw grey where true class will be drawn
            mask = mask & (cls != y_true_s)
        if mask.any():
            ax.scatter(x[mask], np.full(mask.sum(), cls) + jitter[mask],
                       s=45, color=PALETTE["in_set"], zorder=2,
                       linewidths=0, alpha=0.8)

    # ── predicted class (orange) ──────────────────────────────────────────────
    ax.scatter(x, y_pred_s + jitter,
               s=85, color=PALETTE["pred"], zorder=4,
               label="Predicted class ŷ",
               linewidths=0.5, edgecolors="white")

    # ── true class ────────────────────────────────────────────────────────────
    if has_true:
        cov_m  = covered_s.astype(bool)
        miss_m = ~cov_m

        if cov_m.any():
            ax.scatter(x[cov_m], y_true_s[cov_m] + jitter[cov_m],
                       s=55, color=PALETTE["true_cov"], zorder=5,
                       label=f"True (covered, n={cov_m.sum()})",
                       linewidths=0.4, edgecolors="white")
        if miss_m.any():
            ax.scatter(x[miss_m], y_true_s[miss_m] + jitter[miss_m],
                       s=80, color=PALETTE["true_miss"], marker="X", zorder=6,
                       label=f"True (missed, n={miss_m.sum()})",
                       linewidths=0)

    # ── set-size dividers ─────────────────────────────────────────────────────
    seen_dividers = set()
    for sz_val in range(2, n_classes + 1):
        first = np.where(sz_s == sz_val)[0]
        if len(first) and sz_val not in seen_dividers:
            xi = first[0] - 0.5
            ax.axvline(xi, color=PALETTE["divider"], lw=1,
                       linestyle="--", alpha=0.55)
            ax.text(xi + 0.3, n_classes - 0.55,
                    f"set size={sz_val}",
                    fontsize=7, color=PALETTE["divider"], va="top")
            seen_dividers.add(sz_val)

    # ── annotation box: counts per set size ───────────────────────────────────
    sz_labels = {1: "singleton", 2: "doubleton", n_classes: "full set"}
    ann_lines = []
    for sz_val in range(1, n_classes + 1):
        cnt = (sz_s == sz_val).sum()
        if cnt == 0:
            continue
        lbl = sz_labels.get(sz_val, f"size={sz_val}")
        ann_lines.append(f"{lbl}: {cnt} ({cnt/n*100:.0f}%)")
    ax.text(0.99, 0.03, "\n".join(ann_lines),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_yticks(classes_idx)
    ax.set_yticklabels(disp_labels, fontsize=9)
    ax.set_ylabel("Class", fontsize=10)
    ax.set_xlabel("Molecules ranked by increasing prediction set size →", fontsize=10)
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.5, n_classes - 0.5)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    if has_true:
        cov_pct = covered_s.mean() * 100
        title = (f"{model} — Prediction sets sorted by size  "
                 f"(α={alpha}, coverage={cov_pct:.1f}%)")
    else:
        title = f"{model} — Prediction sets sorted by size  (α={alpha})"

    ax.set_title(
        title + "\nGrey bar = prediction set span · "
        "Orange = predicted · Blue/Red = true (covered/missed)",
        fontsize=10
    )

    plt.tight_layout()
    path = out_dir / f"c1_dotplot_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# C2 — Set-size distribution by true class
# =============================================================================
def plot_c2_setsize_by_class(df, meta, model, alpha, out_dir, class_names=None):
    """
    Violin plot: CP set size per true class.
    Reveals which classes are harder for the model.
    """
    has_true = df["has_true"].any() if "has_true" in df.columns else False
    if not has_true or "set_size" not in df.columns:
        display(Markdown(f"C2 skipped for {model}: need true labels and set_size."))
        return None

    df = df.dropna(subset=["true", "set_size"]).copy()
    if class_names is None:
        try:
            class_names = sorted(df["true"].unique().astype(int))
        except Exception:
            class_names = sorted(df["true"].unique())

    groups = []
    for cls in class_names:
        g = df.loc[df["true"] == cls, "set_size"].values
        groups.append(g)

    valid = [(g, c) for g, c in zip(groups, class_names) if len(g) > 1]
    if not valid:
        display(Markdown(f"C2: not enough data per class."))
        return None
    groups_v, cls_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(5, len(cls_v) * 1.5), 5))
    vp = ax.violinplot(list(groups_v), positions=range(len(cls_v)),
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(PALETTE["interval"])
        body.set_alpha(0.5)
    vp["cmedians"].set_color(PALETTE["pred"])
    vp["cmedians"].set_linewidth(2)

    # Overlay mean coverage per class
    ax2 = ax.twinx()
    cov_by_class = [df.loc[df["true"] == c, "correct"].mean()
                    if "correct" in df else np.nan
                    for c in cls_v]
    ax2.plot(range(len(cls_v)), cov_by_class, "rs-", ms=8, lw=2, label="Accuracy")
    ax2.axhline(1 - alpha, color="red", linestyle="--", lw=1.5,
                label=f"CP target {1-alpha:.0%}")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Per-class accuracy / coverage", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")

    n_per_class = [len(g) for g in groups_v]
    ax.set_xticks(range(len(cls_v)))
    ax.set_xticklabels(
        [f"{c}\n(n={n_per_class[i]})" for i, c in enumerate(cls_v)],
        fontsize=9)
    ax.set_xlabel("True class", fontsize=10)
    ax.set_ylabel("CP prediction set size", fontsize=10)
    ax.set_title(f"{model} — Set size by true class  (α={alpha})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    path = out_dir / f"c2_setsize_by_class_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# C3 — Singleton rate: molecules sorted by set size
# =============================================================================
def plot_c3_singleton_rank(df, meta, model, alpha, out_dir):
    """
    x = molecule rank (sorted by set size ascending)
    y = set size value (1 = singleton, 2 = doubleton, …)
    Colour encodes whether true class is covered.
    Right panel: cumulative singleton rate.
    Echoes the TOC "efficiency reflects structural novelty" panel.
    """
    if "set_size" not in df.columns:
        display(Markdown(f"C3 skipped for {model}: no set_size column."))
        return None

    df_s = df.sort_values("set_size").reset_index(drop=True)
    idx  = np.arange(len(df_s))
    has_true = df_s["has_true"].any() if "has_true" in df_s.columns else False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Left: set-size staircase
    if has_true and "correct" in df_s.columns:
        c_correct = df_s["correct"].values
        ax1.scatter(idx[c_correct == 1], df_s["set_size"].values[c_correct == 1],
                    s=10, color=PALETTE["covered"], alpha=0.65, label="Correct pred")
        ax1.scatter(idx[c_correct == 0], df_s["set_size"].values[c_correct == 0],
                    s=20, color=PALETTE["missed"], alpha=0.8, marker="x", lw=1.2,
                    label="Wrong pred")
    else:
        ax1.scatter(idx, df_s["set_size"].values,
                    s=8, color=PALETTE["neutral"], alpha=0.55)

    ax1.axhline(1, color=PALETTE["green"], lw=1.5, linestyle="--",
                label="Singleton (ideal)")
    sing_pct = (df_s["set_size"] == 1).mean() * 100
    ax1.set_title(f"{model} — Set size per molecule\n"
                  f"(singleton rate = {sing_pct:.1f}%)", fontsize=10)
    ax1.set_xlabel("Molecules ranked by set size →", fontsize=9)
    ax1.set_ylabel("Prediction set size", fontsize=9)
    ax1.set_yticks(sorted(df_s["set_size"].unique()))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25, linestyle="--")

    # Right: cumulative singleton rate
    cum_sing = (df_s["set_size"] == 1).cumsum().values / (idx + 1)
    ax2.plot(idx, cum_sing, color=PALETTE["covered"], lw=2)
    ax2.axhline(sing_pct / 100, color=PALETTE["pred"], lw=1.5, linestyle="--",
                label=f"Overall singleton rate = {sing_pct:.1f}%")
    ax2.set_xlabel("Molecules ranked by set size →", fontsize=9)
    ax2.set_ylabel("Cumulative singleton rate", fontsize=9)
    ax2.set_title("Cumulative singleton rate", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25, linestyle="--")

    plt.suptitle(f"{model} — CP efficiency  (α={alpha})", fontsize=12)
    plt.tight_layout()

    path = out_dir / f"c3_singleton_rank_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# C4 — Confusion-matrix heatmap with mean set-size overlay
# =============================================================================
def plot_c4_confusion_setsize(df, meta, model, alpha, out_dir, class_names=None):
    """
    Rows = true class, columns = predicted class.
    Cell = count (text) + mean set size (colour).
    Large set sizes on off-diagonal = genuine uncertainty.
    Small sets on diagonal = confident correct predictions.
    """
    has_true = df["has_true"].any() if "has_true" in df.columns else False
    if not has_true or "set_size" not in df.columns:
        display(Markdown(f"C4 skipped for {model}: need true labels and set_size."))
        return None

    df = df.dropna(subset=["true", "pred", "set_size"]).copy()
    if class_names is None:
        try:
            class_names = sorted(set(
                list(df["true"].unique()) + list(df["pred"].unique())
            ))
        except Exception:
            class_names = sorted(set(
                [str(v) for v in df["true"].unique()] +
                [str(v) for v in df["pred"].unique()]
            ))

    n_cls = len(class_names)
    count_mat  = np.zeros((n_cls, n_cls), dtype=int)
    size_mat   = np.full((n_cls, n_cls), np.nan)

    for i, tc in enumerate(class_names):
        for j, pc in enumerate(class_names):
            mask = (df["true"] == tc) & (df["pred"] == pc)
            count_mat[i, j]  = mask.sum()
            if mask.sum() > 0:
                size_mat[i, j] = df.loc[mask, "set_size"].mean()

    fig, ax = plt.subplots(figsize=(max(5, n_cls * 1.2), max(5, n_cls * 1.0)))
    masked_size = np.ma.masked_invalid(size_mat)
    im = ax.imshow(masked_size, cmap=CMAP_WIDTH, aspect="auto",
                   vmin=1, vmax=n_cls)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean CP set size")

    for i in range(n_cls):
        for j in range(n_cls):
            cnt = count_mat[i, j]
            if cnt == 0:
                continue
            sz  = size_mat[i, j]
            txt = f"{cnt}\n(sz={sz:.1f})"
            # White text on dark cells
            bg  = masked_size[i, j] if not np.isnan(size_mat[i, j]) else 1
            col = "white" if bg > (n_cls * 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")

    # Diagonal highlight
    for k in range(n_cls):
        rect = mpatches.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   linewidth=2.5,
                                   edgecolor=PALETTE["green"], facecolor="none")
        ax.add_patch(rect)

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels([str(c) for c in class_names], fontsize=9)
    ax.set_yticklabels([str(c) for c in class_names], fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
    ax.set_title(f"{model} — Confusion matrix with mean CP set size  (α={alpha})\n"
                 "Cell: count / (mean set size)  —  green border = diagonal",
                 fontsize=10)

    plt.tight_layout()
    path = out_dir / f"c4_confusion_setsize_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# C5 — Set size by ADI bin
# =============================================================================
def plot_c5_setsize_by_adi(df, meta, model, alpha, out_dir, adi_col="ADI", n_bins=4):
    """
    Violin: CP set size per ADI quantile.  Analogous to R3 for classification.
    """
    if adi_col not in df.columns or df[adi_col].isna().all():
        display(Markdown(f"C5 skipped for {model}: no `{adi_col}` column."))
        return None
    if "set_size" not in df.columns:
        display(Markdown(f"C5 skipped for {model}: no set_size column."))
        return None

    df = df.dropna(subset=[adi_col, "set_size"]).copy()
    try:
        df["adi_bin"] = pd.qcut(df[adi_col], q=n_bins,
                                labels=[f"Q{i+1}" for i in range(n_bins)],
                                duplicates="drop")
    except ValueError:
        display(Markdown(f"C5 skipped for {model}: not enough unique ADI values."))
        return None

    bin_order = sorted(df["adi_bin"].dropna().unique())
    groups    = [df.loc[df["adi_bin"] == b, "set_size"].values for b in bin_order]
    sizes     = [len(g) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    vp = ax.violinplot(groups, positions=range(len(bin_order)),
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(PALETTE["interval"])
        body.set_alpha(0.55)
    vp["cmedians"].set_color(PALETTE["pred"])
    vp["cmedians"].set_linewidth(2)

    # Singleton rate overlay
    ax2 = ax.twinx()
    sing_by_bin = [(df.loc[df["adi_bin"] == b, "set_size"] == 1).mean()
                   for b in bin_order]
    ax2.plot(range(len(bin_order)), sing_by_bin,
             "gs-", lw=2, ms=8, label="Singleton rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Singleton rate", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(
        [f"{b}\n(n={sizes[i]})" for i, b in enumerate(bin_order)],
        fontsize=9)
    ax.set_xlabel("ADI quantile  (Q1 = lowest similarity / most out-of-domain)",
                  fontsize=9)
    ax.set_ylabel("CP prediction set size", fontsize=10)
    ax.set_title(f"{model} — CP set size by ADI quantile  (α={alpha})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    path = out_dir / f"c5_setsize_by_adi_{model}.png"
    _savefig(fig, path)
    return path


# =============================================================================
# E1 — Regression exchangeability plot
# =============================================================================
def _load_ncm_scores_from_excel(excel_path, model):
    """
    Load regression calibration nonconformity scores from the Excel.
 
    Priority
    --------
    1. Sheet "Calibration"  — exact scores used by MAPIE for the quantile.
    2. Sheet "Training"     — used when calibration sheet is absent.
    3. Recompute from {model}_true / _pred / _sigma in whichever sheet matched.
 
    Returns (scores_1d_float_array, sheet_name_string)
    Raises RuntimeError if nothing works.
    """
    xl      = pd.ExcelFile(excel_path)
    ncm_col = f"{model}_ncm"
 
    for sheet in ["Calibration", "Training"]:
        if sheet not in xl.sheet_names:
            continue
        df_s = pd.read_excel(excel_path, sheet_name=sheet)
 
        # Direct column
        if ncm_col in df_s.columns:
            scores = df_s[ncm_col].dropna().values.astype(float)
            if len(scores) > 0:
                return scores, sheet
 
        # Recompute from true / pred / sigma
        true_col  = f"{model}_true"
        pred_col  = f"{model}_pred"
        sigma_col = f"{model}_sigma"
        if all(c in df_s.columns for c in [true_col, pred_col, sigma_col]):
            eps        = 1e-6
            sigma_safe = np.maximum(df_s[sigma_col].values.astype(float), eps)
            scores     = (
                np.abs(df_s[true_col].values.astype(float) -
                       df_s[pred_col].values.astype(float)) / sigma_safe
            )
            scores = scores[np.isfinite(scores)]
            if len(scores) > 0:
                return scores, f"{sheet} (recomputed)"
 
    raise RuntimeError(
        f"No calibration NCM scores found for '{model}' in {excel_path}. "
        "Expected a 'Calibration' or 'Training' sheet with a "
        f"'{ncm_col}' column, or '{model}_true'/'{model}_pred'/'{model}_sigma' columns."
    )
 
 
def plot_e1_regression_exchangeability(
    regr_excel_path, regr_pickle_path, model, alpha, out_dir
):
    """
    Left  : calibration vs test nonconformity score distributions (KS test).
    Right : conformal p-value distribution vs Uniform(0,1) (KS test).
 
    Cal scores  : Excel "Calibration" sheet  {model}_ncm
                  → fallback Excel "Training" sheet
                  → fallback pickle _mapie_regressor.conformity_scores_
    Test scores : Excel "Prediction Intervals" sheet  {model}_ncm
                  → fallback recomputed from true/pred/sigma
    """
    out_dir = Path(out_dir)
 
    # ── Calibration scores: Excel first, pickle last ───────────────────────────
    try:
        scores_cal, cal_source = _load_ncm_scores_from_excel(regr_excel_path, model)
        display(Markdown(
            f"  Cal scores from Excel sheet '{cal_source}': n={len(scores_cal)}, "
            f"range=[{scores_cal.min():.3f}, {scores_cal.max():.3f}]"
        ))
    except RuntimeError as exc_excel:
        display(Markdown(
            f"  Excel cal scores unavailable ({exc_excel}); trying pickle."
        ))
        try:
            with open(regr_pickle_path, "rb") as fh:
                saved = pickle.load(fh)
            raw        = saved["mapie"]._mapie_regressor.conformity_scores_
            scores_cal = raw[~np.isnan(raw)]
            cal_source = "pickle"
            display(Markdown(
                f"  Cal scores from pickle: n={len(scores_cal)}, "
                f"range=[{scores_cal.min():.3f}, {scores_cal.max():.3f}]"
            ))
        except Exception as exc_pkl:
            display(Markdown(
                f"E1 skipped for {model}: cannot load calibration scores — {exc_pkl}"
            ))
            return None
 
    # ── Test scores from "Prediction Intervals" sheet ─────────────────────────
    df      = pd.read_excel(regr_excel_path, sheet_name="Prediction Intervals")
    ncm_col = f"{model}_ncm"
 
    if ncm_col in df.columns:
        scores_test = df[ncm_col].dropna().values.astype(float)
    else:
        true_col  = f"{model}_true"
        pred_col  = f"{model}_pred"
        sigma_col = f"{model}_sigma"
        if all(c in df.columns for c in [true_col, pred_col, sigma_col]):
            eps         = 1e-6
            sigma_safe  = np.maximum(df[sigma_col].values.astype(float), eps)
            scores_test = (
                np.abs(df[true_col].values.astype(float) -
                       df[pred_col].values.astype(float)) / sigma_safe
            )
            display(Markdown(
                f"Test `{ncm_col}` not found; recomputed from true/pred/sigma."
            ))
        else:
            display(Markdown(
                f"E1 skipped for {model}: no `{ncm_col}` in 'Prediction Intervals' "
                "and cannot recompute (true/pred/sigma columns missing)."
            ))
            return None
 
    # ── Conformal p-values ─────────────────────────────────────────────────────
    p_values = np.array([np.mean(scores_cal >= s_i) for s_i in scores_test])
 
    # ── KS tests ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    ks_stat_dist, ks_p_dist = ks_2samp(scores_cal, scores_test)
    ks_stat_pval, ks_p_pval = ks_2samp(p_values, rng.uniform(size=len(p_values)))
 
    msg_dist = ("OK — no evidence of distributional shift"
                if ks_p_dist > 0.05 else
                "WARNING — possible shift (coverage may deviate)")
    msg_pval = ("OK — p-values consistent with uniformity"
                if ks_p_pval > 0.05 else
                "WARNING — p-values not uniform (exchangeability suspect)")
 
    display(Markdown(
        f"### Regression exchangeability: {model}  (cal from: {cal_source})\n"
        f"- Cal vs Test NCM: KS stat={ks_stat_dist:.4f}, p={ks_p_dist:.4f}  →  {msg_dist}\n"
        f"- P-value unif.:   KS stat={ks_stat_pval:.4f}, p={ks_p_pval:.4f}  →  {msg_pval}"
    ))
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_dist, ax_pval) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE["background"])
 
    ax_dist.hist(scores_cal,  bins="auto", density=True, alpha=0.55,
                 color=_COLORS["CALIBRATION"],
                 label=f"Calibration — {cal_source} (n={len(scores_cal)})")
    ax_dist.hist(scores_test, bins="auto", density=True, alpha=0.55,
                 color=_COLORS["TEST"],
                 label=f"Test (n={len(scores_test)})")
    ax_dist.set_xlabel("Nonconformity score  s = |y−ŷ| / σ̂(x)", fontsize=10)
    ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.set_title(
        f"Cal vs Test NCM scores\nKS p = {ks_p_dist:.4f}  —  {msg_dist}", fontsize=9)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.25, linestyle="--")
 
    ax_pval.hist(p_values, bins=20, density=True, alpha=0.7,
                 color=_COLORS["TEST"],
                 label=f"Conformal p-values (n={len(p_values)})")
    ax_pval.axhline(1.0, color="red", linestyle="--", lw=2,
                    label="Uniform(0,1) reference")
    ax_pval.set_xlabel("p_i = #{s_cal ≥ s_i} / n_cal", fontsize=10)
    ax_pval.set_ylabel("Density", fontsize=9)
    ax_pval.set_title(
        f"P-value uniformity (KS vs Uniform)\nKS p = {ks_p_pval:.4f}  —  {msg_pval}",
        fontsize=9)
    ax_pval.set_xlim(0, 1)
    ax_pval.legend(fontsize=8)
    ax_pval.grid(True, alpha=0.25, linestyle="--")
 
    plt.suptitle(
        f"{model} — Regression exchangeability diagnostics  (α={alpha})\n"
        "Under exchangeability, cal and test distributions should match "
        "and p-values should be uniform.",
        fontsize=11)
    plt.tight_layout()
 
    path = out_dir / f"e1_exchangeability_regr_{model}.png"
    _savefig(fig, path)
    return path


def _get_cal_scores_classifier(mapie_obj):
    """
    Extract calibration conformity scores from SplitConformalClassifier.
    Correct path (confirmed by introspection):
      mapie._mapie_classifier.conformity_scores_   shape (n_cal, 1)
    Returns 1-D array.
    """
    raw = mapie_obj._mapie_classifier.conformity_scores_
    return raw.squeeze()


def _recompute_test_scores(saved, df_excel, model, cache_path):
    """
    Compute test LAC scores: s_i = 1 - p_pseudo(y_true_i | x_i).

    Route A — tutorial Excel (has pseudo_p_* columns):
        No ECFP recomputation needed.

    Route B — main-pipeline Excel (has in_set_class_* columns):
        Reconstruct pseudo-probabilities from the NCM distance model
        already stored in the pickle + ECFP fingerprints.
        Requires Smiles column + cache_path.

    Route C — in_set_class_* + true label available (no ECFP needed):
        Use 1 - in_set score proxy: s_i = 1 - p_pseudo derived from
        saved calibration quantile.  Approximate but avoids cache_path.
        Used as fallback when cache_path is None.

    Returns
    -------
    scores_test   : 1-D float array  (n_test,)
    y_true_mapped : 1-D int array    (n_test,)  contiguous class indices
    class_order   : list of original class labels (for display)
    route         : str
    """
    sigma_model   = saved["sigma_model"]
    ncm_type      = saved["ncm"]
    classes       = saved["classes"]           # contiguous 0-based ints
    classes_orig  = saved["classes_original"]
    class_to_map  = saved["class_to_mapped"]
    mapped_to_cls = saved["mapped_to_class"]

    # ── detect true-label column ───────────────────────────────────────────────
    true_col_candidates = [f"{model}_true", "True", "true", "Experimental", "Exp", "exp"]
    true_col = next((c for c in true_col_candidates if c in df_excel.columns), None)
    has_true = true_col is not None and df_excel[true_col].notna().any()

    # ── Route A: tutorial Excel with pseudo_p_* columns ───────────────────────
    pseudo_cols = [c for c in df_excel.columns if "pseudo_p_" in c.lower()]

    if pseudo_cols and has_true:
        display(Markdown(
            f"  Route A: reconstructing test scores from `pseudo_p_*` columns."
        ))

        def _parse_cls(col):
            sfx = col.lower().split("pseudo_p_")[-1]
            try:
                return int(sfx)
            except ValueError:
                return sfx

        col_cls_order = [_parse_cls(c) for c in pseudo_cols]
        cls_to_col    = {cls: i for i, cls in enumerate(col_cls_order)}

        y_true_raw = df_excel[true_col].dropna().values
        proba_mat  = df_excel[pseudo_cols].values.astype(float)

        valid = np.array([y in cls_to_col for y in y_true_raw])
        y_tv  = y_true_raw[valid]
        pp_v  = proba_mat[valid]
        col_i = np.array([cls_to_col[y] for y in y_tv])
        scores_test   = 1.0 - pp_v[np.arange(len(y_tv)), col_i]
        y_true_mapped = np.array([class_to_map.get(y, -1) for y in y_tv])
        return scores_test, y_true_mapped, list(classes_orig), "tutorial_pseudo_p"


    def _map(v):
        try:
            return class_to_map.get(int(float(v)), class_to_map.get(v, -1))
        except (TypeError, ValueError):
            return class_to_map.get(v, -1)

    # ── Route C: {tag}_lac_score_true — exact, no ECFP, check BEFORE Route B preamble ──
    lac_col = f"{model}_lac_score_true"
    if lac_col in df_excel.columns and has_true:
        display(Markdown(f"  Route C: reading exact LAC scores from `{lac_col}` column."))
        df_lac        = df_excel[[true_col, lac_col]].dropna().reset_index(drop=True)
        y_true_mapped = np.array([_map(v) for v in df_lac[true_col].values])
        valid_lac     = y_true_mapped >= 0
        scores_test   = df_lac.loc[valid_lac, lac_col].values.astype(float)
        y_true_mapped = y_true_mapped[valid_lac]
        return scores_test, y_true_mapped, list(classes_orig), "excel_lac_score_true"

    # ── Route B preamble: build df_v, y_true_mapped, y_pred_mapped ────────────────
    inset_cols = [c for c in df_excel.columns if c.startswith("in_set_class_")]
    if not has_true:
        raise RuntimeError(
            "Cannot compute test LAC scores without true labels. "
            f"Tried columns: {true_col_candidates}"
        )

    pred_candidates = [f"{model}_pred", "pred", "Pred", "Predicted"]
    pred_col = next((c for c in pred_candidates if c in df_excel.columns), None)
    if pred_col is None:
        raise RuntimeError(f"No prediction column found. Tried: {pred_candidates}")


    df_v          = df_excel.dropna(subset=[true_col, pred_col]).reset_index(drop=True)
    y_true_mapped = np.array([_map(v) for v in df_v[true_col].values])
    y_pred_mapped = np.array([_map(v) for v in df_v[pred_col].values])
    valid         = (y_true_mapped >= 0) & (y_pred_mapped >= 0)
    df_v          = df_v[valid].reset_index(drop=True)
    y_true_mapped = y_true_mapped[valid]
    y_pred_mapped = y_pred_mapped[valid]

   # ── Route D: in_set_class_* binary proxy (last resort) ───────────────────
    # Approximate: 0 if true class in prediction set, 1 if not.
    # Works without re-running mapiec.py but gives binary not continuous scores.
    if inset_cols:
        display(Markdown(
            "  Route D (fallback): binary proxy from `in_set_class_*` columns. "
            "Re-run mapiec.py with the lac_score patch for exact continuous scores."
        ))
        def _parse_inset(col):
            sfx = col.replace("in_set_class_", "")
            try:
                v = float(sfx); return int(v) if v == int(v) else v
            except ValueError:
                return sfx
 
        inset_cls_order  = [_parse_inset(c) for c in inset_cols]
        inset_to_j       = {c: i for i, c in enumerate(inset_cls_order)}
        n_v              = len(df_v)
        in_set_m         = np.zeros((n_v, len(inset_cls_order)), dtype=bool)
        for col in inset_cols:
            j = inset_to_j.get(_parse_inset(col), -1)
            if j >= 0:
                in_set_m[:, j] = df_v[col].values.astype(bool)
 
        inset_cls_mapped = {_map(c): i for c, i in inset_to_j.items()}
        scores_test = np.array([
            0.0 if (y_true_mapped[i] in inset_cls_mapped
                    and in_set_m[i, inset_cls_mapped[y_true_mapped[i]]])
            else 1.0
            for i in range(n_v)
        ])
        return scores_test, y_true_mapped, list(classes_orig), "inset_binary_proxy"
 
    if cache_path is None:
        raise RuntimeError(
            "No lac_score column, no in_set_class_* columns, and cache_path is None. "
            "Re-run mapiec.py with the lac_score patch."
        )

    # ── Route B proper: ECFP recompute ────────────────────────────────────────

    display(Markdown("  Route B: recomputing via ECFP + NCM model."))
 
    smiles_col = next(
        (c for c in ["Smiles", "SMILES", "smiles"] if c in df_excel.columns), None
    )
    if smiles_col is None:
        raise RuntimeError("No Smiles column found in Excel for Route B.")
 
    df_v_sm       = df_v.loc[df_v[smiles_col].notna()].reset_index(drop=True)
    y_true_mapped = y_true_mapped[:len(df_v_sm)]
    y_pred_mapped = y_pred_mapped[:len(df_v_sm)]
 
    from qubounds.descriptors.ecfp import smiles_to_ecfp_cached
    from qubounds.mapie_class_lac import NCMProbabilisticClassifier
 
    # cache_path may be None — smiles_to_ecfp_cached works without cache,
    # it just won't persist fingerprints between runs.
    if cache_path is not None:
        from qubounds.descriptors.ecfp import init_cache
        init_cache(cache_path)
 
    X_test = np.array([smiles_to_ecfp_cached(s) for s in df_v_sm[smiles_col].values])
 
    ncm_est = NCMProbabilisticClassifier(
        y_pred    = y_pred_mapped,
        ncm_model = sigma_model,
        ncm_type  = ncm_type,
        classes   = classes,
    )
    ncm_est.fit()
    pseudo_proba = ncm_est.predict_proba(X_test)
    scores_test  = 1.0 - pseudo_proba[np.arange(len(y_true_mapped)), y_true_mapped]
 
    return scores_test, y_true_mapped, list(classes_orig), "ecfp_recompute"
 
 
def _load_lac_scores_from_excel(excel_path, model):
    """
    Load classification calibration LAC scores from the Excel.
    Looks for the {model}_lac_score_true column written by the patched
    predict_conformal_classifier_hard_chunked().
 
    Priority
    --------
    1. Sheet "Calibration PI"  — scores used by MAPIE for the quantile.
    2. Sheet "Training PI"     — fallback when Calibration sheet absent.
 
    Returns (scores_1d_float, sheet_name) or raises RuntimeError.
    """
    xl      = pd.ExcelFile(excel_path)
    lac_col = f"{model}_lac_score_true"
 
    for sheet in ["Calibration PI", "Training PI"]:
        if sheet not in xl.sheet_names:
            continue
        df_s = pd.read_excel(excel_path, sheet_name=sheet)
        if lac_col in df_s.columns:
            scores = df_s[lac_col].dropna().values.astype(float)
            if len(scores) > 0:
                return scores, sheet
 
    raise RuntimeError(
        f"No LAC scores found for '{model}' in {excel_path}. "
        f"Expected sheet 'Calibration PI' or 'Training PI' "
        f"with column '{lac_col}'. "
        "Re-run mapiec.py after applying the lac_score patch."
    )
 
# =============================================================================
# E2 — Classification exchangeability
# =============================================================================
def plot_e2_classification_exchangeability(
    class_excel_path,
    class_pickle_path,
    model,
    alpha,
    out_dir,
    cache_path=None,
):
    """
    Overall panel (cal vs test score distributions + p-value uniformity)
    + class-wise p-value histograms (one per class, KS vs Uniform).

    Parameters
    ----------
    class_excel_path  : path to Excel written by mapiec.py or tutorial task.
    class_pickle_path : path to .pkl written by train_conformal_classifier_hard().
    model             : VEGA model name / column prefix (e.g. "MUTA_CAESAR").
    alpha             : miscoverage level used at training time (annotation only).
    out_dir           : directory for output PNGs.
    cache_path        : ECFP SQLite cache (required for main-pipeline route B).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── pickle ─────────────────────────────────────────────────────────────────
    with open(class_pickle_path, "rb") as fh:
        saved = pickle.load(fh)

    print(class_pickle_path, saved)
    if saved.get("skipped") or saved.get("mapie") is None:
        display(Markdown(f"E2 skipped for {model}: pickle contains no MAPIE object."))
        return None, None

    mapie_obj = saved["mapie"]

    # ── Calibration scores: Excel first (exact), pickle as fallback ───────────
    try:
        scores_cal, cal_source = _load_lac_scores_from_excel(class_excel_path, model)
        display(Markdown(
            f"  Cal scores from Excel '{cal_source}': n={len(scores_cal)}, "
            f"range=[{scores_cal.min():.3f}, {scores_cal.max():.3f}]"
        ))
    except RuntimeError as exc_excel:
        display(Markdown(
            f"  Excel cal scores unavailable ({exc_excel}); trying pickle."
        ))
        try:
            scores_cal = _get_cal_scores_classifier(saved["mapie"])
            cal_source = "pickle"
            display(Markdown(
                f"  Cal scores from pickle: n={len(scores_cal)}, "
                f"range=[{scores_cal.min():.3f}, {scores_cal.max():.3f}]"
            ))
        except AttributeError as exc_pkl:
            display(Markdown(
                f"E2 skipped for {model}: cannot load calibration scores — {exc_pkl}"
            ))
            return None, None

    display(Markdown(
        f"  Calibration scores: n={len(scores_cal)}, "
        f"range=[{scores_cal.min():.3f}, {scores_cal.max():.3f}]"
    ))

    # ── test LAC scores ─────────────────────────────────────────────────────────
    df = pd.read_excel(class_excel_path, sheet_name="Prediction Intervals")
    display(Markdown(f"  Excel columns: `{list(df.columns)}`"))

    try:
        scores_test, y_true_mapped, class_order, route = _recompute_test_scores(
            saved, df, model, cache_path
        )
    except Exception as exc:
        display(Markdown(
            f"E2 skipped for {model}: cannot compute test scores — {exc}"
        ))
        return None, None

    display(Markdown(
        f"  Test scores: n={len(scores_test)}, "
        f"range=[{scores_test.min():.3f}, {scores_test.max():.3f}]  "
        f"(route: {route})"
    ))

    # ── conformal p-values ─────────────────────────────────────────────────────
    p_values = np.array([np.mean(scores_cal >= s_i) for s_i in scores_test])

    # ── KS tests ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    ks_stat_dist, ks_p_dist = ks_2samp(scores_cal, scores_test)
    ks_stat_pval, ks_p_pval = ks_2samp(p_values, rng.uniform(size=len(p_values)))

    msg_dist = "OK" if ks_p_dist > 0.05 else "WARNING — possible distributional shift"
    msg_pval = "OK" if ks_p_pval > 0.05 else "WARNING — p-values not uniform"

    display(Markdown(
        f"### {model}  exchangeability diagnostics\n"
        f"- Cal vs Test LAC scores : KS stat={ks_stat_dist:.4f}  "
        f"p={ks_p_dist:.4f}  →  {msg_dist}\n"
        f"- P-value uniformity     : KS stat={ks_stat_pval:.4f}  "
        f"p={ks_p_pval:.4f}  →  {msg_pval}"
    ))

    # ── PANEL 1: overall ───────────────────────────────────────────────────────
    fig_ov, (ax_dist, ax_pval) = plt.subplots(1, 2, figsize=(12, 4))
    fig_ov.patch.set_facecolor(PALETTE["background"])

    ax_dist.hist(scores_cal,  bins="auto", density=True, alpha=0.55,
                 color=_COLORS["CALIBRATION"],
                 label=f"Calibration (n={len(scores_cal)})")
    ax_dist.hist(scores_test, bins="auto", density=True, alpha=0.55,
                 color=_COLORS["TEST"],
                 label=f"Test (n={len(scores_test)})")
    ax_dist.set_xlabel("LAC score  1 − p̂(true class | x)", fontsize=10)
    ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.set_title(
        f"Cal vs Test LAC scores\nKS p = {ks_p_dist:.4f}  —  {msg_dist}",
        fontsize=9
    )
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.25, linestyle="--")

    ax_pval.hist(p_values, bins=20, density=True, alpha=0.7,
                 color=_COLORS["TEST"],
                 label=f"Conformal p-values (n={len(p_values)})")
    ax_pval.axhline(1.0, color="red", linestyle="--", lw=2,
                    label="Uniform(0,1) reference")
    ax_pval.set_xlabel("p_i = #{s_cal ≥ s_i} / n_cal", fontsize=10)
    ax_pval.set_ylabel("Density", fontsize=9)
    ax_pval.set_title(
        f"P-value uniformity  (KS vs Uniform)\n"
        f"KS p = {ks_p_pval:.4f}  —  {msg_pval}",
        fontsize=9
    )
    ax_pval.set_xlim(0, 1)
    ax_pval.legend(fontsize=8)
    ax_pval.grid(True, alpha=0.25, linestyle="--")

    plt.suptitle(
        f"{model} — Classification exchangeability  (α={alpha})  [Overall]",
        fontsize=11
    )
    plt.tight_layout()
    path_ov = out_dir / f"e2_exchangeability_class_overall_{model}.png"
    _savefig(fig_ov, path_ov)

    # ── PANEL 2: class-wise p-value histograms ─────────────────────────────────
    unique_cls = np.unique(y_true_mapped[y_true_mapped >= 0])
    if len(unique_cls) < 2:
        display(Markdown(
            "Class-wise panel skipped: fewer than 2 classes in test set."
        ))
        return path_ov, None

    mapped_to_orig = saved.get("mapped_to_class", {})
    ncols  = min(len(unique_cls), 4)
    nrows  = int(np.ceil(len(unique_cls) / ncols))
    fig_cls, axes_cls = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2), squeeze=False
    )
    fig_cls.patch.set_facecolor(PALETTE["background"])

    ks_rows = []
    for idx, cls_m in enumerate(unique_cls):
        ax      = axes_cls[idx // ncols][idx % ncols]
        mask_c  = y_true_mapped == cls_m
        p_cls   = p_values[mask_c]
        cls_lbl = mapped_to_orig.get(int(cls_m), cls_m)

        if len(p_cls) < 5:
            ax.set_title(f"Class {cls_lbl}\ntoo few (n={len(p_cls)})", fontsize=8)
            ax.axis("off")
            continue

        uni_c          = rng.uniform(size=len(p_cls))
        ks_c, ks_p_c   = ks_2samp(p_cls, uni_c)
        ok_c           = ks_p_c > 0.05
        ks_rows.append({
            "class":   cls_lbl,
            "n":       len(p_cls),
            "KS_stat": round(ks_c, 4),
            "KS_p":    round(ks_p_c, 4),
            "status":  "OK" if ok_c else "WARNING",
        })

        color_c = PALETTE["green"] if ok_c else PALETTE["missed"]
        ax.hist(p_cls, bins=10, density=True, alpha=0.75, color=color_c)
        ax.axhline(1.0, color="red", linestyle="--", lw=1.8, label="Uniform(0,1)")
        ax.set_title(
            f"Class {cls_lbl}  (n={len(p_cls)})\n"
            f"KS p = {ks_p_c:.3f}  {'✓ OK' if ok_c else '⚠ WARNING'}",
            fontsize=8
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Conformal p-value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=6)

    for idx in range(len(unique_cls), nrows * ncols):
        axes_cls[idx // ncols][idx % ncols].axis("off")

    plt.suptitle(
        f"{model} — Class-wise p-value uniformity  (α={alpha})\n"
        "Green = exchangeability OK   Red = possible violation",
        fontsize=10
    )
    plt.tight_layout()
    path_cls = out_dir / f"e2_exchangeability_class_classwise_{model}.png"
    _savefig(fig_cls, path_cls)

    if ks_rows:
        display(Markdown("#### Class-wise KS test summary"))
        display(pd.DataFrame(ks_rows))

    return path_ov, path_cls

# MAIN

produced_plots = []

# ── Regression plots ──────────────────────────────────────────────────────────
if model_type == "regression":
    for regr_model in data:
        _path_a = upstream["mapie_*"][f"mapie_{regr_model}_{ncm}"]["data"]
        # mapie_[[data]]_[[ncm]]
        display(Markdown(f"# Regression publication plots — {regr_model}"))

        try:
            df_regr, meta = _load_regr(_path_a, regr_model)
        except Exception as err:
            print(err)
            continue
        display(Markdown(f"Loaded {len(df_regr)} rows from `{_path_a}`.  "
                        f"Columns: `{list(df_regr.columns)}`"))

        produced_plots.append(
            plot_r1_sorted_by_width(df_regr, meta, regr_model, alpha, n_show, out_dir))
        produced_plots.append(
            plot_r2_pred_obs_colored(df_regr, meta, regr_model, alpha, out_dir))
        produced_plots.append(
            plot_r3_width_by_adi(df_regr, meta, regr_model, alpha, out_dir))
        produced_plots.append(
            plot_r4_sigma_vs_width(df_regr, meta, regr_model, alpha, out_dir))

        model_pickle =  upstream["mapie_*"][f"mapie_{regr_model}_{ncm}"]["ncmodel"]
        if model_pickle is not None:
             display(Markdown(f"# Exchangeability — {regr_model}"))
             plot_e1_regression_exchangeability(
                 regr_excel_path  = _path_a,
                 regr_pickle_path = model_pickle,
                 model  = regr_model,
                 alpha  = alpha,
                 out_dir = out_dir,
             )

# ── Classification plots ──────────────────────────────────────────────────────
if model_type == "classification":
    for class_model in data:
        display(Markdown(f"# Classification publication plots — {class_model}"))
        class_data = upstream["mapiec_*"][f"mapiec_{class_model}_{ncm}"]["data"]
        try:
            df_class, meta = _load_class(class_data, class_model)
        except Exception as err:
            print(err)
            continue
        display(Markdown(f"Loaded {len(df_class)} rows. "
                        f"Columns: `{list(df_class.columns)}`"))

        produced_plots.append(
            plot_c1_dotplot(df_class, meta, class_model, alpha, out_dir))
        
        """
        produced_plots.append(
            plot_c2_setsize_by_class(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c3_singleton_rank(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c4_confusion_setsize(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c5_setsize_by_adi(df_class, meta, class_model, alpha, out_dir))
        """

        model_pickle =  upstream["mapiec_*"][f"mapiec_{class_model}_{ncm}"]["ncmodel"]
        if model_pickle is not None:
            display(Markdown(f"# Exchangeability — {class_model}"))
            plot_e2_classification_exchangeability(
                class_excel_path  = class_data,
                class_pickle_path = model_pickle,
                model      = class_model,
                alpha      = alpha,
                out_dir    = out_dir,
                cache_path = cache_path,   # None → uses Route C (in_set proxy)
            )
        

# ── Summary ───────────────────────────────────────────────────────────────────
display(Markdown("## Plots produced"))
for p in produced_plots:
    if p is not None:
        display(Markdown(f"- `{p}`"))

display(Markdown("Done."))
