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

# + tags=["parameters"]
data = None
alpha = 0.1
n_show = 200
model_type = None
ncm = None
product = None
upstream = None
# -

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
# C1 — Prediction-set heatmap
# =============================================================================
def plot_c1_set_heatmap(df, meta, model, alpha, out_dir, max_molecules=80,
                        class_names=None):
    """
    Rows = molecules (sorted by set size desc, then true class),
    Columns = classes.
    Cell colour = pseudo-probability (or 1/0 in-set indicator if not available).
    Green border = true class.  Red X = predicted class outside true.
    """
    if "set_size" not in df.columns:
        display(Markdown(f"C1 skipped for {model}: no set_size column."))
        return None

    pseudo_cols = df.attrs.get("pseudo_cols", [])
    has_true    = df["has_true"].any() if "has_true" in df.columns else False

    # Determine classes
    if class_names is None:
        all_vals = []
        if has_true:
            all_vals += df["true"].dropna().tolist()
        all_vals += df["pred"].dropna().tolist()
        try:
            class_names = sorted(set(int(v) for v in all_vals))
        except Exception:
            class_names = sorted(set(str(v) for v in all_vals))

    n_classes = len(class_names)
    sort_key  = ["set_size"] + (["true"] if has_true else [])
    df_s = df.sort_values(sort_key, ascending=[False] + [True] * (len(sort_key)-1))
    df_s = df_s.head(max_molecules).reset_index(drop=True)
    n_mol = len(df_s)

    # Build probability matrix
    if pseudo_cols:
        mat = np.zeros((n_mol, n_classes))
        for j, cls in enumerate(class_names):
            col_candidates = [c for c in pseudo_cols
                              if str(cls) in c or str(cls).lower() in c.lower()]
            if col_candidates:
                mat[:, j] = df_s[col_candidates[0]].fillna(0).values
    else:
        # Fallback: binary in-set matrix from set_size (not ideal but usable)
        mat = np.zeros((n_mol, n_classes))
        for i in range(n_mol):
            sz = int(df_s["set_size"].iloc[i])
            pred_cls = df_s["pred"].iloc[i]
            try:
                pred_idx = list(class_names).index(pred_cls)
            except ValueError:
                pred_idx = 0
            for j in range(n_classes):
                dist = abs(j - pred_idx)
                mat[i, j] = 1 if dist < sz else 0

    fig, ax = plt.subplots(figsize=(max(5, n_classes * 1.2), max(6, n_mol * 0.22)))
    im = ax.imshow(mat, aspect="auto", cmap=CMAP_PROB,
                   vmin=0, vmax=mat.max() if mat.max() > 0 else 1,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.5, label="Pseudo-probability")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([str(c) for c in class_names], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel(f"Molecules (n={n_mol}, sorted by set size ↓)", fontsize=9)
    ax.set_title(f"{model} — Prediction-set heatmap  (α={alpha})\n"
                 "Colour = pseudo-probability; green outline = true class",
                 fontsize=11)

    # Overlays: true class border, predicted class marker
    for i in range(n_mol):
        pred_cls = df_s["pred"].iloc[i]
        try:
            pred_j = list(class_names).index(pred_cls)
        except ValueError:
            pred_j = None

        if has_true and pd.notna(df_s["true"].iloc[i]):
            true_cls = df_s["true"].iloc[i]
            try:
                true_j = list(class_names).index(true_cls)
                rect = mpatches.FancyBboxPatch(
                    (true_j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    linewidth=1.8, edgecolor=PALETTE["green"], facecolor="none")
                ax.add_patch(rect)
            except ValueError:
                pass

        if pred_j is not None:
            ax.plot(pred_j, i, "k^", ms=4, alpha=0.7)

    plt.tight_layout()
    path = out_dir / f"c1_set_heatmap_{model}.png"
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
# MAIN
# =============================================================================

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
"""
        # R5: two-model contrast
        if regr_model_b is not None:
            _path_b = regr_data[1] if isinstance(regr_data, list) and len(regr_data) > 1 \
                    else _path_a.replace(regr_model, regr_model_b)
            try:
                df_regr_b = _load_regr(_path_b, regr_model_b)
                # Order so the tighter model is on the left (more visually clear)
                if df_regr_b["width"].mean() < df_regr["width"].mean():
                    produced_plots.append(
                        plot_r5_contrast(df_regr_b, regr_model_b,
                                        df_regr,   regr_model,
                                        alpha, n_show, out_dir_regr))
                else:
                    produced_plots.append(
                        plot_r5_contrast(df_regr,   regr_model,
                                        df_regr_b, regr_model_b,
                                        alpha, n_show, out_dir_regr))
            except Exception as e:
                display(Markdown(f"R5 skipped: could not load second dataset — {e}"))
"""                  

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
            plot_c1_set_heatmap(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c2_setsize_by_class(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c3_singleton_rank(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c4_confusion_setsize(df_class, meta, class_model, alpha, out_dir))
        produced_plots.append(
            plot_c5_setsize_by_adi(df_class, meta, class_model, alpha, out_dir))

# ── Summary ───────────────────────────────────────────────────────────────────
display(Markdown("## Plots produced"))
for p in produced_plots:
    if p is not None:
        display(Markdown(f"- `{p}`"))

display(Markdown("Done."))
