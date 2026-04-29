import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown
%matplotlib inline

"""
tasks/tutorial/compare_variants.py
------------------------------------
Compare regression CP variants side-by-side:
  Mode 1  - internal base model (MAPIE trains LightGBM)
  Mode 2  - external predictions (mirrors VEGA / paper pipeline)

Both modes are produced by mapie_native.py and mapie_external.py.
This task reads their Metrics sheets and Predictions sheets and
produces summary plots and a combined Excel.

Inputs (ploomber params)
---------
  dataset    : dataset key
  ncm        : sigma-model key
    alpha      : miscoverage level
  product    : {nb, data, plot_summary, plot_width}
"""

# + tags=["parameters"]
alpha      = 0.1
product    = None
upstream   = None
dataset    = None
ncm        = None
# -

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)

# ── resolve upstream paths ────────────────────────────────────────────────────
tag_load    = f"tutorial_load_{dataset}"
tag_native  = f"tutorial_native_{dataset}_{ncm}"
tag_ext     = f"tutorial_external_{dataset}_{ncm}"
dataset, tag_load,tag_native,tag_ext


meta_path     = upstream["tutorial_load_*"][tag_load]["meta"]
native_data   = upstream["tutorial_native_*"][tag_native]["data"]
external_data = upstream["tutorial_external_*"][tag_ext]["data"]
meta_path

with open(meta_path) as f:
    meta = json.load(f)
print(meta)    
target_col = meta["target_col"]
dataset_name = meta["dataset"]

display(Markdown(f"# Comparison: {dataset_name}  (alpha={alpha}, target={1-alpha:.0%})"))

# ── load metrics from each variant ───────────────────────────────────────────
native_metrics   = pd.read_excel(native_data,   sheet_name="Metrics")
try:
    external_metrics = pd.read_excel(external_data, sheet_name="Metrics")
except Exception as err:
    print(err)    
    external_metrics = pd.DataFrame()

# mapie_native.py writes one row per mode (Mode1_internal, Mode2_external)
# mapie_external.py writes one row per variant (adaptive, plain)
# Standardise to a common format with a 'source' column.
native_metrics["source"]   = "native"
external_metrics["source"] = "external"

# Rename columns to a common schema if needed
def _normalise_metrics(df, source):
    """Map various column names to a standard set."""
    col_map = {
        # mapie_native uses predict_conformal() output keys
        "Average Interval Width":  "mean_width",
        "Empirical coverage":      "coverage",
        "Empirical Coverage":      "coverage",
        # mapie_external uses its own column names
        "coverage":                "coverage",
        "mean_width":              "mean_width",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["source"] = source
    return df

native_metrics   = _normalise_metrics(native_metrics,   "native")
external_metrics = _normalise_metrics(external_metrics, "external")

all_metrics = pd.concat([native_metrics, external_metrics], ignore_index=True)
all_metrics["dataset"] = dataset_name
all_metrics["1-alpha"] = 1 - alpha

display(Markdown("## Metrics summary"))
_show_cols = [c for c in ["source", "variant", "coverage", "mean_width",
                           "median_width", "sigma_r2", "exch_ks_pvalue"]
              if c in all_metrics.columns]
display(all_metrics[_show_cols])

# ── load per-compound predictions for width distribution ─────────────────────
def _try_load_predictions(path, source):
    """Try to load prediction-level data from available sheets."""
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        if "Prediction" in sheet or "Mode" in sheet:
            try:
                df = pd.read_excel(path, sheet_name=sheet)
                # Find interval width columns
                width_cols = [c for c in df.columns
                              if "width" in c.lower() or "Width" in c]
                for wc in width_cols:
                    frames.append(pd.DataFrame({
                        "width":   df[wc].dropna().values,
                        "variant": f"{source}_{sheet}_{wc}",
                    }))
            except Exception:
                pass
    return frames

width_frames = []
width_frames.extend(_try_load_predictions(native_data,   "native"))
width_frames.extend(_try_load_predictions(external_data, "external"))

# ── plot 1: coverage vs mean interval width ───────────────────────────────────
display(Markdown("## Plot 1: Coverage vs mean interval width"))

fig1, ax1 = plt.subplots(figsize=(8, 5))
colors_src  = {"native": "#1f77b4", "external": "#ff7f0e"}
markers_src = {"native": "o", "external": "s"}
for _, row in all_metrics.iterrows():
    if "coverage" not in row or "mean_width" not in row:
        continue
    src = row.get("source", "native")
    variant_label = str(row.get("variant", row.get("Mode", src)))
    try:
        cov = float(row["coverage"])
        wid = float(row["mean_width"])
    except (ValueError, TypeError):
        continue
    ax1.scatter(wid, cov,
                color=colors_src.get(src, "grey"),
                marker=markers_src.get(src, "^"),
                s=120, zorder=3)
    ax1.annotate(f"{src}\n{variant_label}",
                 xy=(wid, cov), xytext=(wid, cov + 0.005),
                 fontsize=7, ha="center")
ax1.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.5,
            label=f"Target {1-alpha:.0%}")
ax1.set_xlabel("Mean Interval Width", fontsize=11)
ax1.set_ylabel("Empirical Coverage", fontsize=11)
ax1.set_title(f"{dataset_name}: Coverage vs Interval Width\n"
              f"(native vs external; alpha={alpha})", fontsize=11)
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(product["plot_summary"], dpi=200, bbox_inches="tight")
plt.show(); plt.close(fig1)
display(Markdown(f"- Summary plot -> {product['plot_summary']}"))

# ── plot 2: interval width distributions ─────────────────────────────────────
display(Markdown("## Plot 2: Interval width distributions"))

if width_frames:
    width_df = pd.concat(width_frames, ignore_index=True)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for vname in width_df["variant"].unique():
        vals = width_df.loc[width_df["variant"] == vname, "width"].dropna()
        try:
            ax2.hist(vals.values, bins="auto", alpha=0.5, label=vname,
                        density=True, histtype="step", linewidth=2)
        except:
            try:
                ax2.hist(vals.values, bins=5, alpha=0.5, label=vname,
                        density=True, histtype="step", linewidth=2)               
            except Exception as err:
                print(err)
    ax2.set_xlabel("Prediction Interval Width", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(f"{dataset_name}: Width Distributions by Variant", fontsize=11)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(product["plot_width"], dpi=200, bbox_inches="tight")
    plt.show(); plt.close(fig2)
    display(Markdown(f"- Width distribution plot -> {product['plot_width']}"))
else:
    display(Markdown("- No per-molecule width data found; skipping width plot."))
    # Write a blank placeholder so the product file exists
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.text(0.5, 0.5, "No width data available", ha="center", va="center")
    fig2.savefig(product["plot_width"], dpi=100)
    plt.close(fig2)
    width_df = pd.DataFrame(columns=["width", "variant"])

# ── save summary Excel ────────────────────────────────────────────────────────
with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    all_metrics.to_excel(w, sheet_name="Metrics", index=False)
    if len(width_df) > 0:
        width_df.to_excel(w, sheet_name="Width_distributions", index=False)

display(Markdown(f"- Summary Excel -> {product['data']}"))
