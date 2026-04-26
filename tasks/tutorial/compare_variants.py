"""
tasks/tutorial/compare_variants.py
---------------------------------
Aggregate and compare all four CP variants across one or more datasets:

  A1 – MAPIE-native base model, adaptive (sigma-normalised)
  A2 – MAPIE-native base model, non-adaptive (plain split CP)
  B1 – External base model, adaptive
  B2 – External base model, non-adaptive

Produces:
  - Summary Excel with per-variant metrics
  - Coverage vs interval-width comparison plot (saved to PNG)
  - Per-compound interval-width distribution plot

Inputs (ploomber params)
---------
  native_data   : str  – product["data"] from mapie_native.py
  external_data : str  – product["data"] from mapie_external.py
  meta_path     : str  – JSON metadata
  alpha         : float
  product       : dict – {nb, data, plot_summary, plot_width}
"""

# + tags=["parameters"]
alpha         = 0.1
product       = None
upstream = None
dataset = None
ncm = None
base_model = None
# -

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)

tag = f"tutorial_native_{dataset}_{ncm}"
native_data = upstream["tutorial_native_*"][tag]["data"]
tag = f"tutorial_external_{dataset}_{ncm}_{base_model}"
external_data = upstream["tutorial_external_*"][tag]["data"]
tag = f"tutorial_load_{dataset}"
meta_path = upstream["tutorial_load_*"][tag]["meta"]


with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]
dataset    = meta["dataset"]

# ── load predictions 
native_pred   = pd.read_excel(native_data,   sheet_name="Predictions")
external_pred = pd.read_excel(external_data, sheet_name="Predictions")

native_metrics   = pd.read_excel(native_data,   sheet_name="Metrics")
external_metrics = pd.read_excel(external_data, sheet_name="Metrics")

# Tag variant source
native_metrics["source"]   = "native"
external_metrics["source"] = "external"

# ── combined metrics table ────────────────────────────────────────────────────
all_metrics = pd.concat([native_metrics, external_metrics], ignore_index=True)
all_metrics["dataset"] = dataset
all_metrics["1-alpha"] = 1 - alpha

print(f"\n=== {dataset}  (α={alpha}, target coverage={1-alpha:.0%}) ===")
print(all_metrics[["variant", "source", "coverage", "mean_width",
                    "median_width", "sigma_r2"]].to_string(index=False))

# ── collect per-compound widths for distribution plot ─────────────────────────
width_frames = []

for src, pred_df in [("native", native_pred), ("external", external_pred)]:
    for var in ("adaptive", "plain"):
        col = f"Width_{var}"
        if col in pred_df.columns:
            width_frames.append(pd.DataFrame({
                "width": pred_df[col].values,
                "variant": f"{var}_{src}",
            }))

width_df = pd.concat(width_frames, ignore_index=True)

# ── plot 1: coverage vs mean interval width ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

colors  = {"native": "#1f77b4", "external": "#ff7f0e"}
markers = {"adaptive": "o", "plain": "s"}
offset  = {"native_adaptive": (-0.003, 0.005),
            "native_plain":   ( 0.003, 0.005),
            "external_adaptive": (-0.003, -0.008),
            "external_plain":    ( 0.003, -0.008)}

for _, row in all_metrics.iterrows():
    src = row["source"]
    var_key = row["variant"].replace("_external", "").replace("_native", "")
    var_simple = "adaptive" if "adaptive" in row["variant"] else "plain"
    label = row["variant"]
    ax.scatter(row["mean_width"], row["coverage"],
               color=colors[src], marker=markers[var_simple],
               s=120, zorder=3, label=label)
    dx, dy = offset.get(f"{src}_{var_simple}", (0, 0.005))
    ax.annotate(label, xy=(row["mean_width"], row["coverage"]),
                xytext=(row["mean_width"] + dx, row["coverage"] + dy),
                fontsize=8, ha="center")

ax.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.5,
           label=f"Target {1-alpha:.0%}")
ax.set_xlabel("Mean Interval Width", fontsize=11)
ax.set_ylabel("Empirical Coverage", fontsize=11)
ax.set_title(f"{dataset}: Coverage vs Interval Width\n"
             f"(native vs external base model; adaptive vs plain CP)",
             fontsize=11)
ax.set_ylim(max(0, (1 - alpha) - 0.15), 1.02)
# deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(product["plot_summary"], dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Summary plot → {product['plot_summary']}")

# ── plot 2: interval-width distributions ─────────────────────────────────────
variants_order = width_df["variant"].unique().tolist()
fig2, ax2 = plt.subplots(figsize=(8, 4))

for vname in variants_order:
    vals = width_df.loc[width_df["variant"] == vname, "width"]
    try:
        ax2.hist(vals, alpha=0.5, label=vname, density=True, histtype="step", linewidth=2)
    except Exception as err:
        print(err)

ax2.set_xlabel("Prediction Interval Width", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title(f"{dataset}: Interval Width Distributions by Variant", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(product["plot_width"], dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Width distribution plot → {product['plot_width']}")

# ── save summary Excel ────────────────────────────────────────────────────────
with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    all_metrics.to_excel(w, sheet_name="Metrics", index=False)
    width_df.to_excel(w, sheet_name="Width_distributions", index=False)

print(f"Summary Excel → {product['data']}")
