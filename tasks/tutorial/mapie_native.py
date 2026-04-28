import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp
from lightgbm import LGBMRegressor

from qubounds.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from qubounds.mapie_diagnostic import make_sigma_model, sigma_diagnostics, detect_residual_degeneracy
from qubounds.mapie_regression import train_conformal_regression, predict_conformal, ExternalPredictor
# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_native.py
-------------------------------
Tutorial: Conformal Prediction for QSAR - Regression
=====================================================

Demonstrates split conformal prediction using the SAME production functions
from the main qubounds pipeline:
  - train_conformal_regression()  from tasks.mapie_regression
  - predict_conformal()           from tasks.mapie_regression
  - ExternalPredictor             from tasks.mapie_regression

Two modes are supported:

  Mode 1 - Built-in base model (standard MAPIE)
    MAPIE trains LightGBM on the fit set and wraps it with split CP.
    The sigma model is trained on fit-set residuals with ECFP fingerprints.

  Mode 2 - External predictions from file (mirrors the paper / VEGA pipeline)
    Predictions are read from a CSV/Excel file (column: pred_col_external).
    The sigma model is trained the same way; only the base predictor changes.
    This is the exact mode used for VEGA models in the paper.

Both variants are compared head-to-head on coverage and efficiency.

Key concepts illustrated (per-prediction vs per-model):
  alpha         - miscoverage level (scalar, user-set)
  sigma model   - predicts local residual magnitude per molecule
  q_hat         - (1-alpha) quantile of calibration scores (per-model)
  Coverage      - per-prediction: in-interval indicator; per-model: mean
  Efficiency    - per-prediction: interval width; per-model: mean width
  Objective     - minimise mean width subject to coverage >= 1-alpha
  Exchangeability - KS test: calibration vs test score distributions
  NCM comparison  - coverage is stable regardless of sigma model quality;
                    efficiency improves with better sigma model

Inputs (ploomber params)
---------
  dataset            : dataset key matching upstream load task
  alpha              : miscoverage level (default 0.1)
  ncm                : sigma-model key (default rlgbmecfp)
  cache_path         : ECFP SQLite cache path
  external_pred_file : optional path to CSV/Excel with external predictions
                       columns: Smiles, <external_pred_col>, [<true_col>]
  external_pred_col  : column name for external predictions (default "Pred")
  external_true_col  : column name for true values in external file (default "Exp")
  product            : {nb, data, ncmodel_adaptive, ncmodel_plain}
"""

# + tags=["parameters"]
alpha              = 0.1
ncm                = "rlgbmecfp"
cache_path         = None
product            = None
upstream           = None
dataset            = None
dataset_config     = None
# -


matplotlib.use("Agg")
%matplotlib inline

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

# ==============================================================================
# S0  Resolve upstream paths
# ==============================================================================
tag        = f"tutorial_load_{dataset}"
train_data = upstream["tutorial_load_*"][tag]["train"]
test_data  = upstream["tutorial_load_*"][tag]["test"]
meta_path  = upstream["tutorial_load_*"][tag]["meta"]

cfg = dataset_config[dataset]
external_pred_file = cfg.get("pred_file", None)   # set to a file path to run Mode 2
external_pred_col  = cfg.get("pred_col", None) 
external_true_col  = cfg.get("target_col", None) 
smiles_col  = cfg.get("smiles_col", None) 

with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - REGRESSION"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))

# ==============================================================================
# S1  Formal definitions
# ==============================================================================
display(Markdown("## S1  Formal definitions"))
display(Markdown(r"""
### Formal definitions (per-prediction vs per-model)

**alpha** -- miscoverage level. *Scalar, set by the user.*
The allowed fraction of test predictions that may fall outside the interval.
alpha=0.1 means 10% may miss => 90% coverage targeted.

---

**Nonconformity score** s(x_i, y_i) -- *per-prediction*, on calibration set.

  Adaptive: s = |y - y_hat| / sigma_hat(x)
  Plain:    s = |y - y_hat|

where sigma_hat(x) is the sigma model prediction of expected local residual.
Higher score = more surprising = model less reliable at this molecule.

---

**Calibration quantile** q_hat -- *per-model*, computed once from all n calibration scores.

  q_hat = quantile({s_1, ..., s_n}, level = ceil((n+1)*(1-alpha)) / n)

The inflated level (n+1)/n is a finite-sample correction ensuring
marginal coverage exactly >= 1-alpha (Vovk et al. 2005).

---

**Prediction interval** C(x) -- *per-prediction*, at inference time.

  Adaptive: y_hat +/- q_hat * sigma_hat(x)   (molecule-specific width)
  Plain:    y_hat +/- q_hat                   (same width for all molecules)

---

**Coverage** -- two granularities:
  Per-prediction:  cov_i = 1[y_i in C(x_i)]        (binary per molecule)
  Per-model:       Cov = mean(cov_i) over test set   (validity criterion)
  CP guarantee:    Cov >= 1-alpha in expectation.

---

**Efficiency** -- two granularities:
  Per-prediction:  w_i = upper_i - lower_i          (interval width)
  Per-model:       W = mean(w_i)                     (primary efficiency metric)

---

**Calibration objective:**
  Minimise W = mean interval width
  subject to: Cov >= 1-alpha

CP achieves this by construction. The adaptive variant achieves lower W than
plain when sigma_hat(x) correctly ranks molecules by local uncertainty.

---

**Conditional vs marginal coverage:**
CP guarantees *marginal* coverage (averaged over all test molecules).
Per-subgroup coverage (e.g. only out-of-AD compounds) is not guaranteed.
The adaptive variant approximates conditional coverage but the formal
guarantee remains marginal.
"""))

# ==============================================================================
# S2  Data splits
# ==============================================================================
display(Markdown("## S2  Data splits"))
display(Markdown("""
Split conformal prediction requires THREE disjoint sets:

  Fit set         -> train base model + sigma model
  Calibration set -> compute nonconformity scores; derive q_hat
  Test set        -> evaluate coverage and efficiency

The calibration set is CONSUMED by the conformal procedure.
Exchangeability between calibration and test is verified with a KS test.
"""))

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

fit_df, cal_df = train_test_split(train_df, test_size=0.2, random_state=42)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"- Fit set         : {len(fit_df)} molecules"))
display(Markdown(f"- Calibration set : {len(cal_df)} molecules"))
display(Markdown(f"- Test set        : {len(test_df)} molecules"))

# ==============================================================================
# S3  Fingerprints
# ==============================================================================
display(Markdown("## S3  Molecular fingerprints (ECFP4, 2048 bits)"))

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df)
X_cal  = to_ecfp(cal_df)
X_test = to_ecfp(test_df)

y_fit  = fit_df[target_col].values.astype(float)
y_cal  = cal_df[target_col].values.astype(float)
y_test = test_df[target_col].values.astype(float)

# ==============================================================================
# S4  Base model and sigma model
# ==============================================================================
display(Markdown("## S4  Base model and sigma model"))
display(Markdown(f"""
The sigma model predicts |y - y_hat| from ECFP fingerprints.
It is trained on fit-set residuals using: {ncm}

Key insight: a low sigma model R2 does NOT invalidate CP coverage.
The calibration quantile q_hat provides the coverage guarantee regardless.
A better sigma model gives narrower intervals (better efficiency).
"""))

# Train base model on fit set
base_model = LGBMRegressor(objective="huber", random_state=42, verbose=-1)
base_model.fit(X_fit, y_fit)
y_fit_pred = base_model.predict(X_fit)
y_cal_pred = base_model.predict(X_cal)
y_test_pred_internal = base_model.predict(X_test)

r2_cal = r2_score(y_cal, y_cal_pred)
display(Markdown(f"- Base model R2 on cal set: {r2_cal:.3f}  (honest estimate)"))

# Train sigma model
residuals_fit = np.abs(y_fit - y_fit_pred)
_, diag_res = detect_residual_degeneracy(residuals_fit, y_fit)
sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, residuals_fit)
sigma_pred_fit = sigma_model.predict(X_fit)
sigma_pred_cal = sigma_model.predict(X_cal)
sigma_pred_test = sigma_model.predict(X_test)
diag_s = sigma_diagnostics(residuals_fit, sigma_pred_fit)
diag_s_cal = sigma_diagnostics(np.abs(y_cal - y_cal_pred), sigma_pred_cal)

display(Markdown(f"- Sigma model R2 on fit set : {diag_s['r2']:.3f}"))
display(Markdown(f"- Sigma model R2 on cal set : {diag_s_cal['r2']:.3f}"))

# Sigma scatter plot
fig_s, ax_s = plt.subplots(figsize=(5, 4))
ax_s.scatter(residuals_fit, sigma_pred_fit, alpha=0.3, s=8, color="#2196F3", label="Fit")
ax_s.scatter(np.abs(y_cal - y_cal_pred), sigma_pred_cal, alpha=0.3, s=8,
             color="#FF9800", label="Cal")
lim = max(residuals_fit.max(), sigma_pred_fit.max()) * 1.05
ax_s.plot([0, lim], [0, lim], "k--", lw=1)
ax_s.set_xlabel("|y - y_hat|  (true residual)")
ax_s.set_ylabel("sigma_hat(x)  (predicted)")
ax_s.set_title(f"Sigma model: predicted vs actual residual\n(R2_cal={diag_s_cal['r2']:.3f})")
ax_s.legend(fontsize=8)
ax_s.grid(True, alpha=0.3)
plt.tight_layout()
fig_s.savefig(out_dir / "sigma_scatter.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_s)

# ==============================================================================
# S5  Formal definitions: calibration scores and q_hat
# ==============================================================================
display(Markdown("## S5  Calibration scores and q_hat"))

eps = 1e-6
sigma_cal_safe = np.maximum(sigma_pred_cal, eps)
scores_adaptive = np.abs(y_cal - y_cal_pred) / sigma_cal_safe
scores_plain    = np.abs(y_cal - y_cal_pred)

n_cal  = len(scores_adaptive)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_level = min(q_level, 1.0)
q_adaptive = np.quantile(scores_adaptive, q_level)
q_plain    = np.quantile(scores_plain,    q_level)

display(Markdown(f"- Calibration set size n : {n_cal}"))
display(Markdown(f"- Quantile level         : {q_level:.4f}  [= ceil((n+1)*(1-alpha)) / n]"))
display(Markdown(f"- q_hat adaptive         : {q_adaptive:.4f}  (in sigma units)"))
display(Markdown(f"- q_hat plain            : {q_plain:.4f}  (in target units)"))

fig_q, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(scores_adaptive, bins=40, density=True, alpha=0.7, color="#2196F3")
ax1.axvline(q_adaptive, color="red", lw=2, linestyle="--",
            label=f"q_hat={q_adaptive:.3f}  ({1-alpha:.0%})")
ax1.set_xlabel("|y - y_hat| / sigma_hat  (adaptive score)")
ax1.set_ylabel("Density")
ax1.set_title("Adaptive nonconformity scores")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax2.hist(scores_plain, bins=40, density=True, alpha=0.7, color="#FF9800")
ax2.axvline(q_plain, color="red", lw=2, linestyle="--",
            label=f"q_hat={q_plain:.3f}  ({1-alpha:.0%})")
ax2.set_xlabel("|y - y_hat|  (plain score)")
ax2.set_ylabel("Density")
ax2.set_title("Plain nonconformity scores")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.suptitle(f"Calibration nonconformity score distributions  (alpha={alpha})", fontsize=11)
plt.tight_layout()
fig_q.savefig(out_dir / "calibration_scores.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_q)

# ==============================================================================
# S6  Mode 1: train_conformal_regression() with internal base model
#     Uses the SAME production function as the VEGA pipeline
# ==============================================================================
display(Markdown("## S6  Mode 1: training with internal base model"))
display(Markdown("""
We use train_conformal_regression() from tasks.mapie_regression -- the same
function used in the main pipeline.  The function receives:
  - df_train with a 'residuals' column (pre-computed from the fit set)
  - df_calibration with Smiles, Exp, Pred, residuals columns
  - ExternalPredictor wrapping y_cal_pred (the base model's calibration predictions)

This is how VEGA models are handled: the model predictions are read from a
file and passed to MAPIE as an ExternalPredictor.
"""))

# Prepare DataFrames in the format expected by train_conformal_regression
df_train_for_cp = fit_df[["Smiles", target_col]].copy()
df_train_for_cp = df_train_for_cp.rename(columns={target_col: "Exp"})
df_train_for_cp["residuals"] = residuals_fit

df_cal_for_cp = cal_df[["Smiles", target_col]].copy()
df_cal_for_cp = df_cal_for_cp.rename(columns={target_col: "Exp"})
df_cal_for_cp["Pred"] = y_cal_pred
df_cal_for_cp["residuals"] = np.abs(y_cal - y_cal_pred)

model_path_mode1 = str(out_dir / "cp_model_mode1.pkl")

train_conformal_regression(
    df_train=df_train_for_cp,
    experimental_tag="Exp",
    df_calibration=df_cal_for_cp,
    sheet_name=f"{dataset}_mode1",
    experimental_tag_test="Exp",
    predicted_tag_test="Pred",
    cache_path=cache_path,
    alpha=alpha,
    output_model_path=model_path_mode1,
    ncm=ncm,
)
display(Markdown(f"- Model saved to: {model_path_mode1}"))

# Predict on test set using predict_conformal()
df_test_for_cp = test_df[["Smiles", target_col]].copy()
df_test_for_cp = df_test_for_cp.rename(columns={target_col: "Exp"})
df_test_for_cp["Pred"] = y_test_pred_internal

results_mode1, metrics_mode1, _ = predict_conformal(
    df=df_test_for_cp,
    pred_column="Pred",
    true_column="Exp",
    model_path=model_path_mode1,
    tag=f"{dataset}",
    smiles_column="Smiles",
    split="Test",
    save_path=str(out_dir / "mode1_residuals.png"),
)

display(Markdown("**Mode 1 results (internal base model):**"))
display(pd.DataFrame([metrics_mode1]))

# ==============================================================================
# S7  Mode 2: external predictions from file (paper / VEGA pattern)
# ==============================================================================
display(Markdown("## S7  Mode 2: external predictions from file"))
display(Markdown(f"""
In the paper, VEGA model predictions are read from a report file.
The conformal wrapper does NOT retrain the base model -- it treats it as
a black box and only needs the predictions.

If external_pred_file is set, we load predictions from that file.
Otherwise we simulate an external predictor by using the same base model
predictions (demonstrating the code path, not a different model).

External file format: Smiles, {external_pred_col}, [{external_true_col}]
"""))

if external_pred_file is not None and Path(external_pred_file).exists():
    display(Markdown(f"- Loading external predictions from: {external_pred_file}"))
    _p = Path(external_pred_file)
    df_ext = pd.read_excel(_p) if _p.suffix in {".xlsx", ".xls"} else pd.read_csv(_p)
    df_ext = df_ext.rename(columns={external_pred_col: "Pred", smiles_col: "Smiles"})
    if external_true_col in df_ext.columns:
        df_ext = df_ext.rename(columns={external_true_col: "Exp"})
    # Use molecules in common with test set
    df_test_ext = df_test_for_cp.merge(df_ext[["Smiles", "Pred"]], on="Smiles", how="inner",
                                        suffixes=("_internal", ""))
    if "Pred_internal" in df_test_ext.columns:
        df_test_ext = df_test_ext.drop(columns=["Pred_internal"])
    display(Markdown(f"- Matched {len(df_test_ext)} molecules with external predictions"))
else:
    display(Markdown("- No external_pred_file provided. Using internal model predictions to "
                     "demonstrate the code path (same predictions as Mode 1)."))
    df_test_ext = df_test_for_cp.copy()

# The calibration set always uses the same sigma model and calibration data.
# Only the test-time predictor changes.
results_mode2, metrics_mode2, _ = predict_conformal(
    df=df_test_ext,
    pred_column="Pred",
    true_column="Exp",
    model_path=model_path_mode1,   # same sigma model, different predictions
    tag=f"{dataset}_ext",
    smiles_column="Smiles",
    split="Test",
    save_path=str(out_dir / "mode2_residuals.png"),
)

display(Markdown("**Mode 2 results (external predictions):**"))
display(pd.DataFrame([metrics_mode2]))

# ==============================================================================
# S8  Coverage and efficiency: formal comparison
# ==============================================================================
display(Markdown("## S8  Coverage and efficiency: formal comparison"))
display(Markdown(f"""
Objective: minimise W = mean interval width, subject to Cov >= {1-alpha:.2f}.
Both modes achieve the coverage constraint by construction.
"""))

summary_rows = []
for name, metrics in [("Mode 1 internal", metrics_mode1), ("Mode 2 external", metrics_mode2)]:
    summary_rows.append({
        "Mode": name,
        "Coverage": f"{metrics.get('Empirical coverage', metrics.get('Empirical Coverage', 'N/A')):.3f}",
        "Target": f">= {1-alpha:.2f}",
        "Mean width": f"{metrics['Average Interval Width']:.4f}",
        "KS p-value": f"{metrics.get('exch_ks_pvalue', float('nan')):.4f}",
    })
display(pd.DataFrame(summary_rows))

# ==============================================================================
# S9  Visualise prediction intervals
# ==============================================================================
display(Markdown("## S9  Prediction intervals on test set"))

n_show = min(100, len(results_mode1))
idx_sorted = np.argsort(results_mode1[f"{dataset}_true"].values)[:n_show]

y_true_show  = results_mode1[f"{dataset}_true"].values[idx_sorted]
y_pred_show  = results_mode1[f"{dataset}_pred"].values[idx_sorted]
lower_show   = results_mode1[f"{dataset}_lower"].values[idx_sorted]
upper_show   = results_mode1[f"{dataset}_upper"].values[idx_sorted]
covered_show = (y_true_show >= lower_show) & (y_true_show <= upper_show)

fig_pi, ax_pi = plt.subplots(figsize=(12, 4))
x_pos = np.arange(n_show)
ax_pi.fill_between(x_pos, lower_show, upper_show, alpha=0.25,
                   color="#2196F3", label="Prediction interval")
ax_pi.scatter(x_pos[covered_show],  y_true_show[covered_show],
              c="green", s=15, zorder=3, label="Covered")
ax_pi.scatter(x_pos[~covered_show], y_true_show[~covered_show],
              c="red", s=25, zorder=4, marker="x", label="Not covered")
ax_pi.plot(x_pos, y_pred_show, color="#FF9800", lw=1, alpha=0.7, label="y_hat")
ax_pi.set_xlabel(f"Test molecules (sorted by true {target_col})")
ax_pi.set_ylabel(target_col)
ax_pi.set_title(f"Prediction intervals  (alpha={alpha}, n={n_show})\n"
                f"coverage={covered_show.mean():.3f}  "
                f"mean width={np.mean(upper_show - lower_show):.3f}")
ax_pi.legend(fontsize=7, loc="upper left")
ax_pi.grid(True, alpha=0.3)
plt.tight_layout()
fig_pi.savefig(out_dir / "prediction_intervals.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_pi)

# ==============================================================================
# S10  Exchangeability (KS test)
# ==============================================================================
display(Markdown("## S10  Exchangeability check (KS test)"))
display(Markdown("""
CP coverage guarantees require that calibration and test nonconformity scores
are exchangeable (look like draws from the same distribution).

- KS p-value >> 0.05 -> no evidence of shift -> guarantee holds
- KS p-value << 0.05 -> possible shift -> coverage may deviate from guarantee

We compare the calibration score distribution against the test scores.
"""))

scores_test_adaptive = results_mode1[f"{dataset}_ncm"].dropna().values
ks_stat, ks_p = ks_2samp(scores_adaptive, scores_test_adaptive)

fig_ks, ax_ks = plt.subplots(figsize=(6, 4))
ax_ks.hist(scores_adaptive,      bins=40, density=True, alpha=0.5,
           label="Calibration", color="#2196F3")
ax_ks.hist(scores_test_adaptive, bins=40, density=True, alpha=0.5,
           label="Test", color="#FF9800")
ax_ks.set_xlabel("Nonconformity score  |y - y_hat| / sigma_hat")
ax_ks.set_ylabel("Density")
ax_ks.set_title(f"Exchangeability: cal vs test scores\n(KS p={ks_p:.4f})")
ax_ks.legend(fontsize=8)
ax_ks.grid(True, alpha=0.3)
plt.tight_layout()
fig_ks.savefig(out_dir / "exchangeability_ks.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ks)

display(Markdown(f"- KS statistic: {ks_stat:.4f}"))
display(Markdown(f"- KS p-value  : {ks_p:.4f}  "
                 f"({'OK exchangeable' if ks_p > 0.05 else 'WARNING possible shift'})"))

# ==============================================================================
# S11  Coverage guarantee sweep across alpha
# ==============================================================================
display(Markdown("## S11  Coverage guarantee across alpha levels"))
display(Markdown("""
Split CP guarantees coverage >= 1-alpha for ANY alpha.
We verify by sweeping alpha and recomputing from already-fitted calibration scores.
"""))

# Load saved conformity scores from the model
with open(model_path_mode1, "rb") as f:
    saved_m1 = pickle.load(f)
mapie_m1 = saved_m1["mapie"]
cal_scores_mapie = mapie_m1._mapie_regressor.conformity_scores_
cal_scores_mapie = cal_scores_mapie[~np.isnan(cal_scores_mapie)]

alphas_sw = np.arange(0.05, 0.51, 0.05)
coverages_sw = []
widths_sw    = []

sigma_test_safe = np.maximum(sigma_pred_test, eps)
y_pred_test_arr = results_mode1[f"{dataset}_pred"].values

for a in alphas_sw:
    _n = len(cal_scores_mapie)
    _ql = min(np.ceil((_n + 1) * (1 - a)) / _n, 1.0)
    _q  = np.quantile(cal_scores_mapie, _ql)
    _lo = y_pred_test_arr - _q * sigma_test_safe
    _hi = y_pred_test_arr + _q * sigma_test_safe
    coverages_sw.append(np.mean((y_test >= _lo) & (y_test <= _hi)))
    widths_sw.append(np.mean(_hi - _lo))

fig_sw, (ax_sw1, ax_sw2) = plt.subplots(1, 2, figsize=(10, 4))
ax_sw1.plot(1 - alphas_sw, coverages_sw, "o-", color="#2196F3")
ax_sw1.plot([0.5, 0.95], [0.5, 0.95], "k:", lw=1, label="y = 1-alpha (ideal)")
ax_sw1.set_xlabel("Target coverage  (1 - alpha)")
ax_sw1.set_ylabel("Empirical coverage")
ax_sw1.set_title("Coverage guarantee across alpha levels")
ax_sw1.legend(fontsize=8); ax_sw1.grid(True, alpha=0.3)
ax_sw2.plot(1 - alphas_sw, widths_sw, "s-", color="#FF9800")
ax_sw2.set_xlabel("Target coverage  (1 - alpha)")
ax_sw2.set_ylabel("Mean interval width")
ax_sw2.set_title("Efficiency-coverage trade-off")
ax_sw2.grid(True, alpha=0.3)
plt.suptitle("Marginal coverage and efficiency across alpha values", fontsize=11)
plt.tight_layout()
fig_sw.savefig(out_dir / "coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_sw)

# ==============================================================================
# S12  NCM quality vs coverage/efficiency
# ==============================================================================
display(Markdown("## S12  NCM model quality vs coverage and efficiency"))
display(Markdown(r"""
Reviewers ask: 'which ML method should be used for the sigma model?'

Key insight from CP theory:
- **Coverage is guaranteed regardless of sigma model quality.**
  The calibration quantile q_hat absorbs whatever the sigma model does.
- **Efficiency depends on sigma model quality.**
  A better sigma model correctly ranks molecules by local uncertainty
  => smaller q_hat in normalised units => narrower intervals.

Below we simulate three sigma quality levels from the same real calibration
residuals and show that coverage is stable while mean width varies.
"""))

np.random.seed(42)
_true_res_cal  = np.abs(y_cal - y_cal_pred)
_true_res_test = np.abs(y_test - y_pred_test_arr)

_configs = [
    ("Poor  (R2~0.1)",  1.5,  "#e74c3c"),
    ("Medium (R2~0.5)", 0.75,  "#3498db"),
    ("Good  (R2~0.9)",  0.25, "#27ae60"),
]

fig_ncm, axes_ncm = plt.subplots(2, 3, figsize=(14, 8))
ncm_rows = []
for col_idx, (label, noise, color) in enumerate(_configs):
    _sc  = np.maximum(_true_res_cal  + np.random.normal(0, noise * _true_res_cal.mean(),
                                                          len(_true_res_cal)), 1e-4)
    _st  = np.maximum(_true_res_test + np.random.normal(0, noise * _true_res_test.mean(),
                                                          len(_true_res_test)), 1e-4)
    _r2  = r2_score(_true_res_cal, _sc)
    _sc_safe = np.maximum(_sc, eps)
    _scores  = _true_res_cal / _sc_safe
    _n = len(_scores)
    _ql = min(np.ceil((_n + 1) * (1 - alpha)) / _n, 1.0)
    _q  = np.quantile(_scores, _ql)
    _lo = y_pred_test_arr - _q * np.maximum(_st, eps)
    _hi = y_pred_test_arr + _q * np.maximum(_st, eps)
    _cov = np.mean((y_test >= _lo) & (y_test <= _hi))
    _w   = np.mean(_hi - _lo)
    ncm_rows.append({"Model": label, "R2": round(_r2, 3),
                     "q_hat": round(_q, 3), "Coverage": round(_cov, 3),
                     "Mean_width": round(_w, 3)})

    ax_top = axes_ncm[0, col_idx]
    ax_top.scatter(_true_res_cal, _sc, alpha=0.35, s=8, color=color)
    _lim = max(_true_res_cal.max(), _sc.max()) * 1.05
    ax_top.plot([0, _lim], [0, _lim], "k--", lw=1)
    ax_top.set_xlabel("|y - y_hat|  (true residual)")
    ax_top.set_ylabel("sigma_hat(x)  (predicted)")
    ax_top.set_title(f"{label}\nR2 = {_r2:.2f}")
    ax_top.grid(True, alpha=0.3)

    ax_bot = axes_ncm[1, col_idx]
    ax_bot.hist(_scores, bins=35, density=True, alpha=0.65, color=color)
    ax_bot.axvline(_q, color="red", lw=2, linestyle="--",
                   label=f"q_hat={_q:.2f}  cov={_cov:.3f}  W={_w:.3f}")
    ax_bot.set_xlabel("|y - y_hat| / sigma  (normalised score)")
    ax_bot.set_ylabel("Density")
    ax_bot.set_title(f"Coverage={_cov:.3f}  Mean width={_w:.3f}")
    ax_bot.legend(fontsize=7)
    ax_bot.grid(True, alpha=0.3)
plt.suptitle(
    f"NCM quality vs CP outcome  (alpha={alpha}, target={1-alpha:.0%})\n"
    "Coverage stable; efficiency improves with better sigma model.",
    fontsize=10)
plt.tight_layout()
fig_ncm.savefig(out_dir / "ncm_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ncm)

display(pd.DataFrame(ncm_rows))
display(Markdown("""
**Take-away:**
- Coverage is robust to NCM choice: all quality levels achieve the target.
- LightGBM Huber (rlgbmecfp) achieves the best sigma R2 on QSAR datasets
  and is the recommended default -- but the guarantee holds regardless.
"""))

# ==============================================================================
# S13  Save results
# ==============================================================================
with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    results_mode1.to_excel(w, sheet_name="Mode1_internal", index=False)
    results_mode2.to_excel(w, sheet_name="Mode2_external", index=False)
    pd.DataFrame([metrics_mode1, metrics_mode2]).to_excel(w, sheet_name="Metrics", index=False)
    pd.DataFrame(ncm_rows).to_excel(w, sheet_name="NCM_comparison", index=False)

display(Markdown("## [OK] Regression tutorial complete."))
display(Markdown(f"- Results: {product['data']}"))
display(Markdown(f"- Plots  : {out_dir}"))
