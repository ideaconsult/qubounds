# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_class_external.py
---------------------------------------
Tutorial: Conformal Prediction for Classification with External Predictions
============================================================================

Mirrors the VEGA/paper pipeline for classification exactly:
  - Hard predictions are read from an external file (pred_file in dataset_config).
  - The conformal wrapper treats the model as a black box.
  - NCMProbabilisticClassifier + train_conformal_classifier_hard from
    qubounds.mapie_class_lac are used (same as the internal classification task).
  - If the external file also contains AD index columns, AD vs CP comparison
    is performed at the end of this task.

SKIP BEHAVIOUR:
  If pred_file is null or the file does not exist, the task writes empty
  product files and exits immediately with an informative message.
  No "fallback to internal model" -- the external task is meaningless
  without real external predictions.

TRAIN/TEST SPLIT:
  If split_col is configured (e.g. "Dataset"), rows where
  split_col == split_train_value are used as the fit/calibration set and
  rows where split_col == split_test_value as the test set.
  If split_col is absent, the task falls back to matching cal/test by
  Smiles against the internally-split sets from the upstream load task.

Inputs (ploomber params) -- all sourced from pipeline.tutorial.yaml
---------
  dataset        : dataset key
  ncm            : sigma-model key (default crfecfp)
  alpha          : miscoverage level
  cache_path     : ECFP SQLite cache path
  dataset_config : full dict from env.tutorial.yaml
  product        : {nb, data, ncmodel}
"""

# + tags=["parameters"]
dataset        = None
ncm            = "crfecfp"
alpha          = 0.1
cache_path     = None
dataset_config = None
product        = None
upstream       = None
threshold      = 0.5
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
from scipy import stats

from qubounds.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from qubounds.mapie_diagnostic import sigma_diagnostics
from qubounds.mapie_class_lac import (
    NCMProbabilisticClassifier,
    train_conformal_classifier_hard,
)
%matplotlib inline

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

# ==============================================================================
# S0  Resolve upstream and dataset config
# ==============================================================================
tag        = f"tutorial_load_class_{dataset}"
train_data = upstream["tutorial_load_class_*"][tag]["train"]
test_data  = upstream["tutorial_load_class_*"][tag]["test"]
meta_path  = upstream["tutorial_load_class_*"][tag]["meta"]

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
# classification-specific column names (for downstream compare tasks)
set_size_col_a    = cfg.get("set_size_col_a", "SetSize_A")
set_size_col_b    = cfg.get("set_size_col_b", "SetSize_B")
covered_col_a     = cfg.get("covered_col_a",  "Cov_A")
covered_col_b     = cfg.get("covered_col_b",  "Cov_B")

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - CLASSIFICATION (external model)"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))
display(Markdown(f"""
Configuration from `env.tutorial.yaml[{dataset}]`:
- pred_file         : `{pred_file}`
- pred_col          : `{pred_col}`
- smiles_col        : `{smiles_col_ext}`
- split_col         : `{split_col}` (train=`{split_train_value}` / test=`{split_test_value}`)
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
  pred_file:          "path/to/predictions.csv"
  pred_col:           "Pred"          # column with hard class labels
  split_col:          "Dataset"       # optional: column separating train/test
  split_train_value:  "Train"         # optional
  split_test_value:   "Test"          # optional
  ad_cols:            ["PROB-STD"]    # optional: AD index columns
  ad_col_directions:  ["similarity"]
```
"""))
    pd.DataFrame().to_excel(str(product["data"]))
    _ncmodel_path = Path(str(product["ncmodel"]))
    _ncmodel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(_ncmodel_path), "wb") as _fh:
        pickle.dump({"skipped": True, "reason": _reason}, _fh)
    sys.exit(0)

# ==============================================================================
# S1  Load external file and resolve train/test split
# ==============================================================================
display(Markdown("## S1  Load external predictions file"))

_p   = Path(pred_file)
_ext = pd.read_excel(_p) if _p.suffix in {".xlsx", ".xls"} else pd.read_csv(_p)

# Normalise SMILES column name
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
    _vals = split_test_value if isinstance(split_test_value, list) else [split_test_value]
    _vals = [v.lower() for v in _vals]
    _ext_test = _ext[_ext[split_col].astype(str).str.lower().isin(_vals)].copy()
    display(Markdown(f"  - Calibration rows : {len(_ext_cal)}"))
    display(Markdown(f"  - Test rows        : {len(_ext_test)}"))
else:
    display(Markdown(
        "- No `split_col` configured. Using all rows for both cal and test "
        "matching; alignment by Smiles against the internally-split sets."))
    _ext_cal  = _ext.copy()
    _ext_test = _ext.copy()

# ==============================================================================
# S2  Internal data splits (from upstream load task)
# ==============================================================================
display(Markdown("## S2  Internal data splits"))
display(Markdown("""
The upstream load task produced stratified train/test splits.
We use those for fingerprint-based NCM training (fit set) and calibration.
The test set provides ground-truth labels for coverage evaluation.
External hard predictions are aligned to these sets by Smiles.
"""))

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

# Derive class mapping from the upstream meta (consistent with internal task)
classes_from_meta = sorted(meta.get("classes", train_df[target_col].unique().tolist()))
classes_original  = np.array(classes_from_meta)
class_to_int      = {c: i for i, c in enumerate(classes_original)}
int_to_class      = {i: c for c, i in class_to_int.items()}
n_classes         = len(classes_original)

train_df["label"] = train_df[target_col].map(class_to_int)
test_df["label"]  = test_df[target_col].map(class_to_int)
test_df = test_df.dropna(subset=["label"]).reset_index(drop=True)

fit_df, cal_df = train_test_split(
    train_df, test_size=0.2, random_state=42,
    stratify=train_df["label"] if n_classes <= 10 else None,
)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"- Fit set         : {len(fit_df)} molecules"))
display(Markdown(f"- Calibration set : {len(cal_df)} molecules"))
display(Markdown(f"- Test set        : {len(test_df)} molecules"))
display(Markdown(f"- Classes         : {classes_original.tolist()}"))

y_cal  = cal_df["label"].values.astype(int)
y_test = test_df["label"].values.astype(int)

min_required = int(np.ceil(1 / alpha)) - 1
for lbl, cnt in cal_df["label"].value_counts().sort_index().items():
    flag = "OK" if cnt >= min_required else f"WARNING: needs >= {min_required}"
    display(Markdown(f"  - {int_to_class[int(lbl)]:>20s}: {cnt:>4d}  {flag}"))

# ==============================================================================
# S3  Align external hard predictions with cal / test sets
# ==============================================================================
display(Markdown("## S3  Align external hard predictions"))
display(Markdown("""
External predictions are matched to the calibration and test sets by Smiles.
Missing entries fall back to a majority-class placeholder (flagged below).
The NCM is trained internally using ECFP fingerprints -- it does not depend on
which model produced the hard labels.
"""))

init_cache(cache_path)

def _to_int_label(val):
    """Map a raw label value to an integer index using class_to_int."""
    if val in class_to_int:
        return class_to_int[val]
    # Try numeric conversion (external file may store ints directly)
    try:
        v = int(val)
        if v in class_to_int:
            return class_to_int[v]
    except (TypeError, ValueError):
        pass
    return None


_ext_cal_idx  = _ext_cal.set_index("Smiles")[pred_col]
_ext_test_idx = _ext_test.set_index("Smiles")[pred_col]

# Map external labels to integer indices
_cal_raw   = cal_df["Smiles"].map(_ext_cal_idx)
_test_raw  = test_df["Smiles"].map(_ext_test_idx)

y_cal_hard_raw  = _cal_raw.apply(lambda v: _to_int_label(v) if pd.notna(v) else None)
y_test_hard_raw = _test_raw.apply(lambda v: _to_int_label(v) if pd.notna(v) else None)

n_cal_matched  = y_cal_hard_raw.notna().sum()
n_test_matched = y_test_hard_raw.notna().sum()
display(Markdown(
    f"- External predictions matched: cal={n_cal_matched}/{len(cal_df)}  "
    f"test={n_test_matched}/{len(test_df)}"
))

if n_cal_matched == 0 or n_test_matched == 0:
    display(Markdown(
        "ERROR: No molecules matched between external file and internal splits.\n"
        "Check `smiles_col` in env.tutorial.yaml and `split_col` configuration."))
    sys.exit(1)

# Fallback for unmatched molecules: majority class in the calibration set
_majority_class = int(cal_df["label"].mode()[0])
y_cal_hard  = y_cal_hard_raw.fillna(_majority_class).astype(int).values
y_test_hard = y_test_hard_raw.fillna(_majority_class).astype(int).values

n_cal_fallback  = int(y_cal_hard_raw.isna().sum())
n_test_fallback = int(y_test_hard_raw.isna().sum())
if n_cal_fallback:
    display(Markdown(
        f"- WARNING: {n_cal_fallback} calibration molecules not in external file; "
        f"filled with majority class ({int_to_class[_majority_class]})."))
if n_test_fallback:
    display(Markdown(
        f"- WARNING: {n_test_fallback} test molecules not in external file; "
        f"filled with majority class ({int_to_class[_majority_class]})."))

# External hard-label accuracy on matched molecules
_acc_mask = y_cal_hard_raw.notna().values
if _acc_mask.sum() > 0:
    ext_acc = np.mean(y_cal_hard[_acc_mask] == y_cal[_acc_mask])
    display(Markdown(f"- External model accuracy on matched cal molecules: {ext_acc:.3f}"))

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
# S4  ECFP fingerprints for NCM training
# ==============================================================================
display(Markdown(f"## S4  ECFP fingerprints for NCM training ({ncm})"))
display(Markdown("""
The NCM model (sigma) is always trained internally on ECFP fingerprints.
It learns P(distance = |y - y_hat| | x) independently of the external model.
Coverage is guaranteed by the calibration quantile regardless of NCM quality.
Better NCM => smaller prediction sets (more efficiency).
"""))

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df)
X_cal  = to_ecfp(cal_df)
X_test = to_ecfp(test_df)

y_fit      = fit_df["label"].values.astype(int)
y_fit_pred = y_cal_hard[:len(fit_df)]   # placeholder; NCM uses cal hard labels

# ==============================================================================
# S5  Train conformal classifier (NCM pseudo-probabilities)
# ==============================================================================
display(Markdown("## S5  Train conformal classifier with external hard labels"))
display(Markdown(f"""
`train_conformal_classifier_hard` from `qubounds.mapie_class_lac`:
  1. Trains an NCM classifier on ordinal distances |y - y_hat| (fit set).
  2. Wraps it in NCMProbabilisticClassifier to produce pseudo-probabilities.
  3. Calibrates a MAPIE SplitConformalClassifier with LAC on the cal set.

Hard labels come from the external file (simulating VEGA output).
alpha={alpha}, target coverage={1-alpha:.0%}.
"""))

df_fit_for_ncm = pd.DataFrame({
    "Smiles": fit_df["Smiles"].values,
    "Exp":    y_fit,
    "Pred":   y_fit_pred,
})
df_cal_for_ncm = pd.DataFrame({
    "Smiles": cal_df["Smiles"].values,
    "Exp":    y_cal,
    "Pred":   y_cal_hard,
})

_ncm_model_path = str(out_dir / "ncm_ext_tutorial_model.pkl")
saved = train_conformal_classifier_hard(
    df_train=df_fit_for_ncm,
    experimental_tag="Exp",
    predicted_tag="Pred",
    df_calibration=df_cal_for_ncm,
    cache_path=cache_path,
    alpha=alpha,
    output_model_path=_ncm_model_path,
    class_order=list(range(n_classes)),
    ncm=ncm,
    method_score="LAC",
)

if saved is None:
    display(Markdown("ERROR: train_conformal_classifier_hard returned None. "
                     "Check class distribution and NCM training data."))
    sys.exit(1)

mapie_ncm       = saved["mapie"]
sigma_model     = saved["sigma_model"]
ncm_estimator   = mapie_ncm.estimator_   # NCMProbabilisticClassifier

diag_fit = sigma_diagnostics(
    np.abs(y_fit - y_fit_pred).astype(float),
    sigma_model.predict(X_fit) if not ncm.startswith(("c", "o"))
    else np.argmax(sigma_model.predict_proba(X_fit), axis=1).astype(float),
)
diag_cal = sigma_diagnostics(
    np.abs(y_cal - y_cal_hard).astype(float),
    sigma_model.predict(X_cal) if not ncm.startswith(("c", "o"))
    else np.argmax(sigma_model.predict_proba(X_cal), axis=1).astype(float),
)
display(Markdown(f"- NCM R² on fit set : {diag_fit['r2']:.3f}"))
display(Markdown(f"- NCM R² on cal set : {diag_cal['r2']:.3f}"))
display(Markdown("""
Note: low NCM R² does NOT invalidate CP coverage.
The calibration quantile provides the guarantee regardless.
NCM quality only affects efficiency (prediction set size).
"""))

# ==============================================================================
# S6  Generate prediction sets on the test set
# ==============================================================================
display(Markdown("## S6  Prediction sets on test set"))

ncm_estimator.y_pred = y_test_hard
pseudo_proba_test = ncm_estimator.predict_proba(X_test)

_, y_sets_ncm = mapie_ncm.predict_set(X_test)
if y_sets_ncm.ndim == 3:
    y_sets_ncm = np.squeeze(y_sets_ncm, axis=2)

covered_ncm = y_sets_ncm[np.arange(len(y_test)), y_test].astype(bool)
sizes_ncm   = y_sets_ncm.sum(axis=1)

# LAC conformity scores for calibration (for diagnostics)
ncm_estimator.y_pred = y_cal_hard
pseudo_proba_cal = ncm_estimator.predict_proba(X_cal)
scores_ncm = 1.0 - pseudo_proba_cal[np.arange(len(y_cal)), y_cal]
n_cal      = len(scores_ncm)
q_level    = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
q_hat_ncm  = np.quantile(scores_ncm, q_level)
prob_threshold_ncm = 1 - q_hat_ncm

display(Markdown(f"- Calibration n         : {n_cal}"))
display(Markdown(f"- q_hat (NCM/external)  : {q_hat_ncm:.4f}"))
display(Markdown(f"- Probability threshold : {prob_threshold_ncm:.4f}"))
display(Markdown(f"**Results:**"))
display(Markdown(f"- Coverage      : {covered_ncm.mean():.3f}  (target >= {1-alpha:.2f})"))
display(Markdown(f"- Mean set size : {sizes_ncm.mean():.3f}"))
display(Markdown(f"- Singleton %   : {np.mean(sizes_ncm==1)*100:.1f}%"))
display(Markdown(f"- Full set %    : {np.mean(sizes_ncm==n_classes)*100:.1f}%"))

# ==============================================================================
# S7  LAC score distribution
# ==============================================================================
display(Markdown("## S7  LAC conformity score distribution"))
display(Markdown(f"""
LAC score: s(x, y) = 1 - p_pseudo(y_true | x)  -- per-calibration-sample.
q_hat is the (1-alpha) quantile over calibration scores.
Prediction set: {{j : p_pseudo(j|x) >= 1 - q_hat}}.
"""))

fig_s, axes_s = plt.subplots(1, 2, figsize=(11, 4))
axes_s[0].hist(scores_ncm, bins="auto", density=True, alpha=0.7, color="#FF9800")
axes_s[0].axvline(q_hat_ncm, color="red", lw=2, linestyle="--",
                  label=f"q_hat={q_hat_ncm:.3f}  ({1-alpha:.0%})")
axes_s[0].set_xlabel("LAC score  1 - p_pseudo(true class | x)")
axes_s[0].set_ylabel("Density")
axes_s[0].set_title("LAC calibration scores (external NCM)")
axes_s[0].legend(fontsize=8); axes_s[0].grid(True, alpha=0.3)
for cls_int in range(n_classes):
    mask = y_cal == cls_int
    if mask.sum() > 0:
        axes_s[1].hist(scores_ncm[mask], bins=20, density=True, alpha=0.5,
                       label=str(classes_original[cls_int]))
axes_s[1].axvline(q_hat_ncm, color="red", lw=2, linestyle="--",
                  label=f"q_hat={q_hat_ncm:.3f}")
axes_s[1].set_xlabel("LAC score"); axes_s[1].set_ylabel("Density")
axes_s[1].set_title("LAC scores by true class")
axes_s[1].legend(fontsize=7); axes_s[1].grid(True, alpha=0.3)
plt.suptitle("External model: LAC calibration score distributions", fontsize=11)
plt.tight_layout()
fig_s.savefig(out_dir / "ext_class_lac_scores.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_s)

# ==============================================================================
# S8  Per-class coverage
# ==============================================================================
display(Markdown("## S8  Per-class coverage (marginal vs conditional)"))
display(Markdown("""
CP guarantees *marginal* coverage averaged over all test molecules.
Per-class coverage may deviate -- particularly for minority classes with few
calibration samples.
"""))

per_class_rows = []
for cls_int, cls_name in int_to_class.items():
    mask = y_test == cls_int
    if mask.sum() == 0:
        continue
    per_class_rows.append({
        "Class":   cls_name,
        "n_test":  int(mask.sum()),
        "Coverage": f"{covered_ncm[mask].mean():.3f}",
        "Mean set size": f"{sizes_ncm[mask].mean():.2f}",
        "Singleton %": f"{np.mean(sizes_ncm[mask]==1)*100:.1f}%",
        "External match %": f"{np.mean(~y_test_hard_raw.isna().values[mask])*100:.1f}%",
    })
display(pd.DataFrame(per_class_rows))

# ==============================================================================
# S9  Exchangeability check (KS test)
# ==============================================================================
display(Markdown("## S9  Exchangeability check (KS test)"))
display(Markdown("""
Calibration and test LAC scores should be exchangeable for the coverage
guarantee to hold.  If the external model behaves differently on the
calibration vs test split, exchangeability may be violated.

KS p > 0.05 -> no evidence of distributional shift -> guarantee holds
KS p < 0.05 -> possible shift -> coverage may deviate from the guarantee
"""))

from scipy.stats import ks_2samp
scores_test_ncm = 1.0 - pseudo_proba_test[np.arange(len(y_test)), y_test]
ks_stat, ks_p = ks_2samp(scores_ncm, scores_test_ncm)
display(Markdown(
    f"- KS statistic = {ks_stat:.4f}  p = {ks_p:.4f}  "
    f"({'OK' if ks_p > 0.05 else 'WARNING: possible shift'})"))

fig_ks, ax_ks = plt.subplots(figsize=(6, 4))
ax_ks.hist(scores_ncm,      bins="auto", density=True, alpha=0.5,
           label="Calibration", color="#FF9800")
ax_ks.hist(scores_test_ncm, bins="auto", density=True, alpha=0.5,
           label="Test", color="#9C27B0")
ax_ks.set_xlabel("LAC score  1 - p_pseudo(true class | x)")
ax_ks.set_ylabel("Density")
ax_ks.set_title(f"Exchangeability: cal vs test  (KS p={ks_p:.4f})")
ax_ks.legend(fontsize=8); ax_ks.grid(True, alpha=0.3)
plt.tight_layout()
fig_ks.savefig(out_dir / "ext_class_exchangeability_ks.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ks)

# ==============================================================================
# S10  Coverage guarantee sweep across alpha levels
# ==============================================================================
display(Markdown("## S10  Coverage guarantee across alpha levels"))

alphas_sw = np.arange(0.05, 0.51, 0.05)
cov_sw = []; sz_sw = []
for a in alphas_sw:
    _ql = min(np.ceil((n_cal + 1) * (1 - a)) / n_cal, 1.0)
    _q  = np.quantile(scores_ncm, _ql)
    _s  = (1.0 - pseudo_proba_test) <= _q
    cov_sw.append(_s[np.arange(len(y_test)), y_test].mean())
    sz_sw.append(_s.sum(axis=1).mean())

fig_sw, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(1 - alphas_sw, cov_sw, "o-", color="#FF9800", label="External NCM")
ax1.plot([0.5, 0.95], [0.5, 0.95], "k:", lw=1, label="y = 1-alpha (ideal)")
ax1.set_xlabel("Target coverage  (1 - alpha)"); ax1.set_ylabel("Empirical coverage")
ax1.set_title("Coverage guarantee"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax2.plot(1 - alphas_sw, sz_sw, "s--", color="#FF9800", label="External NCM")
ax2.set_xlabel("Target coverage  (1 - alpha)"); ax2.set_ylabel("Mean set size")
ax2.set_title("Set size vs coverage target"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.suptitle("External classification CP: coverage across alpha levels", fontsize=11)
plt.tight_layout()
fig_sw.savefig(out_dir / "ext_class_coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_sw)

# ==============================================================================
# S11  AD comparison: set size vs applicability domain indices
# ==============================================================================
display(Markdown("## S11  AD comparison: prediction set size vs AD indices"))
display(Markdown("""
Classification analogue of the regression AD comparison.
Set size plays the role of interval width: larger sets = more uncertain.

Expected relationship:
- similarity-based AD: rho < 0 (in-AD molecules get smaller sets)
- distance-based AD:   rho > 0 (out-of-AD molecules get larger sets)

Coverage is compared across AD quantile bins (should stay >= 1-alpha).
"""))

ad_results = []
if not _available_ad:
    display(Markdown(
        "### Skipped\n\n"
        "No AD columns found in the external file. To enable, add to "
        f"`env.tutorial.yaml` under `{dataset}`:\n\n"
        "```yaml\n  ad_cols: [\"PROB-STD\"]\n  ad_col_directions: [\"similarity\"]\n```"
    ))
else:
    display(Markdown(f"Comparing CP set size against: {_available_ad}"))

    df_test_ad["SetSize_ext"] = sizes_ncm
    df_test_ad["Covered_ext"] = covered_ncm.astype(int)

    for ad_col, ad_dir in zip(_available_ad, ad_col_directions[:len(_available_ad)]):
        display(Markdown(f"### {ad_col}  (direction: {ad_dir})"))

        ad_raw  = df_test_ad[ad_col].astype(float)
        ad_norm = ad_raw.copy()   # no inversion; direction handled by interpretation

        mask     = ad_norm.notna() & pd.Series(sizes_ncm).notna()
        ad_c     = ad_norm[mask].reset_index(drop=True)
        sz_c     = pd.Series(sizes_ncm)[mask].reset_index(drop=True)
        cov_c    = pd.Series(covered_ncm.astype(int))[mask].reset_index(drop=True)

        rho, pval = stats.spearmanr(ad_c.values, sz_c.values)
        rng_b     = np.random.default_rng(42)
        boot_rhos = [
            stats.spearmanr(
                ad_c.values[_i := rng_b.integers(0, len(ad_c), len(ad_c))],
                sz_c.values[_i])[0]
            for _ in range(1000)
        ]
        rho_lo = np.quantile(boot_rhos, 0.025)
        rho_hi = np.quantile(boot_rhos, 0.975)

        expected = "negative" if ad_dir == "similarity" else "positive"
        display(Markdown(
            f"- Spearman rho={rho:.3f}  p={pval:.4f}  "
            f"95% CI [{rho_lo:.3f}, {rho_hi:.3f}]  (expected {expected})"))

        # Stratified analysis by AD quintile
        _tmp = pd.DataFrame({"AD_norm": ad_c.values, "sz": sz_c.values,
                              "covered": cov_c.values})
        _tmp["_bin"] = pd.qcut(ad_c, q=n_quantile_bins, labels=False, duplicates="drop")

        strat_rows = []
        for b in sorted(_tmp["_bin"].dropna().unique()):
            _m = _tmp["_bin"] == b
            strat_rows.append({
                "AD bin": f"Q{int(b)+1} [{ad_c[_m].min():.2f}-{ad_c[_m].max():.2f}]",
                "n":               int(_m.sum()),
                "Mean set size":   round(_tmp.loc[_m, "sz"].mean(), 3),
                "Coverage":        round(_tmp.loc[_m, "covered"].mean(), 3),
                "Singleton %":     round(np.mean(_tmp.loc[_m, "sz"] == 1) * 100, 1),
            })
        strat_df = pd.DataFrame(strat_rows)
        display(strat_df)

        # In-AD vs out-of-AD split
        if ad_dir == "similarity":
            in_ad  = sz_c[ad_c >= threshold]
            out_ad = sz_c[ad_c <  threshold]
        else:
            in_ad  = sz_c[ad_c <= threshold]
            out_ad = sz_c[ad_c >  threshold]

        fig_ad, axes_ad = plt.subplots(2, 2, figsize=(13, 9))
        # scatter
        axes_ad[0, 0].scatter(ad_c, sz_c, alpha=0.25, s=8, color="#FF9800", rasterized=True)
        try:
            z  = np.polyfit(ad_c, sz_c, 1)
            xr = np.linspace(ad_c.min(), ad_c.max(), 100)
            axes_ad[0, 0].plot(xr, np.poly1d(z)(xr), "r-", lw=2, alpha=0.7, label="trend")
        except Exception:
            pass
        axes_ad[0, 0].set_xlabel(f"{ad_col}")
        axes_ad[0, 0].set_ylabel("Prediction set size")
        axes_ad[0, 0].set_title(
            f"Spearman rho={rho:.3f}  p={pval:.4f}\n"
            f"95% CI [{rho_lo:.3f}, {rho_hi:.3f}]  (expected {expected})", fontsize=9)
        axes_ad[0, 0].legend(fontsize=7); axes_ad[0, 0].grid(True, alpha=0.3)
        # boxplot
        if len(in_ad) > 1 and len(out_ad) > 1:
            bp = axes_ad[0, 1].boxplot(
                [in_ad.values, out_ad.values], patch_artist=True,
                widths=0.5, medianprops=dict(color="black", lw=2))
            for patch, c_ in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
                patch.set_facecolor(c_)
            u_stat, u_p = stats.mannwhitneyu(in_ad, out_ad, alternative="two-sided")
            axes_ad[0, 1].text(0.98, 0.97, f"Mann-Whitney p={u_p:.4f}",
                               transform=axes_ad[0, 1].transAxes,
                               ha="right", va="top", fontsize=8,
                               bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        axes_ad[0, 1].set_xticklabels(
            [f"In-AD (n={len(in_ad)})", f"Out-of-AD (n={len(out_ad)})"])
        axes_ad[0, 1].set_ylabel("Prediction set size")
        axes_ad[0, 1].set_title(f"In-AD vs Out-of-AD (threshold={threshold})")
        axes_ad[0, 1].grid(True, alpha=0.3, axis="y")
        # bar: mean set size by quintile + coverage overlay
        x = np.arange(len(strat_df))
        axes_ad[1, 0].bar(x, strat_df["Mean set size"], color="#FF9800",
                          alpha=0.8, edgecolor="black")
        axes_ad[1, 0].set_xticks(x)
        axes_ad[1, 0].set_xticklabels(strat_df["AD bin"], rotation=30,
                                       ha="right", fontsize=8)
        axes_ad[1, 0].set_ylabel("Mean set size")
        axes_ad[1, 0].set_title("Mean set size by AD quintile")
        axes_ad[1, 0].grid(True, alpha=0.3, axis="y")
        ax2b = axes_ad[1, 0].twinx()
        ax2b.plot(x, strat_df["Coverage"], "rs-", lw=2, ms=8, label="Coverage")
        ax2b.axhline(1 - alpha, color="red", linestyle="--", lw=1.5,
                     label=f"Target {1-alpha:.0%}")
        ax2b.set_ylim(0, 1.05); ax2b.set_ylabel("Coverage")
        ax2b.legend(loc="upper left", fontsize=8)
        # histogram
        axes_ad[1, 1].hist(
            in_ad.values,  bins="auto", density=True, alpha=0.5,
            color="#27ae60", label=f"In-AD (n={len(in_ad)})")
        axes_ad[1, 1].hist(
            out_ad.values, bins="auto", density=True, alpha=0.5,
            color="#e74c3c", label=f"Out-of-AD (n={len(out_ad)})")
        if len(in_ad):
            axes_ad[1, 1].axvline(
                in_ad.mean(), color="#27ae60", lw=2, linestyle="--",
                label=f"Mean in={in_ad.mean():.2f}")
        if len(out_ad):
            axes_ad[1, 1].axvline(
                out_ad.mean(), color="#e74c3c", lw=2, linestyle="--",
                label=f"Mean out={out_ad.mean():.2f}")
        axes_ad[1, 1].set_xlabel("Prediction set size")
        axes_ad[1, 1].set_ylabel("Density")
        axes_ad[1, 1].set_title("Set size distribution: In-AD vs Out-of-AD")
        axes_ad[1, 1].legend(fontsize=8); axes_ad[1, 1].grid(True, alpha=0.3)

        interpretation = ("Negative rho = smaller sets inside AD"
                          if ad_dir == "similarity"
                          else "Positive rho = larger sets outside AD")
        plt.suptitle(
            f"{dataset}: set size vs {ad_col}  (rho={rho:.3f})\n{interpretation}",
            fontsize=10)
        plt.tight_layout()
        _plot_path = out_dir / f"ext_class_ad_{ad_col}.png"
        fig_ad.savefig(_plot_path, dpi=150, bbox_inches="tight")
        plt.show(); plt.close(fig_ad)
        display(Markdown(f"- Plot: `{_plot_path}`"))

        ad_results.append({
            "dataset":          dataset,
            "ad_col":           ad_col,
            "direction":        ad_dir,
            "n":                int(mask.sum()),
            "spearman_rho":     round(rho, 4),
            "p_value":          round(pval, 6),
            "rho_CI_lo":        round(rho_lo, 4),
            "rho_CI_hi":        round(rho_hi, 4),
            "mean_sz_in_AD":    round(float(in_ad.mean()),  4) if len(in_ad)  else None,
            "mean_sz_out_AD":   round(float(out_ad.mean()), 4) if len(out_ad) else None,
        })

# ==============================================================================
# S12  Save results
# ==============================================================================
display(Markdown("## S12  Save results"))

result_df = pd.DataFrame({
    "Smiles":      test_df["Smiles"].values,
    "True":        test_df[target_col].values,
    "Pred_ext":    [int_to_class[p] for p in y_test_hard],
    "Matched_ext": (~y_test_hard_raw.isna()).values,
    covered_col_b: covered_ncm.astype(int),
    set_size_col_b: sizes_ncm,
    **{f"pseudo_p_{c}": pseudo_proba_test[:, i]
       for i, c in enumerate(classes_original)},
    **{c: df_test_ad[c].values
       for c in _available_ad if c in df_test_ad.columns},
})

metrics_df = pd.DataFrame([{
    "variant":      "external_ncm",
    "alpha":        alpha,
    "coverage":     covered_ncm.mean(),
    "mean_sz":      sizes_ncm.mean(),
    "singleton_pct": np.mean(sizes_ncm == 1) * 100,
    "full_set_pct": np.mean(sizes_ncm == n_classes) * 100,
    "q_hat_ncm":    q_hat_ncm,
    "ks_stat":      ks_stat,
    "ks_pvalue":    ks_p,
    "ncm_r2_fit":   diag_fit["r2"],
    "ncm_r2_cal":   diag_cal["r2"],
    "n_test":       len(test_df),
    "n_matched_ext": int(n_test_matched),
}])

# Persist the trained model with all metadata
save_dict = {
    **saved,
    "alpha":           alpha,
    "q_hat_ncm":       q_hat_ncm,
    "classes_original": classes_original,
    "class_to_int":    class_to_int,
    "int_to_class":    int_to_class,
    "meta":            meta,
    "ks_stat":         ks_stat,
    "ks_pvalue":       ks_p,
}
with open(str(product["ncmodel"]), "wb") as fh:
    pickle.dump(save_dict, fh)

with pd.ExcelWriter(str(product["data"]), engine="xlsxwriter") as w:
    result_df.to_excel(w,  sheet_name="Predictions",    index=False)
    metrics_df.to_excel(w, sheet_name="Metrics",        index=False)
    pd.DataFrame(per_class_rows).to_excel(w, sheet_name="Per_class", index=False)
    if ad_results:
        pd.DataFrame(ad_results).to_excel(w, sheet_name="AD_CP_correlations", index=False)

display(Markdown("## [OK] External classification tutorial complete."))
display(Markdown(f"- Results : `{product['data']}`"))
display(Markdown(f"- Model   : `{product['ncmodel']}`"))
display(Markdown(f"- Plots   : `{out_dir}`"))
