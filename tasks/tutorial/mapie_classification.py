# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_classification.py
----------------------------------------
Tutorial: Conformal Prediction for QSAR – Classification
=========================================================

This script mirrors the regression tutorial but for classification.
It demonstrates prediction *sets* instead of prediction *intervals*.

Key concepts illustrated
------------------------
  Prediction set  : instead of a point prediction, CP returns a SET of
                    plausible classes guaranteed to contain the true class
                    with probability ≥ 1-α.

  Coverage        : fraction of test samples where the true class is in the
                    predicted set. CP guarantees ≥ 1-α marginally.

  Efficiency      : mean set size. A singleton set {ŷ} is most efficient;
                    a full set {all classes} is least efficient but always covers.
                    Adaptive methods aim for small sets on confident predictions
                    and larger sets on uncertain ones.

  LAC score       : Least Ambiguous Classifier. Conformity s(x, y) = 1 - p̂(y|x).
                    Works with classifiers that output probability estimates.

  Ordinal NCM     : For QSAR classifiers with only hard predictions, we train
                    a separate model to predict ordinal distance |ŷ - y| and
                    convert that to pseudo-class probabilities.

Dataset
-------
  Uses a binary or multiclass toxicity dataset loaded by load_dataset_classification.py.
  Built-in option: Tox21 SR-MMP assay (binary: active/inactive).

Inputs  (ploomber params)
---------
  dataset    : dataset key
  alpha      : miscoverage level (default 0.1)
  ncm        : sigma-model key for the ordinal NCM (default "crfecfp")
  cache_path : ECFP SQLite cache path
  product    : {nb, data, ncmodel}
"""


import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown, HTML

from lightgbm import LGBMClassifier
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import LACConformityScore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_diagnostic import make_sigma_model, sigma_diagnostics
%matplotlib inline


# + tags=["parameters"]
alpha      = 0.1
ncm        = "crfecfp"
cache_path = None
product    = None
upstream   = None
dataset    = None
# -

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

# ═══════════════════════════════════════════════════════════════════════════════
# §0  Resolve upstream paths
# ═══════════════════════════════════════════════════════════════════════════════
tag        = f"tutorial_load_class_{dataset}"
train_data = upstream["tutorial_load_class_*"][tag]["train"]
test_data  = upstream["tutorial_load_class_*"][tag]["test"]
meta_path  = upstream["tutorial_load_class_*"][tag]["meta"]

with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]

display(Markdown(f"#  CONFORMAL PREDICTION TUTORIAL – CLASSIFICATION"))
display(Markdown(f"##  Dataset : {meta['dataset']}   Target : {target_col}"))

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Data splits
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §1  Data splits"))
display(Markdown("""
Same three-way split as regression:
  - Fit set         → train the base classifier
  - Calibration set → determine the score threshold that controls coverage
  - Test set        → evaluate coverage and efficiency

For classification, we also check class balance across splits to ensure
the calibration set is representative of all classes.
"""))

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

# Encode labels to integers
classes_original = np.sort(train_df[target_col].unique())
class_to_int = {c: i for i, c in enumerate(classes_original)}
int_to_class = {i: c for c, i in class_to_int.items()}
n_classes = len(classes_original)

train_df["label"] = train_df[target_col].map(class_to_int)
test_df["label"]  = test_df[target_col].map(class_to_int)
test_df = test_df.dropna(subset=["label"]).reset_index(drop=True)

fit_df, cal_df = train_test_split(
    train_df, test_size=0.2, random_state=42,
    stratify=train_df["label"] if n_classes <= 10 else None
)
fit_df = fit_df.reset_index(drop=True)
cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"-  Fit set          : {len(fit_df):>5d} molecules"))
display(Markdown(f"-  Calibration set  : {len(cal_df):>5d} molecules"))
display(Markdown(f"-  Test set         : {len(test_df):>5d} molecules"))
display(Markdown(f"-  Classes          : {classes_original.tolist()}"))
display(Markdown("-  Class distribution in calibration set:"))
for cls, cnt in zip(*np.unique(cal_df["label"], return_counts=True)):
    display(Markdown(f"-    {int_to_class[cls]:>20s} : {cnt:>4d}  ({cnt/len(cal_df)*100:.1f} %)"))
display(Markdown("""
  ⚠ Minimum calibration set size per class: for target coverage 1-α,
    the calibration set must contain at least ceil(1/α) - 1 samples.
    E.g. for α=0.1 → at least 9 samples per class.
    Very small classes may lead to overcoverage (wide sets).
"""))
min_required = int(np.ceil(1 / alpha)) - 1
class_counts = cal_df["label"].value_counts()
for lbl, cnt in class_counts.items():
    flag = "OK" if cnt >= min_required else f"WARNING: needs >={min_required}"
    display(Markdown(f"-    class {int_to_class[lbl]:>20s}: {cnt:>4d}  {flag}"))

# ═══════════════════════════════════════════════════════════════════════════════
# §2  Fingerprints and base classifier
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §2  Fingerprints and base classifier"))

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df)
X_cal  = to_ecfp(cal_df)
X_test = to_ecfp(test_df)

y_fit  = fit_df["label"].values
y_cal  = cal_df["label"].values
y_test = test_df["label"].values

display(Markdown("  Training LightGBM classifier + Platt scaling calibration..."))
# LightGBM + Platt calibration so predict_proba is well-calibrated
base_lgbm = LGBMClassifier(
    n_estimators=200, learning_rate=0.05,
    class_weight="balanced", random_state=42, verbose=-1
)
# CalibratedClassifierCV gives us well-calibrated probabilities
base_model = CalibratedClassifierCV(base_lgbm, cv=3, method="sigmoid")
base_model.fit(X_fit, y_fit)

y_fit_pred  = base_model.predict(X_fit)
y_cal_pred  = base_model.predict(X_cal)
y_test_pred = base_model.predict(X_test)

acc_fit = np.mean(y_fit_pred == y_fit)
acc_cal = np.mean(y_cal_pred == y_cal)
display(Markdown(f"-  Base model accuracy on fit set  : {acc_fit:.3f}"))
display(Markdown(f"-  Base model accuracy on cal set  : {acc_cal:.3f}  (honest estimate)"))

classification_report(y_test, base_model.predict(X_test),
                             target_names=[str(c) for c in classes_original])

# ═══════════════════════════════════════════════════════════════════════════════
# §3  Conformity scores for classification (LAC)
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §3  Conformity scores for classification"))
display(Markdown(f"""
For classification, the LAC (Least Ambiguous Classifier) conformity score is:

  - s(x, y) = 1 - p̂(y | x)

where p̂(y | x) is the estimated probability for the TRUE class y.

- High score (near 1) → the classifier was uncertain about the true class.
- Low score (near 0)  → the classifier was confident about the true class.

The (1-α) quantile q̂ of calibration scores determines which classes enter
the prediction set for a new molecule:

-  Prediction set  = {{ y : 1 - p̂(y | x_new) ≤ q̂ }}
                  = {{ y : p̂(y | x_new) ≥ 1 - q̂ }}

Here α = {alpha}, target coverage = {1-alpha:.0%}.
"""))

proba_cal = base_model.predict_proba(X_cal)   # (n_cal, n_classes)
# conformity scores for the TRUE class
scores_cal = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]

n_cal = len(scores_cal)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_level = min(q_level, 1.0)
q_hat = np.quantile(scores_cal, q_level)

display(Markdown(f"-  Calibration set size : {n_cal}"))
display(Markdown(f"-  Quantile level       : {q_level:.4f}"))
display(Markdown(f"-  q̂                   : {q_hat:.4f}"))
display(Markdown(f"-  Threshold on p̂      : {1 - q_hat:.4f}  "
      f"(classes with p̂ ≥ this threshold enter the prediction set)"))

fig_q, ax_q = plt.subplots(figsize=(6, 4))
ax_q.hist(scores_cal, bins=40, density=True, alpha=0.7, color="#9C27B0")
ax_q.axvline(q_hat, color="red", lw=2, linestyle="--",
             label=f"q̂ = {q_hat:.3f}  ({1-alpha:.0%} level)")
ax_q.set_xlabel("Conformity score  1 - p̂(true class | x)")
ax_q.set_ylabel("Density")
ax_q.set_title(f"LAC conformity scores on calibration set (α={alpha})")
ax_q.legend(fontsize=9)
ax_q.grid(True, alpha=0.3)
plt.tight_layout()
fig_q.savefig(out_dir / "class_calibration_scores.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_q)

# ═══════════════════════════════════════════════════════════════════════════════
# §4  MAPIE conformal classifier
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## n§4  Fitting MAPIE conformal classifier (LAC)"))

mapie_clf = SplitConformalClassifier(
    estimator=base_model,
    conformity_score=LACConformityScore(),
    prefit=True,
    confidence_level=1 - alpha,
)
mapie_clf.estimator_ = base_model
mapie_clf.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)
display(Markdown("-  Conformal classifier fitted and conformalized."))

# ═══════════════════════════════════════════════════════════════════════════════
# §5  Test set: prediction sets, coverage, efficiency
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §5  Test set: prediction sets, coverage, efficiency"))
display(Markdown(f"""
For each test molecule we obtain:
  - A prediction set (one or more classes)
  - Whether the true class is in the set (coverage)
  - The set size (efficiency: 1 = singleton, most efficient)

Target coverage ≥ {1-alpha:.0%}.
"""))

_, y_pred_sets = mapie_clf.predict_set(X_test)
if y_pred_sets.ndim == 3:
    y_pred_sets = np.squeeze(y_pred_sets, axis=2)

# Coverage
covered = y_pred_sets[np.arange(len(y_test)), y_test].astype(bool)
set_sizes = y_pred_sets.sum(axis=1)
singleton_rate = np.mean(set_sizes == 1)
empty_rate     = np.mean(set_sizes == 0)

display(Markdown(f"-  Empirical coverage  : {covered.mean():.3f}  (target ≥ {1-alpha:.2f})"))
display(Markdown(f"-  Mean set size       : {set_sizes.mean():.3f}  (1 = most efficient)"))
display(Markdown(f"-  Singleton sets (%)  : {singleton_rate*100:.1f}%  (most informative)"))
display(Markdown(f"-  Empty sets (%)      : {empty_rate*100:.1f}%  (should be ~0)"))
display(Markdown(f"-  Set size distribution:"))
for sz, cnt in zip(*np.unique(set_sizes, return_counts=True)):
    display(Markdown(f"  - size {int(sz)} : {cnt:>5d}  ({cnt/len(set_sizes)*100:.1f}%)"))

# ── Coverage by true class
display(Markdown(f"\n  Per-class coverage:"))
for cls_int, cls_name in int_to_class.items():
    mask = y_test == cls_int
    if mask.sum() == 0:
        continue
    cls_cov = covered[mask].mean()
    cls_sz  = set_sizes[mask].mean()
    display(Markdown(f"-{cls_name:>20s}: coverage={cls_cov:.3f}  mean set size={cls_sz:.2f}"))

# ── Prediction set visualisation
proba_test = base_model.predict_proba(X_test)

# Sample 12 molecules to show their prediction sets
n_show = min(12, len(test_df))
sample_idx = np.random.default_rng(0).choice(len(test_df), n_show, replace=False)
sample_idx = sample_idx[np.argsort(y_test[sample_idx])]   # sort by true class

fig_sets, axes = plt.subplots(3, 4, figsize=(14, 9))
axes = axes.flatten()
for k, i in enumerate(sample_idx):
    ax = axes[k]
    probs_i = proba_test[i]
    in_set  = y_pred_sets[i].astype(bool)
    true_cls = y_test[i]
    colors_bar = []
    for j in range(n_classes):
        if j == true_cls and in_set[j]:
            colors_bar.append("#2E7D32")   # green: true class, in set
        elif j == true_cls and not in_set[j]:
            colors_bar.append("#D32F2F")   # red: true class, not covered
        elif in_set[j]:
            colors_bar.append("#FFA726")   # orange: in set but not true
        else:
            colors_bar.append("#BDBDBD")   # grey: not in set

    ax.bar(np.arange(n_classes), probs_i, color=colors_bar)
    ax.axhline(1 - q_hat, color="red", lw=1.5, linestyle="--", alpha=0.7,
               label=f"threshold={1-q_hat:.2f}")
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels([str(c) for c in classes_original], fontsize=7)
    ax.set_ylim(0, 1.05)
    status = "OK" if covered[i] else "MISSED"
    ax.set_title(f"True: {int_to_class[true_cls]}  {status}  "
                 f"set size={int(set_sizes[i])}", fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    if k == 0:
        ax.legend(fontsize=6, loc="upper right")
plt.suptitle(
    f"Prediction sets: class probabilities vs LAC threshold (α={alpha})\n"
    "Green=true class in set  Red=true class missed  Orange=false positive in set",
    fontsize=10
)
plt.tight_layout()
fig_sets.savefig(out_dir / "class_prediction_sets.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_sets)

# ═══════════════════════════════════════════════════════════════════════════════
# §6  Coverage guarantee: sweeping α
# ═══════════════════════════════════════════════════════════════════════════════
display(Markdown("## §6  Coverage guarantee across α levels"))

alphas = np.arange(0.05, 0.51, 0.05)
coverages_sweep = []
sizes_sweep     = []

scores_mapie = mapie_clf._mapie_classifier.conformity_scores_
scores_mapie = scores_mapie[~np.isnan(scores_mapie)]

proba_test_all = base_model.predict_proba(X_test)

for a in alphas:
    n = len(scores_mapie)
    ql = min(np.ceil((n + 1) * (1 - a)) / n, 1.0)
    q_a = np.quantile(scores_mapie, ql)
    # prediction sets: include class j if 1 - p̂(j) ≤ q_a, i.e. p̂(j) ≥ 1 - q_a
    sets_a = (1.0 - proba_test_all) <= q_a   # (n_test, n_classes)
    cov_a  = sets_a[np.arange(len(y_test)), y_test].mean()
    sz_a   = sets_a.sum(axis=1).mean()
    coverages_sweep.append(cov_a)
    sizes_sweep.append(sz_a)

fig_sweep, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(1 - alphas, coverages_sweep, "o-", color="#9C27B0")
ax1.plot([0.5, 0.95], [0.5, 0.95], "k:", lw=1, label="y = 1-α (ideal)")
ax1.set_xlabel("Target coverage  (1 - α)")
ax1.set_ylabel("Empirical coverage")
ax1.set_title("Coverage guarantee across α levels")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax2.plot(1 - alphas, sizes_sweep, "s-", color="#E91E63")
ax2.set_xlabel("Target coverage  (1 - α)")
ax2.set_ylabel("Mean prediction set size")
ax2.set_title("Set size grows as coverage target increases\n(efficiency ↔ coverage trade-off)")
ax2.grid(True, alpha=0.3)
plt.suptitle("Classification CP: marginal coverage and efficiency across α", fontsize=11)
plt.tight_layout()
fig_sweep.savefig(out_dir / "class_coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_sweep)

for a, cov, sz in zip(alphas, coverages_sweep, sizes_sweep):
    display(Markdown(f"-  α={a:.2f}  target={1-a:.2f}  coverage={cov:.3f}  mean set size={sz:.2f}"))

# ═══════════════════════════════════════════════════════════════════════════════
# §7  Save results
# ═══════════════════════════════════════════════════════════════════════════════
pred_sets_decoded = [
    [int_to_class[j] for j in range(n_classes) if y_pred_sets[i, j]]
    for i in range(len(test_df))
]

result_df = pd.DataFrame({
    "Smiles":      test_df["Smiles"].values,
    "True":        test_df[target_col].values,
    "Pred":        [int_to_class[p] for p in y_test_pred],
    "In_Coverage": covered.astype(int),
    "Set_Size":    set_sizes,
    "Pred_Set":    [str(s) for s in pred_sets_decoded],
    **{f"p_{c}": proba_test_all[:, i] for i, c in enumerate(classes_original)},
})

metrics_dict = {
    "dataset": meta["dataset"],
    "target_col": target_col,
    "alpha": alpha,
    "coverage": float(covered.mean()),
    "mean_set_size": float(set_sizes.mean()),
    "singleton_rate": float(singleton_rate),
    "empty_rate": float(empty_rate),
    "base_accuracy_cal": float(acc_cal),
    "n_test": len(test_df),
    "n_classes": n_classes,
}
metrics_df = pd.DataFrame([metrics_dict])

save_dict = {
    "mapie": mapie_clf,
    "base_model": base_model,
    "ncm": ncm,
    "classes_original": classes_original,
    "class_to_int": class_to_int,
    "int_to_class": int_to_class,
    "alpha": alpha,
    "meta": meta,
}
with open(product["ncmodel"], "wb") as fh:
    pickle.dump(save_dict, fh)

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics_df.to_excel(w, sheet_name="Metrics", index=False)

display(Markdown("-[OK] Classification tutorial complete."))
display(Markdown(f"-   Results → {product['data']}"))
display(Markdown(f"-  Plots   → {out_dir}"))
