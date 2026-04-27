# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_classification.py
----------------------------------------
Tutorial: Conformal Prediction for QSAR - Classification
=========================================================

Three CP approaches are demonstrated and compared:

  Approach A  LAC with model probabilities
              The classifier produces predict_proba() output.
              Conformity score: s(x,y) = 1 - p_hat(y|x)
              This is the gold standard when calibrated probabilities are available.

  Approach B  Hard predictions + ordinal distance NCM (pseudo-probabilities)
              The classifier produces only a hard label y_hat.
              An auxiliary NCM model learns P(|y - y_hat| = d | x) over distances.
              These distance probabilities are converted to class pseudo-probabilities:
                  P(class=j | x, y_hat) = P(distance = |j - y_hat| | x)
              Then LAC is applied to the pseudo-probabilities.
              This is the approach used for VEGA models that provide only hard predictions.

  Approach C  Hard predictions + plain ordinal distance (non-adaptive)
              No auxiliary model. Conformity score is simply -|y - y_hat|.
              Simplest possible approach; no efficiency gain from uncertainty estimation.

The tutorial shows:
  - Formal definitions (per-prediction vs per-model)
  - Calibration objective: minimise mean set size subject to coverage >= 1-alpha
  - How LAC threshold translates to prediction sets (bar chart panel)
  - NCM quality vs coverage/efficiency (mirrors the regression NCM comparison)
  - Coverage guarantee sweep across alpha values
  - Side-by-side comparison of all three approaches
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown

from lightgbm import LGBMClassifier
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import LACConformityScore
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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

# ==============================================================================
# S0  Resolve upstream
# ==============================================================================
tag        = f"tutorial_load_class_{dataset}"
train_data = upstream["tutorial_load_class_*"][tag]["train"]
test_data  = upstream["tutorial_load_class_*"][tag]["test"]
meta_path  = upstream["tutorial_load_class_*"][tag]["meta"]

with open(meta_path) as f:
    meta = json.load(f)
target_col = meta["target_col"]

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - CLASSIFICATION"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))

# ==============================================================================
# S1  Formal definitions for classification
# ==============================================================================
display(Markdown("## S1  Formal definitions"))
display(Markdown(r"""
### Key concepts: classification vs regression

In **regression**, CP outputs a *prediction interval* [lower, upper].
In **classification**, CP outputs a *prediction set* -- a subset of classes
guaranteed to contain the true class with probability >= 1-alpha.

---

**Nonconformity score** s(x_i, y_i) -- *per-prediction*, on calibration set.

| Approach | Requires | Score formula |
|---|---|---|
| LAC (real proba) | predict_proba() | s = 1 - p_hat(y_true given x) |
| NCM pseudo-proba | hard label + NCM | s = 1 - p_pseudo(y_true given x, y_hat) |
| Ordinal distance | hard label only | s = abs(y_true - y_hat) |

Higher score = more surprising = less confident the model is about the true class.

---

**Calibration quantile** q_hat -- *per-model*, computed once.
  q_hat = quantile({s_1, ..., s_n}, level = ceil((n+1)*(1-alpha)) / n)

---

**Prediction set** C(x) -- *per-prediction*, at inference time.
For LAC and pseudo-proba:
  C(x) = {y : 1 - p_hat(y given x) <= q_hat}
        = {y : p_hat(y given x) >= 1 - q_hat}
For ordinal distance:
  C(x) = {y : abs(y - y_hat) <= q_hat}    (all classes within distance q_hat)

---

**Coverage** -- two granularities:
  Per-prediction:  cov_i = 1[y_i in C(x_i)]          (binary per molecule)
  Per-model:       Cov = mean(cov_i) over test set     (validity criterion)
  CP guarantees:   Cov >= 1-alpha in expectation.

---

**Efficiency** -- two granularities:
  Per-prediction:  sz_i = |C(x_i)|                    (set size, integer)
  Per-model:       SZ = mean(sz_i)                     (efficiency metric)
  Ideal:           sz_i = 1 for all molecules (singleton sets).

---

**Calibration objective** (same structure as regression):
  Minimise SZ = mean set size
  subject to: Cov >= 1-alpha

CP achieves this by construction via q_hat. LAC with well-calibrated
probabilities achieves smaller SZ than ordinal distance because it can
distinguish between classes with high vs low predicted probability.

---

**Conditional vs marginal coverage** (identical to regression case):
CP guarantees marginal coverage (averaged over all test molecules).
Per-class coverage may deviate; classes with few calibration samples
are most at risk. Check: need at least ceil(1/alpha)-1 samples per class.
"""))

# ==============================================================================
# S2  Data splits and class balance
# ==============================================================================
display(Markdown("## S2  Data splits and class balance"))

train_df = pd.read_excel(train_data, sheet_name="Training")
test_df  = pd.read_excel(test_data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)

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

display(Markdown(f"- Fit set        : {len(fit_df)} molecules"))
display(Markdown(f"- Calibration set: {len(cal_df)} molecules"))
display(Markdown(f"- Test set       : {len(test_df)} molecules"))
display(Markdown(f"- Classes        : {classes_original.tolist()}"))

min_required = int(np.ceil(1 / alpha)) - 1
display(Markdown(f"\nMinimum calibration samples per class at alpha={alpha}: {min_required}"))
class_counts_cal = cal_df["label"].value_counts().sort_index()
for lbl, cnt in class_counts_cal.items():
    flag = "OK" if cnt >= min_required else f"WARNING: needs >={min_required}"
    display(Markdown(f"  - {int_to_class[lbl]:>20s}: {cnt:>4d}  {flag}"))

# ==============================================================================
# S3  Fingerprints and base classifier
# ==============================================================================
display(Markdown("## S3  Fingerprints and base classifier"))
display(Markdown("""
We train ONE base classifier and use it for all three CP approaches.
This isolates the effect of the CP method from the base model.

The base classifier is LightGBM with Platt calibration (sigmoid method),
which produces well-calibrated predict_proba() output. This matters for
Approach A (LAC) but not for Approach C (ordinal distance).

For Approach B (NCM pseudo-proba), we additionally train an auxiliary NCM
classifier that predicts the ordinal distance P(|y - y_hat| = d | x).
"""))

init_cache(cache_path)

def to_ecfp(df_):
    return np.array([smiles_to_ecfp_cached(s) for s in df_["Smiles"].values])

X_fit  = to_ecfp(fit_df)
X_cal  = to_ecfp(cal_df)
X_test = to_ecfp(test_df)

y_fit  = fit_df["label"].values
y_cal  = cal_df["label"].values
y_test = test_df["label"].values

# Base model with calibrated probabilities
base_lgbm = LGBMClassifier(
    n_estimators=200, learning_rate=0.05,
    class_weight="balanced", random_state=42, verbose=-1
)
base_model = CalibratedClassifierCV(base_lgbm, cv=3, method="sigmoid")
base_model.fit(X_fit, y_fit)

y_fit_pred  = base_model.predict(X_fit)
y_cal_pred  = base_model.predict(X_cal)
y_test_pred = base_model.predict(X_test)

acc_fit = accuracy_score(y_fit, y_fit_pred)
acc_cal = accuracy_score(y_cal, y_cal_pred)
acc_test = accuracy_score(y_test, y_test_pred)

display(Markdown(f"- Base model accuracy on fit set  : {acc_fit:.3f}"))
display(Markdown(f"- Base model accuracy on cal set  : {acc_cal:.3f}  (honest)"))
display(Markdown(f"- Base model accuracy on test set : {acc_test:.3f}"))

display(Markdown("### Classification report on test set"))
report = classification_report(y_test, y_test_pred,
                                target_names=[str(c) for c in classes_original])
display(Markdown(f"```\n{report}\n```"))

# Hard predictions (used as "external model" output for Approaches B and C)
# These simulate a VEGA model that provides only a class label, not probabilities.
y_cal_hard  = y_cal_pred   # hard labels on calibration set
y_test_hard = y_test_pred  # hard labels on test set

# Predicted probabilities (Approach A)
proba_cal  = base_model.predict_proba(X_cal)   # (n_cal, n_classes)
proba_test = base_model.predict_proba(X_test)  # (n_test, n_classes)

# ==============================================================================
# S4  LAC conformity scores (Approach A - real probabilities)
# ==============================================================================
display(Markdown("## S4  Approach A: LAC with model probabilities"))
display(Markdown(f"""
LAC (Least Ambiguous Classifier) conformity score:

  s_i = 1 - p_hat(y_true | x_i)

where p_hat(y_true | x_i) is the predicted probability for the TRUE class.

The (1-alpha) quantile q_hat of calibration scores is computed.
A test molecule x gets class j into its prediction set if:
  p_hat(j | x) >= 1 - q_hat

Here alpha = {alpha}, target coverage = {1-alpha:.0%}.

This approach requires a well-calibrated classifier (CalibratedClassifierCV
or a model natively producing calibrated probabilities). If probabilities are
poorly calibrated, coverage may still hold (it is guaranteed by the quantile
mechanism) but efficiency suffers.
"""))

# LAC scores on calibration set
scores_lac = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]

n_cal = len(scores_lac)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_level = min(q_level, 1.0)
q_hat_lac = np.quantile(scores_lac, q_level)
prob_threshold_lac = 1 - q_hat_lac

display(Markdown(f"- Calibration set size : {n_cal}"))
display(Markdown(f"- Quantile level       : {q_level:.4f}  [= ceil((n+1)*(1-alpha)) / n]"))
display(Markdown(f"- q_hat (LAC)          : {q_hat_lac:.4f}"))
display(Markdown(f"- Probability threshold: {prob_threshold_lac:.4f}  "
                 f"(classes with p_hat >= this enter the prediction set)"))

fig_a, axes_a = plt.subplots(1, 2, figsize=(11, 4))
axes_a[0].hist(scores_lac, bins="auto", density=True, alpha=0.7, color="#9C27B0")
axes_a[0].axvline(q_hat_lac, color="red", lw=2, linestyle="--",
                  label=f"q_hat={q_hat_lac:.3f}  ({1-alpha:.0%} level)")
axes_a[0].set_xlabel("LAC score  1 - p_hat(true class | x)")
axes_a[0].set_ylabel("Density")
axes_a[0].set_title("LAC calibration scores (Approach A)")
axes_a[0].legend(fontsize=8)
axes_a[0].grid(True, alpha=0.3)
# Show per-class calibration score distributions
for cls_int in range(n_classes):
    mask = y_cal == cls_int
    if mask.sum() > 0:
        axes_a[1].hist(scores_lac[mask], bins=20, density=True, alpha=0.5,
                       label=str(classes_original[cls_int]))
axes_a[1].axvline(q_hat_lac, color="red", lw=2, linestyle="--", label=f"q_hat={q_hat_lac:.3f}")
axes_a[1].set_xlabel("LAC score")
axes_a[1].set_ylabel("Density")
axes_a[1].set_title("LAC scores by true class")
axes_a[1].legend(fontsize=7)
axes_a[1].grid(True, alpha=0.3)
plt.suptitle("Approach A: LAC calibration score distributions", fontsize=11)
plt.tight_layout()
fig_a.savefig(out_dir / "class_lac_scores.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_a)

# MAPIE conformal classifier - Approach A
mapie_lac = SplitConformalClassifier(
    estimator=base_model,
    conformity_score=LACConformityScore(),
    prefit=True,
    confidence_level=1 - alpha,
)
mapie_lac.estimator_ = base_model
mapie_lac.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

_, y_sets_lac = mapie_lac.predict_set(X_test)
if y_sets_lac.ndim == 3:
    y_sets_lac = np.squeeze(y_sets_lac, axis=2)

covered_lac  = y_sets_lac[np.arange(len(y_test)), y_test].astype(bool)
sizes_lac    = y_sets_lac.sum(axis=1)

display(Markdown(f"**Approach A results:**"))
display(Markdown(f"- Coverage  : {covered_lac.mean():.3f}  (target >= {1-alpha:.2f})"))
display(Markdown(f"- Mean set size: {sizes_lac.mean():.3f}"))
display(Markdown(f"- Singleton rate: {np.mean(sizes_lac==1)*100:.1f}%"))

# ==============================================================================
# S5  NCM pseudo-probability approach (Approach B - hard labels)
# ==============================================================================
display(Markdown("## S5  Approach B: NCM pseudo-probabilities from hard labels"))
display(Markdown(f"""
Many QSAR models (e.g., VEGA) output only a hard class label, not probabilities.
Standard LAC cannot be applied directly.

**Solution**: train an auxiliary NCM classifier on ECFP fingerprints to predict
the ordinal distance P(|y - y_hat| = d | x) over discrete distances d = 0, 1, 2, ...

Then convert distance probabilities to class pseudo-probabilities:
  P_pseudo(class=j | x, y_hat) = P(distance = |j - y_hat| | x)

Example (4-class model, y_hat = 1):
  NCM output: P(d=0)=0.60, P(d=1)=0.30, P(d=2)=0.08, P(d=3)=0.02
  => P(class=0) = P(d=1) = 0.30
     P(class=1) = P(d=0) = 0.60   <- predicted class, highest probability
     P(class=2) = P(d=1) = 0.30
     P(class=3) = P(d=2) = 0.08

Note: classes equidistant from y_hat get equal pseudo-probability (symmetric).
This preserves ordinal structure while allowing LAC to work with hard predictions.

Coverage guarantee is NOT affected by NCM quality -- it comes from the
calibration quantile. NCM quality only affects efficiency (set size).
"""))

# Train NCM model on ordinal distances
ordinal_distances_fit = np.abs(y_fit - y_fit_pred).astype(int)
ordinal_distances_cal = np.abs(y_cal - y_cal_hard).astype(int)

display(Markdown(f"- Unique ordinal distances in fit set: {np.unique(ordinal_distances_fit).tolist()}"))
display(Markdown(f"- Distance distribution in fit set:"))
for d, cnt in zip(*np.unique(ordinal_distances_fit, return_counts=True)):
    display(Markdown(f"  - distance {d}: {cnt} ({cnt/len(ordinal_distances_fit)*100:.1f}%)"))

sigma_model = make_sigma_model(ncm)
sigma_model.fit(X_fit, ordinal_distances_fit)

# Diagnostics on calibration set
sigma_pred_fit = sigma_model.predict(X_fit) if not ncm.startswith("c") else None
sigma_pred_cal = sigma_model.predict(X_cal) if not ncm.startswith("c") else None

if ncm.startswith("c"):
    probs_fit_ncm = sigma_model.predict_proba(X_fit)
    probs_cal_ncm = sigma_model.predict_proba(X_cal)
    probs_test_ncm = sigma_model.predict_proba(X_test)
    distance_classes = sigma_model.classes_
    sigma_pred_fit = (probs_fit_ncm * distance_classes).sum(axis=1)
    sigma_pred_cal = (probs_cal_ncm * distance_classes).sum(axis=1)
    sigma_pred_test = (probs_test_ncm * distance_classes).sum(axis=1)
else:
    probs_fit_ncm = None
    probs_cal_ncm = None
    probs_test_ncm = None
    distance_classes = np.arange(n_classes)
    sigma_pred_test = sigma_model.predict(X_test)

diag_ncm_fit = sigma_diagnostics(ordinal_distances_fit.astype(float), sigma_pred_fit)
diag_ncm_cal = sigma_diagnostics(ordinal_distances_cal.astype(float), sigma_pred_cal)

display(Markdown(f"\n**NCM model diagnostics:**"))
display(Markdown(f"- NCM R2 on fit set : {diag_ncm_fit['r2']:.3f}"))
display(Markdown(f"- NCM R2 on cal set : {diag_ncm_cal['r2']:.3f}"))
display(Markdown("""
Note: A low NCM R2 does NOT invalidate coverage (guaranteed by calibration
quantile). It only affects efficiency -- a better NCM model may produce
narrower prediction sets for confident predictions.
"""))

# Convert NCM distance probabilities to class pseudo-probabilities
def ncm_to_class_proba(probs_ncm, y_hard, n_cls, dist_classes):
    """Convert NCM distance distribution to class pseudo-probabilities."""
    n = len(y_hard)
    class_proba = np.zeros((n, n_cls))
    for i in range(n):
        yh = int(y_hard[i])
        for j in range(n_cls):
            d = abs(j - yh)
            if d < len(dist_classes):
                idx = np.where(dist_classes == d)[0]
                if len(idx) > 0 and probs_ncm is not None:
                    class_proba[i, j] = probs_ncm[i, idx[0]]
                else:
                    class_proba[i, j] = 1e-6
            else:
                class_proba[i, j] = 1e-6
        s = class_proba[i].sum()
        class_proba[i] = class_proba[i] / s if s > 0 else np.ones(n_cls) / n_cls
    return class_proba

if probs_cal_ncm is not None:
    pseudo_proba_cal  = ncm_to_class_proba(probs_cal_ncm,  y_cal_hard,  n_classes, distance_classes)
    pseudo_proba_test = ncm_to_class_proba(probs_test_ncm, y_test_hard, n_classes, distance_classes)
else:
    # Regressor NCM: Gaussian-like pseudo-probabilities
    def regressor_to_proba(sigma_pred, y_hard_, n_cls_):
        n_ = len(y_hard_)
        pp = np.zeros((n_, n_cls_))
        for i in range(n_):
            for j in range(n_cls_):
                pp[i, j] = np.exp(-abs(j - y_hard_[i]) * (1.0 / max(sigma_pred[i], 0.1)))
            pp[i] /= pp[i].sum()
        return pp
    pseudo_proba_cal  = regressor_to_proba(sigma_pred_cal,  y_cal_hard,  n_classes)
    pseudo_proba_test = regressor_to_proba(sigma_pred_test, y_test_hard, n_classes)

# Show pseudo-probability examples
display(Markdown("### Example: pseudo-probability conversion for 6 calibration molecules"))
fig_ex, axes_ex = plt.subplots(2, 3, figsize=(12, 6))
axes_ex = axes_ex.flatten()
sample_ex = np.random.default_rng(7).choice(len(cal_df), 6, replace=False)
for k, i in enumerate(sample_ex):
    ax = axes_ex[k]
    ax.bar(np.arange(n_classes), pseudo_proba_cal[i],
           color="#9C27B0", alpha=0.7, label="Pseudo-proba (NCM)")
    ax.bar(np.arange(n_classes), proba_cal[i],
           color="#2196F3", alpha=0.4, label="Real proba (model)")
    ax.axvline(y_cal_hard[i] - 0.5 + 0.5, color="orange", lw=2,
               linestyle="--", label=f"y_hat={int_to_class[y_cal_hard[i]]}")
    ax.axvline(y_cal[i] - 0.5 + 0.5, color="green", lw=2,
               linestyle=":", label=f"y_true={int_to_class[y_cal[i]]}")
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels([str(c) for c in classes_original], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"True:{int_to_class[y_cal[i]]}  Pred:{int_to_class[y_cal_hard[i]]}", fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    if k == 0:
        ax.legend(fontsize=6, loc="upper right")
plt.suptitle("Pseudo-probabilities (NCM, purple) vs real model probabilities (blue)\n"
             "Orange dashed=predicted class  Green dotted=true class", fontsize=10)
plt.tight_layout()
fig_ex.savefig(out_dir / "pseudo_proba_examples.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_ex)

# LAC scores using pseudo-probabilities
scores_ncm = 1.0 - pseudo_proba_cal[np.arange(len(y_cal)), y_cal]
q_hat_ncm = np.quantile(scores_ncm, q_level)
prob_threshold_ncm = 1 - q_hat_ncm

display(Markdown(f"\n**Approach B calibration:**"))
display(Markdown(f"- q_hat (NCM pseudo-proba): {q_hat_ncm:.4f}"))
display(Markdown(f"- Probability threshold   : {prob_threshold_ncm:.4f}"))

# Manual prediction sets using pseudo-probabilities (no MAPIE needed here)
sets_ncm = (1.0 - pseudo_proba_test) <= q_hat_ncm
covered_ncm = sets_ncm[np.arange(len(y_test)), y_test].astype(bool)
sizes_ncm   = sets_ncm.sum(axis=1)

display(Markdown(f"**Approach B results:**"))
display(Markdown(f"- Coverage  : {covered_ncm.mean():.3f}  (target >= {1-alpha:.2f})"))
display(Markdown(f"- Mean set size: {sizes_ncm.mean():.3f}"))
display(Markdown(f"- Singleton rate: {np.mean(sizes_ncm==1)*100:.1f}%"))

# ==============================================================================
# S6  Plain ordinal distance approach (Approach C - non-adaptive)
# ==============================================================================
display(Markdown("## S6  Approach C: Plain ordinal distance (non-adaptive)"))
display(Markdown(f"""
The simplest CP approach for hard-prediction classifiers:

  Conformity score: s_i = |y_true - y_hat|

No auxiliary model needed. The (1-alpha) quantile q_hat of calibration
ordinal distances determines the prediction set radius:

  C(x) = {{y : |y - y_hat| <= q_hat}}

This is the widest approach: it always includes all classes within q_hat
steps from y_hat, regardless of the molecule's chemical neighborhood.
Coverage is guaranteed, but efficiency may be poor -- especially when the
classifier is sometimes wrong by large margins.
"""))

scores_ord = ordinal_distances_cal.astype(float)  # |y_true - y_hat| on calibration
q_hat_ord  = np.quantile(scores_ord, q_level)

display(Markdown(f"- q_hat (ordinal distance): {q_hat_ord:.4f}  "
                 f"(prediction set includes all classes within {q_hat_ord:.1f} steps of y_hat)"))

# Prediction sets: include class j if |j - y_hat| <= q_hat
sets_ord = np.zeros((len(y_test), n_classes), dtype=bool)
for i in range(len(y_test)):
    for j in range(n_classes):
        if abs(j - y_test_hard[i]) <= q_hat_ord:
            sets_ord[i, j] = True

covered_ord = sets_ord[np.arange(len(y_test)), y_test].astype(bool)
sizes_ord   = sets_ord.sum(axis=1)

display(Markdown(f"**Approach C results:**"))
display(Markdown(f"- Coverage  : {covered_ord.mean():.3f}  (target >= {1-alpha:.2f})"))
display(Markdown(f"- Mean set size: {sizes_ord.mean():.3f}"))
display(Markdown(f"- Singleton rate: {np.mean(sizes_ord==1)*100:.1f}%"))

# ==============================================================================
# S7  Head-to-head comparison of all three approaches
# ==============================================================================
display(Markdown("## S7  Head-to-head comparison: all three approaches"))
display(Markdown(f"""
Objective: minimise mean set size, subject to coverage >= {1-alpha:.2f}.

All three approaches achieve the coverage constraint by construction.
Efficiency (set size) distinguishes them.
"""))

comparison_rows = []
for name, covered_, sizes_, q_ in [
    ("A: LAC (real proba)",         covered_lac, sizes_lac, q_hat_lac),
    ("B: NCM pseudo-proba",         covered_ncm, sizes_ncm, q_hat_ncm),
    ("C: Ordinal distance (plain)", covered_ord, sizes_ord, q_hat_ord),
]:
    comparison_rows.append({
        "Approach":      name,
        "Coverage":      f"{covered_.mean():.3f}",
        "Target":        f">= {1-alpha:.2f}",
        "Mean set size": f"{sizes_.mean():.3f}",
        "Singleton %":   f"{np.mean(sizes_==1)*100:.1f}%",
        "Full set %":    f"{np.mean(sizes_==n_classes)*100:.1f}%",
        "q_hat":         f"{q_:.4f}",
    })
display(pd.DataFrame(comparison_rows))

# Bar chart comparison
fig_cmp, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(11, 4))
labels_cmp = ["A: LAC", "B: NCM", "C: Ordinal"]
colors_cmp = ["#2196F3", "#9C27B0", "#FF9800"]
coverages_cmp = [covered_lac.mean(), covered_ncm.mean(), covered_ord.mean()]
sizes_cmp     = [sizes_lac.mean(),   sizes_ncm.mean(),   sizes_ord.mean()]
ax_c1.bar(labels_cmp, coverages_cmp, color=colors_cmp, alpha=0.8)
ax_c1.axhline(1 - alpha, color="red", lw=2, linestyle="--", label=f"Target {1-alpha:.0%}")
ax_c1.set_ylim(0, 1.05)
ax_c1.set_ylabel("Empirical coverage")
ax_c1.set_title("Coverage (all must reach target)")
ax_c1.legend(fontsize=8)
ax_c1.grid(True, alpha=0.3, axis="y")
ax_c2.bar(labels_cmp, sizes_cmp, color=colors_cmp, alpha=0.8)
ax_c2.axhline(1.0, color="green", lw=1.5, linestyle="--", label="Ideal (singleton)")
ax_c2.set_ylabel("Mean prediction set size")
ax_c2.set_title("Efficiency (smaller = better)")
ax_c2.legend(fontsize=8)
ax_c2.grid(True, alpha=0.3, axis="y")
plt.suptitle(f"Classification CP: three approaches compared  (alpha={alpha})", fontsize=11)
plt.tight_layout()
fig_cmp.savefig(out_dir / "class_approach_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_cmp)

# Per-molecule set size scatter: A vs B vs C
fig_sz, axes_sz = plt.subplots(1, 2, figsize=(10, 4))
axes_sz[0].scatter(sizes_lac, sizes_ncm, alpha=0.3, s=8, color="#9C27B0")
axes_sz[0].plot([0, n_classes], [0, n_classes], "k--", lw=1)
axes_sz[0].set_xlabel("Set size: A (LAC real proba)")
axes_sz[0].set_ylabel("Set size: B (NCM pseudo-proba)")
axes_sz[0].set_title("A vs B: per-molecule set sizes")
axes_sz[0].grid(True, alpha=0.3)
axes_sz[1].scatter(sizes_lac, sizes_ord, alpha=0.3, s=8, color="#FF9800")
axes_sz[1].plot([0, n_classes], [0, n_classes], "k--", lw=1)
axes_sz[1].set_xlabel("Set size: A (LAC real proba)")
axes_sz[1].set_ylabel("Set size: C (ordinal distance)")
axes_sz[1].set_title("A vs C: per-molecule set sizes")
axes_sz[1].grid(True, alpha=0.3)
plt.suptitle("Per-molecule set size: LAC vs NCM pseudo-proba vs ordinal distance", fontsize=10)
plt.tight_layout()
fig_sz.savefig(out_dir / "class_setsize_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_sz)

# ==============================================================================
# S8  Visualise prediction sets on 12 example molecules
# ==============================================================================
display(Markdown("## S8  Prediction set visualisation"))
display(Markdown("""
Each panel shows one test molecule.  Three rows of bars show:
  Blue   = real model probabilities (Approach A threshold)
  Purple = NCM pseudo-probabilities (Approach B threshold)
  Orange = ordinal distance indicator (Approach C: all classes within q_hat)

Green border = true class.  Set membership indicated by bar height vs threshold.
"""))

n_show = min(12, len(test_df))
sample_idx = np.random.default_rng(0).choice(len(test_df), n_show, replace=False)
sample_idx = sample_idx[np.argsort(y_test[sample_idx])]

fig_sets, axes_sets = plt.subplots(3, 4, figsize=(15, 10))
axes_sets = axes_sets.flatten()
for k, i in enumerate(sample_idx):
    ax = axes_sets[k]
    x = np.arange(n_classes)
    w = 0.25
    true_cls = y_test[i]
    y_hat_i  = y_test_hard[i]

    # Approach A: real probabilities
    ax.bar(x - w, proba_test[i], width=w, color="#2196F3", alpha=0.8, label="A real proba")
    # Approach B: pseudo-probabilities
    ax.bar(x,     pseudo_proba_test[i], width=w, color="#9C27B0", alpha=0.8, label="B NCM pseudo")
    # Approach C: within-distance indicator (1 = in set, 0 = not)
    in_set_c = np.array([1.0 if abs(j - y_hat_i) <= q_hat_ord else 0.0 for j in range(n_classes)])
    ax.bar(x + w, in_set_c * 0.5, width=w, color="#FF9800", alpha=0.8, label="C ordinal")

    # Thresholds
    ax.axhline(prob_threshold_lac, color="#2196F3", lw=1.5, linestyle="--", alpha=0.7)
    ax.axhline(prob_threshold_ncm, color="#9C27B0", lw=1.5, linestyle=":",  alpha=0.7)

    # Mark true class
    ax.axvspan(true_cls - 0.5, true_cls + 0.5, alpha=0.1, color="green")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes_original], fontsize=7)
    ax.set_ylim(0, 1.05)

    in_a = "OK" if covered_lac[i] else "X"
    in_b = "OK" if covered_ncm[i] else "X"
    in_c = "OK" if covered_ord[i] else "X"
    ax.set_title(
        f"True:{int_to_class[true_cls]}  Pred:{int_to_class[y_hat_i]}\n"
        f"A:{in_a}({int(sizes_lac[i])})  B:{in_b}({int(sizes_ncm[i])})  C:{in_c}({int(sizes_ord[i])})",
        fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    if k == 0:
        ax.legend(fontsize=5, loc="upper right")
plt.suptitle(
    f"Prediction sets: A=LAC(blue)  B=NCM(purple)  C=Ordinal(orange)  (alpha={alpha})\n"
    "Green band=true class.  OK=covered  X=missed.  Numbers=set size.",
    fontsize=9)
plt.tight_layout()
fig_sets.savefig(out_dir / "class_prediction_sets_3way.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_sets)

# ==============================================================================
# S9  NCM quality vs coverage/efficiency (mirrors regression S9c)
# ==============================================================================
display(Markdown("## S9  NCM quality vs coverage and efficiency"))
display(Markdown(r"""
'which ML method should be used for the NCM model?'

Same answer as regression:
- **Coverage is guaranteed regardless of NCM quality** (from calibration quantile).
- **Efficiency depends on NCM quality**: a better NCM model distinguishes
  molecules where the base classifier is likely correct (distance=0, high
  P(d=0)) from those where it is likely wrong (distance>0, low P(d=0)).
  This allows narrower prediction sets for confident predictions.

Below we simulate three NCM quality levels using the same real calibration
distances, compare q_hat and mean set size, and confirm coverage is stable.
"""))

np.random.seed(42)
_true_d_cal  = ordinal_distances_cal.astype(float)
_true_d_test = np.abs(y_test - y_test_hard).astype(float)

_ncm_configs_c = [
    ("Poor  (R2~0.1)",  2.0,  "#e74c3c"),
    ("Medium (R2~0.5)", 0.8,  "#3498db"),
    ("Good  (R2~0.9)",  0.25, "#27ae60"),
]

fig_ncm_c, axes_ncm_c = plt.subplots(2, 3, figsize=(14, 8))
ncm_c_results = []
for col_idx, (label, noise_scale, color) in enumerate(_ncm_configs_c):
    # Simulate pseudo-probabilities: P(d=0) varies with quality
    # Higher quality -> higher P(d=0) for correct predictions
    # We simulate by creating a soft version of the true distance
    _noisy_d_cal  = np.clip(_true_d_cal  + np.random.normal(0, noise_scale, len(_true_d_cal)), 0, n_classes - 1)
    _noisy_d_test = np.clip(_true_d_test + np.random.normal(0, noise_scale, len(_true_d_test)), 0, n_classes - 1)

    from sklearn.metrics import r2_score as _r2s
    _r2v = _r2s(_true_d_cal, _noisy_d_cal)

    # Build pseudo-proba from noisy distances using exponential decay
    def _d_to_proba(noisy_d, y_hard_, n_cls_):
        n_ = len(noisy_d)
        pp = np.zeros((n_, n_cls_))
        for ii in range(n_):
            for jj in range(n_cls_):
                pp[ii, jj] = np.exp(-abs(jj - y_hard_[ii]) / max(noisy_d[ii], 0.5))
            pp[ii] /= pp[ii].sum()
        return pp

    _pp_cal  = _d_to_proba(_noisy_d_cal,  y_cal_hard,  n_classes)
    _pp_test = _d_to_proba(_noisy_d_test, y_test_hard, n_classes)

    # LAC scores on calibration
    _scores = 1.0 - _pp_cal[np.arange(len(y_cal)), y_cal]
    _q = np.quantile(_scores, q_level)

    # Prediction sets on test
    _sets = (1.0 - _pp_test) <= _q
    _cov  = _sets[np.arange(len(y_test)), y_test].mean()
    _sz   = _sets.sum(axis=1).mean()

    ncm_c_results.append({"Model": label, "R2": round(_r2v, 3),
                           "q_hat": round(_q, 4), "Coverage": round(_cov, 3),
                           "Mean_set_size": round(_sz, 3)})

    # Row 1: noisy distance vs true distance
    ax_top = axes_ncm_c[0, col_idx]
    ax_top.scatter(_true_d_cal, _noisy_d_cal, alpha=0.35, s=8, color=color)
    _lim = max(_true_d_cal.max(), _noisy_d_cal.max()) + 0.5
    ax_top.plot([0, _lim], [0, _lim], "k--", lw=1)
    ax_top.set_xlabel("|y - y_hat|  (true distance)")
    ax_top.set_ylabel("NCM predicted distance")
    ax_top.set_title(f"{label}\nR2 = {_r2v:.2f}")
    ax_top.grid(True, alpha=0.3)

    # Row 2: LAC conformity score distribution
    ax_bot = axes_ncm_c[1, col_idx]
    ax_bot.hist(_scores, bins=20, density=True, alpha=0.65, color=color)
    ax_bot.axvline(_q, color="red", lw=2, linestyle="--",
                   label=f"q_hat={_q:.3f}  cov={_cov:.3f}  sz={_sz:.2f}")
    ax_bot.set_xlabel("LAC score on calibration set")
    ax_bot.set_ylabel("Density")
    ax_bot.set_title(f"Coverage={_cov:.3f}  Mean set size={_sz:.3f}")
    ax_bot.legend(fontsize=7)
    ax_bot.grid(True, alpha=0.3)
plt.suptitle(
    f"NCM quality vs CP outcome  (alpha={alpha}, target={1-alpha:.0%})\n"
    "Coverage is stable; efficiency (set size) improves with better NCM.",
    fontsize=10)
plt.tight_layout()
fig_ncm_c.savefig(out_dir / "class_ncm_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_ncm_c)

ncm_c_df = pd.DataFrame(ncm_c_results)
display(ncm_c_df)
display(Markdown("""
**Summary:**
- Coverage is robust to NCM choice: all quality levels achieve the target.
- A better NCM model (higher R2) produces more peaked pseudo-probabilities
  near the predicted class, resulting in smaller (more informative) prediction sets.
- For hard-prediction QSAR models, NCM approach B outperforms plain ordinal
  distance C in efficiency, while preserving the same coverage guarantee.
- Approach A (real probabilities) provides the best efficiency when a
  well-calibrated probabilistic classifier is available.
"""))

# ==============================================================================
# S10  Coverage guarantee sweep across alpha
# ==============================================================================
display(Markdown("## S10  Coverage guarantee across alpha levels"))
display(Markdown("""
All three approaches guarantee coverage >= 1-alpha for any alpha.
We verify this by sweeping alpha and recomputing coverage and mean set size
using the already-computed calibration scores (no refitting needed).
"""))

alphas_sw = np.arange(0.05, 0.51, 0.05)
res_sw = {name: {"cov": [], "sz": []} for name in ["A_lac", "B_ncm", "C_ord"]}

for a in alphas_sw:
    _ql = min(np.ceil((n_cal + 1) * (1 - a)) / n_cal, 1.0)

    # A: LAC
    _q = np.quantile(scores_lac, _ql)
    _sets = (1.0 - proba_test) <= _q
    res_sw["A_lac"]["cov"].append(_sets[np.arange(len(y_test)), y_test].mean())
    res_sw["A_lac"]["sz"].append(_sets.sum(axis=1).mean())

    # B: NCM pseudo-proba
    _q = np.quantile(scores_ncm, _ql)
    _sets = (1.0 - pseudo_proba_test) <= _q
    res_sw["B_ncm"]["cov"].append(_sets[np.arange(len(y_test)), y_test].mean())
    res_sw["B_ncm"]["sz"].append(_sets.sum(axis=1).mean())

    # C: ordinal distance
    _q = np.quantile(scores_ord, _ql)
    _sets_c = np.zeros((len(y_test), n_classes), dtype=bool)
    for i in range(len(y_test)):
        for j in range(n_classes):
            _sets_c[i, j] = abs(j - y_test_hard[i]) <= _q
    res_sw["C_ord"]["cov"].append(_sets_c[np.arange(len(y_test)), y_test].mean())
    res_sw["C_ord"]["sz"].append(_sets_c.sum(axis=1).mean())

fig_sw, (ax_sw1, ax_sw2) = plt.subplots(1, 2, figsize=(11, 4))
style_map = {"A_lac": ("o-", "#2196F3", "A: LAC"),
             "B_ncm": ("s--", "#9C27B0", "B: NCM pseudo"),
             "C_ord": ("^:", "#FF9800", "C: Ordinal")}
for key, (style, color, label) in style_map.items():
    ax_sw1.plot(1 - alphas_sw, res_sw[key]["cov"], style, color=color, label=label)
    ax_sw2.plot(1 - alphas_sw, res_sw[key]["sz"],  style, color=color, label=label)
ax_sw1.plot([0.5, 0.95], [0.5, 0.95], "k:", lw=1, label="y = 1-alpha (ideal)")
ax_sw1.set_xlabel("Target coverage  (1 - alpha)")
ax_sw1.set_ylabel("Empirical coverage")
ax_sw1.set_title("Coverage guarantee across alpha levels")
ax_sw1.legend(fontsize=7)
ax_sw1.grid(True, alpha=0.3)
ax_sw2.set_xlabel("Target coverage  (1 - alpha)")
ax_sw2.set_ylabel("Mean prediction set size")
ax_sw2.set_title("Set size vs coverage target\n(efficiency-coverage trade-off)")
ax_sw2.legend(fontsize=7)
ax_sw2.grid(True, alpha=0.3)
plt.suptitle("Classification CP: coverage and efficiency across alpha values", fontsize=11)
plt.tight_layout()
fig_sw.savefig(out_dir / "class_coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_sw)

# ==============================================================================
# S11  Per-class coverage breakdown
# ==============================================================================
display(Markdown("## S11  Per-class coverage (marginal vs conditional)"))
display(Markdown("""
Marginal coverage (averaged over all test molecules) is guaranteed >= 1-alpha.
Per-class coverage may vary -- classes with fewer calibration samples or
higher model error may show under- or over-coverage.
This is NOT a violation of CP theory; it reflects the marginal (not conditional)
nature of the guarantee.
"""))

per_class_rows = []
for cls_int, cls_name in int_to_class.items():
    mask = y_test == cls_int
    if mask.sum() == 0:
        continue
    per_class_rows.append({
        "Class": cls_name,
        "n_test": int(mask.sum()),
        "Cov_A": f"{covered_lac[mask].mean():.3f}",
        "Cov_B": f"{covered_ncm[mask].mean():.3f}",
        "Cov_C": f"{covered_ord[mask].mean():.3f}",
        "Sz_A": f"{sizes_lac[mask].mean():.2f}",
        "Sz_B": f"{sizes_ncm[mask].mean():.2f}",
        "Sz_C": f"{sizes_ord[mask].mean():.2f}",
    })
display(pd.DataFrame(per_class_rows))

# ==============================================================================
# S12  Save results
# ==============================================================================
result_df = pd.DataFrame({
    "Smiles":       test_df["Smiles"].values,
    "True":         test_df[target_col].values,
    "Pred":         [int_to_class[p] for p in y_test_pred],
    # Approach A
    "Cov_A":        covered_lac.astype(int),
    "SetSize_A":    sizes_lac,
    # Approach B
    "Cov_B":        covered_ncm.astype(int),
    "SetSize_B":    sizes_ncm,
    # Approach C
    "Cov_C":        covered_ord.astype(int),
    "SetSize_C":    sizes_ord,
    # Probabilities
    **{f"p_{c}": proba_test[:, i] for i, c in enumerate(classes_original)},
    **{f"pseudo_p_{c}": pseudo_proba_test[:, i] for i, c in enumerate(classes_original)},
})

metrics_df = pd.DataFrame(comparison_rows)

save_dict = {
    "base_model": base_model,
    "sigma_model": sigma_model,
    "ncm": ncm,
    "q_hat_lac": q_hat_lac,
    "q_hat_ncm": q_hat_ncm,
    "q_hat_ord": q_hat_ord,
    "classes_original": classes_original,
    "class_to_int": class_to_int,
    "int_to_class": int_to_class,
    "distance_classes": distance_classes,
    "alpha": alpha,
    "meta": meta,
}
with open(product["ncmodel"], "wb") as fh:
    pickle.dump(save_dict, fh)

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w, sheet_name="Predictions", index=False)
    metrics_df.to_excel(w, sheet_name="Metrics", index=False)
    ncm_c_df.to_excel(w, sheet_name="NCM_comparison", index=False)

display(Markdown("## [OK] Classification tutorial complete."))
display(Markdown(f"- Results  : {product['data']}"))
display(Markdown(f"- Model    : {product['ncmodel']}"))
display(Markdown(f"- Plots    : {out_dir}"))
