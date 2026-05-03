import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown

from lightgbm import LGBMClassifier
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import LACConformityScore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
from qubounds.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from qubounds.mapie_diagnostic import sigma_diagnostics
from qubounds.mapie_class_lac import (
    NCMProbabilisticClassifier,
    train_conformal_classifier_hard,
)
from scipy.stats import ks_2samp, spearmanr, mannwhitneyu

# -*- coding: utf-8 -*-
"""
tasks/tutorial/mapie_classification.py
----------------------------------------
Tutorial: Conformal Prediction for QSAR - Classification

Table of Contents
-----------------
- [S0  Resolve upstream](#S0--Resolve-upstream)
- [S1  Formal definitions](#S1--Formal-definitions)
- [S2  Data splits and class balance](#S2--Data-splits-and-class-balance)
- [S3  Fingerprints and base classifier](#S3--Fingerprints-and-base-classifier)
- [S4  Approach A: LAC with real model probabilities](#S4--Approach-A-LAC-with-real-model-probabilities)
- [S5  Approach B: NCM pseudo-probabilities (hard labels)](#S5--Approach-B-NCM-pseudo-probabilities-hard-labels)
- [S6  Head-to-head comparison: A vs B](#S6--Head-to-head-comparison-A-vs-B)
- [S7  Prediction set visualisation: 12 example molecules](#S7--Prediction-set-visualisation-12-example-molecules)
- [S8  NCM quality vs coverage and efficiency](#S8--NCM-quality-vs-coverage-and-efficiency)
- [S9  Exchangeability check (KS test)](#S9--Exchangeability-check-KS-test)
- [S10  Coverage guarantee sweep across alpha](#S10--Coverage-guarantee-sweep-across-alpha)
- [S11  Per-class coverage](#S11--Per-class-coverage)
- [S12  AD comparison (only when ad_cols are available)](#S12--AD-comparison-only-when-ad_cols-are-available)
"""

"""
tasks/tutorial/mapie_classification.py
----------------------------------------
Tutorial: Conformal Prediction for QSAR - Classification
=========================================================

Two CP approaches are demonstrated and compared:

  Approach A  LAC with model probabilities
              The classifier produces predict_proba() output.
              Conformity score: s(x,y) = 1 - p_hat(y|x)
              Gold standard when calibrated probabilities are available.
              Used here with a LightGBM + Platt-calibration base model.

  Approach B  Hard predictions + ordinal NCM (pseudo-probabilities)
              The classifier produces only a hard label y_hat.
              An auxiliary NCM model learns P(|y - y_hat| = d | x).
              Distance probabilities are converted to class pseudo-probabilities:
                P_pseudo(class=j | x, y_hat) = P(distance = |j - y_hat| | x)
              LAC is then applied to pseudo-probabilities.
              This is the approach used for VEGA models in the paper.
              Implemented via NCMProbabilisticClassifier from mapie_class_lac.py.

Both approaches support external predictions from file (simulating VEGA output).

Key concepts:
  - Formal definitions (per-prediction vs per-model)
  - Calibration objective: minimise mean set size subject to coverage >= 1-alpha
  - LAC threshold visualised per molecule
  - NCM quality vs coverage/efficiency
  - Coverage guarantee sweep across alpha
"""

matplotlib.use("Agg")
%matplotlib inline

# + tags=["parameters"]
alpha               = 0.1
ncm                 = "crfecfp"
cache_path          = None
product             = None
upstream            = None
dataset             = None
dataset_config = None
base_model = None
# -

Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
out_dir = Path(product["data"]).parent

# ==============================================================================
# S0  Resolve upstream
# ==============================================================================
tag        = f"tutorial_load_class_{dataset}"
data = upstream["tutorial_load_class_*"][tag]["data"]
meta_path  = upstream["tutorial_load_class_*"][tag]["meta"]

with open(meta_path) as f:
    meta = json.load(f)
meta


target_col = meta["target_col"]
hard_col = meta.get("hard_col", None)
prob_col = meta.get("prob_col", None)

# all classes  should be integer
cfg  = dataset_config.get(dataset, {})
ad_cols           = cfg.get("ad_cols",             [])
ad_col_directions = cfg.get("ad_col_directions",   [])
n_quantile_bins   = int(cfg.get("n_quantile_bins", 5))

print(cfg.get("classes",{}))

int_to_class = {int(k): v for k, v in cfg.get("classes",{}).items()}
print(int_to_class)
class_to_int = {v: k for k, v in int_to_class.items()}  # invert mapping
classes_original = list(int_to_class.values())
print(classes_original)
n_classes = len(classes_original)
valid_classes = set(int_to_class.keys())

if hard_col is None and prob_col is None and base_model == "file": 
   base_model = "catboost" 
ad_cols = meta.get("ad_cols", [])
ad_col_directions = meta.get("ad_col_directions",[])

display(Markdown("# CONFORMAL PREDICTION TUTORIAL - CLASSIFICATION"))
display(Markdown(f"## Dataset: {meta['dataset']}   Target: {target_col}"))

# ==============================================================================
# S1  Formal definitions
# ==============================================================================
display(Markdown("## S1  Formal definitions"))
display(Markdown(f"""
In **regression**, CP outputs a *prediction interval* [lower, upper].
In **classification**, CP outputs a *prediction set* -- a subset of classes
guaranteed to contain the true class with probability >= 1-alpha.

**Nonconformity score** s(x_i, y_i) -- *per-prediction*, on calibration set.

  Approach A (LAC, real proba)  : s = 1 - p_hat(y_true | x)
  Approach B (NCM pseudo-proba) : s = 1 - p_pseudo(y_true | x, y_hat)

Higher score = more surprising = model less confident about the true class.

**Calibration quantile** q_hat -- *per-model*, computed once:
  q_hat = quantile({{s_1,...,s_n}}, level=ceil((n+1)*(1-alpha))/n)

**Prediction set** C(x) -- *per-prediction*, at inference time:
  C(x) = {{y : p_hat(y|x) >= 1 - q_hat}}

**Coverage** -- two granularities:
  Per-prediction : cov_i = 1[y_i in C(x_i)]         (binary per molecule)
  Per-model      : Cov = mean(cov_i) over test set    (validity criterion)
  CP guarantee   : Cov >= 1-alpha in expectation.

**Efficiency** -- two granularities:
  Per-prediction : sz_i = |C(x_i)|                   (set size, integer >= 1)
  Per-model      : SZ = mean(sz_i)                    (efficiency metric)
  Ideal          : sz_i = 1 for all (singleton sets).

**Calibration objective:**
  Minimise SZ = mean set size,  subject to: Cov >= {1-alpha:.2f}

CP achieves this by construction. Approach A achieves smaller SZ than B
when the classifier produces well-calibrated probabilities, because it can
distinguish between classes more finely than the NCM ordinal distance.

**Conditional vs marginal coverage:**
CP guarantees *marginal* coverage (averaged over all test molecules).
Per-class coverage may deviate. Minimum calibration samples needed per
class: ceil(1/alpha) - 1 = {int(np.ceil(1/alpha))-1} (for alpha={alpha}).
"""))

# ==============================================================================
# S2  Data splits and class balance
# ==============================================================================
display(Markdown("## S2  Data splits and class balance"))

train_df = pd.read_excel(data, sheet_name="Training")
test_df  = pd.read_excel(data,  sheet_name="Test")
train_df = train_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["Smiles", target_col]).reset_index(drop=True)
test_df.head()

# check for nans
for col_ in [target_col, hard_col, prob_col]:
    if col_ is not None:
        assert not train_df[col_].isna().any(), f"Train column {col_} contains NaNs"
        assert not test_df[col_].isna().any(), f"Test column {col_} contains NaNs"

train_df["label"] = train_df[target_col]
test_df["label"]  = test_df[target_col]

#--- CLEAN (invalidate bad labels like "-")
train_df[target_col] = train_df[target_col].where(
    train_df[target_col].isin(valid_classes), np.nan
)
test_df[target_col] = test_df[target_col].where(
    test_df[target_col].isin(valid_classes), np.nan
)

test_df = test_df.dropna(subset=["label"]).reset_index(drop=True)



if base_model == "file": 
    df_test_ad = test_df.copy()
    _available_ad = []
    if ad_cols:
        for _ac in ad_cols:
            if _ac in test_df.columns:
                df_test_ad[_ac] = test_df["Smiles"].map(
                    test_df.set_index("Smiles")[_ac]).values
                _available_ad.append(_ac)
            else:
                display(Markdown(f"- WARNING: AD column `{_ac}` not found in external file."))
        if _available_ad:
            display(Markdown(f"- AD columns loaded: {_available_ad}"))
    fit_df = train_df.copy().reset_index(drop=True)
    cal_df, test_df = train_test_split(test_df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)
    cal_df = cal_df.reset_index(drop=True)                
else:   
    df_test_ad = None
    _available_ad = [] 
    fit_df, cal_df = train_test_split(
        train_df, test_size=0.2, random_state=42,
        stratify=train_df["label"] if n_classes <= 10 else None
    )
    fit_df = fit_df.reset_index(drop=True)
    cal_df = cal_df.reset_index(drop=True)

display(Markdown(f"- Fit set        : {len(fit_df)} molecules"))
display(Markdown(f"- Calibration set: {len(cal_df)} molecules"))
display(Markdown(f"- Test set       : {len(test_df)} molecules"))
display(Markdown(f"- Classes        : {classes_original}"))

min_required = int(np.ceil(1 / alpha)) - 1
class_counts_cal = cal_df["label"].value_counts().sort_index()
for lbl, cnt in class_counts_cal.items():
    flag = "OK" if cnt >= min_required else f"WARNING: needs >={min_required}"
    display(Markdown(f"  - {int_to_class[lbl]:>20s}: {cnt:>4d}  {flag}"))

# ==============================================================================
# S3  Fingerprints and base classifier
# ==============================================================================
display(Markdown("## S3  Fingerprints and base classifier"))
display(Markdown("""
ONE base classifier is trained and used for both CP approaches.
LightGBM with Platt scaling produces well-calibrated predict_proba() output,
which is needed for Approach A. Approach B uses only the hard label y_hat.

If pred_col is set, hard predictions are read from that file
(simulating a VEGA model providing only a class label), the same pattern
used for VEGA models 
                 in the paper. Otherwise the internal base model is used
for both approaches.
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

# Base model with calibrated probabilities (for Approach A and for generating
# internal hard labels used as a stand-in when no external file is provided)
if base_model in ["lgbm", "catboost"]:
    if base_model == "lgbm":
        base_lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1
        )
    else:
        base_lgbm = CatBoostClassifier(
            n_estimators=200, learning_rate=0.05,
            random_state=42, 
        )        
    base_model = CalibratedClassifierCV(base_lgbm, cv=3, method="sigmoid")
    base_model.fit(X_fit, y_fit)

    y_fit_hard   = base_model.predict(X_fit)
    y_cal_hard   = base_model.predict(X_cal)
    y_test_hard  = base_model.predict(X_test)
    proba_cal    = base_model.predict_proba(X_cal)
    proba_test   = base_model.predict_proba(X_test)
elif base_model == "file":
    proba_cal = None;  proba_test = None;   proba_fit = None
    y_fit_hard = None; y_cal_hard = None; y_test_hard = None
    if prob_col is not None:
        proba_fit  = fit_df[prob_col].values
        proba_cal  = cal_df[prob_col].values
        proba_test = test_df[prob_col].values
    if hard_col is not None:
        y_fit_hard  = fit_df[hard_col].values # .map(class_to_int)
        y_cal_hard  = cal_df[hard_col].values
        y_test_hard = test_df[hard_col].values
else:
    assert False,f"{base_model} not supported"

acc_cal  = accuracy_score(y_cal,  y_cal_hard)
acc_test = accuracy_score(y_test, y_test_hard)
display(Markdown(f"- Base model {base_model} accuracy on cal set  : {acc_cal:.3f} "))
display(Markdown(f"- Base model {base_model} accuracy on test set : {acc_test:.3f}"))

report = classification_report(y_test, y_test_hard,
                                target_names=[str(c) for c in classes_original])
display(Markdown(f"```\n{report}\n```"))


# ==============================================================================
# S4  Approach A: LAC with real model probabilities
# ==============================================================================

if proba_cal is not None:
    display(Markdown("## S4  Approach A: LAC with model probabilities"))
    display(Markdown(f"""
    LAC conformity score: s_i = 1 - p_hat(y_true | x_i)

    The (1-alpha) quantile q_hat determines the probability threshold:
    class j enters C(x) if p_hat(j|x) >= 1 - q_hat

    alpha={alpha}, target coverage={1-alpha:.0%}.
    """))

    scores_lac  = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n_cal       = len(scores_lac)
    q_level     = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_hat_lac   = np.quantile(scores_lac, q_level)
    prob_threshold_lac = 1 - q_hat_lac

    display(Markdown(f"- Calibration n  : {n_cal}"))
    display(Markdown(f"- Quantile level : {q_level:.4f}  [= ceil((n+1)*(1-alpha)) / n]"))
    display(Markdown(f"- q_hat (LAC)    : {q_hat_lac:.4f}"))
    display(Markdown(f"- p threshold    : {prob_threshold_lac:.4f}"))

    fig_a, axes_a = plt.subplots(1, 2, figsize=(11, 4))
    axes_a[0].hist(scores_lac, bins="auto", density=True, alpha=0.7, color="#9C27B0")
    axes_a[0].axvline(q_hat_lac, color="red", lw=2, linestyle="--",
                    label=f"q_hat={q_hat_lac:.3f}  ({1-alpha:.0%})")
    axes_a[0].set_xlabel("LAC score  1 - p_hat(true class | x)")
    axes_a[0].set_ylabel("Density")
    axes_a[0].set_title("LAC calibration scores (Approach A)")
    axes_a[0].legend(fontsize=8); axes_a[0].grid(True, alpha=0.3)
    for cls_int in range(n_classes):
        mask = y_cal == cls_int
        if mask.sum() > 0:
            axes_a[1].hist(scores_lac[mask], bins=20, density=True, alpha=0.5,
                        label=str(classes_original[cls_int]))
    axes_a[1].axvline(q_hat_lac, color="red", lw=2, linestyle="--",
                    label=f"q_hat={q_hat_lac:.3f}")
    axes_a[1].set_xlabel("LAC score"); axes_a[1].set_ylabel("Density")
    axes_a[1].set_title("LAC scores by true class")
    axes_a[1].legend(fontsize=7); axes_a[1].grid(True, alpha=0.3)
    plt.suptitle("Approach A: LAC calibration score distributions", fontsize=11)
    plt.tight_layout()
    fig_a.savefig(out_dir / "class_lac_scores.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig_a)

    # MAPIE conformal classifier - Approach A
    mapie_lac = SplitConformalClassifier(
        estimator=base_model, conformity_score=LACConformityScore(),
        prefit=True, confidence_level=1 - alpha,
    )
    mapie_lac.estimator_ = base_model
    mapie_lac.conformalize(X_conformalize=X_cal, y_conformalize=y_cal)

    _, y_sets_lac = mapie_lac.predict_set(X_test)
    if y_sets_lac.ndim == 3:
        y_sets_lac = np.squeeze(y_sets_lac, axis=2)

    covered_lac = y_sets_lac[np.arange(len(y_test)), y_test].astype(bool)
    sizes_lac   = y_sets_lac.sum(axis=1)

    display(Markdown(f"**Approach A results:**"))
    display(Markdown(f"- Coverage      : {covered_lac.mean():.3f}  (target >= {1-alpha:.2f})"))
    display(Markdown(f"- Mean set size : {sizes_lac.mean():.3f}"))
    display(Markdown(f"- Singleton %   : {np.mean(sizes_lac==1)*100:.1f}%"))
else:
    covered_lac = None
    sizes_lac = None
    n_cal = 0
    q_hat_lac = None
# ==============================================================================
# S5  Approach B: NCM pseudo-probabilities (hard labels)
# ==============================================================================
if y_cal_hard is not None:
    display(Markdown("## S5  Approach B: NCM pseudo-probabilities from hard labels"))
    display(Markdown(f"""
    VEGA models output only a hard class label y_hat.
    An auxiliary NCM classifier learns P(|y - y_hat| = d | x) for distances d=0,1,...

    Conversion to class pseudo-probabilities:
    P_pseudo(class=j | x, y_hat) = P(distance = |j - y_hat| | x)

    Classes equidistant from y_hat get equal pseudo-probability (ordinal symmetry).

    Example (binary, y_hat=0):
    NCM: P(d=0)=0.70, P(d=1)=0.30
    => P(class=0) = 0.70  (distance 0 from y_hat=0)
        P(class=1) = 0.30  (distance 1 from y_hat=0)

    Implemented via train_conformal_classifier_hard() from tasks.mapie_class_lac --
    the same production function used for all VEGA models in the paper.

    Coverage guarantee is independent of NCM quality (comes from calibration
    quantile). NCM quality only affects efficiency (set size).
    """))

    print("y_fit_hard", np.isnan(y_fit_hard).any())
    print("y_fit", np.isnan(y_fit).any())
    
    ordinal_distances_fit = np.abs(y_fit - y_fit_hard).astype(int)
    ordinal_distances_cal = np.abs(y_cal - y_cal_hard).astype(int)

    display(Markdown(f"### Distances in fit set: {np.unique(ordinal_distances_fit).tolist()}"))
    for d, cnt in zip(*np.unique(ordinal_distances_fit, return_counts=True)):
        display(Markdown(f"- distance {d}: {cnt} ({cnt/len(ordinal_distances_fit)*100:.1f}%)"))

    df_fit_for_ncm = pd.DataFrame({"Smiles": fit_df["Smiles"].values,
                                    "Exp": y_fit, "Pred": y_fit_hard})
    df_cal_for_ncm = pd.DataFrame({"Smiles": cal_df["Smiles"].values,
                                    "Exp": y_cal, "Pred": y_cal_hard})

    _ncm_model_path = str(out_dir / "ncm_tutorial_model.pkl")
    saved_ncm = train_conformal_classifier_hard(
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

    mapie_ncm     = saved_ncm["mapie"]
    ncm_estimator = mapie_ncm.estimator_      # NCMProbabilisticClassifier
    sigma_model   = saved_ncm["sigma_model"]  # underlying RF / GBT classifier
    distance_classes = (sigma_model.classes_
                        if hasattr(sigma_model, "classes_")
                        else np.arange(n_classes))

    # Diagnostics
    if ncm.startswith("c") or ncm.startswith("o"):
        _pf = sigma_model.predict_proba(X_fit)
        _pc = sigma_model.predict_proba(X_cal)
        sigma_pred_fit = (_pf * distance_classes).sum(axis=1)
        sigma_pred_cal = (_pc * distance_classes).sum(axis=1)
    else:
        sigma_pred_fit = sigma_model.predict(X_fit)
        sigma_pred_cal = sigma_model.predict(X_cal)

    diag_fit = sigma_diagnostics(ordinal_distances_fit.astype(float), sigma_pred_fit)
    diag_cal = sigma_diagnostics(ordinal_distances_cal.astype(float), sigma_pred_cal)
    display(Markdown(f"- NCM R2 on fit set : {diag_fit['r2']:.3f}"))
    display(Markdown(f"- NCM R2 on cal set : {diag_cal['r2']:.3f}"))
    display(Markdown("""
    Note: low NCM R2 does NOT invalidate CP coverage.
    The calibration quantile provides the guarantee regardless.
    NCM quality only affects efficiency (set size).
    """))

    # Get pseudo-probabilities from NCMProbabilisticClassifier.predict_proba()
    # -- the exact call MAPIE makes internally.
    ncm_estimator.y_pred = y_cal_hard
    pseudo_proba_cal = ncm_estimator.predict_proba(X_cal)

    ncm_estimator.y_pred = y_test_hard
    pseudo_proba_test = ncm_estimator.predict_proba(X_test)

    # Show pseudo-probability examples vs real probabilities
    display(Markdown("### Pseudo-probability vs real probability: 6 example molecules"))
    fig_ex, axes_ex = plt.subplots(2, 3, figsize=(12, 6))
    axes_ex = axes_ex.flatten()
    sample_ex = np.random.default_rng(7).choice(len(cal_df), 6, replace=False)
    for k, i in enumerate(sample_ex):
        ax = axes_ex[k]
        x = np.arange(n_classes)
        w = 0.35
        if proba_cal is not None:
            ax.bar(x - w/2, proba_cal[i],       width=w, color="#2196F3", alpha=0.8,
                label="Real proba (model)")
        ax.bar(x + w/2, pseudo_proba_cal[i], width=w, color="#FF9800", alpha=0.8,
            label="Pseudo-proba (NCM)")
        ax.axvline(y_cal_hard[i], color="orange", lw=2, linestyle="--",
                label=f"y_hat={int_to_class[int(y_cal_hard[i])]}")
        ax.axvline(y_cal[i],      color="green",  lw=2, linestyle=":",
                label=f"y_true={int_to_class[y_cal[i]]}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes_original], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"True:{int_to_class[y_cal[i]]}  "
                    f"Pred:{int_to_class[int(y_cal_hard[i])]}", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        if k == 0:
            ax.legend(fontsize=6, loc="upper right")
    plt.suptitle("Blue=real model proba  Orange=NCM pseudo-proba\n"
                "Orange dashed=y_hat  Green dotted=y_true", fontsize=10)
    plt.tight_layout()
    fig_ex.savefig(out_dir / "pseudo_proba_examples.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig_ex)

    # LAC scores and prediction sets for Approach B
    scores_ncm = 1.0 - pseudo_proba_cal[np.arange(len(y_cal)), y_cal]
    try:
        q_hat_ncm  = np.quantile(scores_ncm, q_level)
        prob_threshold_ncm = 1 - q_hat_ncm
        display(Markdown(f"- q_hat (NCM pseudo-proba): {q_hat_ncm:.4f}"))
        display(Markdown(f"- Probability threshold   : {prob_threshold_ncm:.4f}"))        
    except Exception:
        q_hat_ncm = None
        prob_threshold_ncm = None

    ncm_estimator.y_pred = y_test_hard
    _, y_sets_ncm = mapie_ncm.predict_set(X_test)
    if y_sets_ncm.ndim == 3:
        y_sets_ncm = np.squeeze(y_sets_ncm, axis=2)
    covered_ncm = y_sets_ncm[np.arange(len(y_test)), y_test].astype(bool)
    sizes_ncm   = y_sets_ncm.sum(axis=1)

    display(Markdown(f"**Approach B results:**"))
    display(Markdown(f"- Coverage      : {covered_ncm.mean():.3f}  (target >= {1-alpha:.2f})"))
    display(Markdown(f"- Mean set size : {sizes_ncm.mean():.3f}"))
    display(Markdown(f"- Singleton %   : {np.mean(sizes_ncm==1)*100:.1f}%"))

# ==============================================================================
# S6  Head-to-head comparison: A vs B
# ==============================================================================
if proba_cal is not None and y_cal_hard is not None:
    display(Markdown("## S6  Head-to-head comparison: A vs B"))
    display(Markdown(f"""
    Objective: minimise mean set size, subject to coverage >= {1-alpha:.2f}.
    Both approaches achieve the coverage constraint by construction.
    """))

    comparison_rows = [
        {"Approach": "A: LAC (real proba)", "Coverage": f"{covered_lac.mean():.3f}",
        "Target": f">= {1-alpha:.2f}", "Mean set size": f"{sizes_lac.mean():.3f}",
        "Singleton %": f"{np.mean(sizes_lac==1)*100:.1f}%",
        "Full set %": f"{np.mean(sizes_lac==n_classes)*100:.1f}%",
        "q_hat": f"{q_hat_lac:.4f}"},
        {"Approach": "B: NCM pseudo-proba", "Coverage": f"{covered_ncm.mean():.3f}",
        "Target": f">= {1-alpha:.2f}", "Mean set size": f"{sizes_ncm.mean():.3f}",
        "Singleton %": f"{np.mean(sizes_ncm==1)*100:.1f}%",
        "Full set %": f"{np.mean(sizes_ncm==n_classes)*100:.1f}%",
        "q_hat": f"{q_hat_ncm}"},
    ]
    display(pd.DataFrame(comparison_rows))

    fig_cmp, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(9, 4))
    labels_cmp    = ["A: LAC", "B: NCM"]
    colors_cmp    = ["#2196F3", "#FF9800"]
    coverages_cmp = [covered_lac.mean(), covered_ncm.mean()]
    sizes_cmp     = [sizes_lac.mean(),   sizes_ncm.mean()]
    ax_c1.bar(labels_cmp, coverages_cmp, color=colors_cmp, alpha=0.8)
    ax_c1.axhline(1 - alpha, color="red", lw=2, linestyle="--",
                label=f"Target {1-alpha:.0%}")
    ax_c1.set_ylim(0, 1.05); ax_c1.set_ylabel("Empirical coverage")
    ax_c1.set_title("Coverage"); ax_c1.legend(fontsize=8)
    ax_c1.grid(True, alpha=0.3, axis="y")
    ax_c2.bar(labels_cmp, sizes_cmp, color=colors_cmp, alpha=0.8)
    ax_c2.axhline(1.0, color="green", lw=1.5, linestyle="--",
                label="Ideal (singleton)")
    ax_c2.set_ylabel("Mean prediction set size")
    ax_c2.set_title("Efficiency (smaller = better)")
    ax_c2.legend(fontsize=8); ax_c2.grid(True, alpha=0.3, axis="y")
    plt.suptitle(f"Classification CP: A vs B  (alpha={alpha})", fontsize=11)
    plt.tight_layout()
    fig_cmp.savefig(out_dir / "class_approach_comparison.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig_cmp)

    # Per-molecule set size scatter
    fig_sz, ax_sz = plt.subplots(figsize=(5, 4))
    ax_sz.scatter(sizes_lac, sizes_ncm, alpha=0.3, s=8, color="#9C27B0")
    ax_sz.plot([0, n_classes], [0, n_classes], "k--", lw=1)
    ax_sz.set_xlabel("Set size: A (LAC real proba)")
    ax_sz.set_ylabel("Set size: B (NCM pseudo-proba)")
    ax_sz.set_title("Per-molecule set sizes: A vs B")
    ax_sz.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_sz.savefig(out_dir / "class_setsize_scatter.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig_sz)
else:
    comparison_rows = [
        {"Approach": "B: NCM pseudo-proba", "Coverage": f"{covered_ncm.mean():.3f}",
        "Target": f">= {1-alpha:.2f}", "Mean set size": f"{sizes_ncm.mean():.3f}",
        "Singleton %": f"{np.mean(sizes_ncm==1)*100:.1f}%",
        "Full set %": f"{np.mean(sizes_ncm==n_classes)*100:.1f}%",
        "q_hat": f"{q_hat_ncm}"},
    ]
    display(pd.DataFrame(comparison_rows))
# ==============================================================================
# S7  Prediction set visualisation: 12 example molecules
# ==============================================================================
display(Markdown("## S7  Prediction set visualisation"))

n_show = min(12, len(test_df))
sample_idx = np.random.default_rng(0).choice(len(test_df), n_show, replace=False)
sample_idx = sample_idx[np.argsort(y_test[sample_idx])]

fig_sets, axes_sets = plt.subplots(3, 4, figsize=(15, 10))
axes_sets = axes_sets.flatten()
for k, i in enumerate(sample_idx):
    ax = axes_sets[k]
    x = np.arange(n_classes)
    w = 0.35
    true_cls = y_test[i]
    y_hat_i  = int(y_test_hard[i])
    if proba_test is not None:
        ax.bar(x - w/2, proba_test[i],        width=w, color="#2196F3", alpha=0.8,
               label="A: real proba")
        ax.axhline(prob_threshold_lac, color="#2196F3", lw=1.5, linestyle="--", alpha=0.7,
               label=f"A threshold={prob_threshold_lac:.2f}")        
        ax.axhline(prob_threshold_ncm, color="#FF9800", lw=1.5, linestyle=":",  alpha=0.7,
               label=f"B threshold={prob_threshold_ncm:.2f}")
        
    ax.bar(x + w/2, pseudo_proba_test[i], width=w, color="#FF9800", alpha=0.8,
           label="B: NCM pseudo")
    ax.axvspan(true_cls - 0.5, true_cls + 0.5, alpha=0.1, color="green")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes_original], fontsize=7)
    ax.set_ylim(0, 1.05)

    try:
        in_a = "OK" if covered_lac[i] else "X"
    except Exception:
        in_a = None
    in_b = "OK" if covered_ncm[i] else "X"
    ax.set_title(
        f"True:{int_to_class[true_cls]}  Pred:{int_to_class[y_hat_i]}\n"
        f"A:{in_a}(sz={None if sizes_lac is None else int(sizes_lac[i])})  B:{in_b}(sz={int(sizes_ncm[i])})",
        fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    if k == 0:
        ax.legend(fontsize=5, loc="upper right")
plt.suptitle(
    f"Blue=real proba (A)  Orange=NCM pseudo (B)  (alpha={alpha})\n"
    "Green band=true class.  Dashed/dotted lines=thresholds.  "
    "OK=covered  X=missed.",
    fontsize=9)
plt.tight_layout()
fig_sets.savefig(out_dir / "class_prediction_sets.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_sets)

# ==============================================================================
# S8  NCM quality vs coverage and efficiency
# ==============================================================================
display(Markdown("## S8  NCM quality vs coverage and efficiency"))
display(Markdown(r"""
Which ML method should be used for the NCM model?

- **Coverage is guaranteed regardless of NCM quality.**
  The calibration quantile provides the guarantee for any NCM.
- **Efficiency depends on NCM quality.**
  A better NCM model produces more peaked pseudo-probabilities near the
  predicted class => smaller prediction sets for confident predictions.

We simulate three quality levels using real calibration distances.
"""))

np.random.seed(42)
_true_d_cal  = ordinal_distances_cal.astype(float)
_true_d_test = np.abs(y_test - y_test_hard).astype(float)

_configs = [
    ("Poor  (R2~0.1)",  2.0,  "#e74c3c"),
    ("Medium (R2~0.5)", 0.8,  "#3498db"),
    ("Good  (R2~0.9)",  0.25, "#27ae60"),
]

fig_ncm_c, axes_ncm_c = plt.subplots(2, 3, figsize=(14, 8))
ncm_c_results = []
for col_idx, (label, noise_scale, color) in enumerate(_configs):
    _nd_cal  = np.clip(_true_d_cal  + np.random.normal(0, noise_scale, len(_true_d_cal)),
                       0, n_classes - 1)
    _nd_test = np.clip(_true_d_test + np.random.normal(0, noise_scale, len(_true_d_test)),
                       0, n_classes - 1)

    from sklearn.metrics import r2_score as _r2s
    _r2v = _r2s(_true_d_cal, _nd_cal)

    def _d_to_proba(nd, yh_, n_cls_):
        pp = np.zeros((len(nd), n_cls_))
        for ii in range(len(nd)):
            for jj in range(n_cls_):
                pp[ii, jj] = np.exp(-abs(jj - yh_[ii]) / max(nd[ii], 0.5))
            pp[ii] /= pp[ii].sum()
        return pp

    _pp_cal  = _d_to_proba(_nd_cal,  y_cal_hard,  n_classes)
    _pp_test = _d_to_proba(_nd_test, y_test_hard, n_classes)

    _scores = 1.0 - _pp_cal[np.arange(len(y_cal)), y_cal]
    try:
        _q  = np.quantile(_scores, q_level)
        _sets = (1.0 - _pp_test) <= _q
        _cov = _sets[np.arange(len(y_test)), y_test].mean()
        _sz  = _sets.sum(axis=1).mean()

        ncm_c_results.append({"Model": label, "R2": round(_r2v, 3),
                            "q_hat": round(_q, 4), "Coverage": round(_cov, 3),
                            "Mean_set_size": round(_sz, 3)})
    except Exception:
        pass

    ax_top = axes_ncm_c[0, col_idx]
    ax_top.scatter(_true_d_cal, _nd_cal, alpha=0.35, s=8, color=color)
    _lim = max(_true_d_cal.max(), _nd_cal.max()) + 0.5
    ax_top.plot([0, _lim], [0, _lim], "k--", lw=1)
    ax_top.set_xlabel("|y - y_hat|  (true distance)")
    ax_top.set_ylabel("NCM predicted distance")
    ax_top.set_title(f"{label}\nR2 = {_r2v:.2f}")
    ax_top.grid(True, alpha=0.3)

    ax_bot = axes_ncm_c[1, col_idx]
    ax_bot.hist(_scores, bins=20, density=True, alpha=0.65, color=color)
    try:
        ax_bot.axvline(_q, color="red", lw=2, linestyle="--",
                   label=f"q_hat={_q:.3f}  cov={_cov:.3f}  sz={_sz:.2f}")
        ax_bot.set_title(f"Coverage={_cov:.3f}  Mean set size={_sz:.3f}")        
    except Exception:
        pass
    ax_bot.set_xlabel("LAC score on calibration")
    ax_bot.set_ylabel("Density")
    
    ax_bot.legend(fontsize=7); ax_bot.grid(True, alpha=0.3)
plt.suptitle(
    f"NCM quality vs CP outcome  (alpha={alpha}, target={1-alpha:.0%})\n"
    "Coverage stable; efficiency (set size) improves with better NCM.",
    fontsize=10)
plt.tight_layout()
fig_ncm_c.savefig(out_dir / "class_ncm_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ncm_c)

ncm_c_df = pd.DataFrame(ncm_c_results)
display(ncm_c_df)
display(Markdown("""
**Take-away:**
- Coverage is robust to NCM choice: all quality levels achieve the target.
- Approach A (real probabilities) is best when a calibrated classifier is available.
- Approach B (NCM pseudo-proba) matches real-world VEGA usage and is more
  efficient than any fixed-width method.
"""))


# ==============================================================================
# S10  Exchangeability check (KS test)
# ==============================================================================
display(Markdown("## S9  Exchangeability check (KS test)"))
display(Markdown("""
Calibration and test LAC scores should be exchangeable for the coverage
guarantee to hold.  If the external model behaves differently on the
calibration vs test split, exchangeability may be violated.

KS p > 0.05 -> no evidence of distributional shift -> guarantee holds
KS p < 0.05 -> possible shift -> coverage may deviate from the guarantee
"""))


# --- Nonconformity scores ---
scores_cal = scores_ncm
scores_test = 1.0 - pseudo_proba_test[np.arange(len(y_test)), y_test]

# --- Conformal p-values (classification analogue) ---
# fraction of calibration scores >= test score
p_values = np.array([
    np.mean(scores_cal >= s_i)
    for s_i in scores_test
])


colors = {
    "TRAINING": "#4CAF50",  
    "CALIBRATION":  "#FF9800",
    "TEST":   "#2196F3",   
}

# --- Organize comparisons ---
scores = {
    "CALIBRATION": scores_cal,
    "TEST": scores_test,
    "p-values TEST": p_values,
    "uniform": "uniform"
}

pairs = [
    ("CALIBRATION", "TEST"),
    ("p-values TEST", "uniform")
]

ks_p = {}
fig_ks, axes = plt.subplots(1, len(pairs), figsize=(12, 4))
for ax, (k1, k2) in zip(axes, pairs):
    s1, s2 = scores[k1], scores[k2]

    if k2 == "uniform":
        # compare p-values to U(0,1)
        ks_stat, _ks_p = ks_2samp(s1, np.random.uniform(size=len(s1)))
    else:
        ks_stat, _ks_p = ks_2samp(s1, s2)

    ks_p[f"{k1}_{k2}"] = _ks_p
    msg = f"KS p={_ks_p:.4f} ({'OK' if _ks_p > 0.05 else 'WARNING: possible shift'})"

    # --- Plot ---
    ax.hist(s1, bins="auto", density=True, alpha=0.5,
            label=k1, color=colors.get(k1, "gray"))

    if k2 == "uniform":
        ax.axhline(y=1, linestyle='--',
                   color=colors.get(k2, "red"),
                   label="uniform", linewidth=2)
        ax.set_xlabel("p-values")
    else:
        ax.hist(s2, bins="auto", density=True, alpha=0.5,
                label=k2, color=colors.get(k2, "red"))
        ax.set_xlabel("Nonconformity score")

    ax.set_title(f"{k1} vs {k2}\n{msg}")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    display(Markdown(f"- [{k1} - {k2}] {msg}"))
plt.tight_layout()
fig_ks.savefig(out_dir / "ext_class_exchangeability_ks.png",
               dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_ks)

# ==============================================================================
# S11  Coverage guarantee sweep across alpha
# ==============================================================================
display(Markdown("## S9  Coverage guarantee across alpha levels"))

alphas_sw = np.arange(0.05, 0.51, 0.05)
res_sw = {"A_lac": {"cov": [], "sz": []}, "B_ncm": {"cov": [], "sz": []}}

for a in alphas_sw:
    _ql = min(np.ceil((n_cal + 1) * (1 - a)) / n_cal, 1.0)
    try:
        _q = np.quantile(scores_lac, _ql)
        _s = (1.0 - proba_test) <= _q
        res_sw["A_lac"]["cov"].append(_s[np.arange(len(y_test)), y_test].mean())
        res_sw["A_lac"]["sz"].append(_s.sum(axis=1).mean())
    except Exception:
        pass

    _q = np.quantile(scores_ncm, _ql)
    _s = (1.0 - pseudo_proba_test) <= _q
    res_sw["B_ncm"]["cov"].append(_s[np.arange(len(y_test)), y_test].mean())
    res_sw["B_ncm"]["sz"].append(_s.sum(axis=1).mean())

fig_sw, (ax_sw1, ax_sw2) = plt.subplots(1, 2, figsize=(11, 4))
for key, (style, color, lbl) in {
        "A_lac": ("o-", "#2196F3", "A: LAC"),
        "B_ncm": ("s--", "#FF9800", "B: NCM")}.items():
    try:
        ax_sw1.plot(1 - alphas_sw, res_sw[key]["cov"], style, color=color, label=lbl)
    except:
        pass
    try:
        ax_sw2.plot(1 - alphas_sw, res_sw[key]["sz"],  style, color=color, label=lbl)
    except:
        pass
ax_sw1.plot([0.5, 0.95], [0.5, 0.95], "k:", lw=1, label="y = 1-alpha (ideal)")
ax_sw1.set_xlabel("Target coverage  (1 - alpha)")
ax_sw1.set_ylabel("Empirical coverage")
ax_sw1.set_title("Coverage guarantee across alpha levels")
ax_sw1.legend(fontsize=7); ax_sw1.grid(True, alpha=0.3)
ax_sw2.set_xlabel("Target coverage  (1 - alpha)")
ax_sw2.set_ylabel("Mean prediction set size")
ax_sw2.set_title("Set size vs coverage target")
ax_sw2.legend(fontsize=7); ax_sw2.grid(True, alpha=0.3)
plt.suptitle("Classification CP: coverage and efficiency across alpha", fontsize=11)
plt.tight_layout()
fig_sw.savefig(out_dir / "class_coverage_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_sw)

# ==============================================================================
# S12  Per-class coverage
# ==============================================================================
display(Markdown("## S10  Per-class coverage (marginal vs conditional)"))

per_class_rows = []
for cls_int, cls_name in int_to_class.items():
    mask = y_test == cls_int
    if mask.sum() == 0:
        continue
    per_class_rows.append({
        "Class": cls_name, "n_test": int(mask.sum()),
        "Cov_A": None if covered_lac is None else f"{covered_lac[mask].mean():.3f}",
        "Cov_B": f"{covered_ncm[mask].mean():.3f}",
        "Sz_A":  None if sizes_lac is None else f"{sizes_lac[mask].mean():.2f}",
        "Sz_B":  f"{sizes_ncm[mask].mean():.2f}",
    })
display(pd.DataFrame(per_class_rows))


# ==============================================================================
# S13  AD comparison (only when ad_cols are available)
# ==============================================================================
display(Markdown("## S13  AD comparison: set size / singleton rate vs AD index"))

ad_results = []
if not _available_ad:
    display(Markdown(
        "### Skipped\n\nNo AD columns found in external file. To enable, add to "
        f"`env.tutorial.yaml` under `{dataset}`:\n\n"
        "```yaml\n  ad_cols: [\"ADI\"]\n  ad_col_directions: [\"similarity\"]\n```"))
else:
    display(Markdown(f"- AD columns: {_available_ad}"))
    display(Markdown("""
**Key question**: Do prediction set sizes correlate with AD metrics?

Expected: larger sets (more uncertain) for out-of-AD molecules.
Singleton rate should be higher for in-AD molecules.
"""))
    for ad_col, ad_dir in zip(_available_ad, ad_col_directions[:len(_available_ad)]):
        display(Markdown(f"### {ad_col}  (direction: {ad_dir})"))
        ad_raw = test_df[ad_col].astype(float)
        ad_s   = pd.Series(ad_raw, index=test_df.index)
        mask   = ad_s.notna()
        ad_c   = ad_s[mask].reset_index(drop=True)
        sz_c   = pd.Series(sizes_ncm)[mask].reset_index(drop=True)
        sing_c = (sz_c == 1).astype(float)
        rho_sz, p_sz   = spearmanr(ad_c.values, sz_c.values)
        rho_sg, p_sg   = spearmanr(ad_c.values, sing_c.values)
        expected = "negative" if ad_dir == "similarity" else "positive"
        display(Markdown(f"- Set size Spearman rho={rho_sz:.3f}  p={p_sz:.4f}  (expected {expected})"))
        display(Markdown(f"- Singleton rate Spearman rho={rho_sg:.3f}  p={p_sg:.4f}"))
        threshold = 0.5
        in_ad  = sz_c[ad_c >= threshold] if ad_dir == "similarity" else sz_c[ad_c <= threshold]
        out_ad = sz_c[ad_c <  threshold] if ad_dir == "similarity" else sz_c[ad_c >  threshold]
        fig_ad, axes_ad = plt.subplots(2, 2, figsize=(13, 10))
        axes_ad[0, 0].scatter(ad_c, sz_c, alpha=0.25, s=8, color="#9C27B0", rasterized=True)
        try:
            z = np.polyfit(ad_c, sz_c, 1)
            xr = np.linspace(ad_c.min(), ad_c.max(), 100)
            axes_ad[0, 0].plot(xr, np.poly1d(z)(xr), "r-", lw=2, alpha=0.7, label="trend")
        except Exception:
            pass
        axes_ad[0, 0].set_xlabel(f"{ad_col}"); axes_ad[0, 0].set_ylabel("Set size")
        axes_ad[0, 0].set_title(f"Set size vs {ad_col}\nSpearman rho={rho_sz:.3f}  p={p_sz:.4f}  (expected {expected})", fontsize=9)
        axes_ad[0, 0].legend(fontsize=7); axes_ad[0, 0].grid(True, alpha=0.3)
        if len(in_ad) > 1 and len(out_ad) > 1:
            bp = axes_ad[0, 1].boxplot([in_ad.values, out_ad.values], patch_artist=True,
                                        widths=0.5, medianprops=dict(color="black", lw=2))
            for patch, c_ in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
                patch.set_facecolor(c_)
            u_stat, u_p = mannwhitneyu(in_ad, out_ad, alternative="two-sided")
            axes_ad[0, 1].text(0.98, 0.97, f"Mann-Whitney p={u_p:.4f}",
                               transform=axes_ad[0, 1].transAxes, ha="right", va="top",
                               fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        axes_ad[0, 1].set_xticklabels([f"In-AD (n={len(in_ad)})", f"Out-of-AD (n={len(out_ad)})"])
        axes_ad[0, 1].set_ylabel("Set size")
        axes_ad[0, 1].set_title("Set size: In-AD vs Out-of-AD  (threshold=0.5)")
        axes_ad[0, 1].grid(True, alpha=0.3, axis="y")
        _tmp = pd.DataFrame({"AD": ad_c.values, "sz": sz_c.values, "singleton": sing_c.values})
        _tmp["_bin"] = pd.qcut(ad_c, q=n_quantile_bins, labels=False, duplicates="drop")
        strat_rows = []
        for b in sorted(_tmp["_bin"].dropna().unique()):
            _m = _tmp["_bin"] == b
            strat_rows.append({
                "AD bin":       f"Q{int(b)+1} [{ad_c[_m].min():.2f}-{ad_c[_m].max():.2f}]",
                "n":            int(_m.sum()),
                "Mean set size": _tmp.loc[_m, "sz"].mean(),
                "Singleton %":  _tmp.loc[_m, "singleton"].mean() * 100,
            })
        strat_df = pd.DataFrame(strat_rows)
        display(strat_df)
        x = np.arange(len(strat_df))
        axes_ad[1, 0].bar(x, strat_df["Mean set size"], color="#3498db", alpha=0.8, edgecolor="black")
        axes_ad[1, 0].set_xticks(x)
        axes_ad[1, 0].set_xticklabels(strat_df["AD bin"], rotation=30, ha="right", fontsize=8)
        axes_ad[1, 0].set_ylabel("Mean set size"); axes_ad[1, 0].grid(True, alpha=0.3, axis="y")
        axes_ad[1, 0].set_title("Mean set size by AD quintile")
        ax2 = axes_ad[1, 0].twinx()
        ax2.plot(x, strat_df["Singleton %"] / 100, "rs-", lw=2, ms=8, label="Singleton rate")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("Singleton rate"); ax2.legend(loc="upper left", fontsize=8)
        axes_ad[1, 1].scatter(ad_c, sing_c, alpha=0.25, s=8, color="#FF9800", rasterized=True)
        try:
            z2 = np.polyfit(ad_c, sing_c, 1)
            axes_ad[1, 1].plot(xr, np.poly1d(z2)(xr), "r-", lw=2, alpha=0.7, label="trend")
        except Exception:
            pass
        axes_ad[1, 1].set_xlabel(f"{ad_col}"); axes_ad[1, 1].set_ylabel("Singleton (1=yes)")
        axes_ad[1, 1].set_title(f"Singleton rate vs {ad_col}\nSpearman rho={rho_sg:.3f}  p={p_sg:.4f}")
        axes_ad[1, 1].legend(fontsize=7); axes_ad[1, 1].grid(True, alpha=0.3)
        interp = "Negative rho = larger sets outside AD" if ad_dir == "similarity" \
                 else "Positive rho = larger sets outside AD"
        plt.suptitle(f"{dataset}: set size vs {ad_col}  (rho={rho_sz:.3f})\n{interp}", fontsize=10)
        plt.tight_layout()
        _plot_path = out_dir / f"ext_class_ad_{ad_col}.png"
        fig_ad.savefig(_plot_path, dpi=150, bbox_inches="tight")
        plt.show(); plt.close(fig_ad)
        display(Markdown(f"- Plot: `{_plot_path}`"))
        ad_results.append({
            "dataset": dataset, "ad_col": ad_col, "direction": ad_dir,
            "n": int(mask.sum()),
            "spearman_rho_setsize":   round(rho_sz, 4), "p_value_setsize": round(p_sz, 6),
            "spearman_rho_singleton": round(rho_sg, 4), "p_value_singleton": round(p_sg, 6),
        })


# ==============================================================================
# S14  Save results
# ==============================================================================
result_df = pd.DataFrame({
    "Smiles": test_df["Smiles"].values,
    "True":   test_df[target_col].values,
    "Pred":   [int_to_class[p] for p in y_test_hard],
    "Cov_A":  None if covered_lac is None else covered_lac.astype(int), "SetSize_A": sizes_lac,
    "Cov_B":  covered_ncm.astype(int), "SetSize_B": sizes_ncm,
    **{f"p_{c}":   None if proba_test is None else  proba_test[:, i]        for i, c in enumerate(classes_original)},
    **{f"pseudo_p_{c}": pseudo_proba_test[:, i] for i, c in enumerate(classes_original)},
})

save_dict = {
    "base_model": base_model, "sigma_model": sigma_model, "ncm": ncm,
    "q_hat_lac": q_hat_lac, "q_hat_ncm": q_hat_ncm,
    "classes_original": classes_original,
    "class_to_int": class_to_int, "int_to_class": int_to_class,
    "distance_classes": distance_classes, "alpha": alpha, "meta": meta,
}
with open(product["ncmodel"], "wb") as fh:
    pickle.dump(save_dict, fh)

with pd.ExcelWriter(product["data"], engine="xlsxwriter") as w:
    result_df.to_excel(w,         sheet_name="Predictions",    index=False)
    pd.DataFrame(comparison_rows).to_excel(w, sheet_name="Metrics", index=False)
    ncm_c_df.to_excel(w,          sheet_name="NCM_comparison", index=False)
    pd.DataFrame(per_class_rows).to_excel(w, sheet_name="Per_class", index=False)

display(Markdown("## [OK] Classification tutorial complete."))
display(Markdown(f"- Results : {product['data']}"))
display(Markdown(f"- Model   : {product['ncmodel']}"))
display(Markdown(f"- Plots   : {out_dir}"))
