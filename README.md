![QU-bounds logo](logo.svg)

# QU-bounds: Conformal Prediction for QSAR Models

A model-agnostic open-source pipeline that retrofits any QSAR model with statistically guaranteed uncertainty estimates using adaptive conformal prediction (CP). The framework produces prediction intervals for regression endpoints and prediction sets for classification endpoints at a user-specified nominal confidence level (e.g. 90%), without retraining the underlying model.

---

## Overview

Traditional QSAR Applicability Domain (AD) methods output heuristic similarity scores without formal statistical guarantees. QU-bounds wraps any existing QSAR model as a **calibration layer**: given only the model's predictions and a calibration set, it derives intervals or label sets with guaranteed marginal coverage under the exchangeability assumption.

Key properties:

- **No retraining required.** Only model predictions and a calibration set are needed.
- **Statistically guaranteed coverage.** At confidence level 1−α, at least (1−α)×100% of test predictions will contain the true value, as a mathematical consequence of the rank-based calibration procedure.
- **Model-agnostic.** Works with regression and classification QSAR models from any platform (VEGA, OPERA, OCHEM, etc.) or trained internally.
- **AD-aware.** Auxiliary models trained on molecular fingerprints (ECFP4) serve as nonconformity measures (NCMs); most existing AD indices can play the same role. Conformal efficiency metrics (interval width, singleton rate) correlate strongly with AD indices.
- **Hard-label classifiers supported.** A novel ordinal distance strategy generates pseudo-probabilities compatible with standard conformal methods, extending the approach to classifiers that produce only hard labels.
- **Large-scale ready.** Validated on over 100 VEGA QSAR models spanning physicochemical, toxicity, and environmental endpoints, and demonstrated on the EPA CompTox chemical inventory.

---

## Repository structure

```
qubounds_clean/
├── src/qubounds/               # Core library
│   ├── mapie_regression.py     # Conformal regression: ExternalPredictor, train/predict functions
│   ├── mapie_class_lac.py      # Conformal classification: NCMProbabilisticClassifier, LAC wrapper
│   ├── mapie_class_proba.py    # Classification variant using native predict_proba()
│   ├── mapie_diagnostic.py     # Sigma model factory, exchangeability tests, diagnostics
│   ├── descriptors/            # ECFP4 fingerprint computation with SQLite cache
│   └── vega/                   # VEGA-specific data loaders and utilities
│
├── tasks/                      # Ploomber pipeline tasks (notebooks-as-scripts)
│   ├── tutorial/               # Self-contained tutorial tasks
│   │   ├── load_dataset.py             # Download/load and split a dataset
│   │   ├── mapie_native.py             # Regression CP: internal + external model
│   │   ├── mapie_classification.py     # Classification CP: Approach with A. class probabilities + B. hard classifier
│   │   ├── mapie_class_external.py     # Classification with external hard-label predictions
│   │   ├── compare_variants.py         # Compare adaptive vs plain CP variants
│   │   └── ad_comparison.py            # Correlate CP width / set size with AD indices
│   │
│   ├── mapie.py                        # VEGA regression: train conformal models
│   ├── mapiec.py                       # VEGA classification: train conformal models
│   ├── mapie_apply.py                  # Apply trained CP models to new compound sets
│   ├── mapie_apply_plot.py             # Visualise application results
│   ├── mapie_plot.py                   # Regression summary plots (Spearman vs AD)
│   ├── mapie_plot_class.py             # Classification summary plots
│   ├── mapie_regression_analysis.py    # Aggregate regression analysis
│   ├── mapie_class_analysis.py         # Aggregate classification analysis
│   ├── conformal_regression_summary.py # Cross-model regression summary table
│   ├── conformal_classification_summary.py
│   ├── make_archive.py
│   └── vega/                           # VEGA-specific preparation tasks
│
├── resources/                  # Bundled example datasets
│   ├── BCF_MEYLAN.xlsx         # Bioconcentration factor (VEGA train/test split + ADI)
│   ├── HENRY_OPERA.xlsx        # Henry's law constant
│   ├── ZEBRAFISH_CORAL.xlsx    # Zebrafish developmental toxicity
│   ├── toxicity_against_t_pyriformis.csv   # T. pyriformis IGC50 (OCHEM)
│   ├── ames_levenberg.csv      # Ames mutagenicity (OCHEM)
│   └── vega_models_withclasses.xlsx        # VEGA model registry with endpoint metadata
│
├── pipeline.tutorial.yaml      # Tutorial pipeline definition (Ploomber)
├── pipeline.mapie.yaml         # Full conformal prediction pipeline definition
├── pipeline.preparevega.yaml   # VEGA data preparation pipeline
├── pipeline.preparecomptox.yaml# CompTox data preparation pipeline
├── pipeline.archive.yaml       # Archive/export pipeline
├── env.tutorial.yaml           # Dataset and parameter configuration for pipeline.tutorial.yaml
├── env.yaml                    # Full pipeline configuration (model lists, paths)
└── pyproject.toml
```

---

## Conformal prediction: key concepts

### Regression

| Concept | Definition |
|---|---|
| **Nonconformity score** | `s(x,y) = \|y − ŷ\| / σ̂(x)` (adaptive) or `\|y − ŷ\|` (plain) |
| **σ̂(x)** | Sigma model: predicts local residual magnitude from ECFP4 fingerprints |
| **q̂** | Calibration quantile at level `⌈(n+1)(1−α)⌉/n` |
| **Prediction interval** | `ŷ ± q̂·σ̂(x)` (adaptive) or `ŷ ± q̂` (plain) |
| **Coverage guarantee** | Marginal coverage ≥ 1−α; holds under exchangeability |
| **Efficiency metric** | Mean interval width; narrowed by a better sigma model |

The adaptive variant achieves molecule-specific interval widths: wider for structurally novel compounds, narrower for well-represented chemical space.

![FATHEAD_EPA](regression_demo.png "VEGA QSAR model for Fathead Minnow LC50 96h (from EPA T.E.S.T software) FATHEAD_EPA")

### Classification

Two approaches are implemented and compared in the tutorial:

**Approach A — LAC with model probabilities.** Requires `predict_proba()` output. Conformity score: `s = 1 − p̂(y_true | x)`. Gold standard when calibrated probabilities are available.

**Approach B — NCM pseudo-probabilities (hard labels).** For classifiers that produce only a hard label (the standard case for VEGA models). An auxiliary NCM classifier learns `P(|y − ŷ| = d | x)` for ordinal distances `d = 0, 1, …`. These are converted to class pseudo-probabilities:

```
P_pseudo(class=j | x, ŷ) = P(distance = |j − ŷ| | x)
```

LAC is then applied to the pseudo-probabilities. **Coverage is guaranteed regardless of NCM quality**; NCM quality determines efficiency (singleton rate and mean set size).

| Concept | Definition |
|---|---|
| **Prediction set** | `C(x) = {y : p̂(y\|x) ≥ 1 − q̂}` |
| **Coverage guarantee** | Marginal coverage ≥ 1−α |
| **Efficiency metric** | Mean set size; singleton rate (fraction of size-1 sets) |

![FISH_IRFMN](classification_demo.png "VEGA QSAR classification model for fish acute (LC50) toxicity FISH_IRFMN")

### Toxicity Classes

| Class | Description |
|---|---|
| 1 | Toxic-1 (< 1 mg/L) |
| 2 | Toxic-2 (1–10 mg/L) |
| 3 | Toxic-3 (10–100 mg/L) |
| 4 | Non-Toxic (> 100 mg/L) |


---

## Nonconformity measures (NCM codes)

The sigma/NCM model is specified by a code string in the pipeline configuration:

| Code | Model | Task |
|---|---|---|
| `rlgbmecfp` | LightGBM regressor on ECFP4 | Regression sigma |
| `rfecfp` | Random Forest regressor on ECFP4 | Regression sigma |
| `knn2ecfp` | k-NN regressor on ECFP4 | Regression sigma |
| `crfecfp` | Random Forest classifier (ordinal distances) | Classification NCM |
| `cgbecfp` | Gradient Boosting classifier (ordinal distances) | Classification NCM |
| `cknn2jecfp` | k-NN classifier with Jaccard (ordinal distances) | Classification NCM |
| `gbecfp` | Gradient Boosting regressor on ECFP4 | Classification NCM (regression mode) |

---

## Installation

Requires Python ≥ 3.12. Dependencies are managed with UV.

```bash
git clone <repository_url>
cd qubounds
```

Key dependencies: `mapie`, `scikit-learn`, `lightgbm`, `catboost`, `mord`, `rdkit`, `ploomber`, `pandas`, `plotly`, `statsmodels`.

---

## Quick start: tutorial pipeline

The tutorial pipeline demonstrates the full CP workflow on public and bundled datasets. It is the recommended as an entry point for new readers.

### Run

```bash
ploomber build --entry-point pipeline.tutorial.yaml --env-file env.tutorial.yaml
```

Output notebooks and Excel files are written to `products/tutorial/`.

### Tutorial datasets (`env.tutorial.yaml`)

| Key | Dataset | Task | Source |
|---|---|---|---|
| `esol` | ESOL aqueous solubility | Regression | DeepChem / Delaney |
| `lipo` | Lipophilicity | Regression | DeepChem |
| `tpyriformis` | *T. pyriformis* IGC50 | Regression + AD comparison | OCHEM (bundled) |
| `bcfmeylan` | BCF (Meylan) | Regression + AD comparison | VEGA (bundled) |
| `zebrafish_coral` | Zebrafish AC50 | Regression + AD comparison | VEGA (bundled) |
| `henry_opera` | Henry's law constant | Regression + AD comparison | VEGA (bundled) |
| `bbbp` | Blood-brain barrier permeability | Classification | DeepChem |
| `clintox` | Clinical toxicity (FDA approval) | Classification | DeepChem |
| `ames_ochem` | Ames mutagenicity | Classification + AD comparison | OCHEM (bundled) |

### Tutorial task descriptions

**Regression datasets** pass through four tasks:

1. **`tutorial_load_regr_<dataset>`** — reads or downloads the dataset, applies train/test splitting (using a `split_col` if provided, otherwise random 80/20), writes `Training` and `Test` sheets to Excel and a JSON metadata file.

2. **`tutorial_native_<dataset>_<ncm>`** — fits an internal LightGBM base model + sigma model and runs split conformal regression (both adaptive and plain variants). Produces:
   - sigma model diagnostics (R², scatter vs true residuals)
   - calibration nonconformity score histograms and q̂ annotation
   - prediction interval plots on sorted test set
   - exchangeability KS test (calibration vs test NCM scores; p-value uniformity)
   - coverage guarantee sweep across α levels
   - NCM quality comparison (simulated poor/medium/good sigma models)
   - AD correlation analysis (Spearman ρ, boxplots, stratified quintile tables) when `ad_cols` is configured

3. **`tutorial_external_<dataset>_<ncm>`** — identical workflow but reads predictions from file, simulating an external QSAR tool (VEGA, OCHEM, OPERA). AD comparison is embedded here because AD indices come from the same prediction file.

4. **`tutorial_compare_<dataset>_<ncm>`** — side-by-side comparison of native vs external variants: coverage, mean width, width distribution, per-molecule width reduction.

**Classification datasets** pass through analogous tasks:

1. **`tutorial_load_class_<dataset>`** — as above, with class-balance reporting.

2. **`tutorial_classify_<dataset>_<ncm>`** — fits LightGBM + Platt calibration (Approach A) and NCM ordinal classifier (Approach B). Produces:
   - LAC score distributions per class
   - pseudo-probability vs real probability comparison for example molecules
   - prediction set visualisations (12 example molecules)
   - head-to-head A vs B: coverage, mean set size, singleton rate
   - NCM quality comparison
   - exchangeability KS test
   - coverage guarantee sweep across α levels
   - per-class coverage table
   - AD correlation analysis (set size and singleton rate vs AD indices) when configured

3. **`tutorial_classify_external_<dataset>_<ncm>`** — external hard-label variant (Approach B only).

### Adding your own dataset

Edit `env.tutorial.yaml`. Minimum required fields:

```yaml
tutorial_regr_datasets:
  mymodel:
    path: "path/to/predictions.xlsx"
    target_col: "Exp value"
    smiles_col: "Smiles"
```

To add external predictions and AD comparison:

```yaml
    pred_col: "Predicted value"        # column with external model predictions
    split_col: "Status"                # column distinguishing train/test rows
    split_train_value: "TRAINING"
    split_test_value: ["TEST"]
    ad_cols: ["ADI", "Accuracy index"] # AD metric columns in the same file
    ad_col_directions: ["similarity", "similarity"]  # or "distance"
    n_quantile_bins: 5
    software: "VEGA"
```

`ad_col_directions: similarity` means higher AD value = more reliable (e.g. ADI); `distance` is the reverse (e.g. leverage).

---

## Full Conformal prdiction with external QSAR models pipeline

The full pipeline (`pipeline.mapie.yaml` + `env.yaml`) trains conformal models for all VEGA endpoints and applies them to an external compound inventory (e.g. EPA CompTox).
- The pipeline reads VEGA prediction form files ; does not run VEGA itself. 
- Therefore it is applicable to arbitrary QSAR models, provided the input files are in the required format.
- Look at pipeline.prepare*.yaml for examples of conversion

### Prepare VEGA input files (optional)

- The data files can be downloaded from Zenodo  https://doi.org/10.5281/zenodo.18444068.
- The pipeline is here for completeness.

Copy `env.preparevega.example.yaml` to `env.preparevega.yaml` and configure:

```yaml
VEGA_TRAINING_SETS: "path/to/vega/datasets"     # VEGA-exported training set files
VEGA_RESULT_FILES:  "path/to/vega/results"      # VEGA prediction output files
VEGA_QUBOUNDS_INPUT: "products/qubounds_input"  # staging area for pipeline input
```

Then:

```bash
ploomber build --entry-point pipeline.preparevega.yaml --env-file env.preparevega.yaml
```

### Configure and run full pipeline

In `env.yaml`, set the compound inventory paths under `config`:

```yaml
config:
  comptox:
    VEGA_REPORTS_INPUT: "path/to/vega/epacomptox"
    SMILES_INPUT: "path/to/epacomptox.txt"
    alpha: 0.1
    id: "DTXSID"
```

Then:

```bash
ploomber build --entry-point pipeline.mapie.yaml --env-file env.yaml
```

### VEGA models covered

`env.yaml` lists 52 regression and 53 classification VEGA endpoints, including:

- **Regression:** BCF (multiple models), LogP, melting point, water solubility, aquatic toxicity (algae, Daphnia, fish), carcinogenicity potency, Henry's law, KOC, KOA, LD50, LOAEL, NOAEL, vapour pressure, persistence (air/soil/water/sediment), zebrafish AC50.
- **Classification:** aromatase, Ames mutagenicity, carcinogenicity, developmental toxicity, eye/skin irritation and sensitisation, estrogen/androgen/glucocorticoid/thyroid receptor activity, hepatotoxicity (NRF2, PPAR-α/γ, PXR), micronucleus (in vitro/in vivo), P-glycoprotein, persistence (sediment/soil/water), TPO.

---

## Exchangeability and coverage diagnostics

The coverage guarantee holds under the **exchangeability assumption**: calibration and test compounds are drawn from the same input space. When this is violated (structural extrapolation), coverage may fall below the nominal level. The pipeline signals this through:

- widening prediction intervals or shrinking singleton rates for out-of-domain compounds,
- Kolmogorov-Smirnov tests comparing calibration and test nonconformity score distributions,
- conformal p-value uniformity tests (p-values should be uniform under exchangeability).

These diagnostics are computed automatically in all tutorial and full pipeline tasks.

regression_exchangeability.png
![FATHEAD_EPA_exchtest](regression_exchangeability.png "Exchangeability tests for VEGA QSAR model for Fathead Minnow LC50 96h (from EPA T.E.S.T software) FATHEAD_EPA")

![FISH_IRFMN_exchtest](classification_exchangeability.png "Exchangeability tests for VEGA QSAR classification model for fish acute (LC50) toxicity FISH_IRFMN")

![FISH_IRFMN_exchtestc](classification_exchangeability_classwise.png "Classwise Exchangeability tests for VEGA QSAR classification model for fish acute (LC50) toxicity FISH_IRFMN")

---

## Pipeline outputs

All tasks write to `products/`:

| Path | Contents |
|---|---|
| `products/tutorial/<task>/` | Per-dataset notebooks (`.ipynb`), Excel with predictions and metrics, PNG diagnostic plots |
| `products/qubounds_output/mapie/vega/regression/<ncm>/<model>/` | Trained conformal models (`.pkl`); per-model predictions and metrics (`.xlsx`) |
| `products/qubounds_output/mapie/vega/regression/summary.xlsx` | Cross-model summary: coverage, mean width, Spearman ρ vs AD indices |
| `products/qubounds_output/mapie/vega/class_lac/<ncm>/<model>/` | Classification conformal models and per-model results |
| `products/qubounds_output/mapie/vega/class_lac/summary.xlsx` | Cross-model classification summary |
| `products/comptox/<input_key>/regression/` | CP results applied to compound inventory (regression) |
| `products/comptox/<input_key>/class_lac/` | CP results applied to compound inventory (classification) |

Each per-prediction Excel file includes: SMILES, true value (if available), model prediction, lower/upper interval bounds (regression) or per-class pseudo-probabilities and prediction sets (classification), interval width or set size, and coverage indicators.

---

## Acknowledgement

🇪🇺 This project has received funding from the European Union's Horizon Europe research and innovation program under grant agreements [101092164](https://cordis.europa.eu/project/id/101092164) and [101130073](https://cordis.europa.eu/project/id/101130073).
