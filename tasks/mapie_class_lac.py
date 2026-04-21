from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import LACConformityScore, APSConformityScore
from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_diagnostic import (
    sigma_diagnostics, make_sigma_model,
    plot_normalized_ordinal_distances,
    plot_ncm_diagnostics, compute_ordinal_sigma
)
from tasks.mapie_class_proba import (
    train_conformal_classifier_proba, predict_conformal_classifier_proba
)
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)

"""
NCM-Based Pseudo-Probabilistic Conformal Prediction
This approach bridges the gap between hard-prediction classifiers and probability-based conformal methods.
Since the external toxicity classifier only provides hard class predictions (e.g., "toxicity = 2") without
confidence scores, we cannot directly use standard MAPIE conformity scores like LAC (Least Ambiguous Classifier),
which require predict_proba(). To solve this, we train a separate Nonconformity Measure (NCM) model that learns
to predict the probability distribution over ordinal distances: P(distance = 0, 1, 2, 3 | molecule). During prediction,
we convert these distance probabilities into pseudo-class probabilities using the relationship: 
    P(class = j | molecule, ŷ) = P(distance = |j - ŷ| | molecule). 
For example, if the external model predicts class 1 and the NCM outputs [0.6, 0.3, 0.08, 0.02] for distances [0,1,2,3],
we assign P(class=0) = 0.3, P(class=1) = 0.6, P(class=2) = 0.3, P(class=3) = 0.08. These synthetic probabilities encode
both the hard prediction (highest probability at predicted class) and uncertainty (spread reflects NCM's confidence).
We then feed these pseudo-probabilities into MAPIE's standard LAC conformity score, which computes conformity as the 
difference between top predicted probability and each class's probability. This allows us to leverage MAPIE's proven 
conformal framework while working with hard predictions, combining the ordinal structure awareness of our NCM with the
rigorous coverage guarantees of split conformal prediction.
"""

# -------------------------------------------
# NCM-Based Probabilistic Classifier
# -------------------------------------------
class NCMProbabilisticClassifier(ClassifierMixin):
    """
    Converts hard predictions + NCM distance probabilities 
    into class probabilities for use with standard MAPIE scores.
    
    For each sample:
      - Hard prediction: ŷ
      - NCM gives: P(distance = 0, 1, 2, 3 | x)
      - Convert to: P(class = j | x) = P(distance = |j - ŷ| | x)
    """
    def __init__(self, y_pred, ncm_model, ncm_type, classes=None):
        self.y_pred = np.asarray(y_pred)
        self.ncm_model = ncm_model
        self.ncm_type = ncm_type
        
        if classes is None:
            classes = np.unique(self.y_pred)
        self.classes_ = np.asarray(classes)
        self.n_classes_ = len(self.classes_)

    def get_params(self, deep=True):
        return {
            'y_pred': self.y_pred,
            'ncm_model': self.ncm_model,
            'ncm_type': self.ncm_type,
            'classes': self.classes_
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.y_pred[:len(X)]

    def predict_proba(self, X):
        """
        Convert NCM distance probabilities to class probabilities.
        
        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            P(class = j | x) based on NCM distance predictions
        """
        n_samples = len(X)
        
        # Get NCM distance probabilities
        if self.ncm_type.startswith("c") or self.ncm_type.startswith("o"):
            # Classifier NCM - returns P(distance = k)
            distance_probs = self.ncm_model.predict_proba(X)
            distance_classes = self.ncm_model.classes_
        else:
            # Regressor NCM - create pseudo-probabilities centered at predicted distance
            # (Less principled, but provides some uncertainty)
            predicted_distances = self.ncm_model.predict(X)
            max_distance = int(self.n_classes_ - 1)
            distance_classes = np.arange(max_distance + 1)
            
            # Gaussian-like pseudo-probabilities around predicted distance
            distance_probs = np.zeros((n_samples, len(distance_classes)))
            for i in range(n_samples):
                for d in distance_classes:
                    # Simple exponential decay: higher prob for closer distances
                    distance_probs[i, d] = np.exp(-abs(d - predicted_distances[i]))
                # Normalize
                distance_probs[i] /= distance_probs[i].sum()
        
        # Convert distance probabilities to class probabilities
        class_probs = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            y_pred_i = self.y_pred[i]
            
            # For each possible class j, find its distance from predicted class
            for j, cls in enumerate(self.classes_):
                distance = abs(cls - y_pred_i)
                
                # Find probability for this distance in NCM output
                if distance < len(distance_classes):
                    distance_idx = np.where(distance_classes == distance)[0]
                    if len(distance_idx) > 0:
                        class_probs[i, j] = distance_probs[i, distance_idx[0]]
                    else:
                        # Distance not in NCM classes, use small probability
                        class_probs[i, j] = 1e-6
                else:
                    class_probs[i, j] = 1e-6
            
            # Normalize (should already sum to 1, but ensure it)
            if class_probs[i].sum() > 0:
                class_probs[i] /= class_probs[i].sum()
            else:
                # Fallback: uniform distribution
                class_probs[i] = 1.0 / self.n_classes_
        
        return class_probs


def train_conformal_classifier(
        df_train, 
        experimental_tag,
        predicted_tag,
        df_calibration, 
        cache_path, 
        alpha, 
        output_model_path,
        class_order=None, 
        ncm="crfecfp",
        method_score="LAC"):
    if method_score.endswith("_proba"):
        conformity_score = method_score.replace("_proba", "").lower()
        logger.info(f"method_score {method_score} conformity_score {conformity_score}")
        return train_conformal_classifier_proba(
            df_calibration=df_calibration,
            experimental_tag=experimental_tag,
            prob_columns=predicted_tag,
            cache_path=cache_path,
            alpha=alpha,
            output_model_path=output_model_path)
    else:
        return train_conformal_classifier_hard(
            df_train,
            experimental_tag,
            predicted_tag,
            df_calibration,
            cache_path,
            alpha, 
            output_model_path,
            class_order=None,
            ncm=ncm,
            method_score=method_score)
    

def train_conformal_classifier_hard(
        df_train, 
        experimental_tag,
        predicted_tag,
        df_calibration, 
        cache_path, 
        alpha, 
        output_model_path,
        class_order=None, 
        ncm="crfecfp",
        method_score="LAC"):
    """
    Train conformal classifier using NCM-based pseudo-probabilities + MAPIE LAC.
    
    Key idea: Convert NCM distance predictions to class probabilities,
    then use standard MAPIE LAC score for conformal prediction.
    """
    smiles_cal = df_calibration["Smiles"].values
    y_cal_original = df_calibration[experimental_tag].values
    y_pred_cal_original = df_calibration[predicted_tag].values

    smiles_train = df_train["Smiles"].values
    y_train_original = df_train[experimental_tag].values
    y_pred_train_original = df_train[predicted_tag].values

    # Get unique classes and create mapping
    classes_in_data = np.unique(
        np.concatenate([y_train_original, y_pred_train_original, 
                       y_pred_cal_original, y_cal_original]))
    
    if class_order is not None:
        classes_original = np.array(class_order)
        missing_classes = set(classes_in_data) - set(classes_original)
        if missing_classes:
            raise ValueError(f"Classes in data but not in class_order: {missing_classes}")
    else:
        classes_original = np.sort(classes_in_data)
    
    logger.info(f"Classes: {classes_original}")
    
    # Create class mapping
    class_to_mapped = {cls: i for i, cls in enumerate(classes_original)}
    mapped_to_class = {i: cls for cls, i in class_to_mapped.items()}
    
    # Remap to contiguous integers
    y_cal = np.array([class_to_mapped[c] for c in y_cal_original])
    y_train = np.array([class_to_mapped[c] for c in y_train_original])
    y_pred_cal = np.array([class_to_mapped[c] for c in y_pred_cal_original])
    y_pred_train = np.array([class_to_mapped[c] for c in y_pred_train_original])
    classes = np.arange(len(classes_original))
    
    # Init fingerprint cache
    init_cache(cache_path)
    
    logger.info("Computing ECFP...")
    X_ecfp_cal = np.array([smiles_to_ecfp_cached(sm) for sm in smiles_cal])
    X_ecfp_train = np.array([smiles_to_ecfp_cached(sm) for sm in smiles_train])

    # -------------------------------------------
    # Train NCM Model (CRITICAL - used for pseudo-probabilities)
    # -------------------------------------------
    ordinal_distances_train = np.abs(y_train - y_pred_train).astype(float)
    ordinal_distances_cal = np.abs(y_cal - y_pred_cal).astype(float)
    
    logger.info(f"Training NCM model (used for pseudo-probabilities)...")
    
    unique_distances = np.unique(ordinal_distances_train.astype(int))
    if len(unique_distances) < 2:
        logger.error("Cannot train: only 1 unique distance value")
        save_dict = {
            "method": "NCM_LAC",
            "ncm": ncm,
            "mapie": None,
            "sigma_model": None,
            "classes": classes,
            "classes_original": classes_original,
            "class_to_mapped": class_to_mapped,
            "mapped_to_class": mapped_to_class,
            "alpha": alpha,
            "is_binary": False,
            "sigma_diagnostics": None,
            "sigma_r2": None,
            "sigma_rmse": None,
            "sigma_mae": None,
            "error": "Cannot train: only 1 unique distance value"
        }
        with open(output_model_path, "wb") as f:
            pickle.dump(save_dict, f)        
        return None
    
    sigma_model = make_sigma_model(ncm)
    sigma_model.fit(X_ecfp_train, ordinal_distances_train.astype(int))
    
    # Get NCM predictions for diagnostics
    if ncm.startswith("c") or ncm.startswith("o"):
        probs_train = sigma_model.predict_proba(X_ecfp_train)
        probs_cal = sigma_model.predict_proba(X_ecfp_cal)
        #distance_classes = sigma_model.classes_
        sigma_pred_train = np.argmax(probs_train, axis=1)  # predicted distance class
        sigma_pred_cal = np.argmax(probs_cal, axis=1)
    else:
        probs_train = None
        probs_cal = None
        sigma_pred_train = sigma_model.predict(X_ecfp_train)
        sigma_pred_cal = sigma_model.predict(X_ecfp_cal)
    
    # Diagnostics
    diag_sigma = sigma_diagnostics(ordinal_distances_train, sigma_pred_train)
    logger.info(f"NCM diagnostics (training): R²={diag_sigma['r2']:.3f}, "
                f"RMSE={diag_sigma['rmse']:.3f}, MAE={diag_sigma['mae']:.3f}")
    
    diag_sigma_cal = sigma_diagnostics(ordinal_distances_cal, sigma_pred_cal)
    logger.info(f"NCM diagnostics (calibration): R²={diag_sigma_cal['r2']:.3f}, "
                f"RMSE={diag_sigma_cal['rmse']:.3f}, MAE={diag_sigma_cal['mae']:.3f}")    
    # Plot diagnostics
    try:
        fig = plot_ncm_diagnostics(ordinal_distances_cal, sigma_pred_cal,
                             None if probs_train is None else np.max(probs_train, axis=1),
                             None if probs_cal is None else np.max(probs_cal, axis=1),
                            title=f"NCM Model [{ncm}]")
        fig.savefig(output_model_path.replace('.pkl', f'{ncm}_ncm_diag.png'), dpi=150)
        plt.close()
    except Exception as x:
        print(x)
        pass

    # -------------------------------------------
    # Create NCM-Based Probabilistic Classifier
    # -------------------------------------------
    logger.info("Creating NCM-based pseudo-probabilistic classifier...")
    
    estimator = NCMProbabilisticClassifier(
        y_pred=y_pred_cal,
        ncm_model=sigma_model,
        ncm_type=ncm,
        classes=classes
    )
    
    # Use standard MAPIE LAC score
    if method_score == "LAC":
        conformity_score = LACConformityScore()
    else:
        conformity_score = APSConformityScore()        
    
    mapie = SplitConformalClassifier(
        estimator=estimator,
        conformity_score=conformity_score,
        prefit=True,
        confidence_level=1 - alpha
    )
    
    logger.info(f"Calibrating with MAPIE {method_score} (alpha={alpha})...")
    mapie.estimator_ = estimator.fit(None, None)
    mapie.conformalize(X_conformalize=X_ecfp_cal, y_conformalize=y_cal)
    
    # -------------------------------------------
    # Validation
    # -------------------------------------------
    y_pred_sets = mapie.predict(X_ecfp_cal)
    
    if hasattr(y_pred_sets, 'shape') and len(y_pred_sets.shape) == 2:
        coverage = y_pred_sets[np.arange(len(y_cal)), y_cal].mean()
        avg_size = y_pred_sets.sum(axis=1).mean()
    else:
        pred_sets = [list(s) if isinstance(s, (list, set, np.ndarray)) else [s] 
                    for s in y_pred_sets]
        coverage = np.mean([y_cal[i] in pred_sets[i] for i in range(len(y_cal))])
        avg_size = np.mean([len(s) for s in pred_sets])
    
    logger.info(f"Calibration coverage: {coverage:.2%} (target: {1-alpha:.2%})")
    logger.info(f"Avg set size: {avg_size:.2f}")
    
    # -------------------------------------------
    # Save Model
    # -------------------------------------------
    save_dict = {
        "method": f"NCM_{method_score}",
        "ncm": ncm,
        "mapie": mapie,
        "sigma_model": sigma_model,
        "classes": classes,
        "classes_original": classes_original,
        "class_to_mapped": class_to_mapped,
        "mapped_to_class": mapped_to_class,
        "alpha": alpha,
        "is_binary": False,
        "sigma_diagnostics": diag_sigma,
        "sigma_r2": diag_sigma["r2"],
        "sigma_rmse": diag_sigma["rmse"],
        "sigma_mae": diag_sigma["mae"],
        "sigma_diagnostics_cal": diag_sigma_cal,  # full diagnostics dict
        "sigma_r2_cal": diag_sigma_cal["r2"],
        "sigma_rmse_cal": diag_sigma_cal["rmse"],
        "sigma_mae_cal": diag_sigma_cal["mae"]        
    }
    
    with open(output_model_path, "wb") as f:
        pickle.dump(save_dict, f)
    
    logger.info(f"Model saved to: {output_model_path}")
    return save_dict


def predict_conformal_classifier_chunked(
    df,
    pred_column,
    true_column=None,
    model_path=None,
    tag="Exp",
    smiles_column="Smiles",
    chunk_size=10000,
    split="Test"):
    if not Path(model_path).exists():
        return None, {}, {}

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    if saved["sigma_model"] is None:
        return None, {}, {}
    
    prob_columns = saved.get("prob_columns", None)
    if prob_columns is None:
        return predict_conformal_classifier_hard_chunked(
            df=df,
            pred_column=pred_column,
            true_column=true_column,
            saved_model=saved,
            tag=tag,
            smiles_column=smiles_column,
            chunk_size=chunk_size,
            split=split
        )
    else:
        return predict_conformal_classifier_proba(
            df=df,
            prob_columns=pred_column,
            true_column=true_column,
            saved_model=saved,
            tag=tag,
            chunk_size=chunk_size,
            split=split
            )


def predict_conformal_classifier_hard_chunked(
    df,
    pred_column,
    true_column=None,
    saved_model=None,
    tag="Exp",
    smiles_column="Smiles",
    chunk_size=10000,
    split="Test",
):
    """
    Chunked conformal classification with MAPIE (LAC / APS) + NCM diagnostics.
    """

    method = saved_model["method"]
    sigma_model = saved_model["sigma_model"]
    ncm = saved_model["ncm"]
    classes = saved_model["classes"]
    classes_original = saved_model["classes_original"]
    class_to_mapped = saved_model["class_to_mapped"]
    mapped_to_class = saved_model["mapped_to_class"]
    alpha = saved_model["alpha"]
    mapie = saved_model["mapie"]

    has_true = (
        true_column is not None
        and true_column in df.columns
        and df[true_column].isin(class_to_mapped).any()
    )
    
    logger.info(f"{has_true} {df.columns}")
    # ---- global validation ----
    y_pred_all_original = df[pred_column].values
    unknown = set(y_pred_all_original) - set(class_to_mapped.keys())
    if unknown:
        raise ValueError(f"Unknown classes in test set: {sorted(unknown)}")

    logger.info(
        f"{tag}\tProcessing {len(df)} samples in chunks of {chunk_size} "
        f"(method={method}, alpha={alpha})"
    )

    # ---- accumulators ----
    results_chunks = []

    all_set_sizes = []
    all_coverage = []
    all_correct = []
    all_distances = []

    num_chunks = int(np.ceil(len(df) / chunk_size))
    # Extract IDs if available
    has_ids = "ID" in df.columns    
    has_ADI = "ADI" in df.columns    

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df))

        logger.info(f"{tag}\tChunk {i+1}/{num_chunks} rows {start}-{end}")

        df_chunk = df.iloc[start:end].copy()

        ids_chunk = df_chunk["ID"].values if has_ids else None
        adi_chunk = df_chunk["ADI"].values if has_ADI else None
        smiles = df_chunk[smiles_column].values
        y_pred_orig = df_chunk[pred_column].values
        y_pred = np.array([class_to_mapped[c] for c in y_pred_orig])

        y_true_orig = (
            df_chunk[true_column].values if has_true else None
        )

        # ---- ECFP ----
        X_ecfp = np.array([smiles_to_ecfp_cached(sm) for sm in smiles])

        # ---- NCM scores (diagnostics) ----
        if ncm.startswith(("c", "o")):
            probs = sigma_model.predict_proba(X_ecfp)
            distance_classes = sigma_model.classes_
            
            # Original: max probability across all distances
            ncm_probs_max = np.max(probs, axis=1)
            
            # New: P(distance=0) - probability of correct prediction
            if 0 in distance_classes:
                idx_zero = np.where(distance_classes == 0)[0][0]
                ncm_probs_zero = probs[:, idx_zero]
            else:
                ncm_probs_zero = np.zeros(len(probs))
                logger.warning(f"Distance class 0 not found in NCM model for chunk {i+1}")
            
            # Also keep argmax for reference (most likely distance)
            ncm_distance_mode = np.argmax(probs, axis=1)
            
        else:
            # Regression NCM
            ncm_distance_mode = sigma_model.predict(X_ecfp)
            ncm_probs_max = None
            ncm_probs_zero = None

        # ---- MAPIE prediction sets ----
        mapie.estimator_.y_pred = y_pred
        y_pred_sets = mapie.predict(X_ecfp)

        # ---- decode prediction sets ----
        if hasattr(y_pred_sets, "shape") and y_pred_sets.ndim == 2:
            set_sizes = y_pred_sets.sum(axis=1)
            pred_sets = [
                classes[row.astype(bool)].tolist()
                for row in y_pred_sets
            ]
        else:
            pred_sets = [
                list(s) if isinstance(s, (list, set, np.ndarray)) else [s]
                for s in y_pred_sets
            ]
            set_sizes = np.array([len(s) for s in pred_sets])

        pred_sets_original = [
            [mapped_to_class[c] for c in s] for s in pred_sets
        ]

        all_set_sizes.extend(set_sizes)

        # ---- metrics per chunk ----
        if has_true:
            n = len(y_true_orig)

            # valid labels = mappable to numeric
            valid_mask = np.array([
                c in class_to_mapped
                for c in y_true_orig
            ])

            # initialize full-length outputs
            coverage_full = np.full(n, np.nan, dtype=float)
            correct_full = np.full(n, np.nan, dtype=float)
            distances_full = np.full(n, np.nan, dtype=float)

            if valid_mask.any():
                idx = np.where(valid_mask)[0]

                # numeric mapping
                y_true_valid = np.array([class_to_mapped[y_true_orig[i]] for i in idx])
                y_pred_valid = y_pred[idx]
                y_pred_orig_valid = y_pred_orig[idx]

                # coverage
                coverage_valid = np.array([
                    y_true_orig[i] in pred_sets_original[i]
                    for i in idx
                ], dtype=float)

                coverage_full[idx] = coverage_valid

                # correctness
                correct_full[idx] = (y_pred_orig_valid == y_true_orig[idx])

                # distances
                distances_full[idx] = np.abs(y_true_valid - y_pred_valid)

            all_coverage.extend(coverage_full)
            all_correct.extend(correct_full)
            all_distances.extend(distances_full)

        # ---- build result frame ----
        chunk_result = {
            "ID" : ids_chunk,
            "ADI" : adi_chunk,
            f"{tag}_true": y_true_orig,
            f"{tag}_pred": y_pred_orig,
            f"{tag}_predicted_distance": ncm_distance_mode,  # Most likely distance (argmax)
            f"{tag}_probs_distance": ncm_probs_max,          # Max probability (original)
            f"{tag}_probs_zero_distance": ncm_probs_zero,    # P(distance=0) - NEW
            "Set_Size": set_sizes,
            "Prediction_Set": [
                str([int(c) if c == int(c) else float(c) for c in s])
                for s in pred_sets_original
            ],
            "In_Coverage": coverage_full if has_true else None,
            "Smiles": smiles,
        }

        if has_true:
            chunk_result[f"{tag}_distance"] = distances_full

        for cls in classes_original:
            chunk_result[f"in_set_class_{cls}"] = [
                cls in s for s in pred_sets_original
            ]

        results_chunks.append(pd.DataFrame(chunk_result))

    # ---- concatenate ----
    results_df = pd.concat(results_chunks, ignore_index=True)

    # ---- final metrics ----
    correct_arr = np.array(all_correct)
    mask_correct = np.isfinite(correct_arr) & (correct_arr.astype(bool))

    metrics = {
        "average_set_size": np.mean(all_set_sizes),
        "Empirical Coverage": np.mean(all_coverage) if has_true else None,
        "Point Accuracy": np.mean(all_correct) if has_true else None,
        "Off-by-One Accuracy": (
            np.mean(np.array(all_distances) <= 1)
            if has_true else None
        ),
        "Mean Ordinal Distance": (
            np.mean(all_distances) if has_true else None
        ),
        "Singleton Efficiency": (
            np.mean(np.array(all_set_sizes)[mask_correct] == 1)
            if has_true and mask_correct.any()
            else None
        ),
        "Alpha": alpha,
        "Target Coverage": 1 - alpha,
        "Method": method,
        "ncm": ncm,
        "n_samples": len(df),
        "n_chunks": num_chunks,
        "Split" : split
    }

    logger.info(
        f"{tag}\tCompleted {len(df)} samples | "
        f"avg set size={metrics['average_set_size']:.3f}, "
        f"coverage={metrics['Empirical Coverage']}"
    )

    return results_df, metrics, saved_model


