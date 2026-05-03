from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import BaseClassificationScore
from qubounds.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from qubounds.mapie_diagnostic import (
    sigma_diagnostics, make_sigma_model,
    plot_ncm_diagnostics, compute_ordinal_sigma
)
from sklearn.base import ClassifierMixin
from mapie.conformity_scores import LACConformityScore, APSConformityScore
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)
EPS = 1e-6


class ExternalProbabilisticClassifier(ClassifierMixin):
    """
    Returns externally provided probabilities; X is ignored.
    Works with MAPIE as a drop-in estimator.
    """
    
    def __init__(self, y_pred_proba, classes):
        self.y_pred_proba = np.asarray(y_pred_proba)
        self.classes_ = np.asarray(classes)
        self.n_classes_ = len(classes)

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.y_pred_proba[:len(X)], axis=1)]

    def predict_proba(self, X):
        return self.y_pred_proba[:len(X)]


def train_conformal_classifier_proba(
    df_calibration,
    experimental_tag,
    prob_columns,
    cache_path,
    alpha,
    output_model_path,
    class_order=None,
    conformity_score="lac",
):
    y_cal_original = df_calibration[experimental_tag].values

    # Get unique classes and create mapping
    classes_in_data = np.unique(np.concatenate([y_cal_original]))
    
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
    classes = np.arange(len(classes_original))

    init_cache(cache_path)

    # Extract probabilities in the correct class order
    P_cal = extract_probabilities(df_calibration, prob_columns, classes_original)

    is_binary = len(classes) == 2

    # Create estimator with precomputed probabilities
    estimator = ExternalProbabilisticClassifier(
        y_pred_proba=P_cal,
        classes=classes
    )

    # Use MAPIE's LACConformityScore
    score = LACConformityScore()

    mapie = SplitConformalClassifier(
        estimator=estimator,
        conformity_score=score,
        prefit=True,
        confidence_level=1 - alpha
    )

    logger.info(f"Calibrating with MAPIE LAC (alpha={alpha})...")
    mapie.estimator_ = estimator.fit(None, None)
    
    mapie.conformalize(
        X_conformalize=np.zeros((len(y_cal), 1)),
        y_conformalize=y_cal
    )

    save_dict = {
        "method": "LAC",
        "classes": classes,
        "classes_original": classes_original,
        "class_to_mapped": class_to_mapped,
        "mapped_to_class": mapped_to_class,
        "alpha": alpha,
        "prob_columns": prob_columns,
        "is_binary": is_binary,
        "mapie": mapie,
    }

    with open(output_model_path, "wb") as f:
        pickle.dump(save_dict, f)

    return save_dict


def extract_probabilities(df, prob_columns, classes_original):
    """
    Extract probabilities in the order of classes_original and normalize strictly
    to prevent MAPIE assertion errors.

    Args:
        df (pd.DataFrame): Input data frame.
        prob_columns (list or dict): Columns corresponding to probabilities.
        classes_original (list or array): Ordered list of original classes.

    Returns:
        np.ndarray: Array of shape (n_samples, n_classes) with row sums exactly 1.
    """
    n_samples = len(df)
    n_classes = len(classes_original)
    prob_matrix = np.zeros((n_samples, n_classes), dtype=float)

    if isinstance(prob_columns, dict):
        # Each class has its own column
        for i, cls in enumerate(classes_original):
            prob_matrix[:, i] = df[prob_columns[cls]].values
    else:
        # List of columns in order
        for i, col in enumerate(prob_columns):
            prob_matrix[:, i] = df[col].values

    # Clip small negative probabilities to zero
    prob_matrix = np.clip(prob_matrix, 0.0, None)

    # Strict normalization to ensure row sums == 1
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    
    # Avoid division by zero for any degenerate rows
    zero_rows = (row_sums == 0).flatten()
    if np.any(zero_rows):
        prob_matrix[zero_rows, :] = 1.0 / n_classes
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
    
    prob_matrix /= row_sums

    # Sanity check: row sums exactly 1
    assert np.allclose(prob_matrix.sum(axis=1), 1.0, atol=1e-12), \
        "Probabilities do not sum to 1 after normalization!"

    return prob_matrix



def compute_prediction_errors(y, prob_matrix, classes):
    idx = np.searchsorted(classes, y)
    return 1.0 - prob_matrix[np.arange(len(y)), idx]


def predict_conformal_classifier_proba(
    df,
    prob_columns,
    true_column=None,
    saved_model=None,
    smiles_column="Smiles",
    tag="Exp",
    chunk_size=10000,
    split="Test",
):
    if saved_model is None:
        logger.error("Saved model not provided")
        return None, None

    classes = saved_model["classes"]
    classes_original = saved_model["classes_original"]
    class_to_mapped = saved_model["class_to_mapped"]
    mapped_to_class = saved_model["mapped_to_class"]
    alpha = saved_model["alpha"]
    mapie = saved_model.get("mapie")
    is_binary = saved_model.get("is_binary", False)

    if prob_columns is None:
        prob_columns = saved_model["prob_columns"]

    has_true = (
        true_column is not None
        and true_column in df.columns
        and df[true_column].notna().any()
    )

    results_chunks = []
    all_set_sizes = []
    all_coverage = []
    all_correct = []


    # Extract IDs if available
    has_ids = "ID" in df.columns
    has_adi = "ADI" in df.columns
    has_true = true_column is not None and true_column in df.columns    
    logger.info(f"ADI {has_adi} ID {has_ids}")
    num_chunks = int(np.ceil(len(df) / chunk_size))

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df))
        df_chunk = df.iloc[start:end].copy()

        ids_chunk = df_chunk["ID"].values if has_ids else None
        adi_chunk = df_chunk["ADI"].values if has_adi else None
        smiles = df_chunk[smiles_column].values
        
        # Extract probabilities in the correct class order
        P = extract_probabilities(df_chunk, prob_columns, classes_original)
        logger.info(f"{prob_columns} classes {classes_original} prob {P}")

        # Update the estimator's probability matrix for this chunk
        mapie.estimator_.y_pred_proba = P

        # Get prediction sets from MAPIE
        mask = mapie.predict(np.zeros((len(df_chunk), 1)))

        # Convert mask to list of sets using mapped classes
        if mask.ndim == 1:  # Binary case: mask is 1D
            # True -> second class, False -> first class
            pred_sets = [[classes[1]] if m else [classes[0]] for m in mask]
            set_sizes = np.ones_like(mask, dtype=int)  # each set has size 1
        else:  # Multiclass case: mask is 2D
            pred_sets = [classes[row].tolist() for row in mask]
            set_sizes = mask.sum(axis=1)

        # Point predictions (MAPIE ignores these, just use max probability)
        y_pred_mapped = classes[np.argmax(P, axis=1)]
        y_pred = np.array([mapped_to_class[c] for c in y_pred_mapped])

        all_set_sizes.extend(set_sizes)

        coverage = None
        correct = None

        if has_true:
            y_true_original = df_chunk[true_column].values
            y_true_mapped = np.array([class_to_mapped[c] for c in y_true_original])
            
            coverage = np.array([y_true_mapped[j] in pred_sets[j] for j in range(len(y_true_mapped))], float)
            correct = (y_pred == y_true_original).astype(float)

            all_coverage.extend(coverage)
            all_correct.extend(correct)

        # Convert prediction sets back to original class labels for output
        pred_sets_original = [[mapped_to_class[c] for c in s] for s in pred_sets]

        chunk_result = {
            "ID": ids_chunk,            
            f"{tag}_true": df_chunk[true_column].values if has_true else None,
            f"{tag}_pred": y_pred,
            "Set_Size": set_sizes,
            "Prediction_Set": [str(s) for s in pred_sets_original],
            "In_Coverage": coverage,
            "Smiles": smiles,
            "top_probability": P.max(axis=1),
            "ADI": adi_chunk
        }

        for cls_orig in classes_original:
            cls_mapped = class_to_mapped[cls_orig]
            chunk_result[f"in_set_class_{cls_orig}"] = [cls_mapped in s for s in pred_sets]

        results_chunks.append(pd.DataFrame(chunk_result))

    results_df = pd.concat(results_chunks, ignore_index=True)

    metrics = {
        "average_set_size": np.mean(all_set_sizes),
        "Empirical Coverage": np.mean(all_coverage) if has_true else None,
        "Point Accuracy": np.mean(all_correct) if has_true else None,
        "Alpha": alpha,
        "Target Coverage": 1 - alpha,
        "Method": saved_model["method"],
        "n_samples": len(df),
        "n_chunks": num_chunks,
        "Split": split,
    }

    return results_df, metrics