from scipy.stats import ks_2samp
import numpy as np
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingClassifier)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier
from mord import LogisticAT, LAD  # or LogisticIT, LogisticSE
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def compute_normalized_conformity_scores(y_true, y_pred, sigma_pred):
    """
    CP-relevant nonconformity scores:
        S = |y - y_hat| / sigma_hat
    """
    mask = sigma_pred > 0
    return np.abs(y_true[mask] - y_pred[mask]) / sigma_pred[mask]


def permutation_test_quantile(scores_a, scores_b, q_level, n_perm=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    observed = abs(
        np.quantile(scores_a, q_level) -
        np.quantile(scores_b, q_level)
    )

    pooled = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)

    diffs = []
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diffs.append(abs(
            np.quantile(pooled[:n_a], q_level) -
            np.quantile(pooled[n_a:], q_level)
        ))

    return observed, np.mean(np.array(diffs) >= observed)


def diagnose_exchangeability(
    scores_cal,
    scores_ref,
    alpha,
    tag="CP"
):
    """
    Diagnose CP exchangeability using:
      - KS test on conformity scores
      - Quantile drift
      - Permutation test on CP quantile
    """
    scores_cal = scores_cal[~np.isnan(scores_cal)]
    scores_ref = scores_ref[~np.isnan(scores_ref)]

    n_cal = len(scores_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)

    # KS test
    ks_stat, ks_p = ks_2samp(scores_cal, scores_ref)

    # Quantiles
    q_cal = np.quantile(scores_cal, q_level)
    q_ref = np.quantile(scores_ref, q_level)
    delta_q = abs(q_cal - q_ref)

    # Permutation test
    obs, perm_p = permutation_test_quantile(
        scores_cal, scores_ref, q_level
    )

    diagnostics = {
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "q_cal": q_cal,
        "q_ref": q_ref,
        "delta_q": delta_q,
        "perm_pvalue": perm_p,
        "n_cal": len(scores_cal),
        "n_ref": len(scores_ref),
    }

    logger.info(
        f"{tag} Exchangeability diagnostics | "
        f"KS p={ks_p:.3g}, Δq={delta_q:.4f}, perm p={perm_p:.3g}"
    )

    return diagnostics


def flag_exchangeability(metrics, ks_thresholds=(0.05, 0.01), delta_q_fraction=(0.05, 0.1)):
    """
    Assign a traffic-light flag to exchangeability diagnostics.
    
    Parameters
    ----------
    metrics : dict
        The model_metrics dictionary containing exch_* keys.
    ks_thresholds : tuple
        (green/yellow threshold, yellow/red threshold) for KS p-value.
    delta_q_fraction : tuple
        (green/yellow, yellow/red) thresholds for relative quantile drift Δq/q_cal.
    
    Returns
    -------
    metrics : dict
        Updated metrics dictionary with 'exch_flag' key added: 'green', 'yellow', or 'red'.
    """
    # Default to green if no exchangeability metrics
    flag = "green"
    
    if all(k in metrics for k in ["exch_ks_pvalue", "exch_q_cal", "exch_delta_q"]):
        ks_p = metrics["exch_ks_pvalue"]
        q_cal = metrics.get("exch_q_cal", None)
        delta_q = metrics.get("exch_delta_q", None)

        if q_cal is not None and delta_q is not None:
            rel_delta = delta_q / q_cal if q_cal != 0 else 0.0
            
            # Red: strong violation
            if ks_p <= ks_thresholds[1] or rel_delta >= delta_q_fraction[1]:
                flag = "red"
            # Yellow: moderate deviation
            elif ks_p <= ks_thresholds[0] or rel_delta >= delta_q_fraction[0]:
                flag = "yellow"
            else:
                flag = "green"
    
    metrics["exch_flag"] = flag
    return metrics


def exchangeability_score_complete(metrics, 
                                   ks_weight=0.5, 
                                   quantile_weight=0.5,
                                   auto_thresholds=True):
    """
    Complete exchangeability scoring with automatic threshold calibration.
    
    Recommended defaults:
    - ks_weight=0.5, quantile_weight=0.5 (balanced)
    - auto_thresholds=True (derives from original KS/delta thresholds)
    """
    if not all(k in metrics for k in ["exch_ks_pvalue", "exch_q_cal", "exch_delta_q"]):
        metrics["exch_score"] = None
        metrics["exch_flag"] = None
        return metrics
    
    # Component scores
    ks_score = metrics["exch_ks_pvalue"]
    q_cal = metrics["exch_q_cal"]
    delta_q = metrics["exch_delta_q"]
    
    if q_cal != 0:
        rel_delta = abs(delta_q) / q_cal
        quantile_score = np.exp(-10 * rel_delta)
    else:
        quantile_score = 1.0 if delta_q == 0 else 0.0
    
    # Composite score
    score = ks_weight * ks_score + quantile_weight * quantile_score
    
    # Thresholds
    if auto_thresholds:
        # Based on your original criteria: 
        # Yellow: p<=0.05 or rel_delta>=0.05
        # Red: p<=0.01 or rel_delta>=0.1
        yellow_boundary = ks_weight * 0.05 + quantile_weight * np.exp(-10 * 0.05)
        red_boundary = ks_weight * 0.01 + quantile_weight * np.exp(-10 * 0.1)
        thresholds = (yellow_boundary, red_boundary)
    else:
        thresholds = (0.7, 0.4)  # Manual defaults
    
    # Assign flag
    if score >= thresholds[0]:
        flag = "green"
    elif score >= thresholds[1]:
        flag = "yellow"
    else:
        flag = "red"
    
    metrics["exch_score"] = score
    metrics["exch_flag"] = flag
    metrics["exch_thresholds"] = thresholds
    metrics["exch_components"] = {
        "ks_score": ks_score,
        "quantile_score": quantile_score,
        "rel_delta": rel_delta if q_cal != 0 else None
    }
    
    return metrics


def detect_residual_degeneracy(residuals, y, frac_zero_thr=0.7, scale_frac=0.01):
    """
    Detect whether calibration residuals are degenerate and require epsilon.

    Returns
    -------
    use_epsilon : bool
    diagnostics : dict
    """
    r = np.asarray(residuals)
    y = np.asarray(y)

    target_scale = np.std(y)
    p90 = np.quantile(r, 0.9)
    p95 = np.quantile(r, 0.95)
    frac_zero = np.mean(r < 1e-12)

    tau = scale_frac * target_scale

    use_epsilon = (
        (frac_zero > frac_zero_thr) or
        (p90 < tau)
    )

    diagnostics = {
        "p90": p90,
        "p95": p95,
        "frac_zero": frac_zero,
        "target_std": target_scale,
        "tau": tau,
    }

    return use_epsilon, diagnostics


def apply_epsilon(residuals, diagnostics):
    """
    Apply epsilon regularization to residuals using diagnostics.
    """
    epsilon = max(
        diagnostics["p95"],
        diagnostics["tau"]
    )

    r_eps = np.maximum(residuals, epsilon)

    return r_eps, epsilon


def make_sigma_model(ncm):
    if ncm == "rfecfp":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=10,
            max_features=0.5,
            bootstrap=True,
            random_state=42,
        )

    elif ncm == "gbecfp":
        return GradientBoostingRegressor(
            loss="huber",
            n_estimators=100,
            random_state=42
        )
    elif ncm == "rlgbmecfp":    
        return LGBMRegressor(
            objective="huber",
            #early_stopping_rounds=50
        )            
    elif ncm == "clgbmecfp":
        return ShapeSafeLGBMClassifier(
            objective="multiclass",
            random_state=42
        )        
    elif ncm == "rnrecfp":
        return RadiusNeighborsRegressor(
            radius=0.3,
            weights="distance",
            metric="jaccard"
        )

    elif ncm == "knnecfp":
        return KNeighborsRegressor(
            n_neighbors=5,          # intentional AD behavior
            weights="distance",     # similarity-based uncertainty
            p=1
        )
    elif ncm == "knn2ecfp":
        return KNeighborsRegressor(
            n_neighbors=2,          # intentional AD behavior
            weights="distance",     # similarity-based uncertainty
            p=1
        )
    elif ncm == "knn2jecfp":
        return KNeighborsRegressor(
            n_neighbors=2,          # intentional AD behavior
            weights="distance",  
            metric="jaccard"
        )       
    elif ncm == "ridgeecfp":
        return Ridge(
        )          
    elif ncm == "knnjecfp":
        return KNeighborsRegressor(
            n_neighbors=5,          # intentional AD behavior
            weights="distance",     # similarity-based uncertainty
            metric="jaccard"
        )    
    elif ncm == "cgbecfp":
        return GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    elif ncm == "ogbecfp":
        return GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )        
    elif ncm == "chgbecfp":
        return HistGradientBoostingClassifier(
            random_state=42
        )    
    elif ncm == "cknnecfp":
        return KNeighborsClassifier(
            n_neighbors=5,          # intentional AD behavior
            weights="distance",     # similarity-based uncertainty
            metric="jaccard"
        )     
    elif ncm == "cknn2jecfp":
        return KNeighborsClassifier(
            n_neighbors=2,          # intentional AD behavior
            weights="distance",     # similarity-based uncertainty
            metric="jaccard"
        )        
    elif ncm == "crfecfp":
        return RandomForestClassifier(
            n_estimators=100, class_weight='balanced'
            #max_features='sqrt' # Less features per split
        )
    elif ncm == "cmlpecfp":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            alpha=0.01,  # L2 regularization
            max_iter=500,
            random_state=42
        )
    elif ncm == "omlpecfp":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            alpha=0.01,  # L2 regularization
            max_iter=500,
            random_state=42
        )    
    elif ncm == "cmordecfp":
        return LogisticAT(alpha=1.0)
    elif ncm == "omordecfp":
        return LogisticAT(alpha=1.0)        
    elif ncm == "ladecfp":
        return LAD()
    else:
        raise ValueError(f"Unsupported NCM {ncm}")


def compute_ordinal_sigma(probs, classes, method='expected'):
    """
    Compute NCM sigma from probability distribution over ordinal distances.
    
    Parameters
    ----------
    probs : array, shape (n_samples, n_classes)
        Probability distribution over distance classes
    classes : array, shape (n_classes,)
        Ordinal distance values (e.g., [0, 1, 2, 3])
    method : str
        'expected' - E[distance]
        'expected_variance' - E[distance] + 0.5*sqrt(Var[distance])
        'quantile90' - 90th percentile of distance distribution
    
    Returns
    -------
    sigma : array, shape (n_samples,)
        Predicted uncertainty (expected ordinal distance)
    """
    expected = (probs * classes).sum(axis=1)
    
    if method == 'expected':
        return expected
    
    elif method == 'expected_variance':
        variance = (probs * (classes - expected[:, None])**2).sum(axis=1)
        return expected + 0.5 * np.sqrt(variance)
    
    elif method == 'quantile90':
        cumprobs = np.cumsum(probs, axis=1)
        quantiles = np.array([
            classes[min(np.searchsorted(cumprobs[i], 0.9), len(classes)-1)]
            for i in range(len(probs))
        ])
        return quantiles
    
    else:
        raise ValueError(f"Unknown method: {method}")


def sigma_diagnostics(y_true, y_pred):
    """
    Informational diagnostics for sigma-model only.
    Not used for model selection or logic.
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "median_pred": np.median(y_pred),
        "p90_pred": np.quantile(y_pred, 0.9),
    }


def plot_normalized_residuals(
    scores_cal,
    scores_train=None,
    scores_test=None,
    bins="auto",
    log_scale=False,
    title="Normalized residual distributions",
    confidence_score=0.9,
    save_path=None
):
    """
    Plot distributions of normalized residuals S = |y - y_hat| / sigma_hat

    Parameters
    ----------
    save_path : str or None
        If provided, path where the figure will be saved (e.g. "residuals.png")
    """

    plt.figure(figsize=(9, 4))

    # Calibration
    plt.hist(
        scores_cal,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Calibration"
    )

    # Train
    if scores_train is not None:
        plt.hist(
            scores_train,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="Train"
        )

    # Test
    if scores_test is not None:
        plt.hist(
            scores_test,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="Test"
        )

    q = np.quantile(scores_cal, confidence_score)
    plt.axvline(q, linestyle="--", label=f"Calibration q at {confidence_score}")

    if log_scale:
        plt.yscale("log")

    plt.xlabel(r"$|y - \hat y| / \hat\sigma(x)$")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # ---- Save if requested ----
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_interval_widths(widths, bins="auto", title="", quantile=None):
    plt.figure(figsize=(5, 3))
    plt.hist(widths, bins=bins, density=True, histtype="step", linewidth=2)
    if quantile is not None:
        plt.axvline(x=quantile, color='r', linestyle='--', label='Conformal quantile')
        plt.legend()
    plt.xlabel("Prediction interval width")
    plt.ylabel("Density")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(result_df, model, n_points=5000, figsize=(6,4), title=None):
    """
    Plot predicted values and prediction intervals vs. the true target value.
    Optionally plot prediction interval width vs ADI if ADI column exists.

    Parameters:
        result_df (pd.DataFrame): DataFrame containing true, predicted, and interval values.
        model (str): Column prefix for the model (e.g., "BCF_MEYLAN").
        n_points (int): Number of points to plot (sampled randomly).
    """

    sample = result_df.sample(n=min(n_points, len(result_df)))
    sample = sample.sort_values(by=f"{model}_true")

    x = sample[f"{model}_true"]
    y_pred = sample[f"{model}_pred"]
    y_lower = sample[f"{model}_lower"]
    y_upper = sample[f"{model}_upper"]

    has_ADI = "ADI" in sample.columns

    # --- Create figure ---
    if has_ADI:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # --- First plot: prediction vs true ---
    ax1.plot(x, y_pred, label="Predicted", marker='x', linestyle='', color='blue')
    ax1.fill_between(x, y_lower, y_upper, alpha=0.3, label="Prediction Interval", color='orange')
    ax1.plot(x, x, label="Ideal (y = x)", linestyle='--', color='gray')

    ax1.set_xlabel("True Value")
    ax1.set_ylabel("Predicted Value / Interval")
    ax1.set_title(f"{model}: Prediction vs True [{title}]")
    ax1.legend()

    # --- Second plot: interval width vs ADI ---
    if has_ADI:
        ax2.scatter(sample["ADI"], sample["Relative Interval Width"], alpha=0.6)
        ax2.set_xlabel("ADI")
        ax2.set_ylabel("Relative Interval Width")
        ax2.set_title("Interval Width vs ADI")

    plt.tight_layout()
    plt.show()


# Wrap σ-model to always return positive scale
class PositiveSigmaWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model=None, eps=1e-6):
        self.model = model
        self.eps = eps

    def fit(self, X, y):
        # Apply log internally
        log_y = np.log(y + self.eps)
        self.model.fit(X, log_y)
        
        # Forward fitted attributes for check_is_fitted
        if hasattr(self.model, "n_features_in_"):
            self.n_features_in_ = self.model.n_features_in_
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names_in_ = self.model.feature_names_in_
        
        return self

    def predict(self, X):
        return np.exp(self.model.predict(X))


def plot_normalized_ordinal_distances(
    distances_cal,
    sigma_cal,
    distances_train=None,
    sigma_train=None,
    distances_test=None,
    sigma_test=None,
    bins="auto",
    log_scale=False,
    title="Normalized ordinal distance distributions",
    confidence_score=0.9
):
    """
    Plot distributions of normalized ordinal distances: |y_true - y_pred| / σ(x)
    
    This shows how well the NCM model predicts ordinal distances.
    Ideally, these should be similar across train/cal/test sets.
    
    Parameters
    ----------
    distances_cal : array-like
        Actual ordinal distances |y_true - y_pred| on calibration set
    sigma_cal : array-like
        NCM predicted distances σ(x) on calibration set
    distances_train : array-like, optional
        Actual ordinal distances on training set
    sigma_train : array-like, optional
        NCM predicted distances on training set
    distances_test : array-like, optional
        Actual ordinal distances on test set (if true labels available)
    sigma_test : array-like, optional
        NCM predicted distances on test set
    bins : int or str, default='auto'
        Histogram bins
    log_scale : bool, default=False
        Use log scale for y-axis
    title : str
        Plot title
    confidence_score : float, default=0.9
        Quantile to mark on plot (corresponds to 1-alpha)
    """
    import matplotlib.pyplot as plt
    
    # Compute normalized scores
    scores_cal = distances_cal / sigma_cal
    
    plt.figure(figsize=(9, 4))
    
    # Calibration
    plt.hist(
        scores_cal,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Calibration"
    )
    
    # Train
    if distances_train is not None and sigma_train is not None:
        scores_train = distances_train / sigma_train
        plt.hist(
            scores_train,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="Train"
        )
    
    # Test
    if distances_test is not None and sigma_test is not None:
        scores_test = distances_test / sigma_test
        plt.hist(
            scores_test,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="Test"
        )
    
    q = np.quantile(scores_cal, confidence_score)
    plt.axvline(q, linestyle="--", color='red', 
                label=f"Cal. {100*confidence_score:.0f}th percentile = {q:.2f}")
    
    if log_scale:
        plt.yscale("log")
    
    plt.xlabel(r"$|y_{true} - y_{pred}| / \hat{\sigma}(x)$")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_prediction_set_sizes(
    set_sizes,
    bins=None,
    title="Prediction Set Size Distribution",
    alpha=None,
    show_stats=True
):
    """
    Plot distribution of prediction set sizes.
    
    Parameters
    ----------
    set_sizes : array-like
        Prediction set sizes for each sample
    bins : int or array-like, optional
        Histogram bins. If None, uses integer bins for discrete sizes.
    title : str
        Plot title
    alpha : float, optional
        Significance level (for display in title)
    show_stats : bool, default=True
        Show mean/median set size on plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(7, 4))
    
    # Use integer bins for discrete set sizes
    if bins is None:
        max_size = int(np.max(set_sizes))
        bins = np.arange(-0.5, max_size + 1.5, 1)
    
    plt.hist(
        set_sizes,
        bins=bins,
        density=True,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add statistics
    mean_size = np.mean(set_sizes)
    median_size = np.median(set_sizes)
    
    plt.axvline(mean_size, color='red', linestyle='--', 
                linewidth=2, label=f'Mean = {mean_size:.2f}')
    plt.axvline(median_size, color='blue', linestyle='--', 
                linewidth=2, label=f'Median = {median_size:.0f}')
    
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Density")
    
    if alpha is not None:
        title = f"{title} (α={alpha:.3f}, coverage≥{1-alpha:.1%})"
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if show_stats:
        # Add text box with stats
        stats_text = (
            f"Empty sets: {np.sum(set_sizes == 0)} ({100*np.mean(set_sizes == 0):.1f}%)\n"
            f"Singletons: {np.sum(set_sizes == 1)} ({100*np.mean(set_sizes == 1):.1f}%)\n"
            f"Mean: {mean_size:.2f}\n"
            f"Median: {median_size:.0f}"
        )
        plt.text(0.98, 0.97, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    return plt.gcf()


def plot_ncm_diagnostics(
    distances_actual,
    distances_predicted,
    probs_train,
    probs_cal,
    title="NCM Model: Predicted vs Actual Distances"
):
    """
    Scatter plot of NCM predictions vs actual ordinal distances.
    
    Parameters
    ----------
    distances_actual : array-like
        True ordinal distances |y_true - y_pred|
    distances_predicted : array-like
        NCM model predictions σ(x)
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot
    axes[0].scatter(distances_predicted, distances_actual, alpha=0.3, s=10)
    
    # Add diagonal line (perfect prediction)
    max_val = max(np.max(distances_actual), np.max(distances_predicted))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    axes[0].set_xlabel(r"Predicted distance $\hat{\sigma}(x)$")
    axes[0].set_ylabel(r"Actual distance $|y_{true} - y_{pred}|$")
    axes[0].set_title("NCM Predictions vs Actual")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot (actual - predicted)
    #residuals = distances_actual - distances_predicted
    if probs_train is not None:
        axes[1].hist(probs_train, bins="auto", label="Training")

    if probs_cal is not None:
        axes[1].hist(probs_cal, bins="auto", label="Calibration")
    axes[1].set_xlabel("Distance to class probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title(title)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_coverage_by_class(
    y_true,
    prediction_sets,
    class_names=None,
    title="Coverage by Class"
):
    """
    Plot empirical coverage for each class.
    
    Parameters
    ----------
    y_true : array-like
        True class labels
    prediction_sets : list of lists or array
        Prediction sets for each sample
    class_names : list, optional
        Names for each class
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    # Convert prediction_sets to list of sets if needed
    if isinstance(prediction_sets, np.ndarray):
        if len(prediction_sets.shape) == 2:  # Binary matrix
            classes = np.arange(prediction_sets.shape[1])
            pred_sets_list = [classes[prediction_sets[i].astype(bool)].tolist() 
                             for i in range(len(prediction_sets))]
        else:
            pred_sets_list = prediction_sets.tolist()
    else:
        pred_sets_list = prediction_sets
    
    # Compute coverage per class
    unique_classes = np.unique(y_true)
    coverages = []
    counts = []
    
    for cls in unique_classes:
        mask = (y_true == cls)
        covered = np.array([cls in pred_sets_list[i] for i in range(len(y_true))])
        coverage = np.mean(covered[mask])
        coverages.append(coverage)
        counts.append(np.sum(mask))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(unique_classes))
    bars = ax.bar(x, coverages, alpha=0.7, edgecolor='black')
    
    # Color bars by coverage (red if below target)
    for i, (bar, cov) in enumerate(zip(bars, coverages)):
        if cov < 0.9:  # Assuming 90% target
            bar.set_color('salmon')
        else:
            bar.set_color('lightgreen')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}',
                ha='center', va='bottom', fontsize=8)
    
    # Add horizontal line at target coverage
    ax.axhline(0.9, color='blue', linestyle='--', linewidth=2, 
               label='Target (90%)', alpha=0.7)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(title)
    ax.set_xticks(x)
    
    if class_names is not None:
        ax.set_xticklabels(class_names)
    else:
        ax.set_xticklabels(unique_classes)
    
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_conformal_diagnostics(
    model_path,
    df_train=None,
    df_cal=None,
    df_test=None,
    experimental_tag="Exp",
    predicted_tag="Pred",
    output_dir=None
):
    """
    Generate all diagnostic plots for a trained conformal classifier.
    
    Parameters
    ----------
    model_path : str
        Path to saved conformal model
    df_train : DataFrame, optional
        Training data with Smiles, experimental_tag, predicted_tag
    df_cal : DataFrame, optional
        Calibration data
    df_test : DataFrame, optional
        Test data
    experimental_tag : str
        Column name for true labels
    predicted_tag : str
        Column name for predictions
    output_dir : str, optional
        Directory to save plots. If None, uses model_path directory.
    """
    import os
    
    # Load model
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    
    sigma_model = saved["sigma_model"]
    classes_original = saved["classes_original"]
    class_to_mapped = saved["class_to_mapped"]
    alpha = saved["alpha"]
    
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
    else:
        base_name = "conformal"
    
    figures = {}
    
    # Process each dataset
    datasets = {}
    if df_train is not None:
        datasets['train'] = df_train
    if df_cal is not None:
        datasets['cal'] = df_cal
    if df_test is not None:
        datasets['test'] = df_test
    
    # Compute normalized distances for all datasets
    all_distances = {}
    all_sigmas = {}
    
    for name, df in datasets.items():
        smiles = df["Smiles"].values
        y_true_orig = df[experimental_tag].values
        y_pred_orig = df[predicted_tag].values
        
        # Remap
        y_true = np.array([class_to_mapped[c] for c in y_true_orig])
        y_pred = np.array([class_to_mapped[c] for c in y_pred_orig])
        
        # Compute distances
        distances = np.abs(y_true - y_pred).astype(float)
        
        # Get NCM predictions
        X_ecfp = np.array([smiles_to_ecfp_cached(sm) for sm in smiles])
        sigmas = sigma_model.predict(X_ecfp)
        
        all_distances[name] = distances
        all_sigmas[name] = sigmas
    
    # Plot 1: Normalized distances
    fig = plot_normalized_ordinal_distances(
        distances_cal=all_distances.get('cal'),
        sigma_cal=all_sigmas.get('cal'),
        distances_train=all_distances.get('train'),
        sigma_train=all_sigmas.get('train'),
        distances_test=all_distances.get('test'),
        sigma_test=all_sigmas.get('test'),
        title=f"Normalized Ordinal Distances (α={alpha})",
        confidence_score=1-alpha
    )
    fig.savefig(os.path.join(output_dir, f"{base_name}_normalized_distances.png"), dpi=150)
    figures['normalized_distances'] = fig
    plt.close()
    
    # Plot 2: NCM diagnostics for each dataset
    for name in datasets.keys():
        fig = plot_ncm_diagnostics(
            distances_actual=all_distances[name],
            distances_predicted=all_sigmas[name],
            title=f"NCM Model {name.capitalize()} Set Performance"
        )
        fig.savefig(os.path.join(output_dir, f"{base_name}_ncm_{name}.png"), dpi=150)
        figures[f'ncm_{name}'] = fig
        plt.close()
    
    logger.info(f"Diagnostic plots saved to {output_dir}/{base_name}_*.png")
    
    return figures


def plot_interval_width_histogram(
    result_df,
    model,
    bins="auto",
    labels=None,
    figsize=(6, 4),
    show_residual_hist=False,
    absolute_residuals=False,
    save_path=None
):
    """
    Plot histogram(s) of prediction interval widths.
    Optionally:
      - Residual histogram
      - Residuals vs interval widths scatter

    Residual = model_true - model_pred
    """

    # ---- Backward compatibility ----
    if isinstance(result_df, pd.DataFrame):
        result_df = [result_df]

    if labels is not None and len(labels) != len(result_df):
        raise ValueError("Number of labels must match number of DataFrames")

    # ---- Figure layout ----
    if show_residual_hist:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ax_w, ax_r, ax_s = axes
    else:
        fig, ax_w = plt.subplots(figsize=figsize)
        ax_r = ax_s = None

    # ---- Interval width histograms ----
    for i, df in enumerate(result_df):
        widths = df[f"{model}_upper"] - df[f"{model}_lower"]
        label = labels[i] if labels is not None else None
        try:
            ax_w.hist(
                widths,
                bins=bins,
                alpha=0.5,
                edgecolor="none",
                label=label
            )
        except Exception as err:
            logger.error(err)
    ax_w.set_xlabel("Prediction Interval Width")
    ax_w.set_ylabel("Count")
    ax_w.set_title(f"{model}: Interval Widths")

    if labels is not None:
        ax_w.legend()

    # ---- Residual-related plots ----
    if show_residual_hist:
        for i, df in enumerate(result_df):
            residuals = abs(df[f"{model}_true"] - df[f"{model}_pred"])
            if absolute_residuals:
                residuals = residuals.abs()

            widths = df[f"{model}_upper"] - df[f"{model}_lower"]
            label = labels[i] if labels is not None else None

            try:
                ax_r.hist(
                    residuals,
                    bins=bins,
                    alpha=0.5,
                    edgecolor="none",
                    label=label
                )
            except Exception as err:
                logger.error(err)
            # Scatter
            ax_s.scatter(
                residuals,
                widths,                
                alpha=0.6,
                s=3,
                label=label
            )

        ax_r.set_xlabel("Residual |True - Predicted|")
        ax_r.set_ylabel("Count")
        ax_r.set_title(f"{model}: Residuals")

        ax_s.set_xlabel("Residual")
        ax_s.set_ylabel("Prediction Interval Width")
        ax_s.set_title(f"{model}: Residuals vs Interval Width")

        if labels is not None:
            ax_r.legend()
            ax_s.legend()
    plt.tight_layout()    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_prediction_intervals_index(result_df, model, n_points=100):
    sample = result_df.sample(n=min(n_points, len(result_df))).sort_values(by=f"{model}_true")
    plt.figure()
    plt.plot(sample[f"{model}_true"].values, label="True", marker='o', linestyle='')
    plt.plot(sample[f"{model}_pred"].values, label="Predicted", marker='x', linestyle='')
    plt.fill_between(
        x=range(len(sample)),
        y1=sample[f"{model}_lower"],
        y2=sample[f"{model}_upper"],
        alpha=0.3,
        label="Prediction Interval"
    )
    plt.legend()
    plt.title(f"{model}: Prediction Intervals (sample of {n_points})")
    plt.xlabel("Sample Index (sorted by true value)")
    plt.ylabel("Target")
    plt.show()


def mark_outlier(df, col, low=0.25, up=0.75):
    Q1 = df[col].quantile(low)
    Q3 = df[col].quantile(up)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[col] >= lower) & (df[col] <= upper)
    return ~mask    


def plot_coverage_efficiency_analysis(combined_df, save_path=None, 
                                     max_labels_panel_a=20,
                                     annotate_top_n=3):
    """
    Four-panel figure showing coverage and efficiency across models.
    Reduced crowding with selective labeling.
    
    Args:
        combined_df: DataFrame with columns ['data', 'covered', 'Relative Interval Width']
        save_path: Path to save figure
        max_labels_panel_a: Maximum number of dataset labels to show in Panel A
        annotate_top_n: Number of best/worst to annotate in Panel D
    """
    
    # Calculate per-dataset statistics
    dataset_stats = combined_df.groupby('data').agg({
        'covered': ['mean', 'count'],
        'Relative Interval Width': ['mean', 'median', 'std']
    }).reset_index()
    
    dataset_stats.columns = ['data', 'coverage', 'n', 'mean_width', 'median_width', 'std_width']
    dataset_stats = dataset_stats.sort_values('coverage')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========== PANEL A: Coverage by Dataset ==========
    colors_coverage = [
        '#2E7D32' if 0.85 <= cov <= 0.95 else  # Green: good coverage
        '#FFA726' if 0.80 <= cov < 0.85 or 0.95 < cov <= 1.0 else  # Orange: acceptable
        '#D32F2F'  # Red: poor coverage
        for cov in dataset_stats['coverage']
    ]
    
    # Show all bars but only label some
    y_pos = np.arange(len(dataset_stats))
    axes[0, 0].barh(y_pos, dataset_stats['coverage'], color=colors_coverage, 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Only show labels for subset
    if len(dataset_stats) > max_labels_panel_a:
        # Show labels for: worst 5, best 5, and middle 10
        n_datasets = len(dataset_stats)
        label_indices = (list(range(5)) +  # worst 5
                        list(range(n_datasets//2 - 5, n_datasets//2 + 5)) +  # middle 10
                        list(range(n_datasets - 5, n_datasets)))  # best 5
        
        axes[0, 0].set_yticks([i for i in y_pos if i in label_indices])
        axes[0, 0].set_yticklabels([dataset_stats.iloc[i]['data'] 
                                   for i in label_indices], fontsize=8)
    else:
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(dataset_stats['data'], fontsize=8)
    
    axes[0, 0].axvline(x=0.9, color='red', linestyle='--', linewidth=2, 
                       label='90% Target')
    axes[0, 0].axvspan(0.85, 0.95, alpha=0.1, color='green', 
                       label='Acceptable Range')
    
    axes[0, 0].set_xlabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('A. Coverage by Endpoint (Sorted)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlim(0.7, 1.02)
    axes[0, 0].legend(fontsize=9, loc='lower right')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Add summary stats
    n_good = sum((0.85 <= cov <= 0.95) for cov in dataset_stats['coverage'])
    axes[0, 0].text(0.02, 0.98, 
                   f'Within target: {n_good}/{len(dataset_stats)}\n'
                   f'Mean: {dataset_stats["coverage"].mean():.3f}\n'
                   f'Median: {dataset_stats["coverage"].median():.3f}',
                   transform=axes[0, 0].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== PANEL B: Coverage vs Dataset Size ==========
    axes[0, 1].scatter(dataset_stats['n'], dataset_stats['coverage'],
                      s=100, alpha=0.6, c=colors_coverage, edgecolor='black', linewidth=1)
    axes[0, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].axhspan(0.85, 0.95, alpha=0.1, color='green')
    
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Dataset Size (n)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('B. Coverage Stability Across Dataset Sizes', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim(0.7, 1.02)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Only annotate extreme outliers (coverage < 0.8 or > 1.0)
    extreme_outliers = dataset_stats[(dataset_stats['coverage'] < 0.8) | 
                                     (dataset_stats['coverage'] > 1.0)]
    for _, row in extreme_outliers.iterrows():
        axes[0, 1].annotate(row['data'], 
                           xy=(row['n'], row['coverage']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=7, 
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', 
                                         connectionstyle='arc3,rad=0.2',
                                         linewidth=0.5))
    
    # Correlation test
    rho_size, p_size = spearmanr(dataset_stats['n'], dataset_stats['coverage'])
    axes[0, 1].text(0.02, 0.02, 
                   f'Spearman ρ = {rho_size:.3f}\np = {p_size:.3f}',
                   transform=axes[0, 1].transAxes,
                   fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PANEL C: Interval Width Distribution ==========
    # Don't show individual labels - just boxplots
    data_for_box = [combined_df[combined_df['data'] == d]['Relative Interval Width'].values 
                    for d in dataset_stats['data']]
    
    bp = axes[1, 0].boxplot(data_for_box, 
                           patch_artist=True,
                           showfliers=False,
                           vert=False)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.6)
    
    # Only label y-axis if reasonable number
    if len(dataset_stats) <= 20:
        axes[1, 0].set_yticks(range(1, len(dataset_stats) + 1))
        axes[1, 0].set_yticklabels(dataset_stats['data'], fontsize=7)
    else:
        # No labels - just show as distribution
        axes[1, 0].set_ylabel('Endpoints (sorted by coverage)', 
                             fontsize=12, fontweight='bold')
        axes[1, 0].set_yticks([])

    axes[1, 0].set_xlabel('Relative Interval Width', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('C. Interval Width by Endpoint', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Add overall statistics
    axes[1, 0].text(0.98, 0.98,
                   f'Overall:\n'
                   f'Mean: {combined_df["Relative Interval Width"].mean():.3f}\n'
                   f'Median: {combined_df["Relative Interval Width"].median():.3f}',
                   transform=axes[1, 0].transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== PANEL D: Coverage-Efficiency Tradeoff ==========
    # Size by dataset size but use sqrt scale for better visibility
    sizes = np.sqrt(dataset_stats['n']) * 2  # Scale factor for visibility
    
    scatter = axes[1, 1].scatter(dataset_stats['mean_width'], 
                                dataset_stats['coverage'],
                                s=sizes,
                                alpha=0.6, c=colors_coverage, 
                                edgecolor='black', linewidth=0.5)
    
    # Reference lines
    axes[1, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.5, label='Target coverage')
    axes[1, 1].axvline(x=dataset_stats['mean_width'].median(), 
                      color='blue', linestyle='--', linewidth=1.5, 
                      alpha=0.5, label='Median width')
    
    # Shade "ideal" quadrant (high coverage, low width)
    median_width = dataset_stats['mean_width'].median()
    axes[1, 1].fill_between([0, median_width], 0.9, 1.02, 
                           alpha=0.1, color='green', label='Ideal region')
    
    axes[1, 1].set_xlabel('Mean Relative Interval Width (Efficiency)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Coverage Rate (Validity)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('D. Coverage-Efficiency Tradeoff', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0.7, 1.02)
    axes[1, 1].legend(fontsize=9, loc='lower right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Annotate only top N best and worst (non-overlapping)
    dataset_stats['score'] = dataset_stats['coverage'] / (dataset_stats['mean_width'] + 0.01)
    best = dataset_stats.nlargest(annotate_top_n, 'score')
    worst = dataset_stats.nsmallest(annotate_top_n, 'score')
    
    # Best performers - annotate with offset to avoid overlap
    for i, (_, row) in enumerate(best.iterrows()):
        offset_y = 15 + i * 15  # Stagger vertically
        axes[1, 1].annotate(row['data'], 
                           xy=(row['mean_width'], row['coverage']),
                           xytext=(-30, offset_y), textcoords='offset points',
                           fontsize=7, color='darkgreen',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='lightgreen', alpha=0.7, edgecolor='none'),
                           arrowprops=dict(arrowstyle='->', color='darkgreen', 
                                         linewidth=0.5, alpha=0.7))
    
    # Worst performers
    for i, (_, row) in enumerate(worst.iterrows()):
        offset_y = -15 - i * 15
        axes[1, 1].annotate(row['data'], 
                           xy=(row['mean_width'], row['coverage']),
                           xytext=(30, offset_y), textcoords='offset points',
                           fontsize=7, color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='lightcoral', alpha=0.7, edgecolor='none'),
                           arrowprops=dict(arrowstyle='->', color='darkred', 
                                         linewidth=0.5, alpha=0.7))
    
    # Add size legend
    legend_sizes = [100, 1000, 10000]
    legend_points = [plt.scatter([], [], s=np.sqrt(s)*2, c='gray', alpha=0.6, edgecolor='black')
                    for s in legend_sizes]
    legend_labels = [f'n={s:,}' for s in legend_sizes]
    size_legend = axes[1, 1].legend(legend_points, legend_labels, 
                                   scatterpoints=1, frameon=True,
                                   labelspacing=1.5, title='Dataset Size',
                                   loc='lower left', fontsize=8)
    axes[1, 1].add_artist(size_legend)
    axes[1, 1].legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary (same as before)
    print("\n" + "="*70)
    print("COVERAGE AND EFFICIENCY SUMMARY")
    print("="*70)
    print(f"\nDatasets analyzed: {len(dataset_stats)}")
    print(f"\nCoverage Statistics:")
    n_good = sum((0.85 <= cov <= 0.95) for cov in dataset_stats['coverage'])
    print(f"  Mean coverage: {dataset_stats['coverage'].mean():.3f}")
    print(f"  Median coverage: {dataset_stats['coverage'].median():.3f}")
    print(f"  Range: [{dataset_stats['coverage'].min():.3f}, {dataset_stats['coverage'].max():.3f}]")
    print(f"  Within target (0.85-0.95): {n_good}/{len(dataset_stats)} ({n_good/len(dataset_stats)*100:.1f}%)")
    
    print(f"\nInterval Width Statistics:")
    print(f"  Mean width: {dataset_stats['mean_width'].mean():.4f}")
    print(f"  Median width: {dataset_stats['mean_width'].median():.4f}")
    print(f"  Range: [{dataset_stats['mean_width'].min():.4f}, {dataset_stats['mean_width'].max():.4f}]")
    
    print(f"\nBest Performers (high coverage, narrow intervals):")
    for _, row in best.iterrows():
        print(f"  {row['data']:20s}: coverage={row['coverage']:.3f}, width={row['mean_width']:.4f}")
    
    print(f"\nWorst Performers:")
    for _, row in worst.iterrows():
        print(f"  {row['data']:20s}: coverage={row['coverage']:.3f}, width={row['mean_width']:.4f}")
    
    print("="*70)
    
    return dataset_stats


def plot_coverage_efficiency_classification(
        combined_df, distance_col, distance_label=None, 
        dataset_label="Model",
        save_path=None, max_labels_panel_a=20, annotate_top_n=3
                                           ):
    """
    Four-panel coverage and efficiency analysis for classification.
    Uses prediction distance as efficiency metric (lower = more certain).
    
    Args:
        combined_df: DataFrame with ['data', 'In_Coverage', 'ADI', distance_col]
        distance_col: Name of prediction distance column (e.g., 'ALGAE_COMBASECLASS_probs_distance')
        save_path: Path to save figure
        max_labels_panel_a: Maximum labels in Panel A
        annotate_top_n: Number to annotate in Panel D
    """
    
    if distance_label is None:
        distance_label = distance_col
    # Calculate per-dataset statistics (aggregating all data together)
    dataset_stats = combined_df.groupby('data').agg({
        'In_Coverage': ['mean', 'count'],
        distance_col: ['mean', 'median', 'std'],
        'ADI': 'mean'
    }).reset_index()

    print(distance_col)
    mean_efficiency = f"mean_{distance_col}"
    
    dataset_stats.columns = ['data', 'coverage', 'n', 
                            mean_efficiency, 'median_distance', 'std_distance', 'mean_adi']
    dataset_stats = dataset_stats.sort_values('coverage')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ========== PANEL A: Coverage by Dataset ==========
    colors_coverage = [
        '#2E7D32' if 0.85 <= cov <= 0.95 else
        '#FFA726' if 0.80 <= cov < 0.85 or 0.95 < cov <= 1.0 else
        '#D32F2F'
        for cov in dataset_stats['coverage']
    ]

    y_pos = np.arange(len(dataset_stats))
    axes[0, 0].barh(y_pos, dataset_stats['coverage'], color=colors_coverage,
                    alpha=0.8, edgecolor='none', linewidth=0.5)

    axes[0, 0].axvline(x=0.9, color='red', linestyle='--', linewidth=2,
                    label='90% Target')
    axes[0, 0].axvspan(0.85, 0.95, alpha=0.1, color='green',
                    label='Acceptable')

    axes[0, 0].set_xlabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'A. Coverage by {dataset_label} (Sorted)', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].set_xlim(0.7, 1.02)
    axes[0, 0].legend(fontsize=9, loc='lower right')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    if len(dataset_stats) > 20:
        axes[0, 0].set_yticks([])
        axes[0, 0].set_ylabel(f'{dataset_label} (n={len(dataset_stats)}, sorted by coverage)', 
                            fontsize=11, fontweight='bold')
    else:
        # Only if 20 or fewer, show all labels
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(dataset_stats['data'], fontsize=8)

    # Summary stats - move to upper left to avoid blocking bars
    n_good = sum((0.85 <= cov <= 0.95) for cov in dataset_stats['coverage'])
    axes[0, 0].text(0.72, 0.98,  # Changed from 0.02 to 0.72 (upper LEFT of plot area)
                f'Within target: {n_good}/{len(dataset_stats)}\n'
                f'Mean: {dataset_stats["coverage"].mean():.3f}\n'
                f'Median: {dataset_stats["coverage"].median():.3f}',
                transform=axes[0, 0].transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== PANEL B: Coverage vs Dataset Size ==========
    axes[0, 1].scatter(dataset_stats['n'], dataset_stats['coverage'],
                      s=100, alpha=0.6, c=colors_coverage, 
                      edgecolor='black', linewidth=1)
    
    axes[0, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].axhspan(0.85, 0.95, alpha=0.1, color='green')
    
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Dataset Size (n)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('B. Coverage Stability Across Dataset Sizes',
                        fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim(0.7, 1.02)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Annotate extreme outliers
    extreme = dataset_stats[(dataset_stats['coverage'] < 0.8) |
                           (dataset_stats['coverage'] > 1.0)]
    for _, row in extreme.iterrows():
        axes[0, 1].annotate(row['data'],
                           xy=(row['n'], row['coverage']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3,rad=0.2',
                                         linewidth=0.5))
    
    # Correlation test
    rho_size, p_size = spearmanr(dataset_stats['n'], dataset_stats['coverage'])
    axes[0, 1].text(0.02, 0.02,
                   f'Spearman ρ = {rho_size:.3f}\np = {p_size:.3f}',
                   transform=axes[0, 1].transAxes, fontsize=9,
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PANEL C: Prediction Distance Distribution ==========
    # Boxplots of distance by dataset
    """
    data_for_box = [combined_df[combined_df['data'] == d][distance_col].values
                    for d in dataset_stats['data']]
    
    bp = axes[1, 0].boxplot(data_for_box, patch_artist=True,
                        showfliers=False, vert=False)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.6)
    """

    # Panel: Set Size Distribution
    grouped = combined_df.groupby(['data', distance_col]).size().unstack(fill_value=0)
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

    grouped_pct.plot(
        kind='barh',
        stacked=True,
        ax=axes[1, 0],
        cmap='RdYlGn_r',  # Red (size=1) to green (larger sets)
        edgecolor='black',
        linewidth=0.5,
        width=0.8
    )

    axes[1, 0].set_xlabel('Set size percentage (%)', fontweight='bold')
    axes[1, 0].set_ylabel('')
    axes[1, 0].set_title('Set Size Distribution', fontweight='bold')
    axes[1, 0].legend(title='Set Size', bbox_to_anchor=(1.02, 0.5), loc='center left', ncol=1)
    #axes[1, 0].set_xlim([0, 100])        
    # Conditional labeling
    if len(dataset_stats) <= 50:
        # axes[1, 0].set_yticks(range(1, len(dataset_stats) + 1))
        axes[1, 0].set_yticks(range(len(dataset_stats)))
        axes[1, 0].set_yticklabels(dataset_stats['data'], fontsize=7)
    else:
        axes[1, 0].set_ylabel(f'{dataset_label} (sorted by coverage)',
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_yticks([])
    

    axes[1, 0].set_xlim(0, None)    
    # axes[1, 0]..autoscale(enable=True, axis='x', tight=False)
    #     
    #axes[1, 0].set_xlabel(distance_label, fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'C. {distance_label} by {dataset_label}',
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Add statistics
    axes[1, 0].text(0.98, 0.98,
                f'Overall:\n'
                f'Mean: {combined_df[distance_col].mean():.3f}\n'
                f'Median: {combined_df[distance_col].median():.3f}',
                transform=axes[1, 0].transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
#    except Exception as x:
#        print(x)
# ========== PANEL D: Coverage-Efficiency Tradeoff  ==========
    # Higher Confidence = more certain (better)
    # Target: High coverage (Validity) + High Confidence = Top-Right
    
    sizes = np.sqrt(dataset_stats['n']) * 2

    scatter = axes[1, 1].scatter(dataset_stats[mean_efficiency], # Changed from distance
                                dataset_stats['coverage'],
                                s=sizes, alpha=0.6, c=colors_coverage,
                                edgecolor='black', linewidth=0.5)
    
    # Reference lines
    axes[1, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.5, label='Target coverage (90%)')
    
    median_conf = dataset_stats[mean_efficiency].median()
    axes[1, 1].axvline(x=median_conf, color='blue', linestyle='--',
                      linewidth=1.5, alpha=0.5, label=f'Median {distance_col}')
    
    # Ideal region: high coverage + high confidence (Top-Right)
    # x-range: from median/high threshold to 1.0
    axes[1, 1].fill_between([0.8, 1.2], 0.85, .95, # High-confidence zone
                           alpha=0.1, color='green', label='Ideal region')
    
    axes[1, 1].set_xlabel(f'Mean {distance_col}',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Coverage Rate (Validity)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_title('D. Coverage-Efficiency Tradeoff',
                        fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0.7, 1.02)
    #axes[1, 1].set_xlim(0.6, 1.02) # Focused on the confidence range
    axes[1, 1].grid(True, alpha=0.3)
    
    # Annotate best/worst
    # Best = High coverage AND High confidence
    #dataset_stats['score'] = dataset_stats['coverage'] * dataset_stats['mean_distance']
    #best = dataset_stats.nlargest(annotate_top_n, 'score')
    #worst = dataset_stats.nsmallest(annotate_top_n, 'score')
    in_validity_zone = dataset_stats['coverage'].between(0.85, 0.95)
    in_efficiency_zone = dataset_stats[mean_efficiency].between(.8, 1.2)

    # 2. Identify "Best" (Top-Right of the defined box)
    # We prioritize models that satisfy both, then rank by highest confidence
    dataset_stats['is_ideal'] = in_validity_zone & in_efficiency_zone
    best = dataset_stats[dataset_stats['is_ideal']].nlargest(annotate_top_n, mean_efficiency)

    # 3. Identify "Worst" (The Outliers)
    # We define worst as models that fail the validity floor (< 0.85) 
    # or fall significantly below the efficiency threshold (< 0.80)
    worst = dataset_stats[~dataset_stats['is_ideal']].nsmallest(annotate_top_n, 'coverage')
    
    for i, (_, row) in enumerate(best.iterrows()):
        offset_y = 15 + i * 15
        axes[1, 1].annotate(row['data'],
                           xy=(row[mean_efficiency], row['coverage']),
                           xytext=(-30, offset_y), textcoords='offset points',
                           fontsize=7, color='darkgreen',
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='lightgreen', alpha=0.7,
                                    edgecolor='none'),
                           arrowprops=dict(arrowstyle='->',
                                         color='darkgreen',
                                         linewidth=0.5, alpha=0.7))
    
    for i, (_, row) in enumerate(worst.iterrows()):
        offset_y = -15 - i * 15
        axes[1, 1].annotate(row['data'],
                           xy=(row[mean_efficiency], row['coverage']),
                           xytext=(30, offset_y), textcoords='offset points',
                           fontsize=7, color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='lightcoral', alpha=0.7,
                                    edgecolor='none'),
                           arrowprops=dict(arrowstyle='->',
                                         color='darkred',
                                         linewidth=0.5, alpha=0.7))
    
    # Size legend
    legend_sizes = [100, 1000, 10000]
    legend_points = [plt.scatter([], [], s=np.sqrt(s)*2, c='gray',
                                alpha=0.6, edgecolor='black')
                    for s in legend_sizes]
    size_legend = axes[1, 1].legend(legend_points,
                                   [f'n={s:,}' for s in legend_sizes],
                                   scatterpoints=1, frameon=True,
                                   labelspacing=1.5, title='Dataset Size',
                                   loc='lower left', fontsize=8)
    axes[1, 1].add_artist(size_legend)
    axes[1, 1].legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("CLASSIFICATION COVERAGE AND SET SIZE SUMMARY")
    print("="*70)
    print(f"\nDatasets analyzed: {len(dataset_stats)}")
    print(f"\nCoverage Statistics:")
    print(f"  Mean: {dataset_stats['coverage'].mean():.3f}")
    print(f"  Median: {dataset_stats['coverage'].median():.3f}")
    print(f"  Range: [{dataset_stats['coverage'].min():.3f}, {dataset_stats['coverage'].max():.3f}]")
    print(f"  Within target (0.85-0.95): {n_good}/{len(dataset_stats)} ({n_good/len(dataset_stats)*100:.1f}%)")
    
    print(f"\n{distance_label}:")
    print(f"  Mean: {dataset_stats[mean_efficiency].mean():.3f}")
    print(f"  Median: {dataset_stats[mean_efficiency].median():.3f}")
    print(f"  Range: [{dataset_stats[mean_efficiency].min():.3f}, {dataset_stats[mean_efficiency].max():.3f}]")
    
    rho_dist_cov = spearmanr(dataset_stats[mean_efficiency], dataset_stats['coverage'])
    print(f"\nCorrelation Tests:")
    print(f"  {distance_label} vs Coverage: ρ = {rho_dist_cov[0]:.3f}, p = {rho_dist_cov[1]:.3f}")
    print(f"  {distance_label} size vs Coverage: ρ = {rho_size:.3f}, p = {p_size:.3f}")
    
    print(f"\nBest Performers (high coverage, low distance):")
    for _, row in best.iterrows():
        print(f"  {row['data']:25s}: coverage={row['coverage']:.3f}, {distance_label}={row[mean_efficiency]:.3f}")
    
    print(f"\nWorst Performers:")
    for _, row in worst.iterrows():
        print(f"  {row['data']:25s}: coverage={row['coverage']:.3f}, {distance_label}={row[mean_efficiency]:.3f}")
    
    print("="*70)
    
    return dataset_stats


def figure_spearman_classification(df, score_col_label="Score", save_path=None):
    """
    Spearman correlation analysis for classification: ADI vs score.
    Mirrors the regression version but uses prediction distance instead of interval width.
    
    Args:
        df: DataFrame with correlation results per dataset

        save_path: Path to save figure
    """
    # Sort by correlation strength
    df_sorted = df.sort_values('rho').reset_index(drop=True)

    # Assign colors based on p-value significance
    colors = []
    for p in df_sorted['p']:
        if p < 0.001:
            colors.append('#2E7D32')  # Dark green - highly significant
        elif p < 0.05:
            colors.append('#FFA726')  # Orange - significant
        else:
            colors.append('#D32F2F')  # Red - not significant

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 14))

    # === PLOT 1: Correlation bars ===
    y_pos = np.arange(len(df_sorted))
    ax1.barh(y_pos, df_sorted['rho'], color=colors, alpha=0.8, 
            edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['data'], fontsize=9)
    ax1.set_xlabel('Spearman ρ', fontsize=12)
    ax1.set_title(f'ADI vs {score_col_label} Correlation\n(Sorted by Strength)', 
                fontsize=12, pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    # Note: For classification, positive correlation might be expected
    # (higher ADI → higher distance → more certain/singleton predictions)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', label='p < 0.001'),
        Patch(facecolor='#FFA726', label='p < 0.05'),
        Patch(facecolor='#D32F2F', label='p ≥ 0.05')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=10)

    # === PLOT 2: Scatter plot with dataset size ===
    ax2.scatter(df_sorted['rho'], df_sorted['n'], c=colors, s=100, 
                alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dataset Size (n)', fontsize=12, fontweight='bold')
    ax2.set_title('Correlation Strength vs Dataset Size', 
                fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')

    # Annotate top 3 largest datasets
    top_n = df_sorted.nlargest(3, 'n')
    for _, row in top_n.iterrows():
        ax2.annotate(row['data'], 
                    xy=(row['rho'], row['n']), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax2.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Mean ρ: {df['rho'].mean():.3f}")
    print(f"Median ρ: {df['rho'].median():.3f}")
    print(f"Range: [{df['rho'].min():.3f}, {df['rho'].max():.3f}]")
    print(f"\nSignificant correlations (p < 0.05): {(df['p'] < 0.05).sum()}/{len(df)}")
    print(f"Highly significant (p < 0.001): {(df['p'] < 0.001).sum()}/{len(df)}")
    print(f"\nNegative correlations: {(df['rho'] < 0).sum()}/{len(df)}")
    print(f"Positive correlations: {(df['rho'] > 0).sum()}/{len(df)}")
    print(f"\nDataset size range: {df['n'].min()} to {df['n'].max()}")
    print(f"Median dataset size: {df['n'].median():.0f}")

    # Check if dataset size correlates with correlation strength
    size_rho_corr = df['n'].corr(df['rho'], method='spearman')
    print(f"\nCorrelation between dataset size and ρ: {size_rho_corr:.3f}")
    print("(Does larger n lead to stronger/weaker correlations?)")


def distance_by_adi_bins_classification(
        df, distance_col='ncm_score', distance_label="Score", save_path=None):
    """
    Analyze prediction distance and coverage by ADI bins for classification.
    Mirrors the regression coverage_by_adi_bins but uses distance instead of interval width.
    
    Args:
        df: DataFrame with columns ['ADI', 'In_Coverage', distance_col]
        distance_col: Name of prediction distance column
        save_path: Path to save figure
    """
    # Bin ADI into groups
    df['ADI_bin'] = pd.cut(df['ADI'], bins=[0, 0.5, 0.75, 0.85, 1.0], 
                        labels=['Very Low\n(0-0.5)', 'Low\n(0.5-0.75)', 
                                'Moderate\n(0.75-0.85)', 'High\n(0.85-1.0)'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # === PLOT 1: Coverage rate by ADI bin ===
    coverage_by_adi = df.groupby('ADI_bin')['In_Coverage'].agg(['mean', 'count'])
    coverage_by_adi['mean'].plot(kind='bar', ax=ax1, color='#2E7D32', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    ax1.set_title('Coverage Rate by Applicability Domain', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target 90%')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample sizes on bars
    for i, (idx, row) in enumerate(coverage_by_adi.iterrows()):
        ax1.text(i, row['mean'] + 0.02, f"n={row['count']}", 
                ha='center', fontsize=9, fontweight='bold')

    # === PLOT 2: Prediction distance by ADI bin ===
    df.boxplot(column=distance_col, by='ADI_bin', ax=ax2, patch_artist=True)
    ax2.set_ylabel(distance_label, fontsize=12, fontweight='bold')
    ax2.set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    ax2.set_title(f'{distance_label} by Applicability Domain', fontsize=14, fontweight='bold')
    ax2.get_figure().suptitle('')  # Remove auto-title from boxplot
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n=== Coverage by ADI Bin ===")
    print(coverage_by_adi)
    print(f"\nOverall coverage: {df['In_Coverage'].mean():.1%}")
    print(f"Mean {distance_label}: {df[distance_col].mean():.3f}")
    

class ShapeSafeLGBMClassifier(LGBMClassifier):
    def predict_proba(self, X, *args, **kwargs):
        probas = super().predict_proba(X, *args, **kwargs)
        # LightGBM occasionally returns (n,) for binary; 
        # Scikit-learn expects (n, 2)
        if probas.ndim == 1:
            return np.vstack([1 - probas, probas]).T
        return probas    