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
        return LGBMClassifier(
            objective="huber",
            alpha=0.9,
            num_leaves=64,
            max_depth=-1,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.7,
            min_split_gain=0.01,
            lambda_l2=5,
            early_stopping_rounds=50
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
    elif ncm == "crfecfp":
        return RandomForestClassifier(
            n_estimators=100, class_weight='balanced'
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
    residuals = distances_actual - distances_predicted
    axes[1].scatter(distances_predicted, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel(r"Predicted distance $\hat{\sigma}(x)$")
    axes[1].set_ylabel(r"Residual (actual - predicted)")
    axes[1].set_title("NCM Residuals")
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