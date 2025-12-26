from scipy.stats import ks_2samp
import numpy as np
import logging
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
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

    else:
        raise ValueError(f"Unsupported NCM {ncm}")


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
    confidence_score=0.9
):
    """
    Plot distributions of normalized residuals S = |y - y_hat| / sigma_hat
    """

    #plt.figure(figsize=(7, 5))
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
    #plt.xlim(0, np.quantile(scores_cal, 0.99))

    if log_scale:
        plt.yscale("log")

    plt.xlabel(r"$|y - \hat y| / \hat\sigma(x)$")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
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


def plot_prediction_intervals(result_df, model, n_points=100):
    """
    Plot predicted values and prediction intervals vs. the true target value.

    Parameters:
        result_df (pd.DataFrame): DataFrame containing true, predicted, and interval values.
        model (str): Column prefix for the model (e.g., "rf", "xgb").
        n_points (int): Number of points to plot (sampled randomly).
    """
    sample = result_df.sample(n=min(n_points, len(result_df)))
    sample = sample.sort_values(by=f"{model}_true")

    x = sample[f"{model}_true"]
    y_pred = sample[f"{model}_pred"]
    y_lower = sample[f"{model}_lower"]
    y_upper = sample[f"{model}_upper"]

    plt.figure()
    plt.plot(x, y_pred, label="Predicted", marker='x', linestyle='', color='blue')
    plt.fill_between(x, y_lower, y_upper, alpha=0.3, label="Prediction Interval", color='orange')
    plt.plot(x, x, label="Ideal (y = x)", linestyle='--', color='gray')  # Optional y=x line
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value / Interval")
    plt.title(f"{model}: Prediction vs True with Conformal Intervals")
    plt.legend()
    plt.show()
