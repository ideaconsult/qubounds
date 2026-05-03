import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_diagnostic import (
    exchangeability_score_complete,
    detect_residual_degeneracy,
    sigma_diagnostics, make_sigma_model, apply_epsilon,
    plot_normalized_residuals, plot_interval_widths,
    PositiveSigmaWrapper
)
from scipy.stats import ks_2samp

import logging

logger = logging.getLogger(__name__)


def clean_regrdataset(df, model=None):
    values_to_drop = ["-", np.nan]
    return df[~df[model].isin(values_to_drop)]

# -------------------------------------------
# 1. Base predictor that returns precomputed predictions
# -------------------------------------------
class ExternalPredictor(RegressorMixin):
    """
    Returns externally provided predictions; X is ignored.
    Works with MAPIE as a drop-in estimator.
    """
    def __init__(self, y_pred):
        self.y_pred = np.asarray(y_pred)

    def fit(self, X=None, y=None):
        # MAPIE will copy this into estimator_
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # X length determines how many predictions to return
        return self.y_pred[:len(X)]


# ============================================
# Main Training Function
# ============================================
def train_conformal_regression(
        df_train, experimental_tag,
        df_calibration, sheet_name, 
        experimental_tag_test="Exp", predicted_tag_test="Pred",
        cache_path= None,
        alpha=0.1, output_model_path=None, ncm="knnecfp"):

    # -------------------------------------------
    # 1. Load calibration data
    # -------------------------------------------

    smiles = df_calibration["Smiles"].values
    y_cal = df_calibration[experimental_tag_test].values
    y_pred_cal = df_calibration[predicted_tag_test].values.astype(float)

    smiles_train = df_train["Smiles"].values
    y_exp_train = df_train[experimental_tag].values.astype(float)
    # -------------------------------------------
    # 2. Compute ECFP
    # -------------------------------------------
    init_cache(cache_path)

    logger.info(f"{sheet_name}\tComputing ECFP for training set...")
    X_ecfp_train = np.array([smiles_to_ecfp_cached(sm) for sm in smiles_train])
    logger.info(f"{sheet_name}\tECFP completed for training set.")

    logger.info(f"{sheet_name}\tComputing ECFP for calibration set...")
    X_ecfp_cal = np.array([smiles_to_ecfp_cached(sm) for sm in smiles])
    logger.info(f"{sheet_name}\tECFP completed for calibration set.")

    # -------------------------------------------
    # 3. Residual processing (prefitted-safe)
    # -------------------------------------------
    residuals_train = df_train["residuals"]
    use_eps, diag = detect_residual_degeneracy(residuals_train, 
                                               y_exp_train)
    logger.info(
        f"Residual diagnostics: "
        f"p90={diag['p90']:.4g}, "
        f"p95={diag['p95']:.4g}, "
        f"frac_zero={diag['frac_zero']:.2f}, "
        f"use_epsilon={use_eps}"
    )
    #if use_eps:
    #    residuals_train, epsilon = apply_epsilon(residuals_train, diag)
    #    logger.warning(f"Applied epsilon regularization: epsilon={epsilon:.4g}")

    # -------------------------------------------
    # 4. Train sigma model
    # -------------------------------------------
    #if ncm == "knnecfp":
    #    sigma_model = make_sigma_model(ncm)
    sigma_model = make_sigma_model(ncm)
    #else:
    #    sigma_model = PositiveSigmaWrapper(make_sigma_model(ncm))
    sigma_model.fit(X_ecfp_train, residuals_train)
    sigma_pred_train = sigma_model.predict(X_ecfp_train)
    diag_sigma = sigma_diagnostics(residuals_train, sigma_pred_train)
    logger.info(
        f"Sigma model diagnostics (training): "
        f"R2={diag_sigma['r2']:.3f}, "
        f"RMSE={diag_sigma['rmse']:.4g}, "
        f"MAE={diag_sigma['mae']:.4g}, "
        f"median_pred={diag_sigma['median_pred']:.4g}, "
        f"p90_pred={diag_sigma['p90_pred']:.4g}"
    )
    sigma_pred_cal = sigma_model.predict(X_ecfp_cal)
    residuals_cal = df_calibration["residuals"]
    diag_sigma_cal = sigma_diagnostics(residuals_cal, sigma_pred_cal)
    logger.info(
        f"Sigma model diagnostics (test): "
        f"R2_cal={diag_sigma_cal['r2']:.3f}, "
        f"RMSE_cal={diag_sigma_cal['rmse']:.4g}, "
        f"MAE_cal={diag_sigma_cal['mae']:.4g}, "
        f"median_pred_cal={diag_sigma_cal['median_pred']:.4g}, "
        f"p90_pred_cal={diag_sigma_cal['p90_pred']:.4g}"
    )    
    # -------------------------------------------
    # 5. Conformal predictor (MAPIE)
    # -------------------------------------------
    estimator = ExternalPredictor(y_pred_cal)

    conformity_score = ResidualNormalisedScore(
        residual_estimator=sigma_model,
        prefit=True,
        sym=True
    )

    mapie = SplitConformalRegressor(
        estimator=estimator,
        conformity_score=conformity_score,
        prefit=True,
        confidence_level=1 - alpha
    )

    mapie.estimator_ = estimator.fit(None, None)

    mapie.conformalize(
        X_conformalize=X_ecfp_cal,
        y_conformalize=y_cal
    )

    # -------------------------------------------
    # 6. Save model
    # -------------------------------------------
    save_dict = {
        "mapie": mapie,
        "sigma_model": sigma_model,
        "ncm": ncm,
        "use_epsilon": use_eps,
        "residual_diagnostics": diag,
        "sigma_diagnostics": diag_sigma,  # full diagnostics dict
        # Backward-compatible keys
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

    logger.info(f"Conformal model saved to: {output_model_path} E2 train {save_dict['sigma_r2']} R2 test {save_dict['sigma_r2_cal']}")


def predict_conformal(df, pred_column, true_column=None, 
                      model_path=None, tag="Pred", smiles_column="Smiles",
                      chunk_size=10000, split="Test", save_path=None):
    """
    Predict conformal intervals with chunked processing for large datasets.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with SMILES and predictions
    pred_column : str
        Column name containing model predictions
    true_column : str, optional
        Column name containing true values for coverage calculation
    model_path : str
        Path to saved conformal model
    tag : str
        Prefix for output columns
    smiles_column : str
        Column name containing SMILES strings
    chunk_size : int
        Number of rows to process at once (default: 1000)
    
    Returns
    -------
    tuple
        (results_df, metrics_dict)
    """
    # Load model once
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    
    mapie = saved["mapie"]
    sigma_model = saved["sigma_model"]
    ncm = saved["ncm"]
    
    # Get conformity scores for quantile computation
    conformity_scores = mapie._mapie_regressor.conformity_scores_
    conformity_scores = conformity_scores[~np.isnan(conformity_scores)]
    
    # Get alpha
    confidence_level = mapie.confidence_level if hasattr(mapie, 'confidence_level') else 0.9
    alpha = 1 - confidence_level
    
    # Compute quantile once
    n = len(conformity_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    quantile = np.quantile(conformity_scores, min(q_level, 1.0))
    
    logger.info(f"{tag}\tProcessing {len(df)} samples in chunks of {chunk_size}")
    logger.info(f"{tag}\tCalibration set size: {n} Alphas: {mapie._alphas} Quantile: {quantile:.4f}")
    
    # Initialize accumulators for metrics
    all_interval_widths = []
    all_in_interval = []
    results_chunks = []
    
    all_ref_conformity_scores = []

    # Extract IDs if available
    has_ids = "ID" in df.columns
    has_adi = "ADI" in df.columns
    has_true = true_column is not None and true_column in df.columns
    
    # Process in chunks
    num_chunks = int(np.ceil(len(df) / chunk_size))
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        logger.info(f"{tag}\tProcessing chunk {i+1}/{num_chunks} (rows {start_idx}-{end_idx})...")
        
        # Get chunk data
        df_chunk = df.iloc[start_idx:end_idx].copy()
        
        ids_chunk = df_chunk["ID"].values if has_ids else None
        adi_chunk = df_chunk["ADI"].values if has_adi else None
        smiles_chunk = df_chunk[smiles_column].values
        y_pred_chunk = df_chunk[pred_column].values.astype(float)
        y_true_chunk = df_chunk[true_column].values if has_true else None
        
        # Compute ECFP for chunk
        X_ecfp_chunk = np.array([smiles_to_ecfp_cached(sm) for sm in smiles_chunk])
        
        # Predict sigma
        sigma_pred_chunk = sigma_model.predict(X_ecfp_chunk)

        # we could use mapie.predict_interval() instead ...
        # this is now only for comparison,  to be switched later
        # we have fixed estimator taking values from the file
        mapie.estimator_.y_pred = y_pred_chunk       
        try: 
            y_pred, y_pis = mapie.predict_interval(X_ecfp_chunk)
        except Exception as err:
            logger.error(err)
            continue
        
        # DIAGNOSTICS
        # Compute reference conformity scores S = |y - y_hat| / sigma_hat
        eps = 1e-6          
        if has_true:
            #logger.info(sigma_pred_chunk[sigma_pred_chunk <= 0])
            sigma_safe = np.maximum(sigma_pred_chunk, eps)
            valid = sigma_safe > 0
            ref_scores_chunk = (
                np.abs(y_true_chunk[valid] - y_pred_chunk[valid]) /
                sigma_safe[valid]
            )
            all_ref_conformity_scores.append(ref_scores_chunk)        
            #logger.info(f"{len(ref_scores_chunk)}")
            #logger.info(f"{len(y_true_chunk)}")
        else:
            sigma_safe = np.maximum(sigma_pred_chunk, eps)
            ref_scores_chunk = None
        
        # Compute intervals: pred ± quantile * sigma
        y_pi_lower = y_pred_chunk - quantile * sigma_safe
        y_pi_upper = y_pred_chunk + quantile * sigma_safe
        
        print(y_pred_chunk, y_pred)
        print(y_pi_lower, y_pi_upper, y_pis)

        # Interval widths
        interval_widths_chunk = y_pi_upper - y_pi_lower
        all_interval_widths.extend(interval_widths_chunk)
        
        # Coverage (if true values exist)
        if has_true:
            in_interval_chunk = (y_true_chunk >= y_pi_lower) & (y_true_chunk <= y_pi_upper)
            all_in_interval.extend(in_interval_chunk)
        
        # Build chunk result
        chunk_result = {
            "ID": ids_chunk,
            f"{tag}_ncm": ref_scores_chunk,            
            f"{tag}_true": y_true_chunk,
            f"{tag}_pred": y_pred_chunk,
            f"{tag}_lower": y_pi_lower,
            f"{tag}_upper": y_pi_upper,
            f"{tag}_sigma": sigma_pred_chunk,
            "Interval_Width": interval_widths_chunk,
            "Smiles": smiles_chunk,
            "ADI": adi_chunk,
            f"{tag}_pred_mapie": y_pred,
            f"{tag}_lower_mapie": y_pis[:, 0, 0],
            f"{tag}_upper_mapie": y_pis[:, 1, 0],
        }


        logger.info(f"{len(sigma_pred_chunk)}")
        logger.info(f"{len(interval_widths_chunk)}")
        
        results_chunks.append(pd.DataFrame(chunk_result))
    
    # Concatenate all chunks
    if len(results_chunks) == 0:
        model_metrics = {
            "ncm": ncm, 
            "ncm_descriptors": "ECFP",
            "n_samples": len(df),
            "n_chunks": num_chunks
            }        
        return None, model_metrics, saved
    
    results_df = pd.concat(results_chunks, ignore_index=True)
    
    # Compute metrics across all data
    avg_interval_width = np.mean(all_interval_widths)
    empirical_coverage = np.mean(all_in_interval) if has_true else None
    #try:
    #    plot_interval_widths(all_interval_widths, title=f"{tag} {split}")
    #except Exception as err:
    #    logger.warning(err)
    
    model_metrics = {
        "Average Interval Width": avg_interval_width,
        "Empirical coverage": empirical_coverage,
        "ncm": ncm, 
        "ncm_descriptors": "ECFP",
        "n_samples": len(df),
        "n_chunks": num_chunks
        }
        
    if len(all_ref_conformity_scores) > 0:
        ref_scores = np.concatenate(all_ref_conformity_scores)        
        if split != "Calibration":
            plot_normalized_residuals(
                scores_cal=conformity_scores,
                scores_train=ref_scores if split == "Training" else None,
                scores_test=ref_scores if split == "Test" else None,
                log_scale=False,
                title=f"Normalized residual distributions {tag}",
                save_path=save_path
            )        

    # >>> DIAGNOSTICS 
    if has_true and len(all_ref_conformity_scores) > 0:
        ks_stat, ks_p = ks_2samp(conformity_scores, ref_scores)
        try:
            q_ref = np.quantile(ref_scores, q_level)
            delta_q = abs(quantile - q_ref)
            logger.info(
                f"{tag}\tExchangeability diagnostics | "
                f"KS p={ks_p:.3g}, "
                f"q_cal={quantile:.4f}, q_ref={q_ref:.4f}, Delta_q={delta_q:.4f}"
            )
        except Exception as err:
            q_ref = None
            delta_q = None            
            logger.error(err)

        # Flatten metrics with 'exch.' prefix
        model_metrics["exch_ks_stat"] = ks_stat
        model_metrics["exch_ks_pvalue"] = ks_p
        model_metrics["exch_q_cal"] = quantile
        model_metrics["exch_q_ref"] = q_ref
        model_metrics["exch_delta_q"] = delta_q
        model_metrics["exch_n_cal"] = len(conformity_scores)
        model_metrics["exch_n_ref"] = len(ref_scores)

        try:
            model_metrics = exchangeability_score_complete(model_metrics)
        except Exception as err:
            logger.warning(err)
    
    logger.info(f"{tag}\tCompleted processing all {len(df)} samples Average Interval Width: {avg_interval_width:.4f}")
    if empirical_coverage is not None:
        logger.info(f"{tag}\tEmpirical Coverage: {empirical_coverage:.4f}")
    
    return results_df, model_metrics, saved

