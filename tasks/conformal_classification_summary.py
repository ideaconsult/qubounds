import pandas as pd
import pickle
from tasks.assessment.utils import init_logging
from pathlib import Path
import traceback


# + tags=["parameters"]
product = None
upstream = None
# -


"""
In evaluating conformal prediction performance for ordinal classification tasks, we report four complementary
 metrics beyond standard coverage guarantees. Point Accuracy measures the proportion of samples where the base
 classifier's prediction exactly matches the true class, establishing baseline classifier performance. 
 Off-by-One Accuracy quantifies the percentage of predictions within one ordinal step of the true class (|y_true - y_pred| ≤ 1), 
 capturing the classifier's ability to respect ordinal relationships—a critical property for toxicity prediction 
 where adjacent classes (e.g., low vs. moderate toxicity) are semantically related. 
 Mean Ordinal Distance provides the average magnitude of prediction errors across all samples, offering a continuous
 measure of error severity that directly influences prediction set sizes in ordinal conformal prediction. 
 Finally, Singleton Efficiency measures the proportion of correctly predicted samples that receive singleton 
 prediction sets (set size = 1), assessing the efficiency of the conformal procedure: high singleton efficiency 
 indicates tight, informative prediction sets when the base classifier is correct, while low values suggest overly 
 conservative calibration that unnecessarily inflates set sizes. Together, these metrics distinguish between 
 undercoverage due to poor base classifier performance (low point accuracy, high mean distance), distribution shift 
 between calibration and test sets (adequate metrics but coverage below nominal level), and overly conservative
 conformal calibration (high coverage but low singleton efficiency).
"""

logger = init_logging(Path(product["nb"]).parent , "report.log")

combined_df = pd.DataFrame()
for key_star in upstream:
    for key in upstream[key_star]:
        model_path = upstream[key_star][key]["ncmodel"] 
        sigma_model = {}
        with open(model_path, "rb") as f:
            sigma_model = pickle.load(f)
        logger.info(sigma_model)
        file_path = upstream[key_star][key]["data"]
        try:
            df = pd.read_excel(file_path, sheet_name="Metrics")
            if "Split" not in df.columns:
                df["Split"] = "Test"
            _key = key.replace("conformal_external_classification_", "").replace("conformal_vega_classification_", "").replace("mapiec_*","")
            df["Endpoint"] = _key
            meta = pd.read_excel(file_path, sheet_name="Summary sheet")
            df = pd.merge(meta, df, on=['Method Name', 'Split'], how='outer')        
            meta = pd.read_excel(file_path, sheet_name="Cover sheet", header=None, index_col=0).transpose()
            for t in ["Property Name", "Property Description", "Dataset Name", "Dataset Description", "Property Units", "nTraining", "nTEST","Min","Max"]:
                try:
                    df[t] = meta[t].iloc[0]
                except Exception:
                    df[t] = None
            for t in ['r2', 'rmse', 'mae']:
                df[f"sigma_{t}"] = sigma_model.get(f"sigma_{t}", None)
                df[f"sigma_{t}_cal"] = sigma_model.get("sigma_diagnostics_cal", {}).get(t, None)
            #df["classes_original"] = sigma_model.get("classes_original", None)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as err:
            traceback.print_exc()
            logger.error(f"{key} {key_star} {err}")

#combined_df['Relative Interval Width'] = combined_df['average_set_size'] / (combined_df['Max'] - combined_df['Min'])
combined_df.to_excel(product["data"], index=False)        


#combined_df.sort_values(by=['average_set_size'], ascending=True).head(25)

"""
We evaluated twelve nonconformity measure (NCM) models for predicting ordinal distances in conformal toxicity classification across 45 diverse toxicity datasets, including gradient boosting (GB), random forest (RF), k-nearest neighbors (KNN), multilayer perceptron (MLP), linear discriminant analysis (LDA), and ordinal regression variants, both as classifiers (prefix "c") and regressors (no prefix). NCM models yielded remarkably similar mean calibration coverage (range: 0.81–0.92, standard deviation: 0.03), with variation within individual datasets typically below 5 percentage points, indicating that conformity score design and dataset characteristics dominate over NCM model complexity in determining marginal coverage. Random forest classifier (crfecfp) and KNN classifier (cknnecfp) achieved the highest mean calibration coverage (0.92), though simple RF regressor (rfecfp) underperformed substantially (0.81), suggesting the importance of probabilistic outputs from NCM classifiers for converting distance predictions into class pseudo-probabilities. Despite MLP demonstrating superior predictive performance on held-out data (R²=0.96 vs. RF's R²=0.85 in prior analyses), this improvement did not translate to meaningfully better conformal coverage (cmlpecfp: 0.92 vs. crfecfp: 0.92), consistent with theoretical understanding that conformal prediction's marginal coverage depends on calibration set exchangeability rather than base model accuracy, though better NCM models may still provide improved conditional coverage and efficiency. LDA consistently underperformed (ladecfp: 0.89), likely due to its linear decision boundary poorly capturing complex relationships in high-dimensional ECFP molecular fingerprints, particularly evident in challenging datasets like CARC_ANTARES (0.50 vs. 0.57–0.68 for other methods). Ordinal-aware variants (ogb, omlp) incorporating expected distance plus variance showed no coverage advantage over standard classifiers using expected distance alone (0.89–0.92 vs. 0.89–0.92), indicating that additional ordinal structure modeling provides negligible benefit when the base conformity score already respects ordinal relationships through the Least Ambiguous Classifier (LAC) framework applied to NCM-derived pseudo-probabilities. Regressor-based NCM models (gbecfp: 0.84, rfecfp: 0.81) systematically underperformed their classifier counterparts by 5–11 percentage points, demonstrating the value of probabilistic distance predictions for constructing well-calibrated class pseudo-probabilities. We therefore selected random forest classifier (crfecfp) as the NCM model for subsequent analyses, prioritizing its optimal balance of coverage performance (0.92), computational efficiency, robustness to hyperparameter choices, and practical interpretability over marginal and inconsistent gains from more complex alternatives, while noting that the choice among top-performing classifier-based NCMs (crf, cknn, cmlp) has minimal impact on overall system performance.Claude is AI and can make mistakes. Please double-check responses.
Use crfecfp (Random Forest Classifier): tied-best coverage (0.92), simple, robust, and classifier-based NCMs outperform regressors by 7-9 percentage points due to better pseudo-probability estimation for LAC conformity scores.
"""

"""
Figure: "Mean calibration coverage across 45 toxicity datasets. Classifier-based NCMs (c, o* prefixes) substantially outperform regressors (7-9 percentage points) by providing probabilistic distance predictions. Random Forest Classifier (crfecfp, highlighted) selected for optimal balance of performance and simplicity."*
"""