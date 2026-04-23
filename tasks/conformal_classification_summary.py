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
Figure: "Mean calibration coverage across 45 toxicity datasets. Classifier-based NCMs (c, o* prefixes) substantially outperform regressors (7-9 percentage points) by providing probabilistic distance predictions. Random Forest Classifier (crfecfp, highlighted) selected for optimal balance of performance and simplicity."*
"""