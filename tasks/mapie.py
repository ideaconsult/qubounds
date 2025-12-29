import os.path
import numpy as np
import pandas as pd
from tasks.descriptors.ecfp import init_cache
from tasks.mapie_regression import (
    train_conformal_regression, predict_conformal, clean_regrdataset
)
from tasks.assessment.utils import init_logging
from pathlib import Path
from sklearn.model_selection import train_test_split


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
product = None
skip_existing = None
ncm = None
# -


logger = init_logging(Path(product["nb"]).parent / "logs", "report.log")
if skip_existing and os.path.exists(product["ncmodel"]) and os.path.exists(product["data"]):
    logger.info(f"{data}\tCP model exists {product['ncmodel']}")
    pass
else:
    conn = init_cache(cache_path)
    np.random.seed(42)    
    # --- 3. Load Data ---
    input_file = os.path.join(input_folder, f"{data}.xlsx")
    df_calibration = pd.read_excel(input_file, sheet_name=data)
    df_calibration = clean_regrdataset(df_calibration, data)

    test_df = pd.read_excel(input_file, sheet_name=data)
    test_df = clean_regrdataset(test_df, data)

    calibration_set = "test"

    if data in ["LOGP_MEYLAN", "LOGP_ALOGP", "LOGP_MLOGP", "LD50_KNN",
                "BCF_ARNOTGOBAS", "BCF_KNN", "FATHEAD_KNN", "FISH_KNN", "GUPPY_KNN",
                "KOA_OPERA", "KOC_OPERA", "SLUDGE_COMBASEEC50", "TOTALHL_QSARINS"]:
        # we don't have a test set here !
        experimental_tag = "Exp"
        predicted_tag = data
        n = df_calibration.shape[0]
        ratio = 0.2
        if n * 0.2 < 10:
            train_df, df_calibration = train_test_split(df_calibration, test_size=10/n, random_state=42)
            test_df = df_calibration 
            calibration_set = "test"
        else:    
            train_df, df_calibration = train_test_split(df_calibration, test_size=ratio, random_state=42)
            train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
            calibration_set = "train_split_into_3"
        train_df["residuals"] = np.abs(train_df[experimental_tag].astype(float) - train_df[predicted_tag].astype(float))
        train_cols = ["Smiles", "residuals", experimental_tag]        
        
    else:
        calibration_set = "test"
        train_meta = pd.read_excel(input_file, sheet_name="Cover sheet", header=None)
        experimental_tag = train_meta.loc[train_meta[0] == "Experimental", 1].values[0]
        predicted_tag = train_meta.loc[train_meta[0] == "Property Name", 1].values[0]        
        train_df = pd.read_excel(input_file, sheet_name="Training")
        train_df = clean_regrdataset(train_df, predicted_tag)
        train_cols = ["ID", "Smiles", "residuals", experimental_tag]
        train_df["residuals"] = np.abs(train_df[experimental_tag].astype(float) - train_df[predicted_tag].astype(float))        

    if data in ["MELTING_POINT", "MELTING_POINT_KNN"]:
        calibration_set = "test_split_into_2"
        # big test set, will split into calibration and test        
        df_calibration, test_df = train_test_split(df_calibration, test_size=0.5, random_state=42)

    train_conformal_regression(
        train_df[train_cols],
        experimental_tag,
        df_calibration,
        sheet_name=data,
        cache_path=cache_path,
        alpha=0.1,
        output_model_path=product["ncmodel"],
        ncm=ncm
    )

    result_df, metrics_per_model = predict_conformal(
        test_df, pred_column=data,
        true_column="Exp",
        model_path=product["ncmodel"],
        tag=data,
        split="Test"
    )
    model_metrics = {}
    model_metrics[data] = metrics_per_model
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    metrics_df.index.name = 'Method Name'
    metrics_df["alpha"] = alpha
    metrics_df["Split"] = "Test"

    if calibration_set != "test":
        cal_result_df, cal_metrics_per_model = predict_conformal(
            df_calibration, 
            pred_column=data,
            true_column="Exp",
            model_path=product["ncmodel"],
            tag=data,
            split="Calibration"
        )
        model_metrics = {}
        model_metrics[data] = cal_metrics_per_model
        _metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        _metrics_df.index.name = 'Method Name'
        _metrics_df["alpha"] = alpha
        _metrics_df["Split"] = "Calibration"
        metrics_df = pd.concat([metrics_df, _metrics_df])
    else:
        cal_result_df = None

    train_result_df, train_metrics_per_model = predict_conformal(
        train_df, pred_column=predicted_tag,
        true_column=experimental_tag,
        model_path=product["ncmodel"],
        tag=data,
        split="Training"
        
    )
    model_metrics = {}
    model_metrics[data] = train_metrics_per_model
    _metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    _metrics_df.index.name = 'Method Name'
    _metrics_df["alpha"] = alpha
    _metrics_df["Split"] = "Training"
    logger.info(train_metrics_per_model)
    metrics_df = pd.concat([metrics_df, _metrics_df])
    output_data_path = product["data"]
    with pd.ExcelWriter(output_data_path, engine='xlsxwriter') as writer:
        for sheet in ['Cover sheet', 'Summary sheet']:
            _df = pd.read_excel(input_file, sheet_name=sheet)
            _df.to_excel(writer, sheet_name=sheet, index=False)        
        if result_df is not None:
            result_df.to_excel(writer, sheet_name='Prediction Intervals', index=False)        
        if train_result_df is not None:            
            train_result_df.to_excel(writer, sheet_name='Training PI', index=False)        
        if cal_result_df is not None:            
            cal_result_df.to_excel(writer, sheet_name='Calibration PI', index=False)                    
        metrics_df.to_excel(writer, sheet_name='Metrics') 

    logger.info(f"{data}\tResults saved to {product["data"]}")
