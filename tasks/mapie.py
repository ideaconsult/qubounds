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
import traceback
from tasks.mapie_diagnostic import (
    plot_interval_width_histogram
)


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
    test_meta = pd.read_excel(input_file, sheet_name="Cover sheet", header=None)
    experimental_tag_test = test_meta.loc[test_meta[0] == "Experimental", 1].values[0]
    predicted_tag_test = test_meta.loc[test_meta[0] == "Property Name", 1].values[0]    
        
    df_calibration = pd.read_excel(input_file, sheet_name="Test")
    df_calibration = clean_regrdataset(df_calibration, model=predicted_tag_test)

    test_df = pd.read_excel(input_file, sheet_name="Test")
    test_df = clean_regrdataset(test_df, model=predicted_tag_test)

    calibration_set = "test"

    if data in ["LOGP_MEYLAN", "LOGP_ALOGP", "LOGP_MLOGP", "LD50_KNN",
                "BCF_ARNOTGOBAS", "BCF_KNN", "FATHEAD_KNN", "FISH_KNN", "GUPPY_KNN",
                "KOA_OPERA", "KOC_OPERA", "SLUDGE_COMBASEEC50", "TOTALHL_QSARINS"]:
        # we don't have a test set here !
        experimental_tag = experimental_tag_test
        predicted_tag = predicted_tag_test
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
        df_calibration["residuals"] = np.abs(df_calibration[experimental_tag].astype(float) - df_calibration[predicted_tag].astype(float))
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
        df_calibration["residuals"] = np.abs(df_calibration[experimental_tag].astype(float) - df_calibration[predicted_tag].astype(float))
        print(test_df.columns)

    if data in ["MELTING_POINT", "MELTING_POINT_KNN"]:
        calibration_set = "test_split_into_2"
        # big test set, will split into calibration and test        
        df_calibration, test_df = train_test_split(df_calibration, test_size=0.5, random_state=42)

    try:
        train_conformal_regression(
            train_df[train_cols],
            experimental_tag,
            df_calibration,
            sheet_name=data,
            predicted_tag_test=predicted_tag_test,
            experimental_tag_test=experimental_tag_test,
            cache_path=cache_path,
            alpha=0.1,
            output_model_path=product["ncmodel"],
            ncm=ncm
        )
    except Exception:
        traceback.print_exc()

    save_path = product["data"].replace(".xlsx", "normalized_hist_test.png")
    logger.info(save_path)
    result_df, metrics_per_model, sigma_model = predict_conformal(
        test_df, pred_column=predicted_tag_test,
        true_column=experimental_tag_test,
        model_path=product["ncmodel"],
        tag=data,
        split="Test",
        save_path=save_path
    )
    model_metrics = {}
    model_metrics[data] = metrics_per_model
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    metrics_df.index.name = 'Method Name'
    metrics_df["alpha"] = alpha
    metrics_df["Split"] = "Test"
    metrics_df["sigma_r2"] = sigma_model["sigma_r2_cal"]

    if calibration_set != "test":
        cal_result_df, cal_metrics_per_model, sigma_model = predict_conformal(
            df_calibration, 
            pred_column=predicted_tag_test,
            true_column=experimental_tag_test,
            model_path=product["ncmodel"],
            tag=data,
            split="Calibration",
            save_path=product["data"].replace(".xlsx", "normalized_hist_cal.png")
        )
        model_metrics = {}
        model_metrics[data] = cal_metrics_per_model
        _metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        _metrics_df.index.name = 'Method Name'
        _metrics_df["alpha"] = alpha
        _metrics_df["Split"] = "Calibration"
        _metrics_df["sigma_r2"] = sigma_model["sigma_r2_cal"]
        metrics_df = pd.concat([metrics_df, _metrics_df])
    else:
        cal_result_df = None

    train_result_df, train_metrics_per_model, sigma_model = predict_conformal(
        train_df, pred_column=predicted_tag,
        true_column=experimental_tag,
        model_path=product["ncmodel"],
        tag=data,
        split="Training",
        save_path=product["data"].replace(".xlsx", "normalized_hist_train.png")
        
    )
    model_metrics = {}
    model_metrics[data] = train_metrics_per_model
    _metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    _metrics_df.index.name = 'Method Name'
    _metrics_df["alpha"] = alpha
    _metrics_df["Split"] = "Training"
    _metrics_df["sigma_r2"] = sigma_model["sigma_r2"]
    logger.info(train_metrics_per_model)
    metrics_df = pd.concat([metrics_df, _metrics_df])
    output_data_path = product["data"]
    df2plot = []
    labels2plot = []
    with pd.ExcelWriter(output_data_path, engine='xlsxwriter') as writer:
        for sheet in ['Cover sheet', 'Summary sheet']:
            _df = pd.read_excel(input_file, sheet_name=sheet)
            _df.to_excel(writer, sheet_name=sheet, index=False)        
        if train_result_df is not None:            
            train_result_df.to_excel(writer, sheet_name='Training PI', index=False)        
            df2plot.append(train_result_df)        
            labels2plot.append("Training")            
        if cal_result_df is not None:            
            cal_result_df.to_excel(writer, sheet_name='Calibration PI', index=False)
            df2plot.append(cal_result_df)             
            labels2plot.append("Calibration")                    
        if result_df is not None:
            result_df.to_excel(writer, sheet_name='Prediction Intervals', index=False)
            df2plot.append(result_df)        
            labels2plot.append("Test")            
        metrics_df.to_excel(writer, sheet_name='Metrics') 

    plot_interval_width_histogram(
        df2plot, 
        model=data,
        labels=labels2plot,
        figsize=(12,4),
        show_residual_hist=True, absolute_residuals=False,
        save_path=product["data"].replace(".xlsx", "interval_hist.png"))
    logger.info(f"{data}\tResults saved to {product["data"]}")
