import os.path
import numpy as np
import pandas as pd
from tasks.descriptors.ecfp import init_cache
from tasks.mapie_class_ordinal import train_conformal_classifier, predict_conformal_classifier
from tasks.vega.utils_vega import (
    replace_labels_with_keys, parse_classvalues)
from tasks.assessment.utils import init_logging
from pathlib import Path


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
product = None
id = "Smiles"
skip_existing = True
vega_models = None
ncm = None
# -

logger = init_logging(Path(product["nb"]).parent / "logs", "report.log")

def get_clas_values(vega_models):
    df_models = pd.read_excel(vega_models, engine="openpyxl")
    df_models = df_models.loc[df_models["Key"] == data]
    df_models.head(2)
    needs_fix = df_models['ClassValues'].astype(str).str.contains('Âµ').any()
    if needs_fix:
        logger.info("Column ClassValues contains garbled encoding — fixing...")
        df_models['ClassValues'] = df_models['ClassValues'].str.replace('Âµ', 'µ', regex=False)
    classvalues_str = df_models.loc[df_models["Key"] == data, "ClassValues"].values[0]
    logger.info(classvalues_str)
    classvalues_dict = parse_classvalues(classvalues_str)
    return classvalues_dict


def clean_classdataset(df, model=None, classvalues_dict=None):
    values_to_drop = ["Not Classifiable", 
                        "Not classifiable",
                        "Not classifiable", "Unknown", "", 
                        "N.A.", "No class found",
                        "Non Predicted",
                        "Not predicted", "NA", "-", np.nan]
    # values_to_drop = []    
    logger.info(df[model].unique())
    cleaned_df = df[~df[model].isin(values_to_drop)]
    logger.info(f"Drop not classifiable {df.shape} --> {cleaned_df.shape}")
    cleaned_df, label_pred = replace_labels_with_keys(
        cleaned_df, model, classvalues_dict) 
    logger.info(cleaned_df[label_pred].unique())   
    return cleaned_df, label_pred


if skip_existing and os.path.exists(product["ncmodel"]) and os.path.exists(product["data"]):
    print(f"CP model exists {product['ncmodel']}")
    pass
else:
    conn = init_cache(cache_path)
    # --- Load Data ---
    classvalues_dict = get_clas_values(vega_models)
    conn = init_cache(cache_path)
    np.random.seed(42)
    input_file = os.path.join(input_folder, f"{data}.xlsx")
    # Load calibration data
    df_calibration = pd.read_excel(input_file, sheet_name=data)
    df_calibration, label_pred = clean_classdataset(df_calibration, data, classvalues_dict)

    test_df = pd.read_excel(input_file, sheet_name=data)
    test_df, label_pred_test = clean_classdataset(test_df, data, classvalues_dict)

    df_train = pd.read_excel(input_file, sheet_name="Training")
    train_meta = pd.read_excel(input_file, sheet_name="Cover sheet", header=None)
    experimental_tag = train_meta.loc[train_meta[0] == "Experimental", 1].values[0]
    predicted_tag = train_meta.loc[train_meta[0] == "Property Name", 1].values[0]         
    df_train, label_pred_train = clean_classdataset(df_train, predicted_tag, classvalues_dict)    

    logger.info(f"Calibration: {label_pred} Test: {label_pred_test} Train: {label_pred_train}")

    train_conformal_classifier(
        df_train=df_train,
        experimental_tag=experimental_tag,
        predicted_tag=label_pred_train,
        df_calibration=df_calibration,
        pred_column=label_pred,
        cache_path=cache_path,
        alpha=0.1,
        output_model_path=product["ncmodel"],
        ncm=ncm
    )

    test_df = pd.read_excel(input_file, sheet_name=data)
    test_df, label_pred_test = clean_classdataset(test_df, data, classvalues_dict)

    result_df, metrics_per_model = predict_conformal_classifier(
        test_df, pred_column=label_pred_test,
        true_column="Exp",
        model_path=product["ncmodel"],
        tag=data
    )

    model_metrics = {}
    model_metrics[data] = metrics_per_model
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    metrics_df.index.name = 'Method Name'
    metrics_df["alpha"] = alpha
    metrics_df

    output_data_path = product["data"]
    with pd.ExcelWriter(output_data_path, engine='xlsxwriter') as writer:
        for sheet in ['Cover sheet', 'Summary sheet']:
            _df = pd.read_excel(input_file, sheet_name=sheet)
            _df.to_excel(writer, sheet_name=sheet, index=False)        
        if result_df is not None:
            result_df.to_excel(writer, sheet_name='Prediction Intervals', index=False)        
        metrics_df.to_excel(writer, sheet_name='Metrics') 

    logger.info(f"Results saved to {product["data"]}")