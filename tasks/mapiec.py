import os.path
import numpy as np
import pandas as pd
from tasks.descriptors.ecfp import init_cache
from tasks.mapie_class_lac import (
    train_conformal_classifier, predict_conformal_classifier_chunked)
from tasks.vega.utils_vega import (
    get_class_values, clean_classdataset, map_class_to_probability_label)
from tasks.assessment.utils import init_logging
from pathlib import Path
from tasks.mapie_diagnostic import plot_conformal_diagnostics

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
method_score="LAC"
# -


logger = init_logging(Path(product["nb"]).parent / "logs", "report.log")


if skip_existing and os.path.exists(product["ncmodel"]) and os.path.exists(product["data"]):
    print(f"CP model exists {product['ncmodel']}")
    pass
else:
    conn = init_cache(cache_path)
    # --- Load Data ---

    conn = init_cache(cache_path)
    np.random.seed(42)
    input_file = os.path.join(input_folder, f"{data}.xlsx")
    meta = pd.read_excel(input_file, sheet_name="Cover sheet", header=None)
    experimental_tag = meta.loc[meta[0] == "Experimental", 1].values[0]
    predicted_tag = meta.loc[meta[0] == "Property Name", 1].values[0]         

    # Load calibration data
    df_calibration = pd.read_excel(input_file, sheet_name="Test")
    test_df = pd.read_excel(input_file, sheet_name="Test")
    df_train = pd.read_excel(input_file, sheet_name="Training")

    if method_score.endswith("_proba"):
        label_pred = predicted_tag
        class_values = test_df[predicted_tag].unique()
        class_values = df_train[predicted_tag].unique()
        experimental_tag = "Experimental"
        predicted_tags = meta.loc[meta[0] == "Property Description", 1].values[0].split(";")
        logger.info(f"LABELS {class_values}\t{predicted_tags}")
        label_pred_train = map_class_to_probability_label(class_values, predicted_tags)
        label_pred_test = label_pred_train
        logger.info(f"LABELS {experimental_tag}\t{label_pred_train}")
    else:
        classvalues_dict = get_class_values(vega_models, data)
        df_calibration, label_pred = clean_classdataset(df_calibration, predicted_tag, classvalues_dict)        
        test_df, label_pred_test = clean_classdataset(test_df, predicted_tag, classvalues_dict)            
        df_train, label_pred_train = clean_classdataset(df_train, predicted_tag, classvalues_dict)    
        logger.info(f"Calibration: {label_pred} Test: {label_pred_test} Train: {label_pred_train}")

    train_conformal_classifier(
        df_train=df_train,
        experimental_tag=experimental_tag,
        predicted_tag=label_pred_train,
        df_calibration=df_calibration,
        cache_path=cache_path,
        alpha=0.1,
        output_model_path=product["ncmodel"],
        ncm=ncm,
        method_score=method_score
    )

    results = {}
    metrics_df = None
    _all = list(zip(["Test", "Training", "Calibration"],
                    [test_df, df_train, df_calibration], 
                    ["Prediction Intervals", "Training PI", "Calibration PI"]))
    for (split, df, sheet_name) in _all:
        logger.info(f"{split} {df.columns}")
        result_df, metrics_per_model = predict_conformal_classifier_chunked(
            df, 
            pred_column=label_pred_test,
            true_column=experimental_tag,
            model_path=product["ncmodel"],
            tag=data
        )
        if result_df is None:
            continue
        results[sheet_name] = result_df
        model_metrics = {}
        model_metrics[data] = metrics_per_model
        _metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        _metrics_df.index.name = 'Method Name'
        _metrics_df["alpha"] = alpha
        _metrics_df["Split"] = split
        metrics_df = _metrics_df if metrics_df is None else pd.concat([metrics_df, _metrics_df])

    #plot_conformal_diagnostics(
    #    model_path=product["ncmodel"],
    #    df_train=df_train,
    #    df_cal=df_calibration,
    #    df_test=test_df,  # Add test data
    #    experimental_tag=experimental_tag,
    #    predicted_tag=label_pred_test,
    #    output_dir="analysis/plots"
    #)
    output_data_path = product["data"]
    with pd.ExcelWriter(output_data_path, engine='xlsxwriter') as writer:
        for sheet in ['Cover sheet', 'Summary sheet']:
            _df = pd.read_excel(input_file, sheet_name=sheet)
            _df.to_excel(writer, sheet_name=sheet, index=False)       
        for sheet_name in results:
            results_df = results[sheet_name]
            if results_df is not None:
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)        
        if metrics_df is not None:
            metrics_df.to_excel(writer, sheet_name='Metrics') 

    logger.info(f"Results saved to {product["data"]}")