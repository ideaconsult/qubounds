import os.path
import numpy as np
import pandas as pd
from tasks.descriptors.ecfp import init_cache
from tasks.mapie_regression import predict_conformal, clean_regrdataset
from tasks.vega.utils_vega import (
    load_vega_report, writeExcel_epa, get_adi_cols, clean_vega_report_df, 
    get_main_prediction)
from tasks.assessment.utils import init_logging
import traceback
import time
from pathlib import Path
import glob


# + tags=["parameters"]
config = None
cache_path = None
ncmodel = None
product = None
data = None
input_key = None
ncm_code = None
skip_existing = None
# -


conn = init_cache(cache_path)
np.random.seed(42) 
Path(product["nb"]).parent.mkdir(parents=True, exist_ok=True)
logger = init_logging(Path(product["nb"]).parent / "logs", "report.log")

input_root = config[input_key]["VEGA_REPORTS_INPUT"]
alpha = config[input_key]["alpha"]
calculate_coverage_metrics = config[input_key]["calculate_coverage_metrics"]
id = config[input_key]["id"]

_ncmodel_path = ncmodel.format(ncm_code=ncm_code, data=data)
for file in glob.glob(_ncmodel_path):
    _ncmodel_path = file
    break
_ncmodel_path


input_file = None
for report_prefix in ["report_", "resultsw_"]:
    input_file = os.path.join(input_root, f"{report_prefix}{data}.txt")
    if os.path.exists(input_file):
        break
    else:
        input_file = None

if input_file is None:
    pd.DataFrame().to_excel(product["results"], index=False, sheet_name="error")
else:
    if skip_existing and os.path.exists(product["results"]):
        logger.info(f"{data}\tCP results exists {product['results']}")
    else:
        logger.info(f"{data}\t{input_file}")
        df, metadata = load_vega_report(file=input_file)
        logger.debug(metadata)
        logger.debug(df.columns)
        df, predicted_columns = clean_vega_report_df(df)
        logger.debug(predicted_columns)
        main_column, main_unit, main_index = get_main_prediction(data, predicted_columns)    
        logger.info(f"{data}\t{main_column}, {main_unit}, {main_index}")
        logger.debug(f"*********** {df.columns}")
        df = df[["ID", "Smiles", main_column]]
        values_to_drop = ["-", np.nan]
        df = df[~df[main_column].isin(values_to_drop)]    
        logger.debug("{data}\tpredict_conformal start")
        start_time = time.time()
        result_df, metrics_per_model = predict_conformal(
            df, pred_column=main_column,
            true_column="Experimental",
            model_path=_ncmodel_path,
            tag=data,
            smiles_column="Smiles",
            chunk_size=50000
        )
        elapsed_time = time.time() - start_time
        logger.debug(f"{data}\tpredict_conformal end (elapsed: {elapsed_time:.2f}s)")
        model_metrics = {}
        model_metrics[data] = metrics_per_model
        metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        metrics_df.index.name = 'Method Name'
        metrics_df["alpha"] = alpha
        output_data_path = product["results"]
        logger.info(f"{data}\tWriting results to {output_data_path}")        
        with pd.ExcelWriter(output_data_path, engine='xlsxwriter') as writer:
            #for sheet in ['Cover sheet', 'Summary sheet']:
            #    _df = pd.read_excel(input_file, sheet_name=sheet)
            #    _df.to_excel(writer, sheet_name=sheet, index=False)        
            if result_df is not None:
                result_df.to_excel(writer, sheet_name='Prediction Intervals', index=False)        
            metrics_df.to_excel(writer, sheet_name='Metrics') 