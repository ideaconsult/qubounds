import pandas as pd
from pathlib import Path
import numpy as np
from tasks.assessment.utils import init_logging
from tasks.mapie_diagnostic import (
    plot_prediction_intervals,
    plot_interval_width_histogram,
    plot_prediction_intervals_index
)
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# + tags=["parameters"]
product = None
upstream = None
vega_models = None
mode = "regression"
data = ["BCF_MEYLAN"]
ncm = "rfecfp"
# -


df_models = pd.read_excel(vega_models, engine="openpyxl")
logger = init_logging(Path(product["nb"]).parent / "logs", "plots.log")

combined_df = pd.DataFrame()
for key_star in upstream:
    logger.info(f"{key_star}")
    for key in upstream[key_star]:
        _data = key.replace("mapie_","").replace(f"_{ncm}","")
        if _data not in data:
            continue
        model_path = upstream[key_star][key].get("ncmodel", None)
        if model_path is None:
            continue
        sigma_model = {}
        with open(model_path, "rb") as f:
            sigma_model = pickle.load(f)
        logger.info(f"{ncm}\t{_data}")
        file_path = upstream[key_star][key]["data"]

        df_metrics = pd.read_excel(file_path, sheet_name="Metrics")
        
        df_train = pd.read_excel(file_path, sheet_name="Training PI")
        df_test = pd.read_excel(file_path, sheet_name="Prediction Intervals")
        plot_interval_width_histogram(
            [df_train, df_test], model=_data, figsize=(12,3), bins="auto",
            labels=["Training", "Test"], show_residual_hist=True)
        plot_prediction_intervals(df_train, model=_data)