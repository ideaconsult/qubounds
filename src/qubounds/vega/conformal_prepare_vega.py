import pandas as pd
import os.path
from pathlib import Path
from qubounds.vega.utils_vega import load_vega_report, writeExcel_epa, get_adi_cols
import re


def take_first_predicted_col(model):
    print(model)
    return model in ["EW_TOXICITY", "BCF_MEYLAN"]


# pay attention to exp columns when coming from test set or report - they are not encoded !
def get_main_prediction_smthwrong(model, predicted_columns):
    print("predicted_columns", predicted_columns)
    main_predicted_column = None
    main_unit = None
    main_index = None
    for index, col in enumerate(predicted_columns):
        match = re.search(r"\[(.*?)\]", col)
        _unit = match.group(1) if match else None
        # generally, we take the first column
        if main_predicted_column is None:
            main_predicted_column = col
            main_unit = _unit
            main_index = index
        # but preferrence for units used in regulatory
        if not take_first_predicted_col(model):
            if _unit in ["mg/l", "mg/L", "mg/Kg", "ug/l", "days", "1/(mg/kg-day)", "1/(mg/kg-day)"]:
                main_predicted_column = col
                main_unit = _unit
                main_index = index
    return main_predicted_column, main_unit, main_index


def get_main_prediction(model, predicted_columns):
    print("predicted_columns", predicted_columns)
    main_predicted_column = predicted_columns[0]
    match = re.search(r"\[(.*?)\]", main_predicted_column)
    main_unit = match.group(1) if match else None
    return main_predicted_column, main_unit, 0


