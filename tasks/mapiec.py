import os.path
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsRegressor
from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_class_ordinal import train_conformal_classifier, predict_conformal_classifier
from tasks.vega.utils_vega import (
    replace_labels_with_keys, parse_classvalues)


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
product = None
id = "Smiles"
skip_existing = True
vega_models = None
# -


def get_clas_values(vega_models):
    df_models = pd.read_excel(vega_models, engine="openpyxl")
    df_models = df_models.loc[df_models["Key"] == data]
    df_models.head(2)
    needs_fix = df_models['ClassValues'].astype(str).str.contains('Âµ').any()
    if needs_fix:
        print("Column ClassValues contains garbled encoding — fixing...")
        df_models['ClassValues'] = df_models['ClassValues'].str.replace('Âµ', 'µ', regex=False)
    classvalues_str = df_models.loc[df_models["Key"] == data, "ClassValues"].values[0]
    print(classvalues_str)
    classvalues_dict = parse_classvalues(classvalues_str)
    return classvalues_dict


def clean_classdataset(df, model=None, classvalues_dict=None):
    values_to_drop = ["Not Classifiable", "Not classifiable",
                        "Not Classifiable", "Unknown", "", 
                        "N.A.", "No class found",
                        "Non Predicted", "NA", "-", np.nan]
    cleaned_df = df[~df[model].isin(values_to_drop)]
    print("Drop not classifiable", df.shape, cleaned_df.shape)
    cleaned_df, label_pred = replace_labels_with_keys(
        cleaned_df, model, classvalues_dict)    
    return cleaned_df, label_pred


conn = init_cache(cache_path)
# --- Load Data ---
classvalues_dict = get_clas_values(vega_models)
conn = init_cache(cache_path)
np.random.seed(42)
input_file = os.path.join(input_folder, f"{data}.xlsx")
# Load calibration data
df_calibration = pd.read_excel(input_file, sheet_name=data)
df_calibration, label_pred = clean_classdataset(df_calibration, data, classvalues_dict)
print(label_pred)
df_calibration.head()


train_conformal_classifier(
    df=df_calibration,
    pred_column=label_pred,
    cache_path=cache_path,
    alpha=0.1,
    output_model_path=product["ncmodel"]
)

test_df = pd.read_excel(input_file, sheet_name=data)
test_df, label_pred = clean_classdataset(test_df, data, classvalues_dict)
print(label_pred)
test_df.head()


result_df, metrics_per_model = predict_conformal_classifier(
    test_df, pred_column=label_pred,
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

print(f"\n✓ Results saved to {product["data"]}")