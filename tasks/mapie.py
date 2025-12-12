import os.path
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsRegressor
from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached
from tasks.mapie_regression import train_conformal, predict_conformal


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
product = None
# -

conn = init_cache(cache_path)
np.random.seed(42)

# --- 3. Load Data ---
input_file = os.path.join(input_folder, f"{data}.xlsx")

train_conformal(
    input_excel=input_file,
    sheet_name=data,
    cache_path=cache_path,
    alpha=0.1,
    output_model_path=product["ncmodel"]
)

test_df = pd.read_excel(input_file, sheet_name=data)

result_df, metrics_per_model = predict_conformal(
    test_df, pred_column=data,
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