import os.path
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from tasks.descriptors.ecfp import init_cache
from tasks.descriptors.ecfp import smiles_to_ecfp_cached


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
# -


conn = init_cache(cache_path)

# --- 1. Define the Proxy Model (Same as before) ---
class PrecalculatedPredictor(RegressorMixin):
    def __init__(self):
        self.is_fitted_ = True
    def fit(self, X, y):
        return self
    def predict(self, X):
        check_is_fitted(self)
        # X[:, 0] holds the point prediction (ŷ)
        return X[:, 0].flatten() 

# --- 2. Prepare data ---


input_file = os.path.join(input_folder, f"{data}.xlsx")
calibration_df = pd.read_excel(input_file, sheet_name=data)
calibration_df.head()


N_CAL = calibration_df.shape[0]
N_TEST = 10

y_cal = calibration_df["Exp"].values

# Simulate the two pieces of information we have for each point:
y_pred_cal = calibration_df[data].values

# Combine them into the single input matrix X (Features now have 2 columns!)
# X = [ [ŷ1, σ1], [ŷ2, σ2], ... ]
#X_cal = np.vstack([y_pred_cal, sigma_pred_cal]).T
X_cal = np.array([smiles_to_ecfp_cached(sm) for sm in calibration_df["Smiles"]])

# Simulate Test Data
y_pred_test = y_pred_cal[:N_TEST] 
X_test = X_cal[:N_TEST] 

# --- 3. Initialize and Conformalize with ResidualNormalisedScore ---

confidence_level = 1- alpha 

mapie_calibrator = SplitConformalRegressor(
    estimator=PrecalculatedPredictor(),
    confidence_level=confidence_level,
    prefit=True,
    # 🔥 CRITICAL: Use the score for feature-dependent CP
    conformity_score=ResidualNormalisedScore(sym=True) 
)

print("Starting Calibration (Feature-Dependent)...")

mapie_calibrator.conformalize(
    X_conformalize=X_cal, 
    y_conformalize=y_cal
)

# --- 4. Generate Prediction Intervals ---

# The predict_interval call implicitly uses X_test[:, 0] for ŷ 
# and X_test[:, 1] for σ, as defined by the ResidualNormalisedScore logic.
y_pred, y_pis = mapie_calibrator.predict_interval(
    X=X_test 
)

# --- 5. Display Results ---

print(f"Calibration Complete. Normalized residual threshold Q_norm learned.")
print("\n" + "="*70)
print(f"   MAPIE Normalized CP Intervals ({100 * confidence_level:.0f}% Confidence)")
print("="*70)

results = pd.DataFrame({
    'Point_Prediction': y_pred.flatten(),
    'Sigma_Prediction': X_test[:, 1], # Display the local sigma
    f'Lower_Bound': y_pis[:, 0, 0],
    f'Upper_Bound': y_pis[:, 1, 0]
})

results['Interval_Width'] = results['Upper_Bound'] - results['Lower_Bound']
print(results.round(3))