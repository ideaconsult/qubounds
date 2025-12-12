# + tags=["parameters"]
# add default values for parameters here
# -

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

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

# --- 2. Prepare Simulated Data with Auxillary Info (Sigma) ---

N_CAL = 100
N_TEST = 10
np.random.seed(123)

y_cal = 10 + 2 * np.random.randn(N_CAL)  # True values

# Simulate the two pieces of information we have for each point:
y_pred_cal = y_cal + np.random.randn(N_CAL) # ŷ (Point Prediction)
# Simulate the pre-calculated local standard deviation (our 'sigma' prediction)
sigma_pred_cal = 0.5 + 0.1 * np.random.rand(N_CAL) 

# Combine them into the single input matrix X (Features now have 2 columns!)
# X = [ [ŷ1, σ1], [ŷ2, σ2], ... ]
X_cal = np.vstack([y_pred_cal, sigma_pred_cal]).T 

# Simulate Test Data
y_pred_test = y_pred_cal[:N_TEST] 
sigma_pred_test = sigma_pred_cal[:N_TEST]
X_test = np.vstack([y_pred_test, sigma_pred_test]).T

# --- 3. Initialize and Conformalize with ResidualNormalisedScore ---

confidence_level = 0.90 

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