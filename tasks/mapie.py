import os.path
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from tasks.descriptors.ecfp import init_cache
from tasks.descriptors.ecfp import smiles_to_ecfp_cached


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
# -

# Validate parameters
if input_folder is None or data is None:
    raise ValueError("Must specify input_folder and data parameters")

conn = init_cache(cache_path)
np.random.seed(42)

# --- 1. Define the Robust Proxy Model with ECFP Lookup ---
class PrecalculatedPredictor(RegressorMixin):
    """
    Proxy model that looks up the pre-calculated point prediction (ŷ) 
    by matching the input ECFP features (X) against a stored map.
    This is necessary because MAPIE MUST pass ECFP to the sigma model.
    """
    def __init__(self, X_cal_features, y_pred_cal):
        self.is_fitted_ = True
        # Create a dictionary map: ECFP_tuple -> y_pred_value (for fast lookup)
        self.prediction_map = {
            tuple(row): y_pred_cal[i] 
            for i, row in enumerate(X_cal_features)
        }
        
        # Check for hash collisions
        if len(self.prediction_map) != len(X_cal_features):
            n_duplicates = len(X_cal_features) - len(self.prediction_map)
            print(f"⚠️  WARNING: {n_duplicates} duplicate ECFP fingerprints found in calibration set!")

    def fit(self, X, y):
        # Required by scikit-learn API
        return self

    def predict(self, X):
        check_is_fitted(self)
        
        y_out = np.empty(X.shape[0])
        
        # Look up the pre-calculated prediction for each row of ECFP features (X)
        for i, row in enumerate(X):
            key = tuple(row)
            if key not in self.prediction_map:
                raise KeyError(
                    f"Sample {i} not found in calibration set. "
                    f"ECFP lookup failed - this should not happen for test samples from cal set."
                )
            y_out[i] = self.prediction_map[key]
        
        return y_out.flatten() 

# --- 2. Prepare data and Train Sigma Model ---
input_file = os.path.join(input_folder, f"{data}.xlsx")
calibration_df = pd.read_excel(input_file, sheet_name=data)

N_CAL = calibration_df.shape[0]
y_cal = calibration_df["Exp"].values
y_pred_cal = calibration_df[data].values

print(f"Loaded calibration data: {N_CAL} samples")

# 🔥 ECFP features are the official X input for MAPIE
X_ecfp_cal = np.array([smiles_to_ecfp_cached(sm) for sm in calibration_df["Smiles"]])
X_cal = X_ecfp_cal

# 🔥 STEP 1: Calculate absolute residuals (the target for the sigma model)
residuals_cal = np.abs(y_cal - y_pred_cal)

# 🔥 STEP 2: Train the Uncertainty Model (Sigma Model) using ECFP features
sigma_model = KNeighborsRegressor(n_neighbors=5, weights="distance", p=1)
print("Training Sigma Model (KNN Regressor)...")
sigma_model.fit(X_cal, residuals_cal)

print(f"\nCalibration Diagnostics:")
print(f"  Mean absolute residual: {np.mean(residuals_cal):.3f}")
print(f"  Std absolute residual: {np.std(residuals_cal):.3f}")
print(f"  Mean predicted sigma: {sigma_model.predict(X_cal).mean():.3f}")

# --- Prepare Test Data (first N_TEST samples for pipeline testing) ---
N_TEST = 10
print(f"\n⚠️  Using first {N_TEST} calibration samples for pipeline testing")
print("   (Load separate test file for real evaluation)")

X_ecfp_test = np.array([smiles_to_ecfp_cached(sm) for sm in calibration_df["Smiles"][:N_TEST]])
X_test = X_ecfp_test # ECFP features for test set
y_true_test = calibration_df["Exp"].values[:N_TEST]  # For coverage check

# Predict sigma for the test data (for display only)
sigma_pred_test = sigma_model.predict(X_test)


# --- 3. Initialize and Conformalize with ResidualNormalisedScore ---
confidence_level = 1 - alpha 

# 1. Fully configured conformity score with the trained sigma model
conformity_score = ResidualNormalisedScore(
    residual_estimator=sigma_model,  # The sigma model trained on ECFP
    sym=True 
)

# 2. SplitConformalRegressor wrapper
mapie_calibrator = SplitConformalRegressor(
    # Primary Estimator (The lookup model, initialized with ECFP data)
    estimator=PrecalculatedPredictor(X_ecfp_cal, y_pred_cal),
    confidence_level=confidence_level,
    prefit=True, 
    conformity_score=conformity_score
)

print(f"\nStarting Calibration (Target coverage: {confidence_level:.1%})...")

# MAPIE calls both predictors (PrecalculatedPredictor and sigma_model) with X_cal
mapie_calibrator.conformalize(
    X_conformalize=X_cal, 
    y_conformalize=y_cal
)


# --- 4. Generate Prediction Intervals ---
# MAPIE calls both predictors with X_test
y_pred, y_pis = mapie_calibrator.predict_interval(
    X=X_test 
)

# --- 5. Display Results ---

print(f"\n✓ Calibration Complete. Normalized conformity threshold learned.")
print("\n" + "="*70)
print(f"   MAPIE Normalized CP Intervals ({100 * confidence_level:.0f}% Confidence)")
print("="*70)

results = pd.DataFrame({
    'True_Value': y_true_test,
    'Point_Prediction': y_pred.flatten(),
    'Sigma_Prediction': sigma_pred_test, 
    f'Lower_Bound': y_pis[:, 0, 0],
    f'Upper_Bound': y_pis[:, 1, 0]
})

results['Interval_Width'] = results['Upper_Bound'] - results['Lower_Bound']
results['In_Interval'] = (
    (results['True_Value'] >= results['Lower_Bound']) & 
    (results['True_Value'] <= results['Upper_Bound'])
)

print(results.round(3))

# Calculate coverage
coverage = results['In_Interval'].mean()
print(f"\n{'='*70}")
print(f"Pipeline Test Coverage: {coverage:.1%} (Target: {confidence_level:.1%})")
print(f"  ✓ {results['In_Interval'].sum()}/{N_TEST} samples in interval")
print(f"Mean Interval Width: {results['Interval_Width'].mean():.3f}")
print(f"Std Interval Width: {results['Interval_Width'].std():.3f}")
print(f"{'='*70}")

print("\n✓ Pipeline test complete. Ready to load separate test data.")