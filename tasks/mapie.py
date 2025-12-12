import os.path
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsRegressor
from tasks.descriptors.ecfp import init_cache, smiles_to_ecfp_cached


# + tags=["parameters"]
input_folder = None
data = None
alpha = 0.1
cache_path = None
# -

conn = init_cache(cache_path)

# Lock down all randomness sources
np.random.seed(42)
import random
random.seed(42)
import os
os.environ['PYTHONHASHSEED'] = '0'


# --- 1. Simple Index-Based Predictor ---
class PrecomputedPredictor(RegressorMixin):
    """Returns precomputed predictions by index lookup."""
    def __init__(self, y_pred_cal):
        self.y_pred_cal = y_pred_cal
        self.is_fitted_ = True

    def get_params(self, deep=True):
        return {'y_pred_cal': self.y_pred_cal}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        check_is_fitted(self)
        indices = X.flatten().astype(int)
        return self.y_pred_cal[indices]


# --- 2. ECFP-Translating Sigma Model ---
class ECFPSigmaWrapper(RegressorMixin):
    """
    Wraps sigma model to translate indices -> ECFP before prediction.
    This is what gets passed to ResidualNormalisedScore.
    """
    def __init__(self, sigma_model, X_ecfp_cal):
        self.sigma_model = sigma_model
        self.X_ecfp_cal = X_ecfp_cal
        self.is_fitted_ = True

    def get_params(self, deep=True):
        return {
            'sigma_model': self.sigma_model,
            'X_ecfp_cal': self.X_ecfp_cal
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        # X is indices during conformalize
        indices = X.flatten().astype(int)
        X_ecfp = self.X_ecfp_cal[indices]
        self.sigma_model.fit(X_ecfp, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        # X is indices, translate to ECFP
        indices = X.flatten().astype(int)
        X_ecfp = self.X_ecfp_cal[indices]
        return self.sigma_model.predict(X_ecfp)


# --- 3. Load Data ---
input_file = os.path.join(input_folder, f"{data}.xlsx")
calibration_df = pd.read_excel(input_file, sheet_name=data)

N_CAL = calibration_df.shape[0]
y_cal = calibration_df["Exp"].values
y_pred_cal = calibration_df[data].values

print("Computing ECFP features...")
X_ecfp_cal = np.array([smiles_to_ecfp_cached(sm) for sm in calibration_df["Smiles"]])

# Diagnostic: Check ECFP quality
print(f"DEBUG: ECFP shape: {X_ecfp_cal.shape}")
print(f"DEBUG: ECFP dtype: {X_ecfp_cal.dtype}")
ecfp_sums = X_ecfp_cal.sum(axis=1)
print(f"DEBUG: Non-zero ECFP count: {(ecfp_sums > 0).sum()}/{len(ecfp_sums)}")
print(f"DEBUG: ECFP sums [0:5]: {ecfp_sums[:5]}")
print(f"DEBUG: First 5 SMILES: {calibration_df['Smiles'].values[:5].tolist()}")

if (ecfp_sums == 0).any():
    zero_indices = np.where(ecfp_sums == 0)[0]
    print(f"WARNING: {len(zero_indices)} molecules have all-zero ECFP!")
    print(f"  First zero ECFP index: {zero_indices[0]}, SMILES: {calibration_df['Smiles'].values[zero_indices[0]]}")

# Indices for both models
X_indices_cal = np.arange(N_CAL).reshape(-1, 1)

# Calculate residuals
residuals_cal = np.abs(y_cal - y_pred_cal)


# --- 4. Train Sigma Model on ECFP ---
base_sigma_model = KNeighborsRegressor(
    n_neighbors=5, 
    weights="distance", 
    p=2,
    algorithm='brute',  # More deterministic than 'auto' or 'kd_tree'
    metric='euclidean'  # Explicit metric
)
print("Training Sigma Model (KNN on ECFP features)...")
base_sigma_model.fit(X_ecfp_cal, residuals_cal)

# Verify determinism
test_sigma_1 = base_sigma_model.predict(X_ecfp_cal[:5])
test_sigma_2 = base_sigma_model.predict(X_ecfp_cal[:5])
print(f"DEBUG: Sigma model deterministic? {np.allclose(test_sigma_1, test_sigma_2)}")

# Wrap it to handle indices
sigma_wrapper = ECFPSigmaWrapper(base_sigma_model, X_ecfp_cal)


# --- 5. Setup MAPIE ---
confidence_level = 1 - alpha

conformity_score = ResidualNormalisedScore(
    residual_estimator=sigma_wrapper,
    sym=True
)

mapie_calibrator = SplitConformalRegressor(
    estimator=PrecomputedPredictor(y_pred_cal),
    confidence_level=confidence_level,
    prefit=True,
    conformity_score=conformity_score
)

print("Starting Calibration...")

# Debug: check if data order is consistent
print(f"DEBUG: First ECFP sum: {X_ecfp_cal[0].sum():.6f}")
print(f"DEBUG: First y_pred: {y_pred_cal[0]:.6f}")

mapie_calibrator.conformalize(
    X_conformalize=X_indices_cal,
    y_conformalize=y_cal
)

# Check what quantile was learned
if hasattr(mapie_calibrator, 'quantiles_'):
    print(f"DEBUG: Learned quantile: {mapie_calibrator.quantiles_}")
if hasattr(mapie_calibrator, 'conformity_scores_'):
    print(f"DEBUG: Conformity scores shape: {mapie_calibrator.conformity_scores_.shape}")
    print(f"DEBUG: Conformity scores [0:5]: {mapie_calibrator.conformity_scores_[:5]}")


# --- 6. Test ---
N_TEST = 10
X_indices_test = np.arange(N_TEST).reshape(-1, 1)
y_true_test = calibration_df["Exp"].values[:N_TEST]  # For coverage check

y_pred, y_pis = mapie_calibrator.predict_interval(X=X_indices_test)

# Get sigma for display
sigma_pred_test = base_sigma_model.predict(X_ecfp_cal[:N_TEST])


# --- 7. Results ---
print(f"\nCalibration Complete.")
print("\n" + "="*70)
print(f"   MAPIE Normalized CP Intervals ({100 * confidence_level:.0f}% Confidence)")
print("="*70)


results = pd.DataFrame({
    'True_Value': y_true_test,
    'Point_Prediction': y_pred.flatten(),
    'Sigma_Prediction': sigma_pred_test,
    'Lower_Bound': y_pis[:, 0, 0],
    'Upper_Bound': y_pis[:, 1, 0]
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