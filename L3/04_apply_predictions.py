"""
L3 Apply Predictions
Applies trained ensemble model to new tournament data (e.g., 2026 predictions)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration
MODEL_DIR = Path('outputs/03_ensemble_models')
OUTPUT_DIR = Path('outputs/04_predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which model to use
USE_DATASET = 'long'  # or 'rich'

print("="*80)
print("L3 APPLY PREDICTIONS TO NEW TOURNAMENT DATA")
print("="*80)

# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================
print("\n[1] LOADING TRAINED ENSEMBLE MODEL")
print("-" * 80)

model_file = MODEL_DIR / f'trained_ensemble_{USE_DATASET}.pkl'

with open(model_file, 'rb') as f:
    model_package = pickle.load(f)

models = model_package['models']
scaler = model_package['scaler']
feature_list = model_package['features']
ensemble_weights = model_package['ensemble_weights']['ROC-AUC Weighted']
calibrated_gnb = model_package['calibrated_gnb']

print(f"Loaded {USE_DATASET} dataset model")
print(f"Required features: {len(feature_list)}")
print(f"Models in ensemble: {list(models.keys())}")

# ============================================================================
# LOAD NEW DATA TO PREDICT
# ============================================================================
print("\n[2] LOADING NEW DATA FOR PREDICTION")
print("-" * 80)

# TODO: User provides path to their joined 2026 tournament data
# This should be in the same format as training data (same features)
# For now, using a placeholder - user will replace this

# Option 1: User has already joined 2026 data
# new_data = pd.read_csv('path/to/2026_tournament_teams.csv')

# Option 2: We'll create a template showing what's needed
print("TEMPLATE: Your input data should have these columns:")
print("  Required: Year, Team, Index")
print(f"  Features: {feature_list[:5]}... (and {len(feature_list)-5} more)")
print("\nExample structure:")
print("  Year,Team,Index,bartTorvik_WAB,kenpom_NetRtg,BPI,tournamentSeed,...")
print("  2026,Duke,45,0.95,28.5,32.1,1,...")

# For demonstration, let's show how it works with test data
# In production, user would load their actual 2026 data here
print("\n[DEMO MODE - Using 2024 test data as example]")
print("Replace this section with your actual 2026 tournament data")

# Load the test predictions we already made as an example
demo_data = pd.read_csv(MODEL_DIR / f'best_ensemble_predictions_{USE_DATASET}.csv')
print(f"\nDemo data loaded: {len(demo_data)} teams")
print(demo_data.head())

# ============================================================================
# PREPARE DATA FOR PREDICTION
# ============================================================================
print("\n[3] PREPARING DATA FOR PREDICTION")
print("-" * 80)

def prepare_new_data(df, feature_list, scaler):
    """Prepare new data in same format as training data"""
    
    # This is where user's actual data preparation would happen
    # For now, we're just demonstrating the structure
    
    print(f"Input data: {len(df)} teams")
    print(f"Required features: {len(feature_list)}")
    
    # User would need to:
    # 1. Ensure all required features are present
    # 2. Handle missing values same way as training
    # 3. Scale features using the saved scaler
    
    # For demo, we'll use the existing predictions
    return df

# ============================================================================
# APPLY ENSEMBLE MODEL
# ============================================================================
print("\n[4] GENERATING ELITE 8 PREDICTIONS")
print("-" * 80)

def predict_ensemble(X, models, scaler, calibrated_gnb, weights):
    """Apply ensemble model to new data"""
    
    # This is the actual prediction code
    # User would implement this with their real data
    
    predictions = {}
    
    # Logistic Regression (scaled)
    X_scaled = scaler.transform(X)
    predictions['Logistic Regression'] = models['Logistic Regression'].predict_proba(X_scaled)[:, 1]
    
    # Random Forest (unscaled)
    predictions['Random Forest'] = models['Random Forest'].predict_proba(X)[:, 1]
    
    # SVM (scaled)
    predictions['SVM'] = models['SVM'].predict_proba(X_scaled)[:, 1]
    
    # Gaussian NB (calibrated)
    predictions['Gaussian NB (Calibrated)'] = calibrated_gnb.predict_proba(X)[:, 1]
    
    # Create ensemble
    pred_stack = np.column_stack([
        predictions['Logistic Regression'],
        predictions['Random Forest'],
        predictions['SVM'],
        predictions['Gaussian NB (Calibrated)']
    ])
    
    ensemble_pred = np.average(pred_stack, axis=1, weights=weights)
    ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
    
    return ensemble_pred, predictions

# For demo purposes, we already have predictions
# In production, user would call predict_ensemble with their 2026 data

# ============================================================================
# CREATE OUTPUT
# ============================================================================
print("\n[5] CREATING OUTPUT")
print("-" * 80)

# Create output dataframe
# User would do this with their 2026 data + new predictions

output_template = """
# This is what your output will look like:

Team,Year,tournamentSeed,elite8_probability,predicted_elite8
Duke,2026,1,0.645,Yes
UConn,2026,2,0.523,Yes
Purdue,2026,1,0.487,No
Houston,2026,2,0.441,No
...

Where:
- elite8_probability: Model's probability this team makes Elite 8
- predicted_elite8: Binary prediction (Yes if probability >= 0.12, which is base rate)

Teams would be sorted by elite8_probability descending.
"""

print(output_template)

# ============================================================================
# INSTRUCTIONS FOR USER
# ============================================================================
print("\n" + "="*80)
print("INSTRUCTIONS TO USE THIS SCRIPT WITH YOUR 2026 DATA")
print("="*80)

print("""
STEP 1: Prepare your 2026 tournament data
  - Join all your L2 data sources for 2026 tournament teams
  - Must include ALL features that were in training data
  - Should have columns: Year, Team, Index, [all features]
  
STEP 2: Modify this script
  - Replace the demo data loading with your actual 2026 data
  - Ensure feature names match exactly
  - Handle missing values same way as training (median fill)
  
STEP 3: Run prediction
  - Script will apply the trained ensemble model
  - Output predictions for each 2026 tournament team
  - Results saved to: outputs/04_predictions/elite8_predictions_2026.csv

EXAMPLE CODE TO ADD:

# Load your 2026 joined data
new_data_2026 = pd.read_csv('data/trainingData/training_set_2026.csv')

# Extract features
X_new = new_data_2026[feature_list].copy()

# Fill missing values (same as training)
for col in X_new.columns:
    if X_new[col].isnull().any():
        median_val = X_new[col].median()
        if pd.isna(median_val):
            X_new[col].fillna(0, inplace=True)
        else:
            X_new[col].fillna(median_val, inplace=True)

# Generate predictions
ensemble_probs, individual_probs = predict_ensemble(
    X_new, models, scaler, calibrated_gnb, ensemble_weights
)

# Create output
results_2026 = pd.DataFrame({
    'Team': new_data_2026['Team'],
    'Year': new_data_2026['Year'],
    'tournamentSeed': new_data_2026['tournamentSeed'],
    'elite8_probability': ensemble_probs,
    'predicted_elite8': (ensemble_probs >= 0.12).astype(str).replace({'True': 'Yes', 'False': 'No'})
})

# Sort by probability
results_2026 = results_2026.sort_values('elite8_probability', ascending=False)

# Save
results_2026.to_csv('outputs/04_predictions/elite8_predictions_2026.csv', index=False)
print("Predictions saved!")
""")

print("\n" + "="*80)