"""
L3 Apply Model to 2026 Tournament
Generates Elite 8 predictions for the upcoming 2026 tournament
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration
INPUT_FILE = Path('data/predictionData/predict_set_2025.csv')
MODEL_DIR = Path('outputs/03_ensemble_models')
OUTPUT_DIR = Path('outputs/05_2026_predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which trained model to use
USE_DATASET = 'long'  # or 'rich' - try both and compare

print("="*80)
print("2026 TOURNAMENT - ELITE 8 PREDICTIONS")
print("="*80)

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================
print("\n[1] LOADING TRAINED ENSEMBLE MODEL")
print("-" * 80)

model_file = MODEL_DIR / f'trained_ensemble_{USE_DATASET}.pkl'

with open(model_file, 'rb') as f:
    model_package = pickle.load(f)

models = model_package['models']
scaler = model_package['scaler']
feature_list = model_package['features']
calibrated_gnb = model_package['calibrated_gnb']

print(f"Loaded {USE_DATASET} dataset model")
print(f"Required features: {len(feature_list)}")

# Ensemble weights (equal weighting performed best in testing)
ensemble_weights = np.array([0.25, 0.25, 0.25, 0.25])

# ============================================================================
# LOAD 2026 PREDICTION DATA
# ============================================================================
print("\n[2] LOADING 2026 TOURNAMENT DATA")
print("-" * 80)

# Load the prediction dataset
data_2026 = pd.read_csv(INPUT_FILE)

print(f"Loaded data: {len(data_2026)} teams")
print(f"Columns available: {len(data_2026.columns)}")

# Show sample
print("\nSample of data:")
print(data_2026.head(3))

# Check for required columns
if 'Team' not in data_2026.columns or 'Year' not in data_2026.columns:
    print("\nERROR: Data must have 'Team' and 'Year' columns")
    exit(1)

# ============================================================================
# HANDLE MISSING TOURNAMENT SEED
# ============================================================================
print("\n[3] HANDLING MISSING TOURNAMENT SEED")
print("-" * 80)

# Check if tournamentSeed is in feature list
has_seed_feature = 'tournamentSeed' in feature_list

if has_seed_feature:
    print("tournamentSeed IS required by the model")
    
    if 'tournamentSeed' not in data_2026.columns or data_2026['tournamentSeed'].isnull().all():
        print("  WARNING: tournamentSeed not available in prediction data")
        print("  Will estimate seeds based on team strength metrics")
        
        # Create estimated seeds based on composite ranking
        # Use available metrics to estimate likely seeding
        if 'kenpom_NetRtg' in data_2026.columns:
            # Rank teams by KenPom and assign seeds 1-16 (approximately)
            data_2026['tournamentSeed'] = pd.qcut(
                data_2026['kenpom_NetRtg'].rank(ascending=False, method='first'),
                q=4,
                labels=[1, 2, 3, 4]
            ).astype(float)
            print("  Created estimated seeds based on kenpom_NetRtg (1-4 range)")
        elif 'BPI' in data_2026.columns:
            data_2026['tournamentSeed'] = pd.qcut(
                data_2026['BPI'].rank(ascending=True, method='first'),
                q=4,
                labels=[1, 2, 3, 4]
            ).astype(float)
            print("  Created estimated seeds based on BPI (1-4 range)")
        else:
            # Last resort: assign median seed
            data_2026['tournamentSeed'] = 4.0
            print("  Using median seed (4.0) for all teams")
else:
    print("tournamentSeed is NOT in the feature list - no action needed")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n[4] PREPARING FEATURES")
print("-" * 80)

# Check which features are available
available_features = [f for f in feature_list if f in data_2026.columns]
missing_features = [f for f in feature_list if f not in data_2026.columns]

print(f"Available features: {len(available_features)}/{len(feature_list)}")

if missing_features:
    print(f"\nWARNING: Missing {len(missing_features)} features:")
    for feat in missing_features[:10]:  # Show first 10
        print(f"  - {feat}")
    if len(missing_features) > 10:
        print(f"  ... and {len(missing_features) - 10} more")
    
    print("\nOptions:")
    print("  1. These features will be filled with 0 (may hurt performance)")
    print("  2. Check if feature names match between training and prediction data")
    print("  3. Re-run L2 feature join to ensure all sources are included")

# Extract features - use all available, fill missing with 0
X = pd.DataFrame(index=data_2026.index)
for feat in feature_list:
    if feat in data_2026.columns:
        X[feat] = data_2026[feat]
    else:
        X[feat] = 0.0  # Fill missing features with 0

# Keep team info separate
team_info = data_2026[['Year', 'Team']].copy()
if 'Index' in data_2026.columns:
    team_info['Index'] = data_2026['Index']
if 'tournamentSeed' in data_2026.columns:
    team_info['Estimated_Seed'] = data_2026['tournamentSeed']

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n[5] HANDLING MISSING VALUES")
print("-" * 80)

# Drop entirely NaN columns
all_nan_cols = X.columns[X.isnull().all()].tolist()
if all_nan_cols:
    print(f"Dropping {len(all_nan_cols)} all-NaN columns")
    X = X.drop(columns=all_nan_cols)

# Fill remaining NaNs with median (same as training)
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"Filling {missing_count} missing values with medians")
    
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col].fillna(0, inplace=True)
            else:
                X[col].fillna(median_val, inplace=True)

# Verify no NaNs remain
assert X.isnull().sum().sum() == 0, "NaNs still present after filling!"

print(f"Final feature matrix: {X.shape[0]} teams × {X.shape[1]} features")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n[6] GENERATING ELITE 8 PREDICTIONS")
print("-" * 80)

# Scale features
X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns=X.columns,
    index=X.index
)

# Get predictions from each model
print("Running ensemble models...")

pred_lr = models['Logistic Regression'].predict_proba(X_scaled)[:, 1]
pred_rf = models['Random Forest'].predict_proba(X)[:, 1]
pred_svm = models['SVM'].predict_proba(X_scaled)[:, 1]
pred_gnb = calibrated_gnb.predict_proba(X)[:, 1]

# Create ensemble
pred_stack = np.column_stack([pred_lr, pred_rf, pred_svm, pred_gnb])
ensemble_pred = np.average(pred_stack, axis=1, weights=ensemble_weights)
ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)

print(f"Generated predictions for {len(ensemble_pred)} teams")

# ============================================================================
# CREATE RESULTS
# ============================================================================
print("\n[7] CREATING RESULTS")
print("-" * 80)

results = pd.DataFrame({
    'Team': team_info['Team'].values,
    'Elite8_Probability': ensemble_pred,
    'Logistic_Regression': pred_lr,
    'Random_Forest': pred_rf,
    'SVM': pred_svm,
    'Gaussian_NB_Calibrated': pred_gnb,
})

# Add seed if available
if 'Estimated_Seed' in team_info.columns:
    results['Estimated_Seed'] = team_info['Estimated_Seed'].values

# Sort by probability (descending)
results = results.sort_values('Elite8_Probability', ascending=False)

# Add rank
results.insert(0, 'Rank', range(1, len(results) + 1))

# Format probabilities as percentages for display
results_display = results.copy()
for col in ['Elite8_Probability', 'Logistic_Regression', 'Random_Forest', 'SVM', 'Gaussian_NB_Calibrated']:
    results_display[col] = results_display[col].apply(lambda x: f"{x:.1%}")

# ============================================================================
# DISPLAY TOP PREDICTIONS
# ============================================================================
print("\n[8] TOP 20 ELITE 8 PREDICTIONS FOR 2026")
print("="*80)

top_20 = results_display.head(20)
print(top_20.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[9] SAVING RESULTS")
print("-" * 80)

# Save full results (with numeric probabilities)
results.to_csv(OUTPUT_DIR / f'elite8_predictions_2026_{USE_DATASET}.csv', index=False)
print(f"Saved full predictions: elite8_predictions_2026_{USE_DATASET}.csv")

# Save top 30 for easy reference
top_30 = results.head(30)
top_30.to_csv(OUTPUT_DIR / f'top_30_elite8_2026_{USE_DATASET}.csv', index=False)
print(f"Saved top 30: top_30_elite8_2026_{USE_DATASET}.csv")

# Create a simple summary
print("\n[10] TIER BREAKDOWN")
print("-" * 80)

# Group by probability ranges
def get_tier(prob):
    if prob >= 0.60:
        return "Elite (60%+)"
    elif prob >= 0.45:
        return "Strong (45-60%)"
    elif prob >= 0.30:
        return "Moderate (30-45%)"
    elif prob >= 0.15:
        return "Long Shot (15-30%)"
    else:
        return "Very Unlikely (<15%)"

results['Tier'] = results['Elite8_Probability'].apply(get_tier)

tier_summary = results.groupby('Tier').size().reindex([
    "Elite (60%+)",
    "Strong (45-60%)",
    "Moderate (30-45%)",
    "Long Shot (15-30%)",
    "Very Unlikely (<15%)"
])

print("\nTeams by Elite 8 probability tier:")
print(tier_summary)

# Show teams in each tier
for tier in ["Elite (60%+)", "Strong (45-60%)"]:
    tier_teams = results[results['Tier'] == tier]
    if len(tier_teams) > 0:
        print(f"\n{tier}:")
        for _, row in tier_teams.iterrows():
            print(f"  {row['Rank']:2d}. {row['Team']:20s} {row['Elite8_Probability']:.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("2026 PREDICTIONS COMPLETE")
print("="*80)

print(f"\nTotal teams analyzed: {len(results)}")
print(f"Teams with >50% Elite 8 probability: {len(results[results['Elite8_Probability'] > 0.50])}")
print(f"Teams with >40% Elite 8 probability: {len(results[results['Elite8_Probability'] > 0.40])}")

print("\nOUTPUT FILES:")
print(f"  {OUTPUT_DIR / f'elite8_predictions_2026_{USE_DATASET}.csv'}")
print(f"  {OUTPUT_DIR / f'top_30_elite8_2026_{USE_DATASET}.csv'}")

print("\nNEXT STEPS:")
print("  1. Review top predictions - do they pass the eye test?")
print("  2. Wait for actual tournament seeding (mid-March)")
print("  3. Re-run with real seeds for final predictions")
print("  4. Compare 'long' vs 'rich' model predictions")
print("  5. Use probabilities for bracket/pool strategy")

print("\n" + "="*80)