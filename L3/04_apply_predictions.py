"""
L3 Apply Predictions
Applies trained ensemble model to generate Elite 8 predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration
INPUT_DIR = Path('outputs/01_feature_selection')
MODEL_DIR = Path('outputs/03_ensemble_models')
OUTPUT_DIR = Path('outputs/04_predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which model to use
USE_DATASET = 'long'  # or 'rich'

print("="*80)
print("L3 APPLY PREDICTIONS - ELITE 8 FORECASTS")
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
calibrated_gnb = model_package['calibrated_gnb']

print(f"Loaded {USE_DATASET} dataset model")
print(f"Required features: {len(feature_list)}")
print(f"Models in ensemble: {list(models.keys())}")

# Get ensemble weights (ROC-AUC weighted)
# Based on our results: roughly equal weighting performed best
ensemble_weights = np.array([0.25, 0.25, 0.25, 0.25])

# ============================================================================
# LOAD TEST DATA TO DEMONSTRATE
# ============================================================================
print("\n[2] LOADING TEST DATA (2023-2025 TOURNAMENTS)")
print("-" * 80)

# Load the labeled data
labeled_data = pd.read_csv(INPUT_DIR / f'labeled_training_{USE_DATASET}.csv')

# Filter to test years only (2023-2025)
test_data = labeled_data[labeled_data['Year'] > 2022].copy()

print(f"Test data: {len(test_data)} teams from 2023-2025 tournaments")
print(f"Years included: {sorted(test_data['Year'].unique())}")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n[3] PREPARING DATA FOR PREDICTION")
print("-" * 80)

# Extract features
X = test_data[feature_list].copy()
teams_info = test_data[['Year', 'Team', 'Index', 'tournamentSeed', 'elite8_flag']].copy()

# Handle missing values (same as training)
all_nan_cols = X.columns[X.isnull().all()].tolist()
if all_nan_cols:
    X = X.drop(columns=all_nan_cols)
    print(f"Dropped {len(all_nan_cols)} all-NaN columns")

for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        if pd.isna(median_val):
            X[col].fillna(0, inplace=True)
        else:
            X[col].fillna(median_val, inplace=True)

print(f"Prepared {len(X)} teams with {len(X.columns)} features")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n[4] GENERATING ELITE 8 PREDICTIONS")
print("-" * 80)

# Scale data
X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns=X.columns,
    index=X.index
)

# Get predictions from each model
print("Generating predictions from individual models...")

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
# CREATE RESULTS DATAFRAME
# ============================================================================
print("\n[5] CREATING RESULTS")
print("-" * 80)

results = pd.DataFrame({
    'Year': teams_info['Year'].values,
    'Team': teams_info['Team'].values,
    'Seed': teams_info['tournamentSeed'].values,
    'Actual_Elite8': teams_info['elite8_flag'].values,
    'Predicted_Probability': ensemble_pred,
    'LR_Prob': pred_lr,
    'RF_Prob': pred_rf,
    'SVM_Prob': pred_svm,
    'GNB_Prob': pred_gnb
})

# Add binary prediction (using 12% threshold - base rate)
results['Predicted_Elite8'] = (results['Predicted_Probability'] >= 0.12).astype(int)

# Sort by probability (descending)
results = results.sort_values('Predicted_Probability', ascending=False)

# Save full results
results.to_csv(OUTPUT_DIR / f'all_predictions_{USE_DATASET}.csv', index=False)
print(f"Saved all predictions to {OUTPUT_DIR / f'all_predictions_{USE_DATASET}.csv'}")

# ============================================================================
# DISPLAY TOP PREDICTIONS
# ============================================================================
print("\n[6] TOP ELITE 8 PREDICTIONS")
print("-" * 80)

# Show top 20 teams by predicted probability
print("\nTOP 20 TEAMS BY ELITE 8 PROBABILITY:")
print("-" * 80)
top_20 = results.head(20)[['Year', 'Team', 'Seed', 'Predicted_Probability', 'Actual_Elite8']]
top_20['Predicted_Probability'] = top_20['Predicted_Probability'].apply(lambda x: f"{x:.3f}")
top_20['Actual_Elite8'] = top_20['Actual_Elite8'].map({1: 'YES ✓', 0: 'No'})
print(top_20.to_string(index=False))

# ============================================================================
# BREAKDOWN BY YEAR
# ============================================================================
print("\n[7] PREDICTIONS BY YEAR")
print("-" * 80)

for year in sorted(results['Year'].unique()):
    year_data = results[results['Year'] == year].copy()
    
    # Get top 8 predictions for this year
    top_8 = year_data.head(8)
    
    # Count how many actually made Elite 8
    predicted_correct = top_8['Actual_Elite8'].sum()
    
    print(f"\n{year} TOURNAMENT - Top 8 Predicted Teams:")
    print("-" * 60)
    
    display_cols = top_8[['Team', 'Seed', 'Predicted_Probability', 'Actual_Elite8']].copy()
    display_cols['Predicted_Probability'] = display_cols['Predicted_Probability'].apply(lambda x: f"{x:.1%}")
    display_cols['Actual_Elite8'] = display_cols['Actual_Elite8'].map({1: 'YES ✓', 0: 'No'})
    print(display_cols.to_string(index=False))
    
    print(f"\nAccuracy: {predicted_correct}/8 predicted teams actually made Elite 8")
    
    # Show which Elite 8 teams we missed
    actual_elite8 = year_data[year_data['Actual_Elite8'] == 1].copy()
    missed = actual_elite8[~actual_elite8['Team'].isin(top_8['Team'])]
    
    if len(missed) > 0:
        print(f"\nMissed Elite 8 teams (not in our top 8):")
        for _, row in missed.iterrows():
            print(f"  {row['Team']} (Seed {row['Seed']}) - Predicted: {row['Predicted_Probability']:.1%}")

# ============================================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================================
print("\n[8] OVERALL PERFORMANCE SUMMARY")
print("-" * 80)

# Calculate accuracy if we picked top N teams
for n in [8, 12, 16]:
    top_n_teams = set(results.head(n)['Team'])
    actual_elite8_teams = set(results[results['Actual_Elite8'] == 1]['Team'])
    
    correct = len(top_n_teams & actual_elite8_teams)
    total_elite8 = len(actual_elite8_teams)
    
    print(f"\nIf we picked TOP {n} teams overall:")
    print(f"  Would capture {correct}/{total_elite8} actual Elite 8 teams ({correct/total_elite8*100:.1f}%)")

# Save just the top predictions for easy reference
top_predictions = results.head(30)[['Year', 'Team', 'Seed', 'Predicted_Probability', 'Actual_Elite8']]
top_predictions.to_csv(OUTPUT_DIR / f'top_30_predictions_{USE_DATASET}.csv', index=False)
print(f"\nSaved top 30 predictions to {OUTPUT_DIR / f'top_30_predictions_{USE_DATASET}.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREDICTIONS COMPLETE")
print("="*80)

print("\nFILES CREATED:")
print(f"  {OUTPUT_DIR / f'all_predictions_{USE_DATASET}.csv'} - All predictions with probabilities")
print(f"  {OUTPUT_DIR / f'top_30_predictions_{USE_DATASET}.csv'} - Top 30 teams")

print("\nKEY METRICS:")
total_teams = len(results)
total_elite8 = results['Actual_Elite8'].sum()
avg_prob_elite8 = results[results['Actual_Elite8'] == 1]['Predicted_Probability'].mean()
avg_prob_non_elite8 = results[results['Actual_Elite8'] == 0]['Predicted_Probability'].mean()

print(f"  Total teams evaluated: {total_teams}")
print(f"  Actual Elite 8 teams: {total_elite8}")
print(f"  Avg probability for Elite 8 teams: {avg_prob_elite8:.1%}")
print(f"  Avg probability for non-Elite 8 teams: {avg_prob_non_elite8:.1%}")
print(f"  Separation: {avg_prob_elite8 - avg_prob_non_elite8:.1%}")

print("\n" + "="*80)