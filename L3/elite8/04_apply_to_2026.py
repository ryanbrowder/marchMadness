"""
L3 Apply Model to 2026 Tournament
Configure via config.py: USE_SEEDS = True/False

UNIFIED DATASET (March 2026):
- Single trained model (no long/rich comparison)
- Generates Elite 8 probabilities for all 2026 tournament teams
- Outputs: Top picks, probability distribution, full predictions CSV
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configuration
INPUT_FILE = config.PREDICT_DATA_FILE
MODEL_DIR = config.OUTPUT_02
OUTPUT_DIR = config.OUTPUT_04
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("2026 TOURNAMENT - ELITE 8 PREDICTIONS")
config.print_config()
print("="*80)

# ============================================================================
# LOAD 2026 PREDICTION DATA
# ============================================================================
print("\n[1] LOADING 2026 TOURNAMENT DATA")
print("-" * 80)

data_2026 = pd.read_csv(INPUT_FILE)

print(f"Loaded data: {len(data_2026)} teams")
print(f"Columns available: {len(data_2026.columns)}")

# Check for required columns
if 'Team' not in data_2026.columns or 'Year' not in data_2026.columns:
    print("\nERROR: Data must have 'Team' and 'Year' columns")
    exit(1)

# Extract team info IMMEDIATELY (so it's always available)
team_info = data_2026[['Year', 'Team']].copy()
if 'Index' in data_2026.columns:
    team_info['Index'] = data_2026['Index']

print(f"\nTeam info extracted for {len(team_info)} teams")

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================
print("\n[2] LOADING TRAINED ENSEMBLE MODEL")
print("-" * 80)

# Load the unified model
model_file = MODEL_DIR / 'trained_ensemble_unified_production.pkl'
with open(model_file, 'rb') as f:
    model_package = pickle.load(f)

feature_list = model_package['features']
print(f"Loaded unified model - {len(feature_list)} features")

# ============================================================================
# HANDLE MISSING TOURNAMENT SEED
# ============================================================================
print("\n[3] HANDLING MISSING TOURNAMENT SEED")
print("-" * 80)

# Check if tournamentSeed is required
needs_seed = 'tournamentSeed' in feature_list

if needs_seed:
    print("tournamentSeed IS required by model")
    
    if 'tournamentSeed' not in data_2026.columns or data_2026['tournamentSeed'].isnull().all():
        print("  WARNING: tournamentSeed not available in prediction data")
        print("  Estimating seeds based on team strength metrics")
        
        # Create estimated seeds
        if 'kenpom_NetRtg' in data_2026.columns:
            data_2026['tournamentSeed'] = pd.qcut(
                data_2026['kenpom_NetRtg'].rank(ascending=False, method='first'),
                q=4,
                labels=[1, 2, 3, 4],
                duplicates='drop'
            ).astype(float)
            print("  Created estimated seeds based on kenpom_NetRtg (1-4 range)")
        elif 'BPI' in data_2026.columns:
            data_2026['tournamentSeed'] = pd.qcut(
                data_2026['BPI'].rank(ascending=True, method='first'),
                q=4,
                labels=[1, 2, 3, 4],
                duplicates='drop'
            ).astype(float)
            print("  Created estimated seeds based on BPI (1-4 range)")
        else:
            data_2026['tournamentSeed'] = 4.0
            print("  Using median seed (4.0) for all teams")
    else:
        print(f"  ✓ Using actual tournament seeds from data")
    
    # Add seed to team_info
    team_info['Estimated_Seed'] = data_2026['tournamentSeed']
else:
    print("tournamentSeed is NOT required - no action needed")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n[4] GENERATING PREDICTIONS")
print("-" * 80)

# Extract components
models = model_package['models']
scaler = model_package['scaler']
calibrated_gnb = model_package['gnb_calibrated']
best_weights = model_package['ensemble_weights']
best_strategy = model_package['ensemble_strategy']

print(f"Strategy: {best_strategy}")
print(f"Weights: {dict(zip(['LR', 'RF', 'SVM', 'GNB'], np.round(best_weights, 3)))}")

# Prepare features - TRY TO USE SCALER'S FEATURE NAMES IF AVAILABLE
try:
    scaler_features = scaler.feature_names_in_
    print(f"Using scaler's feature list: {len(scaler_features)} features")
    actual_features = scaler_features
except AttributeError:
    print(f"Using model package feature list: {len(feature_list)} features")
    actual_features = feature_list

available_features = [f for f in actual_features if f in data_2026.columns]
missing_features = [f for f in actual_features if f not in data_2026.columns]

print(f"Available: {len(available_features)}/{len(actual_features)}")

if missing_features:
    print(f"Missing {len(missing_features)} features (will fill with 0):")
    for feat in missing_features[:5]:  # Show first 5
        print(f"  - {feat}")
    if len(missing_features) > 5:
        print(f"  ... and {len(missing_features) - 5} more")

# Extract features - USE EXACT FEATURE ORDER FROM SCALER/MODEL
X = pd.DataFrame(index=data_2026.index)
for feat in actual_features:
    if feat in data_2026.columns:
        X[feat] = data_2026[feat].copy()
    else:
        X[feat] = 0.0

print(f"Extracted features shape: {X.shape}")

# Replace infinite values FIRST
inf_count = np.isinf(X.values).sum()
if inf_count > 0:
    print(f"Replacing {inf_count} infinite values with NaN")
    X = X.replace([np.inf, -np.inf], np.nan)

# Check for all-NaN columns
all_nan_cols = X.columns[X.isnull().all()].tolist()
if all_nan_cols:
    print(f"WARNING: {len(all_nan_cols)} columns are entirely NaN:")
    for col in all_nan_cols:
        print(f"  - {col}")
    print(f"Dropping these columns")
    X = X.drop(columns=all_nan_cols)

# Fill remaining NaNs
nan_cols_before = X.columns[X.isnull().any()].tolist()
if nan_cols_before:
    print(f"Filling NaNs in {len(nan_cols_before)} columns")
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)

# Final verification BEFORE scaling
if X.isnull().any().any():
    print("⚠️ ERROR: NaNs still present before scaling!")
    nan_cols = X.columns[X.isnull().any()].tolist()
    print(f"Columns with NaNs: {nan_cols}")
    for col in nan_cols:
        print(f"  {col}: {X[col].isnull().sum()} NaNs")
    # Force fill
    X = X.fillna(0)
    print("Forced all remaining NaNs to 0")

print(f"Pre-scale: {X.shape[1]} columns, {X.shape[0]} rows, no NaNs ✓")

# Verify feature count matches scaler
try:
    expected_features = scaler.n_features_in_
    if X.shape[1] != expected_features:
        print(f"❌ FEATURE MISMATCH: Scaler expects {expected_features}, got {X.shape[1]}")
        print(f"This may cause errors!")
except AttributeError:
    pass

# Scale features
try:
    X_scaled_array = scaler.transform(X)
    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X.columns,
        index=X.index
    )
except Exception as e:
    print(f"❌ ERROR during scaling: {e}")
    print(f"Attempting to diagnose...")
    print(f"X shape: {X.shape}")
    print(f"X dtypes: {X.dtypes.value_counts()}")
    raise

# Check for NaNs AFTER scaling
if X_scaled.isnull().any().any():
    print("⚠️ ERROR: NaNs introduced during scaling!")
    nan_cols_scaled = X_scaled.columns[X_scaled.isnull().any()].tolist()
    print(f"Columns with NaNs after scaling: {nan_cols_scaled}")
    for col in nan_cols_scaled:
        nan_count = X_scaled[col].isnull().sum()
        print(f"  {col}: {nan_count} NaNs")
        # Show pre-scale values
        if col in X.columns:
            print(f"    Pre-scale: min={X[col].min():.3f}, max={X[col].max():.3f}")
    # Force fill
    X_scaled = X_scaled.fillna(0)
    print("Fixed by filling with 0")

print(f"Post-scale: {X_scaled.shape[1]} columns, no NaNs ✓")

# Get predictions from each model
try:
    pred_lr = models['Logistic Regression'].predict_proba(X_scaled)[:, 1]
    pred_rf = models['Random Forest'].predict_proba(X)[:, 1]
    pred_svm = models['SVM'].predict_proba(X_scaled)[:, 1]
    pred_gnb = calibrated_gnb.predict_proba(X)[:, 1]
except Exception as e:
    print(f"❌ ERROR during prediction: {e}")
    print(f"X_scaled has NaNs: {X_scaled.isnull().any().any()}")
    print(f"X has NaNs: {X.isnull().any().any()}")
    raise

# Create ensemble using OPTIMAL WEIGHTS from model package
pred_stack = np.column_stack([pred_lr, pred_rf, pred_svm, pred_gnb])
ensemble_pred = np.average(pred_stack, axis=1, weights=best_weights)
ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)

print(f"✓ Generated predictions for {len(ensemble_pred)} teams")

# ============================================================================
# CREATE RESULTS DATAFRAME
# ============================================================================
print("\n[5] CREATING RESULTS")
print("-" * 80)

results = pd.DataFrame({
    'Team': team_info['Team'].values,
    'P_E8': ensemble_pred,
    'P_E8_LR': pred_lr,
    'P_E8_RF': pred_rf,
    'P_E8_SVM': pred_svm,
    'P_E8_GNB': pred_gnb
})

# Add index if available
if 'Index' in team_info.columns:
    results['Index'] = team_info['Index'].values

# Add seed if available
if 'Estimated_Seed' in team_info.columns:
    results['Estimated_Seed'] = team_info['Estimated_Seed'].values

# Sort by probability
results = results.sort_values('P_E8', ascending=False)

# Add rank
results.insert(0, 'Rank', range(1, len(results) + 1))

# ============================================================================
# DISPLAY TOP PREDICTIONS
# ============================================================================
print("\n[6] TOP 25 ELITE 8 PREDICTIONS")
print("="*80)

# Format for display
display_results = results.head(25).copy()

# Show key columns
display_cols = ['Rank', 'Team', 'P_E8']
if 'Estimated_Seed' in display_results.columns:
    display_cols.append('Estimated_Seed')

display_df = display_results[display_cols].copy()
display_df['P_E8'] = display_df['P_E8'].apply(lambda x: f"{x:.1%}")

print(display_df.to_string(index=False))

# ============================================================================
# TIER ANALYSIS
# ============================================================================
print("\n[7] PROBABILITY TIERS")
print("-" * 80)

# Define tiers
tiers = [
    ('Elite (>60%)', results['P_E8'] > 0.60),
    ('Strong (40-60%)', (results['P_E8'] >= 0.40) & (results['P_E8'] <= 0.60)),
    ('Moderate (25-40%)', (results['P_E8'] >= 0.25) & (results['P_E8'] < 0.40)),
    ('Long Shot (10-25%)', (results['P_E8'] >= 0.10) & (results['P_E8'] < 0.25)),
    ('Very Low (<10%)', results['P_E8'] < 0.10)
]

for tier_name, tier_mask in tiers:
    tier_teams = results[tier_mask]
    print(f"\n{tier_name}: {len(tier_teams)} teams")
    if len(tier_teams) > 0 and len(tier_teams) <= 15:
        for _, row in tier_teams.iterrows():
            print(f"  {row['Team']:25s} {row['P_E8']:.1%}")

# ============================================================================
# MODEL COMPONENT ANALYSIS
# ============================================================================
print("\n[8] MODEL COMPONENT ANALYSIS (Top 15)")
print("-" * 80)

component_analysis = results.head(15)[['Rank', 'Team', 'P_E8', 'P_E8_LR', 'P_E8_RF', 'P_E8_SVM', 'P_E8_GNB']].copy()

# Format probabilities
for col in ['P_E8', 'P_E8_LR', 'P_E8_RF', 'P_E8_SVM', 'P_E8_GNB']:
    component_analysis[col] = component_analysis[col].apply(lambda x: f"{x:.1%}")

print(component_analysis.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[9] SAVING RESULTS")
print("-" * 80)

# Save full predictions
output_file = OUTPUT_DIR / 'elite8_predictions_2026.csv'
results.to_csv(output_file, index=False)
print(f"Saved full predictions: {output_file.name}")

# Save top 25
output_file_top = OUTPUT_DIR / 'elite8_predictions_2026_top25.csv'
results.head(25).to_csv(output_file_top, index=False)
print(f"Saved top 25: {output_file_top.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("2026 PREDICTIONS COMPLETE")
print("="*80)

print("\nKEY STATISTICS:")
print(f"  Total teams predicted: {len(results)}")
print(f"  Average probability: {results['P_E8'].mean():.1%}")
print(f"  Highest probability: {results['P_E8'].max():.1%} ({results.iloc[0]['Team']})")
print(f"  Teams >50%: {(results['P_E8'] > 0.50).sum()}")
print(f"  Teams >25%: {(results['P_E8'] > 0.25).sum()}")

print("\nTOP 8 PICKS (Most Likely Elite 8):")
for i, row in results.head(8).iterrows():
    seed_str = f" (Seed {row['Estimated_Seed']:.0f})" if 'Estimated_Seed' in row else ""
    print(f"  {row['Rank']}. {row['Team']:25s} {row['P_E8']:.1%}{seed_str}")

print("\nOUTPUTS:")
print(f"  {OUTPUT_DIR}/elite8_predictions_2026.csv (all teams)")
print(f"  {OUTPUT_DIR}/elite8_predictions_2026_top25.csv (top 25)")

if config.USE_SEEDS:
    print("\n✓ Predictions include tournament seed information")
else:
    print("\n✓ Predictions based on pure team metrics (no seeds)")

print("\nNEXT STEP:")
print("  → Proceed to 05_tournament_type_indicator.py for chalk score")

print("\n" + "="*80)
