"""
L3 Apply Model to 2026 Tournament
Configure via config.py: USE_SEEDS = True/False
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
# LOAD BOTH TRAINED MODELS (for seed checking)
# ============================================================================
print("\n[2] LOADING TRAINED ENSEMBLE MODELS (initial check)")
print("-" * 80)

# Quick load to check if seeds are needed
temp_models = {}
for dataset in ['long', 'rich']:
    model_file = MODEL_DIR / f'trained_ensemble_{dataset}_production.pkl'
    with open(model_file, 'rb') as f:
        model_package = pickle.load(f)
    temp_models[dataset] = model_package['features']
    print(f"Loaded {dataset} model - {len(model_package['features'])} features")

# ============================================================================
# HANDLE MISSING TOURNAMENT SEED
# ============================================================================
print("\n[3] HANDLING MISSING TOURNAMENT SEED")
print("-" * 80)

# Check if tournamentSeed exists in either feature list
needs_seed = 'tournamentSeed' in temp_models['long'] or 'tournamentSeed' in temp_models['rich']

if needs_seed:
    print("tournamentSeed IS required by model(s)")
    
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
# GENERATE PREDICTIONS FROM BOTH MODELS
# ============================================================================
print("\n[4] GENERATING PREDICTIONS FROM BOTH MODELS")
print("-" * 80)

predictions = {}

for dataset in ['long', 'rich']:
    # Load the model package for this dataset
    model_file = MODEL_DIR / f'trained_ensemble_{dataset}_production.pkl'
    with open(model_file, 'rb') as f:
        model_package = pickle.load(f)
    
    print(f"\n{dataset.upper()} model:")
    
    # Extract components
    feature_list = model_package['features']
    models = model_package['models']
    scaler = model_package['scaler']
    calibrated_gnb = model_package['calibrated_gnb']
    best_weights = model_package['best_weights']
    best_strategy = model_package['best_strategy']
    
    print(f"  Strategy: {best_strategy}")
    print(f"  Weights: {dict(zip(['LR', 'RF', 'SVM', 'GNB'], np.round(best_weights, 3)))}")
    
    # Prepare features - TRY TO USE SCALER'S FEATURE NAMES IF AVAILABLE
    try:
        scaler_features = scaler.feature_names_in_
        print(f"  Using scaler's feature list: {len(scaler_features)} features")
        actual_features = scaler_features
    except AttributeError:
        print(f"  Using model package feature list: {len(feature_list)} features")
        actual_features = feature_list
    
    available_features = [f for f in actual_features if f in data_2026.columns]
    missing_features = [f for f in actual_features if f not in data_2026.columns]
    
    print(f"  Available: {len(available_features)}/{len(actual_features)}")
    
    if missing_features:
        print(f"  Missing {len(missing_features)} features (will fill with 0):")
        for feat in missing_features[:5]:  # Show first 5
            print(f"    - {feat}")
        if len(missing_features) > 5:
            print(f"    ... and {len(missing_features) - 5} more")
    
    # Extract features - USE EXACT FEATURE ORDER FROM SCALER/MODEL
    X = pd.DataFrame(index=data_2026.index)
    for feat in actual_features:
        if feat in data_2026.columns:
            X[feat] = data_2026[feat].copy()
        else:
            X[feat] = 0.0
    
    print(f"  Extracted features shape: {X.shape}")
    
    # Replace infinite values FIRST
    inf_count = np.isinf(X.values).sum()
    if inf_count > 0:
        print(f"  Replacing {inf_count} infinite values with NaN")
        X = X.replace([np.inf, -np.inf], np.nan)
    
    # Check for all-NaN columns
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"  WARNING: {len(all_nan_cols)} columns are entirely NaN:")
        for col in all_nan_cols:
            print(f"    - {col}")
        print(f"  Dropping these columns")
        X = X.drop(columns=all_nan_cols)
    
    # Fill remaining NaNs
    nan_cols_before = X.columns[X.isnull().any()].tolist()
    if nan_cols_before:
        print(f"  Filling NaNs in {len(nan_cols_before)} columns")
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
    
    # Final verification BEFORE scaling
    if X.isnull().any().any():
        print("  ⚠️ ERROR: NaNs still present before scaling!")
        nan_cols = X.columns[X.isnull().any()].tolist()
        print(f"  Columns with NaNs: {nan_cols}")
        for col in nan_cols:
            print(f"    {col}: {X[col].isnull().sum()} NaNs")
        # Force fill
        X = X.fillna(0)
        print("  Forced all remaining NaNs to 0")
    
    print(f"  Pre-scale: {X.shape[1]} columns, {X.shape[0]} rows, no NaNs ✓")
    
    # Verify feature count matches scaler
    try:
        expected_features = scaler.n_features_in_
        if X.shape[1] != expected_features:
            print(f"  ❌ FEATURE MISMATCH: Scaler expects {expected_features}, got {X.shape[1]}")
            print(f"  This may cause errors!")
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
        print(f"  ❌ ERROR during scaling: {e}")
        print(f"  Attempting to diagnose...")
        print(f"  X shape: {X.shape}")
        print(f"  X dtypes: {X.dtypes.value_counts()}")
        raise
    
    # Check for NaNs AFTER scaling
    if X_scaled.isnull().any().any():
        print("  ⚠️ ERROR: NaNs introduced during scaling!")
        nan_cols_scaled = X_scaled.columns[X_scaled.isnull().any()].tolist()
        print(f"  Columns with NaNs after scaling: {nan_cols_scaled}")
        for col in nan_cols_scaled:
            nan_count = X_scaled[col].isnull().sum()
            print(f"    {col}: {nan_count} NaNs")
            # Show pre-scale values
            if col in X.columns:
                print(f"      Pre-scale: min={X[col].min():.3f}, max={X[col].max():.3f}")
        # Force fill
        X_scaled = X_scaled.fillna(0)
        print("  Fixed by filling with 0")
    
    print(f"  Post-scale: {X_scaled.shape[1]} columns, no NaNs ✓")
    
    # Get predictions from each model
    try:
        pred_lr = models['Logistic Regression'].predict_proba(X_scaled)[:, 1]
        pred_rf = models['Random Forest'].predict_proba(X)[:, 1]
        pred_svm = models['SVM'].predict_proba(X_scaled)[:, 1]
        pred_gnb = calibrated_gnb.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"  ❌ ERROR during prediction: {e}")
        print(f"  X_scaled has NaNs: {X_scaled.isnull().any().any()}")
        print(f"  X has NaNs: {X.isnull().any().any()}")
        raise
    
    # Create ensemble using OPTIMAL WEIGHTS from model package
    pred_stack = np.column_stack([pred_lr, pred_rf, pred_svm, pred_gnb])
    ensemble_pred = np.average(pred_stack, axis=1, weights=best_weights)
    ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
    
    predictions[dataset] = {
        'ensemble': ensemble_pred,
        'lr': pred_lr,
        'rf': pred_rf,
        'svm': pred_svm,
        'gnb': pred_gnb
    }
    
    print(f"  ✓ Generated predictions for {len(ensemble_pred)} teams")

# ============================================================================
# CREATE COMPARISON RESULTS
# ============================================================================
print("\n[5] CREATING COMPARISON RESULTS")
print("-" * 80)

results = pd.DataFrame({
    'Team': team_info['Team'].values,
    'Long_Probability': predictions['long']['ensemble'],
    'Rich_Probability': predictions['rich']['ensemble'],
})

# Add seed if available
if 'Estimated_Seed' in team_info.columns:
    results['Estimated_Seed'] = team_info['Estimated_Seed'].values

# Calculate agreement metrics
results['Avg_Probability'] = (results['Long_Probability'] + results['Rich_Probability']) / 2
results['Difference'] = abs(results['Long_Probability'] - results['Rich_Probability'])
results['Agreement'] = results['Difference'] < 0.10  # Within 10 percentage points

# Sort by average probability
results = results.sort_values('Avg_Probability', ascending=False)

# Add rank
results.insert(0, 'Rank', range(1, len(results) + 1))

# ============================================================================
# DISPLAY COMPARISON
# ============================================================================
print("\n[6] MODEL COMPARISON - TOP 25 PREDICTIONS")
print("="*80)

# Format for display
display_results = results.head(25).copy()
display_results['Long_Probability'] = display_results['Long_Probability'].apply(lambda x: f"{x:.1%}")
display_results['Rich_Probability'] = display_results['Rich_Probability'].apply(lambda x: f"{x:.1%}")
display_results['Avg_Probability'] = display_results['Avg_Probability'].apply(lambda x: f"{x:.1%}")
display_results['Difference'] = display_results['Difference'].apply(lambda x: f"{x:.1%}")
display_results['Agreement'] = display_results['Agreement'].map({True: '✓', False: '✗'})

print(display_results[['Rank', 'Team', 'Long_Probability', 'Rich_Probability', 
                       'Avg_Probability', 'Difference', 'Agreement']].to_string(index=False))

# ============================================================================
# SHOW DISAGREEMENTS
# ============================================================================
print("\n[7] BIGGEST DISAGREEMENTS BETWEEN MODELS")
print("-" * 80)

# Find biggest disagreements
disagreements = results[results['Difference'] > 0.15].sort_values('Difference', ascending=False).head(10)

if len(disagreements) > 0:
    print(f"\nTop {len(disagreements)} teams where models disagree (>15% difference):")
    print()
    
    for _, row in disagreements.iterrows():
        print(f"{row['Team']:20s} Long: {row['Long_Probability']:.1%}  |  Rich: {row['Rich_Probability']:.1%}  |  Diff: {row['Difference']:.1%}")
else:
    print("\nNo major disagreements - models are highly aligned!")

# ============================================================================
# CONSENSUS PREDICTIONS
# ============================================================================
print("\n[8] HIGH-CONFIDENCE CONSENSUS PICKS")
print("-" * 80)

# Teams where BOTH models agree on high probability
high_consensus = results[
    (results['Long_Probability'] > 0.50) & 
    (results['Rich_Probability'] > 0.50) &
    (results['Agreement'] == True)
].head(15)

print(f"\nTeams with >50% probability from BOTH models:")
print()

for _, row in high_consensus.iterrows():
    print(f"{row['Rank']:2d}. {row['Team']:20s} Long: {row['Long_Probability']:.1%}  Rich: {row['Rich_Probability']:.1%}  Avg: {row['Avg_Probability']:.1%}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[9] SAVING RESULTS")
print("-" * 80)

# Save full comparison
results.to_csv(OUTPUT_DIR / 'elite8_predictions_2026_comparison.csv', index=False)
print(f"Saved full comparison: elite8_predictions_2026_comparison.csv")

# Save individual model predictions
for dataset in ['long', 'rich']:
    model_results = pd.DataFrame({
        'Team': team_info['Team'].values,
        'Elite8_Probability': predictions[dataset]['ensemble'],
        'Logistic_Regression': predictions[dataset]['lr'],
        'Random_Forest': predictions[dataset]['rf'],
        'SVM': predictions[dataset]['svm'],
        'Gaussian_NB_Calibrated': predictions[dataset]['gnb']
    })
    
    if 'Estimated_Seed' in team_info.columns:
        model_results['Estimated_Seed'] = team_info['Estimated_Seed'].values
    
    model_results = model_results.sort_values('Elite8_Probability', ascending=False)
    model_results.insert(0, 'Rank', range(1, len(model_results) + 1))
    
    model_results.to_csv(OUTPUT_DIR / f'elite8_predictions_2026_{dataset}.csv', index=False)
    print(f"Saved {dataset} predictions: elite8_predictions_2026_{dataset}.csv")

# Save top 30 consensus picks
top_30 = results.head(30)[['Rank', 'Team', 'Long_Probability', 'Rich_Probability', 
                            'Avg_Probability', 'Difference', 'Agreement']]
if 'Estimated_Seed' in results.columns:
    top_30.insert(2, 'Estimated_Seed', results.head(30)['Estimated_Seed'])

top_30.to_csv(OUTPUT_DIR / 'top_30_elite8_2026_consensus.csv', index=False)
print(f"Saved top 30 consensus: top_30_elite8_2026_consensus.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[10] SUMMARY STATISTICS")
print("-" * 80)

print(f"\nTotal teams analyzed: {len(results)}")
print(f"Teams in high consensus (both >50%): {len(high_consensus)}")
print(f"Teams with strong agreement (diff <10%): {results['Agreement'].sum()}")
print(f"Average difference between models: {results['Difference'].mean():.1%}")
print(f"Max difference between models: {results['Difference'].max():.1%}")

# Tier breakdown
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

results['Avg_Tier'] = results['Avg_Probability'].apply(get_tier)

tier_summary = results.groupby('Avg_Tier').size().reindex([
    "Elite (60%+)",
    "Strong (45-60%)",
    "Moderate (30-45%)",
    "Long Shot (15-30%)",
    "Very Unlikely (<15%)"
], fill_value=0)

print("\nTeams by probability tier (using average):")
for tier, count in tier_summary.items():
    print(f"  {tier:25s} {int(count):3d} teams")

# Show correlation between models
correlation = results['Long_Probability'].corr(results['Rich_Probability'])
print(f"\nCorrelation between Long and Rich predictions: {correlation:.3f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("2026 PREDICTIONS COMPLETE")
print("="*80)

print("\nMODEL INFO:")
print("  ✓ Using PRODUCTION models (trained on ≤2025 data)")
print("  ✓ Using optimal ensemble weights from validation")
print("  ✓ Expected performance: ~0.87-0.90 ROC-AUC (based on backtesting)")

print("\nOUTPUT FILES:")
print(f"  {OUTPUT_DIR / 'elite8_predictions_2026_comparison.csv'} - Side-by-side comparison")
print(f"  {OUTPUT_DIR / 'elite8_predictions_2026_long.csv'} - Long model predictions")
print(f"  {OUTPUT_DIR / 'elite8_predictions_2026_rich.csv'} - Rich model predictions")
print(f"  {OUTPUT_DIR / 'top_30_elite8_2026_consensus.csv'} - Top 30 consensus picks")

print("\nRECOMMENDATIONS:")
if correlation > 0.95:
    print("  ✓ Models are highly aligned - trust the consensus predictions")
elif correlation > 0.85:
    print("  ✓ Models generally agree - use average probability for best estimate")
else:
    print("  ⚠ Models show meaningful differences - investigate disagreements")

print("\nNEXT STEPS:")
print("  1. Review consensus top picks - do they pass the eye test?")
print("  2. Investigate any major disagreements between models")
print("  3. Run 06_tournament_type_indicator.py to estimate chalk vs chaos")
print("  4. Wait for actual tournament seeding (Selection Sunday)")
print("  5. Re-run with real seeds for final predictions")
print("  6. Use probabilities for bracket/pool strategy")

print("\n" + "="*80)