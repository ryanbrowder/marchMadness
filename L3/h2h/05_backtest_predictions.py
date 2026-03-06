"""
H2H 05_backtest_predictions.py

Purpose: Backtest and compare 5 prediction methods on historical NCAA tournaments.
         Evaluates performance across all games (2008-2025) to benchmark model quality.

Prediction Methods:
    1. Seeds - Pick higher seed (lower seed number = better)
    2. PowerRank - Pick team with higher PowerRank rating
    3. bartTorvik_Barthag - Pick team with higher Barthag
    4. kenpom_NetRtg - Pick team with higher NetRtg
    5. H2H Model - Use trained NO SEEDS production model

Inputs:
    - L2/data/srcbb/srcbb_analyze_L2.csv (tournament results)
    - L3/data/trainingData/training_set_unified.csv (team features)
    - L3/h2h/models/ (H2H NO SEEDS production model)

Outputs:
    - outputs/05_backtest_predictions/accuracy_by_season.csv
    - outputs/05_backtest_predictions/accuracy_by_round.csv
    - outputs/05_backtest_predictions/overall_accuracy.csv
    - Console summary with comparison table

Author: Ryan Browder
Date: 2026-03-06
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='sklearn')

# ============================================================================
# Configuration
# ============================================================================

# Input paths
TOURNAMENT_RESULTS_PATH = '../../L2/data/srcbb/srcbb_analyze_L2.csv'
TRAINING_FEATURES_PATH = '../data/trainingData/training_set_unified.csv'
MODEL_DIR = 'models/'

# Feature columns to use from selected_features.csv
SELECTED_FEATURES_PATH = 'outputs/02_feature_correlation/selected_features.csv'

# Output directory
OUTPUT_DIR = 'outputs/05_backtest_predictions'

# ============================================================================
# Helper Functions
# ============================================================================

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}/")

def load_data():
    """Load tournament results and team features."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load tournament results
    print(f"\nLoading tournament results from: {TOURNAMENT_RESULTS_PATH}")
    results = pd.read_csv(TOURNAMENT_RESULTS_PATH)
    print(f"  Shape: {results.shape}")
    print(f"  Years: {results['Year'].min()}-{results['Year'].max()}")
    print(f"  Games: {len(results)}")
    
    # Load team features
    print(f"\nLoading team features from: {TRAINING_FEATURES_PATH}")
    features = pd.read_csv(TRAINING_FEATURES_PATH)
    print(f"  Shape: {features.shape}")
    print(f"  Years: {features['Year'].min()}-{features['Year'].max()}")
    print(f"  Teams: {features['Index'].nunique()} unique")
    
    return results, features

def load_h2h_model():
    """Load H2H NO SEEDS production model."""
    print("\n" + "="*80)
    print("LOADING H2H MODEL")
    print("="*80)
    
    print(f"\nLoading models from: {MODEL_DIR}")
    
    # Load ensemble config
    config_path = os.path.join(MODEL_DIR, 'ensemble_config.json')
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
    
    # Load feature scaler
    scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load models
    models = {}
    for model_name in ['random_forest', 'gradient_boosting', 'neural_network', 
                       'gaussian_naive_bayes', 'svm']:
        model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    
    print(f"  ✓ Loaded {len(models)} models")
    print(f"  ✓ Feature scaler loaded")
    print(f"  ✓ Ensemble config loaded")
    
    return models, scaler, config

def calculate_pct_diff(val_a, val_b):
    """Calculate symmetric percentage difference."""
    avg = (val_a + val_b) / 2.0
    if avg == 0 or np.isnan(avg):
        return 0.0
    return (val_a - val_b) / avg

def get_h2h_prediction(teamA_features, teamB_features, feature_cols, models, scaler, config, debug_first=False):
    """
    Generate H2H model prediction for a single matchup.
    
    Returns probability that TeamA wins.
    """
    try:
        # Calculate percentage differentials
        pct_diffs = []
        for col in feature_cols:
            val_a = teamA_features.get(col, 0)
            val_b = teamB_features.get(col, 0)
            
            # Handle missing values
            if pd.isna(val_a) or pd.isna(val_b):
                pct_diffs.append(0.0)
            else:
                pct_diffs.append(calculate_pct_diff(val_a, val_b))
        
        if debug_first:
            print(f"\n[DEBUG] First prediction:")
            print(f"  feature_cols length: {len(feature_cols)}")
            print(f"  pct_diffs length: {len(pct_diffs)}")
            print(f"  First 3 pct_diffs: {pct_diffs[:3]}")
        
        # Scale features
        X = np.array([pct_diffs])
        X_scaled = scaler.transform(X)
        
        if debug_first:
            print(f"  X shape: {X.shape}")
            print(f"  X_scaled shape: {X_scaled.shape}")
        
        # Get predictions from each model with error handling
        predictions = {}
        
        # Tree-based models (use raw features)
        try:
            predictions['random_forest'] = models['random_forest'].predict_proba(X)[0][1]
            if debug_first:
                print(f"  ✓ RF: {predictions['random_forest']:.3f}")
        except Exception as e:
            if debug_first:
                print(f"  ✗ RF failed: {e}")
            
        try:
            predictions['gradient_boosting'] = models['gradient_boosting'].predict_proba(X)[0][1]
            if debug_first:
                print(f"  ✓ GB: {predictions['gradient_boosting']:.3f}")
        except Exception as e:
            if debug_first:
                print(f"  ✗ GB failed: {e}")
        
        # Models that need scaled features
        try:
            pred = models['neural_network'].predict_proba(X_scaled)[0][1]
            if not np.isnan(pred) and not np.isinf(pred):
                predictions['neural_network'] = pred
                if debug_first:
                    print(f"  ✓ NN: {pred:.3f}")
        except Exception as e:
            if debug_first:
                print(f"  ✗ NN failed: {e}")
        
        # GNB is problematic - catch its errors
        try:
            pred = models['gaussian_naive_bayes'].predict_proba(X_scaled)[0][1]
            if not np.isnan(pred) and not np.isinf(pred):
                predictions['gaussian_naive_bayes'] = pred
                if debug_first:
                    print(f"  ✓ GNB: {pred:.3f}")
        except Exception as e:
            if debug_first:
                print(f"  ✗ GNB failed: {e}")
        
        try:
            pred = models['svm'].predict_proba(X_scaled)[0][1]
            if not np.isnan(pred) and not np.isinf(pred):
                predictions['svm'] = pred
                if debug_first:
                    print(f"  ✓ SVM: {pred:.3f}")
        except Exception as e:
            if debug_first:
                print(f"  ✗ SVM failed: {e}")
        
        if debug_first:
            print(f"  Valid predictions: {len(predictions)}")
        
        # Need at least one valid prediction
        if len(predictions) == 0:
            if debug_first:
                print(f"  ✗ No valid predictions!")
            return None
        
        # Check for problematic predictions and exclude
        valid_predictions = {}
        for model_name, prob in predictions.items():
            if 0.001 <= prob <= 0.999:  # Not extreme
                valid_predictions[model_name] = prob
        
        # Need at least one valid non-extreme prediction
        if len(valid_predictions) == 0:
            # All predictions are extreme, try using fallback weights if available
            if 'fallback_weights' in config and len(config['fallback_weights']) > 0:
                fallback = config['fallback_weights'][0]
                # Use fallback weights with available predictions
                fallback_valid = {m: predictions[m] for m in fallback['weights'].keys() 
                                if m in predictions and 0.001 <= predictions[m] <= 0.999}
                if len(fallback_valid) > 0:
                    total_weight = sum(fallback['weights'][m] for m in fallback_valid.keys())
                    if total_weight > 0:
                        ensemble_prob = sum(fallback_valid[m] * fallback['weights'][m] 
                                          for m in fallback_valid.keys()) / total_weight
                        return ensemble_prob
            # Still no valid predictions
            if debug_first:
                print(f"  ✗ No valid non-extreme predictions!")
            return None
        
        # Map model names (lowercase_underscore) to config keys (Title Case)
        name_map = {
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'neural_network': 'Neural Network',
            'gaussian_naive_bayes': 'Gaussian Naive Bayes',
            'svm': 'SVM'
        }
        
        # Ensemble with original weights, excluding problematic models
        weights = config['weights']
        total_weight = sum(weights.get(name_map.get(m, m), 0) for m in valid_predictions.keys())
        
        if debug_first:
            print(f"  Total weight: {total_weight}")
        
        if total_weight == 0:
            if debug_first:
                print(f"  ✗ Total weight is 0!")
            return None
        
        # Weighted average
        ensemble_prob = sum(valid_predictions[m] * weights.get(name_map.get(m, m), 0) 
                           for m in valid_predictions.keys()) / total_weight
        
        if debug_first:
            print(f"  ✓ Ensemble prob: {ensemble_prob:.3f}")
        
        return ensemble_prob
        
    except Exception as e:
        # Something went wrong, return None
        return None

def join_features_to_results(results_df, features_df):
    """Join team features to tournament results."""
    print("\n" + "="*80)
    print("JOINING FEATURES TO RESULTS")
    print("="*80)
    
    # Get feature columns (everything except Year, Index, Team)
    feature_cols = [c for c in features_df.columns if c not in ['Year', 'Index', 'Team']]
    
    # Prepare features for TeamA - rename BEFORE merge
    features_a = features_df.copy()
    rename_dict_a = {col: f'TeamA_{col}' for col in feature_cols}
    features_a = features_a.rename(columns={'Index': 'TeamA_ID', **rename_dict_a})
    
    # Prepare features for TeamB - rename BEFORE merge
    features_b = features_df.copy()
    rename_dict_b = {col: f'TeamB_{col}' for col in feature_cols}
    features_b = features_b.rename(columns={'Index': 'TeamB_ID', **rename_dict_b})
    
    # Get renamed column names
    teamA_feature_cols = [f'TeamA_{col}' for col in feature_cols]
    teamB_feature_cols = [f'TeamB_{col}' for col in feature_cols]
    
    # Join TeamA features
    print("\nJoining TeamA features...")
    df = results_df.merge(
        features_a[['Year', 'TeamA_ID'] + teamA_feature_cols],
        on=['Year', 'TeamA_ID'],
        how='left'
    )
    
    teamA_missing = df[teamA_feature_cols].isnull().any(axis=1).sum()
    print(f"  Games with missing TeamA features: {teamA_missing}")
    
    # Join TeamB features
    print("Joining TeamB features...")
    df = df.merge(
        features_b[['Year', 'TeamB_ID'] + teamB_feature_cols],
        on=['Year', 'TeamB_ID'],
        how='left'
    )
    
    teamB_missing = df[teamB_feature_cols].isnull().any(axis=1).sum()
    print(f"  Games with missing TeamB features: {teamB_missing}")
    
    print(f"\n✓ Feature join complete: {len(df)} games")
    
    return df

def make_predictions(df, feature_cols, models, scaler, config):
    """Generate predictions using all 5 methods."""
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    print(f"\n[DEBUG] make_predictions called:")
    print(f"  df shape: {df.shape}")
    print(f"  feature_cols length: {len(feature_cols)}")
    print(f"  First 5 feature_cols: {feature_cols[:5]}")
    print(f"  Checking if TeamA columns exist in df:")
    for col in feature_cols[:5]:
        col_name = f'TeamA_{col}'
        exists = col_name in df.columns
        print(f"    {col_name}: {exists}")
    
    predictions = []
    h2h_success = 0
    h2h_failed = 0
    
    print(f"\nProcessing {len(df)} games...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)} games (H2H: {h2h_success} success, {h2h_failed} failed)")
        
        game_pred = {
            'Year': row['Year'],
            'Round': row['Round'],
            'TeamA': row['TeamA'],
            'TeamB': row['TeamB'],
            'SeedA': row['SeedA'],
            'SeedB': row['SeedB'],
            'Actual_Winner': 'TeamA' if row['TeamA_Won'] == 1 else 'TeamB'
        }
        
        # Method 1: Seeds (lower seed number = better)
        game_pred['Seeds_Pick'] = 'TeamA' if row['SeedA'] < row['SeedB'] else 'TeamB'
        game_pred['Seeds_Correct'] = (game_pred['Seeds_Pick'] == game_pred['Actual_Winner'])
        
        # Method 2: PowerRank (higher = better)
        if pd.notna(row['TeamA_PowerRank']) and pd.notna(row['TeamB_PowerRank']):
            game_pred['PowerRank_Pick'] = 'TeamA' if row['TeamA_PowerRank'] > row['TeamB_PowerRank'] else 'TeamB'
            game_pred['PowerRank_Correct'] = (game_pred['PowerRank_Pick'] == game_pred['Actual_Winner'])
        else:
            game_pred['PowerRank_Pick'] = None
            game_pred['PowerRank_Correct'] = None
        
        # Method 3: Barthag (higher = better)
        if pd.notna(row['TeamA_bartTorvik_Barthag']) and pd.notna(row['TeamB_bartTorvik_Barthag']):
            game_pred['Barthag_Pick'] = 'TeamA' if row['TeamA_bartTorvik_Barthag'] > row['TeamB_bartTorvik_Barthag'] else 'TeamB'
            game_pred['Barthag_Correct'] = (game_pred['Barthag_Pick'] == game_pred['Actual_Winner'])
        else:
            game_pred['Barthag_Pick'] = None
            game_pred['Barthag_Correct'] = None
        
        # Method 4: NetRtg (higher = better)
        if pd.notna(row['TeamA_kenpom_NetRtg']) and pd.notna(row['TeamB_kenpom_NetRtg']):
            game_pred['NetRtg_Pick'] = 'TeamA' if row['TeamA_kenpom_NetRtg'] > row['TeamB_kenpom_NetRtg'] else 'TeamB'
            game_pred['NetRtg_Correct'] = (game_pred['NetRtg_Pick'] == game_pred['Actual_Winner'])
        else:
            game_pred['NetRtg_Pick'] = None
            game_pred['NetRtg_Correct'] = None
        
        # Method 5: H2H Model
        if h2h_success == 0 and h2h_failed == 0:
            print(f"\n[DEBUG] Attempting first H2H prediction:")
            print(f"  feature_cols length: {len(feature_cols)}")
            print(f"  First 5 feature_cols: {feature_cols[:5]}")
            print(f"  Checking if columns exist in df:")
            for col in feature_cols[:5]:
                col_name = f'TeamA_{col}'
                exists = col_name in df.columns
                print(f"    {col_name}: {exists}")
                if exists:
                    val = row[col_name]
                    print(f"      value: {val}, type: {type(val)}")
        
        try:
            teamA_features = {col: row[f'TeamA_{col}'] for col in feature_cols}
            teamB_features = {col: row[f'TeamB_{col}'] for col in feature_cols}
            
            if h2h_success == 0 and h2h_failed == 0:
                print(f"  ✓ Dict comprehension succeeded")
                print(f"    teamA_features length: {len(teamA_features)}")
                print(f"    First 3 values: {list(teamA_features.values())[:3]}")
        except Exception as e:
            if h2h_success == 0 and h2h_failed == 0:
                print(f"  ✗ Dict comprehension FAILED: {e}")
            h2h_failed += 1
            game_pred['H2H_Prob_TeamA'] = None
            game_pred['H2H_Pick'] = None
            game_pred['H2H_Correct'] = None
            predictions.append(game_pred)
            continue
        
        # Debug first prediction only
        debug_first = (h2h_success == 0 and h2h_failed == 0)
        
        prob_teamA = get_h2h_prediction(teamA_features, teamB_features, feature_cols, models, scaler, config, debug_first)
        
        if prob_teamA is not None:
            game_pred['H2H_Prob_TeamA'] = prob_teamA
            game_pred['H2H_Pick'] = 'TeamA' if prob_teamA > 0.5 else 'TeamB'
            game_pred['H2H_Correct'] = (game_pred['H2H_Pick'] == game_pred['Actual_Winner'])
            h2h_success += 1
        else:
            game_pred['H2H_Prob_TeamA'] = None
            game_pred['H2H_Pick'] = None
            game_pred['H2H_Correct'] = None
            h2h_failed += 1
        
        predictions.append(game_pred)
    
    print(f"\n✓ Generated predictions for {len(predictions)} games")
    print(f"  H2H Model: {h2h_success} successful, {h2h_failed} failed")
    
    if h2h_failed > 0:
        print(f"  ⚠ H2H model failed on {h2h_failed} games ({h2h_failed/len(predictions)*100:.1f}%)")
    
    return pd.DataFrame(predictions)

def analyze_results(predictions_df):
    """Analyze and summarize prediction accuracy."""
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    methods = ['Seeds', 'PowerRank', 'Barthag', 'NetRtg', 'H2H']
    
    # Overall accuracy
    print("\n" + "="*80)
    print("OVERALL ACCURACY (2008-2025)")
    print("="*80)
    
    overall = []
    for method in methods:
        correct_col = f'{method}_Correct'
        if correct_col in predictions_df.columns:
            total = predictions_df[correct_col].notna().sum()
            correct = predictions_df[correct_col].sum()
            accuracy = correct / total if total > 0 else 0
            overall.append({
                'Method': method,
                'Correct': int(correct),
                'Total': int(total),
                'Accuracy': accuracy
            })
    
    overall_df = pd.DataFrame(overall).sort_values('Accuracy', ascending=False)
    print("\n" + overall_df.to_string(index=False))
    
    # Accuracy by season
    print("\n" + "="*80)
    print("ACCURACY BY SEASON")
    print("="*80)
    
    season_results = []
    for year in sorted(predictions_df['Year'].unique()):
        year_data = predictions_df[predictions_df['Year'] == year]
        row = {'Year': year}
        
        for method in methods:
            correct_col = f'{method}_Correct'
            if correct_col in year_data.columns:
                total = year_data[correct_col].notna().sum()
                correct = year_data[correct_col].sum()
                accuracy = correct / total if total > 0 else 0
                row[f'{method}_Acc'] = accuracy
                row[f'{method}_Record'] = f"{int(correct)}-{int(total-correct)}"
        
        season_results.append(row)
    
    season_df = pd.DataFrame(season_results)
    print("\n" + season_df.to_string(index=False))
    
    # Accuracy by round
    print("\n" + "="*80)
    print("ACCURACY BY ROUND")
    print("="*80)
    
    round_results = []
    for round_name in ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship']:
        round_data = predictions_df[predictions_df['Round'] == round_name]
        if len(round_data) == 0:
            continue
            
        row = {'Round': round_name, 'Games': len(round_data)}
        
        for method in methods:
            correct_col = f'{method}_Correct'
            if correct_col in round_data.columns:
                total = round_data[correct_col].notna().sum()
                correct = round_data[correct_col].sum()
                accuracy = correct / total if total > 0 else 0
                row[f'{method}_Acc'] = accuracy
        
        round_results.append(row)
    
    round_df = pd.DataFrame(round_results)
    print("\n" + round_df.to_string(index=False))
    
    return overall_df, season_df, round_df

def save_results(overall_df, season_df, round_df, predictions_df, output_dir):
    """Save analysis results to CSV files."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save overall accuracy
    overall_path = os.path.join(output_dir, 'overall_accuracy.csv')
    overall_df.to_csv(overall_path, index=False)
    print(f"\n✓ Overall accuracy saved to: {overall_path}")
    
    # Save season accuracy
    season_path = os.path.join(output_dir, 'accuracy_by_season.csv')
    season_df.to_csv(season_path, index=False)
    print(f"✓ Season accuracy saved to: {season_path}")
    
    # Save round accuracy
    round_path = os.path.join(output_dir, 'accuracy_by_round.csv')
    round_df.to_csv(round_path, index=False)
    print(f"✓ Round accuracy saved to: {round_path}")
    
    # Save full predictions
    predictions_path = os.path.join(output_dir, 'all_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Full predictions saved to: {predictions_path}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("H2H 05_backtest_predictions.py")
    print("="*80)
    print("\nBacktesting prediction methods on historical NCAA tournaments...")
    
    # Create output directory
    create_output_directory(OUTPUT_DIR)
    
    # Load data
    results, features = load_data()
    
    # Load H2H model
    models, scaler, config = load_h2h_model()
    
    # Load selected features (NO SEEDS - 27 features)
    selected_features = pd.read_csv(SELECTED_FEATURES_PATH)
    feature_cols = selected_features[selected_features['status'] == 'KEEP']['feature'].tolist()
    # Remove 'pct_diff_' prefix to get base feature names
    feature_cols = [f.replace('pct_diff_', '') for f in feature_cols if f != 'pct_diff_tournamentSeed']
    
    print(f"\n✓ Using {len(feature_cols)} features (NO SEEDS)")
    
    # Join features to results
    df = join_features_to_results(results, features)
    
    # Generate predictions
    predictions_df = make_predictions(df, feature_cols, models, scaler, config)
    
    # Analyze results
    overall_df, season_df, round_df = analyze_results(predictions_df)
    
    # Save results
    save_results(overall_df, season_df, round_df, predictions_df, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✓ BACKTEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
