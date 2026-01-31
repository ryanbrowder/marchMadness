"""
H2H 01_build_training_matchups.py

Purpose: Join historical tournament matchups with team-level features to create
         matchup-level training data with symmetric percentage differential features.

Inputs:
    - L3/data/tournamentResults.csv (historical game outcomes)
    - L3/data/training_set_long.csv (team features by year)

Outputs:
    - L3/h2h/outputs/01_build_training_matchups/training_matchups.csv
    - Diagnostic reports to console

Author: Ryan Browder
Date: 2025-01-31
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Input paths
TOURNAMENT_RESULTS_PATH = '../../L2/data/srcbb/srcbb_analyze_L2.csv'
TRAINING_FEATURES_PATH = '../data/trainingData/training_set_long.csv'

# Output directory
OUTPUT_DIR = 'outputs/01_build_training_matchups'
OUTPUT_FILE = 'training_matchups.csv'

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
    print(f"  Teams: {features['Index'].nunique()} unique teams")
    print(f"  Feature columns: {len(features.columns) - 2}")  # Exclude Year, Index
    
    return results, features

def diagnose_missing_joins(results_df, features_df):
    """
    Identify which teams are failing to join with features.
    """
    print("\n" + "="*80)
    print("DIAGNOSING MISSING JOINS")
    print("="*80)
    
    # Get unique (Year, TeamA_ID, TeamA) from results
    teamA_combos = results_df[['Year', 'TeamA_ID', 'TeamA']].drop_duplicates()
    teamB_combos = results_df[['Year', 'TeamB_ID', 'TeamB']].drop_duplicates()
    
    # Rename for consistency
    teamA_combos = teamA_combos.rename(columns={'TeamA_ID': 'Index', 'TeamA': 'Team'})
    teamB_combos = teamB_combos.rename(columns={'TeamB_ID': 'Index', 'TeamB': 'Team'})
    
    # Combine both
    all_teams = pd.concat([teamA_combos, teamB_combos]).drop_duplicates()
    
    print(f"\nTotal unique team-year combinations in tournament results: {len(all_teams)}")
    
    # Check which exist in features
    features_keys = features_df[['Year', 'Index']].drop_duplicates()
    
    print(f"Total unique team-year combinations in training features: {len(features_keys)}")
    
    # Find missing
    missing = all_teams.merge(
        features_keys,
        on=['Year', 'Index'],
        how='left',
        indicator=True
    )
    
    missing = missing[missing['_merge'] == 'left_only'][['Year', 'Index', 'Team']].sort_values(['Year', 'Team'])
    
    if len(missing) > 0:
        print(f"\n⚠ Found {len(missing)} team-year combinations with no matching features:")
        print(missing.to_string(index=False))
        
        # Show which games are affected
        affected_games = results_df[
            results_df.apply(lambda row: 
                (row['Year'] in missing['Year'].values and row['TeamA_ID'] in missing['Index'].values) or
                (row['Year'] in missing['Year'].values and row['TeamB_ID'] in missing['Index'].values),
                axis=1
            )
        ][['Year', 'Round', 'TeamA', 'TeamB']]
        
        print(f"\n⚠ {len(affected_games)} games affected:")
        print(affected_games.to_string(index=False))
    else:
        print("\n✓ All team-year combinations have matching features")
    
    return missing

def identify_feature_columns(features_df):
    """Identify numeric feature columns to use for differentials."""
    # Exclude non-feature columns
    exclude_cols = ['Year', 'Index', 'Team']
    
    # Get all numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return feature_cols

def calculate_pct_diff(val_a, val_b):
    """
    Calculate symmetric percentage difference: (A - B) / ((A + B) / 2)
    
    This is symmetric: reversing A and B gives opposite sign but same magnitude.
    Handles edge cases where both values are 0 or very close to 0.
    """
    avg = (val_a + val_b) / 2.0
    
    # Avoid division by zero
    if avg == 0 or np.isnan(avg):
        return 0.0
    
    return (val_a - val_b) / avg

def join_team_features(results_df, features_df, feature_cols):
    """
    Join team features for both TeamA and TeamB, calculate differentials.
    """
    print("\n" + "="*80)
    print("JOINING TEAM FEATURES")
    print("="*80)
    
    # Prepare features for joining
    # Rename Index to TeamA_ID for first join
    features_a = features_df.copy()
    features_a = features_a.rename(columns={'Index': 'TeamA_ID'})
    
    # Rename Index to TeamB_ID for second join
    features_b = features_df.copy()
    features_b = features_b.rename(columns={'Index': 'TeamB_ID'})
    
    # Add suffix to feature columns to distinguish TeamA vs TeamB
    feature_rename_a = {col: f'TeamA_{col}' for col in feature_cols}
    feature_rename_b = {col: f'TeamB_{col}' for col in feature_cols}
    
    features_a = features_a.rename(columns=feature_rename_a)
    features_b = features_b.rename(columns=feature_rename_b)
    
    print(f"\nJoining TeamA features...")
    print(f"  Results before join: {len(results_df)} games")
    
    # Join TeamA features
    matchups = results_df.merge(
        features_a[['Year', 'TeamA_ID'] + list(feature_rename_a.values())],
        on=['Year', 'TeamA_ID'],
        how='left'
    )
    
    teamA_nulls = matchups[[col for col in matchups.columns if col.startswith('TeamA_')]].isnull().any(axis=1).sum()
    print(f"  Games after join: {len(matchups)}")
    print(f"  Games with missing TeamA features: {teamA_nulls}")
    
    print(f"\nJoining TeamB features...")
    
    # Join TeamB features
    matchups = matchups.merge(
        features_b[['Year', 'TeamB_ID'] + list(feature_rename_b.values())],
        on=['Year', 'TeamB_ID'],
        how='left'
    )
    
    teamB_nulls = matchups[[col for col in matchups.columns if col.startswith('TeamB_')]].isnull().any(axis=1).sum()
    print(f"  Games after join: {len(matchups)}")
    print(f"  Games with missing TeamB features: {teamB_nulls}")
    
    # Drop games with missing features
    initial_count = len(matchups)
    matchups = matchups.dropna(subset=[col for col in matchups.columns if col.startswith('TeamA_') or col.startswith('TeamB_')])
    dropped_count = initial_count - len(matchups)
    
    print(f"\n✓ Feature join complete")
    print(f"  Final games: {len(matchups)}")
    print(f"  Dropped due to missing features: {dropped_count}")
    
    return matchups, feature_cols

def calculate_differentials(matchups_df, feature_cols):
    """
    Calculate symmetric percentage differentials for all feature columns.
    """
    print("\n" + "="*80)
    print("CALCULATING PERCENTAGE DIFFERENTIALS")
    print("="*80)
    
    diff_cols = []
    
    for feature in feature_cols:
        col_a = f'TeamA_{feature}'
        col_b = f'TeamB_{feature}'
        diff_col = f'pct_diff_{feature}'
        
        # Calculate symmetric percentage difference
        matchups_df[diff_col] = matchups_df.apply(
            lambda row: calculate_pct_diff(row[col_a], row[col_b]),
            axis=1
        )
        
        diff_cols.append(diff_col)
    
    print(f"\n✓ Created {len(diff_cols)} percentage differential features")
    
    # Show sample statistics
    print(f"\nSample differential statistics:")
    sample_diffs = [col for col in diff_cols[:5]]  # First 5 for display
    print(matchups_df[sample_diffs].describe())
    
    return matchups_df, diff_cols

def prepare_final_dataset(matchups_df, diff_cols):
    """
    Select final columns for training dataset.
    """
    print("\n" + "="*80)
    print("PREPARING FINAL DATASET")
    print("="*80)
    
    # Select metadata columns
    metadata_cols = [
        'Year', 'Region', 'Round',
        'TeamA', 'TeamA_ID', 'SeedA',
        'TeamB', 'TeamB_ID', 'SeedB',
        'Winner', 'TeamA_Won',
        'SeedDiff', 'IsUpset'
    ]
    
    # Check which metadata columns exist
    available_metadata = [col for col in metadata_cols if col in matchups_df.columns]
    
    # Combine metadata + differentials
    final_cols = available_metadata + diff_cols
    
    final_df = matchups_df[final_cols].copy()
    
    print(f"\n✓ Final dataset prepared")
    print(f"  Metadata columns: {len(available_metadata)}")
    print(f"  Differential features: {len(diff_cols)}")
    print(f"  Total columns: {len(final_cols)}")
    print(f"  Total games: {len(final_df)}")
    
    # Show target distribution
    if 'TeamA_Won' in final_df.columns:
        wins = final_df['TeamA_Won'].sum()
        losses = len(final_df) - wins
        print(f"\nTarget distribution:")
        print(f"  TeamA wins: {wins} ({wins/len(final_df)*100:.1f}%)")
        print(f"  TeamA losses: {losses} ({losses/len(final_df)*100:.1f}%)")
    
    return final_df

def save_output(df, output_dir, output_file):
    """Save final training matchups dataset."""
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Training matchups saved to: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

def print_summary_stats(df):
    """Print summary statistics about the dataset."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nYears covered: {df['Year'].min()}-{df['Year'].max()}")
    print(f"Total games: {len(df)}")
    
    if 'Round' in df.columns:
        print(f"\nGames by round:")
        round_counts = df['Round'].value_counts().sort_index()
        for round_name, count in round_counts.items():
            print(f"  {round_name}: {count}")
    
    if 'SeedDiff' in df.columns:
        print(f"\nSeed differential statistics:")
        print(f"  Mean: {df['SeedDiff'].mean():.2f}")
        print(f"  Median: {df['SeedDiff'].median():.2f}")
        print(f"  Range: [{df['SeedDiff'].min()}, {df['SeedDiff'].max()}]")
    
    if 'IsUpset' in df.columns:
        upsets = df['IsUpset'].sum()
        print(f"\nUpsets: {upsets} ({upsets/len(df)*100:.1f}%)")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("H2H 01_build_training_matchups.py")
    print("="*80)
    print("\nBuilding head-to-head training matchups with percentage differentials...")
    
    # Create output directory
    create_output_directory(OUTPUT_DIR)
    
    # Load data
    results, features = load_data()
    
    # Diagnose missing joins BEFORE joining
    missing_teams = diagnose_missing_joins(results, features)
    
    # Identify feature columns
    feature_cols = identify_feature_columns(features)
    print(f"\nIdentified {len(feature_cols)} feature columns for differentials")
    print(f"Sample features: {feature_cols[:5]}")
    
    # Join team features
    matchups, feature_cols = join_team_features(results, features, feature_cols)
    
    # Calculate percentage differentials
    matchups, diff_cols = calculate_differentials(matchups, feature_cols)
    
    # Prepare final dataset
    final_df = prepare_final_dataset(matchups, diff_cols)
    
    # Save output
    save_output(final_df, OUTPUT_DIR, OUTPUT_FILE)
    
    # Print summary statistics
    print_summary_stats(final_df)
    
    print("\n" + "="*80)
    print("✓ TRAINING MATCHUPS BUILD COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
