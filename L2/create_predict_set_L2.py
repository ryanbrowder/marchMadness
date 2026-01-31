"""
create_predict_set_L2.py

Purpose: Join clean L2 prediction sources to create prediction dataset
- Uses *_predict_L2.csv files (current season 2025 data)
- Joins all available sources (bartTorvik, kenPom, espnBPI, masseyComposite required)
- Optional sources: LRMCB, powerRank (skipped if files don't exist)
- No tournament filtering (prediction is pre-tournament)
- Outputs to L3/data/predictionData/

Author: Ryan Browder
Created: 2025-01-30
"""

import pandas as pd
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths (relative to script location in L2/)
INPUTS = {
    'bartTorvik': 'data/bartTorvik/bartTorvik_predict_L2.csv',
    'kenPom': 'data/kenPom/kenPom_predict_L2.csv',
    'espnBPI': 'data/espnBPI/espnBPI_predict_L2.csv',
    'masseyComposite': 'data/masseyComposite/masseyComposite_predict_L2.csv',
    'LRMCB': 'data/LRMCB/LRMCB_predict_L2.csv',
    'powerRank': 'data/powerRank/powerRank_predict_L2.csv'
}

# Optional sources (skip if file doesn't exist)
OPTIONAL_SOURCES = ['LRMCB', 'powerRank']

# All sources (will be filtered to available sources at runtime)
ALL_SOURCES = ['bartTorvik', 'kenPom', 'espnBPI', 'masseyComposite', 'LRMCB', 'powerRank']

# Output path (relative to script location in L2/)
OUTPUT_DIR = '../L3/data/predictionData'
OUTPUT_FILE = 'predict_set_2025.csv'

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def validate_file_exists(filepath, optional=False):
    """Check if input file exists, raise error if not (unless optional)"""
    if not os.path.exists(filepath):
        if optional:
            print(f"⊘ Skipped (optional): {filepath}")
            return False
        else:
            raise FileNotFoundError(f"Required input file not found: {filepath}")
    print(f"✓ Found: {filepath}")
    return True

def load_source(filepath, source_name):
    """Load a source CSV and report basic stats"""
    df = pd.read_csv(filepath)
    print(f"\n{source_name}:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    if 'Year' in df.columns:
        print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")
    return df

def diagnose_lost_teams(before_df, after_df, source_name):
    """Identify teams lost during a join"""
    before_teams = set(before_df[['Year', 'Index']].apply(tuple, axis=1))
    after_teams = set(after_df[['Year', 'Index']].apply(tuple, axis=1))
    lost_teams = before_teams - after_teams
    
    if len(lost_teams) > 0:
        print(f"\n  ⚠ {len(lost_teams)} teams lost after {source_name} join:")
        # Get the full records for lost teams
        lost_mask = before_df[['Year', 'Index']].apply(tuple, axis=1).isin(lost_teams)
        lost_records = before_df[lost_mask][['Year', 'Team', 'Index']].sort_values(['Year', 'Team'])
        for _, row in lost_records.head(10).iterrows():  # Show max 10
            print(f"    {row['Year']} {row['Team']:<30} (Index: {row['Index']:>3})")
        if len(lost_teams) > 10:
            print(f"    ... and {len(lost_teams) - 10} more")
    else:
        print(f"  ✓ No teams lost after {source_name} join")

def main():
    print("="*80)
    print("CREATE PREDICTION SET - L2 → L3")
    print("="*80)
    
    # Validate inputs exist
    print("\n[1/5] Validating input files...")
    available_sources = []
    for source, path in INPUTS.items():
        is_optional = source in OPTIONAL_SOURCES
        if validate_file_exists(path, optional=is_optional):
            available_sources.append(source)
    
    print(f"\n  Available sources: {len(available_sources)}/{len(INPUTS)}")
    print(f"  Using: {', '.join(available_sources)}")
    
    # Load sources
    print("\n[2/5] Loading source data...")
    sources = {}
    for name in available_sources:
        sources[name] = load_source(INPUTS[name], name)
    
    bart = sources['bartTorvik']
    
    # ========================================================================
    # CREATE PREDICTION SET (available sources)
    # ========================================================================
    print(f"\n[3/5] Creating prediction set ({len(available_sources)} sources)...")
    print("-" * 80)
    
    df = bart.copy()
    print(f"  Starting with bartTorvik: {len(df):,} rows")
    
    # Join all available sources (skip bartTorvik - already loaded)
    sources_to_join = [s for s in available_sources if s != 'bartTorvik']
    for source_name in sources_to_join:
        df = df.merge(
            sources[source_name],
            on=['Year', 'Index'],
            how='inner',
            suffixes=('', '_DROP')
        )
        df = df.drop(columns=[col for col in df.columns if col.endswith('_DROP')])
        print(f"  After {source_name} join: {len(df):,} rows, {len(df.columns)} columns")
        diagnose_lost_teams(bart, df, source_name)
    
    # Check for duplicates
    print(f"\n  Checking for duplicates on (Year, Index)...")
    dupes_before = df.duplicated(subset=['Year', 'Index']).sum()
    if dupes_before > 0:
        print(f"  ⚠ Found {dupes_before} duplicate team-seasons")
        df = df.drop_duplicates(subset=['Year', 'Index'], keep='first')
        print(f"  Removed duplicates, kept first occurrence")
    else:
        print(f"  ✓ No duplicates found")
    
    print(f"\n  Final feature count: {len(df.columns)}")
    print(f"  Final team count: {len(df):,}")
    
    # ========================================================================
    # VALIDATE AND WRITE OUTPUT
    # ========================================================================
    print("\n[4/5] Validating output...")
    
    # Validate no nulls in key columns
    null_counts = df[['Year', 'Team', 'Index']].isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ⚠ WARNING: Null values found in key columns:")
        print(null_counts[null_counts > 0])
    else:
        print(f"  ✓ No nulls in key columns")
    
    # Create output directory if needed
    print(f"\n[5/5] Writing output...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Write prediction set
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Written: {output_path}")
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Sources joined: {', '.join(available_sources)}")
    print(f"Output file: {output_path}")
    print(f"Total teams: {len(df):,}")
    if 'Year' in df.columns:
        print(f"Year: {df['Year'].unique()[0]}")
    print(f"Feature columns: {len(df.columns)}")
    print("="*80)

if __name__ == "__main__":
    main()
