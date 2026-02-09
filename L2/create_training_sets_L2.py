"""
create_training_sets_L2.py

Purpose: Join clean L2 feature sources to create training datasets
- Creates TWO views: long (2008+) and rich (2016+)
- LONG view: bartTorvik, kenPom, espnBPI, masseyComposite (2008-2025)
- RICH view: All long sources + LRMCB, powerRank (2016-2025)
- Filters to tournament teams only
- Removes tournament metadata to keep features pure
- Outputs to L3/data/trainingData/

Author: Ryan Browder
Created: 2025-01-29
"""

import pandas as pd
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths (relative to script location in L2/)
INPUTS = {
    'bartTorvik': 'data/bartTorvik/bartTorvik_analyze_L2.csv',
    'kenPom': 'data/kenPom/kenPom_analyze_L2.csv',
    'espnBPI': 'data/espnBPI/espnBPI_analyze_L2.csv',
    'masseyComposite': 'data/masseyComposite/masseyComposite_analyze_L2.csv',
    'LRMCB': 'data/LRMCB/LRMCB_analyze_L2.csv',
    'powerRank': 'data/powerRank/powerRank_analyze_L2.csv'
}

# Source groups by coverage period
SOURCES_2008 = ['bartTorvik', 'kenPom', 'espnBPI', 'masseyComposite']  # Long view
SOURCES_2016 = ['LRMCB', 'powerRank']  # Additional sources for rich view (2016+)

# Output path (relative to script location in L2/)
OUTPUT_DIR = '../L3/data/trainingData'
OUTPUT_FILES = {
    'long': 'training_set_long.csv',   # 2008+ with 4 sources
    'rich': 'training_set_rich.csv'    # 2016+ with 6 sources
}

# Columns to drop (tournament metadata - not features)
TOURNAMENT_COLS = ['tournamentSeed', 'tournamentOutcome']

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def validate_file_exists(filepath):
    """Check if input file exists, raise error if not"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required input file not found: {filepath}")
    print(f"✓ Found: {filepath}")

def load_source(filepath, source_name):
    """Load a source CSV and report basic stats"""
    df = pd.read_csv(filepath)
    print(f"\n{source_name}:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")
    return df

def diagnose_lost_teams(before_df, after_df, source_name, min_year=None):
    """Identify tournament teams lost during a join"""
    # Find tournament teams in before_df that aren't in after_df
    before_teams = set(before_df[before_df['tournamentSeed'].notna()][['Year', 'Index']].apply(tuple, axis=1))
    after_teams = set(after_df[['Year', 'Index']].apply(tuple, axis=1))
    lost_teams = before_teams - after_teams
    
    # Filter to min_year if specified (for rich view - only show 2016+ losses)
    if min_year is not None and len(lost_teams) > 0:
        lost_mask = before_df[['Year', 'Index']].apply(tuple, axis=1).isin(lost_teams)
        lost_records = before_df[lost_mask][['Year', 'Team', 'Index', 'tournamentSeed', 'tournamentOutcome']]
        lost_records = lost_records[lost_records['Year'] >= min_year]
        lost_teams_filtered = set(lost_records[['Year', 'Index']].apply(tuple, axis=1))
    else:
        lost_teams_filtered = lost_teams
    
    if len(lost_teams_filtered) > 0:
        print(f"\n  ⚠ {len(lost_teams_filtered)} tournament teams lost after {source_name} join:")
        # Get the full records for lost teams
        lost_mask = before_df[['Year', 'Index']].apply(tuple, axis=1).isin(lost_teams_filtered)
        lost_records = before_df[lost_mask][['Year', 'Team', 'Index', 'tournamentSeed', 'tournamentOutcome']].sort_values(['Year', 'tournamentSeed'])
        for _, row in lost_records.iterrows():
            print(f"    {row['Year']} {row['Team']:<25} (Index: {row['Index']:>3}, Seed: {row['tournamentSeed']:>2.0f}, Round: {row['tournamentOutcome']})")
    else:
        print(f"  ✓ No tournament teams lost after {source_name} join")

def main():
    print("="*80)
    print("CREATE TRAINING SETS - L2 → L3")
    print("="*80)
    
    # Validate inputs exist
    print("\n[1/6] Validating input files...")
    for source, path in INPUTS.items():
        validate_file_exists(path)
    
    # Load sources
    print("\n[2/6] Loading source data...")
    sources = {}
    for name, path in INPUTS.items():
        sources[name] = load_source(path, name)
    
    bart = sources['bartTorvik']
    
    # ========================================================================
    # CREATE LONG VIEW (2008+, 4 sources)
    # ========================================================================
    print("\n[3/6] Creating LONG view (2008+, 4 sources)...")
    print("-" * 80)
    
    df_long = bart.copy()
    print(f"  Starting with bartTorvik: {len(df_long):,} rows")
    print(f"  Tournament teams in bartTorvik: {df_long['tournamentSeed'].notna().sum()}")
    
    # Join each source in SOURCES_2008
    for source_name in SOURCES_2008[1:]:  # Skip bartTorvik (already loaded)
        df_long = df_long.merge(
            sources[source_name],
            on=['Year', 'Index'],
            how='inner',
            suffixes=('', '_DROP')
        )
        df_long = df_long.drop(columns=[col for col in df_long.columns if col.endswith('_DROP')])
        print(f"  After {source_name} join: {len(df_long):,} rows, {len(df_long.columns)} columns")
        diagnose_lost_teams(bart, df_long, source_name)
    
    # Check for duplicates
    print(f"\n  Checking for duplicates on (Year, Index)...")
    dupes_before = df_long.duplicated(subset=['Year', 'Index']).sum()
    if dupes_before > 0:
        print(f"  ⚠ Found {dupes_before} duplicate team-seasons")
        df_long = df_long.drop_duplicates(subset=['Year', 'Index'], keep='first')
        print(f"  Removed duplicates, kept first occurrence")
    else:
        print(f"  ✓ No duplicates found")
    
    # Filter to tournament teams
    print(f"\n  Filtering to tournament teams...")
    print(f"  Before filter: {len(df_long):,} rows")
    df_long = df_long[df_long['tournamentSeed'].notna()].copy()
    print(f"  After filter: {len(df_long):,} rows")
    
    # Drop tournament metadata
    df_long = df_long.drop(columns=TOURNAMENT_COLS)
    print(f"  Final feature count: {len(df_long.columns)}")
    
    # Invert rank columns (lower rank = better → higher value = better)
    print(f"\n  Inverting rank columns for pct_diff compatibility...")
    rank_cols = [c for c in df_long.columns if 'Rank' in c or c.endswith('_Rk')]
    
    if rank_cols:
        print(f"    Found {len(rank_cols)} rank columns to invert:")
        for col in rank_cols:
            if col in df_long.columns and df_long[col].notna().any():
                max_rank = df_long[col].max()
                min_rank = df_long[col].min()
                df_long[col] = max_rank - df_long[col]
                print(f"      {col}: inverted (range now 0-{max_rank - min_rank:.0f})")
        print(f"    ✓ Rank inversion complete for LONG view")
    else:
        print(f"    No rank columns found to invert")
    
    # ========================================================================
    # CREATE RICH VIEW (2016+, 6 sources)
    # ========================================================================
    print("\n[4/6] Creating RICH view (2016+, 6 sources)...")
    print("-" * 80)
    
    df_rich = bart.copy()
    print(f"  Starting with bartTorvik: {len(df_rich):,} rows")
    
    # Join all sources (2008 sources + 2016 sources)
    all_sources = SOURCES_2008[1:] + SOURCES_2016
    for source_name in all_sources:
        df_rich = df_rich.merge(
            sources[source_name],
            on=['Year', 'Index'],
            how='inner',
            suffixes=('', '_DROP')
        )
        df_rich = df_rich.drop(columns=[col for col in df_rich.columns if col.endswith('_DROP')])
        print(f"  After {source_name} join: {len(df_rich):,} rows, {len(df_rich.columns)} columns")
        diagnose_lost_teams(bart, df_rich, source_name, min_year=2016)  # Only show 2016+ losses
    
    # Check for duplicates
    print(f"\n  Checking for duplicates on (Year, Index)...")
    dupes_before = df_rich.duplicated(subset=['Year', 'Index']).sum()
    if dupes_before > 0:
        print(f"  ⚠ Found {dupes_before} duplicate team-seasons")
        df_rich = df_rich.drop_duplicates(subset=['Year', 'Index'], keep='first')
        print(f"  Removed duplicates, kept first occurrence")
    else:
        print(f"  ✓ No duplicates found")
    
    # Filter to 2016+ and tournament teams
    print(f"\n  Filtering to 2016+ and tournament teams...")
    print(f"  Before filter: {len(df_rich):,} rows")
    df_rich = df_rich[(df_rich['Year'] >= 2016) & (df_rich['tournamentSeed'].notna())].copy()
    print(f"  After filter: {len(df_rich):,} rows")
    
    # Drop tournament metadata
    df_rich = df_rich.drop(columns=TOURNAMENT_COLS)
    print(f"  Final feature count: {len(df_rich.columns)}")
    
    # Invert rank columns (lower rank = better → higher value = better)
    print(f"\n  Inverting rank columns for pct_diff compatibility...")
    rank_cols = [c for c in df_rich.columns if 'Rank' in c or c.endswith('_Rk')]
    
    if rank_cols:
        print(f"    Found {len(rank_cols)} rank columns to invert:")
        for col in rank_cols:
            if col in df_rich.columns and df_rich[col].notna().any():
                max_rank = df_rich[col].max()
                min_rank = df_rich[col].min()
                df_rich[col] = max_rank - df_rich[col]
                print(f"      {col}: inverted (range now 0-{max_rank - min_rank:.0f})")
        print(f"    ✓ Rank inversion complete for RICH view")
    else:
        print(f"    No rank columns found to invert")
    
    # ========================================================================
    # VALIDATE AND WRITE OUTPUTS
    # ========================================================================
    print("\n[5/6] Validating outputs...")
    
    # Validate no nulls in key columns
    for name, df in [('LONG', df_long), ('RICH', df_rich)]:
        print(f"\n  {name} view:")
        null_counts = df[['Year', 'Team', 'Index']].isnull().sum()
        if null_counts.sum() > 0:
            print(f"    ⚠ WARNING: Null values found in key columns:")
            print(null_counts[null_counts > 0])
        else:
            print(f"    ✓ No nulls in key columns")
    
    # Create output directory if needed
    print(f"\n[6/6] Writing outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Write long view
    output_path_long = os.path.join(OUTPUT_DIR, OUTPUT_FILES['long'])
    df_long.to_csv(output_path_long, index=False)
    print(f"  ✓ Written: {output_path_long}")
    print(f"    Rows: {len(df_long):,}")
    print(f"    Columns: {len(df_long.columns)}")
    
    # Write rich view
    output_path_rich = os.path.join(OUTPUT_DIR, OUTPUT_FILES['rich'])
    df_rich.to_csv(output_path_rich, index=False)
    print(f"  ✓ Written: {output_path_rich}")
    print(f"    Rows: {len(df_rich):,}")
    print(f"    Columns: {len(df_rich.columns)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nLONG VIEW (2008-2025, 4 sources):")
    print(f"  Sources: {', '.join(SOURCES_2008)}")
    print(f"  Output: {output_path_long}")
    print(f"  Rows: {len(df_long):,}")
    print(f"  Year range: {df_long['Year'].min()}-{df_long['Year'].max()}")
    print(f"  Features: {len(df_long.columns)}")
    print(f"  Teams/year: {len(df_long) / df_long['Year'].nunique():.1f}")
    
    print(f"\nRICH VIEW (2016-2025, 6 sources):")
    print(f"  Sources: {', '.join(SOURCES_2008 + SOURCES_2016)}")
    print(f"  Output: {output_path_rich}")
    print(f"  Rows: {len(df_rich):,}")
    print(f"  Year range: {df_rich['Year'].min()}-{df_rich['Year'].max()}")
    print(f"  Features: {len(df_rich.columns)}")
    print(f"  Teams/year: {len(df_rich) / df_rich['Year'].nunique():.1f}")
    print("="*80)

if __name__ == "__main__":
    main()
