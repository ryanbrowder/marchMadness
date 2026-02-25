"""
create_predict_set_L2.py

Purpose: Join clean L2 prediction sources to create prediction dataset
- Uses *_predict_L2.csv files (current season 2026 data)
- Joins all available sources (bartTorvik, kenPom, espnBPI, masseyComposite required)
- Optional sources: LRMCB, powerRank (skipped if files don't exist)
- Adds ESPN bracketology data (tournamentSeed, tournamentRegion)
- Adds Vegas odds (elite8_prob, final4_prob, champ_prob from DraftKings)
- No tournament filtering (prediction is pre-tournament)
- Outputs to L3/data/predictionData/

Author: Ryan Browder
Created: 2025-01-30
Updated: 2025-02-09 (added Vegas odds integration)
"""

import pandas as pd
import os
from pathlib import Path

# ============================================================================
# FEATURE TOGGLES
# ============================================================================
INCLUDE_VEGAS_ODDS = False  # Set to False to disable Vegas odds integration
INCLUDE_LRMCB = False  # Set to False to exclude LRMCB (if not available for current year)

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

# Bracketology input (ESPN projected tournament seeds)
BRACKETOLOGY_PATH = 'data/bracketology/espn_bracketology_2026.csv'

# Vegas odds input (DraftKings Elite 8 probabilities)
VEGAS_ODDS_PATH = 'data/vegasOdds/vegasOdds_analyze_L2.csv'

# Optional sources (skip if file doesn't exist OR if disabled by flag)
OPTIONAL_SOURCES = []
if INCLUDE_LRMCB:
    OPTIONAL_SOURCES.append('LRMCB')
OPTIONAL_SOURCES.append('powerRank')  # powerRank always optional

# All sources (will be filtered to available sources at runtime)
ALL_SOURCES = ['bartTorvik', 'kenPom', 'espnBPI', 'masseyComposite', 'LRMCB', 'powerRank']

# Output path (relative to script location in L2/)
OUTPUT_DIR = '../L3/data/predictionData'
OUTPUT_FILE = 'predict_set_2026.csv'

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
    if not INCLUDE_LRMCB:
        print(f"  Note: LRMCB excluded by flag (INCLUDE_LRMCB = False)")
    
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
    
    # ========================================================================
    # ADD BRACKETOLOGY DATA (projected tournament seeds)
    # ========================================================================
    print(f"\n  Adding ESPN bracketology data...")
    
    # Drop any existing tournamentSeed/tournamentRegion columns to avoid merge conflicts
    cols_to_drop = [col for col in ['tournamentSeed', 'tournamentRegion'] if col in df.columns]
    if cols_to_drop:
        print(f"    Dropping existing columns: {', '.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    
    if os.path.exists(BRACKETOLOGY_PATH):
        # Read with utf-8-sig to handle BOM if present
        bracket = pd.read_csv(BRACKETOLOGY_PATH, encoding='utf-8-sig')
        print(f"    Loaded: {len(bracket)} projected tournament teams")
        
        # Rename columns to match our schema
        bracket = bracket.rename(columns={
            'Team Index': 'Index', 
            'Seed': 'tournamentSeed',
            'Region': 'tournamentRegion'
        })
        
        # Keep Index, tournamentSeed, and tournamentRegion for the join
        bracket = bracket[['Index', 'tournamentSeed', 'tournamentRegion']]
        
        # Left join to add tournament data (null for non-tournament teams)
        df = df.merge(bracket, on='Index', how='left')
        
        teams_with_seeds = df['tournamentSeed'].notna().sum()
        print(f"    ✓ Added tournamentSeed and tournamentRegion columns from ESPN bracketology")
        print(f"    {teams_with_seeds} teams projected to make tournament")
        print(f"    {len(df) - teams_with_seeds} teams not projected (tournamentSeed/Region = null)")
    else:
        print(f"    ⊘ Bracketology file not found: {BRACKETOLOGY_PATH}")
        print(f"    Continuing without tournamentSeed/tournamentRegion columns")
    
    # ========================================================================
    # ADD VEGAS ODDS (Elite 8 probabilities from DraftKings)
    # ========================================================================
    if INCLUDE_VEGAS_ODDS:
        print(f"\n  Adding Vegas odds data...")
        
        if os.path.exists(VEGAS_ODDS_PATH):
            vegas = pd.read_csv(VEGAS_ODDS_PATH)
            print(f"    Loaded: {len(vegas)} teams with Vegas odds")
            
            # Select columns for join (teamIndex maps to Index)
            vegas_cols = ['teamIndex', 'vegas_elite8_prob', 'vegas_final4_prob', 'vegas_champ_prob']
            vegas = vegas[vegas_cols].rename(columns={'teamIndex': 'Index'})
            
            # Left join to add Vegas odds (null for teams without odds)
            df = df.merge(vegas, on='Index', how='left')
            
            teams_with_odds = df['vegas_elite8_prob'].notna().sum()
            print(f"    ✓ Added vegas_elite8_prob, vegas_final4_prob, vegas_champ_prob columns")
            print(f"    {teams_with_odds} teams have Vegas odds")
            print(f"    {len(df) - teams_with_odds} teams without odds (will use model-only predictions)")
        else:
            print(f"    ⊘ Vegas odds file not found: {VEGAS_ODDS_PATH}")
            print(f"    Continuing without Vegas odds (model-only predictions)")
    else:
        print(f"\n  ⊘ Vegas odds integration DISABLED (INCLUDE_VEGAS_ODDS = False)")
        print(f"    Running in model-only mode")
    
    print(f"\n  Final feature count: {len(df.columns)}")
    print(f"  Final team count: {len(df):,}")
    
    # ========================================================================
    # INVERT RANK COLUMNS (lower rank = better → higher value = better)
    # ========================================================================
    print(f"\n  Inverting rank columns for pct_diff compatibility...")
    rank_cols = [c for c in df.columns if 'Rank' in c or c.endswith('_Rk')]
    
    if rank_cols:
        print(f"    Found {len(rank_cols)} rank columns to invert:")
        for col in rank_cols:
            if col in df.columns and df[col].notna().any():
                max_rank = df[col].max()
                min_rank = df[col].min()
                df[col] = max_rank - df[col]
                print(f"      {col}: inverted (was {min_rank:.0f}-{max_rank:.0f}, now {0:.0f}-{max_rank - min_rank:.0f})")
        print(f"    ✓ Rank inversion complete (higher values now = better teams)")
    else:
        print(f"    No rank columns found to invert")
    
    # ========================================================================
    # NORMALIZE RANKS TO STANDARD 0-364 SCALE
    # ========================================================================
    print(f"\n  Normalizing ranks to standard 0-364 scale...")
    STANDARD_MAX_RANK = 364  # Modern D1 basketball field size
    
    if rank_cols:
        print(f"    Normalizing {len(rank_cols)} rank columns to 0-364 scale:")
        for col in rank_cols:
            if col in df.columns and df[col].notna().any():
                current_max = df[col].max()
                if current_max > 0:
                    df[col] = (df[col] / current_max) * STANDARD_MAX_RANK
                    print(f"      {col}: normalized (max was {current_max:.0f}, now {STANDARD_MAX_RANK})")
        print(f"    ✓ Rank normalization complete (all ranks now on 0-364 scale)")
    else:
        print(f"    No rank columns to normalize")
    
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
    
    # Report tournamentSeed column if present
    if 'tournamentSeed' in df.columns:
        teams_with_seeds = df['tournamentSeed'].notna().sum()
        print(f"  ✓ tournamentSeed column: {teams_with_seeds} teams with projected seeds")
        if 'tournamentRegion' in df.columns:
            regions = df[df['tournamentRegion'].notna()]['tournamentRegion'].unique()
            print(f"  ✓ tournamentRegion column: {len(regions)} regions ({', '.join(sorted(regions))})")
    
    # Report Vegas odds coverage if present
    if 'vegas_elite8_prob' in df.columns:
        teams_with_vegas = df['vegas_elite8_prob'].notna().sum()
        print(f"  ✓ Vegas odds columns: {teams_with_vegas} teams with DraftKings probabilities")
    
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
    if 'tournamentSeed' in df.columns:
        teams_with_seeds = df['tournamentSeed'].notna().sum()
        print(f"Projected tournament teams: {teams_with_seeds}")
        if 'tournamentRegion' in df.columns:
            regions = df[df['tournamentRegion'].notna()]['tournamentRegion'].unique()
            print(f"Tournament regions: {', '.join(sorted(regions))}")
    if 'vegas_elite8_prob' in df.columns:
        teams_with_vegas = df['vegas_elite8_prob'].notna().sum()
        print(f"Teams with Vegas odds: {teams_with_vegas}")
    print("="*80)

if __name__ == "__main__":
    main()