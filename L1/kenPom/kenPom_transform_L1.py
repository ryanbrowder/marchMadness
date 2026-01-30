"""
KenPom Data Transform - L1 (Transform Layer)
Cleans and standardizes KenPom raw data
Outputs: 
  - kenpom_predict_L2.csv (2026 data for predictions)
  - kenpom_analyze_L2.csv (historical data for training)
Note: Run from L1/kenPom/ directory
"""

import pandas as pd
import os

# Configuration
CURRENT_YEAR = 2026
INPUT_FILE = "../data/kenPom/kenPom_raw_L1.csv"
TEAM_INDEX_FILE = "../../utils/teamsIndex.csv"
OUTPUT_DIR = "../../L2/data/kenPom"
PREDICT_FILE = f"{OUTPUT_DIR}/kenPom_predict_L2.csv"
ANALYZE_FILE = f"{OUTPUT_DIR}/kenPom_analyze_L2.csv"

# Columns to delete
DELETE_COLUMNS = ['Conf', 'W-L']

# Columns to exclude from prefixing
EXCLUDE_PREFIX = ['Team', 'Year', 'Index']


def load_team_index():
    """Load team name to index mapping"""
    try:
        team_index = pd.read_csv(TEAM_INDEX_FILE)
        # Create mapping dict: Team name -> Index
        return dict(zip(team_index['Team'], team_index['Index']))
    except FileNotFoundError:
        print(f"Warning: Team index file not found at {TEAM_INDEX_FILE}")
        print("Run utils/generate_team_index.py first to create the index")
        return {}


def clean_team_name(team_str):
    """
    Clean team name by removing seed numbers and tournament indicators
    Examples: 
      "Kansas 1" -> "Kansas"
      "Duke 2*" -> "Duke"
      "Akron 13**" -> "Akron"
    """
    if pd.isna(team_str):
        return team_str
    
    # Split and take all parts except last if last part contains only digits/asterisks
    parts = str(team_str).strip().split()
    if len(parts) > 1:
        last_part = parts[-1]
        # Check if last part is only digits and/or asterisks
        if all(c.isdigit() or c == '*' for c in last_part):
            return ' '.join(parts[:-1])
    
    return str(team_str).strip()


def remove_duplicate_headers(df):
    """Remove any header rows that were mixed into data"""
    # Header rows will have 'Rk' in the Rk column
    df = df[df['Rk'] != 'Rk'].copy()
    return df


def add_team_index(df, team_index_map):
    """Add Index column based on team name lookup and clean Team column"""
    # Clean team names (remove seed numbers)
    df['Team'] = df['Team'].apply(clean_team_name)
    
    # Map to index
    df['Index'] = df['Team'].map(team_index_map)
    
    # Report missing mappings
    missing = df[df['Index'].isna()]['Team'].unique()
    if len(missing) > 0:
        print(f"Warning: {len(missing)} teams not found in index:")
        for team in sorted(missing)[:10]:  # Show first 10
            print(f"  - {team}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    return df


def reorder_columns(df):
    """
    Reorder columns: Year -> Team -> Index -> Rk -> [remaining features]
    """
    # Get all columns
    cols = list(df.columns)
    
    # Remove the ordered columns from list
    remaining = [c for c in cols if c not in ['Year', 'Team', 'Index', 'Rk']]
    
    # Build new order
    new_order = ['Year', 'Team', 'Index', 'Rk'] + remaining
    
    return df[new_order]


def prefix_columns(df):
    """Add 'kenpom_' prefix to all columns except Team, Year, Index"""
    rename_map = {}
    for col in df.columns:
        if col not in EXCLUDE_PREFIX:
            rename_map[col] = f'kenpom_{col}'
    
    return df.rename(columns=rename_map)


def convert_to_numeric(df):
    """Convert all columns to numeric except Team"""
    for col in df.columns:
        if col != 'Team':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def transform_data():
    """Main transformation function"""
    print("="*60)
    print("KenPom Data Transform - L1")
    print("="*60)
    
    # Load raw data
    print(f"Loading raw data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Loaded {len(df)} rows")
    
    # Remove duplicate headers
    print("Cleaning duplicate header rows...")
    df = remove_duplicate_headers(df)
    print(f"  ✓ {len(df)} rows after cleaning")
    
    # Delete unwanted columns
    print(f"Deleting columns: {DELETE_COLUMNS}...")
    df = df.drop(columns=DELETE_COLUMNS)
    print(f"  ✓ Columns remaining: {len(df.columns)}")
    
    # Load team index
    print("Loading team index...")
    team_index_map = load_team_index()
    print(f"  ✓ Loaded {len(team_index_map)} team mappings")
    
    # Add team index
    print("Adding team indices...")
    df = add_team_index(df, team_index_map)
    print(f"  ✓ Matched {df['Index'].notna().sum()} teams")
    
    # Reorder columns (moves Rk after Team)
    print("Reordering columns...")
    df = reorder_columns(df)
    print(f"  ✓ New order: {', '.join(df.columns[:5])}...")
    
    # Prefix columns
    print("Adding 'kenpom_' prefix...")
    df = prefix_columns(df)
    print(f"  ✓ Prefixed columns (sample): {', '.join([c for c in df.columns if c.startswith('kenpom_')][:3])}...")
    
    # Convert to numeric
    print("Converting columns to numeric (except Team)...")
    df = convert_to_numeric(df)
    print(f"  ✓ Converted {len([c for c in df.columns if c != 'Team'])} columns to numeric")
    
    # Split into predict (2026) and analyze (historical)
    print(f"Splitting data: {CURRENT_YEAR} -> predict, rest -> analyze...")
    df_predict = df[df['Year'] == CURRENT_YEAR].copy()
    df_analyze = df[df['Year'] != CURRENT_YEAR].copy()
    print(f"  ✓ Predict: {len(df_predict)} rows (year {CURRENT_YEAR})")
    print(f"  ✓ Analyze: {len(df_analyze)} rows (years {df_analyze['Year'].min()}-{df_analyze['Year'].max()})")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save outputs
    print("Saving transformed datasets...")
    df_predict.to_csv(PREDICT_FILE, index=False)
    print(f"  ✓ Saved: {PREDICT_FILE}")
    df_analyze.to_csv(ANALYZE_FILE, index=False)
    print(f"  ✓ Saved: {ANALYZE_FILE}")
    
    print("="*60)
    print("Transform complete!")
    print("="*60)


if __name__ == "__main__":
    transform_data()