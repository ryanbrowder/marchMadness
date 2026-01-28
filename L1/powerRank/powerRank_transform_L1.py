"""
Power Rank L1 Transform
Transforms raw Power Rank data from L0 to clean, standardized L2 format
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_team_lookup(lookup_path: str) -> pd.DataFrame:
    """Load team index from CSV"""
    return pd.read_csv(lookup_path)


def standardize_team_name(raw_name: str, lookup_df: pd.DataFrame) -> str:
    """
    Standardize a raw team name using the lookup table
    
    Args:
        raw_name: Original team name from Power Rank data
        lookup_df: DataFrame with 'Team' and 'Index' columns (includes all variants)
        
    Returns:
        Standardized team name or original if no match found
    """
    if pd.isna(raw_name):
        return None
    
    raw_name = raw_name.strip()
    
    # Direct lookup - all variants are now rows in the table
    match = lookup_df[lookup_df['Team'] == raw_name]
    if not match.empty:
        return match.iloc[0]['Team']
    
    # No match found - return original
    return raw_name


def fix_encoding_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Fix common encoding issues in team names"""
    df = df.copy()
    
    # Common encoding replacements
    encoding_fixes = {
        '‚Äô': "'",  # Smart apostrophe
        '√©': "é",   # e with accent
        '√°': "à",   # a with grave
        '√≠': "í",   # i with accent
        '√±': "ñ",   # n with tilde
        '‚Äì': "-",  # Em dash
        '√ò': "ó",   # o with accent
    }
    
    for bad, good in encoding_fixes.items():
        df['Team'] = df['Team'].str.replace(bad, good, regex=False)
    
    return df


def clean_powerrank_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize Power Rank data"""
    df = df.copy()
    
    # Fix encoding issues
    df = fix_encoding_issues(df)
    
    # Strip records from team names (e.g., "Duke (15-2)" -> "Duke")
    df['Team'] = df['Team'].str.replace(r'\s*\([^)]*\)\s*$', '', regex=True)
    
    # Strip whitespace from team names
    df['Team'] = df['Team'].str.strip()
    
    # Convert PowerRank to float (handles both integers and decimals)
    df['PowerRank'] = pd.to_numeric(df['PowerRank'], errors='coerce')
    
    # Convert Year to integer
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['Team', 'Year'], keep='first')
    
    return df


def add_team_standardization(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized team names and IDs using lookup table"""
    df = df.copy()
    
    # Standardize team names
    df['Team'] = df['Team'].apply(lambda x: standardize_team_name(x, lookup_df))
    
    # Add Index from lookup
    team_to_index = lookup_df.set_index('Team')['Index'].to_dict()
    df['Index'] = df['Team'].map(team_to_index)
    
    # Flag unmatched teams
    unmatched = df[df['Index'].isna()]['Team'].unique()
    if len(unmatched) > 0:
        print(f"\nWarning: {len(unmatched)} unmatched teams:")
        for team in sorted(unmatched):
            print(f"  - {team}")
    
    return df


def add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add data quality flag for validation"""
    df = df.copy()
    
    # Add data quality flag (True if all required fields present)
    df['DataQuality'] = (
        df['Index'].notna() & 
        df['PowerRank'].notna() & 
        df['Year'].notna()
    )
    
    return df


def reorder_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder columns for final output"""
    
    final_columns = [
        'Year',
        'Team',
        'Index',
        'PowerRank'
    ]
    
    return df[final_columns]


def transform_powerrank_data(input_path: str, output_path: str, lookup_path: str):
    """
    Main transformation pipeline for Power Rank data
    
    Args:
        input_path: Path to raw Power Rank CSV (L1)
        output_path: Path for cleaned output CSV (L2)
        lookup_path: Path to team lookup CSV (teamsIndex.csv with 'Team' and 'Index' columns)
    """
    
    print("Starting Power Rank transformation...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_path, encoding='utf-8-sig')  # Handle BOM
    print(f"   Loaded {len(df)} rows")
    
    # Load team lookup
    lookup_df = load_team_lookup(lookup_path)
    print(f"   Loaded {len(lookup_df)} teams in lookup table")
    
    # Clean data
    print("\n2. Cleaning data...")
    initial_rows = len(df)
    df = clean_powerrank_data(df)
    print(f"   Removed {initial_rows - len(df)} duplicate rows")
    print(f"   Remaining rows: {len(df)}")
    
    # Add team standardization
    print("\n3. Standardizing team names...")
    df = add_team_standardization(df, lookup_df)
    matched_teams = df['Index'].notna().sum()
    print(f"   Matched {matched_teams}/{len(df)} rows to lookup table")
    
    # Add metadata
    print("\n4. Adding metadata columns...")
    df = add_metadata_columns(df)
    
    # Data quality summary (before removing DataQuality column)
    print("\n5. Data Quality Summary:")
    print(f"   Total rows: {len(df)}")
    print(f"   Clean rows: {df['DataQuality'].sum()}")
    print(f"   Incomplete rows: {(~df['DataQuality']).sum()}")
    print(f"   Years covered: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Unique teams: {df['Index'].nunique()}")
    
    # Final column selection and sorting
    print("\n6. Finalizing output...")
    df = reorder_and_select_columns(df)
    df = df.sort_values(['Year', 'PowerRank']).reset_index(drop=True)
    
    # PowerRank statistics by year
    print("\n7. PowerRank Statistics by Year:")
    year_stats = df.groupby('Year')['PowerRank'].agg(['count', 'min', 'max', 'mean'])
    print(year_stats.to_string())
    
    # Save output
    print(f"\n8. Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("   Done!")
    
    return df


if __name__ == "__main__":
    # Define paths (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up from L1/powerRank to project root
    
    # Input files
    historical_file = project_root / "L1" / "data" / "powerRank" / "powerRank_raw_L1.csv"
    current_file = project_root / "L1" / "data" / "powerRank" / "powerRank_rawCurrent_L1.csv"
    
    # Output files
    analyze_file = project_root / "L2" / "data" / "powerRank" / "powerRank_analyze_L2.csv"
    predict_file = project_root / "L2" / "data" / "powerRank" / "powerRank_predict_L2.csv"
    
    # Team lookup
    lookup_file = script_dir / ".." / ".." / "utils" / "teamsIndex.csv"
    
    # Transform historical data (2016-2024)
    print("="*60)
    print("TRANSFORMING HISTORICAL DATA (2016-2024)")
    print("="*60)
    df_analyze = transform_powerrank_data(
        str(historical_file),
        str(analyze_file),
        str(lookup_file)
    )
    
    print("\n" + "="*60)
    print("TRANSFORMING CURRENT SEASON DATA (2026)")
    print("="*60)
    
    # Transform current season data (2026)
    df_predict = transform_powerrank_data(
        str(current_file),
        str(predict_file),
        str(lookup_file)
    )
    
    print("\n" + "="*60)
    print("TRANSFORMATION SUMMARY")
    print("="*60)
    print(f"✓ Historical data: {len(df_analyze)} rows → {analyze_file}")
    print(f"✓ Current season: {len(df_predict)} rows → {predict_file}")
    print("\n✓ All transformations complete!")
