"""
Bart Torvik L1 Transform
Cleans and standardizes raw Torvik data for downstream use
"""

import pandas as pd
import sys
import os
import re

# Add the marchMadness directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.utils import CURRENT_YEAR


def prefix_columns(df: pd.DataFrame, prefix: str = "bartTorvik_", 
                   exclude_cols: list = None) -> pd.DataFrame:
    """
    Add source prefix to column names except specified columns.
    
    Args:
        df: Input DataFrame
        prefix: Prefix to add to column names
        exclude_cols: List of columns to not prefix (join keys)
    
    Returns:
        DataFrame with prefixed columns
    """
    if exclude_cols is None:
        exclude_cols = ['Team', 'Year', 'Index', 'tournamentSeed', 'tournamentOutcome']
    
    rename_map = {
        col: f"{prefix}{col}" 
        for col in df.columns 
        if col not in exclude_cols
    }
    
    return df.rename(columns=rename_map)


def remove_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where Rk column contains 'Rk' (header rows mixed in data)."""
    df = df.copy()
    if 'Rk' in df.columns:
        df = df[df['Rk'] != 'Rk']
    return df


def parse_team_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Team column into Team, tournamentSeed, and tournamentOutcome."""
    df = df.copy()
    
    if 'Team' not in df.columns:
        return df
    
    def parse_team_info(team_string):
        """Parse team string like 'Kansas1 seed,CHAMPS' or 'Michigan(H) 13 Nebraska' into components."""
        if pd.isna(team_string) or team_string == '':
            return team_string, None, None
        
        # First, remove game info: (H) or (A) and everything after it
        # "Michigan(H) 13 Nebraska" becomes "Michigan"
        team_string = re.sub(r'\(H\).*$', '', team_string)
        team_string = re.sub(r'\(A\).*$', '', team_string)
        team_string = team_string.strip()
        
        # Then parse tournament info
        # Pattern: TeamName + Digit + " seed," + Outcome
        # Example: "Kansas1 seed,CHAMPS"
        match = re.match(r'^(.+?)(\d+)\s*seed,(.+)$', team_string)
        
        if match:
            team_name = match.group(1).strip()
            seed = int(match.group(2))
            outcome = match.group(3).strip()
            return team_name, seed, outcome
        else:
            # No tournament info, just return team name
            return team_string, None, None
    
    df[['Team', 'tournamentSeed', 'tournamentOutcome']] = df['Team'].apply(
        lambda x: pd.Series(parse_team_info(x))
    )
    
    return df


def add_team_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add team index from teamsIndex.csv lookup."""
    df = df.copy()
    
    if 'Team' not in df.columns:
        return df
    
    # Load team index lookup table
    teams_index_path = os.path.join(os.path.dirname(__file__), '../../utils/teamsIndex.csv')
    
    if not os.path.exists(teams_index_path):
        print(f"  ⚠ Warning: teamsIndex.csv not found at {teams_index_path}")
        return df
    
    teams_index = pd.read_csv(teams_index_path)
    
    # Merge on Team
    df = df.merge(teams_index[['Team', 'Index']], on='Team', how='left')
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on Team and Year."""
    df = df.copy()
    
    initial_count = len(df)
    
    # Drop duplicates keeping first occurrence
    df = df.drop_duplicates(subset=['Team', 'Year'], keep='first')
    
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0:
        print(f"  ⚠ Removed {duplicates_removed} duplicate Team-Year combinations")
    
    return df


def reorder_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to place Index, tournamentSeed and tournamentOutcome after Team."""
    df = df.copy()
    
    if 'Team' not in df.columns:
        return df
    
    # Get all columns
    cols = df.columns.tolist()
    
    # Remove Index, tournamentSeed and tournamentOutcome from their current positions
    if 'Index' in cols:
        cols.remove('Index')
    if 'tournamentSeed' in cols:
        cols.remove('tournamentSeed')
    if 'tournamentOutcome' in cols:
        cols.remove('tournamentOutcome')
    
    # Find position of Team and insert after it
    team_index = cols.index('Team')
    if 'Index' in df.columns:
        cols.insert(team_index + 1, 'Index')
        team_index += 1  # Shift position for next inserts
    if 'tournamentSeed' in df.columns:
        cols.insert(team_index + 1, 'tournamentSeed')
    if 'tournamentOutcome' in df.columns:
        cols.insert(team_index + 2, 'tournamentOutcome')
    
    return df[cols]


def split_record_column(df: pd.DataFrame) -> pd.DataFrame:
    """Extract overall wins from record."""
    df = df.copy()
    
    if 'Rec' not in df.columns:
        return df
    
    def parse_record(rec_string):
        """Parse record string to extract overall wins."""
        if pd.isna(rec_string) or rec_string == '':
            return None
        
        # Replace various dash encodings with standard dash
        rec_string = rec_string.replace('–', '-').replace('—', '-').replace('‚Äì', '-')
        
        # Look for pattern: digits-digits (wins-losses)
        matches = re.findall(r'(\d+)-(\d+)', rec_string)
        
        if len(matches) >= 1:
            # First match is overall record
            overall_wins = int(matches[0][0])
            return overall_wins
        else:
            return None
    
    df['Wins_Overall'] = df['Rec'].apply(parse_record)
    
    # Drop original Rec column
    df = df.drop(columns=['Rec'])
    
    return df


def reorder_wins_column(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to place Wins_Overall after G."""
    df = df.copy()
    
    if 'Wins_Overall' not in df.columns or 'G' not in df.columns:
        return df
    
    # Get all columns
    cols = df.columns.tolist()
    
    # Remove Wins_Overall from its current position
    cols.remove('Wins_Overall')
    
    # Find position of G and insert Wins_Overall after it
    g_index = cols.index('G')
    cols.insert(g_index + 1, 'Wins_Overall')
    
    return df[cols]


def reorder_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns: Year first, bartTorvik_Rk after tournamentOutcome."""
    df = df.copy()
    
    # Get all columns
    cols = df.columns.tolist()
    
    # Remove Year and bartTorvik_Rk from current positions
    if 'Year' in cols:
        cols.remove('Year')
    if 'bartTorvik_Rk' in cols:
        cols.remove('bartTorvik_Rk')
    
    # Insert Year at the beginning
    if 'Year' in df.columns:
        cols.insert(0, 'Year')
    
    # Insert bartTorvik_Rk after tournamentOutcome
    if 'bartTorvik_Rk' in df.columns and 'tournamentOutcome' in cols:
        outcome_index = cols.index('tournamentOutcome')
        cols.insert(outcome_index + 1, 'bartTorvik_Rk')
    
    return df[cols]


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    df = df.copy()
    
    # Columns that should remain as strings
    string_cols = ['Team', 'tournamentOutcome', 'Conf']
    
    # Columns that should be integers
    integer_cols = ['Year', 'Index', 'tournamentSeed', 'Rk', 'G', 'Wins_Overall']
    
    # Convert integer columns
    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Convert all other columns to float (except string columns)
    for col in df.columns:
        if col not in string_cols and col not in integer_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def clean_torvik_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Torvik raw data.
    
    Args:
        df: Raw Torvik DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Apply cleaning steps in order
    df = remove_header_rows(df)
    df = parse_team_column(df)
    df = add_team_index(df)
    df = remove_duplicates(df)
    df = reorder_team_columns(df)
    df = split_record_column(df)
    df = reorder_wins_column(df)
    df = convert_data_types(df)
    
    return df


def save_tournament_results(df: pd.DataFrame, output_base_dir: str) -> None:
    """
    Save tournament results view with Team, Index, tournamentSeed, tournamentOutcome, Year.
    
    Args:
        df: Cleaned DataFrame (before prefixing)
        output_base_dir: Base L2/data directory
    """
    # Select only tournament columns
    results_cols = ['Team', 'Index', 'tournamentSeed', 'tournamentOutcome', 'Year']
    
    # Filter to only teams that made the tournament (have seed data)
    tournament_teams = df[df['tournamentSeed'].notna()].copy()
    
    # Select columns
    results_df = tournament_teams[results_cols]
    
    # Save to L2/data/ (not in bartTorvik subfolder)
    output_path = os.path.join(output_base_dir, 'tournamentResults.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"  ✓ Tournament results: {output_path}")
    print(f"    • {len(results_df)} tournament teams across {results_df['Year'].nunique()} years")


def split_predict_analyze(df: pd.DataFrame, 
                          current_year: int = CURRENT_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into prediction (current year) and analysis (historical) sets.
    
    Args:
        df: Combined DataFrame
        current_year: Year to use for prediction set
    
    Returns:
        Tuple of (predict_df, analyze_df)
    """
    predict_df = df[df['Year'] == current_year].copy()
    analyze_df = df[df['Year'] < current_year].copy()
    
    return predict_df, analyze_df


def transform_torvik(input_file: str = None,
                     predict_output: str = None,
                     analyze_output: str = None) -> None:
    """
    Main transform function: read raw, clean, prefix, and split.
    
    Args:
        input_file: Path to raw CSV file
        predict_output: Path for prediction output
        analyze_output: Path for analysis output
    """
    # Set default paths relative to this file
    input_dir = os.path.join(os.path.dirname(__file__), '../data/bartTorvik')
    output_dir = os.path.join(os.path.dirname(__file__), '../../L2/data/bartTorvik')
    output_base_dir = os.path.join(os.path.dirname(__file__), '../../L2/data')
    
    if input_file is None:
        input_file = os.path.join(input_dir, 'bartTorvik_raw_L1.csv')
    if predict_output is None:
        predict_output = os.path.join(output_dir, 'bartTorvik_predict_L2.csv')
    if analyze_output is None:
        analyze_output = os.path.join(output_dir, 'bartTorvik_analyze_L2.csv')
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("="*70)
    print("Torvik L1 Transform")
    print("="*70)
    print()
    
    # Read raw data
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Clean data
    print("Cleaning data...")
    df = clean_torvik_data(df)
    print(f"  ✓ Cleaned")
    
    # Save tournament results BEFORE prefixing
    print("\nSaving tournament results...")
    save_tournament_results(df, output_base_dir)
    
    # Prefix columns
    print("\nPrefixing columns...")
    df = prefix_columns(df, prefix="bartTorvik_", exclude_cols=['Team', 'Year', 'Index', 'tournamentSeed', 'tournamentOutcome'])
    print(f"  ✓ Prefixed (kept Team, Year, Index, tournamentSeed, tournamentOutcome unprefixed)")
    
    # Reorder final columns
    print("Reordering columns...")
    df = reorder_final_columns(df)
    print(f"  ✓ Reordered (Year first, bartTorvik_Rk after tournamentOutcome)")
    
    # Split into predict and analyze
    print(f"\nSplitting data (current year: {CURRENT_YEAR})...")
    predict_df, analyze_df = split_predict_analyze(df)
    print(f"  ✓ Predict: {len(predict_df)} rows (year {CURRENT_YEAR})")
    print(f"  ✓ Analyze: {len(analyze_df):,} rows (years {analyze_df['Year'].min()}-{analyze_df['Year'].max()})")
    
    # Save outputs
    print("\nSaving outputs...")
    predict_df.to_csv(predict_output, index=False)
    print(f"  ✓ {predict_output}")
    
    analyze_df.to_csv(analyze_output, index=False)
    print(f"  ✓ {analyze_output}")
    
    print()
    print("="*70)
    print("Transform complete")
    print("="*70)
    
    # Show sample
    print("\nPredict dataset sample:")
    print(predict_df.head())
    
    print("\nAnalyze dataset sample:")
    print(analyze_df.head())
    
    print(f"\nColumns: {predict_df.columns.tolist()}")


if __name__ == "__main__":
    transform_torvik()