"""
ESPN BPI Transform Script (L1 → L2)

Concatenates historical and current ESPN BPI data, then transforms to analysis-ready format.
Reads: espnBPI_historical_raw.csv + espnBPI_current_raw.csv
Handles column structure: Team, BPI, BPI_Rk, Off, Def, Year
Renames to: Team, BPI, BPI_Rank, BPI_Off, BPI_Def, Year

Splits data:
- Year <= (CURRENT_YEAR - 1) → espnBPI_analyze_L2.csv (historical training data)
- Year == CURRENT_YEAR → espnBPI_predict_L2.csv (current season for predictions)

Outputs:
- L2/data/espnBPI/espnBPI_analyze_L2.csv (historical)
- L2/data/espnBPI/espnBPI_predict_L2.csv (current)
"""

import sys
import os

# Add utils to path for CURRENT_YEAR
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from utils import CURRENT_YEAR

import pandas as pd
import re

def clean_team_name(name):
    """
    Clean team name for standardization.
    
    Args:
        name: Raw team name string
        
    Returns:
        Cleaned team name string
    """
    if pd.isna(name):
        return name
    
    # Strip leading/trailing whitespace
    name = name.strip()
    
    # ESPN BPI uses abbreviated names (DUKE, ARIZ) - no additional cleaning needed
    
    return name

def load_teams_index(index_path='../../utils/teamsIndex.csv'):
    """
    Load the unified team index for standardization.
    
    Args:
        index_path: Path to teamsIndex.csv (relative to L1/espnBPI/)
        
    Returns:
        DataFrame with Team and Index columns
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"teamsIndex.csv not found at {index_path}")
    
    teams_index = pd.read_csv(index_path)
    
    # teamsIndex uses vertical expansion: same Index appears multiple times
    # for different name variations of the same team
    print(f"Loaded teamsIndex: {len(teams_index)} rows (includes name variants)")
    
    return teams_index

def transform_espn_bpi(df, teams_index, dataset_name):
    """
    Transform ESPN BPI data.
    
    Args:
        df: Raw ESPN BPI DataFrame
        teams_index: Team standardization lookup
        dataset_name: Name for logging (e.g., "historical" or "current")
        
    Returns:
        Cleaned DataFrame with Index column added
    """
    print(f"\nProcessing {dataset_name} data...")
    print(f"  Input rows: {len(df)}")
    print(f"  Input columns: {df.columns.tolist()}")
    
    # Clean team names
    df['Team'] = df['Team'].apply(clean_team_name)
    
    # Merge with teamsIndex to get Index
    df = df.merge(teams_index[['Team', 'Index']], on='Team', how='left')
    
    # Check for unmatched teams
    unmatched = df[df['Index'].isna()]
    if len(unmatched) > 0:
        print(f"\n  WARNING: {len(unmatched)} teams not found in teamsIndex:")
        unique_unmatched = unmatched['Team'].unique()
        for team in unique_unmatched[:20]:  # Show first 20
            print(f"    - {team}")
        if len(unique_unmatched) > 20:
            print(f"    ... and {len(unique_unmatched) - 20} more")
        print("  Add these variants to teamsIndex.csv and re-run")
    
    matched_pct = (1 - len(unmatched) / len(df)) * 100
    print(f"  Match rate: {matched_pct:.1f}%")
    
    # Rename columns for clarity
    df = df.rename(columns={
        'BPI_Rk': 'BPI_Rank',
        'Off': 'BPI_Off',
        'Def': 'BPI_Def'
    })
    
    # Ensure all columns are numeric except Team
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['Index'] = pd.to_numeric(df['Index'], errors='coerce').astype('Int64')
    df['BPI'] = pd.to_numeric(df['BPI'], errors='coerce')
    df['BPI_Rank'] = pd.to_numeric(df['BPI_Rank'], errors='coerce').astype('Int64')
    df['BPI_Off'] = pd.to_numeric(df['BPI_Off'], errors='coerce')
    df['BPI_Def'] = pd.to_numeric(df['BPI_Def'], errors='coerce')
    
    # Reorder columns: Year, Team, Index, BPI, BPI_Rank, BPI_Off, BPI_Def
    df = df[['Year', 'Team', 'Index', 'BPI', 'BPI_Rank', 'BPI_Off', 'BPI_Def']]
    
    # Sort by Year, then BPI rank (ascending = best teams first)
    df = df.sort_values(['Year', 'BPI_Rank']).reset_index(drop=True)
    
    print(f"  Output rows: {len(df)}")
    print(f"  Years: {sorted(df['Year'].unique())}")
    
    return df

def main():
    """Main execution function."""
    
    print("="*60)
    print("ESPN BPI Transform (L1 → L2)")
    print("="*60)
    
    # Define paths (relative to L1/espnBPI/)
    historical_path = '../data/espnBPI/espnBPI_historical_raw.csv'
    current_path = '../data/espnBPI/espnBPI_current_raw.csv'
    teams_index_path = '../../utils/teamsIndex.csv'
    output_dir = '../../L2/data/espnBPI'
    analyze_output = os.path.join(output_dir, 'espnBPI_analyze_L2.csv')
    predict_output = os.path.join(output_dir, 'espnBPI_predict_L2.csv')
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load team index
    print("\nLoading team index...")
    teams_index = load_teams_index(teams_index_path)
    
    # Load and concatenate raw data files
    print("\nLoading raw data files...")
    
    dfs_to_concat = []
    
    # Load historical data
    if os.path.exists(historical_path):
        df_historical_raw = pd.read_csv(historical_path)
        dfs_to_concat.append(df_historical_raw)
        print(f"  ✓ Historical: {len(df_historical_raw)} rows from {historical_path}")
        print(f"    Years: {sorted(df_historical_raw['Year'].unique())}")
    else:
        print(f"  ⚠ Historical file not found: {historical_path}")
    
    # Load current data
    if os.path.exists(current_path):
        df_current_raw = pd.read_csv(current_path)
        dfs_to_concat.append(df_current_raw)
        print(f"  ✓ Current: {len(df_current_raw)} rows from {current_path}")
        print(f"    Year: {df_current_raw['Year'].unique()[0]}")
    else:
        print(f"  ⚠ Current file not found: {current_path}")
    
    # Check if we have any data
    if len(dfs_to_concat) == 0:
        print("\nERROR: No data files found!")
        print(f"  Expected: {historical_path}")
        print(f"  Expected: {current_path}")
        return
    
    # Concatenate all data
    df_raw = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"\n  ✓ Combined: {len(df_raw)} total rows")
    print(f"    Years present: {sorted(df_raw['Year'].unique())}")
    print(f"    Columns: {df_raw.columns.tolist()}")
    
    # Split data: Historical (up to CURRENT_YEAR-1) vs Current (CURRENT_YEAR)
    df_historical = df_raw[df_raw['Year'] < CURRENT_YEAR].copy()
    df_current = df_raw[df_raw['Year'] == CURRENT_YEAR].copy()
    
    print(f"\nData split (CURRENT_YEAR: {CURRENT_YEAR}):")
    print(f"  Historical (up to {CURRENT_YEAR-1}): {len(df_historical)} rows")
    print(f"  Current ({CURRENT_YEAR}): {len(df_current)} rows")
    
    # Process historical data (for training/validation)
    print("\n" + "-"*60)
    print(f"HISTORICAL DATA (up to {CURRENT_YEAR-1})")
    print("-"*60)
    
    if len(df_historical) > 0:
        df_historical_clean = transform_espn_bpi(df_historical, teams_index, "historical")
        
        # Save to analyze file
        df_historical_clean.to_csv(analyze_output, index=False)
        print(f"\n✓ Saved to: {analyze_output}")
    else:
        print("No historical data to process")
        df_historical_clean = None
    
    # Process current season data (for predictions)
    print("\n" + "-"*60)
    print(f"CURRENT SEASON DATA ({CURRENT_YEAR})")
    print("-"*60)
    
    if len(df_current) > 0:
        df_current_clean = transform_espn_bpi(df_current, teams_index, "current")
        
        # Save to predict file
        df_current_clean.to_csv(predict_output, index=False)
        print(f"\n✓ Saved to: {predict_output}")
    else:
        print("No current season data to process")
        df_current_clean = None
    
    # Summary
    print("\n" + "="*60)
    print("TRANSFORM COMPLETE")
    print("="*60)
    
    if df_historical_clean is not None:
        print(f"Analyze file: {len(df_historical_clean)} rows → {analyze_output}")
        print(f"  Years: {sorted(df_historical_clean['Year'].unique())}")
    
    if df_current_clean is not None:
        print(f"Predict file: {len(df_current_clean)} rows → {predict_output}")
    
    print("\nOutput schema: Year, Team, Index, BPI, BPI_Rank, BPI_Off, BPI_Def")
    print("\nReady for feature engineering and modeling!")
    print("All data sources join on (Index, Year)")

if __name__ == "__main__":
    main()
