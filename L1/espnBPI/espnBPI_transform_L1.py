"""
ESPN BPI Transform Script (L1 → L2)

Transforms ESPN BPI data from raw format to analysis-ready format.
Handles column structure: Team, BPI, BPI_Rk, Off, Def, Year
Renames to: Team, BPI, BPI_Rank, BPI_Off, BPI_Def, Year

Splits data:
- Year 2026 → espnBPI_predict_L2.csv (current season for predictions)
- Years 2008-2025 → espnBPI_analyze_L2.csv (historical data for training)

Outputs:
- L2/data/espnBPI/espnBPI_analyze_L2.csv (historical)
- L2/data/espnBPI/espnBPI_predict_L2.csv (2026)
"""

import pandas as pd
import os
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
    input_path = '../data/espnBPI/espnBPI_raw_L1.csv'
    teams_index_path = '../../utils/teamsIndex.csv'
    output_dir = '../../L2/data/espnBPI'
    analyze_output = os.path.join(output_dir, 'espnBPI_analyze_L2.csv')
    predict_output = os.path.join(output_dir, 'espnBPI_predict_L2.csv')
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load team index
    print("\nLoading team index...")
    teams_index = load_teams_index(teams_index_path)
    
    # Load raw data
    print("\nLoading raw data...")
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    df_raw = pd.read_csv(input_path)
    print(f"Loaded {len(df_raw)} rows from {input_path}")
    print(f"Years present: {sorted(df_raw['Year'].unique())}")
    print(f"Columns: {df_raw.columns.tolist()}")
    
    # Split data: 2026 (predict) vs 2008-2025 (analyze)
    df_historical = df_raw[df_raw['Year'] <= 2025].copy()
    df_current = df_raw[df_raw['Year'] == 2026].copy()
    
    print(f"\nData split:")
    print(f"  Historical (2008-2025): {len(df_historical)} rows")
    print(f"  Current (2026): {len(df_current)} rows")
    
    # Process historical data (for training/validation)
    print("\n" + "-"*60)
    print("HISTORICAL DATA (for training/validation)")
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
    print("CURRENT SEASON DATA (2026 - for predictions)")
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
