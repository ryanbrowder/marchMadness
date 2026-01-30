"""
LRMCB Transform Layer (L1)

Transforms raw LRMCB data into clean, standardized format for modeling.

Input:
    - L1/data/LRMCB/LRMCB_raw_L1.csv (all years including current)

Output:
    - L2/data/LRMCB/LRMCB_analyze_L2.csv (all years: 2016-2025, excluding 2020)

Processing:
    1. Clean team names (records, special characters, spacing)
    2. Match to unified team index
    3. Output all data (2016-2025, excluding 2020) to analyze file
"""

import pandas as pd
import re
import os

# Define paths
INPUT_FILE = '../data/LRMCB/LRMCB_raw_L1.csv'
OUTPUT_DIR = '../../L2/data/LRMCB'
OUTPUT_ANALYZE = os.path.join(OUTPUT_DIR, 'LRMCB_analyze_L2.csv')
TEAMS_INDEX = '../../utils/teamsIndex.csv'

# All years to include (excluding 2020 - tournament cancelled)
ALL_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

def clean_team_name(team_name):
    """
    Clean team name to match teamsIndex format.
    
    Steps:
        1. Strip any records: "Duke (15-2)" -> "Duke"
        2. Fix special characters (smart quotes, en-dashes)
        3. Strip whitespace
    """
    if pd.isna(team_name):
        return team_name
    
    # Convert to string
    team = str(team_name)
    
    # Strip records: anything in parentheses with optional whitespace
    team = re.sub(r'\s*\([^)]*\)', '', team)
    
    # Replace smart apostrophes with regular apostrophes
    team = team.replace('\u2019', "'")  # U+2019 -> U+0027
    
    # Replace en-dashes with hyphens
    team = team.replace('\u2013', '-')  # U+2013 -> U+002D
    
    # Strip leading/trailing whitespace
    team = team.strip()
    
    return team

def match_to_index(df, teams_index_df):
    """
    Match team names to unified index using teamsIndex lookup.
    
    Returns:
        DataFrame with Index column added
    """
    # Clean team names
    df['Team_Clean'] = df['Team'].apply(clean_team_name)
    
    # Merge with teams index
    df_matched = df.merge(
        teams_index_df[['Team', 'Index']],
        left_on='Team_Clean',
        right_on='Team',
        how='left',
        suffixes=('', '_index')
    )
    
    # Check for unmatched teams
    unmatched = df_matched[df_matched['Index'].isna()]
    if len(unmatched) > 0:
        print(f"\nWARNING: {len(unmatched)} unmatched teams found:")
        print(unmatched[['Year', 'Team_Clean', 'LRMCB']].head(20))
        print("\nThese teams need to be added to teamsIndex.csv")
    
    # Drop temporary columns and reorder
    df_final = df_matched[['Year', 'Team_Clean', 'Index', 'LRMCB']].copy()
    df_final.rename(columns={'Team_Clean': 'Team'}, inplace=True)
    
    # Ensure all non-Team columns are numeric
    df_final['Year'] = pd.to_numeric(df_final['Year'], errors='coerce').astype('Int64')
    df_final['Index'] = pd.to_numeric(df_final['Index'], errors='coerce').astype('Int64')
    df_final['LRMCB'] = pd.to_numeric(df_final['LRMCB'], errors='coerce')
    
    return df_final

def main():
    """Main transformation pipeline."""
    
    print("=" * 60)
    print("LRMCB Transform Layer (L1)")
    print("=" * 60)
    
    # Load raw data
    print("\n1. Loading raw LRMCB data...")
    df_raw = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')  # Handle BOM
    print(f"   Loaded {len(df_raw)} rows from {os.path.basename(INPUT_FILE)}")
    print(f"   Years: {sorted(df_raw['Year'].unique())}")
    
    # Load teams index
    print("\n2. Loading teams index...")
    teams_index = pd.read_csv(TEAMS_INDEX)
    print(f"   Loaded {len(teams_index)} team name variants")
    print(f"   Unique teams: {teams_index['Index'].nunique()}")
    
    # Match to index
    print("\n3. Matching teams to unified index...")
    df_matched = match_to_index(df_raw, teams_index)
    
    # Calculate match rate
    match_rate = (1 - df_matched['Index'].isna().sum() / len(df_matched)) * 100
    print(f"   Match rate: {match_rate:.1f}%")
    
    # Filter for valid years
    print("\n4. Filtering for valid years...")
    
    df_final = df_matched[df_matched['Year'].isin(ALL_YEARS)].copy()
    
    print(f"   Valid years: {ALL_YEARS}")
    print(f"   Total rows: {len(df_final)}")
    print(f"   Years present: {sorted(df_final['Year'].unique())}")
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save output
    print("\n5. Saving output...")
    
    df_final.to_csv(OUTPUT_ANALYZE, index=False)
    print(f"   ✓ {OUTPUT_ANALYZE}")
    print(f"     {len(df_final)} rows (all data)")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows processed: {len(df_raw)}")
    print(f"Valid years (excl. 2020): {len(df_final)}")
    print(f"Match rate: {match_rate:.1f}%")
    
    # Data type validation
    print("\nData Types:")
    print(f"  Year: {df_final['Year'].dtype}")
    print(f"  Team: {df_final['Team'].dtype}")
    print(f"  Index: {df_final['Index'].dtype}")
    print(f"  LRMCB: {df_final['LRMCB'].dtype}")
    
    # Preview
    print("\nSample Output (first 3 rows):")
    print(df_final.head(3).to_string(index=False))
    
    if match_rate < 100:
        print("\n⚠ WARNING: Not all teams matched to index!")
        print("  Add missing team name variants to teamsIndex.csv")
    else:
        print("\n✓ All teams successfully matched!")
        print(f"✓ All data output to: {OUTPUT_ANALYZE}")
    
    print("=" * 60)

if __name__ == '__main__':
    main()