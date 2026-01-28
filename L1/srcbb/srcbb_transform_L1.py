#!/usr/bin/env python3
"""
Sports Reference Basketball Tournament Games - L1 Transform
Maps team names to teamsIndex.csv and prepares data for L2 analysis

Input:  ../data/srcbb/srcbb_transform_L1.csv (raw scraped data)
Reference: ../../utils/teamsIndex.csv (team name standardization)
Output: ../../L2/data/srcbb/srcbb_analyze_L2.csv (clean, standardized data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Paths
INPUT_FILE = '../data/srcbb/srcbb_transform_L1.csv'
TEAMS_INDEX = '../../utils/teamsIndex.csv'
OUTPUT_DIR = '../../L2/data/srcbb'
OUTPUT_FILE = f'{OUTPUT_DIR}/srcbb_analyze_L2.csv'


def load_data():
    """Load raw scraped data and teams index"""
    print("Loading data...")
    
    # Load scraped tournament games
    df = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Loaded {len(df)} games from {INPUT_FILE}")
    
    # Load teams index
    teams_index = pd.read_csv(TEAMS_INDEX)
    print(f"  ✓ Loaded {len(teams_index)} rows from {TEAMS_INDEX}")
    
    return df, teams_index


def build_team_mapping(teams_index):
    """
    Build mapping dictionary from team names to Index
    Structure: Team,Index where multiple rows can have same Index
    """
    print("\nBuilding team name mapping...")
    
    mapping = {}
    
    # Iterate through all rows
    for _, row in teams_index.iterrows():
        team_name = str(row['Team']).strip()
        team_index = row['Index']
        
        if team_name and team_name != '':
            mapping[team_name] = team_index
    
    print(f"  ✓ Created mapping for {len(mapping)} team name variants")
    print(f"  ✓ Mapping to {len(set(mapping.values()))} unique team indices")
    
    return mapping


def map_team_names(df, mapping):
    """Map team names to standardized TeamIDs"""
    print("\nMapping team names to TeamIDs...")
    
    # Create new columns for TeamIDs
    df['TeamA_ID'] = df['TeamA'].map(mapping)
    df['TeamB_ID'] = df['TeamB'].map(mapping)
    
    # Check for unmapped teams
    unmapped_a = df[df['TeamA_ID'].isna()]['TeamA'].unique()
    unmapped_b = df[df['TeamB_ID'].isna()]['TeamB'].unique()
    unmapped = sorted(set(list(unmapped_a) + list(unmapped_b)))
    
    if unmapped:
        print(f"\n  ⚠️  WARNING: {len(unmapped)} unmapped teams found:")
        
        # Show first 30 with their closest matches
        for i, team in enumerate(unmapped[:30]):
            # Find close matches in mapping
            close_matches = [k for k in mapping.keys() if team.lower() in k.lower() or k.lower() in team.lower()]
            if close_matches:
                print(f"      - '{team}' (similar in index: {close_matches[:3]})")
            else:
                print(f"      - '{team}'")
        
        if len(unmapped) > 30:
            print(f"      ... and {len(unmapped) - 30} more")
        
        print(f"\n  These teams need to be added to {TEAMS_INDEX}")
        print(f"  Games with unmapped teams will be dropped.\n")
    else:
        print(f"  ✓ All teams successfully mapped!")
    
    # Drop rows with unmapped teams
    initial_count = len(df)
    df = df.dropna(subset=['TeamA_ID', 'TeamB_ID'])
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        print(f"  ✓ Dropped {dropped_count} games with unmapped teams")
    
    return df, unmapped


def clean_and_transform(df):
    """Clean data and add derived columns"""
    print("\nCleaning and transforming data...")
    
    # Convert seeds to integers (handle empty strings)
    df['SeedA'] = pd.to_numeric(df['SeedA'], errors='coerce').fillna(0).astype(int)
    df['SeedB'] = pd.to_numeric(df['SeedB'], errors='coerce').fillna(0).astype(int)
    
    # Convert scores to integers (handle empty strings for future games)
    df['ScoreA'] = pd.to_numeric(df['ScoreA'], errors='coerce')
    df['ScoreB'] = pd.to_numeric(df['ScoreB'], errors='coerce')
    
    # Add seed differential (positive = TeamA is higher seed)
    df['SeedDiff'] = df['SeedB'] - df['SeedA']
    
    # Add binary outcome (1 if TeamA won, 0 if TeamB won)
    df['TeamA_Won'] = (df['Winner'] == df['TeamA']).astype(int)
    
    # Add upset flag (1 if lower seed won)
    df['IsUpset'] = ((df['SeedA'] > df['SeedB']) & (df['TeamA_Won'] == 1)) | \
                    ((df['SeedB'] > df['SeedA']) & (df['TeamA_Won'] == 0))
    df['IsUpset'] = df['IsUpset'].astype(int)
    
    # Add margin of victory
    df['MarginOfVictory'] = abs(df['ScoreA'] - df['ScoreB'])
    
    # Add total points
    df['TotalPoints'] = df['ScoreA'] + df['ScoreB']
    
    print(f"  ✓ Data cleaned and transformed")
    
    return df


def generate_summary_stats(df, unmapped_teams):
    """Print summary statistics about the transformed data"""
    print("\n" + "="*60)
    print("TRANSFORMATION SUMMARY")
    print("="*60)
    
    print(f"\nData Coverage:")
    print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  Total games: {len(df)}")
    print(f"  Unique teams: {len(set(df['TeamA_ID'].tolist() + df['TeamB_ID'].tolist()))}")
    
    print(f"\nGames by Year:")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} games")
    
    print(f"\nGames by Round:")
    round_counts = df['Round'].value_counts()
    for round_name, count in round_counts.items():
        print(f"  {round_name}: {count} games")
    
    print(f"\nUpset Statistics:")
    total_with_seeds = len(df[(df['SeedA'] > 0) & (df['SeedB'] > 0)])
    upsets = df['IsUpset'].sum()
    if total_with_seeds > 0:
        upset_pct = (upsets / total_with_seeds) * 100
        print(f"  Total upsets: {upsets} of {total_with_seeds} ({upset_pct:.1f}%)")
    
    print(f"\nData Quality:")
    missing_scores = df['ScoreA'].isna().sum()
    print(f"  Games with scores: {len(df) - missing_scores}")
    print(f"  Games without scores: {missing_scores}")
    print(f"  Unmapped teams: {len(unmapped_teams)}")
    
    print("="*60 + "\n")


def save_output(df):
    """Save transformed data to L2"""
    print(f"Saving output to {OUTPUT_FILE}...")
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select and order columns for output
    output_columns = [
        'Year', 'Region', 'Round',
        'TeamA', 'TeamA_ID', 'SeedA', 'ScoreA',
        'TeamB', 'TeamB_ID', 'SeedB', 'ScoreB',
        'Winner', 'TeamA_Won', 'SeedDiff', 'IsUpset',
        'MarginOfVictory', 'TotalPoints', 'Location'
    ]
    
    df_output = df[output_columns].copy()
    
    # Save to CSV
    df_output.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved {len(df_output)} games to {OUTPUT_FILE}")


def main():
    """Main transform pipeline"""
    try:
        # Load data
        df, teams_index = load_data()
        
        # Build team mapping
        mapping = build_team_mapping(teams_index)
        
        # Map team names to IDs
        df, unmapped_teams = map_team_names(df, mapping)
        
        # Clean and transform
        df = clean_and_transform(df)
        
        # Generate summary stats
        generate_summary_stats(df, unmapped_teams)
        
        # Save output
        save_output(df)
        
        print("✓ Transform complete!\n")
        
        if unmapped_teams:
            print("⚠️  ACTION REQUIRED:")
            print(f"   Add {len(unmapped_teams)} unmapped teams to {TEAMS_INDEX}")
            print(f"   Run identify_unmapped_teams.py to see full list with suggestions.")
            print(f"   Then re-run this script to include all games.\n")
            return 1
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        print(f"  Make sure you've run the scraper first to generate {INPUT_FILE}")
        return 1
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())