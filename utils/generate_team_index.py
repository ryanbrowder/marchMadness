"""
Generate Team Index from Torvik Data
Creates a fresh team index based on unique teams in Torvik historical data
"""

import pandas as pd
import os


def generate_team_index(
    torvik_raw_file: str = None,
    old_index_file: str = None,
    output_file: str = None
) -> pd.DataFrame:
    """
    Generate team index from Torvik raw data.
    
    Args:
        torvik_raw_file: Path to bartTorvik_raw.csv
        old_index_file: Path to existing teamsIndex.csv (optional, for comparison)
        output_file: Path to save new teamsIndex.csv
    
    Returns:
        DataFrame with Team and Index columns
    """
    # Set default paths
    if torvik_raw_file is None:
        torvik_raw_file = os.path.join(
            os.path.dirname(__file__), 
            '../L1/data/bartTorvik/bartTorvik_raw.csv'
        )
    
    if old_index_file is None:
        old_index_file = os.path.join(
            os.path.dirname(__file__),
            'teamsIndex.csv'
        )
    
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(__file__),
            'teamsIndex.csv'
        )
    
    print("="*70)
    print("Team Index Generator")
    print("="*70)
    print()
    
    # Read Torvik raw data
    print(f"Reading Torvik data from: {torvik_raw_file}")
    df = pd.read_csv(torvik_raw_file)
    print(f"  ✓ Loaded {len(df):,} rows")
    
    # Extract unique teams
    # Need to parse team names to remove tournament info
    import re
    
    def extract_team_name(team_string):
        """Extract clean team name from strings like 'Kansas1 seed,CHAMPS'."""
        if pd.isna(team_string):
            return None
        
        # Pattern: TeamName + Digit + " seed," + Outcome
        match = re.match(r'^(.+?)(\d+)\s*seed,(.+)$', team_string)
        
        if match:
            return match.group(1).strip()
        else:
            return team_string.strip()
    
    df['CleanTeam'] = df['Team'].apply(extract_team_name)
    
    # Get unique teams sorted alphabetically
    unique_teams = sorted(df['CleanTeam'].dropna().unique())
    print(f"  ✓ Found {len(unique_teams)} unique teams")
    
    # Create new index
    new_index = pd.DataFrame({
        'Team': unique_teams,
        'Index': range(1, len(unique_teams) + 1)
    })
    
    # Compare with old index if it exists
    if os.path.exists(old_index_file):
        print(f"\nComparing with old index: {old_index_file}")
        old_index = pd.read_csv(old_index_file)
        print(f"  Old index had {len(old_index)} teams")
        
        # Teams in old but not in new
        old_teams = set(old_index['Team'])
        new_teams = set(new_index['Team'])
        
        missing_from_new = old_teams - new_teams
        missing_from_old = new_teams - old_teams
        
        if missing_from_new:
            print(f"\n  ⚠ Teams in old index but NOT in Torvik ({len(missing_from_new)}):")
            for team in sorted(missing_from_new):
                print(f"    - {team}")
        
        if missing_from_old:
            print(f"\n  ℹ New teams in Torvik not in old index ({len(missing_from_old)}):")
            for team in sorted(missing_from_old)[:10]:  # Show first 10
                print(f"    - {team}")
            if len(missing_from_old) > 10:
                print(f"    ... and {len(missing_from_old) - 10} more")
    
    # Save new index
    print(f"\nSaving new index to: {output_file}")
    new_index.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(new_index)} teams")
    
    print()
    print("="*70)
    print("Index generation complete")
    print("="*70)
    
    # Show sample
    print("\nSample of new index:")
    print(new_index.head(20))
    
    return new_index


if __name__ == "__main__":
    generate_team_index()