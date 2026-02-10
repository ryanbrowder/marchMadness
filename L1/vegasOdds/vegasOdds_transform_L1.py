"""
L1 Transform: Vegas Odds (DraftKings)
=====================================
Processes raw DraftKings odds data for Championship and Final Four markets.

Input:
    - ../data/vegasOdds/vegasOdds_Champion.csv
    - ../data/vegasOdds/vegasOdds_Final4.csv

Output:
    - ../../L2/data/vegasOdds/vegasOdds_analyze_L2.csv

Transformations:
    1. Parse alternating line format (team, odds, team, odds...)
    2. Standardize team names via teamsIndex
    3. Convert American odds → probabilities
    4. Remove vig:
       - Championship: normalize to 1.0 (one winner)
       - Final Four: normalize to 4.0 (four teams make it)
    5. Derive Elite 8 probabilities from Final Four odds using seed-based multipliers
       (more stable than deriving from Championship odds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# File paths
DATA_DIR = Path(__file__).parent / '../data/vegasOdds'
TEAMS_INDEX_PATH = Path(__file__).parent / '../../utils/teamsIndex.csv'
OUTPUT_PATH = Path(__file__).parent / '../../L2/data/vegasOdds'

# Historical Elite 8 rates by seed (calculated from tournament history)
# These represent P(reach Elite 8 | seed) from historical data
SEED_E8_RATES = {
    1: 0.705,   # 1-seeds reach E8 ~70.5% of time
    2: 0.500,   # 2-seeds reach E8 ~50% of time
    3: 0.347,   # 3-seeds reach E8 ~34.7% of time
    4: 0.250,   # 4-seeds reach E8 ~25% of time
    5: 0.150,   # 5-seeds reach E8 ~15% of time
    6: 0.100,   # 6-seeds reach E8 ~10% of time
    7: 0.080,   # 7-seeds reach E8 ~8% of time
    8: 0.060,   # 8-seeds reach E8 ~6% of time
    9: 0.045,   # 9-seeds reach E8 ~4.5% of time
    10: 0.030,  # 10-seeds reach E8 ~3% of time
    11: 0.025,  # 11-seeds reach E8 ~2.5% of time
    12: 0.020,  # 12-seeds reach E8 ~2% of time
    13: 0.012,  # 13-seeds reach E8 ~1.2% of time
    14: 0.008,  # 14-seeds reach E8 ~0.8% of time
    15: 0.005,  # 15-seeds reach E8 ~0.5% of time
    16: 0.003,  # 16-seeds reach E8 ~0.3% of time
}

# Historical Final Four rates by seed (calculated from tournament history)
# These represent P(reach Final Four | seed) from historical data
SEED_F4_RATES = {
    1: 0.545,   # 1-seeds reach F4 ~54.5% of time
    2: 0.312,   # 2-seeds reach F4 ~31.2% of time
    3: 0.198,   # 3-seeds reach F4 ~19.8% of time
    4: 0.125,   # 4-seeds reach F4 ~12.5% of time
    5: 0.075,   # 5-seeds reach F4 ~7.5% of time
    6: 0.050,   # 6-seeds reach F4 ~5% of time
    7: 0.035,   # 7-seeds reach F4 ~3.5% of time
    8: 0.025,   # 8-seeds reach F4 ~2.5% of time
    9: 0.015,   # 9-seeds reach F4 ~1.5% of time
    10: 0.010,  # 10-seeds reach F4 ~1% of time
    11: 0.008,  # 11-seeds reach F4 ~0.8% of time
    12: 0.006,  # 12-seeds reach F4 ~0.6% of time
    13: 0.003,  # 13-seeds reach F4 ~0.3% of time
    14: 0.002,  # 14-seeds reach F4 ~0.2% of time
    15: 0.001,  # 15-seeds reach F4 ~0.1% of time
    16: 0.0005, # 16-seeds reach F4 ~0.05% of time
}


def parse_alternating_odds_file(filepath):
    """
    Parse DraftKings CSV format: alternating lines of team_name, odds.
    
    Returns DataFrame with columns: team_name, odds
    """
    with open(filepath, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse alternating lines
    teams = []
    odds_values = []
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            team = lines[i]
            odds = lines[i + 1]
            
            # Handle special minus sign character (−) vs standard (-)
            odds = odds.replace('−', '-')
            
            # Handle special plus sign character if present
            odds = odds.replace('+', '')
            
            # Convert to int
            try:
                odds_int = int(odds)
                teams.append(team)
                odds_values.append(odds_int)
            except ValueError:
                print(f"Warning: Could not parse odds '{odds}' for team '{team}'. Skipping.")
                continue
    
    return pd.DataFrame({
        'team_name': teams,
        'odds': odds_values
    })


def american_to_prob(odds):
    """
    Convert American odds to implied probability.
    
    Negative odds (favorites): P = |odds| / (|odds| + 100)
    Positive odds (underdogs): P = 100 / (odds + 100)
    
    Note: Allows very high odds (>10000) for unmatched teams with fallback odds.
    """
    if pd.isna(odds):
        return None
    
    # Allow extreme odds for fallback assignments
    if odds < -100000 or odds > 200000:
        raise ValueError(f"Odds out of reasonable range: {odds}")
    
    if odds < 0:
        # Favorite: -150 means bet $150 to win $100
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog: +200 means bet $100 to win $200
        return 100 / (odds + 100)


def remove_vig(probs, target_sum=1.0):
    """
    Remove bookmaker vig (overround) by normalizing probabilities.
    
    Bookmakers set odds so probabilities sum > target (their profit margin).
    We normalize to get "fair" implied probabilities.
    
    Args:
        probs: Series of probabilities
        target_sum: What probabilities should sum to after normalization
                   - 1.0 for championship (one winner)
                   - 4.0 for Final Four (four teams make it)
    """
    total = probs.sum()
    if total == 0:
        return probs
    return (probs / total) * target_sum


def estimate_seed_from_final4_prob(final4_prob):
    """
    Estimate likely seed based on Final Four probability.
    Uses historical seed Final Four rates to find closest match.
    """
    # Find seed with closest Final Four probability
    min_diff = float('inf')
    best_seed = 8
    
    for seed, rate in SEED_F4_RATES.items():
        diff = abs(rate - final4_prob)
        if diff < min_diff:
            min_diff = diff
            best_seed = seed
    
    return best_seed


def derive_elite8_prob(final4_prob, estimated_seed):
    """
    Derive Elite 8 probability from Final Four probability.
    
    Uses historical ratios: P(E8) / P(F4) varies by seed.
    This is more stable than deriving from Championship odds (fewer rounds to traverse).
    
    For example: 1-seeds reach E8 70% of time and F4 54% of time,
    so multiplier is 70/54 = 1.30x
    """
    f4_rate = SEED_F4_RATES.get(estimated_seed, 0.01)
    e8_rate = SEED_E8_RATES.get(estimated_seed, 0.02)
    
    if f4_rate > 0:
        multiplier = e8_rate / f4_rate
    else:
        multiplier = 1.5  # Default fallback
    
    # Apply multiplier but cap at 95% (even 1-seeds aren't guaranteed)
    e8_prob = min(final4_prob * multiplier, 0.95)
    
    return e8_prob


def load_and_match_teams(teams_df, teams_index_path):
    """
    Match DraftKings team names to canonical teamIndex.
    Uses case-insensitive matching with whitespace normalization.
    
    Returns DataFrame with teamIndex added, and list of unmatched teams.
    """
    # Load teams index
    teams_index = pd.read_csv(teams_index_path)
    
    # Create normalized lookup for better matching
    # Normalize: lowercase, strip whitespace, remove periods
    def normalize_name(name):
        if pd.isna(name):
            return ""
        return str(name).lower().strip().replace('.', '').replace("'", '')
    
    # Create lookup with both original and normalized names
    teams_index['normalized'] = teams_index['Team'].apply(normalize_name)
    lookup_normalized = teams_index.set_index('normalized')['Index'].to_dict()
    lookup_original = teams_index.set_index('Team')['Index'].to_dict()
    
    # Normalize team names in input data
    teams_df['normalized'] = teams_df['team_name'].apply(normalize_name)
    
    # Try exact match first
    teams_df['teamIndex'] = teams_df['team_name'].map(lookup_original)
    
    # For unmatched, try normalized match
    unmatched_mask = teams_df['teamIndex'].isna()
    teams_df.loc[unmatched_mask, 'teamIndex'] = teams_df.loc[unmatched_mask, 'normalized'].map(lookup_normalized)
    
    # Find still-unmatched teams
    unmatched = teams_df[teams_df['teamIndex'].isna()]['team_name'].tolist()
    
    if unmatched:
        print(f"\nWarning: {len(unmatched)} unmatched teams:")
        for team in unmatched:
            print(f"  - {team}")
        print("\nAssigning fallback odds for unmatched teams...")
        
        # Assign +150000 odds for unmatched teams (virtually impossible)
        teams_df.loc[teams_df['teamIndex'].isna(), 'odds'] = 150000
        
        # Still need teamIndex - we'll assign them temporary indices
        max_index = teams_index['Index'].max()
        for i, team in enumerate(unmatched):
            new_index = max_index + i + 1
            teams_df.loc[teams_df['team_name'] == team, 'teamIndex'] = new_index
            print(f"  Assigned temporary index {new_index} to {team}")
    
    teams_df['teamIndex'] = teams_df['teamIndex'].astype(int)
    teams_df = teams_df.drop(columns=['normalized'])  # Clean up temp column
    
    return teams_df, unmatched


def main():
    """Main transform pipeline."""
    
    print("=" * 80)
    print("Vegas Odds Transform (L1)")
    print("=" * 80)
    print()
    
    # Parse Championship odds
    print("Loading Championship odds...")
    champ_path = DATA_DIR / 'vegasOdds_Champion.csv'
    champ_df = parse_alternating_odds_file(champ_path)
    print(f"  Loaded {len(champ_df)} teams")
    
    # Parse Final Four odds
    print("\nLoading Final Four odds...")
    final4_path = DATA_DIR / 'vegasOdds_Final4.csv'
    final4_df = parse_alternating_odds_file(final4_path)
    print(f"  Loaded {len(final4_df)} teams")
    
    # Merge on team name
    print("\nMerging Championship and Final Four odds...")
    merged_df = champ_df.merge(
        final4_df,
        on='team_name',
        how='outer',
        suffixes=('_champ', '_final4')
    )
    print(f"  Merged dataset has {len(merged_df)} teams")
    
    # Match to teamsIndex
    print("\nMatching teams to canonical teamIndex...")
    merged_df, unmatched = load_and_match_teams(merged_df, TEAMS_INDEX_PATH)
    print(f"  Successfully matched {len(merged_df) - len(unmatched)} teams")
    
    # Convert odds to probabilities
    print("\nConverting odds to probabilities...")
    merged_df['champ_prob_raw'] = merged_df['odds_champ'].apply(american_to_prob)
    merged_df['final4_prob_raw'] = merged_df['odds_final4'].apply(american_to_prob)
    
    # Remove vig (normalize probabilities)
    print("Removing vig (normalizing probabilities)...")
    merged_df['champ_prob_adjusted'] = remove_vig(merged_df['champ_prob_raw'], target_sum=1.0)
    merged_df['final4_prob_adjusted'] = remove_vig(merged_df['final4_prob_raw'], target_sum=4.0)
    
    # Validate vig removal
    champ_sum = merged_df['champ_prob_adjusted'].sum()
    final4_sum = merged_df['final4_prob_adjusted'].sum()
    print(f"  Championship probs sum: {champ_sum:.4f} (should be ~1.0)")
    print(f"  Final Four probs sum: {final4_sum:.4f} (should be ~4.0)")
    
    if not (0.99 <= champ_sum <= 1.01):
        print(f"  WARNING: Championship probabilities sum to {champ_sum}, not ~1.0")
    if not (3.95 <= final4_sum <= 4.05):
        print(f"  WARNING: Final Four probabilities sum to {final4_sum}, not ~4.0")
    
    # Derive Elite 8 probabilities
    print("\nDeriving Elite 8 probabilities...")
    merged_df['estimated_seed'] = merged_df['final4_prob_adjusted'].apply(estimate_seed_from_final4_prob)
    merged_df['elite8_prob'] = merged_df.apply(
        lambda row: derive_elite8_prob(row['final4_prob_adjusted'], row['estimated_seed']),
        axis=1
    )
    
    # Validate Elite 8 derivation (E8 >= F4 >= Champ)
    violations = merged_df[
        (merged_df['elite8_prob'] < merged_df['final4_prob_adjusted']) |
        (merged_df['final4_prob_adjusted'] < merged_df['champ_prob_adjusted'])
    ]
    if len(violations) > 0:
        print(f"  WARNING: {len(violations)} teams violate E8 >= F4 >= Champ constraint")
        print(violations[['team_name', 'elite8_prob', 'final4_prob_adjusted', 'champ_prob_adjusted']].head())
    else:
        print("  ✓ All teams satisfy E8 >= F4 >= Champ constraint")
    
    # Prepare output dataset
    print("\nPreparing L2 output...")
    output_df = merged_df[[
        'teamIndex',
        'team_name',
        'odds_champ',
        'odds_final4',
        'champ_prob_raw',
        'champ_prob_adjusted',
        'final4_prob_raw',
        'final4_prob_adjusted',
        'elite8_prob',
        'estimated_seed'
    ]].copy()
    
    # Rename for clarity
    output_df.columns = [
        'teamIndex',
        'team_name',
        'championship_odds_american',
        'final4_odds_american',
        'vegas_champ_prob_raw',
        'vegas_champ_prob',
        'vegas_final4_prob_raw',
        'vegas_final4_prob',
        'vegas_elite8_prob',
        'estimated_seed'
    ]
    
    # Add metadata
    output_df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
    output_df['source'] = 'draftkings'
    
    # Sort by Elite 8 probability (best candidates first)
    output_df = output_df.sort_values('vegas_elite8_prob', ascending=False)
    
    # Save output
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH / 'vegasOdds_analyze_L2.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"  Saved to: {output_path}")
    print(f"  Output shape: {output_df.shape}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"\nTop 10 Elite 8 Candidates (by Vegas):")
    print(output_df[['teamIndex', 'team_name', 'vegas_elite8_prob', 'estimated_seed']].head(10).to_string(index=False))
    
    print(f"\nElite 8 Probability Distribution:")
    print(f"  Mean:   {output_df['vegas_elite8_prob'].mean():.4f}")
    print(f"  Median: {output_df['vegas_elite8_prob'].median():.4f}")
    print(f"  Max:    {output_df['vegas_elite8_prob'].max():.4f} ({output_df.iloc[0]['team_name']})")
    print(f"  Min:    {output_df['vegas_elite8_prob'].min():.4f}")
    
    print(f"\nTeams by Estimated Seed:")
    seed_counts = output_df['estimated_seed'].value_counts().sort_index()
    for seed, count in seed_counts.items():
        print(f"  Seed {seed:2d}: {count:2d} teams")
    
    if unmatched:
        print(f"\n⚠️  WARNING: {len(unmatched)} teams could not be matched to teamsIndex")
        print("These teams were assigned fallback odds (+150000) and temporary indices.")
        print("You may want to add these team name variants to teamsIndex.csv:")
        for team in unmatched:
            print(f"  - {team}")
    
    print("\n✓ Transform complete!")


if __name__ == '__main__':
    main()