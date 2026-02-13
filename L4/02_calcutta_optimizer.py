"""
L4.02 - Calcutta Portfolio Optimizer

Generates optimal team portfolios for Calcutta auction using:
  - Historical tournament performance by seed (2008-2025)
  - Auction market analysis (2022-2025)
  - 2026 model predictions from L4.01
  - Your scoring system with upset bonuses

Outputs:
  - Team valuations (expected points, fair value, market price)
  - Portfolio strategies (Value Hunter, Patient Accumulator, Hybrid)
  - Value opportunities (best picks by price tier)
  - Visualizations and diagnostics

Usage:
  python 02_calcutta_optimizer.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = Path("data")
TOURNAMENT_RESULTS = DATA_DIR / "tournament_results.csv"
AUCTION_HISTORY = DATA_DIR / "auction_history.csv"

# L4.01 outputs (2026 predictions)
ROUND_PROBS_2026 = Path("outputs/01_tournament_simulator/round_probabilities.csv")
ELITE8_DIRECT_2026 = Path("../L3/elite8/outputs/05_2026_predictions/elite8_predictions_2026_long.csv")

# Outputs
OUTPUT_DIR = Path("outputs/02_calcutta_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Scoring system
SCORING = {
    'Play-In': 1,
    'R64': 2,
    'R32': 4,
    'S16': 6,
    'E8': 8,
    'FF': 10,
    'Championship': 15
}

# Round mapping for tournament results
ROUND_MAP = {
    'R64': 'R64',
    'R32': 'R32',
    'S16': 'S16',
    'E8': 'E8',
    'FF': 'FF',
    'Championship': 'Championship'
}

# Portfolio parameters
BUDGET = 100
N_ENTRANTS_TYPICAL = 7  # Typical pool size

# Model blending weight (adjust to control historical vs model influence)
# 0.0 = 100% historical baseline (ignore model)
# 0.3 = 30% model, 70% historical (default conservative)
# 0.5 = 50/50 balanced
# 0.7 = 70% model, 30% historical (trust the model)
# 1.0 = 100% model (no historical anchor)
MODEL_WEIGHT = 0.7  # ADJUST THIS VALUE (0.0 to 1.0)

# Strategy definitions
STRATEGIES = {
    'value_hunter': {
        'name': 'Value Hunter',
        'description': 'Target 5-8 high-value teams, concentrated portfolio',
        'target_teams': (5, 8),
        'early_budget': 0.8,  # Spend 80% early
        'value_threshold': 1.2  # Only bid on teams with E[pts]/price > 1.2
    },
    'patient_accumulator': {
        'name': 'Patient Accumulator',
        'description': 'Wait for late auction, accumulate 20-25 cheap teams',
        'target_teams': (20, 25),
        'early_budget': 0.3,  # Spend only 30% early
        'value_threshold': 0.8  # Accept lower value for volume
    },
    'hybrid': {
        'name': 'Hybrid',
        'description': 'Balance 3-4 early targets with 15-20 late volume picks',
        'target_teams': (15, 20),
        'early_budget': 0.5,  # 50/50 split
        'value_threshold': 1.0  # Market value
    }
}

# ============================================================================
# HISTORICAL ANALYSIS
# ============================================================================

def load_tournament_results():
    """Load historical tournament results (2008-2025)."""
    df = pd.read_csv(TOURNAMENT_RESULTS)
    return df


def calculate_historical_seed_performance(results_df):
    """
    Calculate P(reach round) for each seed based on 2008-2025 data.
    Returns DataFrame with columns: Seed, P_R64, P_R32, ..., P_Championship
    """
    print("Calculating historical seed performance (2008-2025)...")
    
    # Count appearances by seed and round
    seed_rounds = defaultdict(lambda: defaultdict(int))
    seed_totals = defaultdict(int)
    
    for _, row in results_df.iterrows():
        winner = row['Winner']
        round_name = row['Round']
        
        # Determine winner's seed
        if winner == row['TeamA']:
            seed = row['SeedA']
        else:
            seed = row['SeedB']
        
        # Track seed appearances
        if round_name == 'R64':
            seed_totals[seed] += 1
        
        # Count round achievements
        if round_name in ROUND_MAP:
            mapped_round = ROUND_MAP[round_name]
            seed_rounds[seed][mapped_round] += 1
    
    # Calculate probabilities
    rows = []
    for seed in range(1, 17):
        if seed not in seed_totals or seed_totals[seed] == 0:
            continue
        
        total = seed_totals[seed]
        row = {'Seed': seed, 'N_Tournaments': total}
        
        # P(reach each round)
        for round_name in ['R64', 'R32', 'S16', 'E8', 'FF', 'Championship']:
            count = seed_rounds[seed].get(round_name, 0)
            
            # R64 is automatic (everyone starts there)
            if round_name == 'R64':
                prob = 1.0
            else:
                prob = count / total
            
            row[f'P_{round_name}'] = prob
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Seed').reset_index(drop=True)
    
    return df


def calculate_upset_bonuses(results_df):
    """Calculate average upset bonus by seed matchup."""
    print("Calculating historical upset patterns...")
    
    upset_data = []
    
    for _, row in results_df.iterrows():
        if row['IsUpset'] == 1:
            # Determine upset details
            if row['TeamA_Won'] == 1:
                underdog_seed = row['SeedA']
                favorite_seed = row['SeedB']
            else:
                underdog_seed = row['SeedB']
                favorite_seed = row['SeedA']
            
            seed_diff = abs(row['SeedDiff'])
            round_name = ROUND_MAP.get(row['Round'], None)
            
            if round_name:
                upset_data.append({
                    'Round': round_name,
                    'Underdog_Seed': underdog_seed,
                    'Favorite_Seed': favorite_seed,
                    'Seed_Diff': seed_diff,
                    'Upset_Bonus': seed_diff
                })
    
    df = pd.DataFrame(upset_data)
    
    # Average upset bonus by seed
    avg_bonus = df.groupby('Underdog_Seed')['Upset_Bonus'].mean().to_dict()
    
    return avg_bonus


def calculate_expected_points_historical(seed_perf_df, upset_bonuses):
    """
    Calculate E[points] for each seed using historical performance.
    Uses round-by-round upset bonus calculation.
    """
    print("Calculating expected points by seed (historical baseline with round-by-round upset bonuses)...")
    
    # R64 bracket structure
    R64_MATCHUPS = {1: 16, 2: 15, 3: 14, 4: 13, 5: 12, 6: 11, 7: 10, 8: 9,
                    16: 1, 15: 2, 14: 3, 13: 4, 12: 5, 11: 6, 10: 7, 9: 8}
    
    # Expected opponent seeds for later rounds
    EXPECTED_OPPONENT_SEED = {
        'R64': R64_MATCHUPS,
        'R32': 8.0,
        'S16': 4.5,
        'E8': 3.0,
        'FF': 2.0,
        'Championship': 1.5
    }
    
    rows = []
    
    for _, row in seed_perf_df.iterrows():
        seed = row['Seed']
        
        # Base expected points (no upset bonuses)
        base_points = 0
        base_points += row['P_R64'] * SCORING['R64']
        base_points += row['P_R32'] * SCORING['R32']
        base_points += row['P_S16'] * SCORING['S16']
        base_points += row['P_E8'] * SCORING['E8']
        base_points += row['P_FF'] * SCORING['FF']
        base_points += row['P_Championship'] * SCORING['Championship']
        
        # Calculate round-by-round upset bonuses
        upset_bonus_total = 0.0
        
        # Winning a round means reaching the NEXT round
        # R64 win = reach R32, R32 win = reach S16, etc.
        round_sequence = [
            ('R64', row['P_R32'], 1.0),           # Win R64 → reach R32
            ('R32', row['P_S16'], row['P_R32']),  # Win R32 → reach S16  
            ('S16', row['P_E8'], row['P_S16']),   # Win S16 → reach E8
            ('E8', row['P_FF'], row['P_E8']),     # Win E8 → reach FF
            ('FF', row['P_Championship'], row['P_FF'])  # Win FF → reach Championship
        ]
        
        for round_name, round_prob, prev_round_prob in round_sequence:
            if round_prob == 0 or prev_round_prob == 0:
                continue
            
            # P(win this round | reached this round)
            p_win_round = round_prob / prev_round_prob
            
            # Expected opponent seed
            if round_name == 'R64':
                expected_opp = EXPECTED_OPPONENT_SEED['R64'].get(seed, seed)
            else:
                expected_opp = EXPECTED_OPPONENT_SEED[round_name]
            
            # Upset bonus only if underdog (higher seed number)
            if seed > expected_opp:
                seed_diff = seed - expected_opp
                upset_bonus_total += p_win_round * seed_diff
        
        expected_points = base_points + upset_bonus_total
        
        rows.append({
            'Seed': seed,
            'E_Points_Historical': round(expected_points, 2),
            'Base_Points': round(base_points, 2),
            'Upset_Bonus': round(upset_bonus_total, 2)
        })
    
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# AUCTION MARKET ANALYSIS
# ============================================================================

def load_auction_history():
    """Load auction history (2022-2025)."""
    df = pd.read_csv(AUCTION_HISTORY)
    return df


def analyze_auction_market(auction_df):
    """
    Analyze historical bidding patterns by seed.
    Returns avg bid, avg points, value (pts/$) by seed.
    """
    print("Analyzing auction market dynamics (2022-2025)...")
    
    # Filter to teams with known seeds
    df = auction_df[auction_df['Seed'] > 0].copy()
    
    # Calculate metrics by seed
    seed_stats = df.groupby('Seed').agg({
        'Bid': ['mean', 'median', 'std', 'count'],
        'Points': ['mean', 'median', 'std']
    }).round(2)
    
    seed_stats.columns = ['_'.join(col).strip() for col in seed_stats.columns.values]
    seed_stats = seed_stats.reset_index()
    
    # Calculate value (points per dollar)
    seed_stats['Value_Pts_Per_Dollar'] = (
        seed_stats['Points_mean'] / seed_stats['Bid_mean']
    ).round(3)
    
    # Calculate ROI
    seed_stats['ROI'] = (
        (seed_stats['Points_mean'] - seed_stats['Bid_mean']) / seed_stats['Bid_mean']
    ).round(3)
    
    return seed_stats


def identify_historical_value_picks(auction_df):
    """Find specific teams that historically outperformed their price."""
    print("Identifying historical value picks...")
    
    # Filter to successful bids (bid > 0)
    df = auction_df[auction_df['Bid'] > 0].copy()
    
    # Calculate value for each purchase
    df['Pts_Per_Dollar'] = df['Points'] / df['Bid']
    df['ROI'] = (df['Points'] - df['Bid']) / df['Bid']
    
    # Best value picks (min $5 bid to avoid noise)
    best_value = df[df['Bid'] >= 5].nlargest(20, 'Pts_Per_Dollar')[
        ['Year', 'Player', 'Team', 'Seed', 'Bid', 'Points', 'Pts_Per_Dollar', 'ROI']
    ]
    
    # Worst value picks
    worst_value = df[df['Bid'] >= 15].nsmallest(20, 'Pts_Per_Dollar')[
        ['Year', 'Player', 'Team', 'Seed', 'Bid', 'Points', 'Pts_Per_Dollar', 'ROI']
    ]
    
    return best_value, worst_value


# ============================================================================
# 2026 TEAM VALUATIONS
# ============================================================================

def load_2026_predictions():
    """Load 2026 model predictions from L4.01."""
    print("Loading 2026 model predictions...")
    
    # Round probabilities
    round_probs = pd.read_csv(ROUND_PROBS_2026)
    
    # Elite 8 direct (optional, for comparison)
    try:
        elite8_direct = pd.read_csv(ELITE8_DIRECT_2026)
    except FileNotFoundError:
        elite8_direct = None
    
    return round_probs, elite8_direct


def calculate_2026_expected_points(round_probs_df, upset_bonuses):
    """
    Calculate E[points] for 2026 tournament teams.
    Uses L4.01 model predictions + round-by-round upset bonus calculation.
    """
    print("Calculating 2026 expected points with round-by-round upset bonuses...")
    
    # R64 bracket structure (known matchups)
    R64_MATCHUPS = {1: 16, 2: 15, 3: 14, 4: 13, 5: 12, 6: 11, 7: 10, 8: 9,
                    16: 1, 15: 2, 14: 3, 13: 4, 12: 5, 11: 6, 10: 7, 9: 8}
    
    # Expected average opponent seed for later rounds (based on historical data)
    # These represent "typical team that reaches this round"
    EXPECTED_OPPONENT_SEED = {
        'R64': R64_MATCHUPS,  # Known from bracket
        'R32': 8.0,   # Average R32 team is ~8-9 seed
        'S16': 4.5,   # Average S16 team is ~4-5 seed
        'E8': 3.0,    # Average E8 team is ~2-3 seed
        'FF': 2.0,    # Average FF team is ~1-2 seed
        'Championship': 1.5  # Average championship participant is ~1-2 seed
    }
    
    rows = []
    
    for _, row in round_probs_df.iterrows():
        team = row['Team']
        seed = int(row['Seed'])
        
        # Base points from round probabilities (no upset bonuses)
        base_points = 0
        base_points += row['P_R64'] * SCORING['R64']
        base_points += row['P_R32'] * SCORING['R32']
        base_points += row['P_S16'] * SCORING['S16']
        base_points += row['P_E8'] * SCORING['E8']
        base_points += row['P_FF'] * SCORING['FF']
        base_points += row['P_Championship'] * SCORING['Championship']
        
        # Calculate round-by-round upset bonuses
        upset_bonus_total = 0.0
        
        # Winning a round means reaching the NEXT round
        round_sequence = [
            ('R64', row['P_R32'], 1.0),           # Win R64 → reach R32
            ('R32', row['P_S16'], row['P_R32']),  # Win R32 → reach S16  
            ('S16', row['P_E8'], row['P_S16']),   # Win S16 → reach E8
            ('E8', row['P_FF'], row['P_E8']),     # Win E8 → reach FF
            ('FF', row['P_Championship'], row['P_FF'])  # Win FF → reach Championship
        ]
        
        for round_name, round_prob, prev_round_prob in round_sequence:
            if round_prob == 0 or prev_round_prob == 0:
                continue
            
            # P(win this round | reached this round)
            p_win_round = round_prob / prev_round_prob
            
            # Expected opponent seed
            if round_name == 'R64':
                expected_opp = EXPECTED_OPPONENT_SEED['R64'].get(seed, seed)
            else:
                expected_opp = EXPECTED_OPPONENT_SEED[round_name]
            
            # Upset bonus only if underdog (higher seed number)
            if seed > expected_opp:
                seed_diff = seed - expected_opp
                upset_bonus_total += p_win_round * seed_diff
        
        expected_points = base_points + upset_bonus_total
        
        rows.append({
            'Team': team,
            'Seed': seed,
            'Region': row.get('Region', ''),
            'E_Points_Model': round(expected_points, 2),
            'Base_Points': round(base_points, 2),
            'Upset_Bonus': round(upset_bonus_total, 2),
            'P_R64': round(row['P_R64'], 4),
            'P_R32': round(row['P_R32'], 4),
            'P_S16': round(row['P_S16'], 4),
            'P_E8': round(row['P_E8'], 4),
            'P_FF': round(row['P_FF'], 4),
            'P_Championship': round(row['P_Championship'], 4)
        })
    
    df = pd.DataFrame(rows)
    return df


def blend_historical_and_model(historical_seed_df, model_2026_df):
    """
    Blend historical baseline with 2026 model predictions.
    Historical provides long-term baseline, model adjusts for 2026 specifics.
    Blend weight controlled by MODEL_WEIGHT parameter.
    """
    historical_weight = 1.0 - MODEL_WEIGHT
    
    print(f"Blending historical baseline ({historical_weight:.0%}) with 2026 model ({MODEL_WEIGHT:.0%})...")
    
    # Merge on seed
    merged = model_2026_df.merge(
        historical_seed_df[['Seed', 'E_Points_Historical']],
        on='Seed',
        how='left'
    )
    
    # Blend: (1-MODEL_WEIGHT) × historical + MODEL_WEIGHT × model
    merged['E_Points_Blended'] = (
        historical_weight * merged['E_Points_Historical'] + 
        MODEL_WEIGHT * merged['E_Points_Model']
    ).round(2)
    
    return merged


def estimate_market_prices(team_valuations_df, auction_market_df):
    """
    Estimate expected market price for each team based on historical patterns.
    Uses seed-based average bids.
    """
    print("Estimating market prices...")
    
    # Create seed → avg bid lookup
    seed_prices = dict(zip(
        auction_market_df['Seed'],
        auction_market_df['Bid_mean']
    ))
    
    # Apply to teams
    team_valuations_df['Expected_Market_Price'] = (
        team_valuations_df['Seed'].map(seed_prices)
    ).round(2)
    
    # Calculate fair value (E[points] as fraction of total pool points)
    total_expected_points = team_valuations_df['E_Points_Blended'].sum()
    total_pool_money = BUDGET * N_ENTRANTS_TYPICAL
    
    team_valuations_df['Fair_Value'] = (
        (team_valuations_df['E_Points_Blended'] / total_expected_points) * 
        total_pool_money
    ).round(2)
    
    # Calculate value rating
    team_valuations_df['Value_Rating'] = (
        team_valuations_df['E_Points_Blended'] / 
        team_valuations_df['Expected_Market_Price']
    ).round(3)
    
    return team_valuations_df


# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

def generate_value_hunter_portfolio(valuations_df, budget=BUDGET):
    """
    Value Hunter: Target 5-8 teams with best value ratings.
    Concentrated bet on high-value picks.
    """
    config = STRATEGIES['value_hunter']
    
    # Filter to teams with value > threshold
    candidates = valuations_df[
        valuations_df['Value_Rating'] >= config['value_threshold']
    ].copy()
    
    # Sort by value rating
    candidates = candidates.sort_values('Value_Rating', ascending=False)
    
    # Target 5-8 teams
    target_n = config['target_teams'][1]  # 8 teams
    
    # Allocate budget proportionally to E[points]
    top_teams = candidates.head(target_n).copy()
    
    total_exp_points = top_teams['E_Points_Blended'].sum()
    top_teams['Allocated_Bid'] = (
        (top_teams['E_Points_Blended'] / total_exp_points) * budget
    ).round(0).astype(int)
    
    # Adjust for rounding
    diff = budget - top_teams['Allocated_Bid'].sum()
    if diff != 0:
        # Add/subtract from highest value team
        top_teams.loc[top_teams.index[0], 'Allocated_Bid'] += diff
    
    portfolio = top_teams[[
        'Team', 'Seed', 'E_Points_Blended', 'Expected_Market_Price',
        'Fair_Value', 'Value_Rating', 'Allocated_Bid'
    ]].copy()
    
    portfolio['Strategy'] = config['name']
    
    return portfolio


def generate_patient_accumulator_portfolio(valuations_df, budget=BUDGET):
    """
    Patient Accumulator: Wait for late auction, grab 20-25 teams at $1-5.
    Focus on seeds 8-15 with upset potential.
    """
    config = STRATEGIES['patient_accumulator']
    
    # Filter to mid/low seeds (likely leftovers)
    candidates = valuations_df[
        (valuations_df['Seed'] >= 8) & 
        (valuations_df['Expected_Market_Price'] <= 10)
    ].copy()
    
    # Sort by E[points] (want best teams in leftover tier)
    candidates = candidates.sort_values('E_Points_Blended', ascending=False)
    
    # Target 20-25 teams
    target_n = config['target_teams'][1]  # 25 teams
    
    # Allocate small bids ($1-5 range)
    top_teams = candidates.head(target_n).copy()
    
    # Distribute budget: most to best teams, but keep all bids low
    weights = np.linspace(1.5, 0.5, len(top_teams))  # Decreasing weights
    weights = weights / weights.sum()
    
    top_teams['Allocated_Bid'] = (weights * budget).round(0).astype(int)
    
    # Ensure no bid > $10 (stay patient)
    top_teams['Allocated_Bid'] = top_teams['Allocated_Bid'].clip(upper=10)
    
    # Adjust for budget constraint
    diff = budget - top_teams['Allocated_Bid'].sum()
    if diff != 0:
        top_teams.loc[top_teams.index[0], 'Allocated_Bid'] += diff
    
    portfolio = top_teams[[
        'Team', 'Seed', 'E_Points_Blended', 'Expected_Market_Price',
        'Fair_Value', 'Value_Rating', 'Allocated_Bid'
    ]].copy()
    
    portfolio['Strategy'] = config['name']
    
    return portfolio


def generate_hybrid_portfolio(valuations_df, budget=BUDGET):
    """
    Hybrid: 3-4 early value targets + 15-20 late accumulation picks.
    Balance concentrated value with volume upside.
    """
    config = STRATEGIES['hybrid']
    
    early_budget = budget * config['early_budget']  # $50
    late_budget = budget - early_budget  # $50
    
    # EARLY: Top value picks (seeds 1-7)
    early_candidates = valuations_df[
        (valuations_df['Seed'] <= 7) &
        (valuations_df['Value_Rating'] >= 1.0)
    ].copy()
    
    early_candidates = early_candidates.sort_values('Value_Rating', ascending=False)
    early_picks = early_candidates.head(4).copy()
    
    total_exp_points = early_picks['E_Points_Blended'].sum()
    early_picks['Allocated_Bid'] = (
        (early_picks['E_Points_Blended'] / total_exp_points) * early_budget
    ).round(0).astype(int)
    
    # LATE: Volume picks (seeds 8-15)
    late_candidates = valuations_df[
        (valuations_df['Seed'] >= 8) &
        (valuations_df['Expected_Market_Price'] <= 10)
    ].copy()
    
    late_candidates = late_candidates.sort_values('E_Points_Blended', ascending=False)
    late_picks = late_candidates.head(16).copy()
    
    weights = np.linspace(1.2, 0.5, len(late_picks))
    weights = weights / weights.sum()
    late_picks['Allocated_Bid'] = (weights * late_budget).round(0).astype(int)
    late_picks['Allocated_Bid'] = late_picks['Allocated_Bid'].clip(upper=8)
    
    # Combine
    portfolio = pd.concat([early_picks, late_picks], ignore_index=True)
    
    # Adjust for budget
    diff = budget - portfolio['Allocated_Bid'].sum()
    if diff != 0:
        portfolio.loc[portfolio.index[0], 'Allocated_Bid'] += diff
    
    portfolio = portfolio[[
        'Team', 'Seed', 'E_Points_Blended', 'Expected_Market_Price',
        'Fair_Value', 'Value_Rating', 'Allocated_Bid'
    ]].copy()
    
    portfolio['Strategy'] = config['name']
    
    return portfolio


def evaluate_portfolios(portfolios):
    """Calculate expected points and variance for each portfolio."""
    results = []
    
    for name, portfolio in portfolios.items():
        total_bid = portfolio['Allocated_Bid'].sum()
        expected_points = portfolio['E_Points_Blended'].sum()
        n_teams = len(portfolio)
        
        results.append({
            'Strategy': name,
            'N_Teams': n_teams,
            'Total_Bid': total_bid,
            'Expected_Points': round(expected_points, 2),
            'Pts_Per_Dollar': round(expected_points / total_bid, 3),
            'Avg_Value_Rating': round(portfolio['Value_Rating'].mean(), 3)
        })
    
    return pd.DataFrame(results)


# ============================================================================
# VALUE IDENTIFICATION
# ============================================================================

def identify_value_opportunities(valuations_df):
    """
    Flag best value picks by price tier:
    - Bargain tier ($1-5): Best upset potential
    - Value tier ($5-15): Best E[points]/price
    - Premium tier ($15-30): Avoid overpriced favorites
    """
    print("Identifying value opportunities...")
    
    opportunities = {}
    
    # Bargain tier ($1-5 expected price)
    bargains = valuations_df[
        valuations_df['Expected_Market_Price'] <= 5
    ].nlargest(10, 'E_Points_Blended')
    opportunities['Bargain_Tier_1_5'] = bargains[[
        'Team', 'Seed', 'E_Points_Blended', 'Expected_Market_Price', 'Value_Rating'
    ]]
    
    # Value tier ($5-15 expected price)
    value_tier = valuations_df[
        (valuations_df['Expected_Market_Price'] > 5) &
        (valuations_df['Expected_Market_Price'] <= 15)
    ].nlargest(10, 'Value_Rating')
    opportunities['Value_Tier_5_15'] = value_tier[[
        'Team', 'Seed', 'E_Points_Blended', 'Expected_Market_Price', 'Value_Rating'
    ]]
    
    # Premium tier (avoid overpriced)
    premium = valuations_df[
        valuations_df['Expected_Market_Price'] > 15
    ].nsmallest(10, 'Value_Rating')  # Worst value = avoid
    opportunities['Avoid_Overpriced'] = premium[[
        'Team', 'Seed', 'E_Points_Blended', 'Expected_Market_Price', 'Value_Rating'
    ]]
    
    return opportunities


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_visualizations(valuations_df, auction_market_df, portfolios_eval):
    """Create diagnostic visualizations."""
    print("Creating visualizations...")
    sns.set_style("whitegrid")
    
    # 1. Value rating by seed
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seed_value = valuations_df.groupby('Seed')['Value_Rating'].mean().reset_index()
    
    ax.bar(seed_value['Seed'], seed_value['Value_Rating'], 
           color='steelblue', edgecolor='black', alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Market Value (1.0)')
    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value Rating (E[pts] / Expected Price)', fontsize=12, fontweight='bold')
    ax.set_title('Value Rating by Seed (2026)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'value_rating_by_seed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Expected points vs expected price
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        valuations_df['Expected_Market_Price'],
        valuations_df['E_Points_Blended'],
        c=valuations_df['Value_Rating'],
        s=100,
        cmap='RdYlGn',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.7
    )
    
    # Annotate top value picks
    top_value = valuations_df.nlargest(8, 'Value_Rating')
    for _, row in top_value.iterrows():
        ax.annotate(
            f"{row['Team']} ({row['Seed']})",
            (row['Expected_Market_Price'], row['E_Points_Blended']),
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6)
        )
    
    ax.set_xlabel('Expected Market Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Points', fontsize=12, fontweight='bold')
    ax.set_title('Expected Points vs Market Price (2026)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Value Rating', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'expected_points_vs_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Historical auction ROI by seed
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(auction_market_df['Seed'], auction_market_df['ROI'], 
           color='coral', edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI (Points - Bid) / Bid', fontsize=12, fontweight='bold')
    ax.set_title('Historical ROI by Seed (2022-2025)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'historical_roi_by_seed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Portfolio comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(portfolios_eval))
    width = 0.35
    
    ax.bar(x - width/2, portfolios_eval['Expected_Points'], width,
           label='Expected Points', color='steelblue', edgecolor='black', alpha=0.8)
    ax.bar(x + width/2, portfolios_eval['N_Teams'], width,
           label='N Teams', color='coral', edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios_eval['Strategy'], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualizations saved to {VIZ_DIR}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("           L4.02 — CALCUTTA PORTFOLIO OPTIMIZER")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: Historical Analysis
    # ========================================================================
    print("STEP 1: Historical Tournament Analysis")
    print("-" * 70)
    
    results_df = load_tournament_results()
    print(f"  ✓ Loaded {len(results_df)} games (2008-2025)")
    
    seed_performance = calculate_historical_seed_performance(results_df)
    print(f"  ✓ Calculated seed performance for {len(seed_performance)} seeds")
    
    upset_bonuses = calculate_upset_bonuses(results_df)
    print(f"  ✓ Calculated upset bonus patterns")
    
    expected_points_hist = calculate_expected_points_historical(
        seed_performance, upset_bonuses
    )
    print(f"  ✓ Calculated historical E[points] by seed")
    print()
    
    # ========================================================================
    # STEP 2: Auction Market Analysis
    # ========================================================================
    print("STEP 2: Auction Market Analysis")
    print("-" * 70)
    
    auction_df = load_auction_history()
    print(f"  ✓ Loaded {len(auction_df)} team purchases (2022-2025)")
    
    auction_market = analyze_auction_market(auction_df)
    print(f"  ✓ Analyzed market dynamics for {len(auction_market)} seeds")
    
    best_value, worst_value = identify_historical_value_picks(auction_df)
    print(f"  ✓ Identified historical value picks")
    print()
    
    # ========================================================================
    # STEP 3: 2026 Model Predictions
    # ========================================================================
    print("STEP 3: Loading 2026 Model Predictions")
    print("-" * 70)
    
    round_probs_2026, elite8_direct = load_2026_predictions()
    print(f"  ✓ Loaded predictions for {len(round_probs_2026)} teams")
    
    expected_points_2026 = calculate_2026_expected_points(
        round_probs_2026, upset_bonuses
    )
    print(f"  ✓ Calculated 2026 E[points] from model")
    print()
    
    # ========================================================================
    # STEP 4: Team Valuations
    # ========================================================================
    print("STEP 4: Calculating 2026 Team Valuations")
    print("-" * 70)
    
    # Merge historical baseline with model predictions
    historical_with_expected = seed_performance.merge(
        expected_points_hist, on='Seed'
    )
    
    team_valuations = blend_historical_and_model(
        historical_with_expected, expected_points_2026
    )
    print(f"  ✓ Blended historical + model (70/30)")
    
    team_valuations = estimate_market_prices(team_valuations, auction_market)
    print(f"  ✓ Estimated market prices")
    print()
    
    # ========================================================================
    # STEP 5: Portfolio Generation
    # ========================================================================
    print("STEP 5: Generating Portfolio Strategies")
    print("-" * 70)
    
    portfolios = {
        'Value Hunter': generate_value_hunter_portfolio(team_valuations),
        'Patient Accumulator': generate_patient_accumulator_portfolio(team_valuations),
        'Hybrid': generate_hybrid_portfolio(team_valuations)
    }
    
    for name, portfolio in portfolios.items():
        print(f"  ✓ {name}: {len(portfolio)} teams, "
              f"${portfolio['Allocated_Bid'].sum()} budget, "
              f"{portfolio['E_Points_Blended'].sum():.1f} exp points")
    print()
    
    portfolios_eval = evaluate_portfolios(portfolios)
    
    # ========================================================================
    # STEP 6: Value Opportunities
    # ========================================================================
    print("STEP 6: Identifying Value Opportunities")
    print("-" * 70)
    
    value_opps = identify_value_opportunities(team_valuations)
    for tier, teams in value_opps.items():
        print(f"  ✓ {tier}: {len(teams)} teams")
    print()
    
    # ========================================================================
    # STEP 7: Outputs
    # ========================================================================
    print("STEP 7: Saving Outputs")
    print("-" * 70)
    
    # Historical baseline
    historical_output = seed_performance.merge(expected_points_hist, on='Seed')
    historical_output.to_csv(OUTPUT_DIR / 'historical_baseline.csv', index=False)
    print(f"  ✓ historical_baseline.csv")
    
    # Auction market analysis
    auction_market.to_csv(OUTPUT_DIR / 'auction_market_analysis.csv', index=False)
    print(f"  ✓ auction_market_analysis.csv")
    
    # Team valuations
    team_valuations_sorted = team_valuations.sort_values(
        'E_Points_Blended', ascending=False
    )
    team_valuations_sorted.to_csv(OUTPUT_DIR / 'team_valuations_2026.csv', index=False)
    print(f"  ✓ team_valuations_2026.csv")
    
    # Portfolios
    for name, portfolio in portfolios.items():
        filename = f"portfolio_{name.lower().replace(' ', '_')}.csv"
        portfolio.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"  ✓ {filename}")
    
    portfolios_eval.to_csv(OUTPUT_DIR / 'portfolio_comparison.csv', index=False)
    print(f"  ✓ portfolio_comparison.csv")
    
    # Value opportunities
    for tier, teams in value_opps.items():
        filename = f"value_opportunities_{tier.lower()}.csv"
        teams.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"  ✓ {filename}")
    
    # Best/worst historical picks
    best_value.to_csv(OUTPUT_DIR / 'historical_best_value.csv', index=False)
    worst_value.to_csv(OUTPUT_DIR / 'historical_worst_value.csv', index=False)
    print(f"  ✓ historical_best_value.csv")
    print(f"  ✓ historical_worst_value.csv")
    print()
    
    # Visualizations
    create_visualizations(team_valuations, auction_market, portfolios_eval)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 70)
    print("                         SUMMARY")
    print("=" * 70)
    print()
    
    print("  HISTORICAL INSIGHTS (2022-2025)")
    print("  " + "-" * 66)
    print(f"  Best value seeds: {auction_market.nlargest(3, 'Value_Pts_Per_Dollar')['Seed'].tolist()}")
    print(f"  Worst value seeds: {auction_market.nsmallest(3, 'Value_Pts_Per_Dollar')['Seed'].tolist()}")
    print()
    
    print("  TOP 10 VALUE PICKS (2026)")
    print("  " + "-" * 66)
    print("  Value Rating = E[pts] / Expected Price (>1.0 = undervalued)")
    print()
    top10_value = team_valuations.nlargest(10, 'Value_Rating')
    for _, row in top10_value.iterrows():
        print(f"  {row['Team']:<20} ({row['Seed']}) "
              f"E[pts]={row['E_Points_Blended']:>5.1f}  "
              f"Price=${row['Expected_Market_Price']:>4.0f}  "
              f"Value={row['Value_Rating']:>4.2f}")
    print()
    
    print("  PORTFOLIO RECOMMENDATIONS")
    print("  " + "-" * 66)
    for _, row in portfolios_eval.iterrows():
        print(f"  {row['Strategy']:<25} "
              f"{row['N_Teams']:>2} teams  "
              f"{row['Expected_Points']:>5.1f} pts  "
              f"${row['Total_Bid']:>3}  "
              f"({row['Pts_Per_Dollar']:.2f} pts/$)")
    print()
    
    print("  AVOID (Worst Value Ratings)")
    print("  " + "-" * 66)
    avoid = team_valuations.nsmallest(5, 'Value_Rating')
    for _, row in avoid.iterrows():
        print(f"  {row['Team']:<20} ({row['Seed']}) "
              f"E[pts]={row['E_Points_Blended']:>5.1f}  "
              f"Price=${row['Expected_Market_Price']:>4.0f}  "
              f"Value={row['Value_Rating']:>4.2f}")
    print()
    
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
