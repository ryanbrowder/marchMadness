"""
L4.02 - Calcutta Bid Guidance Optimizer

Analyzes historical auction data to provide actionable bid ranges for 2026 teams.

Based on:
  - Historical auction prices by seed (2013-2025)
  - Historical points scored by seed
  - What it takes to win 1st/2nd place
  - 2026 model predictions (expected points)
  - Pot structure and payout

Outputs:
  - Bid ranges per team (Conservative / Fair / Aggressive / Pass)
  - Value opportunities
  - Portfolio recommendations
  - Historical performance analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = Path("data")
AUCTION_HISTORY = DATA_DIR / "auction_history.csv"
ROUND_PROBS_2026 = Path("outputs/01_tournament_simulator/round_probabilities.csv")

# Outputs
OUTPUT_DIR = Path("outputs/02_calcutta_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Calcutta rules
ENTRY_FEE = 20  # Per player
PAYOUT_1ST = 0.70  # 70% to winner
PAYOUT_2ND = 0.30  # 30% to runner-up

# ============================================================================
# LOAD DATA
# ============================================================================

def load_auction_history():
    """Load historical auction data (2013-2025)."""
    print("Loading historical auction data...")
    df = pd.read_csv(AUCTION_HISTORY)
    print(f"  ✓ Loaded {len(df)} auction records ({df['Year'].min()}-{df['Year'].max()})")
    return df


def load_2026_predictions():
    """Load 2026 team predictions from L4.01."""
    print("Loading 2026 model predictions...")
    df = pd.read_csv(ROUND_PROBS_2026)
    
    # Rename P_FF to P_F4 if needed
    if 'P_FF' in df.columns:
        df = df.rename(columns={'P_FF': 'P_F4'})
    
    # Exclude 16-seeds (not in auction)
    df = df[df['Seed'] != 16].copy()
    
    # Calculate expected points from probabilities
    df = calculate_expected_points(df)
    
    print(f"  ✓ Loaded predictions for {len(df)} teams (16-seeds excluded)")
    return df


def calculate_expected_points(df):
    """Calculate expected points from round probabilities with upset bonuses."""
    
    # Scoring system
    SCORING = {
        'R64': 2,
        'R32': 4,
        'S16': 6,
        'E8': 8,
        'F4': 10,
        'Championship': 12
    }
    
    # R64 matchups (known from bracket structure)
    R64_MATCHUPS = {1: 16, 2: 15, 3: 14, 4: 13, 5: 12, 6: 11, 7: 10, 8: 9,
                    16: 1, 15: 2, 14: 3, 13: 4, 12: 5, 11: 6, 10: 7, 9: 8}
    
    # Expected opponent seeds for later rounds (if chalk holds)
    EXPECTED_OPPONENT = {
        'R32': 8.0,   # Winner of 8/9 game
        'S16': 4.5,   # Winner of 4/5 vs 12/13
        'E8': 3.0,    # Winner of 2/3 region
        'F4': 2.0,    # Winner of 1/2 region
    }
    
    expected_points = []
    
    for _, row in df.iterrows():
        seed = int(row['Seed'])
        
        # Base points (no upset bonus)
        base = 0
        base += row.get('P_R64', 1.0) * SCORING['R64']
        base += row.get('P_R32', 0) * SCORING['R32']
        base += row.get('P_S16', 0) * SCORING['S16']
        base += row.get('P_E8', 0) * SCORING['E8']
        base += row.get('P_F4', 0) * SCORING['F4']
        base += row.get('P_Championship', 0) * SCORING['Championship']
        
        # Upset bonuses (using ABSOLUTE probabilities)
        upset_bonus = 0.0
        
        # R64: P(win R64) = P_R32
        r64_opponent = R64_MATCHUPS.get(seed, seed)
        if seed > r64_opponent:
            p_win_r64 = row.get('P_R32', 0)
            upset_bonus += p_win_r64 * (seed - r64_opponent)
        
        # R32: P(win R32) = P_S16
        if seed > EXPECTED_OPPONENT['R32']:
            p_win_r32 = row.get('P_S16', 0)
            upset_bonus += p_win_r32 * (seed - EXPECTED_OPPONENT['R32'])
        
        # S16: P(win S16) = P_E8
        if seed > EXPECTED_OPPONENT['S16']:
            p_win_s16 = row.get('P_E8', 0)
            upset_bonus += p_win_s16 * (seed - EXPECTED_OPPONENT['S16'])
        
        # E8: P(win E8) = P_F4
        if seed > EXPECTED_OPPONENT['E8']:
            p_win_e8 = row.get('P_F4', 0)
            upset_bonus += p_win_e8 * (seed - EXPECTED_OPPONENT['E8'])
        
        # F4: P(win F4) = P_Championship
        if seed > EXPECTED_OPPONENT['F4']:
            p_win_f4 = row.get('P_Championship', 0)
            upset_bonus += p_win_f4 * (seed - EXPECTED_OPPONENT['F4'])
        
        expected_points.append(base + upset_bonus)
    
    df['E_Points_Blended'] = [round(pt, 2) for pt in expected_points]
    
    return df


# ============================================================================
# HISTORICAL ANALYSIS
# ============================================================================

def analyze_winners(auction_df):
    """Analyze historical winners and runner-ups."""
    print("Analyzing historical winners...")
    
    # Calculate total points per player per year
    player_totals = auction_df.groupby(['Year', 'Player']).agg({
        'Points': 'sum',
        'Bid': 'sum',
        'Team': 'count'
    }).rename(columns={'Team': 'N_Teams'}).reset_index()
    
    # Find winners and runner-ups
    winners = []
    for year in sorted(player_totals['Year'].unique()):
        year_data = player_totals[player_totals['Year'] == year].sort_values('Points', ascending=False)
        
        if len(year_data) >= 2:
            winner = year_data.iloc[0]
            runner_up = year_data.iloc[1]
            
            winners.append({
                'Year': year,
                'Winner': winner['Player'],
                'Winner_Points': winner['Points'],
                'Winner_Teams': winner['N_Teams'],
                'Runner_Up': runner_up['Player'],
                'Runner_Up_Points': runner_up['Points'],
                'Runner_Up_Teams': runner_up['N_Teams']
            })
    
    winners_df = pd.DataFrame(winners)
    
    print(f"  ✓ Average winning total: {winners_df['Winner_Points'].mean():.1f} points")
    print(f"  ✓ Average 2nd place: {winners_df['Runner_Up_Points'].mean():.1f} points")
    
    return winners_df


def analyze_seed_performance(auction_df):
    """Analyze historical performance by seed."""
    print("Analyzing seed performance...")
    
    # Filter to valid seeds
    valid_df = auction_df[auction_df['Seed'] > 0].copy()
    
    # Aggregate by seed - include percentiles
    seed_stats = valid_df.groupby('Seed').agg({
        'Bid': ['mean', 'median', 'std', 'count', 
                lambda x: x.quantile(0.25),  # 25th percentile
                lambda x: x.quantile(0.75)], # 75th percentile
        'Points': ['mean', 'median', 'std']
    }).round(2)
    
    seed_stats.columns = ['Bid_mean', 'Bid_median', 'Bid_std', 'Bid_count', 
                          'Bid_p25', 'Bid_p75',
                          'Points_mean', 'Points_median', 'Points_std']
    seed_stats = seed_stats.reset_index()
    
    # Calculate ROI
    seed_stats['ROI'] = ((seed_stats['Points_mean'] - seed_stats['Bid_mean']) / 
                         seed_stats['Bid_mean']).round(3)
    seed_stats['Points_Per_Dollar'] = (seed_stats['Points_mean'] / 
                                       seed_stats['Bid_mean']).round(3)
    
    good_value_seeds = seed_stats[seed_stats['ROI'] > 0]['Seed'].tolist()
    print(f"  ✓ Seeds with positive historical ROI: {good_value_seeds}")
    
    return seed_stats


def estimate_pot_structure(auction_df):
    """Estimate pot structure from historical data."""
    players_per_year = auction_df.groupby('Year')['Player'].nunique()
    avg_players = players_per_year.mean()
    
    total_pot = avg_players * ENTRY_FEE
    first_place = total_pot * PAYOUT_1ST
    second_place = total_pot * PAYOUT_2ND
    
    return {
        'avg_players': avg_players,
        'total_pot': total_pot,
        'first_place': first_place,
        'second_place': second_place
    }


# ============================================================================
# BID GUIDANCE CALCULATION
# ============================================================================

def calculate_bid_guidance(team_valuations, seed_stats, winners_df, pot_info):
    """Calculate executable auction strategy with tier assignments."""
    print("Building executable auction strategy...")
    
    # Benchmarks
    WINNING_THRESHOLD = winners_df['Winner_Points'].quantile(0.25)
    
    print(f"  ✓ Winning threshold: {WINNING_THRESHOLD:.0f} points")
    print(f"  ✓ Building tier-based strategy")
    
    # Merge with historical seed stats
    team_valuations = team_valuations.merge(
        seed_stats[['Seed', 'Bid_mean', 'Bid_median', 'Bid_p25', 'Bid_p75', 
                   'Points_mean', 'ROI']],
        on='Seed',
        how='left',
        suffixes=('', '_hist')
    )
    
    # Calculate model advantage
    team_valuations['Points_Advantage'] = (
        team_valuations['E_Points_Blended'] - team_valuations['Points_mean']
    ).round(1)
    
    # Market price = historical median
    team_valuations['Market_Price'] = team_valuations['Bid_median'].round(0)
    
    # VALUE METRICS
    team_valuations['Pts_Per_Market_Dollar'] = (
        team_valuations['E_Points_Blended'] / team_valuations['Market_Price']
    ).round(3)
    
    # TIER ASSIGNMENT
    def assign_tier(row):
        exp_pts = row['E_Points_Blended']
        advantage = row['Points_Advantage']
        efficiency = row['Pts_Per_Market_Dollar']
        
        if pd.isna(advantage):
            return 'PASS'
        
        # TIER 1: ANCHORS - High floor, high ceiling teams
        if exp_pts >= 20 and advantage >= 5:
            return 'ANCHOR'
        elif exp_pts >= 15 and advantage >= 8:
            return 'ANCHOR'
        
        # TIER 2: QUALITY FILL - Solid supporting cast
        elif exp_pts >= 12 and advantage >= 4:
            return 'FILL'
        elif exp_pts >= 10 and advantage >= 6:
            return 'FILL'
        
        # TIER 3: VALUE - Efficiency plays
        elif exp_pts >= 8 and efficiency >= 1.1:
            return 'VALUE'
        elif exp_pts >= 8 and advantage >= 3:
            return 'VALUE'
        
        # TIER 4: OPPORTUNISTIC - Only if major discount
        elif exp_pts >= 6 and advantage >= 2:
            return 'OPPORTUNISTIC'
        
        # TIER 5: FADE - Model doesn't like them
        else:
            return 'FADE'
    
    team_valuations['Tier'] = team_valuations.apply(assign_tier, axis=1)
    
    # BID GUIDANCE BY PHASE
    # Phase 1 (0-2 teams, securing anchor): Pay up for quality
    # Phase 2 (3-5 teams, building core): Pay market for good value
    # Phase 3 (6+ teams, value hunting): Only pay below market
    
    def calculate_phase_bids(row):
        market = row['Market_Price']
        tier = row['Tier']
        advantage = row['Points_Advantage']
        
        if tier == 'ANCHOR':
            # Phase 1: Willing to pay premium
            phase1 = market + 5 if advantage >= 10 else market + 3
            # Phase 2: Pay market
            phase2 = market
            # Phase 3: Pass (already have anchor)
            phase3 = 0
            
        elif tier == 'FILL':
            # Phase 1: Pay market + small premium if exceptional
            phase1 = market + 2 if advantage >= 7 else market
            # Phase 2: Pay market
            phase2 = market if advantage >= 6 else market - 2
            # Phase 3: Only if discount
            phase3 = market - 3
            
        elif tier == 'VALUE':
            # Phase 1: Pass (focus on anchors)
            phase1 = 0
            # Phase 2: Pay market
            phase2 = market
            # Phase 3: Best value here
            phase3 = market - 1
            
        elif tier == 'OPPORTUNISTIC':
            # Only bid if major discount in any phase
            phase1 = market - 3
            phase2 = market - 2
            phase3 = market - 1
            
        else:  # FADE
            phase1 = 0
            phase2 = 0
            phase3 = 0
        
        return pd.Series({
            'Bid_Phase1': max(0, phase1),
            'Bid_Phase2': max(0, phase2),
            'Bid_Phase3': max(0, phase3)
        })
    
    phase_bids = team_valuations.apply(calculate_phase_bids, axis=1)
    team_valuations = pd.concat([team_valuations, phase_bids], axis=1)
    
    return team_valuations


# ============================================================================
# PORTFOLIO RECOMMENDATIONS
# ============================================================================


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def print_summary(team_valuations, seed_stats, winners_df, pot_info):
    """Print executable auction strategy cheat sheet."""
    print()
    print("=" * 80)
    print("                    AUCTION STRATEGY CHEAT SHEET")
    print("=" * 80)
    print()
    
    # Quick reference
    print("  TARGET: 76+ points to win | YOUR BUDGET: $100 | TARGET: 7-8 teams")
    print()
    
    # Auction phases
    print("  AUCTION PHASES")
    print("  " + "-" * 76)
    print("  Phase 1 (Teams 0-2): SECURE YOUR ANCHOR - Be aggressive on top targets")
    print("  Phase 2 (Teams 3-5): BUILD YOUR CORE - Pay market for quality fill")
    print("  Phase 3 (Teams 6-8): VALUE HUNTING - Only bid below market")
    print()
    
    # Tier 1: Anchors
    anchors = team_valuations[team_valuations['Tier'] == 'ANCHOR'].sort_values(
        'E_Points_Blended', ascending=False
    )
    
    if len(anchors) > 0:
        print("  TIER 1: ANCHORS (Pick 1-2, your foundation)")
        print("  " + "-" * 76)
        print("  Team            | Seed | Pts | Advantage | Market | Phase 1 | Phase 2 | Phase 3")
        print("  ----------------|------|-----|-----------|--------|---------|---------|--------")
        for _, row in anchors.head(8).iterrows():
            p1 = f"${row['Bid_Phase1']:.0f}" if row['Bid_Phase1'] > 0 else "PASS"
            p2 = f"${row['Bid_Phase2']:.0f}" if row['Bid_Phase2'] > 0 else "PASS"
            p3 = f"${row['Bid_Phase3']:.0f}" if row['Bid_Phase3'] > 0 else "PASS"
            
            print(f"  {row['Team']:<15} | {row['Seed']:>4} | "
                  f"{row['E_Points_Blended']:>3.0f} | "
                  f"+{row['Points_Advantage']:>5.1f}    | "
                  f"${row['Market_Price']:>5.0f}  | "
                  f"{p1:>7} | {p2:>7} | {p3:>6}")
        print()
    
    # Tier 2: Fill
    fills = team_valuations[team_valuations['Tier'] == 'FILL'].sort_values(
        'E_Points_Blended', ascending=False
    )
    
    if len(fills) > 0:
        print("  TIER 2: QUALITY FILL (Pick 3-4, supporting cast)")
        print("  " + "-" * 76)
        print("  Team            | Seed | Pts | Advantage | Market | Phase 1 | Phase 2 | Phase 3")
        print("  ----------------|------|-----|-----------|--------|---------|---------|--------")
        for _, row in fills.head(10).iterrows():
            p1 = f"${row['Bid_Phase1']:.0f}" if row['Bid_Phase1'] > 0 else "PASS"
            p2 = f"${row['Bid_Phase2']:.0f}" if row['Bid_Phase2'] > 0 else "PASS"
            p3 = f"${row['Bid_Phase3']:.0f}" if row['Bid_Phase3'] > 0 else "PASS"
            
            print(f"  {row['Team']:<15} | {row['Seed']:>4} | "
                  f"{row['E_Points_Blended']:>3.0f} | "
                  f"+{row['Points_Advantage']:>5.1f}    | "
                  f"${row['Market_Price']:>5.0f}  | "
                  f"{p1:>7} | {p2:>7} | {p3:>6}")
        print()
    
    # Tier 3: Value
    values = team_valuations[team_valuations['Tier'] == 'VALUE'].sort_values(
        'Pts_Per_Market_Dollar', ascending=False
    )
    
    if len(values) > 0:
        print("  TIER 3: VALUE PLAYS (Pick 2-3, efficiency targets)")
        print("  " + "-" * 76)
        print("  Team            | Seed | Pts | Pts/$  | Market | Phase 1 | Phase 2 | Phase 3")
        print("  ----------------|------|-----|--------|--------|---------|---------|--------")
        for _, row in values.head(8).iterrows():
            p1 = f"${row['Bid_Phase1']:.0f}" if row['Bid_Phase1'] > 0 else "PASS"
            p2 = f"${row['Bid_Phase2']:.0f}" if row['Bid_Phase2'] > 0 else "PASS"
            p3 = f"${row['Bid_Phase3']:.0f}" if row['Bid_Phase3'] > 0 else "PASS"
            
            print(f"  {row['Team']:<15} | {row['Seed']:>4} | "
                  f"{row['E_Points_Blended']:>3.0f} | "
                  f"{row['Pts_Per_Market_Dollar']:>6.2f} | "
                  f"${row['Market_Price']:>5.0f}  | "
                  f"{p1:>7} | {p2:>7} | {p3:>6}")
        print()
    
    # Strategic portfolio paths
    print("  STRATEGIC PORTFOLIO PATHS (Pick One)")
    print("  " + "-" * 76)
    print("  Each path fits $100 budget and exceeds 76-point threshold")
    print()
    
    # PATH A: Premium Anchor Strategy
    print("  PATH A: PREMIUM ANCHOR (Quality over quantity)")
    print("  " + "-" * 76)
    
    path_a = []
    path_a_budget = 100
    
    # Get best anchor
    if len(anchors) > 0:
        best_anchor = anchors.iloc[0]
        anchor_cost = min(best_anchor['Bid_Phase1'], 43)  # Cap to leave room
        path_a.append({
            'team': best_anchor['Team'],
            'seed': best_anchor['Seed'],
            'pts': best_anchor['E_Points_Blended'],
            'cost': anchor_cost
        })
        path_a_budget -= anchor_cost
    
    # Get 3 best fills at Phase 2 prices
    fill_count = 0
    for _, row in fills.iterrows():
        if fill_count >= 3 and path_a_budget < 15:
            break
        if fill_count >= 3:
            break
        cost = min(row['Bid_Phase2'], path_a_budget - 15)
        if cost > 0:
            path_a.append({
                'team': row['Team'],
                'seed': row['Seed'],
                'pts': row['E_Points_Blended'],
                'cost': cost
            })
            path_a_budget -= cost
            fill_count += 1
    
    # Fill rest with value at Phase 3 prices
    for _, row in values.iterrows():
        if len(path_a) >= 8:
            break
        if path_a_budget <= 0:
            break
        cost = min(row['Bid_Phase3'], path_a_budget)
        if cost > 0 and row['Team'] not in [t['team'] for t in path_a]:
            path_a.append({
                'team': row['Team'],
                'seed': row['Seed'],
                'pts': row['E_Points_Blended'],
                'cost': cost
            })
            path_a_budget -= cost
    
    path_a_total_pts = sum(t['pts'] for t in path_a)
    path_a_total_cost = sum(t['cost'] for t in path_a)
    
    for team in path_a:
        print(f"  {team['team']:<18} (Seed {team['seed']:>2}) → "
              f"${team['cost']:>3.0f}  [{team['pts']:>5.1f} pts]")
    
    print(f"  {'-'*76}")
    print(f"  TOTAL: {len(path_a)} teams, {path_a_total_pts:.1f} points, ${path_a_total_cost:.0f} spent")
    print(f"  Strategy: Secure 1 elite anchor, surround with best available value")
    print()
    
    # PATH B: Dual Mid-Tier Strategy
    print("  PATH B: DUAL MID-TIER (Balanced foundation)")
    print("  " + "-" * 76)
    
    path_b = []
    path_b_budget = 100
    
    # Get 2 mid-tier anchors (Houston + Iowa St type)
    anchor_count = 0
    for _, row in anchors.iterrows():
        if anchor_count >= 2:
            break
        # Skip most expensive, take 2nd/3rd best
        if row['Market_Price'] <= 30 and anchor_count < 2:
            cost = row['Bid_Phase2']
            path_b.append({
                'team': row['Team'],
                'seed': row['Seed'],
                'pts': row['E_Points_Blended'],
                'cost': cost
            })
            path_b_budget -= cost
            anchor_count += 1
    
    # Get 3 fills at market or below
    fill_count = 0
    for _, row in fills.iterrows():
        if fill_count >= 3:
            break
        if row['Team'] not in [t['team'] for t in path_b]:
            cost = min(row['Bid_Phase2'], path_b_budget - 10)
            if cost > 0:
                path_b.append({
                    'team': row['Team'],
                    'seed': row['Seed'],
                    'pts': row['E_Points_Blended'],
                    'cost': cost
                })
                path_b_budget -= cost
                fill_count += 1
    
    # Add value plays
    for _, row in values.iterrows():
        if len(path_b) >= 8:
            break
        if row['Team'] not in [t['team'] for t in path_b] and path_b_budget > 0:
            cost = min(row['Bid_Phase2'], path_b_budget)
            if cost > 0:
                path_b.append({
                    'team': row['Team'],
                    'seed': row['Seed'],
                    'pts': row['E_Points_Blended'],
                    'cost': cost
                })
                path_b_budget -= cost
    
    path_b_total_pts = sum(t['pts'] for t in path_b)
    path_b_total_cost = sum(t['cost'] for t in path_b)
    
    for team in path_b:
        print(f"  {team['team']:<18} (Seed {team['seed']:>2}) → "
              f"${team['cost']:>3.0f}  [{team['pts']:>5.1f} pts]")
    
    print(f"  {'-'*76}")
    print(f"  TOTAL: {len(path_b)} teams, {path_b_total_pts:.1f} points, ${path_b_total_cost:.0f} spent")
    print(f"  Strategy: Two solid anchors (Houston/Iowa St tier), deep supporting cast")
    print()
    
    # PATH C: Deep Value Strategy
    print("  PATH C: DEEP VALUE (Efficiency maximization)")
    print("  " + "-" * 76)
    
    path_c = []
    path_c_budget = 100
    
    # Get 1 cheap anchor
    for _, row in anchors.iterrows():
        if row['Market_Price'] <= 20:
            cost = row['Bid_Phase2']
            path_c.append({
                'team': row['Team'],
                'seed': row['Seed'],
                'pts': row['E_Points_Blended'],
                'cost': cost
            })
            path_c_budget -= cost
            break
    
    # Get fills at Phase 3 prices (discounts)
    for _, row in fills.iterrows():
        if len(path_c) >= 8:
            break
        if row['Team'] not in [t['team'] for t in path_c]:
            cost = row['Bid_Phase3']
            if cost > 0 and path_c_budget >= cost:
                path_c.append({
                    'team': row['Team'],
                    'seed': row['Seed'],
                    'pts': row['E_Points_Blended'],
                    'cost': cost
                })
                path_c_budget -= cost
    
    # Add values
    for _, row in values.iterrows():
        if len(path_c) >= 9:
            break
        if row['Team'] not in [t['team'] for t in path_c] and path_c_budget > 0:
            cost = min(row['Bid_Phase3'], path_c_budget)
            if cost > 0:
                path_c.append({
                    'team': row['Team'],
                    'seed': row['Seed'],
                    'pts': row['E_Points_Blended'],
                    'cost': cost
                })
                path_c_budget -= cost
    
    path_c_total_pts = sum(t['pts'] for t in path_c)
    path_c_total_cost = sum(t['cost'] for t in path_c)
    
    for team in path_c:
        print(f"  {team['team']:<18} (Seed {team['seed']:>2}) → "
              f"${team['cost']:>3.0f}  [{team['pts']:>5.1f} pts]")
    
    print(f"  {'-'*76}")
    print(f"  TOTAL: {len(path_c)} teams, {path_c_total_pts:.1f} points, ${path_c_total_cost:.0f} spent")
    print(f"  Strategy: Max teams, bargain hunting, volume over star power")
    print()
    
    # Path comparison
    print("  PATH COMPARISON")
    print("  " + "-" * 76)
    print(f"  Path A: {len(path_a)} teams, {path_a_total_pts:.0f} pts, ${path_a_total_cost:.0f} → "
          f"{'✓ VIABLE' if path_a_total_pts >= 76 and path_a_total_cost <= 100 else '✗ ADJUST'}")
    print(f"  Path B: {len(path_b)} teams, {path_b_total_pts:.0f} pts, ${path_b_total_cost:.0f} → "
          f"{'✓ VIABLE' if path_b_total_pts >= 76 and path_b_total_cost <= 100 else '✗ ADJUST'}")
    print(f"  Path C: {len(path_c)} teams, {path_c_total_pts:.0f} pts, ${path_c_total_cost:.0f} → "
          f"{'✓ VIABLE' if path_c_total_pts >= 76 and path_c_total_cost <= 100 else '✗ ADJUST'}")
    print()
    print("  Pick based on auction flow and which teams you can actually win")
    print()
    
    # Quick decision rules
    print("  AUCTION DECISION RULES")
    print("  " + "-" * 76)
    print("  When team comes up, ask:")
    print("    1. What tier? (ANCHOR / FILL / VALUE)")
    print("    2. What phase am I in? (0-2 teams / 3-5 teams / 6+ teams)")
    print("    3. Check table above for max bid")
    print()
    print("  Key principles:")
    print("    - ANCHORS: Pay up in Phase 1, pay market in Phase 2, pass in Phase 3")
    print("    - FILLS: Small premium in Phase 1, market in Phase 2, discount in Phase 3")
    print("    - VALUES: Pass Phase 1, market Phase 2, best deals Phase 3")
    print("    - If bidding war starts above your phase bid → LET THEM HAVE IT")
    print()
    
    # Historical ROI reminder
    print("  HISTORICAL CONTEXT (Why our targets make sense)")
    print("  " + "-" * 76)
    print("  Historically overpriced seeds (market overpays):")
    worst_roi = seed_stats.nsmallest(3, 'ROI')[['Seed', 'ROI']]
    for _, row in worst_roi.iterrows():
        print(f"    Seed {row['Seed']:>2}: {row['ROI']:>+6.1%} ROI")
    
    print()
    print("  Our model identifies WHICH teams in these seeds are actually good:")
    if len(anchors) > 0:
        for _, row in anchors.head(3).iterrows():
            print(f"    {row['Team']:<15} (Seed {row['Seed']:>2}): "
                  f"+{row['Points_Advantage']:.1f} advantage over avg {int(row['Seed'])}-seed")
    print()
    print("=" * 80)


def save_outputs(team_valuations, seed_stats, winners_df):
    """Save output files."""
    print("  OUTPUTS")
    print("  " + "-" * 76)
    
    # Main bid guidance
    output = team_valuations.sort_values('E_Points_Blended', ascending=False)
    output.to_csv(OUTPUT_DIR / 'team_valuations_2026.csv', index=False)
    print(f"  ✓ team_valuations_2026.csv")
    
    # Historical analysis
    seed_stats.to_csv(OUTPUT_DIR / 'historical_seed_performance.csv', index=False)
    print(f"  ✓ historical_seed_performance.csv")
    
    winners_df.to_csv(OUTPUT_DIR / 'historical_winners.csv', index=False)
    print(f"  ✓ historical_winners.csv")
    
    print()
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("           L4.02 — CALCUTTA BID GUIDANCE OPTIMIZER")
    print("=" * 80)
    print()
    
    # Load data
    print("STEP 1: Load Data")
    print("-" * 80)
    auction_df = load_auction_history()
    predictions_2026 = load_2026_predictions()
    print()
    
    # Historical analysis
    print("STEP 2: Historical Analysis")
    print("-" * 80)
    winners_df = analyze_winners(auction_df)
    seed_stats = analyze_seed_performance(auction_df)
    pot_info = estimate_pot_structure(auction_df)
    print()
    
    # Calculate bid guidance
    print("STEP 3: Calculate 2026 Bid Guidance")
    print("-" * 80)
    team_valuations = calculate_bid_guidance(
        predictions_2026, seed_stats, winners_df, pot_info
    )
    print()
    
    # Print summary
    print_summary(team_valuations, seed_stats, winners_df, pot_info)
    
    # Save outputs
    save_outputs(team_valuations, seed_stats, winners_df)


if __name__ == "__main__":
    main()
