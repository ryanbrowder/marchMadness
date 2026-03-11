"""
L4.02 - Calcutta Bid Guidance Optimizer
March Madness Computron | Ryan Browder | 2026

PHILOSOPHY:
  The market prices teams by seed. We price teams by expected points.
  That gap is the edge.

  Fair_Bid     = (E_Points / TARGET_POINTS) × $100   ← what the team is worth to you
  Market_Price = historical median bid for that seed   ← what the room pays
  Value_Gap    = Fair_Bid - Market_Price   ← your edge (positive = buy, negative = fade)

  TARGET_POINTS is derived from historical chalk/normal/chaos year classification:
    Chalk years  avg winner: ~69 pts  (top seeds advance, no upset bonus explosions)
    Normal years avg winner: ~83 pts
    Chaos years  avg winner: ~107 pts
  2026 is projected 80% chalk → TARGET_POINTS = chalk year median = 70.

  Walk-away rule: never bid above Fair_Bid, regardless of phase.
  Dynamic ceiling: (E_Points / Points_Still_Needed) × Budget_Remaining
    → tightens automatically as your portfolio fills

INPUTS:
  - outputs/01_tournament_simulator/round_probabilities.csv
  - data/auction_history.csv
  - data/calcuttaValueHistory.csv

OUTPUTS:
  - outputs/02_calcutta_optimizer/team_bidders_2026.csv
  - outputs/02_calcutta_optimizer/team_valuations_2026.csv
  - outputs/02_calcutta_optimizer/historical_seed_performance.csv
  - outputs/02_calcutta_optimizer/historical_winners.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
AUCTION_HISTORY      = DATA_DIR / "auction_history.csv"
CALCUTTA_VALUE_HIST  = DATA_DIR / "calcuttaValueHistory.csv"
ROUND_PROBS_2026     = Path("outputs/01_tournament_simulator/round_probabilities.csv")

OUTPUT_DIR = Path("outputs/02_calcutta_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Auction rules
ENTRY_FEE   = 20
PAYOUT_1ST  = 0.70
PAYOUT_2ND  = 0.30
BUDGET      = 100

# Tournament type classification — used to select the right TARGET_POINTS
# Chalk:  top seeds dominate, few upsets, low winning scores
# Chaos:  multi-round upsets, deep cinderella runs, high winning scores
# Normal: mixed
TOURNAMENT_TYPES = {
    2006: 'normal', 2007: 'chalk',  2008: 'chalk',  2009: 'chalk',
    2010: 'normal', 2011: 'chaos',  2012: 'chalk',  2013: 'normal',
    2014: 'chaos',  2015: 'chalk',  2016: 'normal', 2017: 'chalk',
    2018: 'chaos',  2019: 'chalk',  2021: 'chalk',  2022: 'chaos',
    2023: 'normal', 2024: 'chaos',  2025: 'chalk',
}

# 2026 projection: 80% chalk
TOURNAMENT_TYPE_2026 = 'chalk'

# Chalk-year market prices — median bid by seed across chalk years in THIS room.
# These differ from all-years medians: room bids $47 on 1-seeds in chalk years
# (vs $45 all-years), $10 on 6-seeds (vs $9), etc.
# Used instead of all-years median when TOURNAMENT_TYPE_2026 == 'chalk'.
# Source: auction_history.csv, chalk years only (2015/17/19/21/25), n=20 per seed.
CHALK_YEAR_MARKET_PRICES = {
    1: 47,   # chalk n=20, all-yr=$45
    2: 30,   # chalk n=20, all-yr=$30
    3: 19,   # chalk n=20, all-yr=$18
    4: 12,   # chalk n=20, all-yr=$14
    5: 11,   # chalk n=20, all-yr=$11
    6: 10,   # chalk n=19, all-yr=$9
    7:  7,   # chalk n=19, all-yr=$7
    8:  7,   # chalk n=19, all-yr=$6
    9:  5,   # chalk n=19, all-yr=$5
    10:  7,  # chalk n=19, all-yr=$7
    11:  6,  # chalk n=19, all-yr=$6
    12:  6,  # chalk n=20, all-yr=$6
    13:  4,  # chalk n=20, all-yr=$3
    14:  3,  # chalk n=19, all-yr=$2
    15:  2,  # chalk n=20, all-yr=$2
}


def compute_target_points():
    """
    Derive TARGET_POINTS from calcuttaValueHistory.csv using chalk year winning scores.
    Returns the median winning points across chalk years.
    Falls back to 70 if file not found.
    """
    try:
        df = pd.read_csv(CALCUTTA_VALUE_HIST)
        # First column is year (unnamed in CSV)
        df.columns = ['Year'] + list(df.columns[1:])
        df = df[pd.to_numeric(df['Year'], errors='coerce').notna()].copy()
        df['Year'] = df['Year'].astype(int)
        df['Winning Pts'] = pd.to_numeric(df['Winning Pts'], errors='coerce')
        df = df.dropna(subset=['Winning Pts'])
        df['Type'] = df['Year'].map(TOURNAMENT_TYPES)
        chalk = df[df['Type'] == TOURNAMENT_TYPE_2026]['Winning Pts']
        target = int(chalk.median())
        print(f"  ✓ TARGET_POINTS derived from {len(chalk)} chalk years: "
              f"median={target}, mean={chalk.mean():.1f}, range={chalk.min():.0f}-{chalk.max():.0f}")
        return target
    except Exception as e:
        print(f"  ⚠ Could not load calcuttaValueHistory.csv ({e}). Using fallback TARGET_POINTS=70.")
        return 70


# Compute at module load — used throughout
TARGET_POINTS = compute_target_points()
WIN_THRESHOLD = TARGET_POINTS  # In chalk year, target IS the threshold


# ============================================================================
# LOAD DATA
# ============================================================================

def load_auction_history():
    """Load historical auction data (2013-2025)."""
    print("Loading historical auction data...")
    df = pd.read_csv(AUCTION_HISTORY)
    # Drop invalid seeds
    df = df[(df['Seed'] > 0) & (df['Seed'] != 16)].copy()
    print(f"  ✓ Loaded {len(df)} auction records ({df['Year'].min()}-{df['Year'].max()})")
    return df


def load_2026_predictions():
    """Load 2026 team predictions from L4.01."""
    print("Loading 2026 model predictions...")
    df = pd.read_csv(ROUND_PROBS_2026)

    if 'P_FF' in df.columns:
        df = df.rename(columns={'P_FF': 'P_F4'})

    df = df[df['Seed'] != 16].copy()
    df = calculate_expected_points(df)

    print(f"  ✓ Loaded predictions for {len(df)} teams (16-seeds excluded)")
    return df


def calculate_expected_points(df):
    """
    Calculate expected points from round probabilities with upset bonuses.
    Also tracks base points and upset bonus separately so chalk risk can be assessed.
    In a chalk-heavy year, teams with high upset_bonus dependency are riskier.
    """
    SCORING = {'R64': 2, 'R32': 4, 'S16': 6, 'E8': 8, 'F4': 10, 'Championship': 12}

    R64_MATCHUPS = {1:16, 2:15, 3:14, 4:13, 5:12, 6:11, 7:10, 8:9,
                    16:1, 15:2, 14:3, 13:4, 12:5, 11:6, 10:7, 9:8}

    EXPECTED_OPPONENT = {'R32': 8.0, 'S16': 4.5, 'E8': 3.0, 'F4': 2.0}

    expected_points = []
    base_points_list = []
    upset_bonus_list = []

    for _, row in df.iterrows():
        seed = int(row['Seed'])

        base = 0
        base += row.get('P_R64', 1.0) * SCORING['R64']
        base += row.get('P_R32', 0)   * SCORING['R32']
        base += row.get('P_S16', 0)   * SCORING['S16']
        base += row.get('P_E8', 0)    * SCORING['E8']
        base += row.get('P_F4', 0)    * SCORING['F4']
        base += row.get('P_Championship', 0) * SCORING['Championship']

        upset_bonus = 0.0

        r64_opp = R64_MATCHUPS.get(seed, seed)
        if seed > r64_opp:
            upset_bonus += row.get('P_R32', 0) * (seed - r64_opp)
        if seed > EXPECTED_OPPONENT['R32']:
            upset_bonus += row.get('P_S16', 0) * (seed - EXPECTED_OPPONENT['R32'])
        if seed > EXPECTED_OPPONENT['S16']:
            upset_bonus += row.get('P_E8', 0) * (seed - EXPECTED_OPPONENT['S16'])
        if seed > EXPECTED_OPPONENT['E8']:
            upset_bonus += row.get('P_F4', 0) * (seed - EXPECTED_OPPONENT['E8'])
        if seed > EXPECTED_OPPONENT['F4']:
            upset_bonus += row.get('P_Championship', 0) * (seed - EXPECTED_OPPONENT['F4'])

        expected_points.append(round(base + upset_bonus, 2))
        base_points_list.append(round(base, 2))
        upset_bonus_list.append(round(upset_bonus, 2))

    df['E_Points_Blended'] = expected_points
    df['E_Points_Base']    = base_points_list    # Points from round advancement only
    df['E_Points_Upset']   = upset_bonus_list    # Points from upset bonuses only
    return df


# ============================================================================
# HISTORICAL ANALYSIS
# ============================================================================

def analyze_winners(auction_df):
    """Analyze historical winners, split by tournament type."""
    print("Analyzing historical winners...")

    player_totals = auction_df.groupby(['Year', 'Player']).agg(
        Points=('Points', 'sum'),
        Spend=('Bid', 'sum'),
        N_Teams=('Team', 'count')
    ).reset_index()

    winners = []
    for year in sorted(player_totals['Year'].unique()):
        yr = player_totals[player_totals['Year'] == year].sort_values('Points', ascending=False)
        if len(yr) >= 2:
            w, r = yr.iloc[0], yr.iloc[1]
            winners.append({
                'Year':      year,
                'Type':      TOURNAMENT_TYPES.get(year, 'unknown'),
                'Winner':    w['Player'],
                'W_Points':  w['Points'],
                'W_Spend':   w['Spend'],
                'W_Teams':   w['N_Teams'],
                'R_Points':  r['Points'],
                'R_Spend':   r['Spend'],
            })

    winners_df = pd.DataFrame(winners)

    chalk  = winners_df[winners_df['Type'] == 'chalk']
    normal = winners_df[winners_df['Type'] == 'normal']
    chaos  = winners_df[winners_df['Type'] == 'chaos']

    print(f"  ✓ Chalk  years ({len(chalk)}):  avg winner {chalk['W_Points'].mean():.1f} pts, "
          f"${chalk['W_Spend'].mean():.0f} spend")
    print(f"  ✓ Normal years ({len(normal)}): avg winner {normal['W_Points'].mean():.1f} pts, "
          f"${normal['W_Spend'].mean():.0f} spend")
    print(f"  ✓ Chaos  years ({len(chaos)}):  avg winner {chaos['W_Points'].mean():.1f} pts, "
          f"${chaos['W_Spend'].mean():.0f} spend")
    return winners_df


def analyze_seed_performance(auction_df):
    """
    Historical bid and points by seed.
    Market_Price = median bid for that seed. This is what the room will pay.
    """
    print("Analyzing seed performance...")

    valid_df = auction_df[auction_df['Bid'] > 0].copy()

    seed_stats = valid_df.groupby('Seed').agg(
        Bid_mean=('Bid', 'mean'),
        Bid_median=('Bid', 'median'),
        Bid_std=('Bid', 'std'),
        Bid_count=('Bid', 'count'),
        Bid_p25=('Bid', lambda x: x.quantile(0.25)),
        Bid_p75=('Bid', lambda x: x.quantile(0.75)),
        Points_mean=('Points', 'mean'),
        Points_median=('Points', 'median'),
    ).round(2).reset_index()

    # Pts_Per_Dollar: points scored per dollar spent historically.
    # Pts_Per_Dollar > 1.0 = scored more points than dollars paid (efficient)
    # Pts_Per_Dollar < 1.0 = paid more dollars than points scored (inefficient)
    # This is the correct efficiency signal — NOT a percentage ROI.
    # A 1-seed at $45 averaging 17 pts = 0.38 pts/$. A 11-seed at $6 averaging
    # 8 pts = 1.37 pts/$. Market systematically overvalues high seeds.
    seed_stats['Pts_Per_Dollar'] = (
        seed_stats['Points_mean'] / seed_stats['Bid_mean']
    ).round(3)

    print(f"  ✓ Most efficient seed (pts/$): {seed_stats.loc[seed_stats['Pts_Per_Dollar'].idxmax(), 'Seed']:.0f}")
    print(f"  ✓ Least efficient seed (pts/$): {seed_stats.loc[seed_stats['Pts_Per_Dollar'].idxmin(), 'Seed']:.0f}")
    return seed_stats


def estimate_pot_structure(auction_df):
    """Estimate pot structure from historical data."""
    players_per_year = auction_df.groupby('Year')['Player'].nunique()
    avg_players = players_per_year.mean()
    total_pot = avg_players * ENTRY_FEE
    return {
        'avg_players': avg_players,
        'total_pot': total_pot,
        'first_place': total_pot * PAYOUT_1ST,
        'second_place': total_pot * PAYOUT_2ND
    }


# ============================================================================
# CORE BID GUIDANCE
# ============================================================================

def calculate_bid_guidance(predictions_df, seed_stats, winners_df, pot_info):
    """
    Build bid guidance grounded in expected points, not seed.

    Key outputs per team:
      Fair_Bid     = (E_Points / TARGET_POINTS) × BUDGET
                     This is the walk-away price. Never bid above it.
      Market_Price = historical median bid for this seed
                     This is what the room will likely pay.
      Value_Gap    = Fair_Bid - Market_Price
                     Positive → you can win the team at or under your ceiling.
                     Negative → market overpays relative to model; let them.
      Action       = BUY / COMPETE / FADE based on gap magnitude
    """
    print("Building bid guidance (E_Points-based pricing)...")

    df = predictions_df.copy()

    # ── Merge historical seed data ──────────────────────────────────────────
    df = df.merge(
        seed_stats[['Seed', 'Bid_mean', 'Bid_median', 'Bid_p25', 'Bid_p75',
                    'Points_mean', 'Points_median', 'Pts_Per_Dollar']],
        on='Seed', how='left'
    )

    # ── Core pricing columns ────────────────────────────────────────────────

    # Fair_Bid: what this team is worth to you based on model output (whole dollars)
    df['Fair_Bid'] = ((df['E_Points_Blended'] / TARGET_POINTS) * BUDGET).round(0).astype(int)

    # Market_Price: what this room will pay in 2026.
    # In chalk years, use chalk-year medians — this room bids $47 on 1-seeds
    # in chalk years, not $45. Using all-years medians understates competition
    # for top seeds and overstates it for mid/low seeds.
    if TOURNAMENT_TYPE_2026 == 'chalk':
        df['Market_Price'] = df['Seed'].map(CHALK_YEAR_MARKET_PRICES).fillna(
            df['Bid_median']).round(0).astype(int)
    else:
        df['Market_Price'] = df['Bid_median'].round(0).astype(int)

    # Value_Gap: your edge. Positive = you can compete at or below your ceiling.
    df['Value_Gap'] = (df['Fair_Bid'] - df['Market_Price']).astype(int)

    # Value_Gap_Pct: gap as % of market price (normalizes across price tiers)
    df['Value_Gap_Pct'] = (df['Value_Gap'] / df['Market_Price']).round(3)

    # Points_Advantage: model vs. historical central tendency for this seed.
    # Seeds 1-8: use mean — distributions are relatively symmetric.
    # Seeds 9-15: use MEDIAN — these are heavily right-skewed by rare Cinderella
    # runs (e.g., seed-11 mean=8.2, median=4.0 — UCLA 45pts, NC State 41pts drag
    # the mean far above what a typical 9-15 seed scores). Using mean here would
    # penalize legitimately good teams for failing to match an inflated benchmark.
    df['Points_Benchmark'] = df.apply(
        lambda r: r['Points_median'] if r['Seed'] >= 9 else r['Points_mean'], axis=1
    )
    df['Points_Advantage'] = (df['E_Points_Blended'] - df['Points_Benchmark']).round(1)

    # Points per market dollar (efficiency for low-cost teams)
    df['Pts_Per_Market_Dollar'] = (df['E_Points_Blended'] / df['Market_Price']).round(3)

    # ── Chalk Risk ───────────────────────────────────────────────────────────
    # 2026 is projected 80% chalk. Teams whose E_Points depend heavily on upset
    # bonuses are riskier this year — those bonuses may not materialize.
    # Upset_Pct = upset bonus / total E_Points
    # HIGH (>30%): meaningful downside in chalk year
    # MEDIUM (15-30%): some risk, model still likes them
    # LOW (<15%): base points driven by round advancement, chalk-safe
    df['Upset_Pct'] = (df['E_Points_Upset'] / df['E_Points_Blended'].clip(lower=0.01)).round(3)

    def assign_chalk_risk(row):
        if row['Upset_Pct'] > 0.30:
            return 'HIGH'
        elif row['Upset_Pct'] > 0.15:
            return 'MEDIUM'
        else:
            return 'LOW'

    df['Chalk_Risk'] = df.apply(assign_chalk_risk, axis=1)

    # ── Action signal ────────────────────────────────────────────────────────
    # BUY:           Genuine edge — gap >= $3 (real dollar upside)
    #                OR gap >= $2 AND gap_pct >= 15% (small-price teams)
    #                Both conditions anchor BUY to meaningful edges only.
    #                Avoids cheap teams with tiny absolute gaps inflating pct.
    # COMPETE:       gap_pct >= -15% — within fighting range of market
    # DISCOUNT_ONLY: gap_pct < -15% AND E_Points > 10 — market outbids you
    #                but team is strong enough to track
    # FADE:          gap_pct < -15% AND E_Points <= 10, OR Tier=FADE
    # PASS:          E_Points < 6

    def assign_tier(row):
        pts = row['E_Points_Blended']
        adv = row['Points_Advantage']
        eff = row['Pts_Per_Market_Dollar']   # model pts / market $

        if pd.isna(adv):
            return 'PASS'

        # Minimum efficiency gate: if the market price is so inflated that you'd
        # get less than 0.35 model pts per dollar spent, tier is FADE regardless
        # of E_Points or advantage. Michigan St. at $30 market / 11.4 pts = 0.38 —
        # barely clears this, so we also add an absolute gap floor: if gap_pct < -40%
        # the market will reliably beat your ceiling by a wide margin.
        gap = row['Value_Gap']
        mkt = row['Market_Price']
        gap_pct = gap / mkt if mkt > 0 else 0
        if gap_pct < -0.40:
            return 'FADE'

        if (pts >= 20 and adv >= 5) or (pts >= 15 and adv >= 8):
            return 'ANCHOR'
        elif (pts >= 12 and adv >= 4) or (pts >= 10 and adv >= 6):
            return 'FILL'
        elif (pts >= 8 and eff >= 1.1) or (pts >= 8 and adv >= 3):
            return 'VALUE'
        elif pts >= 6 and adv >= 2:
            return 'OPPORTUNISTIC'
        else:
            return 'FADE'

    df['Tier'] = df.apply(assign_tier, axis=1)

    # ── Action signal ────────────────────────────────────────────────────────
    # Single metric throughout: gap% = Value_Gap / Market_Price
    #
    # BUY:           gap% >= +15%   Market meaningfully underprices you.
    #                Genuine edge — expect to win at or near market.
    #                Iowa St. (+32%) is a BUY. Houston (+10%) is not.
    #
    # COMPETE:       -15% <= gap% < +15%   Fair fight.
    #                You can win this team but expect resistance.
    #                Hard ceiling is Fair_Bid — never exceed it.
    #                Houston (+10%) lives here — best available COMPETE anchor.
    #
    # DISCOUNT_ONLY: gap% < -15%,  E_Pts > 10
    #                Strong team but market overprices by seed.
    #                You will lose at market. Only bid if room leaves a gift.
    #
    # FADE:          gap% < -15%,  E_Pts <= 10  OR  Tier == FADE
    #                Bad value and/or weak team. Never bid.
    #
    # PASS:          E_Pts < 6 — not worth a roster slot at any price
    #
    # Why pure gap%: mixing dollar thresholds with % thresholds creates
    # inconsistency across price tiers. A $3 gap on a $6 team (+50%) is a
    # screaming BUY. A $3 gap on a $30 team (+10%) is a moderate edge.
    # Same rule, same unit, clean interpretation.

    def assign_action(row):
        pts     = row['E_Points_Blended']
        tier    = row['Tier']
        mkt     = row['Market_Price']
        gap_pct = row['Value_Gap'] / mkt if mkt > 0 else 0

        if pts < 6:
            return 'PASS'
        # Check gap% BEFORE tier — gap% is the primary signal.
        # tier==FADE is set by pts/adv/eff thresholds in assign_tier, not by market value.
        # A team with gap% >= -15% should never be FADE: the market is pricing it
        # at or below fair value regardless of tier. Virginia at 0% gap = COMPETE.
        # Saint Mary's at +43% gap = BUY. Tier can't override that.
        if gap_pct >= 0.15:
            return 'BUY'
        if gap_pct >= -0.15:
            return 'COMPETE'
        # Only below -15% gap does tier matter
        if tier == 'FADE':
            return 'FADE'
        if pts > 10:
            return 'DISCOUNT_ONLY'
        return 'FADE'

    df['Action'] = df.apply(assign_action, axis=1)

    print(f"  ✓ BUY targets:      {len(df[df['Action']=='BUY'])} teams")
    print(f"  ✓ COMPETE:          {len(df[df['Action']=='COMPETE'])} teams")
    print(f"  ✓ DISCOUNT_ONLY:    {len(df[df['Action']=='DISCOUNT_ONLY'])} teams")
    print(f"  ✓ FADE:             {len(df[df['Action']=='FADE'])} teams")
    print(f"  ✓ PASS:             {len(df[df['Action']=='PASS'])} teams")

    return df


# ============================================================================
# RANKED BID LIST BUILDER
# ============================================================================

def build_ranked_bid_list(df):
    """
    Replaces static portfolio paths with a ranked bid list + pivot scenarios.

    Expert auction strategy: you don't control sequence, so you need:
      1. A pre-auction anchor decision — top 2 ANCHOR-tier teams, derived dynamically
      2. A ranked fill list by E_Points value with dual budget columns
         (one per anchor choice) so you always know your remaining runway
      3. Contested pair flags — teams you can win ONE of, not both
      4. Explicit pivot scenarios for the most likely plan breaks

    Anchor selection is fully dynamic — no team names hardcoded. The top 2
    ANCHOR-tier BUY/COMPETE teams by E_Points become the two options:
      Anchor 1: highest E_Pts ceiling (typically a COMPETE team)
      Anchor 2: best gap% among remaining anchors (the value option)

    Returns:
      anchors  — ordered dict of anchor options keyed by team name
      fills    — list of non-anchor BUY/COMPETE targets, ranked by E_Points
      pivots   — dict of named pivot scenarios with team lists + budget math
    """

    active = df[
        df['Action'].isin(['BUY', 'COMPETE']) &
        (df['Tier'] != 'FADE')
    ].copy()

    # ── Anchor options: top 2 ANCHOR-tier teams, derived from data ───────────
    # Anchor 1: highest E_Points — best ceiling, likely a COMPETE
    # Anchor 2: best gap% among remaining anchors — the value play
    # Both must be ANCHOR tier and BUY or COMPETE (already filtered above).
    anchor_pool = active[active['Tier'] == 'ANCHOR'].copy()
    anchor_pool['Gap_Pct'] = anchor_pool['Value_Gap'] / anchor_pool['Market_Price']

    anchors = {}
    used_anchor_names = set()

    if len(anchor_pool) >= 1:
        # Anchor 1: highest E_Points among winnable anchors (gap% >= -10%).
        # A team you can't win at market is not a useful anchor option, even if
        # it has the highest ceiling. Duke at -13% is technically COMPETE but
        # nearly impossible to win — Houston at +10% is the actionable choice.
        # Falls back to highest E_Pts if no anchor has gap% >= -10%.
        winnable = anchor_pool[anchor_pool['Gap_Pct'] >= -0.10]
        a1_pool  = winnable if len(winnable) > 0 else anchor_pool
        a1 = a1_pool.sort_values('E_Points_Blended', ascending=False).iloc[0]
        used_anchor_names.add(a1['Team'])
        gap_pct_a1 = a1['Value_Gap'] / a1['Market_Price'] if a1['Market_Price'] > 0 else 0
        anchors[a1['Team']] = {
            'Team':         a1['Team'],
            'Seed':         int(a1['Seed']),
            'E_Pts':        a1['E_Points_Blended'],
            'Fair':         int(a1['Fair_Bid']),
            'Mkt':          int(a1['Market_Price']),
            'Gap_Pct':      gap_pct_a1,
            'Action':       a1['Action'],
            'Chalk_Risk':   a1['Chalk_Risk'],
            'Budget_After': BUDGET - int(a1['Market_Price']),
            'Note':         f'Highest winnable E_Pts ceiling ({a1["E_Points_Blended"]:.1f} pts). '
                            f'{"Genuine edge — expect to win near market." if gap_pct_a1 >= 0.15 else "Fair fight up to Fair$."}'
        }

    if len(anchor_pool) >= 2:
        # Anchor 2: best gap% among remaining anchors
        remaining = anchor_pool[~anchor_pool['Team'].isin(used_anchor_names)]
        a2 = remaining.sort_values('Gap_Pct', ascending=False).iloc[0]
        used_anchor_names.add(a2['Team'])
        gap_pct_a2 = a2['Value_Gap'] / a2['Market_Price'] if a2['Market_Price'] > 0 else 0
        a2_note_gap = f'Best gap% among anchors ({gap_pct_a2:+.0%}). ' if gap_pct_a2 >= 0 else f'Closest to fair value among anchors ({gap_pct_a2:+.0%}). '
        a1_mkt_for_compare = int(a1['Market_Price']) if len(anchor_pool) >= 1 else 0
        a2_note_budget = f'More budget for fills (${BUDGET - int(a2["Market_Price"])} remaining).' if int(a2['Market_Price']) < a1_mkt_for_compare else ''
        anchors[a2['Team']] = {
            'Team':         a2['Team'],
            'Seed':         int(a2['Seed']),
            'E_Pts':        a2['E_Points_Blended'],
            'Fair':         int(a2['Fair_Bid']),
            'Mkt':          int(a2['Market_Price']),
            'Gap_Pct':      gap_pct_a2,
            'Action':       a2['Action'],
            'Chalk_Risk':   a2['Chalk_Risk'],
            'Budget_After': BUDGET - int(a2['Market_Price']),
            'Note':         (a2_note_gap +
                            f'{"Genuine edge — expect to win near market." if gap_pct_a2 >= 0.15 else "Fair fight up to Fair$."} ' +
                            a2_note_budget).strip()
        }

    anchor_name_1 = list(anchors.keys())[0] if len(anchors) >= 1 else None
    anchor_name_2 = list(anchors.keys())[1] if len(anchors) >= 2 else None

    # ── Fill pool: all non-ANCHOR-tier BUY/COMPETE, sorted by E_Points ────────
    # Exclude entire ANCHOR tier — these teams belong in Step 1, not the fill list.
    fills_df = active[
        active['Tier'] != 'ANCHOR'
    ].sort_values(['E_Points_Blended', 'Chalk_Risk'], ascending=[False, True])

    fills = []
    for _, row in fills_df.iterrows():
        fills.append({
            'Team':       row['Team'],
            'Seed':       int(row['Seed']),
            'E_Pts':      row['E_Points_Blended'],
            'Fair':       int(row['Fair_Bid']),
            'Mkt':        int(row['Market_Price']),
            'Gap_Pct':    row['Value_Gap'] / row['Market_Price'] if row['Market_Price'] > 0 else 0,
            'Action':     row['Action'],
            'Tier':       row['Tier'],
            'Chalk_Risk': row['Chalk_Risk'],
            'Contested':  False,
            'Contested_With': None,
        })

    # ── Flag contested clusters ─────────────────────────────────────────────
    # Teams are contested if: same seed AND market prices within $1 of each other.
    # Handle N-way clusters (e.g. Kansas/Texas Tech/Alabama all seed 4 at $12)
    # by collecting all cluster members and listing all rivals, not just adjacent.
    # In a random-order auction with 7-8 bidders, the room will split on these.
    # Plan to win ONE of the cluster. Budget for that one, not all.
    from collections import defaultdict
    seed_mkt_groups = defaultdict(list)
    for idx, f in enumerate(fills):
        seed_mkt_groups[(f['Seed'], f['Mkt'])].append(idx)

    # Expand: merge groups within $1 of each other on same seed
    merged = []
    visited = set()
    for (seed, mkt), idxs in seed_mkt_groups.items():
        if (seed, mkt) in visited:
            continue
        cluster_idxs = set(idxs)
        for (seed2, mkt2), idxs2 in seed_mkt_groups.items():
            if seed2 == seed and abs(mkt2 - mkt) <= 1:
                cluster_idxs.update(idxs2)
                visited.add((seed2, mkt2))
        visited.add((seed, mkt))
        if len(cluster_idxs) > 1:
            merged.append(cluster_idxs)

    for cluster_idxs in merged:
        for i in cluster_idxs:
            # Contested rivals must be within 3 E_Points — true strategic substitutes.
            # Same seed/price but very different E_Pts means they're not interchangeable:
            # Kansas (13.0 pts) vs Virginia (8.9 pts) = 4.1 gap → not contested.
            # Kansas (13.0 pts) vs Texas Tech (11.5 pts) = 1.5 gap → contested ✓
            rivals = [
                fills[j]['Team'] for j in cluster_idxs
                if j != i and abs(fills[j]['E_Pts'] - fills[i]['E_Pts']) <= 3.0
            ]
            if rivals:
                fills[i]['Contested']      = True
                fills[i]['Contested_With'] = ' / '.join(rivals)

    # ── Compute cumulative budget remaining under each anchor ────────────────
    cum_a1 = anchors[anchor_name_1]['Mkt'] if anchor_name_1 else 0
    cum_a2 = anchors[anchor_name_2]['Mkt'] if anchor_name_2 else 0
    for f in fills:
        cum_a1 += f['Mkt']
        cum_a2 += f['Mkt']
        f['Rem_A1'] = BUDGET - cum_a1
        f['Rem_A2'] = BUDGET - cum_a2

    # ── Pivot scenarios ──────────────────────────────────────────────────────
    def run_greedy(budget, exclude_teams, skip_high_risk=False):
        portfolio, rem = [], budget
        for f in fills:
            if f['Team'] in exclude_teams:
                continue
            # Pivot 2 uses skip_high_risk=True — after winning an anchor you're
            # already at market, so adding HIGH chalk-risk teams compounds exposure
            if skip_high_risk and f.get('Chalk_Risk') == 'HIGH':
                continue
            if rem >= f['Mkt']:
                portfolio.append(f)
                rem -= f['Mkt']
        return portfolio, rem

    pivots = {}

    # Pivot 1: Lost anchor entirely — include all chalk risks (you need volume)
    p1_teams, p1_rem = run_greedy(BUDGET, set())
    pivots['lost_anchor'] = {
        'label': 'LOST ANCHOR BID',
        'desc':  'Full $100 available. Load fills by E_Points order.',
        'teams': p1_teams[:8],
        'rem':   p1_rem,
        'pts':   sum(f['E_Pts'] for f in p1_teams[:8]),
    }

    # Pivot 2: Anchor 1 won, top contested fills lost — skip HIGH chalk risk fills
    top_contested = [f['Team'] for f in fills if f['Contested']][:2]
    a1_mkt = anchors[anchor_name_1]['Mkt'] if anchor_name_1 else 0
    a1_pts = anchors[anchor_name_1]['E_Pts'] if anchor_name_1 else 0
    p2_teams, p2_rem = run_greedy(BUDGET - a1_mkt, set(top_contested), skip_high_risk=True)
    contested_desc = ' & '.join(top_contested) if top_contested else 'top fills'
    pivots['anchor1_contested_lost'] = {
        'label':       f'{anchor_name_1 or "ANCHOR 1"} WON + {contested_desc} LOST',
        'desc':        f'{anchor_name_1} won (${a1_mkt}). {contested_desc} lost to other bidders. Fill from next tier.',
        'teams':       p2_teams[:7],
        'rem':         p2_rem,
        'pts':         a1_pts + sum(f['E_Pts'] for f in p2_teams[:7]),
        'anchor_cost': a1_mkt,
    }

    # Pivot 3: Budget low — cheapest BUY targets only
    cheap_buys = [f for f in fills if f['Mkt'] <= 12 and f['Action'] == 'BUY']
    pivots['budget_low'] = {
        'label': 'BUDGET LOW (< $15 remaining)',
        'desc':  'Only consider these BUY targets — market price ≤ $12.',
        'teams': cheap_buys[:6],
        'rem':   None,
        'pts':   None,
    }

    return anchors, anchor_name_1, anchor_name_2, fills, pivots

# ============================================================================
# OUTPUT / CHEAT SHEET
# ============================================================================

def print_summary(df, seed_stats, winners_df, pot_info):
    """Print the auction day cheat sheet."""

    # Compute chalk-year winner stats for the header benchmark
    chalk_winners = winners_df[winners_df['Type'] == 'chalk'] if 'Type' in winners_df.columns else winners_df
    all_winners   = winners_df

    print()
    print("=" * 80)
    print("                    CALCUTTA AUCTION CHEAT SHEET  —  2026")
    print("=" * 80)
    print()
    print(f"  Budget: ${BUDGET} | Target: {WIN_THRESHOLD}+ pts to win | Typical portfolio: 7-8 teams")
    if len(chalk_winners) > 0:
        print(f"  Chalk-year winners: avg {chalk_winners['W_Points'].mean():.0f} pts, "
              f"${chalk_winners['W_Spend'].mean():.0f} spend, "
              f"{chalk_winners['W_Teams'].mean():.0f} teams  "
              f"(all-years avg: {all_winners['W_Points'].mean():.0f} pts)")
    else:
        print(f"  Avg winner: {all_winners['W_Points'].mean():.0f} pts spending ${all_winners['W_Spend'].mean():.0f}")
    print()

    # ── 2026 TOURNAMENT CONTEXT ──────────────────────────────────────────────
    print("  2026 TOURNAMENT OUTLOOK  (informs strategy selection)")
    print("  " + "-" * 76)
    print("  Type:    80% CHALK / 15% NORMAL / 5% CHAOS")
    print("  Impact:  Top seeds more likely to score. Upset bonuses less likely to")
    print("           materialize. Teams with HIGH Chalk_Risk are riskier this year.")
    print()
    print("  Market prices use chalk-year medians (what this room pays in chalk years).")
    print("  1-seeds go for $47 median in chalk years — not $45 all-years avg.")
    print("  Chalk-year winners consistently win 1 deep-scoring anchor + cheap fills.")
    print("  2025: Florida ($53) + Drake ($11) + 5 others. 2021: Gonzaga ($57) + Oregon St. ($6).")
    print("  Strategy: win one anchor that can score 25-30+ pts. Fill with BUY value.")
    print()
    print("  Chalk_Risk flag: % of a team's E_Points from upset bonuses.")
    print("    LOW  (<15% upset-dependent) → safe in chalk year")
    print("    MED  (15-30%)               → some downside if chalk holds")
    print("    HIGH (>30%)                 → significant points at risk in chalk year")
    print()

    # ── PRICING MODEL ────────────────────────────────────────────────────────
    print("  HOW BIDS ARE CALCULATED")
    print("  " + "-" * 76)
    print(f"  Fair_Bid     = (E_Points / {TARGET_POINTS}) × $100   ← your walk-away price (NEVER exceed)")
    print(f"  TARGET_POINTS = {TARGET_POINTS}  (chalk year median — derived from historical data)")
    if len(chalk_winners) > 0:
        print(f"  Chalk-yr winner avg: {chalk_winners['W_Points'].mean():.0f} pts, "
              f"${chalk_winners['W_Spend'].mean():.0f} spend | Normal: 83 pts | Chaos: 107 pts")
    print(f"  Market_Price = chalk-year median bid for that seed (what this room pays in chalk years)")
    print(f"  Value_Gap    = Fair_Bid − Market_Price   ← your edge")
    print()
    print("  NEVER bid above Fair_Bid. Path Exp$ = Market_Price (what you need to WIN the team).")
    print("  Dynamic ceiling mid-auction: (Team_E_Pts / Pts_Still_Needed) × Budget_Left")
    # Build example from actual Anchor 1 so numbers match the live field
    _ex_anchors = df[(df['Tier'] == 'ANCHOR') & df['Action'].isin(['BUY', 'COMPETE'])].sort_values('E_Points_Blended', ascending=False)
    if len(_ex_anchors) > 0:
        _ex      = _ex_anchors.iloc[0]
        _ex_pts  = round(_ex['E_Points_Blended'], 1)
        _ex_need = round(TARGET_POINTS - _ex_pts)
        _ex_left = BUDGET - int(_ex['Market_Price'])
        _ex_ceil = round((_ex_pts / max(_ex_need, 1)) * _ex_left)
        print(f"  Example: {_ex['Team']} {_ex_pts} pts, you need {_ex_need} more, "
              f"${_ex_left} left → ({_ex_pts}/{_ex_need})×{_ex_left} = ${_ex_ceil}")
    else:
        print("  Example: Top team 20 pts, you need 50 more, $70 left → (20/50)×70 = $28")
    print()

    # ── ACTION KEY ───────────────────────────────────────────────────────────
    # Derive live examples from data — no hardcoded team names
    _anchor_pool = df[
        (df['Tier'] == 'ANCHOR') & df['Action'].isin(['BUY', 'COMPETE'])
    ].copy()
    _anchor_pool['Gap_Pct'] = _anchor_pool['Value_Gap'] / _anchor_pool['Market_Price']
    _buy_anchors  = _anchor_pool[_anchor_pool['Action'] == 'BUY'].sort_values('Gap_Pct', ascending=False)
    if len(_buy_anchors) > 0:
        _buy_example = _buy_anchors
    else:
        # No BUY anchors — fall back to best BUY fill.
        # Priority: FILL tier > VALUE > OPPORTUNISTIC, then LOW chalk risk, then gap%.
        # Avoids surfacing HIGH-risk Cinderella teams as the headline BUY example.
        _TIER_ORDER = {'FILL': 0, 'VALUE': 1, 'OPPORTUNISTIC': 2}
        _buy_fills_all = df[
            (df['Action'] == 'BUY') & (~df['Tier'].isin(['FADE']))
        ].copy()
        _buy_fills_all['Gap_Pct']   = _buy_fills_all['Value_Gap_Pct']
        _buy_fills_all['_TierRank'] = _buy_fills_all['Tier'].map(_TIER_ORDER).fillna(3)
        _buy_fills_all['_RiskRank'] = (_buy_fills_all['Chalk_Risk'] != 'LOW').astype(int)
        _buy_example = _buy_fills_all.sort_values(
            ['_TierRank', '_RiskRank', 'Gap_Pct'], ascending=[True, True, False]
        )
    # COMPETE anchor example: prefer positive gap% (teams you can actually win) over
    # highest E_Pts. A -13% COMPETE anchor is nearly impossible to win at market.
    _compete_pool    = _anchor_pool[_anchor_pool['Action'] == 'COMPETE'].copy()
    _compete_pos     = _compete_pool[_compete_pool['Gap_Pct'] >= 0].sort_values('E_Points_Blended', ascending=False)
    _compete_example = _compete_pos if len(_compete_pos) > 0 else _compete_pool.sort_values('E_Points_Blended', ascending=False)
    _buy_label   = "is BUY" if len(_buy_anchors) > 0 else "leads BUY fills"
    _buy_str     = (f"{_buy_example.iloc[0]['Team']} ({_buy_example.iloc[0]['Gap_Pct']:+.0%})"
                    if len(_buy_example) > 0 else "top BUY fill")
    _compete_str = (f"{_compete_example.iloc[0]['Team']} ({_compete_example.iloc[0]['Gap_Pct']:+.0%})"
                    if len(_compete_example) > 0 else "best COMPETE anchor")

    print("  ACTION GUIDE  (gap% = Value_Gap ÷ Market_Price — one metric throughout)")
    print("  " + "-" * 76)
    print("  BUY           gap% ≥ +15%.  Market meaningfully underprices you.")
    print(f"                Genuine edge — expect to win near market. {_buy_str} {_buy_label}.")
    print("  COMPETE       −15% ≤ gap% < +15%.  Fair fight.")
    print("                You can win this team but expect resistance.")
    print("                Hard ceiling is Fair_Bid — walk away above it.")
    print(f"                {_compete_str} is your best COMPETE anchor.")
    print("  DISCOUNT_ONLY gap% < −15%,  E_Pts > 10.  Strong team, market overprices by seed.")
    print("                You will lose at market. Only bid if room leaves a gift.")
    print("                Never plan around these teams.")
    print("  FADE          gap% < −15%,  E_Pts ≤ 10  OR  model flags weak.  Never bid.")
    print()

    # ── FULL TEAM TABLE ──────────────────────────────────────────────────────
    active = df[df['Action'].isin(['BUY', 'COMPETE', 'DISCOUNT_ONLY'])].sort_values(
        'E_Points_Blended', ascending=False)

    print("  TEAM PRICING TABLE  (excludes FADE/PASS)")
    print("  " + "-" * 90)
    print(f"  {'Team':<18} {'Sd':>3} {'E_Pts':>6} {'Base':>5} {'Fair$':>6} {'Mkt$':>5} "
          f"{'Gap%':>6} {'CRisk':>5}  {'Action':<14} {'Tier'}")
    print("  " + "-" * 90)

    for _, row in active.iterrows():
        mkt = row['Market_Price']
        gap_pct = row['Value_Gap'] / mkt if mkt > 0 else 0
        gap_str = f"+{gap_pct:.0%}" if gap_pct >= 0 else f"{gap_pct:.0%}"
        if row['Action'] == 'BUY':
            action_marker = "◀ BUY"
        elif row['Action'] == 'DISCOUNT_ONLY':
            action_marker = "▼ DISC.ONLY"
        else:
            action_marker = "  COMPETE"
        crisk = row.get('Chalk_Risk', '?')
        # Suppress tier when it contradicts the Action signal:
        # - DISC.ONLY: tier is irrelevant (can't use as fill)
        # - BUY/COMPETE with Tier=FADE: gap% overrode the tier — showing FADE is confusing
        if row['Action'] == 'DISCOUNT_ONLY':
            tier_str = "—"
        elif row['Action'] in ('BUY', 'COMPETE') and row['Tier'] == 'FADE':
            tier_str = "—"
        else:
            tier_str = row['Tier']
        # Flag HIGH chalk risk in table too, not just in ranked list
        chalk_flag = " ⚠" if crisk == 'HIGH' else ""
        print(f"  {row['Team']:<18} {int(row['Seed']):>3} "
              f"{row['E_Points_Blended']:>6.1f} "
              f"{row['E_Points_Base']:>5.1f} "
              f"${row['Fair_Bid']:>5} "
              f"${row['Market_Price']:>4} "
              f"{gap_str:>6} "
              f"{crisk:>5}  "
              f"{action_marker:<14} {tier_str}{chalk_flag}")

    print()
    print("  Base = pts from round advancement only (chalk-safe portion of E_Pts)")
    print("  CRisk = Chalk Risk (LOW/MED/HIGH) — ⚠ = HIGH risk, points depend on upsets")
    print()

    # ── FADE LIST ────────────────────────────────────────────────────────────
    fades = df[df['Action'] == 'FADE'].sort_values('Value_Gap')
    if len(fades) > 0:
        print("  FADE LIST  (market overprices by >40% — gap% too negative to overcome — never bid)")
        print("  " + "-" * 64)
        print(f"  {'Team':<18} {'Sd':>3} {'E_Pts':>6} {'Fair$':>6} {'Mkt$':>5} {'Gap%':>6}  {'Tier'}")
        print("  " + "-" * 64)
        for _, row in fades.head(12).iterrows():
            mkt = row['Market_Price']
            gap_pct = row['Value_Gap'] / mkt if mkt > 0 else 0
            gap_str = f"{gap_pct:.0%}"
            print(f"  {row['Team']:<18} {int(row['Seed']):>3} "
                  f"{row['E_Points_Blended']:>6.1f} "
                  f"${row['Fair_Bid']:>5} "
                  f"${row['Market_Price']:>4} "
                  f"{gap_str:>6}  "
                  f"{row['Tier']}")
        print()

    # ── HISTORICAL CONTEXT ───────────────────────────────────────────────────
    print("  HISTORICAL SEED EFFICIENCY  (all-years pts per chalk-year market price)")
    print("  " + "-" * 76)
    print("  Pts/$ > 1.0 = efficient  |  Pts/$ < 1.0 = market overpays")
    print("  Mkt$ = chalk-year median (what this room pays in chalk years)")
    print()

    # Compute Pts_Per_ChalkyDollar using chalk-year market prices for the display
    chalk_mkt = pd.Series(CHALK_YEAR_MARKET_PRICES, name='Chalk_Mkt')
    eff_df = seed_stats.copy()
    eff_df['Chalk_Mkt'] = eff_df['Seed'].map(CHALK_YEAR_MARKET_PRICES).fillna(eff_df['Bid_median'])
    eff_df['Pts_Per_ChalkDollar'] = (eff_df['Points_mean'] / eff_df['Chalk_Mkt']).round(3)

    worst = eff_df.nsmallest(5, 'Pts_Per_ChalkDollar')[['Seed','Chalk_Mkt','Points_mean','Pts_Per_ChalkDollar']]
    best  = eff_df.nlargest(5,  'Pts_Per_ChalkDollar')[['Seed','Chalk_Mkt','Points_mean','Pts_Per_ChalkDollar']]
    print(f"  {'Seed':>5} {'Mkt$':>5} {'Avg Pts':>8} {'Pts/$':>6}   {'Seed':>5} {'Mkt$':>5} {'Avg Pts':>8} {'Pts/$':>6}")
    print(f"  {'(worst)':>5}                        {'(best)':>5}")
    print("  " + "-" * 76)
    for (_, wr), (_, br) in zip(worst.iterrows(), best.iterrows()):
        print(f"  {int(wr['Seed']):>5} ${wr['Chalk_Mkt']:>4.0f} {wr['Points_mean']:>8.1f} {wr['Pts_Per_ChalkDollar']:>6.2f}"
              f"   {int(br['Seed']):>5} ${br['Chalk_Mkt']:>4.0f} {br['Points_mean']:>8.1f} {br['Pts_Per_ChalkDollar']:>6.2f}")
    print()
    print("  High seeds score more pts but cost far more than they deliver per dollar.")
    print("  In chalk years: market pays $47 for ~17 pts from a 1-seed (0.36 pts/$).")
    print("  Note: Avg Pts includes all tournament types. Seeds 9-15 score fewer pts")
    print("  than the mean suggests in chalk years — Cinderella runs inflate the average.")
    print()

    # ── RANKED BID LIST ──────────────────────────────────────────────────────
    anchors, anchor_name_1, anchor_name_2, fills, pivots = build_ranked_bid_list(df)

    # Short labels for column headers (truncate to 8 chars)
    lbl_1 = (anchor_name_1 or 'Anchor1')[:8]
    lbl_2 = (anchor_name_2 or 'Anchor2')[:8]

    W = 80

    # ── STEP 1: ANCHOR DECISION ──────────────────────────────────────────────
    print("  ═" * (W // 2))
    print("  STEP 1 — ANCHOR DECISION  (decide before the auction starts)")
    print("  " + "─" * 76)
    print("  Chalk-year pattern: ONE deep-running anchor wins it.")
    print("  2025: Florida $53 won. 2021: Gonzaga $57 won. Commit before walking in.")
    print()
    print(f"  {'Option':<14} {'Sd':>3} {'E_Pts':>6} {'Fair$':>6} {'Mkt$':>5} {'Gap%':>6}  "
          f"{'After':>6}  {'Trade-off'}")
    print("  " + "─" * 76)
    for key, a in anchors.items():
        gap_str = f"{a['Gap_Pct']:+.0%}"
        print(f"  {a['Team']:<14} {a['Seed']:>3} {a['E_Pts']:>6.1f} "
              f"${a['Fair']:>5} ${a['Mkt']:>4} {gap_str:>6}  "
              f"${a['Budget_After']:>5}  {a['Note']}")
    print()
    # Warn when no anchor has positive gap% — the room is overpriced on all elite teams
    _all_neg = all(a['Gap_Pct'] < 0 for a in anchors.values())
    if _all_neg and len(anchors) > 0:
        print("  ⚠  No anchor has a positive gap%. Market overprices all ANCHOR-tier teams.")
        print("     Consider the no-anchor path: load BUY fills only, target 8-10 teams.")
        print("     Only pursue an anchor if it stalls well below Fair$.")
        print()
    print("  No anchor: $100 for fills. 8-10 teams. Needs volume. Higher variance in chalk year.")
    print()

    # ── STEP 2: RANKED BID LIST ──────────────────────────────────────────────
    print("  ═" * (W // 2))
    print("  STEP 2 — RANKED BID LIST  (use during auction — cross off as teams go)")
    print("  " + "─" * 76)
    print(f"  Fair$ = your ceiling (NEVER exceed)  |  Mkt$ = what you need to win")

    # Collapse to single Rem$ column if both anchor costs are identical
    _a1_mkt = anchors[anchor_name_1]['Mkt'] if anchor_name_1 else 0
    _a2_mkt = anchors[anchor_name_2]['Mkt'] if anchor_name_2 else 0
    _single_rem_col = (_a1_mkt == _a2_mkt)

    if _single_rem_col:
        print(f"  Rem$ = budget remaining AFTER this team  (both anchors cost ${_a1_mkt})")
    else:
        print(f"  Rem$ = budget remaining AFTER this team  "
              f"(left col = {anchor_name_1} anchor, right = {anchor_name_2})")
    print(f"  ⚔  = contested pair — plan to win ONE, not both")
    print(f"  ⚠  = HIGH chalk risk — points depend on upsets that may not happen")
    print()
    if _single_rem_col:
        print(f"  {'#':>3}  {'Team':<16} {'Sd':>3} {'E_Pts':>6} {'Fair$':>6} {'Mkt$':>5} "
              f"{'Action':<9} {'Rem$':>8}  Notes")
    else:
        col1 = f"Rem$({lbl_1})"
        col2 = f"Rem$({lbl_2})"
        print(f"  {'#':>3}  {'Team':<16} {'Sd':>3} {'E_Pts':>6} {'Fair$':>6} {'Mkt$':>5} "
              f"{'Action':<9} {col1:>12} {col2:>12}  Notes")
    print("  " + "─" * 92)

    for i, f in enumerate(fills, 1):
        if _single_rem_col:
            rem_str = f"${f['Rem_A1']:>+4}" if f['Rem_A1'] >= 0 else f"[OVER ${abs(f['Rem_A1'])}]"
        else:
            rem_str  = None
        rem_a1_str = f"${f['Rem_A1']:>+4}" if f['Rem_A1'] >= 0 else f"[OVER ${abs(f['Rem_A1'])}]"
        rem_a2_str = f"${f['Rem_A2']:>+4}" if f['Rem_A2'] >= 0 else f"[OVER ${abs(f['Rem_A2'])}]"

        notes = []
        if f['Contested']:
            notes.append(f"⚔ vs {f['Contested_With']}")
        if f['Chalk_Risk'] == 'HIGH':
            notes.append("⚠ chalk risk")
        if f['Chalk_Risk'] == 'MEDIUM':
            notes.append("~ med risk")
        note_str = "  " + " | ".join(notes) if notes else ""

        # Budget cut lines
        prev_a1 = fills[i-2]['Rem_A1'] if i > 1 else BUDGET
        prev_a2 = fills[i-2]['Rem_A2'] if i > 1 else BUDGET
        if f['Rem_A1'] < 0 and prev_a1 >= 0:
            if _single_rem_col:
                print(f"  ───  ─── budget limit — teams below need no-anchor path ───")
            else:
                print(f"  ───  ─── {anchor_name_1} budget limit (teams below need {anchor_name_2} or no-anchor) ───")
        if not _single_rem_col and f['Rem_A2'] < 0 and prev_a2 >= 0:
            print(f"  ───  ─── {anchor_name_2} budget limit (teams below need no-anchor path) ───")

        if _single_rem_col:
            print(f"  {i:>3}  {f['Team']:<16} {f['Seed']:>3} {f['E_Pts']:>6.1f} "
                  f"${f['Fair']:>5} ${f['Mkt']:>4} "
                  f"{f['Action']:<9} {rem_str:>8} {note_str}")
        else:
            print(f"  {i:>3}  {f['Team']:<16} {f['Seed']:>3} {f['E_Pts']:>6.1f} "
                  f"${f['Fair']:>5} ${f['Mkt']:>4} "
                  f"{f['Action']:<9} {rem_a1_str:>12} {rem_a2_str:>12} {note_str}")

    print()

    # ── STEP 3: PIVOT SCENARIOS ───────────────────────────────────────────────
    print("  ═" * (W // 2))
    print("  STEP 3 — PIVOT SCENARIOS  (pre-plan for the most likely breaks)")
    print("  " + "─" * 76)

    for key, pv in pivots.items():
        print(f"  IF {pv['label']}")
        print(f"  {pv['desc']}")
        if pv['teams']:
            team_strs = [f"{t['Team']}(${t['Mkt']})" for t in pv['teams']]
            total_mkt = sum(t['Mkt'] for t in pv['teams'])
            total_pts = pv['pts']
            print(f"  → {' + '.join(team_strs)}")
            if total_pts:
                anchor_cost = pv.get('anchor_cost', 0)
                if anchor_cost > 0:
                    print(f"  → Fill Mkt: ${total_mkt} | Total Spend: ${total_mkt + anchor_cost} | E_Pts: {total_pts:.1f} | Rem: ${pv['rem']}")
                else:
                    print(f"  → Total Mkt: ${total_mkt} | E_Pts: {total_pts:.1f} | Rem: ${pv['rem']}")
        print()

    # ── STEP 4: IN-AUCTION RULES ─────────────────────────────────────────────
    print("  ═" * (W // 2))
    print("  STEP 4 — IN-AUCTION RULES")
    print("  " + "─" * 76)
    print("  Every team: max bid = min(Fair$, Dynamic_Ceiling)")
    print("    Dynamic_Ceiling = (Team_E_Pts ÷ Pts_Still_Needed) × Budget_Remaining")
    print("  Lose a bid → cross it off, check Rem$ column, continue down the list.")
    print("  Contested pair comes up → bid your one target, let the other go.")
    print("  Lose your anchor → jump to PIVOT: LOST ANCHOR section above.")
    print("  Budget under $15 → BUDGET LOW pivot only. Don't reach.")
    print("  DISCOUNT_ONLY team → never chase above Fair$. Only bid if room stalls.")
    print("  FADE team → never bid. Not in the list. Do not get pulled in.")
    print()
    print("=" * 80)


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(df, seed_stats, winners_df):
    """Save CSV outputs."""
    print("  OUTPUTS")
    print("  " + "-" * 76)

    # Full valuation file
    out = df.sort_values('Fair_Bid', ascending=False)
    out.to_csv(OUTPUT_DIR / 'team_valuations_2026.csv', index=False)
    print(f"  ✓ team_valuations_2026.csv")

    # Auction day cheat sheet (lean columns)
    bidder_cols = [
        'Team', 'Seed', 'Region',
        'E_Points_Blended', 'E_Points_Base', 'E_Points_Upset',
        'P_R64', 'P_R32', 'P_S16', 'P_E8', 'P_F4', 'P_Championship', 'P_WIN_Championship',
        'Fair_Bid', 'Market_Price', 'Value_Gap', 'Value_Gap_Pct',
        'Action', 'Tier', 'Chalk_Risk', 'Upset_Pct',
        'Points_Advantage', 'Pts_Per_Market_Dollar'
    ]
    # Only include columns that exist (P_WIN_Championship may not always be present)
    bidder_cols = [c for c in bidder_cols if c in out.columns]
    bidders = out[bidder_cols].copy()
    bidders.to_csv(OUTPUT_DIR / 'team_bidders_2026.csv', index=False)
    print(f"  ✓ team_bidders_2026.csv")

    seed_stats.to_csv(OUTPUT_DIR / 'historical_seed_performance.csv', index=False)
    print(f"  ✓ historical_seed_performance.csv")

    winners_df.to_csv(OUTPUT_DIR / 'historical_winners.csv', index=False)
    print(f"  ✓ historical_winners.csv")

    print()
    print("  USE DURING AUCTION: team_bidders_2026.csv")
    print("  USE FOR ANALYSIS:   team_valuations_2026.csv")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("           L4.02 — CALCUTTA BID GUIDANCE OPTIMIZER")
    print("=" * 80)
    print()

    print("STEP 1: Load Data")
    print("-" * 80)
    auction_df = load_auction_history()
    predictions_2026 = load_2026_predictions()
    print()

    print("STEP 2: Historical Analysis")
    print("-" * 80)
    winners_df = analyze_winners(auction_df)
    seed_stats = analyze_seed_performance(auction_df)
    pot_info = estimate_pot_structure(auction_df)
    print()

    print("STEP 3: Calculate 2026 Bid Guidance")
    print("-" * 80)
    team_valuations = calculate_bid_guidance(
        predictions_2026, seed_stats, winners_df, pot_info
    )
    print()

    print_summary(team_valuations, seed_stats, winners_df, pot_info)
    save_outputs(team_valuations, seed_stats, winners_df)


if __name__ == "__main__":
    main()
