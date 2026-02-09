"""
L4.02 - Bracket Simulator & Optimizer

Takes validated H2H model and generates optimal tournament brackets.

Features:
  - Monte Carlo simulation (50K runs, configurable)
  - Round-by-round probability tracking for all teams
  - Multiple optimization strategies (chalk, EV, Elite8-focus)
  - Bracket export in CSV and readable text formats
  - Convergence diagnostics

Inputs (same as L4.01):
  - Elite 8 predictions (L3 output CSV)
  - H2H models and config
  - predict_set_2026.csv (features)
  - Bracket from tournamentSeed assignments

Outputs:
  - round_probabilities.csv — P(reach round) for all teams, all rounds
  - optimal_bracket_[strategy].csv — optimal picks per strategy
  - bracket_trace_[strategy].txt — readable bracket format
  - simulation_summary.json — convergence, champion odds, diagnostics
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# L3 output paths
ELITE8_PREDICTIONS_PATH = Path("../L3/elite8/outputs/05_2026_predictions/elite8_predictions_2026_long.csv")
H2H_MODEL_DIR            = Path("../L3/h2h/models")
PREDICTION_DATA_PATH     = Path("../L3/data/predictionData/predict_set_2026.csv")

# L4 output
OUTPUT_DIR = Path("outputs/02_bracket_simulator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
N_SIMULATIONS      = 50000
CONVERGENCE_CHECK  = 10000  # Check convergence every N sims
CONVERGENCE_THRESH = 0.001  # Stop if champion probs change < 0.1%
RANDOM_SEED        = 42
np.random.seed(RANDOM_SEED)

# Scoring rules (ESPN standard)
SCORING = {
    'R64': 10,
    'R32': 20,
    'S16': 40,
    'E8':  80,
    'FF':  160,
    'Championship': 320
}

# Config name → short key mapping
NAME_TO_KEY = {
    'Gradient Boosting':    'GB',
    'SVM':                  'SVM',
    'Random Forest':        'RF',
    'Neural Network':       'NN',
    'Gaussian Naive Bayes': 'GNB'
}

H2H_MODEL_FILES = {
    'GB':  'gradient_boosting.pkl',
    'SVM': 'svm.pkl',
    'RF':  'random_forest.pkl',
    'NN':  'neural_network.pkl',
    'GNB': 'gaussian_naive_bayes.pkl'
}

# ============================================================================
# LOADING (reused from L4.01)
# ============================================================================

def load_elite8_predictions():
    """Load pre-computed Elite 8 predictions from L3."""
    df = pd.read_csv(ELITE8_PREDICTIONS_PATH)
    return df


def load_h2h():
    """Load H2H models, scaler, and config."""
    with open(H2H_MODEL_DIR / 'ensemble_config.json', 'r') as f:
        config = json.load(f)

    feature_columns = config['feature_columns']
    source_columns  = [c.replace('pct_diff_', '') for c in feature_columns]

    weights = {NAME_TO_KEY[k]: v for k, v in config['weights'].items()}

    fallback_weights = {}
    for scenario, w in config.get('fallback_weights', {}).items():
        fallback_weights[scenario] = {NAME_TO_KEY[k]: v for k, v in w.items()}

    models = {}
    for key, filename in H2H_MODEL_FILES.items():
        with open(H2H_MODEL_DIR / filename, 'rb') as f:
            models[key] = pickle.load(f)

    with open(H2H_MODEL_DIR / 'feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return {
        'models':           models,
        'scaler':           scaler,
        'weights':          weights,
        'fallback_weights': fallback_weights,
        'feature_columns':  feature_columns,
        'source_columns':   source_columns,
    }


def load_prediction_data():
    """Load raw 2026 prediction features."""
    df = pd.read_csv(PREDICTION_DATA_PATH)
    return df


def construct_bracket(prediction_df):
    """
    Build tournament bracket from tournamentSeed.
    Returns: bracket dict, bracket_df
    """
    df = prediction_df.copy()
    df = df[(df['tournamentSeed'] >= 1) & (df['tournamentSeed'] <= 16)].copy()

    if len(df) == 0:
        raise ValueError("No teams with seeds 1-16")

    df = df.sort_values('tournamentSeed').reset_index(drop=True)

    region_names = ['East', 'West', 'South', 'Midwest']
    regions = []
    for seed in sorted(df['tournamentSeed'].unique()):
        seed_teams = df[df['tournamentSeed'] == seed]
        for idx in range(len(seed_teams)):
            regions.append(region_names[idx % len(region_names)])
    df['Region'] = regions

    df = df.sort_values(['Region', 'tournamentSeed']).reset_index(drop=True)

    matchup_pairs = [
        (1, 16), (8, 9),
        (5, 12), (4, 13),
        (6, 11), (3, 14),
        (7, 10), (2, 15)
    ]

    bracket = {}
    for region in sorted(df['Region'].unique()):
        region_teams = df[df['Region'] == region].sort_values('tournamentSeed')
        bracket[region] = []

        for seed1, seed2 in matchup_pairs:
            t1 = region_teams[region_teams['tournamentSeed'] == seed1]
            t2 = region_teams[region_teams['tournamentSeed'] == seed2]
            if len(t1) > 0 and len(t2) > 0:
                bracket[region].append((t1.iloc[0]['Team'], t2.iloc[0]['Team']))

    return bracket, df


# ============================================================================
# H2H PREDICTION (reused from L4.01)
# ============================================================================

def compute_h2h_features(team1, team2, prediction_df, source_columns):
    """Compute 29 pct_diff features for a matchup."""
    t1_row = prediction_df[prediction_df['Team'] == team1]
    t2_row = prediction_df[prediction_df['Team'] == team2]

    if len(t1_row) == 0 or len(t2_row) == 0:
        return None

    t1_vals = t1_row[source_columns].values[0].astype(float)
    t2_vals = t2_row[source_columns].values[0].astype(float)

    denom = np.abs(t1_vals) + np.abs(t2_vals)
    diffs = np.where(denom == 0, 0.0, (t1_vals - t2_vals) / denom)

    return diffs


def predict_h2h_matchup(team1, team2, prediction_df, h2h):
    """Predict P(team1 beats team2) using H2H ensemble."""
    diffs = compute_h2h_features(team1, team2, prediction_df, h2h['source_columns'])
    if diffs is None:
        return 0.5, {'excluded': [], 'note': 'Unknown team'}

    X = h2h['scaler'].transform(diffs.reshape(1, -1))

    predictions = {}
    excluded = []

    for key, model in h2h['models'].items():
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[0, 1]
            else:
                pred = float(model.predict(X)[0])

            if key == 'GNB' and (pred < 0.01 or pred > 0.99):
                excluded.append(key)
                continue

            predictions[key] = pred
        except Exception as e:
            excluded.append(key)

    if len(predictions) == 0:
        return 0.5, {'excluded': excluded, 'note': 'All models excluded'}

    if 'GNB' in excluded and 'exclude_gaussian_naive_bayes' in h2h['fallback_weights']:
        weights = h2h['fallback_weights']['exclude_gaussian_naive_bayes']
    else:
        weights = h2h['weights']

    active_w = {k: weights[k] for k in predictions if k in weights}
    total_w  = sum(active_w.values())

    if total_w == 0:
        return 0.5, {'excluded': excluded, 'note': 'Zero weight'}

    prob = sum(predictions[k] * active_w[k] for k in active_w) / total_w

    return prob, {'excluded': excluded}


# ============================================================================
# ENHANCED SIMULATION (tracks all rounds)
# ============================================================================

def simulate_tournament_full(bracket, prediction_df, h2h, n_sims=50000, 
                             convergence_check=10000, convergence_thresh=0.001):
    """
    Monte Carlo simulation with full round tracking and convergence checking.
    
    Returns:
        round_probs: {team: {round: probability}}
        champion_probs: {team: probability}
        convergence_history: list of champion prob snapshots
        cache: matchup probability cache
    """
    print(f"Running {n_sims:,} tournament simulations...")
    print(f"  Convergence: check every {convergence_check:,}, threshold {convergence_thresh:.3f}")

    # Validate source columns
    missing = [c for c in h2h['source_columns'] if c not in prediction_df.columns]
    if missing:
        print(f"  ⚠ {len(missing)} source columns missing")
        for c in missing:
            prediction_df[c] = 0.0
    else:
        print(f"  ✓ All {len(h2h['source_columns'])} source columns present")

    # Initialize trackers
    round_counts = defaultdict(lambda: defaultdict(int))  # {team: {round: count}}
    
    # Matchup cache
    cache = {}
    total_matchups = 0
    gnb_exclusion_count = 0
    
    # Convergence tracking
    convergence_history = []
    last_champion_probs = None

    def get_prob(t1, t2):
        """Cached matchup probability."""
        nonlocal gnb_exclusion_count, total_matchups
        total_matchups += 1

        if (t1, t2) in cache:
            return cache[(t1, t2)]

        p, info = predict_h2h_matchup(t1, t2, prediction_df, h2h)
        cache[(t1, t2)]  = p
        cache[(t2, t1)]  = 1.0 - p

        if 'GNB' in info.get('excluded', []):
            gnb_exclusion_count += 1

        return p

    for sim_idx in range(n_sims):
        if sim_idx % 5000 == 0:
            print(f"    [{sim_idx:>6}/{n_sims}] ...", flush=True)

        # Track teams that reach each round in this simulation
        sim_rounds = defaultdict(set)  # {round: set of teams}

        # --- R64 ---
        r64_winners = {}
        for region, matchups in bracket.items():
            r64_winners[region] = []
            for t1, t2 in matchups:
                sim_rounds['R64'].add(t1)
                sim_rounds['R64'].add(t2)
                
                p = get_prob(t1, t2)
                winner = t1 if np.random.random() < p else t2
                r64_winners[region].append(winner)

        # --- R32 ---
        r32_winners = {}
        for region, winners in r64_winners.items():
            r32_winners[region] = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    t1, t2 = winners[i], winners[i+1]
                    sim_rounds['R32'].add(t1)
                    sim_rounds['R32'].add(t2)
                    
                    p = get_prob(t1, t2)
                    winner = t1 if np.random.random() < p else t2
                    r32_winners[region].append(winner)

        # --- S16 (2 games per region) ---
        s16_winners = {}
        for region, winners in r32_winners.items():
            s16_winners[region] = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    t1, t2 = winners[i], winners[i+1]
                    sim_rounds['S16'].add(t1)
                    sim_rounds['S16'].add(t2)
                    
                    p = get_prob(t1, t2)
                    winner = t1 if np.random.random() < p else t2
                    s16_winners[region].append(winner)

        # --- Elite 8 (1 game per region) ---
        e8_winners = {}
        for region, winners in s16_winners.items():
            if len(winners) == 2:
                t1, t2 = winners[0], winners[1]
                sim_rounds['E8'].add(t1)
                sim_rounds['E8'].add(t2)
                
                p = get_prob(t1, t2)
                winner = t1 if np.random.random() < p else t2
                e8_winners[region] = winner

        # --- Final Four (2 semifinal games) ---
        ff_matchups = [('East', 'South'), ('West', 'Midwest')]
        ff_winners = []
        for r1, r2 in ff_matchups:
            if r1 in e8_winners and r2 in e8_winners:
                t1, t2 = e8_winners[r1], e8_winners[r2]
                sim_rounds['FF'].add(t1)
                sim_rounds['FF'].add(t2)
                
                p = get_prob(t1, t2)
                winner = t1 if np.random.random() < p else t2
                ff_winners.append(winner)

        # --- Championship ---
        champion = None
        if len(ff_winners) == 2:
            t1, t2 = ff_winners[0], ff_winners[1]
            sim_rounds['Championship'].add(t1)
            sim_rounds['Championship'].add(t2)
            
            p = get_prob(t1, t2)
            champion = t1 if np.random.random() < p else t2

        # Update round counts
        for round_name, teams in sim_rounds.items():
            for team in teams:
                round_counts[team][round_name] += 1

        # Convergence check
        if (sim_idx + 1) % convergence_check == 0:
            current_champion_probs = {
                team: round_counts[team].get('Championship', 0) / (sim_idx + 1)
                for team in round_counts
            }
            convergence_history.append({
                'sim': sim_idx + 1,
                'champion_probs': current_champion_probs.copy()
            })
            
            if last_champion_probs is not None:
                # Check if top 5 champion probs have converged
                top_teams = sorted(current_champion_probs.items(), 
                                 key=lambda x: -x[1])[:5]
                max_change = max(
                    abs(current_champion_probs.get(t, 0) - last_champion_probs.get(t, 0))
                    for t, _ in top_teams
                )
                
                if max_change < convergence_thresh:
                    print(f"\n  ✓ Converged at {sim_idx + 1:,} sims "
                          f"(max change: {max_change:.4f})")
                    n_sims = sim_idx + 1  # Update actual sim count
                    break
            
            last_champion_probs = current_champion_probs.copy()

    # Convert counts to probabilities
    round_probs = {}
    for team, rounds in round_counts.items():
        round_probs[team] = {
            round_name: count / n_sims 
            for round_name, count in rounds.items()
        }

    champion_probs = {
        team: round_counts[team].get('Championship', 0) / n_sims
        for team in round_counts
    }

    stats = {
        'n_simulations':     n_sims,
        'total_matchups':    total_matchups,
        'unique_matchups':   len(cache) // 2,
        'gnb_exclusions':    gnb_exclusion_count,
        'gnb_exclusion_rate': round(gnb_exclusion_count / max(len(cache) // 2, 1), 4),
        'converged':         n_sims < N_SIMULATIONS,
        'convergence_history': convergence_history
    }

    print(f"  ✓ Complete")
    print(f"    - Total sims: {n_sims:,}")
    print(f"    - Unique matchups: {stats['unique_matchups']:,}")
    print(f"    - GNB exclusion rate: {stats['gnb_exclusion_rate']:.1%}")
    
    top_champs = sorted(champion_probs.items(), key=lambda x: -x[1])[:5]
    print(f"    - Top 5 champions:")
    for team, prob in top_champs:
        print(f"        {team}: {prob:.1%}")

    return round_probs, champion_probs, stats, cache


# ============================================================================
# OPTIMIZATION STRATEGIES
# ============================================================================

def optimize_chalk(bracket, round_probs):
    """
    Chalk strategy: always pick the team with higher probability.
    Simple but effective baseline.
    """
    picks = {}
    
    # For each matchup, pick higher probability team
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            # Use R32 probability as tiebreaker (who's more likely to advance)
            p1 = round_probs.get(t1, {}).get('R32', 0)
            p2 = round_probs.get(t2, {}).get('R32', 0)
            
            winner = t1 if p1 >= p2 else t2
            picks[(t1, t2)] = winner
    
    return picks, "Chalk (always pick favorite)"


def optimize_expected_value(bracket, round_probs, bracket_df, scoring=SCORING):
    """
    Expected Value strategy: maximize expected points given scoring rules.
    
    For each game, calculate:
      EV(pick team1) = P(team1 wins) * points_if_correct
      EV(pick team2) = P(team2 wins) * points_if_correct
    
    Pick whichever has higher EV.
    """
    picks = {}
    
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            # Get probabilities of advancing to next round
            p1_r32 = round_probs.get(t1, {}).get('R32', 0)
            p2_r32 = round_probs.get(t2, {}).get('R32', 0)
            
            # EV for picking each team (R64 round)
            ev1 = p1_r32 * scoring['R64']
            ev2 = p2_r32 * scoring['R64']
            
            winner = t1 if ev1 >= ev2 else t2
            picks[(t1, t2)] = winner
    
    return picks, "Expected Value (maximize expected points)"


def optimize_elite8_focus(bracket, round_probs, elite8_probs_df):
    """
    Elite 8 Focus: optimize through Sweet 16, then pick chalk.
    
    Uses Elite 8 direct probabilities for early rounds where they're most
    reliable, then switches to H2H simulation probabilities for later rounds.
    """
    picks = {}
    
    # Create lookup for Elite 8 probs
    e8_lookup = dict(zip(elite8_probs_df['Team'], 
                        elite8_probs_df['Elite8_Probability']))
    
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            # Use Elite 8 direct probability as decision metric
            e8_prob1 = e8_lookup.get(t1, 0)
            e8_prob2 = e8_lookup.get(t2, 0)
            
            winner = t1 if e8_prob1 >= e8_prob2 else t2
            picks[(t1, t2)] = winner
    
    return picks, "Elite 8 Focus (optimize through Sweet 16)"


# ============================================================================
# BRACKET SIMULATION (apply picks to get full bracket)
# ============================================================================

def simulate_bracket_from_picks(bracket, picks, round_probs, cache):
    """
    Given R64 picks, simulate the rest of the bracket using probabilities.
    Returns full bracket path to championship.
    """
    result = {
        'R64': {},
        'R32': {},
        'S16': {},
        'E8': {},
        'FF': {},
        'Championship': None
    }
    
    # R64: use provided picks
    r64_winners = {}
    for region, matchups in bracket.items():
        r64_winners[region] = []
        result['R64'][region] = []
        
        for t1, t2 in matchups:
            winner = picks.get((t1, t2), t1)  # Default to t1 if no pick
            r64_winners[region].append(winner)
            
            # Get probability for this pick
            p = cache.get((t1, t2), 0.5)
            if winner == t2:
                p = 1.0 - p
            
            result['R64'][region].append({
                'team1': t1,
                'team2': t2,
                'winner': winner,
                'prob': p
            })
    
    # R32: pick higher probability from remaining teams
    r32_winners = {}
    for region, winners in r64_winners.items():
        r32_winners[region] = []
        result['R32'][region] = []
        
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                t1, t2 = winners[i], winners[i+1]
                
                # Pick team with higher S16 probability
                p1 = round_probs.get(t1, {}).get('S16', 0)
                p2 = round_probs.get(t2, {}).get('S16', 0)
                winner = t1 if p1 >= p2 else t2
                r32_winners[region].append(winner)
                
                p = cache.get((t1, t2), 0.5)
                if winner == t2:
                    p = 1.0 - p
                
                result['R32'][region].append({
                    'team1': t1,
                    'team2': t2,
                    'winner': winner,
                    'prob': p
                })
    
    # S16
    s16_winners = {}
    for region, winners in r32_winners.items():
        s16_winners[region] = []
        result['S16'][region] = []
        
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                t1, t2 = winners[i], winners[i+1]
                
                p1 = round_probs.get(t1, {}).get('E8', 0)
                p2 = round_probs.get(t2, {}).get('E8', 0)
                winner = t1 if p1 >= p2 else t2
                s16_winners[region].append(winner)
                
                p = cache.get((t1, t2), 0.5)
                if winner == t2:
                    p = 1.0 - p
                
                result['S16'][region].append({
                    'team1': t1,
                    'team2': t2,
                    'winner': winner,
                    'prob': p
                })
    
    # E8
    e8_winners = {}
    for region, winners in s16_winners.items():
        if len(winners) == 2:
            t1, t2 = winners[0], winners[1]
            
            p1 = round_probs.get(t1, {}).get('FF', 0)
            p2 = round_probs.get(t2, {}).get('FF', 0)
            winner = t1 if p1 >= p2 else t2
            e8_winners[region] = winner
            
            p = cache.get((t1, t2), 0.5)
            if winner == t2:
                p = 1.0 - p
            
            result['E8'][region] = {
                'team1': t1,
                'team2': t2,
                'winner': winner,
                'prob': p
            }
    
    # FF
    ff_matchups = [('East', 'South'), ('West', 'Midwest')]
    ff_winners = []
    result['FF'] = {}
    
    for r1, r2 in ff_matchups:
        if r1 in e8_winners and r2 in e8_winners:
            t1, t2 = e8_winners[r1], e8_winners[r2]
            
            p1 = round_probs.get(t1, {}).get('Championship', 0)
            p2 = round_probs.get(t2, {}).get('Championship', 0)
            winner = t1 if p1 >= p2 else t2
            ff_winners.append(winner)
            
            p = cache.get((t1, t2), 0.5)
            if winner == t2:
                p = 1.0 - p
            
            matchup_name = f"{r1} vs {r2}"
            result['FF'][matchup_name] = {
                'team1': t1,
                'team2': t2,
                'winner': winner,
                'prob': p
            }
    
    # Championship
    if len(ff_winners) == 2:
        t1, t2 = ff_winners[0], ff_winners[1]
        
        # Pick higher champion probability
        p1 = round_probs.get(t1, {}).get('Championship', 0)
        p2 = round_probs.get(t2, {}).get('Championship', 0)
        winner = t1 if p1 >= p2 else t2
        
        p = cache.get((t1, t2), 0.5)
        if winner == t2:
            p = 1.0 - p
        
        result['Championship'] = {
            'team1': t1,
            'team2': t2,
            'winner': winner,
            'prob': p
        }
    
    return result


# ============================================================================
# EXPORTS
# ============================================================================

def export_round_probabilities(round_probs, bracket_df):
    """Export round probability table."""
    rows = []
    
    rounds = ['R64', 'R32', 'S16', 'E8', 'FF', 'Championship']
    
    for team in round_probs:
        row = {'Team': team}
        
        # Add seed and region if available
        team_info = bracket_df[bracket_df['Team'] == team]
        if len(team_info) > 0:
            row['Seed'] = int(team_info.iloc[0]['tournamentSeed'])
            row['Region'] = team_info.iloc[0]['Region']
        else:
            row['Seed'] = None
            row['Region'] = None
        
        # Add probabilities for each round
        for round_name in rounds:
            row[f'P_{round_name}'] = round_probs[team].get(round_name, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by championship probability
    df = df.sort_values('P_Championship', ascending=False)
    
    return df


def export_bracket_csv(bracket_result, strategy_name):
    """Export bracket picks to CSV."""
    rows = []
    
    for round_name in ['R64', 'R32', 'S16', 'E8']:
        if round_name in bracket_result:
            for region, games in bracket_result[round_name].items():
                if isinstance(games, list):
                    for game in games:
                        rows.append({
                            'Round': round_name,
                            'Region': region,
                            'Team1': game['team1'],
                            'Team2': game['team2'],
                            'Winner': game['winner'],
                            'Probability': round(game['prob'], 4)
                        })
                elif isinstance(games, dict):  # E8 is dict not list
                    rows.append({
                        'Round': round_name,
                        'Region': region,
                        'Team1': games['team1'],
                        'Team2': games['team2'],
                        'Winner': games['winner'],
                        'Probability': round(games['prob'], 4)
                    })
    
    # FF
    if 'FF' in bracket_result:
        for matchup_name, game in bracket_result['FF'].items():
            rows.append({
                'Round': 'FF',
                'Region': matchup_name,
                'Team1': game['team1'],
                'Team2': game['team2'],
                'Winner': game['winner'],
                'Probability': round(game['prob'], 4)
            })
    
    # Championship
    if bracket_result.get('Championship'):
        game = bracket_result['Championship']
        rows.append({
            'Round': 'Championship',
            'Region': 'Final',
            'Team1': game['team1'],
            'Team2': game['team2'],
            'Winner': game['winner'],
            'Probability': round(game['prob'], 4)
        })
    
    df = pd.DataFrame(rows)
    
    filename = f"optimal_bracket_{strategy_name.lower().replace(' ', '_')}.csv"
    filepath = OUTPUT_DIR / filename
    df.to_csv(filepath, index=False)
    
    return filepath


def export_bracket_text(bracket_result, strategy_name, champion_prob):
    """Export human-readable bracket trace."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(f"  OPTIMAL BRACKET: {strategy_name}")
    lines.append("=" * 70)
    lines.append("")
    
    # Champion
    if bracket_result.get('Championship'):
        champ = bracket_result['Championship']['winner']
        lines.append(f"🏆 CHAMPION: {champ} ({champion_prob:.1%})")
        lines.append("")
    
    # Round by round
    for round_name in ['R64', 'R32', 'S16', 'E8', 'FF', 'Championship']:
        if round_name not in bracket_result:
            continue
        
        lines.append(f"{round_name}")
        lines.append("-" * 70)
        
        if round_name == 'Championship':
            game = bracket_result['Championship']
            lines.append(f"  {game['winner']:<25} def. {game['team2'] if game['winner'] == game['team1'] else game['team1']:<25} ({game['prob']:.0%})")
        elif round_name == 'FF':
            for matchup_name, game in bracket_result['FF'].items():
                loser = game['team2'] if game['winner'] == game['team1'] else game['team1']
                lines.append(f"  {game['winner']:<25} def. {loser:<25} ({game['prob']:.0%})")
        else:
            # R64, R32, S16, E8
            for region in ['East', 'Midwest', 'South', 'West']:
                if region not in bracket_result[round_name]:
                    continue
                
                games = bracket_result[round_name][region]
                if isinstance(games, dict):  # E8
                    games = [games]
                
                for game in games:
                    loser = game['team2'] if game['winner'] == game['team1'] else game['team1']
                    lines.append(f"  {game['winner']:<25} def. {loser:<25} ({game['prob']:.0%})")
        
        lines.append("")
    
    lines.append("=" * 70)
    
    text = "\n".join(lines)
    
    filename = f"bracket_trace_{strategy_name.lower().replace(' ', '_')}.txt"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w') as f:
        f.write(text)
    
    return filepath


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("           L4.02 — BRACKET SIMULATOR & OPTIMIZER")
    print("=" * 70)
    print()

    # ================================================================
    # STEP 1: Load
    # ================================================================
    print("STEP 1: Loading data and models")
    print("-" * 70)

    print("Loading Elite 8 predictions...")
    elite8_df = load_elite8_predictions()
    print(f"  ✓ {len(elite8_df)} teams")

    print("Loading H2H models and config...")
    h2h = load_h2h()
    print(f"  ✓ {len(h2h['models'])} models, {len(h2h['feature_columns'])} features")

    print("Loading prediction data...")
    prediction_df = load_prediction_data()
    print(f"  ✓ {len(prediction_df)} teams, {len(prediction_df.columns)} columns")
    
    print()

    # ================================================================
    # STEP 2: Construct bracket
    # ================================================================
    print("STEP 2: Constructing tournament bracket")
    print("-" * 70)
    
    bracket, bracket_df = construct_bracket(prediction_df)
    
    print(f"  ✓ {len(bracket_df)} tournament teams")
    print(f"  ✓ {len(bracket)} regions, {sum(len(m) for m in bracket.values())} R64 matchups")
    print()

    # ================================================================
    # STEP 3: Monte Carlo simulation
    # ================================================================
    print("STEP 3: Monte Carlo tournament simulation")
    print("-" * 70)
    
    round_probs, champion_probs, sim_stats, cache = simulate_tournament_full(
        bracket, prediction_df, h2h, 
        n_sims=N_SIMULATIONS,
        convergence_check=CONVERGENCE_CHECK,
        convergence_thresh=CONVERGENCE_THRESH
    )
    print()

    # ================================================================
    # STEP 4: Generate optimal brackets (multiple strategies)
    # ================================================================
    print("STEP 4: Generating optimal brackets")
    print("-" * 70)
    
    strategies = [
        ('chalk', optimize_chalk),
        ('expected_value', optimize_expected_value),
        ('elite8_focus', optimize_elite8_focus)
    ]
    
    brackets = {}
    
    for strat_key, optimizer_func in strategies:
        print(f"\n  {strat_key}...")
        
        if strat_key == 'chalk':
            picks, desc = optimizer_func(bracket, round_probs)
        elif strat_key == 'expected_value':
            picks, desc = optimizer_func(bracket, round_probs, bracket_df)
        elif strat_key == 'elite8_focus':
            picks, desc = optimizer_func(bracket, round_probs, elite8_df)
        
        # Simulate full bracket from R64 picks
        bracket_result = simulate_bracket_from_picks(bracket, picks, round_probs, cache)
        
        champion = bracket_result['Championship']['winner'] if bracket_result.get('Championship') else None
        champ_prob = champion_probs.get(champion, 0) if champion else 0
        
        brackets[strat_key] = {
            'description': desc,
            'result': bracket_result,
            'champion': champion,
            'champion_prob': champ_prob
        }
        
        print(f"    Strategy: {desc}")
        print(f"    Champion: {champion} ({champ_prob:.1%})")
    
    print()

    # ================================================================
    # STEP 5: Export outputs
    # ================================================================
    print("STEP 5: Exporting outputs")
    print("-" * 70)
    
    # Round probabilities
    round_probs_df = export_round_probabilities(round_probs, bracket_df)
    round_probs_path = OUTPUT_DIR / 'round_probabilities.csv'
    round_probs_df.to_csv(round_probs_path, index=False)
    print(f"  ✓ round_probabilities.csv ({len(round_probs_df)} teams)")
    
    # Optimal brackets (one per strategy)
    for strat_key, bracket_data in brackets.items():
        csv_path = export_bracket_csv(bracket_data['result'], strat_key)
        print(f"  ✓ {csv_path.name}")
        
        txt_path = export_bracket_text(
            bracket_data['result'], 
            bracket_data['description'],
            bracket_data['champion_prob']
        )
        print(f"  ✓ {txt_path.name}")
    
    # Simulation summary
    summary = {
        'simulation': {
            'n_sims': sim_stats['n_simulations'],
            'converged': sim_stats['converged'],
            'unique_matchups': sim_stats['unique_matchups'],
            'gnb_exclusion_rate': sim_stats['gnb_exclusion_rate']
        },
        'top_10_champions': {
            team: round(prob, 4)
            for team, prob in sorted(champion_probs.items(), 
                                   key=lambda x: -x[1])[:10]
        },
        'strategies': {
            strat_key: {
                'description': data['description'],
                'champion': data['champion'],
                'champion_probability': round(data['champion_prob'], 4)
            }
            for strat_key, data in brackets.items()
        }
    }
    
    summary_path = OUTPUT_DIR / 'simulation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ simulation_summary.json")
    print()

    # ================================================================
    # RESULTS SUMMARY
    # ================================================================
    print("=" * 70)
    print("                    SIMULATION SUMMARY")
    print("=" * 70)
    print()
    print(f"  Simulations:      {sim_stats['n_simulations']:,}")
    print(f"  Converged:        {'Yes' if sim_stats['converged'] else 'No'}")
    print(f"  Unique matchups:  {sim_stats['unique_matchups']:,}")
    print()
    
    print("  TOP 10 CHAMPIONSHIP PROBABILITIES")
    print("  " + "-" * 66)
    print(f"  {'Team':<25} {'Probability':<15} {'P(Elite 8)':<15}")
    print("  " + "-" * 66)
    
    for team, prob in sorted(champion_probs.items(), key=lambda x: -x[1])[:10]:
        e8_prob = round_probs.get(team, {}).get('E8', 0)
        print(f"  {team:<25} {prob:>8.1%}          {e8_prob:>8.1%}")
    print()
    
    print("=" * 70)
    print("                    OPTIMAL BRACKETS")
    print("=" * 70)
    print()
    
    for strat_key, data in brackets.items():
        print(f"  {data['description'].upper()}")
        print(f"    Champion: {data['champion']} ({data['champion_prob']:.1%})")
        print()
    
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()