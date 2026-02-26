"""
L4.01 - Tournament Simulator (Validation + Optimization)

Unified script for model validation and bracket optimization.

MODES:
  Validation (--validate):
    - Quick 5K simulation
    - Correlation analysis (Elite 8 direct vs simulated)
    - Disagreement breakdown and visualizations
    - Use during development to verify models

  Production (default):
    - Full 50K simulation with convergence
    - Multiple optimization strategies
    - Optimal bracket generation
    - Use on Selection Sunday for pool submissions

USAGE:
  python 01_tournament_simulator.py --validate   # Run validation
  python 01_tournament_simulator.py              # Generate brackets
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# L3 output paths - Primary models (WITH SEEDS - bracket aware, post-Selection Sunday)
ELITE8_PREDICTIONS_PATH = Path("../L3/elite8/outputs/04_2026_predictions/elite8_predictions_2026_long.csv")
H2H_MODEL_DIR            = Path("../L3/h2h/models_with_seeds")
PREDICTION_DATA_PATH     = Path("../L3/data/predictionData/predict_set_2026.csv")

# Comparison models (NO SEEDS - pure metrics, pre-Selection Sunday)
ELITE8_PREDICTIONS_NO_SEEDS_PATH = Path("../L3/elite8/outputs/04_2026_predictions_no_seeds/elite8_predictions_2026_long.csv")
H2H_MODEL_DIR_NO_SEEDS           = Path("../L3/h2h/models")

# L4 output
OUTPUT_DIR = Path("outputs/01_tournament_simulator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
N_SIMS_VALIDATE   = 5000
N_SIMS_PRODUCTION = 50000
CONVERGENCE_CHECK = 10000
CONVERGENCE_THRESH = 0.001
RANDOM_SEED = 42

# Validation thresholds
HIGH_AGREEMENT_THRESHOLD = 0.85
MODERATE_AGREEMENT_THRESHOLD = 0.70
DISAGREEMENT_THRESHOLD = 0.15

# Scoring rules (ESPN standard)
# Scoring systems
SCORING_ESPN = {
    'R64': 10,
    'R32': 20,
    'S16': 40,
    'E8': 80,
    'FF': 160,
    'Championship': 320
}
# Maximum ESPN: 1,920 points (32×10 + 16×20 + 8×40 + 4×80 + 2×160 + 1×320)

SCORING_YAHOO = {
    'R64': 1,
    'R32': 2,
    'S16': 4,
    'E8': 8,
    'FF': 16,
    'Championship': 32
}
# Maximum Yahoo: 192 points (32×1 + 16×2 + 8×4 + 4×8 + 2×16 + 1×32)

# Default to ESPN for backward compatibility
SCORING = SCORING_ESPN

# Model config
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
# LOADING
# ============================================================================

def load_elite8_predictions():
    """Load pre-computed Elite 8 predictions from L3."""
    df = pd.read_csv(ELITE8_PREDICTIONS_PATH)
    return df


def load_both_elite8_predictions():
    """
    Load both WITH SEEDS and NO SEEDS Elite 8 predictions for comparison.
    
    Returns:
        tuple: (elite8_with_seeds, elite8_no_seeds)
    """
    elite8_with_seeds = pd.read_csv(ELITE8_PREDICTIONS_PATH)
    elite8_no_seeds = pd.read_csv(ELITE8_PREDICTIONS_NO_SEEDS_PATH)
    return elite8_with_seeds, elite8_no_seeds


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


def load_h2h_from_dir(model_dir):
    """Load H2H models from a specific directory."""
    with open(model_dir / 'ensemble_config.json', 'r') as f:
        config = json.load(f)

    feature_columns = config['feature_columns']
    source_columns  = [c.replace('pct_diff_', '') for c in feature_columns]

    weights = {NAME_TO_KEY[k]: v for k, v in config['weights'].items()}

    fallback_weights = {}
    for scenario, w in config.get('fallback_weights', {}).items():
        fallback_weights[scenario] = {NAME_TO_KEY[k]: v for k, v in w.items()}

    models = {}
    for key, filename in H2H_MODEL_FILES.items():
        with open(model_dir / filename, 'rb') as f:
            models[key] = pickle.load(f)

    with open(model_dir / 'feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return {
        'models':           models,
        'scaler':           scaler,
        'weights':          weights,
        'fallback_weights': fallback_weights,
        'feature_columns':  feature_columns,
        'source_columns':   source_columns,
    }


def load_both_h2h_models():
    """
    Load both WITH SEEDS and NO SEEDS H2H models for strategic comparison.
    
    Returns:
        tuple: (h2h_with_seeds, h2h_no_seeds)
    """
    print("Loading H2H models (both WITH and NO SEEDS)...")
    
    h2h_with_seeds = load_h2h_from_dir(H2H_MODEL_DIR)
    h2h_no_seeds = load_h2h_from_dir(H2H_MODEL_DIR_NO_SEEDS)
    
    print(f"  ✓ WITH SEEDS: {len(h2h_with_seeds['models'])} models, {len(h2h_with_seeds['feature_columns'])} features")
    print(f"  ✓ NO SEEDS: {len(h2h_no_seeds['models'])} models, {len(h2h_no_seeds['feature_columns'])} features")
    
    return h2h_with_seeds, h2h_no_seeds


def load_prediction_data():
    """Load raw 2026 prediction features."""
    df = pd.read_csv(PREDICTION_DATA_PATH)
    return df


def load_public_picks():
    """
    Load ESPN public pick percentages for contrarian analysis.
    
    Returns DataFrame with columns:
        - team_name
        - champion_pct (public % picking team to win championship)
        - final4_pct (public % picking team to reach Final Four)
        - elite8_pct, sweet16_pct, round32_pct, round64_pct
    
    Returns None if file doesn't exist (allows graceful degradation).
    """
    public_picks_path = Path("../L2/data/bracketology/espn_public_picks_2026_clean.csv")
    
    if not public_picks_path.exists():
        print(f"  ⚠ Public picks file not found: {public_picks_path}")
        return None
    
    df = pd.read_csv(public_picks_path)
    return df


def construct_bracket(prediction_df):
    """Build tournament bracket from tournamentSeed."""
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
# H2H PREDICTION
# ============================================================================

def compute_h2h_features(team1, team2, prediction_df, source_columns):
    """
    Compute pct_diff features for a matchup.
    
    Uses L3's formula: (A - B) / average(A, B)
    This matches how the models were trained in L3/h2h/03_train_models.py
    """
    t1_row = prediction_df[prediction_df['Team'] == team1]
    t2_row = prediction_df[prediction_df['Team'] == team2]

    if len(t1_row) == 0 or len(t2_row) == 0:
        return None

    t1_vals = t1_row[source_columns].values[0].astype(float)
    t2_vals = t2_row[source_columns].values[0].astype(float)

    # L3's formula: (A - B) / ((A + B) / 2) = 2(A - B) / (A + B)
    avg = (t1_vals + t2_vals) / 2.0
    diffs = np.where(avg == 0, 0.0, (t1_vals - t2_vals) / avg)

    return diffs


def predict_h2h_matchup(team1, team2, prediction_df, h2h):
    """
    Predict P(team1 beats team2) using H2H ensemble.
    
    CRITICAL: Matches L3's exact implementation:
    - Neural Network, Gaussian Naive Bayes, SVM use SCALED features
    - Random Forest, Gradient Boosting use RAW (unscaled) features
    """
    diffs = compute_h2h_features(team1, team2, prediction_df, h2h['source_columns'])
    if diffs is None:
        return 0.5, {'excluded': [], 'note': 'Unknown team'}

    # Keep both raw and scaled features
    X_raw = diffs.reshape(1, -1)
    X_scaled = h2h['scaler'].transform(X_raw)

    predictions = {}
    excluded = []

    # Models that need scaled features (matches L3)
    SCALED_MODELS = {'NN', 'GNB', 'SVM'}

    for key, model in h2h['models'].items():
        try:
            # Use scaled features only for NN, GNB, SVM (like L3)
            if key in SCALED_MODELS:
                X_model = X_scaled
            else:
                X_model = X_raw
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_model)[0, 1]
            else:
                pred = float(model.predict(X_model)[0])

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



def identify_seed_bias(bracket_df, elite8_with_seeds, elite8_no_seeds, threshold=0.02):
    """
    Compare WITH SEEDS vs NO SEEDS Elite 8 probabilities to identify seed bias.
    
    Positive bias: Committee seed helps team (potentially overseeded - FADE)
    Negative bias: Committee seed hurts team (potentially underseeded - VALUE)
    
    Args:
        bracket_df: Tournament teams with seeds
        elite8_with_seeds: Elite 8 predictions using WITH SEEDS models
        elite8_no_seeds: Elite 8 predictions using NO SEEDS models
        threshold: Minimum difference to flag as FADE/VALUE (default 3%)
    
    Returns:
        DataFrame with seed bias analysis sorted by bias magnitude
    """
    results = []
    
    # Create lookups for Elite 8 probabilities
    e8_with_lookup = dict(zip(elite8_with_seeds['Team'], elite8_with_seeds['Elite8_Probability']))
    e8_no_lookup = dict(zip(elite8_no_seeds['Team'], elite8_no_seeds['Elite8_Probability']))
    
    for _, row in bracket_df.iterrows():
        team = row['Team']
        seed = row['tournamentSeed']
        region = row['Region']
        
        # Get Elite 8 probabilities from both models
        e8_with = e8_with_lookup.get(team, 0)
        e8_no = e8_no_lookup.get(team, 0)
        
        # Seed bias = how much the seed helps/hurts compared to pure metrics
        seed_bias = e8_with - e8_no
        
        # Categorize
        if seed_bias > threshold:
            category = 'FADE'  # Seed helps them (potentially overseeded)
        elif seed_bias < -threshold:
            category = 'VALUE'  # Seed hurts them (potentially underseeded)
        else:
            category = 'NEUTRAL'
        
        results.append({
            'Team': team,
            'Seed': seed,
            'Region': region,
            'E8_With_Seeds': e8_with,
            'E8_No_Seeds': e8_no,
            'Seed_Bias': seed_bias,
            'Category': category
        })
    
    df = pd.DataFrame(results)
    return df.sort_values('Seed_Bias', ascending=False)



# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def simulate_tournament(bracket, prediction_df, h2h, n_sims=50000, 
                       convergence_check=10000, convergence_thresh=0.001,
                       n_trace=5):
    """
    Monte Carlo simulation with full round tracking.
    
    Args:
        n_trace: Number of sample sims to log for trace export (0 = none)
    
    Returns:
        round_probs: {team: {round: probability}}
        champion_probs: {team: probability}
        stats: simulation summary
        cache: matchup probability cache
        traces: list of sample simulation game logs (if n_trace > 0)
    """
    print(f"Running {n_sims:,} tournament simulations...")
    if convergence_check:
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
    round_counts = defaultdict(lambda: defaultdict(int))
    
    cache = {}
    total_matchups = 0
    gnb_exclusion_count = 0
    
    convergence_history = []
    last_champion_probs = None
    
    traces = []  # For sample bracket traces

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

        log_this_sim = (sim_idx < n_trace)
        sim_rounds = defaultdict(set)

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

                if log_this_sim:
                    traces.append({
                        'sim': sim_idx + 1,
                        'region': region,
                        'round': 'R64',
                        'team1': t1,
                        'team2': t2,
                        'p_team1': round(p, 4),
                        'winner': winner
                    })

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

                    if log_this_sim:
                        traces.append({
                            'sim': sim_idx + 1,
                            'region': region,
                            'round': 'R32',
                            'team1': t1,
                            'team2': t2,
                            'p_team1': round(p, 4),
                            'winner': winner
                        })

        # --- S16 ---
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

                    if log_this_sim:
                        traces.append({
                            'sim': sim_idx + 1,
                            'region': region,
                            'round': 'S16',
                            'team1': t1,
                            'team2': t2,
                            'p_team1': round(p, 4),
                            'winner': winner,
                            'note': 'winner makes Elite 8'
                        })

        # --- Elite 8 ---
        e8_winners = {}
        for region, winners in s16_winners.items():
            if len(winners) == 2:
                t1, t2 = winners[0], winners[1]
                sim_rounds['E8'].add(t1)
                sim_rounds['E8'].add(t2)
                
                p = get_prob(t1, t2)
                winner = t1 if np.random.random() < p else t2
                e8_winners[region] = winner

                if log_this_sim:
                    traces.append({
                        'sim': sim_idx + 1,
                        'region': region,
                        'round': 'E8',
                        'team1': t1,
                        'team2': t2,
                        'p_team1': round(p, 4),
                        'winner': winner,
                        'note': 'winner makes Final Four'
                    })

        # --- Final Four ---
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

                if log_this_sim:
                    traces.append({
                        'sim': sim_idx + 1,
                        'region': f'{r1} vs {r2}',
                        'round': 'FF',
                        'team1': t1,
                        'team2': t2,
                        'p_team1': round(p, 4),
                        'winner': winner,
                        'note': 'Final Four semifinal'
                    })

        # --- Championship ---
        champion = None
        if len(ff_winners) == 2:
            t1, t2 = ff_winners[0], ff_winners[1]
            sim_rounds['Championship'].add(t1)
            sim_rounds['Championship'].add(t2)
            
            p = get_prob(t1, t2)
            champion = t1 if np.random.random() < p else t2

            if log_this_sim:
                traces.append({
                    'sim': sim_idx + 1,
                    'region': 'Championship',
                    'round': 'Championship',
                    'team1': t1,
                    'team2': t2,
                    'p_team1': round(p, 4),
                    'winner': champion,
                    'note': 'CHAMPION'
                })

        # Per-sim summary
        if log_this_sim:
            e8_list = [t for region in s16_winners for t in s16_winners[region]]
            ff_list = [e8_winners[r] for r in ['East', 'South', 'West', 'Midwest'] if r in e8_winners]
            traces.append({
                'sim': sim_idx + 1,
                'region': '---',
                'round': 'SUMMARY',
                'team1': 'E8: ' + ', '.join(e8_list),
                'team2': 'FF: ' + ', '.join(ff_list),
                'p_team1': '---',
                'winner': champion or '?',
                'note': 'CHAMPION: ' + (champion or '?')
            })

        # Update round counts
        for round_name, teams in sim_rounds.items():
            for team in teams:
                round_counts[team][round_name] += 1

        # Convergence check
        if convergence_check and (sim_idx + 1) % convergence_check == 0:
            current_champion_probs = {
                team: round_counts[team].get('Championship', 0) / (sim_idx + 1)
                for team in round_counts
            }
            convergence_history.append({
                'sim': sim_idx + 1,
                'champion_probs': current_champion_probs.copy()
            })
            
            if last_champion_probs is not None:
                top_teams = sorted(current_champion_probs.items(), key=lambda x: -x[1])[:5]
                max_change = max(
                    abs(current_champion_probs.get(t, 0) - last_champion_probs.get(t, 0))
                    for t, _ in top_teams
                )
                
                if max_change < convergence_thresh:
                    print(f"\n  ✓ Converged at {sim_idx + 1:,} sims (max change: {max_change:.4f})")
                    n_sims = sim_idx + 1
                    break
            
            last_champion_probs = current_champion_probs.copy()

    # Convert to probabilities
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
        'n_simulations': n_sims,
        'total_matchups': total_matchups,
        'unique_matchups': len(cache) // 2,
        'gnb_exclusions': gnb_exclusion_count,
        'gnb_exclusion_rate': round(gnb_exclusion_count / max(len(cache) // 2, 1), 4),
        'converged': convergence_check and n_sims < N_SIMS_PRODUCTION,
        'convergence_history': convergence_history,
        'top_5_champions': {t: round(p, 4) for t, p in sorted(champion_probs.items(), key=lambda x: -x[1])[:5]}
    }

    print(f"  ✓ Complete")
    print(f"    - Total sims: {n_sims:,}")
    print(f"    - Unique matchups: {stats['unique_matchups']:,}")
    print(f"    - GNB exclusion rate: {stats['gnb_exclusion_rate']:.1%}")
    if n_trace > 0:
        print(f"    - Trace records logged: {len(traces)}")
    
    top_champs = sorted(champion_probs.items(), key=lambda x: -x[1])[:5]
    print(f"    - Top 5 champions:")
    for team, prob in top_champs:
        print(f"        {team}: {prob:.1%}")

    return round_probs, champion_probs, stats, cache, traces


# ============================================================================
# VALIDATION MODE
# ============================================================================

def analyze_agreement(elite8_direct, elite8_simulated):
    """Compare Elite 8 direct vs simulated probabilities."""
    df = elite8_direct.copy()
    df['P_Elite8_Simulated'] = df['Team'].map(elite8_simulated)
    df = df.dropna(subset=['P_Elite8_Direct', 'P_Elite8_Simulated'])

    pearson_corr, pearson_p = pearsonr(df['P_Elite8_Direct'], df['P_Elite8_Simulated'])
    spearman_corr, spearman_p = spearmanr(df['P_Elite8_Direct'], df['P_Elite8_Simulated'])

    rmse = np.sqrt(mean_squared_error(df['P_Elite8_Direct'], df['P_Elite8_Simulated']))
    mae = np.mean(np.abs(df['P_Elite8_Direct'] - df['P_Elite8_Simulated']))

    df['Abs_Difference'] = np.abs(df['P_Elite8_Direct'] - df['P_Elite8_Simulated'])
    df['Direct_Higher'] = df['P_Elite8_Direct'] > df['P_Elite8_Simulated']
    df['Significant_Disagreement'] = df['Abs_Difference'] > DISAGREEMENT_THRESHOLD

    df['P_Elite8_Ensemble'] = (df['P_Elite8_Direct'] + df['P_Elite8_Simulated']) / 2

    if pearson_corr >= HIGH_AGREEMENT_THRESHOLD:
        strategy = "HIGH_AGREEMENT"
        recommendation = f"Models agree strongly (r={pearson_corr:.3f}). H2H simulation is primary."
        w_h2h, w_e8 = 0.70, 0.30
    elif pearson_corr >= MODERATE_AGREEMENT_THRESHOLD:
        strategy = "MODERATE_AGREEMENT"
        recommendation = f"Moderate agreement (r={pearson_corr:.3f}). Blend 60% H2H + 40% Elite8."
        w_h2h, w_e8 = 0.60, 0.40
    else:
        strategy = "LOW_AGREEMENT"
        recommendation = f"Low agreement (r={pearson_corr:.3f}). Use H2H for brackets, Elite 8 for Calcutta separately."
        w_h2h, w_e8 = 0.50, 0.50

    summary = {
        'n_teams': len(df),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'rmse': float(rmse),
        'mae': float(mae),
        'max_absolute_difference': float(df['Abs_Difference'].max()),
        'median_absolute_difference': float(df['Abs_Difference'].median()),
        'teams_with_significant_disagreement': int(df['Significant_Disagreement'].sum()),
        'pct_teams_with_disagreement': float(df['Significant_Disagreement'].mean()),
        'integration_strategy': strategy,
        'recommendation': recommendation,
        'suggested_weight_h2h': w_h2h,
        'suggested_weight_elite8': w_e8
    }

    df['Avg_Probability'] = (df['P_Elite8_Direct'] + df['P_Elite8_Simulated']) / 2
    df = df.sort_values('Avg_Probability', ascending=False).reset_index(drop=True)

    return df, summary


def create_validation_visualizations(analysis_df, summary, sim_stats):
    """Create validation diagnostic plots."""
    print("Creating validation visualizations...")
    sns.set_style("whitegrid")

    # Scatter + disagreement histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red' if x else 'steelblue' for x in analysis_df['Significant_Disagreement']]
    axes[0].scatter(analysis_df['P_Elite8_Direct'], analysis_df['P_Elite8_Simulated'],
                    c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect agreement')
    axes[0].set_xlabel('Elite 8 Model (Direct)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('H2H Model (Simulated)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Model Agreement\nr = {summary["pearson_correlation"]:.3f}', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for _, row in analysis_df.nlargest(5, 'Abs_Difference').iterrows():
        axes[0].annotate(row['Team'], (row['P_Elite8_Direct'], row['P_Elite8_Simulated']),
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

    axes[1].hist(analysis_df['Abs_Difference'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(DISAGREEMENT_THRESHOLD, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({DISAGREEMENT_THRESHOLD:.0%})')
    axes[1].set_xlabel('Absolute Difference', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Disagreement Distribution\nMAE={summary["mae"]:.3f}', fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'validation_model_agreement.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ validation_model_agreement.png")
    plt.close()

    # Top 20 comparison
    top20 = analysis_df.head(20)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top20))
    w = 0.35

    ax.bar(x - w/2, top20['P_Elite8_Direct'], w, label='Elite 8 (Direct)', alpha=0.8, color='steelblue', edgecolor='black')
    ax.bar(x + w/2, top20['P_Elite8_Simulated'], w, label='H2H (Simulated)', alpha=0.8, color='coral', edgecolor='black')

    ax.set_xlabel('Team', fontsize=11, fontweight='bold')
    ax.set_ylabel('P(Elite 8)', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Elite 8 Candidates', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top20['Team'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'validation_top_teams.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ validation_top_teams.png")
    plt.close()


def run_validation_mode(bracket, bracket_df, prediction_df, elite8_df, h2h):
    """Run validation analysis."""
    print()
    print("=" * 70)
    print("                    VALIDATION MODE")
    print("=" * 70)
    print()

    # Quick simulation
    print("Running validation simulation...")
    print("-" * 70)
    
    round_probs, champion_probs, sim_stats, cache, traces = simulate_tournament(
        bracket, prediction_df, h2h,
        n_sims=N_SIMS_VALIDATE,
        convergence_check=None,  # No convergence in validation mode
        n_trace=5
    )
    print()

    # Elite 8 comparison
    print("Analyzing model agreement...")
    print("-" * 70)
    
    tournament_teams = set(bracket_df['Team'])
    elite8_direct = (
        elite8_df[elite8_df['Team'].isin(tournament_teams)]
        [['Team', 'Elite8_Probability']]
        .rename(columns={'Elite8_Probability': 'P_Elite8_Direct'})
        .merge(bracket_df[['Team', 'tournamentSeed', 'Region']], on='Team')
        .rename(columns={'tournamentSeed': 'Seed'})
    )

    elite8_simulated = {team: round_probs[team].get('E8', 0) for team in round_probs}
    
    analysis_df, summary = analyze_agreement(elite8_direct, elite8_simulated)
    
    print(f"  ✓ Pearson r={summary['pearson_correlation']:.4f}")
    print(f"  ✓ Strategy: {summary['integration_strategy']}")
    print(f"  ✓ Disagreements: {summary['teams_with_significant_disagreement']} teams")
    print()

    # Save outputs
    print("Saving validation outputs...")
    print("-" * 70)
    
    analysis_df.to_csv(OUTPUT_DIR / 'validation_disagreement_analysis.csv', index=False)
    print(f"  ✓ validation_disagreement_analysis.csv")
    
    with open(OUTPUT_DIR / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ validation_summary.json")
    
    with open(OUTPUT_DIR / 'validation_simulation_stats.json', 'w') as f:
        json.dump(sim_stats, f, indent=2)
    print(f"  ✓ validation_simulation_stats.json")
    
    if traces:
        traces_df = pd.DataFrame(traces)
        traces_df.to_csv(OUTPUT_DIR / 'validation_sample_traces.csv', index=False)
        print(f"  ✓ validation_sample_traces.csv")
    
    create_validation_visualizations(analysis_df, summary, sim_stats)
    print()

    # Print summary
    print("=" * 70)
    print("                    VALIDATION SUMMARY")
    print("=" * 70)
    print()
    print(f"  Correlation:   Pearson  {summary['pearson_correlation']:>6.4f}  (p={summary['pearson_p_value']:.2e})")
    print(f"                 Spearman {summary['spearman_correlation']:>6.4f}  (p={summary['spearman_p_value']:.2e})")
    print(f"  Error:         MAE      {summary['mae']:>6.4f}")
    print(f"                 RMSE     {summary['rmse']:>6.4f}")
    print(f"                 Max Δ    {summary['max_absolute_difference']:>6.4f}")
    print(f"  Disagreements: {summary['teams_with_significant_disagreement']} teams "
          f"({summary['pct_teams_with_disagreement']:.0%})")
    print()
    print(f"  Strategy: {summary['integration_strategy']}")
    print(f"  → {summary['recommendation']}")
    print()

    print("  TOP 10 DISAGREEMENTS")
    print("  " + "-" * 66)
    print(f"  {'Team':<18} {'Direct':<9} {'Simulated':<10} {'Δ':<8} {'Direction'}")
    print("  " + "-" * 66)
    for _, row in analysis_df.nlargest(10, 'Abs_Difference').iterrows():
        direction = "E8 ↑" if row['Direct_Higher'] else "H2H ↑"
        print(f"  {row['Team']:<18} {row['P_Elite8_Direct']:>6.1%}   "
              f"{row['P_Elite8_Simulated']:>6.1%}     {row['Abs_Difference']:>5.1%}   {direction}")
    print()
    print("=" * 70)


# ============================================================================
# PRODUCTION MODE - OPTIMIZATION
# ============================================================================

def optimize_chalk(bracket, round_probs):
    """Chalk strategy: always pick favorite."""
    picks = {}
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            p1 = round_probs.get(t1, {}).get('R32', 0)
            p2 = round_probs.get(t2, {}).get('R32', 0)
            winner = t1 if p1 >= p2 else t2
            picks[(t1, t2)] = winner
    return picks, "Chalk (always pick favorite)"


def optimize_expected_value(bracket, round_probs, bracket_df, scoring=SCORING):
    """
    Expected Value strategy: maximize expected points.
    
    For ESPN/Yahoo scoring (no upset bonuses):
    Expected Value = P(win) × round_points
    
    Since round_points is constant for all teams in R64, this simplifies to
    picking the team with highest win probability, which is identical to CHALK.
    
    NOTE: This strategy only differs from CHALK in Calcutta scoring (which has
    upset bonuses). For bracket pools, Expected Value = CHALK.
    """
    picks = {}
    
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            # Get win probabilities
            p1_win = round_probs.get(t1, {}).get('R32', 0)
            p2_win = round_probs.get(t2, {}).get('R32', 0)
            
            # Expected Value without upset bonuses:
            # EV = P(win) × round_points
            # Since round_points is constant, just pick higher probability
            ev1 = p1_win * scoring['R64']
            ev2 = p2_win * scoring['R64']
            
            # Pick team with higher expected value
            winner = t1 if ev1 >= ev2 else t2
            picks[(t1, t2)] = winner
    
    return picks, "Expected Value (maximize expected points)"


def optimize_elite8_focus(bracket, round_probs, elite8_probs_df):
    """Elite 8 Focus: optimize through Sweet 16 using direct probabilities."""
    picks = {}
    e8_lookup = dict(zip(elite8_probs_df['Team'], elite8_probs_df['Elite8_Probability']))
    
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            e8_prob1 = e8_lookup.get(t1, 0)
            e8_prob2 = e8_lookup.get(t2, 0)
            winner = t1 if e8_prob1 >= e8_prob2 else t2
            picks[(t1, t2)] = winner
    
    return picks, "Elite 8 Focus (optimize through Sweet 16)"


def calculate_bracket_expected_points(bracket_result, bracket_df, scoring=SCORING):
    """
    Calculate expected points for a bracket (ESPN/Yahoo scoring).
    
    Only calculates base points: probability × round points
    NO upset bonuses (those are only for Calcutta scoring in 02_calcutta_optimizer.py)
    
    Returns total expected points across all rounds.
    """
    total_points = 0
    
    rounds = [
        ('R64', scoring['R64']),
        ('R32', scoring['R32']),
        ('S16', scoring['S16']),
        ('E8', scoring['E8']),
        ('FF', scoring['FF']),
        ('Championship', scoring['Championship'])
    ]
    
    for round_name, round_points in rounds:
        round_data = bracket_result.get(round_name, {})
        
        if round_name == 'Championship':
            # Championship is a single game
            if round_data and 'winner' in round_data:
                prob = round_data['prob']
                total_points += prob * round_points
        
        elif round_name in ['E8', 'FF']:
            # E8 is dict by region, FF is dict by matchup
            for key, game in round_data.items():
                if isinstance(game, dict) and 'winner' in game:
                    prob = game['prob']
                    total_points += prob * round_points
        
        else:
            # R64, R32, S16 are dicts by region with lists of games
            for region, games in round_data.items():
                if isinstance(games, list):
                    for game in games:
                        prob = game['prob']
                        total_points += prob * round_points
    
    return total_points


def simulate_bracket_from_picks(bracket, picks, round_probs, cache, prediction_df, h2h):
    """
    Given R64 picks, simulate rest of bracket using ACTUAL H2H probabilities.
    
    FIXED: Previously used Monte Carlo round probabilities for all rounds after R64,
    causing all strategies to converge to the same champion. Now calculates actual
    head-to-head matchup probabilities for each game.
    """
    result = {'R64': {}, 'R32': {}, 'S16': {}, 'E8': {}, 'FF': {}, 'Championship': None}
    
    # R64 - Use the picks provided by the strategy
    r64_winners = {}
    for region, matchups in bracket.items():
        r64_winners[region] = []
        result['R64'][region] = []
        
        for t1, t2 in matchups:
            winner = picks.get((t1, t2), t1)
            r64_winners[region].append(winner)
            
            p = cache.get((t1, t2), 0.5)
            if winner == t2:
                p = 1.0 - p
            
            result['R64'][region].append({'team1': t1, 'team2': t2, 'winner': winner, 'prob': p})
    
    # R32 - Calculate actual H2H probabilities between R64 winners
    r32_winners = {}
    for region, winners in r64_winners.items():
        r32_winners[region] = []
        result['R32'][region] = []
        
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                t1, t2 = winners[i], winners[i+1]
                
                # Calculate actual H2H probability
                p_t1_wins, _ = predict_h2h_matchup(t1, t2, prediction_df, h2h)
                
                # Pick winner based on H2H probability
                winner = t1 if p_t1_wins >= 0.5 else t2
                r32_winners[region].append(winner)
                
                prob = p_t1_wins if winner == t1 else (1.0 - p_t1_wins)
                result['R32'][region].append({'team1': t1, 'team2': t2, 'winner': winner, 'prob': prob})
    
    # S16 - Calculate actual H2H probabilities between R32 winners
    s16_winners = {}
    for region, winners in r32_winners.items():
        s16_winners[region] = []
        result['S16'][region] = []
        
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                t1, t2 = winners[i], winners[i+1]
                
                # Calculate actual H2H probability
                p_t1_wins, _ = predict_h2h_matchup(t1, t2, prediction_df, h2h)
                
                # Pick winner based on H2H probability
                winner = t1 if p_t1_wins >= 0.5 else t2
                s16_winners[region].append(winner)
                
                prob = p_t1_wins if winner == t1 else (1.0 - p_t1_wins)
                result['S16'][region].append({'team1': t1, 'team2': t2, 'winner': winner, 'prob': prob})
    
    # E8 - Calculate actual H2H probabilities for Elite 8 games
    e8_winners = {}
    for region, winners in s16_winners.items():
        if len(winners) == 2:
            t1, t2 = winners[0], winners[1]
            
            # Calculate actual H2H probability
            p_t1_wins, _ = predict_h2h_matchup(t1, t2, prediction_df, h2h)
            
            # Pick winner based on H2H probability
            winner = t1 if p_t1_wins >= 0.5 else t2
            e8_winners[region] = winner
            
            prob = p_t1_wins if winner == t1 else (1.0 - p_t1_wins)
            result['E8'][region] = {'team1': t1, 'team2': t2, 'winner': winner, 'prob': prob}
    
    # FF - Calculate actual H2H probabilities for Final Four
    ff_matchups = [('East', 'South'), ('West', 'Midwest')]
    ff_winners = []
    result['FF'] = {}
    
    for r1, r2 in ff_matchups:
        if r1 in e8_winners and r2 in e8_winners:
            t1, t2 = e8_winners[r1], e8_winners[r2]
            
            # Calculate actual H2H probability
            p_t1_wins, _ = predict_h2h_matchup(t1, t2, prediction_df, h2h)

            # Pick winner based on H2H probability
            winner = t1 if p_t1_wins >= 0.5 else t2
            ff_winners.append(winner)
            
            prob = p_t1_wins if winner == t1 else (1.0 - p_t1_wins)
            result['FF'][f"{r1} vs {r2}"] = {'team1': t1, 'team2': t2, 'winner': winner, 'prob': prob}
    
    # Championship - Calculate actual H2H probability for final game
    if len(ff_winners) == 2:
        t1, t2 = ff_winners[0], ff_winners[1]
        
        # Calculate actual H2H probability
        p_t1_wins, _ = predict_h2h_matchup(t1, t2, prediction_df, h2h)
        
        # Pick winner based on H2H probability
        winner = t1 if p_t1_wins >= 0.5 else t2
        
        prob = p_t1_wins if winner == t1 else (1.0 - p_t1_wins)
        result['Championship'] = {'team1': t1, 'team2': t2, 'winner': winner, 'prob': prob}
    
    return result


def export_round_probabilities(round_probs, bracket_df, public_picks_df=None):
    """
    Export round probability table with optional public pick contrarian analysis.
    
    Args:
        round_probs: Dictionary of team probabilities by round
        bracket_df: DataFrame with team info (seed, region)
        public_picks_df: Optional DataFrame with ESPN public pick percentages
    
    Returns:
        DataFrame with probabilities and contrarian metrics
    """
    rows = []
    rounds = ['R64', 'R32', 'S16', 'E8', 'FF', 'Championship']
    
    for team in round_probs:
        row = {'Team': team}
        team_info = bracket_df[bracket_df['Team'] == team]
        if len(team_info) > 0:
            row['Seed'] = int(team_info.iloc[0]['tournamentSeed'])
            row['Region'] = team_info.iloc[0]['Region']
            row['teamsIndex'] = int(team_info.iloc[0]['teamsIndex'])  # Add teamsIndex for merge
        else:
            row['Seed'] = None
            row['Region'] = None
            row['teamsIndex'] = None
        
        for round_name in rounds:
            row[f'P_{round_name}'] = round_probs[team].get(round_name, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add public pick data if available
    if public_picks_df is not None:
        # Select columns from public picks (using teamsIndex for merge)
        public_subset = public_picks_df[[
            'teamsIndex',
            'round64_pct',
            'round32_pct',
            'sweet16_pct',
            'elite8_pct',
            'final4_pct',
            'champion_pct'
        ]].copy()
        
        # Rename columns
        public_subset.columns = [
            'teamsIndex',
            'Public_R64_Pct',
            'Public_R32_Pct',
            'Public_S16_Pct',
            'Public_E8_Pct',
            'Public_FF_Pct',
            'Public_Champion_Pct'
        ]
        
        # Merge on teamsIndex (not team name!)
        df = df.merge(public_subset, on='teamsIndex', how='left')
        
        # Calculate contrarian ratios for each round (Public / Model)
        # Lower ratio = public undervalues (contrarian opportunity)
        round_mappings = [
            ('R64', 'Public_R64_Pct', 'R64_Contrarian_Ratio'),
            ('R32', 'Public_R32_Pct', 'R32_Contrarian_Ratio'),
            ('S16', 'Public_S16_Pct', 'S16_Contrarian_Ratio'),
            ('E8', 'Public_E8_Pct', 'E8_Contrarian_Ratio'),
            ('FF', 'Public_FF_Pct', 'FF_Contrarian_Ratio'),
            ('Championship', 'Public_Champion_Pct', 'Champion_Contrarian_Ratio')
        ]
        
        for round_name, public_col, ratio_col in round_mappings:
            model_col = f'P_{round_name}'
            df[ratio_col] = df[public_col] / (df[model_col] * 100)
            df[ratio_col] = df[ratio_col].round(2)
        
        # Reorder columns to group model/public/ratio for each round together
        column_order = ['Team', 'Seed', 'Region']
        
        # Add columns for each round: Model %, Public %, Ratio
        rounds_in_order = [
            ('R64', 'Public_R64_Pct', 'R64_Contrarian_Ratio'),
            ('R32', 'Public_R32_Pct', 'R32_Contrarian_Ratio'),
            ('S16', 'Public_S16_Pct', 'S16_Contrarian_Ratio'),
            ('E8', 'Public_E8_Pct', 'E8_Contrarian_Ratio'),
            ('FF', 'Public_FF_Pct', 'FF_Contrarian_Ratio'),
            ('Championship', 'Public_Champion_Pct', 'Champion_Contrarian_Ratio')
        ]
        
        for round_name, public_col, ratio_col in rounds_in_order:
            column_order.append(f'P_{round_name}')
            column_order.append(public_col)
            column_order.append(ratio_col)
        
        df = df[column_order]
    
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
                elif isinstance(games, dict):
                    rows.append({
                        'Round': round_name,
                        'Region': region,
                        'Team1': games['team1'],
                        'Team2': games['team2'],
                        'Winner': games['winner'],
                        'Probability': round(games['prob'], 4)
                    })
    
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
    filename = f"bracket_{strategy_name.lower().replace(' ', '_')}.csv"
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
    
    if bracket_result.get('Championship'):
        champ = bracket_result['Championship']['winner']
        lines.append(f"🏆 CHAMPION: {champ} ({champion_prob:.1%})")
        lines.append("")
    
    for round_name in ['R64', 'R32', 'S16', 'E8', 'FF', 'Championship']:
        if round_name not in bracket_result:
            continue
        
        lines.append(f"{round_name}")
        lines.append("-" * 70)
        
        if round_name == 'Championship':
            game = bracket_result['Championship']
            loser = game['team2'] if game['winner'] == game['team1'] else game['team1']
            lines.append(f"  {game['winner']:<25} def. {loser:<25} ({game['prob']:.0%})")
        elif round_name == 'FF':
            for matchup_name, game in bracket_result['FF'].items():
                loser = game['team2'] if game['winner'] == game['team1'] else game['team1']
                lines.append(f"  {game['winner']:<25} def. {loser:<25} ({game['prob']:.0%})")
        else:
            for region in ['East', 'Midwest', 'South', 'West']:
                if region not in bracket_result[round_name]:
                    continue
                games = bracket_result[round_name][region]
                if isinstance(games, dict):
                    games = [games]
                for game in games:
                    loser = game['team2'] if game['winner'] == game['team1'] else game['team1']
                    lines.append(f"  {game['winner']:<25} def. {loser:<25} ({game['prob']:.0%})")
        lines.append("")
    
    lines.append("=" * 70)
    
    text = "\n".join(lines)
    filename = f"bracket_{strategy_name.lower().replace(' ', '_')}.txt"
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        f.write(text)
    return filepath


def run_production_mode(bracket, bracket_df, prediction_df, elite8_df, h2h, h2h_no_seeds=None, elite8_no_seeds=None):
    """Run bracket optimization with narrative output."""
    print()
    print("=" * 70)
    print("                    PRODUCTION MODE")
    print("=" * 70)
    print()

    # ========================================================================
    # STEP 1: Run simulations (quiet except progress)
    # ========================================================================
    print("Running Monte Carlo simulation...")
    print("-" * 70)
    
    round_probs, champion_probs, sim_stats, cache, _ = simulate_tournament(
        bracket, prediction_df, h2h,
        n_sims=N_SIMS_PRODUCTION,
        convergence_check=CONVERGENCE_CHECK,
        convergence_thresh=CONVERGENCE_THRESH,
        n_trace=0
    )
    print()

    # Load supporting data (quiet)
    public_picks_df = load_public_picks()
    
    # Calculate seed bias (quiet)
    seed_bias_df = None
    fade_candidates = pd.DataFrame()
    value_picks = pd.DataFrame()
    if h2h_no_seeds is not None and elite8_no_seeds is not None:
        seed_bias_df = identify_seed_bias(bracket_df, elite8_df, elite8_no_seeds)
        fade_candidates = seed_bias_df[seed_bias_df['Category'] == 'FADE'].head(5)
        value_picks = seed_bias_df[seed_bias_df['Category'] == 'VALUE'].head(5)
    
    # Calculate contrarian opportunities (quiet)
    contrarian_df = None
    contrarian_champ = pd.DataFrame()
    contrarian_ff = pd.DataFrame()
    if public_picks_df is not None:
        contrarian_df = pd.DataFrame()
        contrarian_df['Team'] = list(round_probs.keys())
        contrarian_df['P_Championship'] = [round_probs[t].get('Championship', 0) for t in contrarian_df['Team']]
        contrarian_df['P_FF'] = [round_probs[t].get('FF', 0) for t in contrarian_df['Team']]
        
        # Add teamsIndex for merge
        team_index_lookup = bracket_df.set_index('Team')['teamsIndex'].to_dict()
        contrarian_df['teamsIndex'] = contrarian_df['Team'].map(team_index_lookup)
        
        # Merge on teamsIndex
        public_subset = public_picks_df[['teamsIndex', 'champion_pct', 'final4_pct']].copy()
        public_subset.columns = ['teamsIndex', 'Public_Champion_Pct', 'Public_FF_Pct']
        contrarian_df = contrarian_df.merge(public_subset, on='teamsIndex', how='left')
        
        contrarian_df['Champion_Ratio'] = contrarian_df['Public_Champion_Pct'] / (contrarian_df['P_Championship'] * 100)
        contrarian_df['FF_Ratio'] = contrarian_df['Public_FF_Pct'] / (contrarian_df['P_FF'] * 100)
        
        contrarian_champ = contrarian_df[contrarian_df['P_Championship'] > 0.05].sort_values('Champion_Ratio').head(5)
        contrarian_ff = contrarian_df[contrarian_df['P_FF'] > 0.10].sort_values('FF_Ratio').head(5)
    
    # Generate brackets (quiet)
    strategies = [
        ('chalk', optimize_chalk),
        ('expected_value', optimize_expected_value),
        ('elite8_focus', optimize_elite8_focus)
    ]
    
    brackets = {}
    for strat_key, optimizer_fn in strategies:
        # Call optimizer with correct arguments
        if strat_key == 'chalk':
            picks, desc = optimizer_fn(bracket, round_probs)
        elif strat_key == 'expected_value':
            picks, desc = optimizer_fn(bracket, round_probs, bracket_df)
        elif strat_key == 'elite8_focus':
            picks, desc = optimizer_fn(bracket, round_probs, elite8_df)
        
        # Simulate bracket from picks
        bracket_result = simulate_bracket_from_picks(bracket, picks, round_probs, cache, prediction_df, h2h)
        champion = bracket_result['Championship']['winner'] if bracket_result.get('Championship') else None
        champ_prob = champion_probs.get(champion, 0) if champion else 0
        
        # Calculate expected points
        expected_points_espn = calculate_bracket_expected_points(bracket_result, bracket_df, SCORING_ESPN)
        expected_points_yahoo = calculate_bracket_expected_points(bracket_result, bracket_df, SCORING_YAHOO)
        
        brackets[strat_key] = {
            'description': desc,
            'result': bracket_result,
            'champion': champion,
            'champion_prob': champ_prob,
            'expected_points_espn': expected_points_espn,
            'expected_points_yahoo': expected_points_yahoo
        }
    
    # ========================================================================
    # STEP 2: STRATEGIC RECOMMENDATIONS (THE "SO WHAT")
    # ========================================================================
    print()
    print("=" * 70)
    print("              STRATEGIC RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Get top teams
    top_mc = sorted(champion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    bracket_winner = brackets['chalk']['champion']
    bracket_h2h_prob = brackets['chalk']['result']['Championship']['prob']  # H2H win probability
    
    print("YOUR BRACKET PICK")
    print("-" * 70)
    print(f"  Champion: {bracket_winner} ({bracket_h2h_prob*100:.1f}% to win when they meet opponents)")
    print()
    print(f"  Why: {bracket_winner} is the optimal pick because they win head-to-head")
    print(f"       matchups against likely opponents, even if another team has")
    print(f"       more overall paths to the championship.")
    print()
    
    # Monte Carlo vs Bracket explanation
    if top_mc[0][0] != bracket_winner:
        mc_leader = top_mc[0][0]
        mc_prob = top_mc[0][1]
        print(f"  Note: {mc_leader} has highest Monte Carlo probability ({mc_prob*100:.1f}%)")
        print(f"        because they have an easier path, but {bracket_winner} beats them")
        print(f"        head-to-head when they meet.")
    print()
    
    # Contrarian opportunities
    if len(contrarian_champ) > 0:
        print("CONTRARIAN EDGE (Public Severely Undervalues)")
        print("-" * 70)
        
        for idx, (_, row) in enumerate(contrarian_champ.head(3).iterrows(), 1):
            print(f"  {idx}. {row['Team']}")
            print(f"     Model: {row['P_Championship']*100:>5.1f}% champion | Public picks: {row['Public_Champion_Pct']:>5.1f}% (Ratio: {row['Champion_Ratio']:.2f})")
            
            # Add seed bias context if available
            if seed_bias_df is not None:
                team_bias = seed_bias_df[seed_bias_df['Team'] == row['Team']]
                if len(team_bias) > 0:
                    bias_val = team_bias.iloc[0]['Seed_Bias']
                    if abs(bias_val) > 0.02:
                        bias_type = "overseeded" if bias_val > 0 else "underseeded"
                        print(f"     Seed bias: {bias_type} ({bias_val*100:+.1f}% Elite 8)")
            print()
        
        print(f"  Strategy: Public severely undervalues these teams. Pick them in")
        print(f"            bracket pools for differentiation and edge over casual fans.")
        print()
    
    # Teams to avoid
    if len(fade_candidates) > 0:
        print("TEAMS TO AVOID (Overpriced Due to Seeding)")
        print("-" * 70)
        
        for idx, (_, row) in enumerate(fade_candidates.head(3).iterrows(), 1):
            print(f"  {idx}. {row['Team']} ({int(row['Seed'])}-seed)")
            print(f"     Gets +{row['Seed_Bias']*100:.1f}% Elite 8 boost from favorable seeding")
            print(f"     Market will overprice based on inflated probabilities")
            print()
        
        print(f"  Strategy: Don't overpay in Calcutta auctions. Some of their Elite 8")
        print(f"            probability comes from bracket luck, not pure team strength.")
        print()
    
    # ========================================================================
    # STEP 3: SUPPORTING ANALYSIS (THE "WHY")
    # ========================================================================
    print()
    print("=" * 70)
    print("               SUPPORTING ANALYSIS")
    print("=" * 70)
    print()
    
    # Contrarian details
    if len(contrarian_champ) > 0:
        print("Full Contrarian Champion List")
        print("-" * 70)
        for _, row in contrarian_champ.iterrows():
            print(f"  {row['Team']:<20} Model: {row['P_Championship']*100:>5.1f}%  Public: {row['Public_Champion_Pct']:>5.1f}%  Ratio: {row['Champion_Ratio']:.2f}")
        print()
    
    # Seed bias details
    if seed_bias_df is not None:
        print("Seed Bias Analysis (Committee vs Pure Metrics)")
        print("-" * 70)
        
        if len(fade_candidates) > 0:
            print("\n  FADE (Overseeded - Committee helps them):")
            for _, row in fade_candidates.iterrows():
                print(f"    {row['Team']:<20} ({row['Seed']:>2}): +{row['Seed_Bias']*100:>4.1f}% Elite 8 boost")
        
        if len(value_picks) > 0:
            print("\n  VALUE (Underseeded - Better than seed suggests):")
            for _, row in value_picks.iterrows():
                print(f"    {row['Team']:<20} ({row['Seed']:>2}): {row['Seed_Bias']*100:>5.1f}% Elite 8 penalty")
        print()
    
    # Monte Carlo details
    print("Monte Carlo Champion Probabilities")
    print("-" * 70)
    for idx, (team, prob) in enumerate(top_mc[:5], 1):
        print(f"  {idx}. {team:<20} {prob*100:>5.1f}%")
    print()
    
    # ========================================================================
    # STEP 4: SIMULATION SUMMARY (THE DATA)
    # ========================================================================
    print()
    print("=" * 70)
    print("                SIMULATION SUMMARY")
    print("=" * 70)
    print()
    
    print(f"  Simulations:      {sim_stats['n_simulations']:,}")
    print(f"  Converged:        {'Yes' if sim_stats['converged'] else 'No'}")
    print(f"  Unique matchups:  {sim_stats['unique_matchups']:,}")
    print()
    
    print("  TOP 10 CHAMPIONSHIP PROBABILITIES")
    print("  " + "-" * 110)
    if public_picks_df is not None:
        print(f"  {'Team':<20} {'Champion':<12} {'Champion':<12} {'Champion':<10} {'Elite 8':<12} {'Elite 8':<12}")
        print(f"  {'':<20} {'Model %':<12} {'Public %':<12} {'Ratio':<10} {'Model %':<12} {'Public %':<12}")
    else:
        print(f"  {'Team':<24} {'P(Champion)':<15} {'P(Elite 8)':<15}")
    print("  " + "-" * 110)
    
    top_10 = sorted(champion_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if public_picks_df is not None:
        # Create team name to teamsIndex lookup
        team_to_index = bracket_df.set_index('Team')['teamsIndex'].to_dict()
        
        # Create teamsIndex to public picks lookups
        public_champ_lookup = public_picks_df.set_index('teamsIndex')['champion_pct'].to_dict()
        public_e8_lookup = public_picks_df.set_index('teamsIndex')['elite8_pct'].to_dict()
        
        for team, prob in top_10:
            e8_prob = round_probs.get(team, {}).get('E8', 0)
            
            # Get teamsIndex for this team, then lookup public picks
            team_index = team_to_index.get(team)
            public_champ_pct = public_champ_lookup.get(team_index, 0) if team_index else 0
            public_e8_pct = public_e8_lookup.get(team_index, 0) if team_index else 0
            
            ratio = public_champ_pct / (prob * 100) if prob > 0 else 0
            print(f"  {team:<20} {prob*100:>6.1f}%      {public_champ_pct:>6.1f}%      {ratio:>5.2f}       {e8_prob*100:>6.1f}%      {public_e8_pct:>6.1f}%")
    else:
        for team, prob in top_10:
            e8_prob = round_probs.get(team, {}).get('E8', 0)
            print(f"  {team:<24} {prob*100:>6.1f}%         {e8_prob*100:>6.1f}%")
    print()
    
    print("  OPTIMAL BRACKETS")
    print("  " + "-" * 66)
    for strat_key in ['chalk', 'expected_value', 'elite8_focus']:
        data = brackets[strat_key]
        strat_display = {
            'chalk': 'CHALK (Always Pick Favorite)',
            'expected_value': 'EXPECTED VALUE (Maximize Points)',
            'elite8_focus': 'ELITE 8 FOCUS (Optimize Through Sweet 16)'
        }
        
        print(f"\n  {strat_display[strat_key]}")
        print(f"    Champion: {data['champion']} ({data['champion_prob']*100:.1f}%)")
        print(f"    ESPN Scoring:  {data['expected_points_espn']:.1f} pts (max 1,920)")
        print(f"    Yahoo Scoring: {data['expected_points_yahoo']:.1f} pts (max 192)")
    
    print()
    print("=" * 70)
    print()

    # ========================================================================
    # STEP 5: EXPORT FILES
    # ========================================================================
    print("Exporting outputs...")
    print("-" * 70)
    
    round_probs_df = export_round_probabilities(round_probs, bracket_df, public_picks_df)
    round_probs_path = OUTPUT_DIR / 'round_probabilities.csv'
    round_probs_df.to_csv(round_probs_path, index=False)
    print(f"  ✓ round_probabilities.csv ({len(round_probs_df)} teams)")
    
    # Export seed bias analysis if available
    if seed_bias_df is not None:
        seed_bias_path = OUTPUT_DIR / 'seed_bias_analysis.csv'
        seed_bias_df.to_csv(seed_bias_path, index=False)
        print(f"  ✓ seed_bias_analysis.csv (strategic intelligence)")
    
    for strat_key, bracket_data in brackets.items():
        csv_path = export_bracket_csv(bracket_data['result'], strat_key)
        print(f"  ✓ {csv_path.name}")
        
        txt_path = export_bracket_text(bracket_data['result'], bracket_data['description'], bracket_data['champion_prob'])
        print(f"  ✓ {txt_path.name}")
    
    summary = {
        'simulation': {
            'n_sims': sim_stats['n_simulations'],
            'converged': sim_stats['converged'],
            'unique_matchups': sim_stats['unique_matchups'],
            'gnb_exclusion_rate': sim_stats['gnb_exclusion_rate']
        },
        'top_10_champions': {
            team: round(prob, 4)
            for team, prob in sorted(champion_probs.items(), key=lambda x: -x[1])[:10]
        },
        'strategies': {
            strat_key: {
                'description': data['description'],
                'champion': data['champion'],
                'champion_probability': round(data['champion_prob'], 4),
                'expected_points_espn': round(data['expected_points_espn'], 2),
                'expected_points_yahoo': round(data['expected_points_yahoo'], 2)
            }
            for strat_key, data in brackets.items()
        }
    }
    
    summary_path = OUTPUT_DIR / 'simulation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ simulation_summary.json")
    
    print()
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tournament Simulator (Validation + Optimization)')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation mode (5K sims, correlation analysis)')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("           L4.01 — TOURNAMENT SIMULATOR")
    print("=" * 70)
    print()

    # Load data
    print("Loading data and models...")
    print("-" * 70)
    
    print("  Elite 8 predictions...")
    elite8_with_seeds, elite8_no_seeds = load_both_elite8_predictions()
    elite8_df = elite8_with_seeds  # Use WITH SEEDS for primary simulation
    print(f"    ✓ {len(elite8_df)} teams (WITH SEEDS + NO SEEDS loaded)")
    
    print("  H2H models and config...")
    h2h_with_seeds, h2h_no_seeds = load_both_h2h_models()
    h2h = h2h_with_seeds  # Use WITH SEEDS for primary simulation
    
    print("  Prediction data...")
    prediction_df = load_prediction_data()
    print(f"    ✓ {len(prediction_df)} teams, {len(prediction_df.columns)} columns")
    print()

    # Construct bracket
    print("Constructing tournament bracket...")
    print("-" * 70)
    
    bracket, bracket_df = construct_bracket(prediction_df)
    print(f"  ✓ {len(bracket_df)} tournament teams")
    print(f"  ✓ {len(bracket)} regions, {sum(len(m) for m in bracket.values())} R64 matchups")

    # Run appropriate mode
    if args.validate:
        run_validation_mode(bracket, bracket_df, prediction_df, elite8_df, h2h)
    else:
        run_production_mode(bracket, bracket_df, prediction_df, elite8_df, h2h, h2h_no_seeds, elite8_no_seeds)
    
    print()
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
