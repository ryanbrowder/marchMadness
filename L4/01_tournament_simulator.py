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

# L3 output paths
ELITE8_PREDICTIONS_PATH = Path("../L3/elite8/outputs/05_2026_predictions/elite8_predictions_2026_long.csv")
H2H_MODEL_DIR            = Path("../L3/h2h/models")
PREDICTION_DATA_PATH     = Path("../L3/data/predictionData/predict_set_2026.csv")

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
SCORING = {
    'R64': 10,
    'R32': 20,
    'S16': 40,
    'E8':  80,
    'FF':  160,
    'Championship': 320
}

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
    """Compute 29 pct_diff features for a matchup."""
    t1_row = prediction_df[prediction_df['Team'] == team1]
    t2_row = prediction_df[prediction_df['Team'] == team2]

    if len(t1_row) == 0 or len(t2_row) == 0:
        return None

    t1_vals = t1_row[source_columns].values[0].astype(float)
    t2_vals = t2_row[source_columns].values[0].astype(float)

    # OLD (WRONG): denom = np.abs(t1_vals) + np.abs(t2_vals)
    # NEW (MATCHES TRAINING): denom = (np.abs(t1_vals) + np.abs(t2_vals)) / 2.0
    
    avg = (np.abs(t1_vals) + np.abs(t2_vals)) / 2.0
    diffs = np.where(avg == 0, 0.0, (t1_vals - t2_vals) / avg)

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
    """Expected Value strategy: maximize expected points."""
    picks = {}
    for region, matchups in bracket.items():
        for t1, t2 in matchups:
            p1_r32 = round_probs.get(t1, {}).get('R32', 0)
            p2_r32 = round_probs.get(t2, {}).get('R32', 0)
            ev1 = p1_r32 * scoring['R64']
            ev2 = p2_r32 * scoring['R64']
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


def simulate_bracket_from_picks(bracket, picks, round_probs, cache):
    """Given R64 picks, simulate rest of bracket using probabilities."""
    result = {'R64': {}, 'R32': {}, 'S16': {}, 'E8': {}, 'FF': {}, 'Championship': None}
    
    # R64
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
    
    # R32
    r32_winners = {}
    for region, winners in r64_winners.items():
        r32_winners[region] = []
        result['R32'][region] = []
        
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                t1, t2 = winners[i], winners[i+1]
                p1 = round_probs.get(t1, {}).get('S16', 0)
                p2 = round_probs.get(t2, {}).get('S16', 0)
                winner = t1 if p1 >= p2 else t2
                r32_winners[region].append(winner)
                
                p = cache.get((t1, t2), 0.5)
                if winner == t2:
                    p = 1.0 - p
                result['R32'][region].append({'team1': t1, 'team2': t2, 'winner': winner, 'prob': p})
    
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
                result['S16'][region].append({'team1': t1, 'team2': t2, 'winner': winner, 'prob': p})
    
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
            result['E8'][region] = {'team1': t1, 'team2': t2, 'winner': winner, 'prob': p}
    
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
            result['FF'][f"{r1} vs {r2}"] = {'team1': t1, 'team2': t2, 'winner': winner, 'prob': p}
    
    # Championship
    if len(ff_winners) == 2:
        t1, t2 = ff_winners[0], ff_winners[1]
        p1 = round_probs.get(t1, {}).get('Championship', 0)
        p2 = round_probs.get(t2, {}).get('Championship', 0)
        winner = t1 if p1 >= p2 else t2
        
        p = cache.get((t1, t2), 0.5)
        if winner == t2:
            p = 1.0 - p
        result['Championship'] = {'team1': t1, 'team2': t2, 'winner': winner, 'prob': p}
    
    return result


def export_round_probabilities(round_probs, bracket_df):
    """Export round probability table."""
    rows = []
    rounds = ['R64', 'R32', 'S16', 'E8', 'FF', 'Championship']
    
    for team in round_probs:
        row = {'Team': team}
        team_info = bracket_df[bracket_df['Team'] == team]
        if len(team_info) > 0:
            row['Seed'] = int(team_info.iloc[0]['tournamentSeed'])
            row['Region'] = team_info.iloc[0]['Region']
        else:
            row['Seed'] = None
            row['Region'] = None
        
        for round_name in rounds:
            row[f'P_{round_name}'] = round_probs[team].get(round_name, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
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


def run_production_mode(bracket, bracket_df, prediction_df, elite8_df, h2h):
    """Run bracket optimization."""
    print()
    print("=" * 70)
    print("                    PRODUCTION MODE")
    print("=" * 70)
    print()

    # Full simulation
    print("Running Monte Carlo simulation...")
    print("-" * 70)
    
    round_probs, champion_probs, sim_stats, cache, _ = simulate_tournament(
        bracket, prediction_df, h2h,
        n_sims=N_SIMS_PRODUCTION,
        convergence_check=CONVERGENCE_CHECK,
        convergence_thresh=CONVERGENCE_THRESH,
        n_trace=0  # No traces in production mode
    )
    print()

    # Generate optimal brackets
    print("Generating optimal brackets...")
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
        
        bracket_result = simulate_bracket_from_picks(bracket, picks, round_probs, cache)
        champion = bracket_result['Championship']['winner'] if bracket_result.get('Championship') else None
        champ_prob = champion_probs.get(champion, 0) if champion else 0
        
        brackets[strat_key] = {
            'description': desc,
            'result': bracket_result,
            'champion': champion,
            'champion_prob': champ_prob
        }
        
        print(f"    Champion: {champion} ({champ_prob:.1%})")
    
    print()

    # Export outputs
    print("Exporting outputs...")
    print("-" * 70)
    
    round_probs_df = export_round_probabilities(round_probs, bracket_df)
    round_probs_path = OUTPUT_DIR / 'round_probabilities.csv'
    round_probs_df.to_csv(round_probs_path, index=False)
    print(f"  ✓ round_probabilities.csv ({len(round_probs_df)} teams)")
    
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

    # Print summary
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
    print(f"  {'Team':<25} {'P(Champion)':<15} {'P(Elite 8)':<15}")
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
    elite8_df = load_elite8_predictions()
    print(f"    ✓ {len(elite8_df)} teams")
    
    print("  H2H models and config...")
    h2h = load_h2h()
    print(f"    ✓ {len(h2h['models'])} models, {len(h2h['feature_columns'])} features")
    
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
        run_production_mode(bracket, bracket_df, prediction_df, elite8_df, h2h)
    
    print()
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()