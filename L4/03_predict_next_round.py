"""
L4.03 - Live Tournament Round Predictor

Calculates win probabilities for upcoming round based on actual tournament results.
Use during tournament for betting/gambling decisions.

USAGE:
  python predict_next_round.py --results actual_results.csv --next-round R32
  python predict_next_round.py --results actual_results.csv --next-round S16

INPUT FORMAT (actual_results.csv):
  round,winner,loser
  R64,Duke,Central Arkansas
  R64,St Louis,Iowa
  ...

OUTPUT:
  - next_round_probabilities.txt (human-readable betting sheet)
  - next_round_probabilities.csv (Excel-friendly)
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (same as main simulator)
# ============================================================================

TEAMSINDEX_PATH = Path("../utils/teamsIndex.csv")
H2H_MODEL_DIR = Path("../L3/h2h/models_with_seeds")
PREDICTION_DATA_PATH = Path("../L3/data/predictionData/predict_set_2026.csv")
ESPN_BRACKET_PATH = Path("../L2/data/bracketology/espn_bracketology_2026.csv")
OUTPUT_DIR = Path("outputs/03_live_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default path for actual results (can be overridden with --results)
DEFAULT_RESULTS_PATH = Path("data/actual_results.csv")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

NAME_TO_KEY = {
    'Gradient Boosting': 'GB',
    'SVM': 'SVM',
    'Random Forest': 'RF',
    'Neural Network': 'NN',
    'Gaussian Naive Bayes': 'GNB'
}

H2H_MODEL_FILES = {
    'GB': 'gradient_boosting.pkl',
    'SVM': 'svm.pkl',
    'RF': 'random_forest.pkl',
    'NN': 'neural_network.pkl',
    'GNB': 'gaussian_naive_bayes.pkl'
}

ROUND_NAMES = {
    'R64': 'Round of 64',
    'R32': 'Round of 32',
    'S16': 'Sweet 16',
    'E8': 'Elite 8',
    'FF': 'Final Four',
    'Championship': 'Championship'
}

# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_h2h():
    """Load H2H models and config."""
    with open(H2H_MODEL_DIR / 'ensemble_config.json', 'r') as f:
        config = json.load(f)

    feature_columns = config['feature_columns']
    source_columns = [c.replace('pct_diff_', '') for c in feature_columns]
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
        'models': models,
        'scaler': scaler,
        'weights': weights,
        'fallback_weights': fallback_weights,
        'feature_columns': feature_columns,
        'source_columns': source_columns,
    }


def load_prediction_data():
    """Load raw 2026 prediction features."""
    df = pd.read_csv(PREDICTION_DATA_PATH)
    
    # Load Region from ESPN bracket
    try:
        regions_df = pd.read_csv(ESPN_BRACKET_PATH)[["Team Index", "Region"]]
        regions_df = regions_df.rename(columns={"Team Index": "Index"})
        df = df.merge(regions_df, on="Index", how="left")
    except:
        pass
    
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
    """Compute pct_diff features for a matchup."""
    t1_row = prediction_df[prediction_df['Team'] == team1]
    t2_row = prediction_df[prediction_df['Team'] == team2]

    if len(t1_row) == 0 or len(t2_row) == 0:
        return None

    t1_vals = t1_row[source_columns].values[0].astype(float)
    t2_vals = t2_row[source_columns].values[0].astype(float)

    avg = (t1_vals + t2_vals) / 2.0
    diffs = np.where(avg == 0, 0.0, (t1_vals - t2_vals) / avg)

    return diffs


def predict_h2h_matchup(team1, team2, prediction_df, h2h):
    """Predict P(team1 beats team2) using H2H ensemble."""
    diffs = compute_h2h_features(team1, team2, prediction_df, h2h['source_columns'])
    if diffs is None:
        return 0.5, {'excluded': [], 'note': 'Unknown team'}

    X_raw = diffs.reshape(1, -1)
    X_scaled = h2h['scaler'].transform(X_raw)

    predictions = {}
    excluded = []

    SCALED_MODELS = {'NN', 'GNB', 'SVM'}

    for key, model in h2h['models'].items():
        try:
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
    total_w = sum(active_w.values())

    if total_w == 0:
        return 0.5, {'excluded': excluded, 'note': 'Zero weight'}

    prob = sum(predictions[k] * active_w[k] for k in active_w) / total_w

    return prob, {'excluded': excluded}


# ============================================================================
# ACTUAL RESULTS PROCESSING
# ============================================================================

def load_actual_results(results_path):
    """
    Load actual tournament results from CSV.
    
    Expected format:
        round,Team IndexA,winner,Team IndexB,loser
        R64,76,Duke,45,Central Arkansas
        R64,265,St Louis,132,Iowa
        ...
    
    Returns:
        DataFrame with columns: round, Team IndexA, winner, Team IndexB, loser
    """
    df = pd.read_csv(results_path)
    
    # Validate columns
    required_cols = ['round', 'Team IndexA', 'winner', 'Team IndexB', 'loser']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Results CSV must have columns: {required_cols}")
    
    return df


def remove_losing_teams(prediction_df, actual_results):
    """
    Remove teams that lost from prediction data.
    
    Args:
        prediction_df: Full team prediction data (with Index column)
        actual_results: DataFrame with actual game results (with Team IndexB for losers)
    
    Returns:
        prediction_df with losing teams removed
    """
    # Get all losing team Indexes
    all_loser_indexes = actual_results['Team IndexB'].unique()
    
    # Remove losers by Index
    remaining_df = prediction_df[~prediction_df['Index'].isin(all_loser_indexes)].copy()
    
    print(f"  Removed {len(all_loser_indexes)} losing teams")
    print(f"  {len(remaining_df[remaining_df['tournamentSeed'].between(1, 16)])} teams remaining in tournament")
    
    return remaining_df


def determine_next_round_matchups(actual_results, bracket_df, next_round):
    """
    Determine next round's matchups based on actual results.
    
    Args:
        actual_results: DataFrame with actual game results (columns: round, Team IndexA, winner, Team IndexB, loser)
        bracket_df: Original bracket structure with Index, Team, tournamentSeed, Region
        next_round: Next round to predict ('R32', 'S16', 'E8', 'FF', 'Championship')
    
    Returns:
        List of dicts: [{'team1': str, 'team2': str, 'region': str, 'seed1': int, 'seed2': int}, ...]
    """
    matchups = []
    
    # Get all winners from completed rounds
    completed_rounds = {
        'R32': ['R64'],
        'S16': ['R64', 'R32'],
        'E8': ['R64', 'R32', 'S16'],
        'FF': ['R64', 'R32', 'S16', 'E8'],
        'Championship': ['R64', 'R32', 'S16', 'E8', 'FF']
    }
    
    if next_round not in completed_rounds:
        raise ValueError(f"Invalid next_round: {next_round}")
    
    # Get winners by round and region (using Index)
    winners_by_round_region = defaultdict(lambda: defaultdict(list))
    
    for _, row in actual_results.iterrows():
        round_name = row['round']
        winner_index = row['Team IndexA']
        
        # Get winner's info from bracket_df using Index
        winner_info = bracket_df[bracket_df['Index'] == winner_index]
        if len(winner_info) > 0:
            team_name = winner_info.iloc[0]['Team']
            region = winner_info.iloc[0]['Region']
            seed = winner_info.iloc[0]['tournamentSeed']
            
            winners_by_round_region[round_name][region].append({
                'index': winner_index,
                'team': team_name,
                'seed': seed
            })
    
    # Determine matchups based on next round
    if next_round in ['R32', 'S16', 'E8']:
        # Regional matchups
        for region in ['East', 'Midwest', 'South', 'West']:
            # Get previous round
            prev_round = {
                'R32': 'R64',
                'S16': 'R32',
                'E8': 'S16'
            }[next_round]
            
            winners = winners_by_round_region[prev_round][region]
            
            # Pair up winners (adjacent winners play each other)
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    matchups.append({
                        'team1': winners[i]['team'],
                        'team2': winners[i+1]['team'],
                        'seed1': winners[i]['seed'],
                        'seed2': winners[i+1]['seed'],
                        'region': region
                    })
    
    elif next_round == 'FF':
        # Final Four: East vs Midwest, South vs West
        ff_pairings = [('East', 'Midwest'), ('South', 'West')]
        
        for r1, r2 in ff_pairings:
            e8_winners_r1 = winners_by_round_region['E8'][r1]
            e8_winners_r2 = winners_by_round_region['E8'][r2]
            
            if len(e8_winners_r1) > 0 and len(e8_winners_r2) > 0:
                matchups.append({
                    'team1': e8_winners_r1[0]['team'],
                    'team2': e8_winners_r2[0]['team'],
                    'seed1': e8_winners_r1[0]['seed'],
                    'seed2': e8_winners_r2[0]['seed'],
                    'region': f'{r1} vs {r2}'
                })
    
    elif next_round == 'Championship':
        # Championship: FF winners
        ff_winners = winners_by_round_region['FF']
        all_ff_winners = []
        
        for region in ff_winners.values():
            all_ff_winners.extend(region)
        
        if len(all_ff_winners) == 2:
            matchups.append({
                'team1': all_ff_winners[0]['team'],
                'team2': all_ff_winners[1]['team'],
                'seed1': all_ff_winners[0]['seed'],
                'seed2': all_ff_winners[1]['seed'],
                'region': 'National Championship'
            })
    
    return matchups


# ============================================================================
# PREDICTION AND EXPORT
# ============================================================================

def predict_next_round_probabilities(matchups, prediction_df, h2h):
    """
    Calculate H2H probabilities for next round matchups.
    
    Args:
        matchups: List of matchup dicts
        prediction_df: Remaining team data (losers removed)
        h2h: H2H model dictionary
    
    Returns:
        List of matchup dicts with probabilities added
    """
    results = []
    
    for matchup in matchups:
        team1 = matchup['team1']
        team2 = matchup['team2']
        
        # Calculate H2H probability
        prob, info = predict_h2h_matchup(team1, team2, prediction_df, h2h)
        
        # Determine favorite
        favorite = team1 if prob > 0.5 else team2
        favorite_prob = max(prob, 1 - prob)
        
        results.append({
            'team1': team1,
            'team2': team2,
            'seed1': matchup['seed1'],
            'seed2': matchup['seed2'],
            'region': matchup['region'],
            'team1_prob': prob,
            'team2_prob': 1 - prob,
            'favorite': favorite,
            'favorite_prob': favorite_prob,
            'confidence': classify_confidence(favorite_prob)
        })
    
    return results


def classify_confidence(prob):
    """Classify betting confidence based on probability."""
    if prob >= 0.90:
        return 'LOCK'
    elif prob >= 0.70:
        return 'STRONG'
    elif prob >= 0.60:
        return 'LEAN'
    elif prob >= 0.55:
        return 'SLIGHT'
    else:
        return 'TOSS-UP'


def export_betting_sheet(predictions, next_round, output_dir):
    """
    Export predictions in betting-friendly format.
    
    Creates:
        - TXT: Human-readable betting sheet
        - CSV: Excel-friendly data
    """
    txt_lines = []
    
    # Header
    txt_lines.append("=" * 80)
    txt_lines.append(f"          {ROUND_NAMES[next_round].upper()} - LIVE PROBABILITIES")
    txt_lines.append("=" * 80)
    txt_lines.append("")
    txt_lines.append("Based on actual tournament results")
    txt_lines.append("")
    
    # Group by region
    by_region = defaultdict(list)
    for pred in predictions:
        by_region[pred['region']].append(pred)
    
    # Show all matchups
    txt_lines.append("ALL MATCHUPS")
    txt_lines.append("-" * 80)
    txt_lines.append("")
    
    for region in sorted(by_region.keys()):
        txt_lines.append(f"  {region}:")
        
        for pred in by_region[region]:
            txt_lines.append(f"    ({pred['seed1']}) {pred['team1']} vs ({pred['seed2']}) {pred['team2']}")
            txt_lines.append(f"      → {pred['favorite']} {pred['favorite_prob']*100:.1f}% [{pred['confidence']}]")
        
        txt_lines.append("")
    
    txt_lines.append("")
    
    # Betting tiers
    txt_lines.append("=" * 80)
    txt_lines.append("                    BETTING TIERS")
    txt_lines.append("=" * 80)
    txt_lines.append("")
    
    # LOCKS (>90%)
    locks = [p for p in predictions if p['confidence'] == 'LOCK']
    txt_lines.append(f"LOCKS (>90% - Bet Heavy) [{len(locks)} games]")
    txt_lines.append("-" * 80)
    if locks:
        for pred in locks:
            txt_lines.append(f"  {pred['favorite']} vs {pred['team2'] if pred['favorite'] == pred['team1'] else pred['team1']}")
            txt_lines.append(f"    → {pred['favorite']} {pred['favorite_prob']*100:.1f}%")
            txt_lines.append(f"    - Model gives {pred['favorite']} {int(pred['favorite_prob']*100)} of 100 wins")
            txt_lines.append("")
    else:
        txt_lines.append("  None")
        txt_lines.append("")
    
    # STRONG (70-90%)
    strong = [p for p in predictions if p['confidence'] == 'STRONG']
    txt_lines.append(f"STRONG FAVORITES (70-90% - Bet Moderate) [{len(strong)} games]")
    txt_lines.append("-" * 80)
    if strong:
        for pred in strong:
            loser = pred['team2'] if pred['favorite'] == pred['team1'] else pred['team1']
            txt_lines.append(f"  {pred['favorite']} vs {loser}")
            txt_lines.append(f"    → {pred['favorite']} {pred['favorite_prob']*100:.1f}%")
            txt_lines.append(f"    - Good betting opportunity with solid edge")
            txt_lines.append("")
    else:
        txt_lines.append("  None")
        txt_lines.append("")
    
    # LEANS (60-70%)
    leans = [p for p in predictions if p['confidence'] == 'LEAN']
    txt_lines.append(f"LEANS (60-70% - Small Bet) [{len(leans)} games]")
    txt_lines.append("-" * 80)
    if leans:
        for pred in leans:
            loser = pred['team2'] if pred['favorite'] == pred['team1'] else pred['team1']
            txt_lines.append(f"  {pred['favorite']} vs {loser}")
            txt_lines.append(f"    → {pred['favorite']} {pred['favorite_prob']*100:.1f}%")
            txt_lines.append(f"    - Modest edge, small bet recommended")
            txt_lines.append("")
    else:
        txt_lines.append("  None")
        txt_lines.append("")
    
    # SLIGHT EDGES (55-60%)
    slight = [p for p in predictions if p['confidence'] == 'SLIGHT']
    txt_lines.append(f"SLIGHT EDGES (55-60% - Minimal Bet or Pass) [{len(slight)} games]")
    txt_lines.append("-" * 80)
    if slight:
        for pred in slight:
            loser = pred['team2'] if pred['favorite'] == pred['team1'] else pred['team1']
            txt_lines.append(f"  {pred['favorite']} vs {loser}")
            txt_lines.append(f"    → {pred['favorite']} {pred['favorite_prob']*100:.1f}%")
            txt_lines.append(f"    - Very small edge, consider passing")
            txt_lines.append("")
    else:
        txt_lines.append("  None")
        txt_lines.append("")
    
    # TOSS-UPS (<55%)
    tossups = [p for p in predictions if p['confidence'] == 'TOSS-UP']
    txt_lines.append(f"TOSS-UPS (<55% - Avoid or Hedge) [{len(tossups)} games]")
    txt_lines.append("-" * 80)
    if tossups:
        for pred in tossups:
            txt_lines.append(f"  {pred['team1']} vs {pred['team2']}")
            txt_lines.append(f"    → Near 50/50 ({pred['team1_prob']*100:.1f}% vs {pred['team2_prob']*100:.1f}%)")
            txt_lines.append(f"    - No clear edge, avoid betting or bet underdog for value")
            txt_lines.append("")
    else:
        txt_lines.append("  None")
        txt_lines.append("")
    
    txt_lines.append("=" * 80)
    txt_lines.append("")
    txt_lines.append("NOTE: Model probabilities ≠ betting odds.")
    txt_lines.append("      Use as ONE input to betting decisions.")
    txt_lines.append("      Consider injuries, matchups, and market odds.")
    txt_lines.append("")
    txt_lines.append("=" * 80)
    
    # Export TXT
    txt_path = output_dir / f'{next_round.lower()}_probabilities.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))
    
    # Export CSV
    csv_data = []
    for pred in predictions:
        csv_data.append({
            'Region': pred['region'],
            'Team1': pred['team1'],
            'Seed1': pred['seed1'],
            'Team2': pred['team2'],
            'Seed2': pred['seed2'],
            'Team1_Prob': round(pred['team1_prob'], 4),
            'Team2_Prob': round(pred['team2_prob'], 4),
            'Favorite': pred['favorite'],
            'Favorite_Prob': round(pred['favorite_prob'], 4),
            'Confidence': pred['confidence']
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / f'{next_round.lower()}_probabilities.csv'
    csv_df.to_csv(csv_path, index=False)
    
    return txt_path, csv_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predict next tournament round probabilities')
    parser.add_argument('--results', default=str(DEFAULT_RESULTS_PATH),
                       help=f'Path to actual_results.csv (default: {DEFAULT_RESULTS_PATH})')
    parser.add_argument('--next-round', required=True, choices=['R32', 'S16', 'E8', 'FF', 'Championship'],
                       help='Next round to predict')
    args = parser.parse_args()
    
    print()
    print("=" * 80)
    print("           L4.03 — LIVE TOURNAMENT ROUND PREDICTOR")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data and models...")
    print("-" * 80)
    
    h2h = load_h2h()
    print(f"  ✓ H2H models loaded ({len(h2h['models'])} models)")
    
    prediction_df = load_prediction_data()
    print(f"  ✓ Prediction data loaded ({len(prediction_df)} teams)")
    
    bracket, bracket_df = construct_bracket(prediction_df)
    print(f"  ✓ Original bracket constructed ({len(bracket_df)} tournament teams)")
    print()
    
    # Load actual results
    print("Processing actual results...")
    print("-" * 80)
    
    actual_results = load_actual_results(args.results)
    print(f"  ✓ Loaded {len(actual_results)} completed games")
    
    # Show completed rounds
    completed_rounds = actual_results['round'].value_counts().sort_index()
    for round_name, count in completed_rounds.items():
        print(f"    - {round_name}: {count} games")
    print()
    
    # Remove losing teams
    print("Updating remaining teams...")
    print("-" * 80)
    
    remaining_df = remove_losing_teams(prediction_df, actual_results)
    print()
    
    # Determine next round matchups
    print(f"Determining {ROUND_NAMES[args.next_round]} matchups...")
    print("-" * 80)
    
    matchups = determine_next_round_matchups(actual_results, bracket_df, args.next_round)
    print(f"  ✓ {len(matchups)} matchups determined")
    print()
    
    # Calculate probabilities
    print("Calculating probabilities...")
    print("-" * 80)
    
    predictions = predict_next_round_probabilities(matchups, remaining_df, h2h)
    print(f"  ✓ Probabilities calculated for {len(predictions)} games")
    print()
    
    # Export
    print("Exporting betting sheet...")
    print("-" * 80)
    
    txt_path, csv_path = export_betting_sheet(predictions, args.next_round, OUTPUT_DIR)
    print(f"  ✓ {txt_path.name}")
    print(f"  ✓ {csv_path.name}")
    print()
    
    # Summary
    print("=" * 80)
    print(f"                    {ROUND_NAMES[args.next_round].upper()} SUMMARY")
    print("=" * 80)
    print()
    
    confidence_counts = defaultdict(int)
    for pred in predictions:
        confidence_counts[pred['confidence']] += 1
    
    print(f"  Total games: {len(predictions)}")
    print()
    print("  Confidence breakdown:")
    for tier in ['LOCK', 'STRONG', 'LEAN', 'SLIGHT', 'TOSS-UP']:
        count = confidence_counts[tier]
        if count > 0:
            print(f"    {tier:<10} {count} games")
    print()
    
    print(f"  Betting sheet saved to: {OUTPUT_DIR}/")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
