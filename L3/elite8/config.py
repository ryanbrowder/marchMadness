"""
L3 Elite 8 Model Configuration
Toggle USE_SEEDS to compare models with vs without tournamentSeed
"""

# ============================================================================
# SEED CONFIGURATION - TOGGLE THIS
# ============================================================================
USE_SEEDS = True  # Set to False for "pure metrics" model

# ============================================================================
# AUTO-GENERATED PATHS (DON'T EDIT)
# ============================================================================
from pathlib import Path

# Suffix for output directories
SUFFIX = "" if USE_SEEDS else "_no_seeds"

# Get absolute paths based on this config file's location
CONFIG_DIR = Path(__file__).parent  # L3/elite8/
L3_DIR = CONFIG_DIR.parent  # L3/
L2_DIR = L3_DIR.parent / 'L2'  # L2/ (sibling of L3)
PROJECT_ROOT = L3_DIR.parent  # marchMadness/

# Input/Output directories - YOUR DATA IS IN L3/data/ and L2/data/
RESULTS_DIR = L3_DIR / 'data' / 'trainingData'  # L3/data/trainingData/
PREDICT_DATA_FILE = L3_DIR / 'data' / 'predictionData' / 'predict_set_2026.csv'
TOURNAMENT_RESULTS_FILE = L2_DIR / 'data' / 'tournamentResults.csv'  # L2/data/

# Output directories (automatically add suffix)
OUTPUT_01 = CONFIG_DIR / f'outputs/01_feature_selection{SUFFIX}'
OUTPUT_02 = CONFIG_DIR / f'outputs/02_exploratory_models{SUFFIX}'
OUTPUT_03 = CONFIG_DIR / f'outputs/03_ensemble_models{SUFFIX}'
OUTPUT_04 = CONFIG_DIR / f'outputs/04_backtest{SUFFIX}'
OUTPUT_05 = CONFIG_DIR / f'outputs/05_2026_predictions{SUFFIX}'
OUTPUT_06 = CONFIG_DIR / f'outputs/06_tournament_indicator{SUFFIX}'

# Model configuration
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.90

# Display configuration
MODEL_TYPE = "With Seeds (Bracket-Aware)" if USE_SEEDS else "No Seeds (Pure Metrics)"

def print_config():
    """Print current configuration"""
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Include tournamentSeed: {USE_SEEDS}")
    print(f"Output Suffix: '{SUFFIX}'")
    print(f"Output Directory: outputs/*{SUFFIX}/")
    print(f"\nResolved Paths:")
    print(f"  L3 Directory: {L3_DIR}")
    print(f"  L2 Directory: {L2_DIR}")
    print(f"  Training Data: {RESULTS_DIR}")
    print(f"  Prediction Data: {PREDICT_DATA_FILE}")
    print(f"  Tournament Results: {TOURNAMENT_RESULTS_FILE}")
    print("="*80)

def get_excluded_columns():
    """Get list of columns to exclude from features"""
    base_exclude = ['Year', 'Team', 'Index', 'tournamentOutcome', 'elite8_flag']
    
    if not USE_SEEDS:
        base_exclude.append('tournamentSeed')
    
    return base_exclude
