"""
L3 Elite 8 Model Configuration
Toggle USE_SEEDS to compare models with vs without tournamentSeed
"""

# ============================================================================
# SEED CONFIGURATION - TOGGLE THIS
# ============================================================================
USE_SEEDS = True  # Set to False for "pure metrics" model
#USE_SEEDS = False  # Set to False for "pure metrics" mode

# ============================================================================
# ENSEMBLE MODE - TOGGLE THIS
# ============================================================================
MODE = 'production'  # 'validation' or 'production'
#MODE = 'validation'  # 'validation' or 'production'
# validation: train on 2008-2024, test on 2025 (for strategy selection)
# production: train on 2008-2025 (for 2026 predictions)

# ============================================================================
# PRODUCTION STRATEGIES (only used if MODE='production')
# ============================================================================
# H2H strategies - update after running validations
H2H_PRODUCTION_STRATEGY_NO_SEEDS = 'De-emphasize Worst'
H2H_PRODUCTION_STRATEGY_WITH_SEEDS = 'Emphasize Top 2'

# ============================================================================
# AUTO-GENERATED PATHS (DON'T EDIT)
# ============================================================================
from pathlib import Path

# Suffix for output directories
SUFFIX = "" if USE_SEEDS else "_no_seeds"  # Keep for Elite 8 backwards compatibility
ELITE8_SUFFIX = SUFFIX  # Alias for clarity
H2H_SUFFIX = "_with_seeds" if USE_SEEDS else ""

# Get absolute paths based on this config file's location
CONFIG_DIR = Path(__file__).parent  # L3/
L3_DIR = CONFIG_DIR  # L3/
L2_DIR = CONFIG_DIR.parent / 'L2'
PROJECT_ROOT = CONFIG_DIR.parent
ELITE8_DIR = CONFIG_DIR / 'elite8'
H2H_DIR = CONFIG_DIR / 'h2h'

# Input/Output directories
RESULTS_DIR = L3_DIR / 'data' / 'trainingData'
PREDICT_DATA_FILE = L3_DIR / 'data' / 'predictionData' / 'predict_set_2026.csv'
TOURNAMENT_RESULTS_FILE = L2_DIR / 'data' / 'tournamentResults.csv'

# Elite 8 Output directories
OUTPUT_01 = ELITE8_DIR / f'outputs/01_feature_selection{ELITE8_SUFFIX}'
OUTPUT_02 = ELITE8_DIR / f'outputs/02_ensemble_models{ELITE8_SUFFIX}'
OUTPUT_03 = ELITE8_DIR / f'outputs/03_backtest{ELITE8_SUFFIX}'
OUTPUT_04 = ELITE8_DIR / f'outputs/04_2026_predictions{ELITE8_SUFFIX}'
OUTPUT_05 = ELITE8_DIR / f'outputs/05_tournament_indicator{ELITE8_SUFFIX}'

# H2H Output directories
H2H_TRAINING_MATCHUPS = H2H_DIR / 'outputs/01_build_training_matchups/training_matchups.csv'
H2H_SELECTED_FEATURES = H2H_DIR / 'outputs/02_feature_correlation/selected_features.csv'

# H2H Model directories (with mode and seeds suffix)
mode_suffix = '_validation' if MODE == 'validation' else ''
H2H_MODELS_DIR = H2H_DIR / f'models{H2H_SUFFIX}{mode_suffix}'
H2H_OUTPUT_DIR = H2H_DIR / f'outputs/03_train_models{H2H_SUFFIX}{mode_suffix}'

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