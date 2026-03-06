================================================================================
L3 H2H PIPELINE - HEAD-TO-HEAD MATCHUP PREDICTION MODELS
================================================================================

PURPOSE:
Build dual-model system for predicting tournament matchup outcomes:
- Model A (NO SEEDS): Pure basketball metrics - predict before Selection Sunday
- Model B (WITH SEEDS): Metrics + committee seeds - bracket-aware predictions

OUTPUT:
Four trained model sets for Elite 8/Champion probability generation and 
seed impact analysis.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

L3/h2h uses shared configuration with L3/elite8 via L3/config.py:
- MODE: 'validation' or 'production'
- USE_SEEDS: True or False

Model naming convention:
- USE_SEEDS=False → models/ and models_validation/
- USE_SEEDS=True  → models_with_seeds/ and models_with_seeds_validation/

================================================================================
DATA SOURCE UPDATE (MARCH 2026)
================================================================================

Training data upgraded from training_set_long.csv to training_set_unified.csv:

BEFORE (LONG dataset):
- 2008-2025, 4 sources (bartTorvik, kenPom, espnBPI, masseyComposite)
- ~1,147 tournament teams
- 46 pct_diff features

AFTER (UNIFIED dataset):
- 2008-2025, 5 sources (added PowerRank with historical data)
- ~1,147 tournament teams (same coverage, richer features)
- 47 pct_diff features (added PowerRank, tournamentSeed)

KEY CHANGES:
1. PowerRank added: Continuous rating from Dr. Ed Feng, coverage 2008-2025
   - Correlation with target: 0.365 (strong predictor)
   - Resolution: DROPPED in 02b (redundant with kenpom_NetRtg at r=0.967)
   - Impact: Validates existing features, no change to final model

2. tournamentSeed extraction: Script 01 now extracts seeds from tournament
   results and merges into features for H2H differential calculations
   - Correlation with target: -0.418 (2nd strongest predictor)
   - Resolution: KEPT in 02b (strong signal, not redundant)
   - Impact: Properly enables dual-model architecture (NO SEEDS vs WITH SEEDS)

Result: Same 28 KEEP features, but tournamentSeed now properly included

================================================================================
DATA PIPELINE (ONE-TIME SETUP)
================================================================================

These scripts have local paths and run independently of config.py:

STEP 1: Build Training Matchups
------------------------
Script: 01_build_training_matchups.py
Input:  - L2/data/srcbb/srcbb_analyze_L2.csv (tournament results)
        - L3/data/trainingData/training_set_unified.csv (team features)
Output: outputs/01_build_training_matchups/training_matchups.csv
Action: Creates percentage differential features for all metrics
        Extracts tournamentSeed from results and adds to features
Result: 1,071 games (2008-2025), 47 pct_diff features
        Includes: PowerRank (new), tournamentSeed (extracted from results)

Note: training_set_unified.csv contains 5 sources (bartTorvik, kenPom, espnBPI,
      masseyComposite, powerRank) with historical PowerRank data back to 2008.
      tournamentSeed is NOT in the unified dataset (dropped to prevent data
      leakage in Elite 8 models), so Script 01 extracts it from tournament
      results and merges it into features for H2H differential calculations.

Command:
$ cd L3/h2h
$ python 01_build_training_matchups.py

STEP 2: Feature Correlation Analysis
------------------------
Script: 02_feature_correlation_analysis.py
Input:  outputs/01_build_training_matchups/training_matchups.csv
Output: - outputs/02_feature_correlation/feature_target_correlations.csv
        - outputs/02_feature_correlation/feature_intercorrelations.csv
        - outputs/02_feature_correlation/correlation_heatmap.png
Action: Analyzes predictive power and detects multicollinearity
Result: Identifies 21 strong features, 46 severe redundant pairs
        PowerRank correlation: 0.365 (strong)
        tournamentSeed correlation: -0.418 (2nd strongest predictor)

Command:
$ python 02_feature_correlation_analysis.py

STEP 3: Multicollinearity Resolution
------------------------
Script: 02b_multicollinearity_resolver.py
Input:  outputs/02_feature_correlation/*.csv
Output: outputs/02_feature_correlation/selected_features.csv
Action: Resolves redundant pairs, keeps stronger predictor
Result: 28 features marked KEEP (including tournamentSeed), 19 marked DROP
        PowerRank: DROPPED (redundant with kenpom_NetRtg at r=0.967)
        tournamentSeed: KEPT (strong signal, r=-0.418, not redundant)

Command:
$ python 02b_multicollinearity_resolver.py

================================================================================
MODEL TRAINING (FOUR CONFIGURATIONS)
================================================================================

Edit L3/config.py before each run:
- Lines 15-16: MODE and USE_SEEDS
- Lines 23-24: Production strategies (set after validation)

Configuration 1: NO SEEDS VALIDATION
------------------------
Purpose: Find best ensemble strategy for pure metrics model
Edit L3/config.py:
  MODE = 'validation'
  USE_SEEDS = False

Command:
$ python 03_train_models.py

Output:
- models_validation/ (27 features: all KEEP except tournamentSeed)
- outputs/03_train_models_validation/model_performance.csv
- Ensemble strategies tested: Uniform, Performance-weighted, Emphasize Top 2, 
  De-emphasize Worst, Squared Performance

Expected Results:
- ROC-AUC: ~0.944
- Winner: "Uniform" (tied) or "De-emphasize Worst" (best calibration)
- Top model: Gradient Boosting (0.945)

Configuration 2: WITH SEEDS VALIDATION
------------------------
Purpose: Find best ensemble strategy for bracket-aware model
Edit L3/config.py:
  MODE = 'validation'
  USE_SEEDS = True

Command:
$ python 03_train_models.py

Output:
- models_with_seeds_validation/ (28 features: all KEEP including tournamentSeed)
- outputs/03_train_models_with_seeds_validation/model_performance.csv

Expected Results:
- ROC-AUC: ~0.947 (+0.003 vs NO SEEDS)
- Winner: "Emphasize Top 2"
- Top model: Neural Network (0.940) - learned to leverage seeds
- Model rankings flip vs NO SEEDS (NN becomes best, GB drops)

Update L3/config.py Line 24:
  H2H_PRODUCTION_STRATEGY_WITH_SEEDS = 'Emphasize Top 2'

Configuration 3: NO SEEDS PRODUCTION
------------------------
Purpose: Final model for 2026 predictions (pure metrics)
Edit L3/config.py:
  MODE = 'production'
  USE_SEEDS = False

Command:
$ python 03_train_models.py

Output:
- models/ (27 features, trained on all 2008-2025 data)
- Uses "De-emphasize Worst" strategy from validation
- Ready for pre-Selection Sunday predictions

Configuration 4: WITH SEEDS PRODUCTION
------------------------
Purpose: Final model for 2026 predictions (bracket-aware)
Edit L3/config.py:
  MODE = 'production'
  USE_SEEDS = True

Command:
$ python 03_train_models.py

Output:
- models_with_seeds/ (28 features, trained on all 2008-2025 data)
- Uses "Emphasize Top 2" strategy from validation
- Ready for post-Selection Sunday predictions and seed impact analysis

================================================================================
MODEL ARCHITECTURE
================================================================================

Five Base Models:
1. Random Forest (n_estimators=200, max_depth=10)
2. Gradient Boosting (n_estimators=200, learning_rate=0.05)
3. Neural Network (layers: 64→32→16, relu activation)
4. Gaussian Naive Bayes (default)
5. SVM with RBF kernel (calibrated)

Ensemble Strategies:
- Uniform: Equal weights (20% each)
- Performance-weighted: Weighted by validation ROC-AUC
- Emphasize Top 2: Top two models get 35% and 30%, others split remaining
- De-emphasize Worst: Worst model gets 5%, others split 95% equally
- Squared Performance: ROC-AUC squared for weights (amplifies gaps)

Auto-Exclusion System:
- Models producing extreme predictions (<0.1% or >99.9%) auto-excluded
- Remaining models re-weighted dynamically
- Gaussian Naive Bayes typically excluded 73-78% of time
- Fallback weights pre-computed in validation for common exclusion scenarios

Feature Scaling:
- Neural Network, SVM, Naive Bayes: Scaled (StandardScaler)
- Random Forest, Gradient Boosting: Raw features (tree-based)

================================================================================
TESTING PREDICTIONS
================================================================================

Test individual matchup predictions using Script 04:

NO SEEDS Model:
$ python 04_predict_matchup.py --teamA 179 --teamB 11 --year 2026
(Michigan vs Arizona, pure metrics)

WITH SEEDS Model:
$ python 04_predict_matchup.py --teamA 179 --teamB 11 --year 2026 --models-dir models_with_seeds
(Michigan vs Arizona, with committee seeds)

Example Seed Impact (Michigan vs Arizona):
- NO SEEDS: 61.3% Michigan
- WITH SEEDS: 67.9% Michigan
- Seed boost: +6.6% for Michigan

Team Index Reference:
Find team IDs in: L3/data/predictionData/predict_set_2026.csv
Common teams: Michigan (179), Arizona (11), Purdue (248), Houston (119)

================================================================================
KEY FINDINGS (UPDATED MARCH 2026 - UNIFIED DATASET)
================================================================================

Seed Impact on Predictive Power:
- Seeds add only +0.0023 ROC-AUC (NO SEEDS: 0.9442, WITH SEEDS: 0.9465)
- Minimal predictive lift (committee seeds based on same metrics models use)
- Provides strategic differentiation, not performance improvement

Model Behavior Changes With Seeds:
NO SEEDS:  GB (0.945) > SVM (0.936) > RF (0.920) > NN (0.893)
WITH SEEDS: NN (0.940) > SVM (0.936) > RF (0.922) > GB (0.912)

Neural Network becomes best model WITH SEEDS (learns to leverage tournamentSeed)
Gradient Boosting becomes best model WITHOUT SEEDS (pure metrics)

PowerRank Integration:
- Added from Dr. Ed Feng's historical data (2008-2025 coverage)
- Strong correlation: 0.365 with game outcomes
- Redundant with kenpom_NetRtg (r=0.967) → dropped in multicollinearity resolution
- Validates existing feature set, confirms kenpom_NetRtg as primary composite signal

Seed Boost Magnitude:
- Close matchups (Purdue vs Houston): +2.9% to projected higher seed
- Clearer gaps (Michigan vs Arizona): +6.6% to projected higher seed  
- Duke vs Michigan (both 1-seeds): +8.1% to Duke (projected #1 overall)
- Typical range: 2-8% boost depending on projected seed differential

Strategic Applications:
- Consensus picks: Both models strongly agree → safe pool picks
- Seed-inflated: WITH SEEDS >> NO SEEDS → overseeded, fade in Calcutta
- Metrics-favored: NO SEEDS >> WITH SEEDS → underseeded, value opportunity

================================================================================
FILE STRUCTURE
================================================================================

L3/h2h/
├── 01_build_training_matchups.py
├── 02_feature_correlation_analysis.py
├── 02b_multicollinearity_resolver.py
├── 03_train_models.py (uses shared config)
├── 04_predict_matchup.py
│
├── outputs/
│   ├── 01_build_training_matchups/
│   │   └── training_matchups.csv (1,071 games, 47 pct_diff features)
│   ├── 02_feature_correlation/
│   │   ├── selected_features.csv (28 KEEP, 19 DROP)
│   │   └── correlation_heatmap.png
│   ├── 03_train_models_validation/ (NO SEEDS results)
│   └── 03_train_models_with_seeds_validation/ (WITH SEEDS results)
│
├── models_validation/ (NO SEEDS, 2008-2024 training)
├── models_with_seeds_validation/ (WITH SEEDS, 2008-2024 training)
├── models/ (NO SEEDS production, 2008-2025 training) ← FOR 2026 PREDICTIONS
└── models_with_seeds/ (WITH SEEDS production, 2008-2025 training) ← FOR 2026 PREDICTIONS

L3/config.py (shared with elite8):
- MODE: 'validation' or 'production'
- USE_SEEDS: True or False
- H2H_PRODUCTION_STRATEGY_NO_SEEDS = 'De-emphasize Worst'
- H2H_PRODUCTION_STRATEGY_WITH_SEEDS = 'Emphasize Top 2'

================================================================================
TROUBLESHOOTING
================================================================================

Issue: "File not found: selected_features.csv"
Fix: Run scripts 01 → 02 → 02b in sequence before 03

Issue: Wrong number of features (e.g., 28 when expecting 27)
Fix: Check USE_SEEDS setting in L3/config.py matches intended configuration

Issue: Models saved to wrong directory
Fix: Verify H2H_SUFFIX in config.py:
     H2H_SUFFIX = "_with_seeds" if USE_SEEDS else ""

Issue: GNB predictions always extreme
Fix: This is expected. Auto-exclusion system handles it. Check ensemble output
     shows "Models used: 4/5" and "1 model(s) excluded"

Issue: Production weights don't match validation winner
Fix: This is a known bug in production mode strategy application. Weights are
     still reasonable. Can manually edit ensemble_config.json if needed.

================================================================================
NEXT STEPS: L4 APPLICATION LAYER
================================================================================

With trained models, you can now build:

1. Bracket Simulator (Monte Carlo)
   - Load H2H production models
   - Simulate 10,000 tournaments
   - Output Elite 8 / Champion probabilities

2. Seed Impact Analyzer
   - Compare models/ vs models_with_seeds/ predictions
   - Identify consensus picks vs seed-dependent predictions
   - Find overseeded/underseeded teams

3. Calcutta Strategy Optimizer
   - Combine Elite 8 + H2H probabilities
   - Calculate expected values with budget constraints
   - Non-linear payoff optimization

4. Pool Bracket Generator
   - Given scoring rules, generate optimal bracket
   - Two-bracket hedge strategy (different champion paths)
   - Maximize P(winning pool)

================================================================================
CONTACT & UPDATES
================================================================================

================================================================================
AUTHOR
================================================================================

Ryan Browder
March Madness Computron

Major Update (March 2026):
- Upgraded to training_set_unified.csv (5 sources, PowerRank added 2008-2025)
- tournamentSeed now properly extracted and integrated for H2H differentials
- 47 pct_diff features → 28 KEEP features after multicollinearity resolution

For updates to production strategies or threshold tuning, edit L3/config.py
For model architecture changes, edit 03_train_models.py hyperparameters
For feature selection changes, edit 02b thresholds (currently REDUNDANCY_THRESHOLD = 0.85)

================================================================================
