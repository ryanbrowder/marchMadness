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
DATA PIPELINE (ONE-TIME SETUP)
================================================================================

These scripts have local paths and run independently of config.py:

STEP 1: Build Training Matchups
------------------------
Script: 01_build_training_matchups.py
Input:  - L2/data/srcbb/srcbb_analyze_L2.csv (tournament results)
        - L3/data/trainingData/training_set_long.csv (team features)
Output: outputs/01_build_training_matchups/training_matchups.csv
Action: Creates percentage differential features for all 46 metrics
Result: 1,071 games (2008-2025), 46 pct_diff features including tournamentSeed

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
Result: Identifies 20 strong features, 39 redundant pairs

Command:
$ python 02_feature_correlation_analysis.py

STEP 3: Multicollinearity Resolution
------------------------
Script: 02b_multicollinearity_resolver.py
Input:  outputs/02_feature_correlation/*.csv
Output: outputs/02_feature_correlation/selected_features.csv
Action: Resolves redundant pairs, keeps stronger predictor
Result: 28 features marked KEEP (including tournamentSeed), 18 marked DROP

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
KEY FINDINGS
================================================================================

Seed Impact on Predictive Power:
- Seeds add only +0.003 ROC-AUC (minimal, matches Elite 8 findings)
- Committee seeds based on same metrics models use
- Provides strategic differentiation, not predictive lift

Model Behavior Changes With Seeds:
NO SEEDS:  GB (0.945) > SVM (0.936) > RF (0.920) > NN (0.893)
WITH SEEDS: NN (0.940) > SVM (0.936) > RF (0.922) > GB (0.912)

Neural Network and SVM are "seed learners" - heavily weight tournamentSeed
Random Forest and Gradient Boosting mostly ignore it

Seed Boost Magnitude:
- Close matchups (Purdue vs Houston): +2.9% to projected higher seed
- Clearer gaps (Michigan vs Arizona): +6.6% to projected higher seed
- Typical range: 1-7% boost depending on projected seed differential

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
│   │   └── training_matchups.csv (1,071 games, 46 pct_diff features)
│   ├── 02_feature_correlation/
│   │   ├── selected_features.csv (28 KEEP, 18 DROP)
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
L4 APPLICATION LAYER
================================================================================

With trained models ready, proceed to L4 for decision applications:

  L4.01 — Tournament Simulator
           Runs 50,000 Monte Carlo simulations using H2H production models.
           Outputs round-by-round probabilities, seed bias analysis, and
           optimal bracket recommendations across three strategies.

  L4.02 — Calcutta Optimizer
           Combines Elite 8 + H2H probabilities to generate expected value
           rankings, value picks, and budget-constrained bidding recommendations.

See L4/README.txt for full documentation and execution instructions.

================================================================================
CONTACT & UPDATES
================================================================================

Pipeline Version: 2025-02-24
Author: Ryan Browder
System: March Madness Computron

For updates to production strategies or threshold tuning, edit L3/config.py
For model architecture changes, edit 03_train_models.py hyperparameters
For feature selection changes, edit 02b thresholds (currently REDUNDANCY_THRESHOLD = 0.85)

================================================================================
