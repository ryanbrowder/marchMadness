================================================================================
                      MARCH MADNESS COMPUTRON
                           ROOT README
================================================================================

Author:  Ryan Browder
Version: 2025-02-24
Data:    2008–2025 NCAA Tournament

================================================================================
WHAT THIS IS
================================================================================

A probabilistic forecasting system for the NCAA Tournament that models
tournament outcomes as sequences of binary probabilities rather than bracket
predictions. The system is designed for two primary economic applications:

  1. Bracket pool optimization (maximize expected pool payout)
  2. Calcutta auction strategy (identify mispriced teams, optimize bidding)

This is not a bracket picker. The goal is decision-calibrated output —
actionable recommendations grounded in expected value, not prediction accuracy.

================================================================================
OPERATING PRINCIPLES
================================================================================

THINK IN PROBABILITIES, NOT BRACKETS
  Every outcome is a probability. No single "correct" bracket exists.
  The system generates win likelihoods for every matchup and advancement
  probability for every round. Decisions flow from those numbers.

OPTIMIZE WHERE SKILL BEATS LUCK
  Round of 64 and 32 are high-variance and hard to predict. The championship
  is path-dependent. The Elite 8 is the sweet spot — where underlying team
  quality is most predictive and differentiation has the highest ROI.

SEEDS ADD MINIMAL PREDICTIVE VALUE — BUT MAXIMUM STRATEGIC VALUE
  Committee seeds improve model ROC-AUC by only ~0.003. But comparing
  predictions WITH and WITHOUT seeds reveals overseeded/underseeded teams —
  the core signal for identifying pool fades and Calcutta value opportunities.

BUILD FOR DECISIONS, NOT LEADERBOARDS
  Calibrated probabilities matter more than raw accuracy. A model that
  confidently predicts 70% when the true rate is 70% is more useful than
  one that's "right" more often but poorly calibrated.

================================================================================
ARCHITECTURE
================================================================================

Five pipeline layers, each feeding the next:

  L0  →  L1  →  L2  →  L3  →  L4
  Raw    Clean  Analysis  Models  Applications

L0: DATA COLLECTION
  Web scrapers for basketball analytics sources.
  Handles JavaScript-rendered tables and bot detection (Selenium).

L1: DATA CLEANING
  Source-specific transforms, type corrections, team name standardization.
  All sources mapped through teamsIndex.csv for consistent joins.

L2: ANALYSIS-READY DATASETS
  Joins cleaned sources into unified training and prediction datasets.
  Key transforms: rank inversion (lower rank → higher value), rank normalization
  to 0–364 scale (prevents model extrapolation across historical year ranges).
  Outputs: training_set_long.csv, training_set_rich.csv, predict_set_YYYY.csv

L3: MODEL TRAINING
  Two independent model pipelines:

    L3/elite8   — Predicts each team's probability of reaching the Elite 8.
                   Trained on tournament outcomes 2008–2025.
                   ROC-AUC: ~0.905

    L3/h2h      — Predicts head-to-head matchup win probability.
                   Uses percentage differential features across 46 metrics.
                   Dual-model strategy: NO SEEDS vs. WITH SEEDS.
                   ROC-AUC: ~0.944 (NO SEEDS), ~0.947 (WITH SEEDS)

  Both pipelines share L3/config.py (MODE, USE_SEEDS toggles).

L4: APPLICATIONS
  Two decision tools that consume L3 outputs:

    L4.01 — Tournament Simulator (bracket pool optimization)
             Monte Carlo simulation (50,000 runs), contrarian analysis
             vs. ESPN public pick percentages, three bracket strategies.

    L4.02 — Calcutta Optimizer
             Expected value calculations, portfolio optimization,
             budget-constrained bidding recommendations.

================================================================================
DATA SOURCES
================================================================================

Required (2008–2025):
  bartTorvik       — Team efficiency, tempo, luck
  kenPom           — Adjusted efficiency, strength of schedule
  espnBPI          — Basketball Power Index
  masseyComposite  — Composite of 50+ ranking systems

Optional (2016–2025):
  LRMCB            — Luke Rettig's composite rankings (if available)
  powerRank        — Power rankings

Tournament metadata (post-Selection Sunday):
  ESPN Bracketology — Committee seeds, regions, projected bracket
  ESPN Pick %       — Public bracket pick percentages (for contrarian analysis)

================================================================================
TWO-MODEL STRATEGY (CORE DIFFERENTIATION)
================================================================================

The system trains every model twice:

  NO SEEDS (USE_SEEDS=False)
    Pure basketball metrics. Predictions available before Selection Sunday.
    Represents what the model thinks independent of the committee's opinion.

  WITH SEEDS (USE_SEEDS=True)
    Adds tournamentSeed as a feature. Bracket-aware predictions.
    Reflects the committee's seeding signal layered on top of metrics.

The gap between these two models is the signal:

  WITH SEEDS >> NO SEEDS  →  Overseeded. Team's reputation exceeds metrics.
                              FADE in Calcutta. Contrarian pool opportunity.

  NO SEEDS >> WITH SEEDS  →  Underseeded. Metrics exceed committee placement.
                              VALUE in Calcutta. Safe pool pick with upside.

  Both models agree strongly  →  Consensus. Safe chalk pick.

This comparison is the foundation of the seed bias analysis in L4.

================================================================================
MODEL ARCHITECTURE (L3 SHARED)
================================================================================

Five base models per ensemble:
  1. Random Forest       (n_estimators=200, max_depth=10)
  2. Gradient Boosting   (n_estimators=200, learning_rate=0.05)
  3. Neural Network      (layers: 64→32→16, relu)
  4. Gaussian Naive Bayes
  5. SVM with RBF kernel (calibrated)

Feature scaling — applied conditionally by model type:
  Scaled (StandardScaler):  Neural Network, SVM, Gaussian Naive Bayes
  Raw features:             Random Forest, Gradient Boosting

Auto-exclusion:
  Models producing extreme predictions (<0.1% or >99.9%) are auto-excluded
  per prediction. Remaining models are re-weighted dynamically.
  Gaussian Naive Bayes is excluded ~75% of the time — expected behavior.

Ensemble strategies (configured per model in L3/config.py):
  Uniform, Performance-weighted, Emphasize Top 2,
  De-emphasize Worst, Squared Performance

Validated ensemble winners:
  H2H NO SEEDS:    De-emphasize Worst
  H2H WITH SEEDS:  Emphasize Top 2

================================================================================
FULL PIPELINE EXECUTION ORDER
================================================================================

Pre-Selection Sunday:
  1.  L0  — Scrape current season data (bartTorvik, kenPom, espnBPI, etc.)
  2.  L1  — Run source-specific transforms and cleaning
  3.  L2  — python create_training_sets_L2.py
             python create_predict_set_L2.py
  4.  L3  — Train Elite 8 models (validation → production)
             Train H2H models, 4 configurations:
               NO SEEDS validation  →  NO SEEDS production
               WITH SEEDS validation  →  WITH SEEDS production

Post-Selection Sunday:
  5.  L2  — Scrape ESPN public picks, add tournament seeds to predict set
  6.  L4  — python 01_tournament_simulator.py
             python 02_calcutta_optimizer.py

================================================================================
KEY FILES REFERENCE
================================================================================

Team name standardization:
  utils/teamsIndex.csv            — All team name variants, canonical mapping

Shared model config:
  L3/config.py                    — MODE, USE_SEEDS, ensemble strategies

Training data:
  L3/data/trainingData/training_set_long.csv   — 2008–2025, 4–5 sources
  L3/data/trainingData/training_set_rich.csv   — 2016–2025, 5–6 sources

Prediction data:
  L3/data/predictionData/predict_set_YYYY.csv  — Current season, all D1

Elite 8 predictions (L4 input):
  L3/elite8/outputs/04_YYYY_predictions/elite8_predictions_YYYY_long.csv
  L3/elite8/outputs/04_YYYY_predictions_no_seeds/elite8_predictions_YYYY_long.csv

H2H trained models (L4 input):
  L3/h2h/models/                  — NO SEEDS production
  L3/h2h/models_with_seeds/       — WITH SEEDS production

L4 outputs:
  L4/outputs/01_tournament_simulator/round_probabilities.csv
  L4/outputs/01_tournament_simulator/seed_bias_analysis.csv
  L4/outputs/02_calcutta_optimizer/calcutta_recommendations.csv

================================================================================
UPDATING FOR A NEW SEASON
================================================================================

Each March, the following need to be updated:

  1. Scrape new season data (L0)
  2. Verify teamsIndex.csv includes any new/renamed programs
  3. Update year references in L2 predict script (YEAR = YYYY)
  4. Retrain all L3 models on updated data (adds new tournament year)
  5. Update path references in L4 scripts (predict_set_YYYY.csv, etc.)
  6. Run L4 applications after Selection Sunday

No structural code changes should be required year-to-year if team names
are maintained and data sources remain available.

================================================================================
KNOWN ISSUES AND DESIGN DECISIONS
================================================================================

Feature scaling must be conditional
  Tree-based models (RF, GB) are invariant to feature scale — applying
  StandardScaler to them degrades interpretability with no benefit. Scaling
  is applied only to Neural Network, SVM, and Naive Bayes. This conditional
  logic must be maintained consistently between training and prediction.

Rank normalization is critical
  D1 team counts have grown over time (2008: ~319 teams, 2026: ~364 teams).
  Without normalizing all ranks to a 0–364 scale, models trained on historical
  data extrapolate to out-of-range values when predicting current seasons.
  Both training and prediction datasets must use the same normalization.

Gaussian Naive Bayes extreme predictions
  GNB frequently produces predictions below 0.1% or above 99.9% — this is
  known behavior and handled by the auto-exclusion system. The ensemble
  output should show "1 model(s) excluded" in most predictions. This is correct.

Production strategy weights
  There is a known minor bug where production mode does not always apply the
  exact validation winner's weights. Ensemble performance remains reasonable.
  Manually inspect ensemble_config.json if precise weight control is needed.

================================================================================
LAYER-LEVEL DOCUMENTATION
================================================================================

Detailed documentation for each layer lives alongside the code:

  L0/README.txt        — Scraper inventory, cadence, bot detection notes
  L1/README.txt        — Transform scripts, team name standardization
  L2/README.txt        — Data joining, rank transforms, toggle configuration
  L3/elite8/README.txt — Elite 8 model training, validation, predictions
  L3/h2h/README.txt    — H2H matchup model, dual-model strategy, seed impact
  L4/README.txt        — Tournament simulator and Calcutta optimizer

================================================================================
