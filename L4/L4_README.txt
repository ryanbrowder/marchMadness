================================================================================
                    L4 - APPLICATIONS LAYER
================================================================================

OVERVIEW
------------------------------------------------------------------------
L4 contains two production applications that use L3 model predictions for
decision-making in March Madness contexts:

  01_tournament_simulator.py  - Bracket pool optimization
  02_calcutta_optimizer.py    - Auction bidding strategy

Both applications are designed to run on Selection Sunday (or after) when
the tournament bracket is finalized.


================================================================================
                    PREREQUISITES
================================================================================

L4 applications require outputs from earlier pipeline layers:

FROM L3 (MODELS):
  ✓ Elite 8 predictions (WITH SEEDS)
    → L3/elite8/outputs/04_2026_predictions/elite8_predictions_2026_long.csv
  
  ✓ Elite 8 predictions (NO SEEDS - for seed bias analysis)
    → L3/elite8/outputs/04_2026_predictions_no_seeds/elite8_predictions_2026_long.csv
  
  ✓ H2H models (WITH SEEDS)
    → L3/h2h/models_with_seeds/
  
  ✓ H2H models (NO SEEDS - for seed bias analysis)
    → L3/h2h/models/
  
  ✓ Prediction features (2026 tournament teams)
    → L3/data/predictionData/predict_set_2026.csv

FROM L2 (DATA):
  ✓ ESPN public pick percentages
    → L2/data/bracketology/espn_public_picks_2026_clean.csv
  
  ✓ Historical Calcutta auction data (optional, for L4.02)
    → L2/data/calcutta/auction_history.csv

FROM UTILS:
  ✓ Team name standardization
    → utils/teamsIndex.csv


================================================================================
              L4.01 - TOURNAMENT SIMULATOR
================================================================================

PURPOSE:
  Generate optimal brackets for bracket pools using probabilistic simulation.
  Identify contrarian opportunities where public severely undervalues teams.

WHEN TO RUN:
  After Selection Sunday when tournament bracket is announced.

HOW TO RUN:
  cd L4
  python 01_tournament_simulator.py

WHAT IT DOES:
  1. Loads L3 model predictions (Elite 8 + H2H)
  2. Runs 50,000 Monte Carlo tournament simulations
  3. Calculates round-by-round advancement probabilities
  4. Compares model predictions to ESPN public pick percentages
  5. Identifies seed bias (overseeded vs underseeded teams)
  6. Generates optimal brackets for different strategies
  7. Exports all results to outputs/01_tournament_simulator/

STRATEGIES GENERATED:
  • CHALK - Always pick the favorite (highest probability)
  • EXPECTED VALUE - Maximize expected scoring points
  • ELITE 8 FOCUS - Optimize through Sweet 16 (skill vs variance sweet spot)

OUTPUTS:
  round_probabilities.csv - Team probabilities for each round with public %
  seed_bias_analysis.csv - Overseeded (FADE) vs underseeded (VALUE) teams
  bracket_chalk.csv - Chalk strategy picks
  bracket_expected_value.csv - Expected value strategy picks
  bracket_elite8_focus.csv - Elite 8 focus strategy picks
  [Plus .txt versions of each bracket for readability]
  simulation_summary.json - Summary statistics

KEY FEATURES:
  ✓ Contrarian analysis (Model % vs Public %)
  ✓ Seed bias detection (Committee vs Pure Metrics)
  ✓ Round-by-round contrarian ratios
  ✓ Strategic recommendations with narrative explanations

READING THE OUTPUT:
  Terminal shows:
    • Strategic recommendations (what to pick and why)
    • Contrarian edge opportunities (teams public undervalues)
    • Teams to avoid (overpriced due to seeding)
    • Supporting analysis (full data breakdowns)
    • Simulation summary (probabilities and brackets)

  Key CSV: round_probabilities.csv
    • Columns grouped by round: Model %, Public %, Contrarian Ratio
    • Ratio < 0.5 = Public severely undervalues (PICK THEM)
    • Ratio > 1.2 = Public overvalues (FADE THEM)


================================================================================
              L4.02 - CALCUTTA OPTIMIZER
================================================================================

PURPOSE:
  Generate optimal bidding strategies for Calcutta-style auctions where you
  buy teams and win money based on tournament performance.

WHEN TO RUN:
  Before your Calcutta auction (after tournament bracket is announced).

HOW TO RUN:
  cd L4
  python 02_calcutta_optimizer.py

WHAT IT DOES:
  1. Loads tournament simulator probabilities (from L4.01)
  2. Loads historical auction prices (from auction_history.csv)
  3. Blends historical + model valuations (70/30)
  4. Filters to seeds 1–15 only (16 seeds excluded per auction rules)
  5. Calculates Expected Value for each eligible team
  6. Optimizes portfolio allocation given budget constraints
  7. Identifies value picks and teams to avoid
  8. Exports bidding recommendations

INPUTS:
  • Tournament probabilities from 01_tournament_simulator.py
  • Historical auction prices from auction_history.csv
  • Your auction budget (configurable in script)
  • Payout structure (configurable - linear, top-heavy, winner-take-all)
  • MAX_SEED = 15 (auction rule — 16 seeds not eligible for bidding)

OUTPUTS:
  calcutta_recommendations.csv - All eligible teams ranked by Expected Value
  optimal_portfolio.csv - Recommended team allocations within budget
  value_picks.csv - Teams underpriced relative to model
  avoid_list.csv - Teams overpriced by market

KEY CONCEPTS:
  • Expected Value = (Championship Probability × Payout) - Expected Cost
  • Value Pick = Team where Model EV >> Historical Auction Price
  • Portfolio Optimization = Maximize EV while staying within budget


================================================================================
                    SEQUENCING
================================================================================

FULL PIPELINE ORDER:
  1. L0 - Data collection (scrapers)
  2. L1 - Data cleaning and standardization
  3. L2 - Feature engineering and analysis-ready datasets
  4. L3 - Model training (Elite 8 + H2H predictions)
  5. L4 - Applications (THIS LAYER)
     a. Run 01_tournament_simulator.py first
     b. Run 02_calcutta_optimizer.py second (uses simulator outputs)

L4 SPECIFIC SEQUENCE:
  Step 1: Ensure all L3 models have been trained
  Step 2: Ensure ESPN public picks are scraped (L2)
  Step 3: Run 01_tournament_simulator.py
  Step 4: Review simulator outputs (strategic recommendations)
  Step 5: Run 02_calcutta_optimizer.py (if participating in auction)


================================================================================
                    CONFIGURATION
================================================================================

KEY PATHS (defined at top of each script):
  ELITE8_PREDICTIONS_PATH - L3 Elite 8 model outputs
  H2H_MODEL_DIR - L3 H2H models directory
  PREDICTION_DATA_PATH - L3 feature data for 2026 teams
  TEAMSINDEX_PATH - Team name standardization mapping

SIMULATION PARAMETERS (01_tournament_simulator.py):
  N_SIMS_PRODUCTION = 50000 - Number of Monte Carlo simulations
  CONVERGENCE_CHECK = 10000 - Check convergence every N sims
  CONVERGENCE_THRESH = 0.001 - Convergence threshold

AUCTION PARAMETERS (02_calcutta_optimizer.py):
  MAX_SEED = 15 - Highest seed eligible for bidding (excludes 16 seeds)
                  Based on historical auction data (2013–2016): no 16 seeds
                  have ever been purchased. Change to 16 to include them.

SCORING SYSTEMS (configurable):
  ESPN Standard: R64=10, R32=20, S16=40, E8=80, FF=160, Championship=320
  Yahoo: R64=1, R32=2, S16=4, E8=8, FF=16, Championship=32


================================================================================
                    UPDATING FOR NEW YEARS
================================================================================

TO RUN FOR 2027 TOURNAMENT:
  1. Update L3 to train on 2026 season data
  2. Update paths in L4 scripts:
     - elite8_predictions_2026_long.csv → elite8_predictions_2027_long.csv
     - predict_set_2026.csv → predict_set_2027.csv
     - espn_public_picks_2026_clean.csv → espn_public_picks_2027_clean.csv
  3. Run L4 applications as normal

TO ADD HISTORICAL AUCTION DATA:
  Simply add rows to auction_history.csv with new year's data:
    Year,Player,Team,Seed,Bid,Points
    2026,Ryan,Duke,1,45,30
    2026,Ben,Kansas,2,38,12
    [etc.]
  
  No code changes needed - L4.02 automatically uses all available years.


================================================================================
                    TROUBLESHOOTING
================================================================================

ISSUE: KeyError for team names
FIX: Ensure teamsIndex.csv is up to date with all team name variations

ISSUE: Missing public picks data for some teams
FIX: Check that ESPN scraper in L2 captured all 64 tournament teams
     Verify team names in espn_public_picks_2026_clean.csv match teamsIndex

ISSUE: Model predictions seem unrealistic
FIX: Check L3 model validation metrics
     Verify feature scaling in prediction data matches training data
     Run 01_tournament_simulator.py --validate for model diagnostics

ISSUE: Contrarian ratios all showing 0.00
FIX: Public picks file missing or merge failed
     Check that teamsIndex.csv has mappings for all tournament teams


================================================================================
                    KEY PRINCIPLES
================================================================================

THINK IN PROBABILITIES, NOT BRACKETS
  • Tournament outcomes are probabilistic sequences
  • No single "correct" bracket exists
  • Optimize for expected value, not perfect prediction

OPTIMIZE WHERE SKILL BEATS LUCK
  • Early rounds (R64, R32) = high variance, hard to predict
  • Elite 8 = sweet spot (skill-based, lower variance)
  • Championship = high variance again (path dependency)

BLEND MODEL + MARKET SIGNALS
  • Models provide structural predictions
  • Public picks reveal market inefficiencies
  • Contrarian edge comes from divergence

BUILD DECISION-CALIBRATED SYSTEMS
  • Goal is actionable recommendations, not prediction leaderboards
  • Apply to economic use cases (pools, auctions)
  • Measure success by ROI, not accuracy


================================================================================
                    VALIDATION MODE
================================================================================

For development/debugging, run simulator in validation mode:
  python 01_tournament_simulator.py --validate

This runs a quick 5K simulation and generates:
  • Correlation analysis (Elite 8 model vs simulation)
  • Disagreement breakdown and visualizations
  • Model validation metrics

Use this during development to verify models before production runs.


================================================================================
                    SUPPORT
================================================================================

For questions or issues:
  1. Check this README
  2. Review output files in outputs/ directories
  3. Check terminal output for error messages
  4. Verify all prerequisite files exist and are up to date

System architecture follows modular design:
  L0 → L1 → L2 → L3 → L4
  Each layer depends only on previous layers' outputs.

