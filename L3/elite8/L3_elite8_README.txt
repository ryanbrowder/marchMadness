================================================================================
L3 ELITE 8 PROBABILISTIC FORECASTING SYSTEM
================================================================================

SYSTEM OVERVIEW
================================================================================

A sophisticated March Madness Elite 8 prediction system that models tournament 
outcomes as sequences of binary probabilities rather than traditional bracket 
logic. The system features a unified model architecture comparing bracket-aware 
predictions (WITH seeds) against pure basketball metrics (WITHOUT seeds) to 
identify value opportunities and overseeded teams.

Key Philosophy: "Think in probabilities, not brackets"
Focus: Elite 8 optimization (skill-vs-variance sweet spot)

VALIDATED PERFORMANCE (UNIFIED MODEL - March 2026)
- Training Data: 2008-2025, 1,147 team-years (+87% vs old RICH model)
- Historical ROC-AUC: 0.902 (WITH seeds), 0.897 (WITHOUT seeds)
- Seed Value: +0.005 ROC-AUC improvement (consistent across years)
- Elite 8 Accuracy: 53.8% average (4.3/8 picks)
- Best Year: 2025 (0.998 ROC-AUC, 7/8 correct - extreme chalk)
- Worst Year: 2018 (0.794 ROC-AUC, 3/8 correct - chaos)
- Performance vs Experts: +33-43% better than BartTorvik, PowerRank, ESPN Public

ARCHITECTURE
================================================================================

DIRECTORY STRUCTURE:
marchMadness/
├── L2/
│   └── data/
│       └── tournamentResults.csv          (historical tournament outcomes)
├── L3/
│   ├── config.py                          (shared configuration)
│   ├── data/
│   │   ├── trainingData/
│   │   │   └── training_set_unified.csv   (2008-2025, 1,147 teams)
│   │   └── predictionData/
│   │       └── predict_set_2026.csv       (2026 tournament teams)
│   └── elite8/
│       ├── 01_feature_selection.py
│       ├── 02_ensemble_models.py
│       ├── 03_backtest_historical.py
│       ├── 04_apply_to_2026.py
│       ├── 05_tournament_type_indicator.py
│       ├── 06_compare_seed_impact.py
│       └── outputs/                       (auto-generated)

UNIFIED DATASET ARCHITECTURE (March 2026 Update)
================================================================================

MAJOR CHANGE: Consolidated LONG/RICH dual-model into single unified model

OLD ARCHITECTURE:
- LONG: 2008-2025, 4 sources (BartTorvik, KenPom, BPI, Massey)
- RICH: 2016-2025, 6 sources (added PowerRank, LRMCB)
- Result: Two separate models, complex comparisons

NEW ARCHITECTURE:
- UNIFIED: 2008-2025, 5 sources (BartTorvik, KenPom, BPI, Massey, PowerRank)
- LRMCB removed (source stopped publishing)
- PowerRank historical data obtained from Dr. Ed Feng (thepowerrank@gmail.com)

BENEFITS:
✓ +87% more training data vs old RICH (1,147 vs 612 samples)
✓ Simpler codebase (~15-20% less code)
✓ Faster execution (single model vs two)
✓ Same performance (0.902 vs old LONG 0.902)
✓ PowerRank integration (4th strongest predictor)

DATA SOURCES (5 Total):
1. BartTorvik: Team efficiency metrics, pace, schedule strength
2. KenPom: Advanced basketball analytics, adjusted ratings
3. ESPN BPI: Basketball Power Index, offensive/defensive efficiency  
4. Massey Composite: Aggregate computer rankings
5. PowerRank (NEW): Continuous strength ratings
   * Source: Dr. Ed Feng (thepowerrank@gmail.com)
   * Historical coverage: 2008-2025
   * Scale: -28 to +24 (mean-centered around 0)
   * Type: RATING (continuous), NOT rank
   * Processing: Do NOT normalize to 0-364 scale
   * Performance: 4th strongest Elite 8 predictor (0.380 correlation)
   * Elite teams: typically 15-24 range
   * Average teams: typically -5 to +5 range

FEATURE COUNTS:
- Raw features: 48 total (47 metrics + tournamentSeed)
- After correlation reduction: 30 WITH seeds, 28 WITHOUT seeds
- Top predictors:
  1. kenpom_NetRtg: 0.421 correlation
  2. tournamentSeed: 0.412 correlation ← 2nd best!
  3. BPI: 0.407 correlation
  4. PowerRank: 0.380 correlation ← NEW!

CONFIGURATION
================================================================================

Location: L3/config.py

KEY TOGGLES:
1. USE_SEEDS = True/False
   - True:  Bracket-aware model (30 features, includes tournamentSeed)
   - False: Pure metrics model (28 features, excludes tournamentSeed)

2. MODE = 'validation' / 'production'
   - validation: Train 2008-2024, test on 2025 (strategy selection)
   - production:  Train 2008-2025 (final model for 2026+)

AUTO-GENERATED PATHS:
- Output directories append "_no_seeds" suffix when USE_SEEDS = False
- All scripts read from config.py - no hardcoded paths
- Suffix system: '' for WITH seeds, '_no_seeds' for WITHOUT

CRITICAL BUG FIX (March 2026):
Problem: tournamentSeed was not flowing through to model training
Root Cause: Line 70 in 01_feature_selection.py only pulled 3 columns from 
            tournamentResults.csv (Year, Index, tournamentOutcome)
Fix: Added 'tournamentSeed' to column list in merge operation
Result: WITH seeds model now properly has 30 features (was incorrectly 28)

PIPELINE SEQUENCING
================================================================================

STEP 1: WITH SEEDS (BRACKET-AWARE MODEL)
--------------------------------------------------------------------------------

1a. VALIDATION (determine best ensemble strategy)
    Edit L3/config.py:
        USE_SEEDS = True
        MODE = 'validation'
    
    Run:
        python 01_feature_selection.py
        python 02_ensemble_models.py
    
    Expected Output:
        - 30 features (includes tournamentSeed)
        - tournamentSeed NOT in exclusion list
        - Best strategy: Equal Weight or ROC-AUC Weighted
        - 2025 validation: ~0.998 ROC-AUC (extreme chalk year)

1b. PRODUCTION (train final model)
    Edit L3/config.py:
        USE_SEEDS = True
        MODE = 'production'
    
    Run:
        python 02_ensemble_models.py
        python 03_backtest_historical.py
        python 04_apply_to_2026.py
        python 05_tournament_type_indicator.py
    
    Expected Output:
        - Historical: 0.902 ROC-AUC average
        - 2026: 13 teams >50% probability
        - Tournament type: EXTREME CHALK (80% confidence)
        - Top pick: Arizona 74.7%

STEP 2: WITHOUT SEEDS (PURE METRICS MODEL)
--------------------------------------------------------------------------------

2a. VALIDATION
    Edit L3/config.py:
        USE_SEEDS = False
        MODE = 'validation'
    
    Run:
        python 01_feature_selection.py
        python 02_ensemble_models.py
    
    Expected Output:
        - 28 features (excludes tournamentSeed)
        - tournamentSeed IN exclusion list
        - 2025 validation: ~0.998 ROC-AUC (same as WITH seeds on chalk year)

2b. PRODUCTION
    Edit L3/config.py:
        USE_SEEDS = False
        MODE = 'production'
    
    Run:
        python 02_ensemble_models.py
        python 03_backtest_historical.py
        python 04_apply_to_2026.py
        python 05_tournament_type_indicator.py
    
    Expected Output:
        - Historical: 0.897 ROC-AUC average (-0.005 vs WITH)
        - 2026: 10 teams >50% probability
        - Tournament type: CHALK (60% confidence)
        - Top pick: Arizona 74.9%

STEP 3: COMPARISON (THE PAYOFF)
--------------------------------------------------------------------------------

Run (no config changes needed):
    python 06_compare_seed_impact.py

Output:
    - Side-by-side team probabilities
    - Seed effect = WITH - WITHOUT
    - Value picks (underseeded teams, negative effect)
    - Fade candidates (overseeded teams, positive effect)
    - Strategic recommendations

SCRIPT DETAILS
================================================================================

01_FEATURE_SELECTION.PY
Purpose: Identify predictive features, eliminate multicollinearity
Inputs:  training_set_unified.csv, tournamentResults.csv
Outputs: 
  - labeled_training_unified.csv (joined with Elite 8 labels)
  - reduced_features_unified.csv (30 WITH seeds, 28 WITHOUT)
  - high_correlations_unified.csv (feature-to-feature analysis)
  - label_correlations_unified.csv (feature-to-Elite8 correlations)
  - vif_analysis_unified.csv (multicollinearity detection)
  - drop_recommendations_unified.csv (features to eliminate)

Key Logic:
  - Loads unified training data (1,147 rows)
  - Joins with tournamentResults on (Year, Index)
  - Creates elite8_flag: 1 if Elite Eight or better, 0 otherwise
  - Excludes: Year, Team, Index, tournamentOutcome, elite8_flag
  - Conditionally excludes tournamentSeed when USE_SEEDS = False
  - Drops features with >0.9 correlation to keep strongest predictor
  - PowerRank protected from rank normalization (continuous rating)
  - tournamentSeed protected from auto-removal when USE_SEEDS = True

02_ENSEMBLE_MODELS.PY
Purpose: Train 4-model ensemble, select optimal weighting strategy
Models: 
  - Logistic Regression (scaled, class_weight='balanced')
  - Random Forest (100 trees, no scaling needed)
  - SVM (RBF kernel, scaled, class_weight='balanced')
  - Gaussian Naive Bayes (calibrated via CalibratedClassifierCV)

Strategies Tested:
  - Equal Weight: 0.25 each (simple average)
  - ROC-AUC Weighted: Weight by validation performance
  - Inverse Log-Loss Weighted: Weight by calibration quality

Validation Mode:
  - Train: 2008-2024 (1,079 samples, 128 Elite 8+)
  - Test: 2025 (68 samples, 8 Elite 8+)
  - Selects best strategy based on ROC-AUC
  - 2025 is extreme chalk - Equal Weight typically wins

Production Mode:
  - Train: 2008-2025 (all 1,147 samples, 136 Elite 8+)
  - Uses strategy identified in validation
  - Saves final model for 2026 predictions

Outputs:
  - trained_ensemble_unified_production.pkl (final model)
  - individual_model_performance_unified_validation.csv
  - ensemble_performance_unified_validation.csv
  - best_ensemble_predictions_unified_validation.csv

03_BACKTEST_HISTORICAL.PY
Purpose: Walk-forward validation on 11 historical tournaments (2015-2025)
Method: 
  - For each year: train on ALL prior years, test on that year
  - 2015: train 2008-2014 → test 2015
  - 2016: train 2008-2015 → test 2016
  - ...
  - 2025: train 2008-2024 → test 2025
  - (2020 skipped - no tournament)

Outputs:
  - backtest_summary.csv (year-by-year performance)
  - predictions_YEAR_unified.csv (detailed predictions per year)
  - Chalk vs Chaos classification by ROC-AUC

Performance Tiers:
  - CHALK (ROC-AUC ≥0.90): Favorites win, high predictability
  - NORMAL (0.85-0.90): Moderate upsets
  - CHAOS (ROC-AUC <0.85): Major upsets, low predictability

Historical Results:
  - 2025: 0.998 (extreme chalk - 7/8 correct)
  - 2019: 0.969 (chalk - 6/8 correct)
  - 2021: 0.948 (chalk - 5/8 correct)
  - 2018: 0.794 (chaos - 3/8 correct)

04_APPLY_TO_2026.PY
Purpose: Generate Elite 8 probabilities for 2026 tournament
Inputs:  
  - predict_set_2026.csv (365 teams, 2026 metrics)
  - trained_ensemble_unified_production.pkl
Outputs:
  - elite8_predictions_2026.csv (all 365 teams)
  - elite8_predictions_2026_top25.csv (top picks)

Features:
  - Probability tiers: Elite (>60%), Strong (40-60%), Moderate (25-40%)
  - Model component breakdown: P_E8_LR, P_E8_RF, P_E8_SVM, P_E8_GNB
  - Seed information included when USE_SEEDS = True
  - Missing tournamentSeed handling (fills with median when missing)

Key Columns:
  - Team: Team name
  - P_E8: Ensemble probability of reaching Elite 8
  - Estimated_Seed: ESPN bracketology seed (pre-Selection Sunday)

05_TOURNAMENT_TYPE_INDICATOR.PY
Purpose: Forecast 2026 tournament type (Chalk/Normal/Chaos)
Signals Analyzed (6 total):
  1. Top team probability (≥70% = CHALK signal)
  2. High-confidence picks ≥50% (≥10 = CHALK signal)
  3. Top 10 spread (≤20% = CHALK, ≥30% = CHAOS)
  4. Top 20 standard deviation (≤0.10 = CHALK, ≥0.15 = CHAOS)
  5. Parity 30-50% range (≤10 = CHALK, ≥15 = CHAOS)
  6. Top 5 average (≥65% = CHALK signal)

Chalk Score Calculation:
  - Each signal votes: +2 CHALK, 0 NEUTRAL, -2 CHAOS
  - Range: -13 to +11
  - Thresholds:
    * ≥7: EXTREME CHALK (80% confidence)
    * 4-6: CHALK (60% confidence)
    * -4 to 3: NORMAL (40-50% confidence)
    * <-4: CHAOS (10-30% confidence)

Output:
  - Tournament type forecast
  - Expected ROC-AUC range
  - Expected Elite 8 accuracy (picks/8)
  - Strategy recommendations by pool type

Historical Context:
  - Shows past years by type
  - Helps calibrate expectations
  - 2026 WITH seeds: Chalk score 8 (EXTREME CHALK)
  - 2026 WITHOUT seeds: Chalk score 6 (CHALK)

06_COMPARE_SEED_IMPACT.PY
Purpose: Compare WITH vs WITHOUT seeds to find opportunities
Inputs:
  - elite8_predictions_2026.csv (WITH seeds)
  - elite8_predictions_2026.csv from outputs/*_no_seeds/ (WITHOUT seeds)

Key Metrics:
  - Seed Effect = P_E8_with_seeds - P_E8_no_seeds
  - Positive effect: OVERSEEDED (bracket inflates probability)
  - Negative effect: UNDERSEEDED (metrics better than bracket)

Classification Logic:
  - Consensus: |seed_effect| < 2%
  - Value Picks: seed_effect < -2% (underseeded)
  - Fade Candidates: seed_effect > +2% (overseeded)

Strategic Output:
  - Overseeded teams: Public will overvalue in pools/auctions
  - Underseeded teams: Market inefficiency, value plays
  - Consensus teams: Both models agree, trust them
  - Visualizations: scatter plots, effect distributions

KEY FINDINGS FROM 2026 ANALYSIS
================================================================================

SEED IMPACT (NOW WORKING CORRECTLY!)

Historical Performance:
- WITH seeds: 0.902 ROC-AUC average
- WITHOUT seeds: 0.897 ROC-AUC average
- Seed value: +0.005 ROC-AUC (small but consistent improvement)
- 2019 improvement: +0.000 (chalk year, seeds redundant)
- 2023 improvement: +0.014 (seeds helped identify upsets)
- 2018 improvement: +0.011 (chaos year, seeds provided structure)

2026 Seed Effects (Top Teams):

CONSENSUS PICKS (both models agree):
  1. Arizona:     74.7% WITH, 74.9% WITHOUT → -0.2% (consensus)
  2. Connecticut: 72.0% WITH, 71.0% WITHOUT → +1.0% (minor boost)
  3. Michigan:    64.4% WITH, 64.6% WITHOUT → -0.2% (consensus)
  4. Duke:        63.6% WITH, 63.6% WITHOUT →  0.0% (perfect match)

OVERSEEDED TEAMS (fade candidates):
  1. Michigan St.: 56.9% WITH → 50.7% WITHOUT = +6.2% effect
     - Seed: 2 (committee likes them)
     - Metrics: Barely 50% (model skeptical)
     - Strategy: FADE in pools/auctions
  
  2. Houston: 59.0% WITH → 53.8% WITHOUT = +5.2% effect
     - Seed: 2 (high bracket placement)
     - Metrics: Mid-50s (not as strong)
     - Strategy: FADE, public will overvalue
  
  3. Florida: 64.4% WITH → 60.1% WITHOUT = +4.3% effect
     - Seed: 2 (good seed)
     - Metrics: Still good, but inflated
     - Strategy: Don't overpay
  
  4. Iowa St.: 58.5% WITH → 54.4% WITHOUT = +4.1% effect
     - Seed: 3 (solid)
     - Metrics: Above average, but not elite
     - Strategy: Value play if priced on metrics

UNDERSEEDED TEAMS (value picks):
  - Arizona: 74.7% WITH → 74.9% WITHOUT = -0.2% (minor)
  - Michigan: 64.4% WITH → 64.6% WITHOUT = -0.2% (minor)
  
No strong value plays identified - seeds are generally well-calibrated

TOURNAMENT TYPE FORECAST:

WITH SEEDS Model:
  - Chalk Score: 8/11
  - Type: EXTREME CHALK
  - Confidence: 80%
  - Expected ROC-AUC: 0.93-1.00
  - Expected Elite 8 picks: 6-8 out of 8
  - High-confidence teams (>50%): 13
  - Top team: Arizona 74.7%

WITHOUT SEEDS Model:
  - Chalk Score: 6/11
  - Type: CHALK
  - Confidence: 60%
  - Expected ROC-AUC: 0.88-0.95
  - Expected Elite 8 picks: 4-6 out of 8
  - High-confidence teams (>50%): 10
  - Top team: Arizona 74.9%

KEY INSIGHT: Seeds make 2026 look MORE chalk (+20% confidence boost)

STRATEGIC APPLICATIONS
================================================================================

CONSERVATIVE POOL STRATEGY (Friends, Small Pools):
  • Trust top 6 consensus picks: Arizona, UConn, Michigan, Duke, Florida, Texas Tech
  • Pick teams >55% for Elite 8
  • Expect 5-6 correct (6 would be excellent in 80% chalk)
  • Expected finish: Top 15-25%
  • Risk: Low differentiation in chalk years

COMPETITIVE POOL STRATEGY (ESPN, Large Pools):
  • LOCKS: Arizona, UConn, Michigan, Duke (top 4)
  • FADE: Michigan St., Houston (overseeded, public will overvalue)
  • DIFFERENTIATION: Pick 1-2 from 45-55% range
  • Candidates: Purdue, Kansas, Alabama (40-50% consensus, not overseeded)
  • Expected finish: Top 20-30% (hard to differentiate in extreme chalk)
  • Key: Don't overchase upsets - patience wins in chalk years

CALCUTTA AUCTION STRATEGY:
  • TOP TARGETS: Arizona, UConn, Michigan, Duke
    - Will be expensive but justified (70%+ probabilities)
    - Budget 50-60% of capital on top 4
  
  • AVOID OVERPAYING:
    - Michigan St.: Seed 2 looks good, metrics say barely 50%
    - Houston: Seed 2, everyone will bid high, value isn't there
    - Florida: Good team but 4% overseeded
  
  • VALUE ZONE (45-55% range):
    - Texas Tech (60.8%), Iowa St. (58.5%), Illinois (55.0%)
    - Purdue (51.9%), Kansas (51.8%), Alabama (52.1%)
    - Target: Teams where metrics match seeds (no inflation)
  
  • PORTFOLIO ALLOCATION:
    - 50-60% on top-4 locks
    - 30-35% on value zone (45-55%)
    - 10-15% on wildcards/hedges
  
  • AVOID: Longshots <30% (not a chaos year)

TWO-BRACKET STRATEGY (for pools allowing 2 entries):
  
  Bracket 1 (CHALK):
    - Champion: Arizona or UConn
    - Elite 8: Top 8 by probability
    - Philosophy: Trust the model, play favorites
    - Expected: 5-6 Elite 8 picks, top 20% finish
  
  Bracket 2 (HEDGE):
    - Champion: Duke or Michigan (differentiate from Bracket 1)
    - Elite 8: Top 6 + 2 from 40-55% range (Kansas, Alabama)
    - Philosophy: Same top tier, different champion, variance plays
    - Expected: 4-5 Elite 8 picks, top 30% finish
  
  Combined: One bracket will likely hit top 25%, hedge against chalk variance

DEPLOYMENT INSTRUCTIONS (SELECTION SUNDAY)
================================================================================

WHEN: March 16, 2026 (Selection Sunday - bracket revealed)

PRE-SELECTION SUNDAY (you are here):
  ✓ Models trained on 2008-2025 data
  ✓ Using ESPN bracketology for seed estimates
  ✓ Predictions ready with estimated seeds

SELECTION SUNDAY UPDATES:

1. Update Seed Information
   Location: L3/data/predictionData/predict_set_2026.csv
   Action: Replace "Estimated_Seed" with actual committee seeds
   
   Common Changes:
   - Teams move up/down 1-2 seed lines
   - Region assignments finalized
   - Play-in games (First Four) determined
   
   Note: First Four teams have <2% Elite 8 probability - safe to ignore

2. Re-run WITH SEEDS Pipeline
   Edit L3/config.py:
       USE_SEEDS = True
       MODE = 'production'
   
   Run:
       python 04_apply_to_2026.py
       python 05_tournament_type_indicator.py
   
   Watch For:
   - Probability changes for re-seeded teams
   - New overseeded/underseeded teams
   - Updated chalk score

3. Re-run WITHOUT SEEDS Pipeline  
   Edit L3/config.py:
       USE_SEEDS = False
       MODE = 'production'
   
   Run:
       python 04_apply_to_2026.py
       python 05_tournament_type_indicator.py

4. Compare Models
   Run:
       python 06_compare_seed_impact.py
   
   Review:
   - Did fade candidates change?
   - New value picks from seed changes?
   - Consensus still aligned?

5. Finalize Strategy
   - Lock in your top 8 Elite 8 picks
   - Identify 2-3 differentiation plays
   - Set Calcutta budget allocation
   - Prepare both brackets (if applicable)

FIRST WEEKEND MONITORING:

Thursday-Sunday (Round 1 & Round 2):
  - Monitor for major upsets in top 4 seeds
  - If 2+ top-4 seeds lose → chaos emerging (adjust Sweet 16 picks)
  - If favorites win → chalk confirmed (trust the model)

Sweet 16 (By this point you'll know):
  - Chalk year: Stick with top picks for Elite 8
  - Chaos year: Consider pivoting to surviving mid-seeds
  - Model assumes 60-80% chalk - if very wrong, recalibrate

Elite 8 Weekend:
  - This is what the model optimizes for
  - Expected: 4-6 correct in chalk, 3-4 in chaos
  - Anything above 50% is excellent

TECHNICAL NOTES
================================================================================

ENSEMBLE ARCHITECTURE:
- ROC-AUC Weighted (typical): 24% LR, 24% RF, 23% SVM, 29% GNB
- Equal Weight (2025 chalk): 25% each
- Class weights: 'balanced' to handle 11.9% Elite 8+ class
- Feature scaling: StandardScaler for LR/SVM, raw for RF/GNB
- Calibration: CalibratedClassifierCV on Gaussian NB

DATA REQUIREMENTS:
- Training: 2008-2025 tournament teams (1,147 team-years)
- Sources: BartTorvik, KenPom, ESPN BPI, Massey, PowerRank
- Missing data: Median imputation (rare - most teams have all metrics)
- NaN columns: Automatically dropped (e.g., bartTorvik_Conf)
- tournamentSeed: Fills with median when missing in prediction data

VALIDATION METHODOLOGY:
- Walk-forward: Train on all years before test year
- No future data leakage
- Consistent with H2H model approach
- 2025 used for strategy selection (extreme chalk provides clear winner)

FEATURE ENGINEERING:
- Rank normalization: All ranks scaled to 0-364 range (consistent across years)
- PowerRank exception: Continuous rating, NOT normalized
- Correlation threshold: 0.9 (drop weaker of correlated pair)
- VIF threshold: Review >10 (but don't auto-drop)
- Protected features: tournamentSeed (when USE_SEEDS=True), PowerRank

KNOWN LIMITATIONS:
- Best in chalk years (0.93-1.00 ROC-AUC)
- Struggles in chaos years (0.75-0.85 ROC-AUC)
- Team-level probabilities (not matchup-specific)
- Assumes independence between games (not true in practice)
- 17-year training window (2008-2025) may miss recent regime shifts
- First Four games not modeled (but 16-seeds have ~0% Elite 8 probability)

PERFORMANCE VS EXPERTS (Elite 8 Predictions, 2017-2025):
- March Madness Computron: 0.896 ROC-AUC
- BartTorvik: 0.671 ROC-AUC
- PowerRank: 0.657 ROC-AUC
- ESPN Public: 0.626 ROC-AUC

System is 33-43% better than expert sources at identifying Elite 8 teams!

MAINTENANCE & UPDATES
================================================================================

ANNUAL TASKS (Post-Tournament):

1. Update Training Data
   Location: L3/data/trainingData/training_set_unified.csv
   Action: Add 2026 season metrics
   Sources: Scrape final BartTorvik, KenPom, BPI, Massey, PowerRank
   
2. Update Tournament Results
   Location: L2/data/tournamentResults.csv
   Action: Add 2026 bracket outcomes
   Columns: Team, Index, tournamentSeed, tournamentOutcome, Year
   
3. Re-run Validation
   Purpose: Verify model still performing well
   Steps:
     - Edit config: MODE = 'validation'
     - Run 01_feature_selection.py
     - Run 02_ensemble_models.py
     - Check: Is ROC-AUC Weighted still best strategy?
     - Check: Any major feature changes?
   
4. Monitor Drift
   Metrics to watch:
   - Historical ROC-AUC staying ~0.90?
   - Feature correlations stable?
   - New data sources available?
   - Rule changes affecting tournament?

MID-SEASON UPDATES (Optional, December-February):

- Update predict_set_2026.csv as metrics evolve
- Re-run 04_apply_to_2026.py monthly
- Track probability changes through season
- Identify teams trending up/down
- Get final ESPN bracketology before Selection Sunday

CONTACT POWERRANK (if issues):
- Email: thepowerrank@gmail.com (Dr. Ed Feng)
- Request: Historical data or current season ratings
- Note: PowerRank very responsive, great data source

TROUBLESHOOTING
================================================================================

COMMON ISSUES:

1. "FileNotFoundError: trained_ensemble_unified_production.pkl"
   Cause: Production model not trained yet
   Fix: Run 02_ensemble_models.py with MODE = 'production'
   Check: Verify OUTPUT_02 path in config.py

2. "tournamentSeed not in feature list" or "28 features instead of 30"
   Cause: tournamentSeed not included in merge (the bug we fixed)
   Fix: Verify line 70 in 01_feature_selection.py includes 'tournamentSeed'
   Should be: tournament_results[['Year', 'Index', 'tournamentSeed', 'tournamentOutcome']]

3. "Shape mismatch: expected 30 features, got 28"
   Cause: Model trained WITH seeds but prediction data missing tournamentSeed
   Fix: Check predict_set_2026.csv has tournamentSeed column
   Alternative: Model will fill with median if missing

4. "PowerRank values out of range" or "PowerRank normalized to 0-364"
   Cause: PowerRank treated as rank instead of rating
   Fix: Verify config.is_powerrank_column() protecting it from normalization
   Expected: PowerRank values from -28 to +24, not 0-364

5. Identical predictions from WITH and WITHOUT seeds models
   Cause: tournamentSeed not actually included (the bug)
   Fix: Check reduced_features_unified.csv contains tournamentSeed
   Command: grep "tournamentSeed" L3/elite8/outputs/01_feature_selection/reduced_features_unified.csv
   Should return: tournamentSeed

6. Probabilities seem off (all >90% or <10%)
   Cause: Feature scaling issues or data corruption
   Fix: Check for NaN warnings in script output
   Check: Verify numeric conversion happened (should see "Converting columns to numeric")
   Check: StandardScaler fit on training data before transform

7. "Different results than last run"
   Verify: MODE = 'production' (not validation)
   Verify: USE_SEEDS matches what you expect
   Verify: Same training_set_unified.csv
   Note: Random seed set for reproducibility (RF has randomness)

8. Script reading wrong output directories
   Cause: config.py OUTPUT paths incorrect
   Fix: Check ELITE8_DIR, RESULTS_DIR, L2_DIR in config
   Fix: Verify SUFFIX correctly appending '_no_seeds'
   Command: Check config.print_config() output

DEBUGGING COMMANDS:

# Verify tournamentSeed in final features (WITH seeds)
grep "tournamentSeed" L3/elite8/outputs/01_feature_selection/reduced_features_unified.csv

# Count features (should be 30 WITH, 28 WITHOUT)
wc -l L3/elite8/outputs/01_feature_selection/reduced_features_unified.csv

# Check if tournamentSeed in labeled data
head -1 L3/elite8/outputs/01_feature_selection/labeled_training_unified.csv | tr ',' '\n' | grep "Seed"

# Verify config settings
cd L3/elite8
python -c "import config; config.print_config()"

CONTACT & SUPPORT
================================================================================
AUTHOR
================================================================================

Ryan Browder
March Madness Computron

PHILOSOPHY:
- Probabilistic thinking > bracket logic
- Economic utility > prediction leaderboards  
- Skill optimization > variance chasing
- Expected value > perfect brackets

DESIGN FOR:
✓ Bracket pool optimization
✓ Calcutta auction strategy
✓ Expected value decision-making

NOT DESIGNED FOR:
✗ Gambling or sports betting
✗ Certainty or guarantees
✗ Perfect bracket chasing
✗ Individual game predictions

KEY CONTACTS:
- PowerRank data: Dr. Ed Feng (thepowerrank@gmail.com)
- BartTorvik data: barttorvik.com
- KenPom data: kenpom.com (subscription required)

REMEMBER:
Even with 0.902 ROC-AUC, you're predicting 4-6 out of 8 Elite 8 teams correctly.
The goal is to BEAT YOUR POOL, not predict perfectly.
In chalk years, patience and favorites win.
In chaos years, accept lower accuracy and make smart variance plays.

Good luck! 🏀

================================================================================
END OF README
================================================================================
