================================================================================
                    L4 - APPLICATIONS LAYER
================================================================================

OVERVIEW
------------------------------------------------------------------------
L4 contains three production applications that use L3 model predictions for
decision-making in March Madness contexts:

  01_tournament_simulator.py  - Bracket pool optimization (PRE-TOURNAMENT)
  02_calcutta_optimizer.py    - Auction bidding strategy (PRE-TOURNAMENT)
  03_predict_next_round.py    - Live betting probabilities (DURING TOURNAMENT)

Applications 01 and 02 are designed to run on Selection Sunday (or after) when
the tournament bracket is finalized.

Application 03 runs during the tournament after each round completes.


================================================================================
                    DIRECTORY STRUCTURE
================================================================================

L4/
├── 01_tournament_simulator.py       # Pre-tournament: Full bracket probs
├── 02_calcutta_optimizer.py         # Pre-tournament: Auction strategy
├── 03_predict_next_round.py         # Live: Round-by-round betting probs
├── data/
│   └── actual_results.csv           # Live tournament results (user-maintained)
└── outputs/
    ├── 01_tournament_simulator/     # Bracket probabilities, contrarian
    ├── 02_calcutta_optimizer/       # Team valuations, auction bids
    └── 03_live_predictions/         # Live round predictions


================================================================================
                    PREREQUISITES
================================================================================

L4 applications require outputs from earlier pipeline layers:

FROM L3 (MODELS):
  ✓ Elite 8 predictions (WITH SEEDS)
    → L3/elite8/outputs/04_2026_predictions/elite8_predictions_2026.csv
  
  ✓ Elite 8 predictions (NO SEEDS - for seed bias analysis)
    → L3/elite8/outputs/04_2026_predictions_no_seeds/elite8_predictions_2026.csv
  
  ✓ H2H models (WITH SEEDS)
    → L3/h2h/models_with_seeds/
  
  ✓ H2H models (NO SEEDS - for seed bias analysis)
    → L3/h2h/models/
  
  ✓ Prediction features (2026 tournament teams)
    → L3/data/predictionData/predict_set_2026.csv

FROM L2 (DATA):
  ✓ ESPN public pick percentages
    → L2/data/bracketology/espn_public_picks_2026_clean.csv
  
  ✓ ESPN bracket structure
    → L2/data/bracketology/espn_bracketology_2026.csv
  
  ✓ Historical Calcutta auction data (optional, for L4.02)
    → L2/data/calcutta/auction_history.csv

FROM UTILS:
  ✓ Team name standardization
    → utils/teamsIndex.csv


================================================================================
              L4.01 - TOURNAMENT SIMULATOR
================================================================================

PURPOSE:
  Generate full tournament probabilities for bracket pools. Identify contrarian
  opportunities where model disagrees with public sentiment.

WHEN TO RUN:
  Selection Sunday, after tournament bracket is announced.

HOW TO RUN:
  cd L4
  python 01_tournament_simulator.py

WHAT IT DOES:
  1. Loads L3 Elite 8 predictions (WITH_SEEDS and NO_SEEDS)
  2. Loads L3 H2H ensemble models
  3. Constructs tournament bracket from ESPN bracketology
  4. Runs 10,000 Monte Carlo simulations
  5. Calculates round-by-round advancement probabilities
  6. Compares model predictions to ESPN public pick percentages
  7. Identifies seed bias (teams over/undervalued by seeding)
  8. Generates optimal brackets for different strategies
  9. Exports game-by-game probabilities and analysis

CONFIGURATION:
  N_SIMS = 10000              # Number of Monte Carlo simulations
  USE_ELITE8_NO_SEEDS = False # Use NO_SEEDS model (default: WITH_SEEDS)

OUTPUTS:
  All files saved to: outputs/01_tournament_simulator/

  game_probabilities.txt - Game-by-game probabilities for bracket filling
  game_probabilities.csv - Excel-friendly version
  round_probabilities.csv - Team advancement probabilities by round
  bracket_chalk.csv - Optimal bracket (pick every favorite)
  bracket_contrarian.csv - Contrarian bracket (fade public favorites)
  seed_bias_analysis.csv - Teams over/undervalued by seeding

KEY FEATURES:
  ✓ Play-In handling (First Four resolved deterministically)
  ✓ Dual-model approach (WITH_SEEDS and NO_SEEDS)
  ✓ Contrarian analysis (Model % vs Public %)
  ✓ Seed bias detection (Committee vs Pure Metrics)
  ✓ Two bracket strategies (chalk and contrarian)

READING THE OUTPUT:

  game_probabilities.txt - Use this to fill your bracket
    Shows probabilities for every possible matchup from R64 through Championship
    Format:
      ROUND OF 64
      -----------
      Play-In: (16) BCU vs (16) Howard → BCU 79.8%
      East: (1) Duke vs (16) BCU → Duke 98.5%
      East: (8) St Louis vs (9) Iowa → St Louis 62.3%
      ...
      
      ROUND OF 32 (Expected Matchups)
      --------------------------------
      East: (1) Duke vs (8) St Louis → Duke 87.5%
      ...

  round_probabilities.csv - Full probability breakdown with public comparisons
    Columns by round:
      P_R64 = Probability of reaching Round of 64 (always 1.0 for 1-15 seeds)
      P_R32 = Probability of reaching Round of 32 (i.e., winning R64 game)
      P_S16 = Probability of reaching Sweet 16
      P_E8 = Probability of reaching Elite 8
      P_FF = Probability of reaching Final Four
      P_Championship = Probability of reaching Championship game
      P_WIN_Championship = Probability of winning championship
      
      Public_R32_Pct = ESPN public pick % for reaching R32
      R32_Contrarian_Ratio = Public_R32_Pct / (P_R32 * 100)
      
    Contrarian Ratio interpretation:
      > 1.05 = Public overvalues team (fade candidate)
      < 0.95 = Public undervalues team (value pick)
      0.95-1.05 = Neutral (model agrees with public)

  seed_bias_analysis.csv - Teams where seeds create substantial probability shifts
    P_E8_NO_SEEDS = Elite 8 probability without seed information
    P_E8_WITH_SEEDS = Elite 8 probability with seed information
    Seed_Boost = P_E8_WITH_SEEDS - P_E8_NO_SEEDS
    
    Category:
      SEED_INFLATED = Seeds boost probability >5pp (fade candidates)
      SEED_DEFLATED = Seeds suppress probability >5pp (value picks)
    
    2026 Example fade candidates (seed inflated):
      Houston (2-seed, South): +9pp seed boost
      Purdue (3-seed, Midwest): +7pp seed boost
      Florida (4-seed, East): +6pp seed boost

BRACKET STRATEGIES:

  CHALK STRATEGY (bracket_chalk.csv):
    • Pick the favorite in every game
    • Maximize expected value
    • Best for small pools (<20 people)
    • Edge comes from probability calibration

  CONTRARIAN STRATEGY (bracket_contrarian.csv):
    • Fade public favorites where model disagrees
    • Maximize differentiation
    • Best for large pools (>20 people)
    • Target championship picks with low ownership but high model probability


================================================================================
              L4.02 - CALCUTTA OPTIMIZER
================================================================================

PURPOSE:
  Generate optimal bidding strategies for Calcutta-style auctions where you
  buy teams and win money based on tournament performance.

WHEN TO RUN:
  Before your Calcutta auction (after tournament bracket is announced).
  Must run 01_tournament_simulator.py first.

HOW TO RUN:
  cd L4
  python 02_calcutta_optimizer.py

WHAT IT DOES:
  1. Loads tournament simulator probabilities (from L4.01)
  2. Loads L3 Elite 8 predictions
  3. Calculates Expected Value for each team
  4. Filters to seeds 1-15 (16 seeds excluded per auction rules)
  5. Generates tier-based bidding recommendations
  6. Exports team valuations and strategy guide

CONFIGURATION (edit in script):
  TOTAL_BUDGET = 1000      # Your auction budget
  FIELD_SIZE = 68          # Total tournament teams (64 + 4 play-ins)
  FILTER_16_SEEDS = True   # Exclude 16-seeds from recommendations
  
  PAYOUT_STRUCTURE = {
    'R64': 0,
    'R32': 5,
    'S16': 10,
    'E8': 20,
    'FF': 35,
    'Championship': 50,
    'Champion': 100
  }

OUTPUTS:
  All files saved to: outputs/02_calcutta_optimizer/

  team_valuations_2026.csv - Expected value and recommended bids for each team
  auction_strategy.txt - Bid recommendations by tier

KEY CONCEPTS:
  • Expected Value = Sum of (round probability × payout)
  • Recommended Max Bid = % of budget based on EV tier
  • Tier classification based on championship probability and EV

READING THE OUTPUT:

  team_valuations_2026.csv:
    Team = Team name
    Seed = Tournament seed
    Region = Tournament region
    P_Champion = Probability of winning championship
    Expected_Value = Expected payout based on probabilities
    Recommended_Max_Bid = Maximum you should bid (as % of budget)
    Tier = Value tier (Tier 1 = Top Value, Tier 2 = Good Value, etc.)

  auction_strategy.txt:
    TIER 1 - TOP VALUE (Spend 40-60% of budget):
      Teams with highest expected value
      Premium targets, bid aggressively
    
    TIER 2 - GOOD VALUE (Spend 20-30% of budget):
      Solid value, bid moderately
    
    TIER 3 - LOTTERY TICKETS (Spend <10% of budget):
      Low probability but potential upside

BIDDING STRATEGY:
  • Allocate 40-60% of budget to Tier 1 teams
  • Allocate 20-30% to Tier 2 teams
  • Allocate <10% to Tier 3 teams
  • Reserve 10% for mid-auction opportunities
  • Diversify across regions
  • Don't spend >60% on a single team
  • Avoid 16-seeds (filtered automatically)


================================================================================
              L4.03 - LIVE ROUND PREDICTOR
================================================================================

PURPOSE:
  Calculate win probabilities for upcoming round based on actual tournament
  results. Use for betting/gambling decisions during tournament.

WHEN TO RUN:
  After each round completes, before placing bets on next round.

HOW TO RUN:
  cd L4
  
  # After R64 completes
  python 03_predict_next_round.py --next-round R32
  
  # After R32 completes
  python 03_predict_next_round.py --next-round S16
  
  # Continue for E8, FF, Championship
  python 03_predict_next_round.py --next-round E8
  python 03_predict_next_round.py --next-round FF
  python 03_predict_next_round.py --next-round Championship

WHAT IT DOES:
  1. Loads actual tournament results from data/actual_results.csv
  2. Removes losing teams from prediction data (using Index)
  3. Determines next round's matchups based on bracket structure
  4. Calculates H2H probabilities using L3 ensemble models
  5. Classifies games by betting confidence tier
  6. Exports betting sheet with recommendations

INPUT FILE: data/actual_results.csv

  Format (5 columns):
    round,Team IndexA,winner,Team IndexB,loser
  
  Example:
    round,Team IndexA,winner,Team IndexB,loser
    R64,76,Duke,45,Central Arkansas
    R64,265,St Louis,132,Iowa
    R32,76,Duke,265,St Louis
    S16,76,Duke,108,Gonzaga
    E8,76,Duke,180,Michigan St.
    FF,76,Duke,14,Arizona
    Championship,76,Duke,120,Houston
  
  Columns:
    round = Round name (R64, R32, S16, E8, FF, Championship)
    Team IndexA = Winner's Index from utils/teamsIndex.csv
    winner = Winner's team name (for readability)
    Team IndexB = Loser's Index from utils/teamsIndex.csv
    loser = Loser's team name (for readability)
  
  Finding Team Index values:
    # Search for a team
    grep -i "duke" utils/teamsIndex.csv
    # Output: 76,Duke
    
    # Or open in Excel
    open utils/teamsIndex.csv

  Why Index-based?
    ✓ No team name spelling issues
    ✓ No case sensitivity problems
    ✓ Reliable joins across all data sources
    ✓ Names still included for human readability

OUTPUTS:
  All files saved to: outputs/03_live_predictions/

  r32_probabilities.txt (or s16/e8/ff/championship) - Human-readable betting sheet
  r32_probabilities.csv - Excel-friendly data

BETTING TIERS:
  LOCK (>90%) - Bet heavy (biggest unit size)
    Model has overwhelming confidence
    Example: 1-seed vs 16-seed survivor
  
  STRONG (70-90%) - Bet moderate (standard unit)
    Good betting opportunity with solid edge
    Best value opportunities
  
  LEAN (60-70%) - Bet small (half unit)
    Modest edge, small bet recommended
  
  SLIGHT (55-60%) - Minimal bet or pass
    Very small edge, consider passing
    Edge too small for reliable profit
  
  TOSS-UP (<55%) - Avoid or bet underdog for value
    No clear edge on favorite
    Near 50/50, no betting edge

READING THE OUTPUT:

  Example betting sheet (r32_probabilities.txt):
  
    ============================================================================
                        ROUND OF 32 - LIVE PROBABILITIES
    ============================================================================
    
    Based on actual tournament results
    
    ALL MATCHUPS
    ----------------------------------------------------------------------------
    
      East:
        (1) Duke vs (8) St Louis
          → Duke 87.5% [STRONG]
        (4) Gonzaga vs (5) Tennessee
          → Gonzaga 64.2% [LEAN]
    
    ============================================================================
                        BETTING TIERS
    ============================================================================
    
    LOCKS (>90% - Bet Heavy) [0 games]
    ----------------------------------------------------------------------------
    None
    
    STRONG FAVORITES (70-90% - Bet Moderate) [2 games]
    ----------------------------------------------------------------------------
      Duke vs St Louis
        → Duke 87.5%
        - Good betting opportunity with solid edge
      
      Michigan St. vs Texas A&M
        → Michigan St. 71.3%
        - Good betting opportunity with solid edge
    
    LEANS (60-70% - Small Bet) [3 games]
    ----------------------------------------------------------------------------
      Gonzaga vs Tennessee
        → Gonzaga 64.2%
        - Modest edge, small bet recommended
    
    TOSS-UPS (<55% - Avoid or Hedge) [1 game]
    ----------------------------------------------------------------------------
      Maryland vs Baylor
        → Maryland 58.9%
        - Very small edge, consider passing

  r32_probabilities.csv:
    Region = Tournament region or matchup (e.g., "East" or "East vs Midwest")
    Team1 = First team in matchup
    Seed1 = Team1's seed
    Team2 = Second team in matchup
    Seed2 = Team2's seed
    Team1_Prob = Probability Team1 wins
    Team2_Prob = Probability Team2 wins (1 - Team1_Prob)
    Favorite = Team with higher probability
    Favorite_Prob = Higher of Team1_Prob or Team2_Prob
    Confidence = Betting tier (LOCK, STRONG, LEAN, SLIGHT, TOSS-UP)

WORKFLOW DURING TOURNAMENT:

  THURSDAY EVENING (R64 Day 1):
    1. Update data/actual_results.csv with Thursday results
    2. Optional: Run predictor for remaining R64 games
       python 03_predict_next_round.py --next-round R64

  SATURDAY EVENING (R64 Complete):
    1. Add all R64 results to data/actual_results.csv (should have 32 games)
    2. Predict R32:
       python 03_predict_next_round.py --next-round R32
    3. Review betting sheet:
       cat outputs/03_live_predictions/r32_probabilities.txt
    4. Place R32 bets before Sunday games

  SUNDAY EVENING (R32 Complete):
    1. Add R32 results (should have 48 total lines: 32 R64 + 16 R32)
    2. Predict S16:
       python 03_predict_next_round.py --next-round S16

  CONTINUE PATTERN:
    Thursday Evening (S16 Complete) → Predict E8
    Saturday Evening (E8 Complete) → Predict FF
    Saturday Evening (FF Complete) → Predict Championship

BETTING STRATEGY:
  • Compare model probabilities to Vegas lines
  • Model 70% but Vegas -500 (83% implied) → Pass
  • Model 70% but Vegas -200 (67% implied) → BET!
  • Model is ONE input, not gospel
  • Track performance by tier in Excel
  • Adjust confidence based on results

UNIT SIZING BY TIER:
  LOCK: 3-5 units
  STRONG: 1-2 units
  LEAN: 0.5 units
  SLIGHT/TOSS-UP: Pass or minimal bet


================================================================================
                    SEQUENCING
================================================================================

FULL PIPELINE ORDER:
  1. L0 - Data collection (scrapers)
  2. L1 - Data cleaning and standardization
  3. L2 - Feature engineering and analysis-ready datasets
  4. L3 - Model training (Elite 8 + H2H predictions)
  5. L4 - Applications (THIS LAYER)

PRE-TOURNAMENT (Selection Sunday):
  Step 1: Ensure all L3 models have been trained
  Step 2: Ensure ESPN public picks are scraped (L2)
  Step 3: Run 01_tournament_simulator.py
  Step 4: Review simulator outputs (strategic recommendations)
  Step 5: Run 02_calcutta_optimizer.py (if participating in auction)
  Step 6: Fill bracket using game_probabilities.txt
  Step 7: Submit to pools before deadline
  Step 8: Bid in Calcutta using team_valuations_2026.csv

DURING TOURNAMENT:
  After each round completes:
    1. Update data/actual_results.csv with results
    2. Run 03_predict_next_round.py --next-round [NEXT_ROUND]
    3. Review betting sheet
    4. Place bets before next round starts


================================================================================
                    CONFIGURATION
================================================================================

KEY PATHS (defined at top of each script):

  L4.01 (Tournament Simulator):
    TEAMSINDEX_PATH = ../utils/teamsIndex.csv
    H2H_MODEL_DIR = ../L3/h2h/models_with_seeds
    PREDICTION_DATA_PATH = ../L3/data/predictionData/predict_set_2026.csv
    ESPN_BRACKET_PATH = ../L2/data/bracketology/espn_bracketology_2026.csv
    ESPN_PUBLIC_PATH = ../L2/data/bracketology/espn_public_picks_2026_clean.csv
    ELITE8_PREDICTIONS_PATH = ../L3/elite8/outputs/04_2026_predictions/elite8_predictions_2026.csv
    ELITE8_PREDICTIONS_NO_SEEDS_PATH = ../L3/elite8/outputs/04_2026_predictions_no_seeds/elite8_predictions_2026.csv

  L4.02 (Calcutta Optimizer):
    Same as L4.01 plus:
    ROUND_PROBABILITIES_PATH = outputs/01_tournament_simulator/round_probabilities.csv

  L4.03 (Live Predictor):
    TEAMSINDEX_PATH = ../utils/teamsIndex.csv
    H2H_MODEL_DIR = ../L3/h2h/models_with_seeds
    PREDICTION_DATA_PATH = ../L3/data/predictionData/predict_set_2026.csv
    ESPN_BRACKET_PATH = ../L2/data/bracketology/espn_bracketology_2026.csv
    DEFAULT_RESULTS_PATH = data/actual_results.csv

SIMULATION PARAMETERS:
  N_SIMS = 10000              # Number of Monte Carlo simulations (L4.01)
  USE_ELITE8_NO_SEEDS = False # Use NO_SEEDS model for Elite 8 (L4.01)

AUCTION PARAMETERS:
  TOTAL_BUDGET = 1000         # Your auction budget (L4.02)
  FIELD_SIZE = 68             # Total tournament teams (L4.02)
  FILTER_16_SEEDS = True      # Exclude 16-seeds from recommendations (L4.02)

FINAL FOUR MATCHUPS (bracket structure):
  ff_matchups = [('East', 'Midwest'), ('South', 'West')]


================================================================================
                    UPDATING FOR NEW YEARS
================================================================================

TO RUN FOR 2027 TOURNAMENT:
  1. Update L3 to train on 2026 season data
  2. Update paths in all L4 scripts:
     - elite8_predictions_2026.csv → elite8_predictions_2027.csv
     - predict_set_2026.csv → predict_set_2027.csv
     - espn_public_picks_2026_clean.csv → espn_public_picks_2027_clean.csv
     - espn_bracketology_2026.csv → espn_bracketology_2027.csv
  3. Run L4 applications as normal
  4. Create new data/actual_results.csv for 2027 tournament


================================================================================
                    TROUBLESHOOTING
================================================================================

L4.01 (TOURNAMENT SIMULATOR):

  ISSUE: "Elite 8 predictions file not found"
  FIX: Verify L3 Elite 8 predictions exist:
       ls L3/elite8/outputs/04_2026_predictions/elite8_predictions_2026.csv

  ISSUE: "Region column not found"
  FIX: ESPN bracketology must have Region column. Re-scrape if needed.

  ISSUE: Championship probabilities don't sum to 100%
  FIX: This is expected due to rounding and simulation variance.
       Should be within 0.1% of 100%.

L4.02 (CALCUTTA OPTIMIZER):

  ISSUE: All bids seem too low
  FIX: Check TOTAL_BUDGET setting. Default is $1000.

  ISSUE: Missing teams
  FIX: 16-seeds are filtered by default. Edit FILTER_16_SEEDS if needed.

L4.03 (LIVE PREDICTOR):

  ISSUE: "Results CSV must have columns..."
  FIX: Verify format: round,Team IndexA,winner,Team IndexB,loser
       Check for header row and correct column names

  ISSUE: "No teams with seeds 1-16"
  FIX: Check Index values in data/actual_results.csv
       Verify Index values match utils/teamsIndex.csv

  ISSUE: Wrong matchups shown
  FIX: Verify all previous rounds are complete in data/actual_results.csv
       Example: To predict S16, need complete R64 + R32 results

  ISSUE: Team name not found
  FIX: Use Index values, not names. Search teamsIndex.csv:
       grep -i "team name" utils/teamsIndex.csv

GENERAL:

  ISSUE: KeyError for team names
  FIX: Ensure teamsIndex.csv is up to date with all team name variations

  ISSUE: Missing public picks data for some teams
  FIX: Check that ESPN scraper in L2 captured all 64 tournament teams
       Verify team names in espn_public_picks_2026_clean.csv match teamsIndex


================================================================================
                    KEY METHODOLOGIES
================================================================================

ELITE 8 FIRST APPROACH:
  Why Elite 8?
    • Sweet spot where skill beats variance
    • Championship game has ~50% luck component (single elimination)
    • Elite 8: 4 games, enough data for model edge
    • Model has demonstrated 33% improvement over expert sources at Elite 8

  Implementation:
    1. Train separate Elite 8 binary classifier (L3)
    2. Use Elite 8 probabilities to seed tournament simulation (L4.01)
    3. Calculate H2H probabilities for individual matchups (L4.01, L4.03)
    4. Combine via Monte Carlo simulation

TWO-MODEL STRATEGY:
  WITH_SEEDS model (default):
    • Uses tournamentSeed as feature
    • Better for pre-tournament simulation
    • Seeds capture market sentiment and matchup context

  NO_SEEDS model:
    • Excludes tournamentSeed feature
    • Better for identifying overvalued/undervalued teams
    • Reveals where seeds inflate/deflate probabilities

  Seed Bias Analysis:
    P_E8_WITH_SEEDS - P_E8_NO_SEEDS = Seed Boost
    
    Seed Boost > 5pp → Seed inflates probability (fade candidate)
    Seed Boost < -5pp → Seed deflates probability (value pick)

  2026 Fade Candidates (Seed Inflated):
    • Houston (2-seed, South): +9pp boost
    • Purdue (3-seed, Midwest): +7pp boost
    • Florida (4-seed, East): +6pp boost
    • Illinois (5-seed, West): +5pp boost

H2H ENSEMBLE:
  Models: Gradient Boosting, SVM, Random Forest, Neural Network, Gaussian NB

  Weighting:
    • Default: GB 35%, SVM 25%, RF 20%, NN 15%, GNB 5%
    • GNB fallback: Exclude when predictions are extreme (<1% or >99%)

  Feature Engineering:
    pct_diff = (team1_metric - team2_metric) / avg(team1_metric, team2_metric)

  Sources: BartTorvik, KenPom, ESPN BPI, Massey, PowerRank

CONTRARIAN ANALYSIS:
  Theory: ESPN public picks function as market prices, not quality signals.

  Calculation:
    Contrarian Ratio = Public Pick % / (Model Probability * 100)

  Interpretation:
    Ratio > 1.05: Public overvalues team (fade)
    Ratio < 0.95: Public undervalues team (value)
    Ratio 0.95-1.05: Neutral (model agrees with public)

  Historical Performance (2016-2025):
    • Teams with model prob > public % + 5pp: 67% reach Elite 8
    • Teams with model prob ≈ public %: 28% reach Elite 8
    • Consensus gap is the primary edge signal

  Blue Blood Exception:
    • Kansas and Duke: Only 50% fade discount (not full fade)
    • Public overrepresentation justified by tournament experience


================================================================================
                    BEST PRACTICES
================================================================================

BRACKET POOL STRATEGY:

  Small pools (<20 people):
    • Use CHALK bracket (maximize EV)
    • Pick every favorite
    • Edge comes from probability calibration, not differentiation

  Large pools (>20 people):
    • Use CONTRARIAN bracket (maximize differentiation)
    • Fade public favorites where model disagrees
    • Target championship picks with <5% ownership but >10% model probability

  Pool scoring systems:
    • 1-2-4-8-16-32: Chalk strategy (later rounds worth more)
    • 1-1-1-1-1-1: Balanced (consider contrarian in Final Four)
    • Upset bonus: Lean into contrarian picks

CALCUTTA STRATEGY:

  Budget allocation:
    • 40-60% on Tier 1 (high EV teams)
    • 20-30% on Tier 2 (good value)
    • <10% on Tier 3 (lottery tickets)
    • Reserve 10% for mid-auction opportunities

  Bidding tactics:
    • Let others bid up 1-seeds early
    • Target 2-4 seeds with high model probability
    • Avoid 16-seeds (filtered automatically)
    • Watch for teams with seed boost <-5pp (undervalued)

  Risk management:
    • Diversify across regions
    • Don't spend >60% on a single team
    • Balance high-floor (Elite 8) vs high-ceiling (Championship) teams

LIVE BETTING STRATEGY:

  Tier-based unit sizing:
    • LOCK (>90%): 3-5 units
    • STRONG (70-90%): 1-2 units
    • LEAN (60-70%): 0.5 units
    • SLIGHT/TOSS-UP: Pass or minimal bet

  Compare to market:
    • Model 70% but Vegas -500 (83% implied) → Pass
    • Model 70% but Vegas -200 (67% implied) → BET!
    • Model is ONE input, not gospel

  Track performance:
    • Log all bets in Excel
    • Calculate ROI by tier
    • Calibrate confidence over time


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
  • Apply to economic use cases (pools, auctions, live betting)
  • Measure success by ROI, not accuracy


================================================================================
                    PERFORMANCE METRICS
================================================================================

L3 ELITE 8 MODEL (2016-2025 validation):
  • WITH_SEEDS: 0.902 ROC-AUC
  • NO_SEEDS: 0.897 ROC-AUC
  • Outperforms BartTorvik, ESPN Public, PowerRank by ~33%

H2H ENSEMBLE:
  • Training accuracy: ~73% (2008-2025)
  • Validation: Cross-validated across years
  • Feature importance: BartTorvik rank, KenPom efficiency metrics

CONTRARIAN EDGE:
  • Teams with consensus gap >5pp: 67% Elite 8 rate
  • Teams with consensus gap ≈0pp: 28% Elite 8 rate
  • Blue bloods (Kansas, Duke): 50% fade discount


================================================================================
                    QUICK REFERENCE
================================================================================

PRE-TOURNAMENT COMMANDS:
  cd L4
  python 01_tournament_simulator.py    # Generate bracket probabilities
  python 02_calcutta_optimizer.py      # Generate auction strategy

DURING TOURNAMENT COMMANDS:
  cd L4
  # Update data/actual_results.csv after each round
  python 03_predict_next_round.py --next-round R32
  python 03_predict_next_round.py --next-round S16
  python 03_predict_next_round.py --next-round E8
  python 03_predict_next_round.py --next-round FF
  python 03_predict_next_round.py --next-round Championship

VIEW OUTPUTS:
  cat outputs/01_tournament_simulator/game_probabilities.txt
  cat outputs/03_live_predictions/r32_probabilities.txt
  open outputs/01_tournament_simulator/round_probabilities.csv
  open outputs/03_live_predictions/r32_probabilities.csv

FIND TEAM INDEX:
  grep -i "duke" utils/teamsIndex.csv
  open utils/teamsIndex.csv


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


================================================================================
AUTHOR
================================================================================

Ryan Browder
March Madness Computron

Layer 4 transforms probabilistic predictions into actionable strategy for
bracket pools, Calcutta auctions, and live tournament betting.

================================================================================
