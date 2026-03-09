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
  Provides executable auction strategy for Calcutta pool bidding using:
    • Historical auction data (2013-2025)
    • L4.01 tournament simulation probabilities
    • Tier-based bid guidance by auction phase
    • Strategic portfolio paths

CORE PRINCIPLE:
  You're playing a budget allocation game with $100 to build a 76+ point
  portfolio in a winner-take-all auction against 6-7 competitors. Success
  requires: (1) securing anchor teams, (2) building supporting cast with
  value plays, (3) adapting to auction flow dynamically.

WHEN TO RUN:
  Before your Calcutta auction (after tournament bracket is announced).
  Must run 01_tournament_simulator.py first.

HOW TO RUN:
  cd L4
  python 02_calcutta_optimizer.py

WHAT IT DOES:
  1. Loads tournament simulator probabilities (from L4.01)
  2. Loads historical auction data (2013-2025)
  3. Calculates expected points per team (base + upset bonuses)
  4. Assigns tiers (ANCHOR / FILL / VALUE / FADE)
  5. Generates phase-based bid guidance (Phase 1/2/3)
  6. Builds strategic portfolio paths that fit $100 budget
  7. Exports two files: team_bidders_2026.csv (auction cheat sheet) and
     team_valuations_2026.csv (full analysis)

CALCUTTA RULES:
  Entry: $20 per player
  Budget: $100 fake money per player
  Pot: $20 × N players (typically ~$150)
  Payout: 70% to 1st place, 30% to 2nd place
  Teams: 60 teams (excludes 16-seeds)
  Format: Blind auction (teams called randomly)
  
  Scoring:
    R64: 2 pts  | R32: 4 pts  | S16: 6 pts  | E8: 8 pts
    F4: 10 pts  | Championship: 12 pts
    
    Upset Bonus: If lower seed beats higher seed, add seed difference to points
    Example: Nevada (10) beats Gonzaga (2) in R32 = 4 + 8 = 12 pts

OUTPUTS:
  All files saved to: outputs/02_calcutta_optimizer/
  
  team_bidders_2026.csv - AUCTION CHEAT SHEET (use during auction)
    Columns: Team, Seed, Region, Expected_Points, P_R64-P_WIN_Championship,
             Tier, Bid_Phase1, Bid_Phase2, Bid_Phase3, Efficiency
    Sort by: Team (alphabetical) for quick lookup during auction
  
  team_valuations_2026.csv - FULL ANALYSIS (for pre-auction prep)
    Contains: Everything in bidders file + historical data
    Use: Pre-auction research, post-auction review, understanding WHY
  
  historical_seed_performance.csv - Historical ROI by seed
  historical_winners.csv - Past winners' portfolios

KEY FEATURES:
  ✓ Tier-based team assignments (ANCHOR/FILL/VALUE)
  ✓ Phase-based dynamic bidding (0-2 teams / 3-5 teams / 6+ teams)
  ✓ Three strategic portfolio paths (~$100 each)
  ✓ Historical ROI patterns (fade overpriced seeds)
  ✓ Model advantage identification (+8 vs historical avg)
  ✓ Tournament type adjustment (2026 = 80% CHALK)

TIER SYSTEM:

  ANCHOR (Pick 1-2): Your foundation, high-floor teams
    Criteria: 20+ pts with +5 advantage OR 15+ pts with +8 advantage
    Examples: Duke (28.8 pts), Houston (22.8 pts), Iowa St (17.2 pts)
    Strategy: Secure one early, build around them
  
  FILL (Pick 3-4): Solid supporting cast with good model advantage
    Criteria: 12+ pts with +4 advantage OR 10+ pts with +6 advantage
    Examples: Gonzaga (14.4 pts), Arkansas (13.8 pts), Kansas (13.0 pts)
    Strategy: Core of your portfolio, pay market prices
  
  VALUE (Pick 2-3): Efficiency plays to round out portfolio
    Criteria: 8+ pts with 1.1+ pts/$ OR 8+ pts with +3 advantage
    Examples: UCLA (8.5 pts), Wisconsin (10.1 pts), Louisville (9.2 pts)
    Strategy: Only buy at discounts, maximize points per dollar
  
  FADE: Teams model doesn't like or with negative advantage
    Strategy: Pass entirely

PHASE-BASED BIDDING:

  Phase 1 (0-2 teams, no anchor yet):
    MINDSET: "I need an anchor or I'm toast"
    ANCHORS: Pay up (market + $3-5)
    FILLS: Pass or bid cheap (save budget)
    VALUES: Pass (focus on foundation)
  
  Phase 2 (3-5 teams, have anchor OR building deep roster):
    MINDSET: "Building my core"
    ANCHORS: Pay market only (no premium)
    FILLS: Pay market prices
    VALUES: Start bidding market prices
  
  Phase 3 (6+ teams, close to 76 points):
    MINDSET: "Value hunting to cross threshold"
    ANCHORS: Pass (already have foundation)
    FILLS: Only bid below market
    VALUES: Best opportunities here

STRATEGIC PORTFOLIO PATHS:

  PATH A: PREMIUM ANCHOR (~$100)
    Strategy: 1 elite anchor + deep value supporting cast
    Example: Duke $43 + 6 value/fill teams at ~$9 each
    When: Premium anchor available at reasonable price early
    Risk: High variance (Duke carries you or you're screwed)
  
  PATH B: DUAL MID-TIER (~$96)
    Strategy: 2 solid anchors + balanced supporting cast
    Example: Houston $30 + Iowa St $18 + 5 fills/values
    When: Premium anchors too expensive, pivot to balance
    Risk: Medium variance (diversified foundation)
  
  PATH C: DEEP VALUE (~$95)
    Strategy: 1 cheap anchor + 7-8 value/fill teams
    Example: Iowa St $18 + 7 teams at ~$11 each
    When: Everything expensive, wait for deals
    Risk: Low variance (lots of shots on goal)

HISTORICAL SEED ROI (2013-2025):

  Market systematically overpays for high seeds:
    Seed 6: -74.5% ROI (worst - AVOID)
    Seed 2: -71.4% ROI
    Seed 3: -65.4% ROI
    Seed 1: -61.7% ROI
  
  Model edge: Identifies WHICH high seeds are actually good
    Duke (1): +11.7 advantage over avg 1-seed
    Houston (2): +14.0 advantage over avg 2-seed
    Iowa St (3): +10.4 advantage over avg 3-seed

WINNING REQUIREMENTS:

  Historical Benchmarks (2013-2025):
    Average winning total: 86.1 points
    25th percentile winner: 76 points (minimum to compete)
    Typical portfolio: 7-8 teams, $90-100 spent
  
  Target Portfolio:
    ✓ 76-85 expected points
    ✓ $90-100 spent (don't leave money on table)
    ✓ 1-2 ANCHOR tier teams
    ✓ 3-5 FILL tier teams
    ✓ 1-3 VALUE tier teams
    ✓ Average model advantage: +5 or better

AUCTION EXECUTION:

  Pre-Auction (15 min):
    □ Run optimizer, review terminal output
    □ Pick your strategic path (A/B/C)
    □ Open team_bidders_2026.csv, sort by Team
    □ Print tracking template
  
  During Auction:
    □ Track: teams owned, budget left, points total, have anchor?
    □ Team called → Find in CSV → Check phase → Bid to max
    □ Stop at phase max (discipline beats FOMO)
    □ If bidding exceeds max → LET THEM HAVE IT
    □ Update tracking after each purchase
  
  Decision Rules:
    1. Find team in bidders CSV
    2. Check tier (ANCHOR / FILL / VALUE / FADE)
    3. Check your phase (do you have anchor? how many teams?)
    4. Look at appropriate phase bid column
    5. That's your max bid

BLIND AUCTION ADAPTATION:

  Key Challenge: You might need an anchor but it comes late
  
  Solution: Track "have anchor?" not just team count
  
  Modified Phase Logic:
    Phase 1 = No anchor yet (regardless of team count)
    Phase 2 = Have anchor OR gave up on premium anchors
    Phase 3 = 6+ teams
  
  Example: If you have 4 weak teams but no anchor, and Duke comes up:
    → You're in "Phase 1" for anchors (bid aggressively)
    → But "Phase 3" for values (pass or bid cheap)

CHALK YEAR STRATEGY (2026):

  Tournament Type: 80% CHALK, 15% NORMAL, 5% CHAOS
  
  Implications:
    • High seeds (1-4) MORE valuable than usual
    • Upset bonuses LESS likely
    • Low seeds (12-15) have MINIMAL value
    • Anchors are CRITICAL (reduce variance)
  
  Adjustments:
    • Premium on high-floor anchors (Duke, Houston)
    • Fade overseeded teams (Florida, Purdue per L3 analysis)
    • Don't chase 12-15 seeds even if cheap
    • Build around 1-2 anchors, not deep value strategy

COMMON MISTAKES:

  DON'T:
    × Buy 2+ premium anchors (Duke + Michigan = $90, no money left)
    × Chase every team you like (budget discipline matters)
    × Pay Phase 1 prices in Phase 3 (adapt to portfolio state)
    × Ignore historical ROI patterns (don't overpay for 2/6 seeds)
    × Build portfolio without anchor (need foundation)
    × Underspend (leaving $20 on table = giving up ~15 points)
  
  DO:
    ✓ Pick ONE strategic path before auction (A/B/C)
    ✓ Track: teams owned, budget left, points total, have anchor?
    ✓ Use phase bids as MAX, not TARGET (get discounts when possible)
    ✓ Adapt if your path isn't working (pivot A→B→C as needed)
    ✓ Focus on model advantage (+8 or better)
    ✓ Cross 76-point threshold with budget remaining


================================================================================
              L4.03 - LIVE ROUND PREDICTOR
================================================================================

PURPOSE:
  Generate win probabilities with line value targets for betting decisions.
  Works BOTH pre-tournament (R64) and during tournament (R32+).

WHEN TO RUN:
  
  PRE-TOURNAMENT (Selection Sunday):
    After bracket is announced, before tournament starts.
    Use for R64 betting analysis.
  
  DURING TOURNAMENT (After each round):
    After R64, R32, S16, E8, FF complete.
    Use for next round betting analysis.

HOW TO RUN:

  PRE-TOURNAMENT (No actual results needed):
    cd L4
    python 03_predict_next_round.py --next-round R64
  
  DURING TOURNAMENT (Requires actual_results.csv):
    cd L4
    python 03_predict_next_round.py --next-round R32
    python 03_predict_next_round.py --next-round S16
    python 03_predict_next_round.py --next-round E8
    python 03_predict_next_round.py --next-round FF
    python 03_predict_next_round.py --next-round Championship

WHAT IT DOES:
  1. Loads L3 H2H ensemble models
  2. Loads prediction features for remaining teams
  3. For R64: Generates matchups from bracket structure
     For R32+: Determines matchups from actual results
  4. Calculates H2H win probabilities for each matchup
  5. Computes line value targets (odds needed for different edge levels)
  6. Calculates expected value per $1 bet at each tier
  7. Provides unit sizing recommendations
  8. Exports betting sheet with actionable recommendations

INPUT FILE: data/actual_results.csv (for R32+ only, not needed for R64)

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
  
  r64_probabilities.txt - R64 betting sheet (pre-tournament)
  r32_probabilities.txt - R32 betting sheet (after R64 completes)
  s16_probabilities.txt - Sweet 16 betting sheet (after R32 completes)
  e8_probabilities.txt - Elite 8 betting sheet (after S16 completes)
  ff_probabilities.txt - Final Four betting sheet (after E8 completes)
  championship_probabilities.txt - Championship betting sheet (after FF)
  
  Each file includes:
    • Game-by-game probabilities
    • Line value targets (what odds you need for different bet tiers)
    • Expected value per $1 bet
    • Unit sizing recommendations
    • Betting tier classifications

KEY FEATURES:
  ✓ R64 pre-tournament analysis (no results needed)
  ✓ Live tournament analysis (requires actual results)
  ✓ Line value targets (odds needed for MAX/STRONG/VALUE bets)
  ✓ Expected value calculations per $1 bet
  ✓ Unit sizing recommendations (6-10 units, 2-4 units, 1-2 units)
  ✓ Two-tier classification system (confidence + line value)

TWO CLASSIFICATION SYSTEMS:

  1. CONFIDENCE TIER (Based on model probability)
     • LOCK (>90%): Very high model confidence
     • STRONG (70-90%): High model confidence
     • LEAN (60-70%): Modest model confidence
     • SLIGHT (55-60%): Low model confidence
     • TOSS-UP (<55%): Near 50/50
     
     → Informational only, DOES NOT determine bet size
  
  2. LINE VALUE TIER (Based on edge you get from odds)
     • MAX BET (8% edge): Rare, bet heavy
     • STRONG BET (5% edge): Common, standard bet
     • VALUE BET (2% edge): Most common, minimum bet
     • Marginal (<2% edge): Pass
     
     → DETERMINES bet size (see unit recommendations)

UNIT SIZING RECOMMENDATIONS:

  Line Value Tier    Edge    Units       $ (if unit=$0.50)    Frequency
  ───────────────────────────────────────────────────────────────────────
  MAX BET            8%+     6-10        $3-5                 1-3/tournament
  STRONG BET         5-8%    2-4         $1-2                 8-15/tournament
  VALUE BET          2-5%    1-2         $0.50-1              10-20/tournament
  Pass               <2%     0           $0                   Most games
  
  CRITICAL: Bet size is determined by LINE VALUE tier (edge), NOT confidence!
  
  Example:
    Duke 98% [LOCK confidence] at -5000 odds
      → Edge: ~0% (no value)
      → Line value tier: PASS
      → Bet: 0 units
    
    St Louis 58% [SLIGHT confidence] at +150 odds
      → Edge: ~18% (huge value)
      → Line value tier: MAX BET
      → Bet: 6-10 units

READING THE OUTPUT:

  Example game output:
    
    (5) Tennessee vs (3) Gonzaga
      → Gonzaga 74.9% [STRONG]
      LINE VALUE:
        MAX BET: -210 or better (EV: $0.109 | 6-10 units)
        STRONG BET: -286 or better (EV: $0.063 | 2-4 units)
        VALUE BET: -488 or better (EV: $0.024 | 1-2 units)
        BREAKEVEN: -900 (EV: $0.000 | 0 units - PASS)
  
  How to use this:
    1. Check your sportsbook: Gonzaga -280
    2. Compare to targets: -280 is better than -286 (STRONG BET tier)
    3. Bet recommended units: 2-4 units (if your unit is $0.50, bet $1-2)
    4. Expected profit: $1.50 × 0.063 = $0.095 per bet

EXPECTED VALUE:

  Every line value target shows expected profit per $1 bet:
  
    STRONG BET: -286 or better (EV: $0.063 | 2-4 units)
    
    Interpretation:
      • If you bet at -286, expect $0.063 profit per $1 wagered
      • If you bet $2 (4 units at $0.50), expect $0.126 profit
      • Over 20 similar bets, expect ~$2.52 profit
  
  Typical tournament EV (betting 25-30 games with proper sizing):
    • Conservative: $1-2 profit
    • Moderate: $2-4 profit
    • Aggressive: $3-6 profit
    • ROI: 4-6% (excellent in sports betting)

PRE-TOURNAMENT WORKFLOW:

  Selection Sunday:
    1. python 01_tournament_simulator.py
       → Use game_probabilities.txt to fill bracket
    
    2. python 03_predict_next_round.py --next-round R64
       → Use r64_probabilities.txt for betting with line value targets
    
    3. python 02_calcutta_optimizer.py (if applicable)
       → Use team_valuations_2026.csv for auction

DURING TOURNAMENT WORKFLOW:

  After R64 completes (Saturday evening):
    1. Update data/actual_results.csv with all R64 results
    2. python 03_predict_next_round.py --next-round R32
    3. Review outputs/03_live_predictions/r32_probabilities.txt
    4. Compare line value targets to sportsbook odds
    5. Place R32 bets where you have good value
  
  After R32 completes:
    1. Update data/actual_results.csv with R32 results
    2. python 03_predict_next_round.py --next-round S16
    3. Review s16_probabilities.txt
    4. Place S16 bets
  
  Continue pattern for E8, FF, Championship

UPDATING actual_results.csv:

  After each game (example: Duke beats Central Arkansas):
  
    # Step 1: Look up Index values
    grep -i "duke" utils/teamsIndex.csv
    → 76,Duke
    
    grep -i "central arkansas" utils/teamsIndex.csv
    → 45,Central Arkansas
    
    # Step 2: Add line to actual_results.csv
    echo "R64,76,Duke,45,Central Arkansas" >> data/actual_results.csv
  
  After full round completes:
    python 03_predict_next_round.py --next-round R32


================================================================================
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

  Pre-auction preparation:
    • Pick strategic path: A (premium anchor), B (dual mid-tier), or C (deep value)
    • Open team_bidders_2026.csv, sort by Team (alphabetical)
    • Print tracking template (teams, budget, points, anchor?)
  
  Phase 1 execution (0-2 teams, no anchor):
    • ANCHOR tier: Pay up (market + $3-5), must secure foundation
    • FILL/VALUE tier: Pass or bid cheap (save budget)
    • Goal: Secure 1 anchor before moving to Phase 2
  
  Phase 2 execution (3-5 teams, have anchor):
    • ANCHOR tier: Pay market only (no premium)
    • FILL tier: Pay market prices (core of portfolio)
    • VALUE tier: Start bidding at market
    • Goal: Build supporting cast, get to 60+ points
  
  Phase 3 execution (6+ teams):
    • All tiers: Only bid below market (discounts only)
    • Goal: Cross 76-point threshold with remaining budget
  
  Budget discipline:
    • Use phase bids as MAX, not TARGET (get discounts when possible)
    • Stop at your phase max (discipline beats FOMO)
    • Track running totals: teams, budget, points, anchor status
    • Adapt path if auction doesn't go your way (A→B→C pivot)
  
  Risk management:
    • Don't buy 2+ premium anchors (leaves no budget for depth)
    • Focus on model advantage (+8 or better)
    • Target: 76-85 pts, $90-100 spent, 7-8 teams
    • Historical fade: Seeds 2/6 have worst ROI (-70%+)


LIVE BETTING STRATEGY:

  CRITICAL: Bet size determined by LINE VALUE tier (edge), NOT confidence tier!
  
  Unit sizing by LINE VALUE tier:
    • MAX BET (8%+ edge): 6-10 units - Rare, bet heavy
    • STRONG BET (5-8% edge): 2-4 units - Common, standard bet
    • VALUE BET (2-5% edge): 1-2 units - Most common, minimum bet
    • Marginal (<2% edge): 0 units - Pass entirely
  
  Confidence tier (INFORMATIONAL ONLY - does not determine bet size):
    • LOCK (>90%): High model confidence
    • STRONG (70-90%): Good model confidence
    • LEAN (60-70%): Modest model confidence
    • SLIGHT (55-60%): Low model confidence
    • TOSS-UP (<55%): Near 50/50
  
  How to use line value targets:
    1. Model says: Gonzaga 74.9%
    2. Sportsbook shows: Gonzaga -280
    3. Output shows:
       STRONG BET: -286 or better (EV: $0.063 | 2-4 units)
    4. Compare: -280 is better than -286 (STRONG BET tier)
    5. Bet: 2-4 units (if unit = $0.50, bet $1-2)
  
  Expected value per tournament (proper sizing):
    • Conservative (15-20 bets): $1-2 profit
    • Moderate (25-30 bets): $2-4 profit
    • Aggressive (35-40 bets): $3-6 profit
    • Target ROI: 4-6% (excellent in sports betting)
  
  Track performance:
    • Log all bets in Excel
    • Track: Date, Game, Model %, Line, Units, Tier, EV, Result, Profit
    • Calculate ROI by LINE VALUE tier (not confidence tier)
    • Adjust unit ranges based on results


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
  python 01_tournament_simulator.py         # Generate bracket probabilities
  python 03_predict_next_round.py --next-round R64  # R64 betting analysis
  python 02_calcutta_optimizer.py           # Generate auction strategy

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
  cat outputs/03_live_predictions/r64_probabilities.txt
  cat outputs/03_live_predictions/r32_probabilities.txt
  open outputs/01_tournament_simulator/round_probabilities.csv
  open outputs/03_live_predictions/r64_probabilities.csv

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
