================================================================================
L2 DATA PIPELINE - README
================================================================================

This layer joins clean L1 data sources into analysis-ready datasets for L3
modeling and L4 simulation.

================================================================================
OVERVIEW
================================================================================

**Purpose:** Combine multiple basketball analytics sources into unified dataset

**Key Functions:**
- Join data sources on (Year, Index) using teamsIndex.csv
- Invert rank columns (lower rank = better → higher value = better)
- Normalize all ranks to consistent 0-364 scale
- Preserve PowerRank as continuous rating (NOT normalized)
- Create unified training set: 2008-2025, 5 sources
- Add tournament metadata (seeds, regions) for predictions
- Optional: Add Vegas odds for VEGsemble blending

**Inputs:** Clean L1 data (bartTorvik, kenPom, espnBPI, masseyComposite, powerRank)
**Outputs:** Unified training dataset + prediction dataset for L3

**Major Update (March 2026):**
- PowerRank expanded to 2002-2026 (from Dr. Ed Feng)
- Collapsed LONG/RICH into single unified dataset
- LRMCB removed (source discontinued, no 2026 data)
- PowerRank now core 5th source (was optional)
- Fixed tournamentSeed data leakage

================================================================================
FILE STRUCTURE
================================================================================

L2/
├── create_training_sets_L2.py      # Creates unified training data
├── create_predict_set_L2.py        # Creates 2026 prediction data
├── data/                            # L1 outputs (inputs to L2)
│   ├── bartTorvik/
│   │   ├── bartTorvik_analyze_L2.csv    # Historical (2008-2025)
│   │   └── bartTorvik_predict_L2.csv    # Current season (2026)
│   ├── kenPom/
│   │   ├── kenPom_analyze_L2.csv
│   │   └── kenPom_predict_L2.csv
│   ├── espnBPI/
│   │   ├── espnBPI_analyze_L2.csv
│   │   └── espnBPI_predict_L2.csv
│   ├── masseyComposite/
│   │   ├── masseyComposite_analyze_L2.csv
│   │   └── masseyComposite_predict_L2.csv
│   ├── powerRank/                       # Core source (2002-2026)
│   │   ├── powerRank_analyze_L2.csv     # PowerRank RATINGS (not ranks)
│   │   └── powerRank_predict_L2.csv
│   ├── bracketology/                    # Tournament seeds/regions
│   │   └── espn_bracketology_2026.csv
│   └── vegasOdds/                       # Optional
│       └── vegasOdds_analyze_L2.csv
└── README.txt                           # This file

Outputs to:
../L3/data/trainingData/
    └── training_set_unified.csv         # 2008-2025, 5 sources

../L3/data/predictionData/
    └── predict_set_2026.csv             # 2026, 5 sources + metadata

================================================================================
EXECUTION SEQUENCE
================================================================================

**STEP 1: Verify L1 Data Exists**

Check that L1 transform scripts have run:

```bash
cd L2/data

# Required files (must exist)
ls bartTorvik/bartTorvik_analyze_L2.csv
ls kenPom/kenPom_analyze_L2.csv
ls espnBPI/espnBPI_analyze_L2.csv
ls masseyComposite/masseyComposite_analyze_L2.csv
ls powerRank/powerRank_analyze_L2.csv

ls bartTorvik/bartTorvik_predict_L2.csv
ls kenPom/kenPom_predict_L2.csv
ls espnBPI/espnBPI_predict_L2.csv
ls masseyComposite/masseyComposite_predict_L2.csv
ls powerRank/powerRank_predict_L2.csv

# Optional files (OK if missing)
ls vegasOdds/vegasOdds_analyze_L2.csv  # Optional
```

If required files are missing, run L1 transforms first.

---

**STEP 2: Configure Feature Toggles**

Open predict script and set toggle at top (~line 23):

```python
# create_predict_set_L2.py
INCLUDE_VEGAS_ODDS = False  # Set True to enable Vegas blending
```

**Note:** No other toggles needed. PowerRank is now a required core source.

---

**STEP 3: Generate Training Dataset**

```bash
cd L2
python create_training_sets_L2.py
```

**What it does:**
1. Loads all historical data (2008-2025)
2. Joins 5 core sources (bartTorvik, kenPom, espnBPI, masseyComposite, powerRank)
3. Filters to tournament teams only
4. Drops tournamentSeed and tournamentOutcome (prevents data leakage)
5. Inverts rank columns (higher = better)
6. Normalizes ranks to 0-364 scale (PowerRank excluded - it's a rating)
7. Outputs to L3/data/trainingData/training_set_unified.csv

**Expected output:**
```
UNIFIED TRAINING SET (2008-2025, 5 sources):
  Sources: bartTorvik, kenPom, espnBPI, masseyComposite, powerRank
  Rows: ~1,147 (tournament teams)
  Features: 52-54
  PowerRank coverage: 2002-2026 (using 2008-2025 for this dataset)
```

**Success indicators:**
✓ No teams lost during joins
✓ Rank normalization complete (all ranks now 0-364)
✓ PowerRank preserved as rating (NOT normalized)
✓ tournamentSeed dropped (no data leakage)
✓ One CSV file created: training_set_unified.csv

---

**STEP 4: Generate Prediction Dataset**

```bash
cd L2
python create_predict_set_L2.py
```

**What it does:**
1. Loads all 2026 prediction data
2. Joins 5 core sources (bartTorvik, kenPom, espnBPI, masseyComposite, powerRank)
3. Adds ESPN bracketology (seeds, regions)
4. Optionally adds Vegas odds
5. Inverts rank columns (higher = better)
6. Normalizes ranks to 0-364 scale (PowerRank excluded - it's a rating)
7. Outputs to L3/data/predictionData/predict_set_2026.csv

**Expected output:**
```
Total teams: 365 (all D1)
Feature columns: 54-57 (depends on Vegas toggle)
Projected tournament teams: 64
Tournament regions: EAST, MIDWEST, SOUTH, WEST
PowerRank range: -28 to +24 (continuous rating, not normalized)
```

**Success indicators:**
✓ 365 teams (full D1 field)
✓ 64 projected tournament teams
✓ No teams lost during joins
✓ Rank normalization complete (all ranks now 0-364)
✓ PowerRank preserved as rating (NOT normalized)
✓ One CSV file created in L3/data/predictionData/

---

**STEP 5: Verify Scale Alignment**

Check that training and prediction are on same scale:

```bash
# Check bartTorvik_Rk (5th column after Year,Team,Index,tournamentSeed drop)
# Both should show 364.0
grep "^2026,Michigan," ../L3/data/predictionData/predict_set_2026.csv | cut -d',' -f4
grep "^2008,Kansas," ../L3/data/trainingData/training_set_unified.csv | cut -d',' -f4
```

**Note:** Column position may vary - use `head -1` to find bartTorvik_Rk position.

If both show ~364, scale normalization worked correctly.

================================================================================
FEATURE TOGGLES
================================================================================

**INCLUDE_VEGAS_ODDS** (Predict script only)

When to use TRUE:
- Have DraftKings odds for current season
- Want to blend model + Vegas in L4
- Testing VEGsemble performance

When to use FALSE:
- Vegas odds not scraped yet
- Testing model-only performance
- Don't want blending complexity

**Current Recommendation for 2026:**
```python
INCLUDE_VEGAS_ODDS = False  # Optional - your choice
```

================================================================================
DATA SOURCES
================================================================================

**Required (Core 5 sources):**
- bartTorvik: Team efficiency ratings, tempo, luck (2008-2025)
- kenPom: Adjusted efficiency, tempo, strength of schedule (2008-2025)
- espnBPI: Basketball Power Index ratings (2008-2025)
- masseyComposite: Composite of 50+ ranking systems (2001-2025)
- powerRank: Power ratings from Dr. Ed Feng (2002-2026)

**Optional (Current season):**
- vegasOdds: DraftKings Elite 8/F4/Championship probabilities

**Removed:**
- LRMCB: Luke Rettig's rankings (discontinued, no 2026 data)

================================================================================
KEY TRANSFORMATIONS
================================================================================

**1. Rank Inversion**

Problem: Lower rank = better team (rank 1 = best)
Solution: Invert to higher value = better team

```
Original: Duke rank 1, Youngstown St rank 350
Inverted: Duke rank 364, Youngstown St rank 15
```

Why: pct_diff formula assumes higher = better

**2. Rank Normalization**

Problem: Different years have different max ranks
- 2008: max rank 319 (fewer D1 teams)
- 2026: max rank 364 (modern D1 field)

Solution: Normalize all to 0-364 scale

```
2008 Kansas: (319/319) × 364 = 364
2026 Michigan: (364/364) × 364 = 364
```

Why: Prevents model extrapolation errors

**3. PowerRank Preservation**

**CRITICAL:** PowerRank is a RATING, not a RANK

What it is:
- Continuous strength metric (not ordinal position)
- Mean-centered around 0 by design
- Range: -28 to +24 (historically)
- Negative values = weaker teams
- Already directional (high = good)

What we DON'T do:
- ✗ Invert (already high = good)
- ✗ Normalize to 0-364 (destroys meaning)
- ✓ Preserve as-is for modeling

Example values:
- Florida 2025: 23.79 (elite, championship contender)
- Duke 2025: 21.45 (strong, likely high seed)
- Average team: ~0
- Weak D1 team: -10 to -20

**4. Team Name Standardization**

All sources use teamsIndex.csv for consistent team naming:
- "Michigan" (not "U of Michigan" or "UM")
- "Ohio St." (not "Ohio State" or "OSU")

Clean team names → successful joins across all sources

================================================================================
DATASETS PRODUCED
================================================================================

**training_set_unified.csv**
- Years: 2008-2025 (18 tournaments, excludes 2020)
- Sources: 5 (bartTorvik, kenPom, espnBPI, masseyComposite, powerRank)
- Teams: ~1,147 tournament teams
- Features: 52-54 columns
- Use: Train models with unified feature set
- PowerRank: Continuous rating preserved as-is
- Metadata: tournamentSeed/Outcome dropped (no leakage)

**predict_set_2026.csv**
- Year: 2026 (current season)
- Sources: 5 (same as training)
- Teams: 365 (all D1)
- Features: 54-57 columns
- Extras: tournamentSeed, tournamentRegion, [vegas odds]
- Use: Apply trained models to generate 2026 predictions
- PowerRank: Continuous rating preserved as-is

================================================================================
VALIDATION CHECKS
================================================================================

**After running training script:**

✓ Console shows "No tournament teams lost"
✓ Console shows "Rank normalization complete"
✓ Console shows "PowerRank excluded (it's a rating, not a rank)"
✓ One file created: training_set_unified.csv
✓ Row count: ~1,147
✓ Feature count: 52-54
✓ No tournamentSeed column in output

**After running predict script:**

✓ Console shows "No teams lost"
✓ Console shows "Rank normalization complete"
✓ Console shows "PowerRank excluded (it's a rating, not a rank)"
✓ One file created: predict_set_2026.csv
✓ Row count: 365 teams
✓ 64 teams with tournamentSeed
✓ PowerRank values NOT normalized (should see ~23.8, not 364)

**Scale alignment check:**

```bash
# Check ranks ARE normalized (should show ~364)
python3 << 'EOF'
import pandas as pd
train = pd.read_csv('../L3/data/trainingData/training_set_unified.csv')
pred = pd.read_csv('../L3/data/predictionData/predict_set_2026.csv')
print(f"Training bartTorvik_Rk max: {train['bartTorvik_Rk'].max():.1f}")
print(f"Prediction bartTorvik_Rk max: {pred['bartTorvik_Rk'].max():.1f}")
print(f"Training PowerRank max: {train['PowerRank'].max():.1f}")
print(f"Prediction PowerRank max: {pred['PowerRank'].max():.1f}")
EOF
```

Expected:
- bartTorvik_Rk max: 364.0 (both)
- PowerRank max: ~23-24 (both) ← NOT 364!

================================================================================
TROUBLESHOOTING
================================================================================

**Issue: "Required input file not found"**
Solution: Run L1 transform scripts first to generate L2 inputs

**Issue: "Teams lost after [source] join"**
Solution: Check teamsIndex.csv for team name mismatches
         Run L1 transforms with updated teamsIndex

**Issue: "PowerRank column not found"**
Solution: Verify powerRank_analyze_L2.csv and powerRank_predict_L2.csv exist
         Check PowerRank L1 transform completed successfully

**Issue: "PowerRank values are 364, not 23"**
Problem: PowerRank got normalized when it shouldn't
Solution: Verify exclusion logic in scripts:
         `rank_cols = [c for c in df.columns if ... and c != 'PowerRank']`

**Issue: "Training and prediction max ranks don't match"**
Solution: Normalization didn't run - check script has normalization code
         Should normalize after inversion: (val/max) × 364

**Issue: "tournamentSeed still in training data"**
Problem: Old script running or TOURNAMENT_COLS not updated
Solution: Replace script, verify TOURNAMENT_COLS = ['tournamentSeed', 'tournamentOutcome']

**Issue: "Different feature counts in training vs prediction"**
Solution: Verify both scripts use same 5 core sources
         PowerRank should be required in both

**Issue: "Duplicate team-seasons found"**
Solution: This is normal - script handles it automatically
         Keeps first occurrence, drops duplicates

================================================================================
OUTPUTS FOR L3
================================================================================

L3 modeling layer expects:

**For training:**
- training_set_unified.csv
- Join to game outcomes for labels
- Features only (no tournament metadata)
- Ranks normalized to 0-364
- PowerRank preserved as continuous rating

**For prediction:**
- predict_set_2026.csv
- All 365 D1 teams
- Same features as training
- Ranks normalized to 0-364
- PowerRank preserved as continuous rating
- Includes tournamentSeed/Region for L4 simulation
- Optionally includes Vegas odds for blending

L3 will:
- Train models on historical data
- Generate 2026 predictions
- Validate blend weights (if using Vegas)

================================================================================
REGENERATION WORKFLOW
================================================================================

**When to regenerate:**
- New L1 data available (updated rankings)
- Toggle settings changed (Vegas)
- Team name fixes applied
- Bug fixes in transformation logic

**Full regeneration:**

```bash
# 1. Regenerate training set
cd L2
python create_training_sets_L2.py

# 2. Regenerate prediction set
python create_predict_set_L2.py

# 3. Verify scale alignment
head -1 ../L3/data/trainingData/training_set_unified.csv
# Find bartTorvik_Rk column position, then:
grep "^2026,Michigan," ../L3/data/predictionData/predict_set_2026.csv | cut -d',' -f[N]
grep "^2008,Kansas," ../L3/data/trainingData/training_set_unified.csv | cut -d',' -f[N]
# Both should show ~364

# 4. Verify PowerRank NOT normalized
# Find PowerRank column position, then:
grep "^2025,Florida," ../L3/data/trainingData/training_set_unified.csv | cut -d',' -f[N]
# Should show ~23.79, NOT 364

# 5. Retrain L3 models (critical!)
cd ../L3
# [Run L3 training scripts]
```

**Important:** After regenerating L2 data, ALWAYS retrain L3 models.
Models trained on old data won't work with new data structure.

================================================================================
POWERRANK SPECIFICS
================================================================================

**What PowerRank Is:**
- Strength rating (not rank position)
- Created by Dr. Ed Feng (thepowerrank.com)
- Mean-centered around 0
- Based on margin of victory adjusted for opponents
- Stable methodology since 2002

**Scale Interpretation:**
- ~20+: Elite, championship contender
- 15-20: Strong, likely high seed
- 10-15: Solid, mid-tier tournament team
- 5-10: Bubble/lower seeds
- 0-5: Weak tournament team or strong mid-major
- <0: Typically non-tournament teams
- Range: -28 to +24 (historical extremes)

**Usage in Models:**
- Use as-is (continuous variable)
- Compatible with tree-based models (RF, XGBoost)
- May want to standardize for neural nets
- Consider PowerRank_diff for H2H models
- Shows magnitude of gaps (not just order)

**Why Not Normalize:**
- Already on meaningful scale
- Mean-centered by design
- Negative values have meaning
- Comparable across years
- Normalizing would destroy interpretability

================================================================================
COMPARISON: OLD vs NEW PIPELINE
================================================================================

**Before (Feb 2026):**
- Two datasets: LONG (2008-2025, 4 sources) + RICH (2016-2025, 5-6 sources)
- PowerRank: Optional, only 2016+ (9 years)
- LRMCB: Toggle for optional inclusion
- Training samples: 612 (RICH) or 1,147 (LONG)
- tournamentSeed: Kept as feature (data leakage!)

**After (March 2026):**
- One dataset: UNIFIED (2008-2025, 5 sources)
- PowerRank: Required, 2002-2026 (24 years, using 2008-2025)
- LRMCB: Removed (discontinued)
- Training samples: 1,147 (87% more than old RICH)
- tournamentSeed: Properly dropped

**Benefits:**
- +87% training data vs old RICH view
- Simpler architecture (one dataset, not two)
- No LRMCB dependency
- PowerRank depth (2002-2026 coverage)
- Proper feature handling (no leakage)
- Cleaner documentation

================================================================================
QUICK START
================================================================================

```bash
# Configure Vegas toggle (edit create_predict_set_L2.py)
# INCLUDE_VEGAS_ODDS = False (or True if have odds)

cd L2

# Generate training dataset
python create_training_sets_L2.py

# Generate prediction dataset
python create_predict_set_L2.py

# Verify outputs
ls -lh ../L3/data/trainingData/
ls -lh ../L3/data/predictionData/

# Check column positions
head -1 ../L3/data/trainingData/training_set_unified.csv

# Verify rank normalization (find bartTorvik_Rk column)
grep "^2026,Michigan," ../L3/data/predictionData/predict_set_2026.csv | cut -d',' -f[N]
grep "^2008,Kansas," ../L3/data/trainingData/training_set_unified.csv | cut -d',' -f[N]

# Verify PowerRank NOT normalized (find PowerRank column)
grep "^2025,Florida," ../L3/data/trainingData/training_set_unified.csv | cut -d',' -f[N]

# Proceed to L3
cd ../L3
# [Train models]
```

================================================================================
VERSION HISTORY
================================================================================

2025-01-29: Initial creation (training sets script)
2025-01-30: Added prediction set script
2025-02-09: Added rank inversion for pct_diff compatibility
2025-02-09: Added Vegas odds integration
2025-02-09: Added Vegas odds toggle
2025-02-10: Added rank normalization to 0-364 scale (CRITICAL FIX)
2025-02-10: Fixed corrupted team names (bartTorvik "vs." issue)
2025-02-10: Added LRMCB toggle
2025-03-05: MAJOR UPDATE - PowerRank integration
            - PowerRank expanded to 2002-2026
            - Collapsed LONG/RICH → UNIFIED
            - Removed LRMCB (discontinued)
            - PowerRank now core source (not optional)
            - Fixed tournamentSeed data leakage
            - PowerRank excluded from normalization (it's a rating)

================================================================================
AUTHOR
================================================================================

Ryan Browder
March Madness Computron

For questions or issues, check:
- L2 script comments
- L1 README for input requirements
- L3 README for output requirements
- L2_unified_pipeline_documentation.txt for detailed migration guide

================================================================================
