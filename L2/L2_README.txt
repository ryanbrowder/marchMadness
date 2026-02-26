================================================================================
L2 DATA PIPELINE - README
================================================================================

This layer joins clean L1 data sources into analysis-ready datasets for L3
modeling and L4 simulation.

================================================================================
OVERVIEW
================================================================================

**Purpose:** Combine multiple basketball analytics sources into unified datasets

**Key Functions:**
- Join data sources on (Year, Index) using teamsIndex.csv
- Invert rank columns (lower rank = better → higher value = better)
- Normalize all ranks to consistent 0-364 scale
- Create two views: LONG (more years) and RICH (more features)
- Add tournament metadata (seeds, regions) for predictions
- Optional: Add Vegas odds for VEGsemble blending

**Inputs:** Clean L1 data (bartTorvik, kenPom, espnBPI, etc.)
**Outputs:** Training datasets + prediction dataset for L3

================================================================================
FILE STRUCTURE
================================================================================

L2/
├── create_training_sets_L2.py      # Creates historical training data
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
│   ├── LRMCB/                           # Optional (2016+)
│   │   ├── LRMCB_analyze_L2.csv
│   │   └── LRMCB_predict_L2.csv
│   ├── powerRank/                       # Optional (2016+)
│   │   ├── powerRank_analyze_L2.csv
│   │   └── powerRank_predict_L2.csv
│   └── vegasOdds/                       # Optional
│       └── vegasOdds_analyze_L2.csv
└── README.txt                           # This file

Outputs to:
../L3/data/trainingData/
    ├── training_set_long.csv            # 2008-2025, 4-5 sources
    └── training_set_rich.csv            # 2016-2025, 5-6 sources

../L3/data/predictionData/
    └── predict_set_2026.csv             # 2026, 5-6 sources

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

ls bartTorvik/bartTorvik_predict_L2.csv
ls kenPom/kenPom_predict_L2.csv
ls espnBPI/espnBPI_predict_L2.csv
ls masseyComposite/masseyComposite_predict_L2.csv

# Optional files (OK if missing)
ls LRMCB/LRMCB_analyze_L2.csv          # Optional
ls powerRank/powerRank_analyze_L2.csv  # Optional
ls vegasOdds/vegasOdds_analyze_L2.csv  # Optional
```

If required files are missing, run L1 transforms first.

---

**STEP 2: Configure Feature Toggles**

Open both scripts and set toggles at top (~line 26):

```python
# create_training_sets_L2.py
INCLUDE_LRMCB = True   # Set False if no 2026 LRMCB data

# create_predict_set_L2.py
INCLUDE_VEGAS_ODDS = False  # Set True to enable Vegas blending
INCLUDE_LRMCB = True        # Must match training script setting
```

**Important:** INCLUDE_LRMCB must match in BOTH scripts to ensure
              training and prediction have same features.

---

**STEP 3: Generate Training Datasets**

```bash
cd L2
python create_training_sets_L2.py
```

**What it does:**
1. Loads all historical data (2008-2025)
2. Creates LONG view (2008-2025, 4-5 sources)
3. Creates RICH view (2016-2025, 5-6 sources)
4. Filters to tournament teams only
5. Inverts rank columns (higher = better)
6. Normalizes ranks to 0-364 scale
7. Outputs to L3/data/trainingData/

**Expected output:**
```
LONG VIEW (2008-2025, 4-5 sources):
  Rows: 1,147 (tournament teams)
  Features: 49-51 (depends on sources)
  
RICH VIEW (2016-2025, 5-6 sources):
  Rows: 612 (tournament teams)
  Features: 51-52 (depends on sources)
```

**Success indicators:**
✓ No teams lost during joins
✓ Rank normalization complete (all ranks now 0-364)
✓ Two CSV files created in L3/data/trainingData/

---

**STEP 4: Generate Prediction Dataset**

```bash
cd L2
python create_predict_set_L2.py
```

**What it does:**
1. Loads all 2026 prediction data
2. Joins all available sources
3. Adds ESPN bracketology (seeds, regions)
4. Optionally adds Vegas odds
5. Inverts rank columns (higher = better)
6. Normalizes ranks to 0-364 scale
7. Outputs to L3/data/predictionData/

**Expected output:**
```
Total teams: 365 (all D1)
Feature columns: 53-56 (depends on sources/Vegas)
Projected tournament teams: 64
Tournament regions: EAST, MIDWEST, SOUTH, WEST
```

**Success indicators:**
✓ 365 teams (full D1 field)
✓ 64 projected tournament teams
✓ No teams lost during joins
✓ Rank normalization complete (all ranks now 0-364)
✓ One CSV file created in L3/data/predictionData/

---

**STEP 5: Verify Scale Alignment**

Check that training and prediction are on same scale:

```bash
# Quick check - both should show 364.0
grep "^2026,Michigan," ../L3/data/predictionData/predict_set_2026.csv | cut -d',' -f5
grep "^2008,Kansas," ../L3/data/trainingData/training_set_long.csv | cut -d',' -f4
```

If both show ~364, scale normalization worked correctly.

================================================================================
FEATURE TOGGLES
================================================================================

**INCLUDE_LRMCB** (Both scripts)

When to use TRUE:
- LRMCB data available for current season
- Want maximum feature set
- LRMCB data exists for both training and prediction

When to use FALSE:
- LRMCB data not available for current season (e.g., 2026)
- Want consistent features across all years
- LRMCB source may have shut down

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
INCLUDE_LRMCB = False       # No 2026 LRMCB data available
INCLUDE_VEGAS_ODDS = False  # Optional - your choice
```

================================================================================
DATA SOURCES
================================================================================

**Required (2008-2025):**
- bartTorvik: Team efficiency ratings, tempo, luck
- kenPom: Adjusted efficiency, tempo, strength of schedule
- espnBPI: Basketball Power Index ratings
- masseyComposite: Composite of 50+ ranking systems

**Optional (2016-2025):**
- LRMCB: Luke Rettig's rankings (if available)
- powerRank: Power rankings (working on historical data)

**Optional (Current season):**
- vegasOdds: DraftKings Elite 8/F4/Championship probabilities

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

**3. Team Name Standardization**

All sources use teamsIndex.csv for consistent team naming:
- "Michigan" (not "U of Michigan" or "UM")
- "Ohio St." (not "Ohio State" or "OSU")

Clean team names → successful joins across all sources

================================================================================
DATASETS PRODUCED
================================================================================

**training_set_long.csv**
- Years: 2008-2025 (18 tournaments)
- Sources: 4-5 (bartTorvik, kenPom, espnBPI, masseyComposite, [powerRank])
- Teams: ~1,147 tournament teams
- Features: 49-51 columns
- Use: Train models with maximum historical data

**training_set_rich.csv**
- Years: 2016-2025 (10 tournaments)
- Sources: 5-6 (LONG sources + [LRMCB] + powerRank)
- Teams: ~612 tournament teams
- Features: 51-52 columns
- Use: Train models with maximum features (if sufficient sample size)

**predict_set_2026.csv**
- Year: 2026 (current season)
- Sources: 5-6 (depends on toggles)
- Teams: 365 (all D1)
- Features: 53-56 columns
- Extras: tournamentSeed, tournamentRegion, [vegas odds]
- Use: Apply trained models to generate 2026 predictions

================================================================================
VALIDATION CHECKS
================================================================================

**After running training script:**

✓ Console shows "No tournament teams lost"
✓ Console shows "Rank normalization complete"
✓ Two files created in L3/data/trainingData/
✓ Row counts: LONG ~1,147, RICH ~612
✓ Feature counts: LONG 49-51, RICH 51-52

**After running predict script:**

✓ Console shows "No teams lost"
✓ Console shows "Rank normalization complete"
✓ One file created in L3/data/predictionData/
✓ Row count: 365 teams
✓ 64 teams with tournamentSeed

**Scale alignment check:**

```bash
# Both should show ~364
python3 << 'EOF'
import pandas as pd
train = pd.read_csv('../L3/data/trainingData/training_set_long.csv')
pred = pd.read_csv('../L3/data/predictionData/predict_set_2026.csv')
print(f"Training max rank: {train['bartTorvik_Rk'].max():.1f}")
print(f"Prediction max rank: {pred['bartTorvik_Rk'].max():.1f}")
EOF
```

Expected: Both show 364.0

================================================================================
TROUBLESHOOTING
================================================================================

**Issue: "Required input file not found"**
Solution: Run L1 transform scripts first to generate L2 inputs

**Issue: "Teams lost after [source] join"**
Solution: Check teamsIndex.csv for team name mismatches
         Run L1 transforms with updated teamsIndex

**Issue: "Corrupted team names with 'vs.'"**
Solution: Update bartTorvik L1 transform to clean team names
         Re-run bartTorvik transform, then L2 scripts

**Issue: "Training and prediction max ranks don't match"**
Solution: Normalization didn't run - check script has normalization code
         Should normalize after inversion: (val/max) × 364

**Issue: "LRMCB file not found"**
If intended: Check file exists, verify path
If expected: Set INCLUDE_LRMCB = False in both scripts

**Issue: "Different feature counts in training vs prediction"**
Solution: Ensure INCLUDE_LRMCB matches in both scripts
         Both True or both False

**Issue: "Duplicate team-seasons found"**
Solution: This is normal - script handles it automatically
         Keeps first occurrence, drops duplicates

================================================================================
OUTPUTS FOR L3
================================================================================

L3 modeling layer expects:

**For training:**
- training_set_long.csv OR training_set_rich.csv
- Join to game outcomes for labels
- Features only (no tournament metadata)
- Ranks normalized to 0-364

**For prediction:**
- predict_set_2026.csv
- All 365 D1 teams
- Same features as training (or subset)
- Ranks normalized to 0-364
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
- Toggle settings changed (LRMCB, Vegas)
- Team name fixes applied
- Bug fixes in transformation logic

**Full regeneration:**

```bash
# 1. Regenerate training sets
cd L2
python create_training_sets_L2.py

# 2. Regenerate prediction set
python create_predict_set_L2.py

# 3. Verify scale alignment
grep "^2026,Michigan," ../L3/data/predictionData/predict_set_2026.csv | cut -d',' -f5
grep "^2008,Kansas," ../L3/data/trainingData/training_set_long.csv | cut -d',' -f4

# 4. Retrain L3 models (critical!)
cd ../L3
# [Run L3 training scripts]
```

**Important:** After regenerating L2 data, ALWAYS retrain L3 models.
Models trained on old data won't work with new data structure.

================================================================================
QUICK START
================================================================================

```bash
# Configure toggles (edit scripts)
# INCLUDE_LRMCB = False (no 2026 LRMCB data)
# INCLUDE_VEGAS_ODDS = False (or True if have odds)

cd L2

# Generate training datasets
python create_training_sets_L2.py

# Generate prediction dataset
python create_predict_set_L2.py

# Verify outputs
ls -lh ../L3/data/trainingData/
ls -lh ../L3/data/predictionData/

# Check scale alignment
grep "^2026,Michigan," ../L3/data/predictionData/predict_set_2026.csv | cut -d',' -f5
grep "^2008,Kansas," ../L3/data/trainingData/training_set_long.csv | cut -d',' -f4

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

================================================================================
AUTHOR
================================================================================

Ryan Browder
March Madness Computron

For questions or issues, check:
- L2 script comments
- L1 README for input requirements
- L3 README for output requirements
- Skill documentation for detailed methodology

================================================================================
