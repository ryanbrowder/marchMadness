================================================================================
                    L1 - DATA CLEANING AND TRANSFORM LAYER
                                  README
================================================================================

PURPOSE:
  Apply source-specific cleaning and standardization to raw data files.
  Each source has its own transform script. Output is structured, prefixed,
  and team-indexed CSVs ready for L2 joining and modeling.

  L1 also handles data that has no L0 scraper — LRMCB, vegasOdds, and
  bracketology are manually maintained in L1/data/ and transformed here.

INPUT:  L1/data/{source}/   (raw files from L0 scrapers or manual intake)
OUTPUT: L2/data/{source}/   (analysis-ready CSVs, split by year)

================================================================================
DIRECTORY STRUCTURE
================================================================================

L1/
├── bartTorvik/
│   └── bartTorvik_transform_L1.py
├── bracketology/
│   └── clean_espn_public_picks.py
├── espnBPI/
│   └── espnBPI_transform_L1.py
├── kenPom/
│   └── kenPom_transform_L1.py
├── LRMCB/
│   └── LRMCB_transform_L1.py
├── masseyComposite/
│   └── masseyComposite_transform_L1.py
├── powerRank/
│   └── powerRank_transform_L1.py
├── srcbb/
│   └── srcbb_transform_L1.py
├── vegasOdds/
│   └── vegasOdds_transform_L1.py
│
└── data/                          ← all raw inputs live here
    ├── bartTorvik/                  (written by L0 scraper)
    ├── bracketology/
    │   ├── espn_bracketology_2026.xlsx       (manually updated post-Selection Sunday)
    │   ├── espn_public_picks_2026.csv        (manually updated post-Selection Sunday)
    │   └── espn_public_picks_historical.xlsx (manually maintained archive)
    ├── espnBPI/                     (written by L0 scraper)
    ├── kenPom/                      (written by L0 scraper)
    ├── LRMCB/
    │   └── LRMCB_raw_L1.csv         (manually maintained — no L0 scraper)
    ├── masseyComposite/             (written by L0 scraper)
    ├── powerRank/                   (written by L0 scraper)
    ├── srcbb/                       (written by L0 scraper)
    └── vegasOdds/
        ├── vegasOdds_Champion.csv   (manually maintained — no L0 scraper)
        └── vegasOdds_Final4.csv     (manually maintained — no L0 scraper)

================================================================================
SHARED TRANSFORM LOGIC (ALL SOURCES)
================================================================================

Every transform script applies the same core pattern:

  1. Load raw file(s) from L1/data/{source}/
  2. Remove duplicate header rows (common artifact from JS-rendered tables)
  3. Apply source-specific cleaning (see per-source notes below)
  4. Add team Index via teamsIndex.csv lookup
  5. Prefix all feature columns with {source}_ (see naming rules below)
  6. Convert feature columns to numeric types
  7. Split by year: current season → predict CSV, historical → analyze CSV
  8. Output to L2/data/{source}/

COLUMN NAMING RULES:
  Prefixed:      All feature/metric columns → {source}_ColumnName
  Not prefixed:  Team, Year, Index, tournamentSeed, tournamentOutcome

COLUMN ORDER (standard across all sources):
  Team → Index → tournamentSeed → tournamentOutcome → Year → [prefixed features]

YEAR SPLIT:
  Current year (2026) → {source}_predict_L2.csv
  All prior years     → {source}_analyze_L2.csv

VALIDATION GATE:
  100% team match rate required. If any team fails to join against
  teamsIndex.csv, the script will report unmapped teams. Resolve all
  mismatches in teamsIndex.csv before proceeding to L2.

================================================================================
SOURCE-SPECIFIC NOTES
================================================================================

BART TORVIK
  Script: bartTorvik_transform_L1.py
  Input:  L1/data/bartTorvik/  (written by L0 scraper)
  Output: L2/data/bartTorvik/bartTorvik_analyze_L2.csv
          L2/data/bartTorvik/bartTorvik_predict_L2.csv

  Special logic:
    - Parse Team column to extract tournamentSeed and tournamentOutcome
      (Torvik stores these as a combined string field)
    - Parse Rec column to extract Wins_Overall
    - Filter by pre-tournament cutoff date (defined in utils/utils.py)
      to prevent look-ahead bias in training data

  Expected output:
    analyze: ~6,100+ rows (2008–2025, all D1 teams)
    predict: ~365 rows (2026)

BRACKETOLOGY
  Script: clean_espn_public_picks.py
  Input:  L1/data/bracketology/espn_public_picks_2026.csv
          L1/data/bracketology/espn_public_picks_historical.xlsx
  Output: L2/data/bracketology/espn_public_picks_2026_clean.csv

  Notes:
    These files are manually maintained — no L0 scraper.
    Prior to Selection Sunday, espn_bracketology_2026.xlsx and
    espn_public_picks_2026.csv contain synthetic/placeholder data.
    Once the tournament bracket is announced, both files must be manually
    updated with actual bracket and public pick data before running L1
    and proceeding to L4 contrarian analysis.

    espn_bracketology_2026.xlsx — projected seeds and regions, referenced
    directly by L2/create_predict_set_L2.py to add tournamentSeed and
    tournamentRegion columns to the predict dataset.

    espn_public_picks_2026.csv — public bracket pick percentages, cleaned
    by this script and consumed by L4 for contrarian analysis.

    espn_public_picks_historical.xlsx — archive of prior years' public
    pick data. Manually appended each year post-tournament.

ESPN BPI
  Script: espnBPI_transform_L1.py
  Input:  L1/data/espnBPI/  (written by L0 scraper)
  Output: L2/data/espnBPI/espnBPI_analyze_L2.csv
          L2/data/espnBPI/espnBPI_predict_L2.csv

  Special logic: Standard transform, no source-specific edge cases.

KEN POM
  Script: kenPom_transform_L1.py
  Input:  L1/data/kenPom/  (written by L0 scraper)
  Output: L2/data/kenPom/kenPom_analyze_L2.csv
          L2/data/kenPom/kenPom_predict_L2.csv

  Special logic:
    - Remove Conf, W-L columns (not used in modeling)

  Expected output:
    analyze: ~6,334 rows (2008–2025)
    predict: ~365 rows (2026)

  Good run indicator:
    ✓ Loaded 6699 rows
    ✓ Matched 6699 teams (100%)
    ✓ Predict: 365 / Analyze: 6334

LRMCB
  Script: LRMCB_transform_L1.py
  Input:  L1/data/LRMCB/LRMCB_raw_L1.csv  (manually maintained — no L0 scraper)
  Output: L2/data/LRMCB/LRMCB_analyze_L2.csv
          L2/data/LRMCB/LRMCB_predict_L2.csv

  Notes:
    LRMCB_raw_L1.csv is manually downloaded and placed in L1/data/LRMCB/.
    Coverage begins 2016. Used only in training_set_rich (L2) when available.
    If no current season data is available, set INCLUDE_LRMCB = False in
    both L2 scripts (must match in training and predict scripts).

  Expected output:
    analyze: ~612 rows (2016–2025)
    predict: ~365 rows (2026, if available)

MASSEY COMPOSITE
  Script: masseyComposite_transform_L1.py
  Input:  L1/data/masseyComposite/  (written by L0 scraper)
  Output: L2/data/masseyComposite/masseyComposite_analyze_L2.csv
          L2/data/masseyComposite/masseyComposite_predict_L2.csv

  Special logic:
    - Source uses short year-based column headers ('01, '02, etc.)
    - Transform reshapes to long format (one row per team per year)

POWER RANK
  Script: powerRank_transform_L1.py
  Input:  L1/data/powerRank/  (written by L0 scraper)
  Output: L2/data/powerRank/powerRank_analyze_L2.csv
          L2/data/powerRank/powerRank_predict_L2.csv

  Notes:
    Coverage begins 2016. Used in training_set_rich (L2).

SRCBB (TOURNAMENT GAME RESULTS)
  Script: srcbb_transform_L1.py
  Input:  L1/data/srcbb/  (written by L0 scraper)
  Output: L2/data/srcbb/srcbb_analyze_L2.csv

  Notes:
    Game-level data, not team-level metrics. No year split — all historical
    results go to a single analyze file used as L3 training labels.

  Special logic:
    - Maps both TeamA and TeamB to teamsIndex.csv IDs
    - Derives: SeedDiff, TeamA_Won, IsUpset, MarginOfVictory, TotalPoints

  Output columns:
    Year, Region, Round,
    TeamA, TeamA_ID, SeedA, ScoreA,
    TeamB, TeamB_ID, SeedB, ScoreB,
    Winner, TeamA_Won, SeedDiff, IsUpset, MarginOfVictory, TotalPoints, Location

  Expected output:
    srcbb_analyze_L2.csv — ~1,071 games (2008–2025)

VEGAS ODDS
  Script: vegasOdds_transform_L1.py
  Input:  L1/data/vegasOdds/vegasOdds_Champion.csv
          L1/data/vegasOdds/vegasOdds_Final4.csv
          (manually maintained — no L0 scraper)
  Output: L2/data/vegasOdds/vegasOdds_analyze_L2.csv  (or similar)

  Notes:
    DraftKings Championship and Final Four odds. Manually collected and
    placed in L1/data/vegasOdds/. Optional source — controlled by
    INCLUDE_VEGAS_ODDS toggle in L2/create_predict_set_L2.py.

================================================================================
TEAM NAME STANDARDIZATION
================================================================================

All joins rely on:
  utils/teamsIndex.csv

This file maps every known team name variant to a canonical name and integer
Index. Every transform script loads it and performs a lookup to add the Index
column before outputting.

A 100% match rate is required before proceeding to L2.

To resolve unmapped teams:
  1. Run the transform — unmapped names printed to console
  2. Add missing name variants to teamsIndex.csv
  3. Re-run the transform until 100% match is confirmed

Common failure sources:
  - New D1 programs
  - Teams that changed names
  - Source-specific abbreviations
  - "vs." or opponent string corrupting Torvik team name field

================================================================================
EXECUTION SEQUENCE
================================================================================

Transform scripts are independent and can be run in any order.
All must complete successfully before running L2.

  cd L1/bartTorvik    && python bartTorvik_transform_L1.py
  cd ../kenPom        && python kenPom_transform_L1.py
  cd ../espnBPI       && python espnBPI_transform_L1.py
  cd ../masseyComposite && python masseyComposite_transform_L1.py
  cd ../LRMCB         && python LRMCB_transform_L1.py         (if data available)
  cd ../powerRank     && python powerRank_transform_L1.py
  cd ../srcbb         && python srcbb_transform_L1.py
  cd ../vegasOdds     && python vegasOdds_transform_L1.py     (if using Vegas)
  cd ../bracketology  && python clean_espn_public_picks.py

Verify all expected output files exist in L2/data/ before proceeding:

  ls ../../L2/data/bartTorvik/
  ls ../../L2/data/kenPom/
  ls ../../L2/data/espnBPI/
  ls ../../L2/data/masseyComposite/
  ls ../../L2/data/powerRank/
  ls ../../L2/data/srcbb/
  ls ../../L2/data/bracketology/

================================================================================
VALIDATION — WHAT GOOD OUTPUT LOOKS LIKE
================================================================================

For each source, the console should show:

  ✓ Loaded N rows
  ✓ Matched N teams (100%)
  ✓ Predict: ~365 rows (2026)
  ✓ Analyze: N rows (2008–2025)
  ✓ Saved both CSVs to L2/data/{source}/

Any match rate below 100% must be resolved before proceeding to L2.

================================================================================
CONFIGURATION
================================================================================

utils/utils.py
  CURRENT_YEAR              — Current season (e.g., 2026)
  Tournament cutoff dates   — Pre-Selection Sunday snapshot dates by year,
                              used by bartTorvik transform to filter rows

L2 toggle alignment:
  INCLUDE_LRMCB must be set to the same value (True/False) in both
  L2/create_training_sets_L2.py and L2/create_predict_set_L2.py.
  If LRMCB data is unavailable for the current season, set both to False.

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: "Team not found in teamsIndex" warnings
FIX: Add missing variant to utils/teamsIndex.csv and re-run transform

ISSUE: Corrupted team names containing "vs." (bartTorvik)
FIX: Update bartTorvik L0 scraper to clean team names before writing
     Re-run L0 scraper, then re-run L1 transform

ISSUE: Duplicate team-season rows in output
FIX: Expected — transform auto-deduplicates (keeps first occurrence)
     Investigate source CSV if counts seem significantly off

ISSUE: Wrong year in predict vs. analyze split
FIX: Check CURRENT_YEAR in utils/utils.py

ISSUE: LRMCB file not found
FIX: If intentional — set INCLUDE_LRMCB = False in both L2 scripts
     If unintentional — verify file placed at L1/data/LRMCB/LRMCB_raw_L1.csv

================================================================================
