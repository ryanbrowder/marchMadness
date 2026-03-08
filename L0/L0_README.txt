================================================================================
                    L0 - DATA COLLECTION LAYER
                              README
================================================================================

PURPOSE:
  Scrape raw data from all basketball analytics sources and deposit raw CSVs
  into L1/data/ for downstream cleaning. L0 scripts do not clean or transform
  data — they capture source output exactly as scraped.

OUTPUT LOCATION:
  All L0 scripts write to: L1/data/{source}/

================================================================================
SCRAPER ARCHITECTURE
================================================================================

Each primary source uses a split scraper pattern:

  {source}_scraper_historical_L0.py — Scrapes all prior seasons (2008 through
                                       CURRENT_YEAR - 1). Run once per season
                                       or when historical data needs correcting.
                                       Execution time: several minutes.

  {source}_scraper_current_L0.py    — Scrapes current season only (CURRENT_YEAR).
                                       Run anytime for updated ratings.
                                       Execution time: ~10 seconds.

Both files are read and concatenated by the L1 transform before processing.
L2 and downstream layers see no difference — they consume L1 output as before.

CURRENT_YEAR is defined in utils/utils.py and controls both scrapers
automatically. At season rollover, update that single variable and the
entire pipeline adjusts.

srcbb is a single scraper — tournament game results are static during the
season and only need to be updated once after each tournament concludes.

================================================================================
DIRECTORY STRUCTURE
================================================================================

L0/
├── L0_PRIME.py                               ← runs all 5 primary scrapers
├── bartTorvik/
│   ├── bartTorvik_scraper_historical_L0.py
│   └── bartTorvik_scraper_current_L0.py
├── espnBPI/
│   ├── espnBPI_scraper_historical_L0.py
│   └── espnBPI_scraper_current_L0.py
├── kenPom/
│   ├── kenPom_scraper_historical_L0.py
│   └── kenPom_scraper_current_L0.py
├── masseyComposite/
│   ├── masseyComposite_scraper_historical_L0.py
│   └── masseyComposite_scraper_current_L0.py
├── powerRank/
│   ├── powerRank_scraper_historical_L0.py
│   └── powerRank_scraper_current_L0.py
└── srcbb/
    └── srcbb_scraper_L0.py

================================================================================
DATA SOURCES
================================================================================

PRIMARY (historical 2008–CURRENT_YEAR, split scraper pattern):

  bartTorvik       — Team efficiency, tempo, luck
  kenPom           — Adjusted efficiency, strength of schedule (subscription)
  espnBPI          — Basketball Power Index
  masseyComposite  — Composite of 50+ ranking systems
  powerRank        — Power rankings (2016+)

TOURNAMENT GROUND TRUTH:

  srcbb            — Historical tournament game results (2008–present)
                     Ground truth labels for L3 model training.
                     Re-run once after each tournament completes.

================================================================================
L0_PRIME.py — FULL REFRESH ORCHESTRATOR
================================================================================

Runs all five primary scrapers in sequence (current season only by default).

  cd L0
  python L0_PRIME.py

Use this for routine in-season data refreshes. After completion, run the
full L1 → L2 → L3 pipeline to propagate updated data through to predictions.

To refresh historical data across all sources, run the individual
historical scrapers (see USAGE WORKFLOWS below).

================================================================================
TECHNICAL NOTES BY SOURCE
================================================================================

BART TORVIK
  Scraper:    Selenium (JavaScript-rendered table)
  Auth:       None required
  Output:     L1/data/bartTorvik/bartTorvik_historical_raw.csv
              L1/data/bartTorvik/bartTorvik_current_raw.csv

  Run historical (once per season):
    cd L0/bartTorvik
    python bartTorvik_scraper_historical_L0.py

  Run current (anytime):
    python bartTorvik_scraper_current_L0.py

  Expected output:
    Historical: ~6,300+ rows (all D1 teams × 2008–2025)
    Current:    ~365 rows (all D1 teams, current season)

KEN POM
  Scraper:    Selenium with undetected-chromedriver
  Auth:       Subscription required — must be logged in before running
  Bot detection: Cloudflare on page transitions

  Recommended: Run with HEADLESS=False and manually solve the Cloudflare
  checkbox on first load. Session persists across all year requests.

  Alternative: undetected-chromedriver bypasses automatically:
    pip install undetected-chromedriver --break-system-packages

  Output:     L1/data/kenPom/kenPom_historical_raw.csv
              L1/data/kenPom/kenPom_current_raw.csv

  Run historical (once per season):
    cd L0/kenPom
    python kenPom_scraper_historical_L0.py

  Run current (anytime):
    python kenPom_scraper_current_L0.py

ESPN BPI
  Scraper:    Selenium or requests (see script for active method)
  Auth:       None required
  Output:     L1/data/espnBPI/espnBPI_historical_raw.csv
              L1/data/espnBPI/espnBPI_current_raw.csv

  Run historical:
    cd L0/espnBPI
    python espnBPI_scraper_historical_L0.py

  Run current:
    python espnBPI_scraper_current_L0.py

MASSEY COMPOSITE
  Scraper:    undetected-chromedriver
  Auth:       None required
  Bot detection: Cloudflare — undetected-chromedriver handles automatically

  Install if not present:
    pip install undetected-chromedriver --break-system-packages

  Output:     L1/data/masseyComposite/masseyComposite_historical_raw.csv
              L1/data/masseyComposite/masseyComposite_current_raw.csv

  Notes: Script waits 15 seconds for Cloudflare, then 5 more for data render.
         If output is empty, check massey_debug.html for diagnostic output.

  Run historical:
    cd L0/masseyComposite
    python masseyComposite_scraper_historical_L0.py

  Run current:
    python masseyComposite_scraper_current_L0.py

POWER RANK
  Scraper:    requests + BeautifulSoup
  Auth:       None required
  Coverage:   2016+
  Output:     L1/data/powerRank/powerRank_historical_raw.csv
              L1/data/powerRank/powerRank_current_raw.csv

  Notes: If page becomes JavaScript-heavy and requests fails,
         upgrade script to Selenium.

  Run historical:
    cd L0/powerRank
    python powerRank_scraper_historical_L0.py

  Run current:
    python powerRank_scraper_current_L0.py

SRCBB (TOURNAMENT GAME RESULTS)
  Scraper:    requests or Selenium
  Auth:       None required
  Cadence:    Once per year, after tournament concludes.
              No need to re-run during the season — data is static.
  Output:     L1/data/srcbb/srcbb_raw_L1.csv

  Run:
    cd L0/srcbb
    python srcbb_scraper_L0.py

  Expected output:
    ~1,100 rows (one per tournament game, 2008–present)

================================================================================
USAGE WORKFLOWS
================================================================================

IN-SEASON REFRESH (routine — run frequently):

  cd L0
  python L0_PRIME.py

  Then propagate:
    L1 — re-run all source transforms
    L2 — re-run create_training_sets_L2.py and create_predict_set_L2.py
    L3 — re-run predictions using existing trained models
         (no retraining needed unless training data changed)

START OF SEASON (once — when CURRENT_YEAR rolls over):

  1. Update utils/utils.py: CURRENT_YEAR = YYYY

  2. Run all historical scrapers to include prior season:
       cd L0/bartTorvik  && python bartTorvik_scraper_historical_L0.py
       cd ../kenPom      && python kenPom_scraper_historical_L0.py
       cd ../espnBPI     && python espnBPI_scraper_historical_L0.py
       cd ../masseyComposite && python masseyComposite_scraper_historical_L0.py
       cd ../powerRank   && python powerRank_scraper_historical_L0.py

  3. Run current scrapers for new season:
       cd L0 && python L0_PRIME.py

  4. Update srcbb with prior tournament results:
       cd L0/srcbb && python srcbb_scraper_L0.py

  5. Propagate full pipeline:
       L1 → L2 → L3 (full retrain — new tournament year added to training data)

HISTORICAL DATA CORRECTION (rare):

  Re-run the specific source's historical scraper, then re-run L1 → L2 → L3.

================================================================================
VALIDATION — WHAT GOOD OUTPUT LOOKS LIKE
================================================================================

After running scrapers, spot-check output CSVs:

  ✓ Historical row count consistent with prior runs (~6,300+ for most sources)
  ✓ Current row count ~365 (all D1 teams)
  ✓ Year column present and correct
  ✓ Team names clean — no nulls, no truncated strings
  ✓ No duplicate rows for the same team/year
  ✓ No stray header rows mixed into data

L1 transform console confirms concatenation:
  ✓ Historical: 6,324 rows (2008–2025)
  ✓ Current: 365 rows (year 2026)
  ✓ Combined: 6,689 total rows

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: Cloudflare "Verify you are human" blocking scraper
FIX (manual):   HEADLESS=False, solve checkbox on first load,
                session persists across remaining requests
FIX (auto):     pip install undetected-chromedriver --break-system-packages
                Update driver setup to use uc.Chrome()

ISSUE: Table not found / empty output
FIX: Site may have changed HTML structure. Run with HEADLESS=False
     and inspect what's loading. Update CSS selectors if needed.

ISSUE: KenPom login wall
FIX: Log in manually in the browser, then run with HEADLESS=False.

ISSUE: Duplicate header rows in raw CSV
FIX: Expected artifact from JS-rendered tables.
     Do not fix in L0 — L1 transform handles removal.

ISSUE: powerRank returns empty or partial data
FIX: Page may have gone JavaScript-heavy. Upgrade to Selenium.

ISSUE: Historical and current row counts don't add up after L1 concatenation
FIX: Check that both files exist in L1/data/{source}/ before running transform.
     Re-run the missing scraper if either file is absent or stale.

================================================================================
DEPENDENCIES
================================================================================

  pip install selenium pandas requests beautifulsoup4
  pip install undetected-chromedriver --break-system-packages   (Cloudflare sites)

  ChromeDriver must match installed Chrome version.
  undetected-chromedriver handles version matching automatically.

================================================================================
AUTHOR
================================================================================

Ryan Browder
March Madness Computron

================================================================================
