================================================================================
                    L0 - DATA COLLECTION LAYER
                              README
================================================================================

PURPOSE:
  Scrape raw data from all basketball analytics sources and deposit raw CSVs
  into L1/data/ for downstream cleaning. L0 scripts do not clean or transform
  data — they capture source output exactly as scraped.

  Scrapers are designed to be run at any time throughout the season.
  Each run pulls the most current available data from the source. Re-running
  a scraper updates the raw file in place — simply re-run the full pipeline
  (L1 → L2 → L3) afterward to propagate fresh data through to predictions.

OUTPUT LOCATION:
  All L0 scripts write to: L1/data/{source}/{source}_raw_L1.csv

================================================================================
DATA SOURCES
================================================================================

REQUIRED (historical 2008–2025 + current 2026 season):

  bartTorvik          — Team efficiency, tempo, luck
  kenPom              — Adjusted efficiency, strength of schedule
  espnBPI             — Basketball Power Index
  masseyComposite     — Composite of 50+ ranking systems
  powerRank           — Power rankings (2016+)

TOURNAMENT GROUND TRUTH:

  srcbb               — Historical tournament game results (2008–2025)
                        Labels for L3 model training. Re-run post-tournament
                        each year to add the new season's results.

================================================================================
DIRECTORY STRUCTURE
================================================================================

L0/
├── bartTorvik/
│   └── bartTorvik_scraper_L0.py
├── espnBPI/
│   └── espnBPI_scraper_L0.py
├── kenPom/
│   └── kenPom_scraper_L0.py
├── masseyComposite/
│   └── masseyComposite_scraper_L0.py
├── powerRank/
│   └── powerRank_scraper_L0.py
└── srcbb/
    └── srcbb_scraper_L0.py

================================================================================
TECHNICAL NOTES BY SOURCE
================================================================================

BART TORVIK
  Scraper:    Selenium (JavaScript-rendered table)
  Auth:       None required
  Cadence:    Run anytime — pulls most current season data on each run
  Output:     L1/data/bartTorvik/bartTorvik_raw_L1.csv

  Run:
    cd L0/bartTorvik
    python bartTorvik_scraper_L0.py

  Expected output:
    ~7,000+ rows (all D1 teams × all years 2008–2026)
    Raw column names, no cleaning applied

KEN POM
  Scraper:    Selenium with undetected-chromedriver
  Auth:       Subscription required — must be logged in before running
  Cadence:    Run anytime for most current ratings
  Bot detection: Cloudflare challenge on page transitions
  Output:     L1/data/kenPom/kenPom_raw_L1.csv

  Recommended approach:
    Run with HEADLESS=False and manually solve the Cloudflare checkbox
    on first load. Session persists across year requests.

  Alternative:
    undetected-chromedriver bypasses Cloudflare automatically:
      pip install undetected-chromedriver --break-system-packages

  Run:
    cd L0/kenPom
    python kenPom_scraper_L0.py

  Expected output:
    ~6,700+ rows (all years 2008–2026)

ESPN BPI
  Scraper:    Selenium or requests (see script for active method)
  Auth:       None required
  Cadence:    Run anytime
  Output:     L1/data/espnBPI/espnBPI_raw_L1.csv

  Run:
    cd L0/espnBPI
    python espnBPI_scraper_L0.py

MASSEY COMPOSITE
  Scraper:    undetected-chromedriver
  Auth:       None required
  Cadence:    Run anytime
  Bot detection: Cloudflare — undetected-chromedriver handles automatically
  Output:     L1/data/masseyComposite/masseyComposite_raw_L1.csv

  Install dependency if not already installed:
    pip install undetected-chromedriver --break-system-packages

  Run:
    cd L0/masseyComposite
    python masseyComposite_scraper_L0.py

  Notes:
    Script waits 15 seconds for Cloudflare challenge, then 5 more for
    data render. If output is empty, check massey_debug.html for
    diagnostic output. May need to increase wait times.

POWER RANK
  Scraper:    requests + BeautifulSoup
  Auth:       None required
  Cadence:    Run anytime for updated rankings
  Output:     L1/data/powerRank/powerRank_raw_L1.csv

  Run:
    cd L0/powerRank
    python powerRank_scraper_L0.py

  Notes:
    Coverage begins 2016. If the page becomes JavaScript-heavy and
    requests fails, the script may need to be upgraded to Selenium.

SRCBB (TOURNAMENT GAME RESULTS)
  Scraper:    requests or Selenium (historical game data)
  Auth:       None required
  Cadence:    Run once per year after tournament completes to add new results.
              During the season, this dataset does not change — no need to
              re-run until the following year's tournament is finished.
  Output:     L1/data/srcbb/srcbb_raw_L1.csv

  Run:
    cd L0/srcbb
    python srcbb_scraper_L0.py

  Expected output:
    ~1,100 rows (one row per tournament game, 2008–2025)
    This is the ground truth labels file for L3 model training

================================================================================
EXECUTION — RUNNING THE SCRAPERS
================================================================================

Scrapers are independent and can be run in any order or individually.
Run whichever sources have updated data you want to pull in.

To refresh all sources:

  cd L0/bartTorvik && python bartTorvik_scraper_L0.py
  cd ../kenPom && python kenPom_scraper_L0.py
  cd ../espnBPI && python espnBPI_scraper_L0.py
  cd ../masseyComposite && python masseyComposite_scraper_L0.py
  cd ../powerRank && python powerRank_scraper_L0.py

After any scraper run, propagate changes through the pipeline:

  1. L1 — re-run the corresponding source transform
  2. L2 — re-run create_training_sets_L2.py and create_predict_set_L2.py
  3. L3 — retrain models (if training data changed)
         or re-run predictions only (if only predict_set changed)

If only predict-side data changed (current season ratings updated, no new
historical tournament data), you can skip L3 retraining and re-run
predictions directly using the existing trained models.

================================================================================
VALIDATION — WHAT GOOD OUTPUT LOOKS LIKE
================================================================================

After running each scraper, spot-check the output CSV:

  ✓ Row count consistent with prior runs
  ✓ Year column present and populated
  ✓ Current season (2026) rows present
  ✓ Team names look clean — no null rows, no truncated strings
  ✓ No duplicate rows for the same team/year
  ✓ No stray header rows mixed into data

If team names look malformed or the page didn't load correctly,
run with HEADLESS=False to inspect the browser window directly.

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: Cloudflare "Verify you are human" blocking scraper
FIX (manual):   Run with HEADLESS=False, solve checkbox on first load,
                session persists across remaining year requests
FIX (auto):     pip install undetected-chromedriver --break-system-packages
                Update driver setup in script to use uc.Chrome()

ISSUE: Table not found / empty output
FIX: Site may have changed HTML structure. Run with HEADLESS=False
     and inspect what's actually loading. Update CSS selectors if needed.

ISSUE: KenPom login wall
FIX: Must be logged in to KenPom before running. Log in manually
     in the browser, then run with HEADLESS=False to use active session.

ISSUE: Duplicate header rows in raw CSV
FIX: Expected artifact from some JS-rendered tables.
     Do not fix in L0 — L1 transform handles removal.

ISSUE: powerRank scraper returns empty or partial data
FIX: Page may have gone JavaScript-heavy. Upgrade script to Selenium.

================================================================================
DEPENDENCIES
================================================================================

  pip install selenium pandas requests beautifulsoup4
  pip install undetected-chromedriver --break-system-packages   (for Cloudflare sites)

  ChromeDriver must match installed Chrome version.
  undetected-chromedriver handles ChromeDriver version matching automatically.

================================================================================
