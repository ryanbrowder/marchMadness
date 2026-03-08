import subprocess
import sys
from pathlib import Path

# ── Current-year scrapers only ────────────────────────────────────────────────
# kenPom and masseyComposite require a visible (non-headless) browser to bypass
# Cloudflare. Make sure undetected-chromedriver / Selenium is configured before
# running those two, or they will fail silently / return empty data.
# ─────────────────────────────────────────────────────────────────────────────

SCRIPTS = [
    {
        "path": "bartTorvik/bartTorvik_scraper_current_L0.py",
        "cloudflare": False,
    },
    {
        "path": "espnBPI/espnBPI_scraper_current_L0.py",
        "cloudflare": False,
    },
    {
        "path": "kenPom/kenPom_scraper_current_L0.py",
        "cloudflare": True,
    },
    {
        "path": "masseyComposite/masseyComposite_scraper_L0.py",
        "cloudflare": True,
    },
    {
        "path": "powerRank/powerRank_scraper_L0.py",
        "cloudflare": False,
    },
]


def run_script(entry: dict) -> bool:
    path = entry["path"]
    cloudflare = entry["cloudflare"]

    print(f"\n{'='*55}")
    print(f"  Running: {path}")
    if cloudflare:
        print(f"  ⚠  Cloudflare site — requires non-headless browser")
    print("=" * 55)

    script_path = Path(path).resolve()

    if not script_path.exists():
        print(f"  SKIPPED — file not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, script_path.name],
        cwd=script_path.parent,   # resolve relative paths from script's dir
    )

    if result.returncode != 0:
        print(f"  ERROR: exited with code {result.returncode}")
        return False

    print(f"  ✓ Done: {path}")
    return True


if __name__ == "__main__":
    print("\nMarch Madness Computron — L0 Current-Year Scrape")
    print("=" * 55)

    cloudflare_count = sum(1 for s in SCRIPTS if s["cloudflare"])
    print(f"  {len(SCRIPTS)} scrapers queued  |  {cloudflare_count} require Cloudflare bypass\n")

    failed = []
    for entry in SCRIPTS:
        success = run_script(entry)
        if not success:
            failed.append(entry["path"])

    print(f"\n{'='*55}")
    if failed:
        print(f"COMPLETED WITH ERRORS — {len(failed)} scraper(s) failed:")
        for f in failed:
            print(f"  ✗  {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(SCRIPTS)} SCRAPERS COMPLETED SUCCESSFULLY")
