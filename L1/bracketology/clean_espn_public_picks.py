"""
L1/bracketology/clean_espn_public_picks.py
------------------------------------------
Cleans ESPN public champion pick percentage data and joins teamsIndex.

Inputs:
  - L1/data/bracketology/espn_public_picks_2026.csv      (current season, rich schema)
  - L1/data/bracketology/espn_public_picks_historical.xlsx (2016-2025, stacked year blocks)
  - utils/teamsIndex.csv

Outputs:
  - L2/data/bracketology/espn_public_picks_2026_clean.csv
  - L2/data/bracketology/espn_public_picks_historical_clean.csv

Replace the synthetic 2026 file with the real ESPN file once the tournament is announced.
The historical file only carries champion_pct; the 2026 file carries full round breakdown.
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Path config — adjust ROOT to your project root if running from elsewhere
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))          # L1/bracketology/
L1_DATA   = os.path.join(ROOT, "..", "data", "bracketology")
L2_DATA   = os.path.join(ROOT, "..", "..", "L2", "data", "bracketology")
UTILS_DIR = os.path.join(ROOT, "..", "..", "utils")

INPUT_2026       = os.path.join(L1_DATA, "espn_public_picks_2026.csv")
INPUT_HISTORICAL = os.path.join(L1_DATA, "espn_public_picks_historical.xlsx")
TEAMS_INDEX      = os.path.join(UTILS_DIR, "teamsIndex.csv")

OUTPUT_2026       = os.path.join(L2_DATA, "espn_public_picks_2026_clean.csv")
OUTPUT_HISTORICAL = os.path.join(L2_DATA, "espn_public_picks_historical_clean.csv")

# ---------------------------------------------------------------------------
# Manual name overrides for entries that don't resolve automatically.
# Key = name as it appears in ESPN data, Value = name in teamsIndex.
# Expand as needed when new teams appear.
# ---------------------------------------------------------------------------
ESPN_NAME_OVERRIDES = {
    # Historical variants
    "Michigan St":    "Michigan State",
    "Mich St":        "Michigan State",
    "Michigan State": "Michigan State",
    "Connecticut":    "UConn",
    "St Marys":       "Saint Mary's",
    "St. John's":     "St. John's",
    "St John's":      "St. John's",
    "Ohio St":        "Ohio State",
    "Oklahoma St":    "Oklahoma State",
    "Virginia Tech":  "Virginia Tech",
    "Iowa St":        "Iowa State",
    "West Virginia":  "West Virginia",
    "North Carolina": "North Carolina",
    # 2026 file variants
    "Miami FL":       "Miami (FL)",
    "NC State":       "North Carolina State",
    "Saint Joseph's":  "Saint Joseph's",
    "Stephen F Austin": "Stephen F. Austin",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_teams_index(path: str) -> dict:
    """Return {team_name_lower: index} lookup from teamsIndex.csv."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Build case-insensitive lookup: lowered name → Index
    return {row["Team"].strip().lower(): int(row["Index"]) for _, row in df.iterrows()}


def resolve_index(team_name: str, index_lookup: dict, overrides: dict) -> int | None:
    """
    Resolve a team name to its teamsIndex integer.
    1. Apply manual override if present.
    2. Exact case-insensitive lookup.
    3. Return None (unmatched) if not found.
    """
    # Apply override first
    canonical = overrides.get(team_name, team_name)
    result = index_lookup.get(canonical.strip().lower())
    if result is not None:
        return result
    # Fallback: try original name without override
    return index_lookup.get(team_name.strip().lower())


def parse_historical_xlsx(path: str) -> pd.DataFrame:
    """
    Parse the stacked-year-block format of the historical ESPN file.
    Returns a tidy DataFrame with columns: year, team_name, champion_pct
    """
    raw = pd.read_excel(path, header=None)
    rows = []
    current_year = None

    for _, row in raw.iterrows():
        cell0 = str(row[0]).strip() if pd.notna(row[0]) else ""
        cell1 = str(row[1]).strip() if pd.notna(row[1]) else ""

        # Year header row (numeric value in col 0, NaN in col 1)
        if cell0.isdigit() and len(cell0) == 4 and cell1 == "":
            current_year = int(cell0)
            continue

        # "ESPN Public" label row — skip
        if "espn public" in cell0.lower():
            continue

        # Empty row — skip
        if cell0 == "" and cell1 == "":
            continue

        # Data row: col0 = champion_pct (float), col1 = team_name
        if current_year is not None:
            try:
                pct = float(cell0)
                team = cell1
                if team:
                    rows.append({
                        "year":         current_year,
                        "team_name":    team,
                        "champion_pct": round(pct * 100, 2),  # store as percentage (0-100)
                    })
            except ValueError:
                pass  # non-numeric col0 — skip silently

    return pd.DataFrame(rows)


def clean_pct_column(series: pd.Series) -> pd.Series:
    """Ensure pct columns are float, rounded to 2 decimal places."""
    return pd.to_numeric(series, errors="coerce").round(2)


def add_index(df: pd.DataFrame, index_lookup: dict, team_col: str = "team_name") -> pd.DataFrame:
    """Add teamsIndex column. Warns on any unmatched teams."""
    df = df.copy()
    df["teamsIndex"] = df[team_col].apply(
        lambda t: resolve_index(t, index_lookup, ESPN_NAME_OVERRIDES)
    )
    unmatched = df[df["teamsIndex"].isna()][team_col].unique()
    if len(unmatched):
        print(f"  ⚠️  UNMATCHED teams (no teamsIndex found) — add to ESPN_NAME_OVERRIDES:")
        for t in sorted(unmatched):
            print(f"       '{t}'")
    else:
        print(f"  ✅  All {len(df)} rows matched to teamsIndex.")
    df["teamsIndex"] = df["teamsIndex"].astype("Int64")  # nullable int
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(L2_DATA, exist_ok=True)

    print("Loading teamsIndex...")
    index_lookup = load_teams_index(TEAMS_INDEX)
    print(f"  {len(index_lookup):,} name variants loaded.")

    # ------------------------------------------------------------------
    # 1. Historical (2016–2025)
    # ------------------------------------------------------------------
    print("\n── Historical ESPN picks ──")
    hist = parse_historical_xlsx(INPUT_HISTORICAL)
    print(f"  Parsed {len(hist)} team-year rows across years: {sorted(hist['year'].unique())}")

    hist["champion_pct"] = clean_pct_column(hist["champion_pct"])
    hist = add_index(hist, index_lookup)

    # Sort and finalize columns
    hist = hist.sort_values(["year", "champion_pct"], ascending=[True, False])
    hist = hist[["year", "teamsIndex", "team_name", "champion_pct"]]

    hist.to_csv(OUTPUT_HISTORICAL, index=False)
    print(f"  Saved → {OUTPUT_HISTORICAL}")

    # ------------------------------------------------------------------
    # 2. 2026 current season
    # ------------------------------------------------------------------
    print("\n── 2026 ESPN picks ──")
    curr = pd.read_csv(INPUT_2026)
    curr.columns = curr.columns.str.strip()

    # Standardize pct columns (already in 0–100 scale in this file)
    pct_cols = ["champion_pct", "final4_pct", "elite8_pct", "sweet16_pct", "round32_pct", "round64_pct"]
    for col in pct_cols:
        if col in curr.columns:
            curr[col] = clean_pct_column(curr[col])

    curr["year"] = 2026
    curr = add_index(curr, index_lookup, team_col="team_name")

    # Drop synthetic/placeholder columns not present in real ESPN data
    curr = curr.drop(columns=[c for c in ["seed_estimate", "notes"] if c in curr.columns])

    # Column order: year + identity first, then pcts, then metadata
    col_order = (
        ["year", "teamsIndex", "team_name"]
        + pct_cols
        + [c for c in curr.columns if c not in ["year", "teamsIndex", "team_name"] + pct_cols]
    )
    curr = curr[[c for c in col_order if c in curr.columns]]
    curr = curr.sort_values("champion_pct", ascending=False)

    curr.to_csv(OUTPUT_2026, index=False)
    print(f"  Saved → {OUTPUT_2026}")

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
