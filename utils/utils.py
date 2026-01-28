# utils.py
"""
Tournament configuration and utilities
"""

# Current season year
CURRENT_YEAR = 2026

# Pre-tournament cutoff dates (day before Selection Sunday)
# Format: MMDD (month day)
TOURNEY_CUTOFF_DATES = {
    2008: "0315",  # Selection Sunday: March 16, 2008
    2009: "0314",  # Selection Sunday: March 15, 2009
    2010: "0313",  # Selection Sunday: March 14, 2010
    2011: "0312",  # Selection Sunday: March 13, 2011
    2012: "0310",  # Selection Sunday: March 11, 2012
    2013: "0316",  # Selection Sunday: March 17, 2013
    2014: "0315",  # Selection Sunday: March 16, 2014
    2015: "0314",  # Selection Sunday: March 15, 2015
    2016: "0312",  # Selection Sunday: March 13, 2016
    2017: "0311",  # Selection Sunday: March 12, 2017
    2018: "0310",  # Selection Sunday: March 11, 2018
    2019: "0316",  # Selection Sunday: March 17, 2019
    2020: "0314",  # Selection Sunday: March 15, 2020 (cancelled)
    2021: "0313",  # Selection Sunday: March 14, 2021
    2022: "0312",  # Selection Sunday: March 13, 2022
    2023: "0311",  # Selection Sunday: March 12, 2023
    2024: "0316",  # Selection Sunday: March 17, 2024
    2025: "0315",  # Selection Sunday: March 16, 2025
    2026: "0314",  # Selection Sunday: March 15, 2026
}

def get_cutoff_date(year: int) -> str:
    """
    Get the pre-tournament cutoff date for a given year.
    
    Args:
        year: Season year
        
    Returns:
        Date string in MMDD format (day before Selection Sunday)
    """
    return TOURNEY_CUTOFF_DATES.get(year, "0307")  # Default to March 7 if not found