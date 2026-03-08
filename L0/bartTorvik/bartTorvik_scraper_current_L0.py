"""
Bart Torvik Current Year Scraper
Scrapes only CURRENT_YEAR data for quick updates
Run frequently during the season
Output: bartTorvik_current_raw.csv
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from typing import Optional
import sys
import os

# Add the marchMadness directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.utils import CURRENT_YEAR, get_cutoff_date


def create_driver():
    """Create a Chrome WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def scrape_torvik_year(driver, year: int) -> Optional[pd.DataFrame]:
    """
    Scrape Bart Torvik T-Rank data for a single season (pre-tournament).
    
    Args:
        driver: Selenium WebDriver instance
        year: Season ending year (2008 = 2007-08 season)
    
    Returns:
        DataFrame with team statistics, or None if scrape fails
    """
    end_date = get_cutoff_date(year)
    
    begin_date = f"{year-1}1101"
    end_date_full = f"{year}{end_date}"
    
    url = (
        f"https://barttorvik.com/trank.php?"
        f"year={year}&"
        f"sort=&hteam=&t2value=&"
        f"conlimit=All&state=All&"
        f"begin={begin_date}&end={end_date_full}&"
        f"top=0&revquad=0&quad=5&venue=All&type=All&mingames=0#"
    )
    
    print(f"Scraping {year} (cutoff: {end_date[:2]}/{end_date[2:]})...", end=" ")
    
    try:
        driver.get(url)
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        
        time.sleep(1)
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        table = soup.find('table')
        
        if not table:
            print("✗ No table found")
            return None
        
        rows = table.find_all('tr')
        if len(rows) < 3:
            print("✗ Insufficient data")
            return None
        
        header_row = None
        data_start_idx = 2
        
        for i, row in enumerate(rows):
            cols = [col.get_text(strip=True) for col in row.find_all(['td', 'th'])]
            if 'Team' in cols or 'Rk' in cols:
                header_row = row
                data_start_idx = i + 1
                break
        
        if not header_row:
            print("✗ Could not find header row")
            return None
        
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        data = []
        for row in rows[data_start_idx:]:
            cols = row.find_all(['td', 'th'])
            if cols:
                row_data = [col.get_text(strip=True) for col in cols]
                if len(row_data) == len(headers):
                    data.append(row_data)
        
        if not data:
            print("✗ No data rows")
            return None
        
        df = pd.DataFrame(data, columns=headers)
        df['Year'] = year
        
        print(f"✓ {len(df)} teams")
        return df
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def scrape_torvik_current() -> pd.DataFrame:
    """
    Scrape Torvik data for current year only.
    
    Returns:
        DataFrame with current year team statistics
    """
    print("="*70)
    print(f"Torvik CURRENT YEAR Data Scraper")
    print(f"Year: {CURRENT_YEAR}")
    print("="*70)
    print()
    
    print("Initializing browser...")
    driver = create_driver()
    
    df = None
    
    try:
        df = scrape_torvik_year(driver, CURRENT_YEAR)
    
    finally:
        driver.quit()
        print("\nBrowser closed")
    
    print()
    print("="*70)
    
    if df is None or df.empty:
        print("✗ No data scraped")
        return None
    
    # Save current year raw data
    output_path = os.path.join(os.path.dirname(__file__), '../../L1/data/bartTorvik/bartTorvik_current_raw.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Current year data saved: bartTorvik_current_raw.csv")
    print(f"  • Total teams: {len(df):,}")
    print(f"  • Year: {CURRENT_YEAR}")
    
    print("="*70)
    
    return df


if __name__ == "__main__":
    df = scrape_torvik_current()
    
    if df is not None:
        print("\nCurrent year dataset sample:")
        print(df.head())