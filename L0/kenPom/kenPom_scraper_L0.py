"""
KenPom Data Scraper - L0 (Ingestion Layer)
Scrapes raw data from kenpom.com for years 2008-2026
Outputs: ../../L1/data/kenPom/kenPom_raw_L1.csv with data exactly as scraped
Note: Run from L0/kenPom/ directory
"""

import time
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configuration
START_YEAR = 2008
END_YEAR = 2026
BASE_URL = "https://kenpom.com/index.php?y={year}"
OUTPUT_DIR = "../../L1/data/kenPom"
OUTPUT_FILE = f"{OUTPUT_DIR}/kenPom_raw_L1.csv"

# Set to False to see browser for debugging timeout issues
HEADLESS = False

# Column names based on table structure
COLUMNS = [
    'Rk',
    'Team',
    'Conf',
    'W-L',
    'NetRtg',
    'ORtg',
    'ORtg_Rank',
    'DRtg',
    'DRtg_Rank',
    'AdjT',
    'AdjT_Rank',
    'Luck',
    'Luck_Rank',
    'SOS_NetRtg',
    'SOS_NetRtg_Rank',
    'SOS_ORtg',
    'SOS_ORtg_Rank',
    'SOS_DRtg',
    'SOS_DRtg_Rank',
    'NCSOS_NetRtg',
    'NCSOS_NetRtg_Rank'
]


def setup_driver(headless=True):
    """Initialize Chrome driver with options"""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')  # Run in background
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


def scrape_year(driver, year, retry=0, max_retries=2):
    """
    Scrape KenPom data for a specific year
    
    Args:
        driver: Selenium WebDriver instance
        year: Year to scrape (2008-2026)
        retry: Current retry attempt
        max_retries: Maximum number of retries
    
    Returns:
        List of dictionaries containing team data
    """
    url = BASE_URL.format(year=year)
    print(f"Scraping {year}... ", end='', flush=True)
    
    try:
        driver.get(url)
        
        # Wait for table to load with increased timeout
        wait = WebDriverWait(driver, 30)  # Increased from 10 to 30 seconds
        table = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.ratings-table, table"))
        )
        
        # Additional wait for table to fully populate
        time.sleep(2)
        
        # Find all rows in the table (handles multiple tbody elements or mixed headers)
        rows = table.find_elements(By.TAG_NAME, 'tr')
        
        year_data = []
        
        for row in rows:
            # Only get td elements (data cells, not th header cells)
            cells = row.find_elements(By.TAG_NAME, 'td')
            
            # Skip header rows (they have th elements) or incomplete rows
            if len(cells) < 21:
                continue
            
            # Extract all cell values
            row_data = {
                'Year': year,
                'Rk': cells[0].text.strip(),
                'Team': cells[1].text.strip(),
                'Conf': cells[2].text.strip(),
                'W-L': cells[3].text.strip(),
                'NetRtg': cells[4].text.strip(),
                'ORtg': cells[5].text.strip(),
                'ORtg_Rank': cells[6].text.strip(),
                'DRtg': cells[7].text.strip(),
                'DRtg_Rank': cells[8].text.strip(),
                'AdjT': cells[9].text.strip(),
                'AdjT_Rank': cells[10].text.strip(),
                'Luck': cells[11].text.strip(),
                'Luck_Rank': cells[12].text.strip(),
                'SOS_NetRtg': cells[13].text.strip(),
                'SOS_NetRtg_Rank': cells[14].text.strip(),
                'SOS_ORtg': cells[15].text.strip(),
                'SOS_ORtg_Rank': cells[16].text.strip(),
                'SOS_DRtg': cells[17].text.strip(),
                'SOS_DRtg_Rank': cells[18].text.strip(),
                'NCSOS_NetRtg': cells[19].text.strip(),
                'NCSOS_NetRtg_Rank': cells[20].text.strip()
            }
            
            year_data.append(row_data)
        
        if len(year_data) > 0:
            print(f"✓ ({len(year_data)} teams)")
            return year_data
        else:
            # No data found, retry
            if retry < max_retries:
                print(f"⟳ (no data, retry {retry + 1}/{max_retries})")
                time.sleep(3)
                return scrape_year(driver, year, retry + 1, max_retries)
            else:
                print(f"✗ (no data after {max_retries} retries)")
                return []
        
    except TimeoutException:
        if retry < max_retries:
            print(f"⟳ (timeout, retry {retry + 1}/{max_retries})")
            time.sleep(3)
            return scrape_year(driver, year, retry + 1, max_retries)
        else:
            print(f"✗ (timeout after {max_retries} retries)")
            return []
    except Exception as e:
        print(f"✗ (error: {str(e)})")
        return []


def main():
    """Main scraping function"""
    print("="*60)
    print("KenPom Data Scraper - L0 Ingestion")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print("="*60)
    
    # Initialize driver
    driver = setup_driver(headless=HEADLESS)
    
    all_data = []
    
    try:
        # Scrape each year
        for year in range(START_YEAR, END_YEAR + 1):
            year_data = scrape_year(driver, year)
            all_data.extend(year_data)
            
            # Be respectful with rate limiting
            time.sleep(2)  # Increased from 1 to 2 seconds
        
        # Convert to DataFrame and save
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Reorder columns: Year first, then alphabetically by column name
            cols = ['Year'] + [col for col in COLUMNS]
            df = df[cols]
            
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            df.to_csv(OUTPUT_FILE, index=False)
            print("="*60)
            print(f"✓ Saved {len(df)} total rows to {OUTPUT_FILE}")
            print(f"✓ Years covered: {df['Year'].min()}-{df['Year'].max()}")
            print(f"✓ Teams per year (avg): {len(df) / df['Year'].nunique():.1f}")
            print("="*60)
        else:
            print("✗ No data scraped")
    
    finally:
        driver.quit()


if __name__ == "__main__":
    main()