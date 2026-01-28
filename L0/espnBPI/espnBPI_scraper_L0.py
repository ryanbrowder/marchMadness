"""
ESPN BPI Scraper (L0)

Scrapes ESPN BPI data for all available years (2008-2026).
- 2008-2025: Historical data via season parameter
- 2026: Current season via base URL

Outputs to: L1/data/espnBPI/espnBPI_raw_L1.csv
Columns: Team, BPI, BPI_Rk, Off, Def, Year
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

def scrape_espn_bpi_year(year):
    """
    Scrape ESPN BPI data for a specific year.
    
    Args:
        year: Season year (e.g., 2024)
        
    Returns:
        DataFrame with Team, BPI, BPI_Rk, Off, Def, Year columns
    """
    # Use base URL for 2026 (current season), season parameter for historical
    if year == 2026:
        url = 'https://www.espn.com/mens-college-basketball/bpi'
    else:
        url = f'https://www.espn.com/mens-college-basketball/bpi/_/season/{year}'
    
    print(f"  Scraping {year}...")
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Initialize driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        
        # Wait for tables to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(3)
        
        # Click "show more" to load all teams
        for _ in range(20):  # Max 20 clicks
            try:
                show_more = driver.find_element(By.XPATH, 
                    "//*[contains(text(), 'Show More') or contains(text(), 'show more') or contains(text(), 'SHOW MORE')]")
                show_more.click()
                time.sleep(2)
            except:
                break
        
        time.sleep(2)
        
        # Find tables
        tables = driver.find_elements(By.TAG_NAME, "table")
        
        if len(tables) < 2:
            print(f"    Error: Expected 2 tables, found {len(tables)}")
            return pd.DataFrame()
        
        # Table 1: Extract team names
        team_table = tables[0]
        team_rows = team_table.find_elements(By.CSS_SELECTOR, "tbody tr")
        
        teams = []
        for row in team_rows:
            try:
                # Each row has 2 links: logo (no text), team name (has text)
                team_links = row.find_elements(By.CSS_SELECTOR, "a[href*='/team/']")
                team_name = None
                for link in team_links:
                    text = link.text.strip()
                    if text:
                        team_name = text
                        break
                if team_name and team_name != 'TEAM':
                    teams.append(team_name)
            except:
                continue
        
        # Table 2: Extract data columns
        # Column order: W-L (0), BPI (1), BPI RK (2), TREND (3), OFF (4), DEF (5), ...
        data_table = tables[1]
        data_rows = data_table.find_elements(By.CSS_SELECTOR, "tbody tr")
        
        bpi_data = []
        for row in data_rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                
                if len(cells) >= 6:
                    # Extract: BPI (score), BPI RK (rank), OFF, DEF
                    bpi_score = cells[1].text.strip()
                    bpi_rank = cells[2].text.strip()
                    off = cells[4].text.strip()
                    def_val = cells[5].text.strip()
                    
                    try:
                        bpi_data.append({
                            'BPI': float(bpi_score),
                            'BPI_Rk': int(bpi_rank),
                            'Off': float(off),
                            'Def': float(def_val)
                        })
                    except ValueError:
                        continue
            except:
                continue
        
        # Combine teams and data
        if len(teams) != len(bpi_data):
            min_len = min(len(teams), len(bpi_data))
            teams = teams[:min_len]
            bpi_data = bpi_data[:min_len]
        
        # Create DataFrame
        if len(teams) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(bpi_data)
        df.insert(0, 'Team', teams)
        df['Year'] = year
        
        # Reorder columns: Team, BPI, BPI_Rk, Off, Def, Year
        df = df[['Team', 'BPI', 'BPI_Rk', 'Off', 'Def', 'Year']]
        
        print(f"    ✓ {year}: {len(df)} teams")
        return df
        
    except Exception as e:
        print(f"    ✗ {year}: Error - {e}")
        return pd.DataFrame()
        
    finally:
        driver.quit()

def main():
    """Main execution function."""
    
    print("="*70)
    print("ESPN BPI Scraper")
    print("="*70)
    print("\nScraping historical data (2008-2025) + current season (2026)...")
    print()
    
    # Scrape all years including 2026
    all_data = []
    for year in range(2008, 2027):  # 2008-2026
        df = scrape_espn_bpi_year(year)
        if len(df) > 0:
            all_data.append(df)
        time.sleep(2)  # Delay between years
    
    if len(all_data) == 0:
        print("\n⚠ WARNING: No data scraped!")
        return
    
    # Combine all years
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Display summary
    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"Total rows: {len(df_combined)}")
    print(f"Years: {sorted(df_combined['Year'].unique())}")
    print(f"Columns: {df_combined.columns.tolist()}")
    
    print("\nBreakdown by year:")
    year_counts = df_combined.groupby('Year').size()
    for year, count in year_counts.items():
        print(f"  {year}: {count} teams")
    
    print("\nSample of scraped data:")
    print(df_combined.head(10).to_string(index=False))
    
    # Save to CSV
    output_path = '../../L1/data/espnBPI/espnBPI_raw_L1.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    print("\nNext step: Run espnBPI_transform_L1.py")
    print("  - 2026 data → espnBPI_predict_L2.csv")
    print("  - 2008-2025 data → espnBPI_analyze_L2.csv")

if __name__ == "__main__":
    main()