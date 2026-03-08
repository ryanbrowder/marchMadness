"""
ESPN BPI Current Season Scraper (L0)

Scrapes ESPN BPI data for current season only (CURRENT_YEAR).
Run frequently during active season for updates.

Outputs to: L1/data/espnBPI/espnBPI_current_raw.csv
Columns: Team, BPI, BPI_Rk, Off, Def, Year
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from utils import CURRENT_YEAR

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

def scrape_espn_bpi_current():
    """
    Scrape ESPN BPI data for current season (CURRENT_YEAR).
    Uses base URL without season parameter.
    
    Returns:
        DataFrame with Team, BPI, BPI_Rk, Off, Def, Year columns
    """
    url = 'https://www.espn.com/mens-college-basketball/bpi'
    year = CURRENT_YEAR
    
    print(f"  Scraping {year} (current season)...")
    
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
        
        # Extract team names and data together to ensure alignment
        team_table = tables[0]
        data_table = tables[1]
        
        team_rows = team_table.find_elements(By.CSS_SELECTOR, "tbody tr")
        data_rows = data_table.find_elements(By.CSS_SELECTOR, "tbody tr")
        
        # Process rows together to maintain alignment
        data = []
        skipped_teams = []
        
        for i in range(min(len(team_rows), len(data_rows))):
            team_row = team_rows[i]
            data_row = data_rows[i]
            
            try:
                # Extract team name
                team_links = team_row.find_elements(By.CSS_SELECTOR, "a[href*='/team/']")
                team_name = None
                
                # Try to get team name from links first
                for link in team_links:
                    text = link.text.strip()
                    if text and text != 'TEAM':
                        team_name = text
                        break
                
                # Fallback: If no links found, try to get plain text from the row
                # This handles defunct/dropped D1 teams that ESPN doesn't link
                if not team_name:
                    # Try to find any text in the row
                    row_text = team_row.text.strip()
                    if row_text:
                        # The row text might contain other info, try to get just the team name
                        # Typically the team name is in a span or div after the logo
                        try:
                            # Look for any span with class containing "AnchorLink" or similar
                            spans = team_row.find_elements(By.TAG_NAME, "span")
                            for span in spans:
                                text = span.text.strip()
                                # Skip conference names and other short codes
                                if text and len(text) > 2 and text not in ['CAA', 'Pac-12', 'Big 12', 'SEC', 'ACC', 'Big Ten']:
                                    team_name = text
                                    break
                        except:
                            pass
                    
                    # Last resort: use the raw row text if we have it
                    if not team_name and row_text:
                        team_name = row_text
                
                if not team_name:
                    skipped_teams.append({'row': i+1, 'team': 'UNKNOWN', 'reason': 'No team name found'})
                    continue
                
                # Extract data columns
                # Column order: W-L (0), BPI (1), BPI RK (2), TREND (3), OFF (4), DEF (5), ...
                cells = data_row.find_elements(By.TAG_NAME, "td")
                
                if len(cells) < 6:
                    skipped_teams.append({'row': i+1, 'team': team_name, 'reason': f'Only {len(cells)} cells (need 6+)'})
                    continue
                
                # Extract: BPI (score), BPI RK (rank), OFF, DEF
                bpi_score = cells[1].text.strip()
                bpi_rank = cells[2].text.strip()
                off = cells[4].text.strip()
                def_val = cells[5].text.strip()
                
                # Parse values
                try:
                    data.append({
                        'Team': team_name,
                        'BPI': float(bpi_score),
                        'BPI_Rk': int(bpi_rank),
                        'Off': float(off),
                        'Def': float(def_val)
                    })
                except ValueError as e:
                    skipped_teams.append({
                        'row': i+1, 
                        'team': team_name, 
                        'reason': f'Parse error: BPI={bpi_score}, Rank={bpi_rank}, Off={off}, Def={def_val}'
                    })
                    continue
                    
            except Exception as e:
                team_name = 'UNKNOWN'
                try:
                    # Try to get team name for better error reporting
                    team_links = team_row.find_elements(By.CSS_SELECTOR, "a[href*='/team/']")
                    for link in team_links:
                        text = link.text.strip()
                        if text and text != 'TEAM':
                            team_name = text
                            break
                except:
                    pass
                skipped_teams.append({'row': i+1, 'team': team_name, 'reason': f'Extraction error: {str(e)}'})
                continue
        
        if len(skipped_teams) > 0:
            print(f"    Skipped {len(skipped_teams)} rows:")
            for skip in skipped_teams[:5]:  # Show first 5
                print(f"      Row {skip['row']:3d}: {skip['team']:30s}")
                print(f"               {skip['reason']}")
            if len(skipped_teams) > 5:
                print(f"      ... and {len(skipped_teams) - 5} more")
        
        print(f"    Successfully extracted {len(data)} teams")
        
        # Create DataFrame
        if len(data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
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
    print("ESPN BPI Current Season Scraper")
    print("="*70)
    
    print(f"\nScraping current season data")
    print(f"Year: {CURRENT_YEAR}")
    print()
    
    # Scrape current year
    df = scrape_espn_bpi_current()
    
    if len(df) == 0:
        print("\n⚠ WARNING: No data scraped!")
        return
    
    # Display summary
    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"Total rows: {len(df)}")
    print(f"Year: {CURRENT_YEAR}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nSample of scraped data:")
    print(df.head(10).to_string(index=False))
    
    # Save to CSV
    output_path = '../../L1/data/espnBPI/espnBPI_current_raw.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    print("\nNext: Run espnBPI_transform_L1.py to combine with historical data")

if __name__ == "__main__":
    main()
