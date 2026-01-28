"""
Massey Composite Rankings Scraper (L0)

Scrapes the Massey Composite college basketball rankings from masseyratings.com.
This is the raw data extraction layer - minimal processing, just get the data.

Output: ../../L1/data/masseyComposite/masseyComposite_raw_L1.csv

The Massey Composite aggregates multiple rating systems into a consensus ranking.
Data includes historical rankings from 2001-present.

NOTE: Uses Selenium because the data is loaded dynamically via JavaScript.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import os
import time
from datetime import datetime

def scrape_massey_composite(date='20260118'):
    """
    Scrape Massey Composite rankings for a given date using Selenium.
    
    Args:
        date: String in format YYYYMMDD (default: 20260118, Jan 18 2026)
    
    Returns:
        DataFrame with raw Massey Composite data
    """
    # URL for Massey Composite rankings
    url = f"https://masseyratings.com/ranks?s=cb2026&sym=cmp&d={date}"
    
    print(f"Fetching Massey Composite data for {date}...")
    print(f"URL: {url}")
    
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in background
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # Initialize driver
    driver = webdriver.Chrome(options=options)
    
    try:
        # Load page
        driver.get(url)
        
        # Wait for the table to load (look for rows with team data)
        print("Waiting for table to load...")
        wait = WebDriverWait(driver, 15)
        
        # The table should have rows with team names
        # Wait for at least one row to appear
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        # Give it a moment for all data to render
        time.sleep(2)
        
        # Now parse the rendered HTML
        html = driver.page_source
        
        # Find all tables
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        
        print(f"Found {len(tables)} tables after JavaScript rendering")
        
        # Find the table with the most rows (likely the data table)
        # Based on debug output, the data table has ~400 rows
        data_table = None
        max_rows = 0
        
        for idx, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) > max_rows and len(rows) > 100:  # Data table should have 300+ teams
                data_table = table
                max_rows = len(rows)
                print(f"Found data table at index {idx} with {len(rows)} rows")
        
        if data_table is None:
            raise ValueError("Could not find data table (no table with 100+ rows found)")
        
        # Extract all rows
        rows = data_table.find_all('tr')
        print(f"Processing {len(rows)} rows...")
        
        # Process data
        all_data = []
        years = []  # To store year columns
        header_found = False
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            
            # Skip empty rows
            if not cell_texts or len(cell_texts) < 2:
                continue
            
            # Look for header row with 'Team' and years
            if not header_found and 'Team' in cell_texts:
                # Extract year columns (e.g., '01, '02, etc.)
                years = [col for col in cell_texts if col.startswith("'")]
                print(f"Found header at row {row_idx} with {len(years)} year columns: {years[0]} to {years[-1]}")
                header_found = True
                continue
            
            # Once we have the header, process data rows
            if header_found and cell_texts:
                team_name = cell_texts[0]
                
                # Skip if this is still a header or empty
                if team_name in ['Team', ''] or team_name.startswith("'"):
                    continue
                
                rankings = cell_texts[1:]  # All subsequent cells are rankings
                
                # Create row with team and rankings
                row_data = {'Team': team_name}
                for i, ranking in enumerate(rankings):
                    if i < len(years):
                        year_col = years[i]
                        row_data[year_col] = ranking if ranking not in ['--', ''] else None
                
                all_data.append(row_data)
        
        if not header_found:
            # If we still haven't found "Team", try alternative approach
            # Look for a row where first cell is a team name (contains letters, not just dates)
            print("\nAlternative approach: looking for team data without 'Team' header...")
            
            # Assume year columns are '01 through '26 (we know this from the website)
            years = [f"'{str(i).zfill(2)}" for i in range(1, 27)]  # '01 to '26
            print(f"Using assumed year columns: {years[0]} to {years[-1]}")
            
            for row_idx, row in enumerate(rows[1:], 1):  # Skip first row
                cells = row.find_all(['td', 'th'])
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                
                if not cell_texts or len(cell_texts) < 2:
                    continue
                
                team_name = cell_texts[0]
                
                # Check if this looks like a team name (has letters, not just numbers/dates)
                if team_name and any(c.isalpha() for c in team_name) and not team_name.startswith("'"):
                    rankings = cell_texts[1:]
                    
                    row_data = {'Team': team_name}
                    for i, ranking in enumerate(rankings):
                        if i < len(years):
                            year_col = years[i]
                            row_data[year_col] = ranking if ranking not in ['--', ''] else None
                    
                    all_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"\nScraped {len(df)} teams across {len(years)} years")
        
        # Add metadata
        df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        df['source_date'] = date
        
        return df
        
    finally:
        # Always close the driver
        driver.quit()

def main():
    """Main execution function"""
    
    # Output directory (from L0/masseyComposite to L1/data/masseyComposite)
    output_dir = '../../L1/data/masseyComposite'
    os.makedirs(output_dir, exist_ok=True)
    
    # Scrape data
    df = scrape_massey_composite()
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'masseyComposite_raw_L1.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved raw data to {output_file}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nColumn names:")
    print(df.columns.tolist()[:10])  # Show first 10 columns
    print(f"\nSample team data:")
    if 'Duke' in df['Team'].values:
        duke_data = df[df['Team'] == 'Duke'].iloc[0]
        # Show just a few columns for Duke
        duke_sample = {k: v for k, v in duke_data.items() if k in ['Team', "'24", "'25", "'26", 'scrape_date']}
        print(f"Duke: {duke_sample}")
    else:
        print("First team:")
        first_team = df.iloc[0]
        sample = {k: v for k, v in first_team.items() if k in ['Team', "'24", "'25", "'26", 'scrape_date']}
        print(sample)

if __name__ == "__main__":
    main()