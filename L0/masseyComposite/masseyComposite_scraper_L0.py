"""
Massey Composite Scraper (L0) - Cloudflare Version

Uses undetected-chromedriver to bypass Cloudflare bot detection.
"""

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import datetime


def scrape_massey_composite(date='20260118'):
    """
    Scrape Massey Composite rankings using undetected-chromedriver.
    """
    url = f"https://masseyratings.com/ranks?s=cb2026&sym=cmp&d={date}"
    
    print(f"Fetching Massey Composite data for {date}...")
    print(f"URL: {url}")
    
    # Explicitly specify Chrome version 144 to match your installed Chrome
    print("Starting undetected Chrome (version 144)...")
    
    try:
        driver = uc.Chrome(
            version_main=144,  # Match your Chrome 144.0.7559.134
            headless=False, 
            use_subprocess=True
        )
    except Exception as e:
        print(f"Error starting Chrome: {e}")
        print("\nTrying without version specification...")
        driver = uc.Chrome(headless=False)
    
    try:
        # Load page
        print("Loading page...")
        driver.get(url)
        
        # Wait for Cloudflare challenge
        print("Waiting for Cloudflare challenge... (15 seconds)")
        time.sleep(15)
        
        # Additional wait for table
        print("Waiting for data to load... (5 seconds)")
        time.sleep(5)
        
        # Get HTML
        html = driver.page_source
        
        # Check if stuck on Cloudflare
        if 'Checking your browser' in html or 'Just a moment' in html:
            print("⚠ Still on Cloudflare, waiting 10 more seconds...")
            time.sleep(10)
            html = driver.page_source
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        
        print(f"Found {len(tables)} table(s)")
        
        if len(tables) == 0:
            with open('massey_debug.html', 'w') as f:
                f.write(html)
            raise ValueError("No tables found. Check massey_debug.html")
        
        # Find data table
        data_table = None
        max_rows = 0
        
        for idx, table in enumerate(tables):
            rows = table.find_all('tr')
            print(f"  Table {idx}: {len(rows)} rows")
            
            if len(rows) > max_rows and len(rows) > 100:
                data_table = table
                max_rows = len(rows)
        
        if data_table is None:
            raise ValueError("No data table found (need 100+ rows)")
        
        print(f"✓ Using table with {max_rows} rows")
        
        # Extract data
        rows = data_table.find_all('tr')
        all_data = []
        years = []
        header_found = False
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            
            if not cell_texts or len(cell_texts) < 2:
                continue
            
            # Find header
            if not header_found and 'Team' in cell_texts:
                years = [col for col in cell_texts if col.startswith("'")]
                print(f"✓ Header: {len(years)} years ({years[0]} to {years[-1]})")
                header_found = True
                continue
            
            # Process data
            if header_found:
                team_name = cell_texts[0]
                
                if team_name in ['Team', ''] or team_name.startswith("'"):
                    continue
                
                rankings = cell_texts[1:]
                row_data = {'Team': team_name}
                
                for i, ranking in enumerate(rankings):
                    if i < len(years):
                        row_data[years[i]] = ranking if ranking not in ['--', ''] else None
                
                all_data.append(row_data)
        
        if not header_found:
            raise ValueError("No header row found")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"\n✓ Scraped {len(df)} teams across {len(years)} years")
        
        # Add metadata
        df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        df['source_date'] = date
        
        return df
        
    finally:
        print("Closing browser...")
        driver.quit()


def main():
    """Main execution"""
    
    print("=" * 60)
    print("Massey Composite Scraper (L0)")
    print("=" * 60)
    
    # Output directory
    output_dir = '../../L1/data/masseyComposite'
    os.makedirs(output_dir, exist_ok=True)
    
    # Scrape
    df = scrape_massey_composite()
    
    # Save
    output_file = os.path.join(output_dir, 'masseyComposite_raw_L1.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Shape: {df.shape}")
    
    # Sample
    print("\nSample:")
    if 'Duke' in df['Team'].values:
        duke = df[df['Team'] == 'Duke'].iloc[0]
        print(f"  Duke: {dict(list(duke.items())[:5])}")
    else:
        first = df.iloc[0]
        print(f"  {first['Team']}: {dict(list(first.items())[:5])}")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()