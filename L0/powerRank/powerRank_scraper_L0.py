"""
Power Rank Current Season Scraper (L0) - Simple Version
Attempts to scrape using requests before falling back to Selenium
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time


def scrape_with_requests(url: str) -> pd.DataFrame:
    """
    Attempt to scrape Power Rankings using simple HTTP requests
    
    Args:
        url: URL to Power Rank college basketball rankings
        
    Returns:
        DataFrame with Team, PowerRank, Year columns
    """
    print("Attempting to scrape with requests (simple method)...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Try to find rankings table
    # Adjust these selectors based on actual page structure
    table = soup.find('table')
    
    if not table:
        print("✗ No table found with simple method")
        return None
    
    data = []
    rows = table.find_all('tr')[1:]  # Skip header row
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 2:
            rank = cells[0].get_text(strip=True)
            team = cells[1].get_text(strip=True)
            
            if rank and team:
                data.append({
                    'Team': team,
                    'PowerRank': rank,
                    'Year': 2026
                })
    
    if data:
        print(f"✓ Successfully scraped {len(data)} teams with simple method")
        return pd.DataFrame(data)
    else:
        print("✗ No data extracted with simple method")
        return None


def scrape_power_rankings(url: str, output_path: str):
    """
    Scrape current Power Rankings, trying simple method first
    
    Args:
        url: URL to Power Rank college basketball rankings
        output_path: Path to save scraped CSV
    """
    
    print(f"Starting Power Rank scraper for current season...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print()
    
    # Try simple requests first
    try:
        df = scrape_with_requests(url)
        
        if df is not None and len(df) > 0:
            # Convert PowerRank to numeric
            df['PowerRank'] = pd.to_numeric(df['PowerRank'], errors='coerce')
            
            # Sort by PowerRank
            df = df.sort_values('PowerRank').reset_index(drop=True)
            
            # Display sample
            print("\nSample of scraped data:")
            print(df.head(10).to_string(index=False))
            
            # Save to CSV
            print(f"\nSaving to {output_path}...")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            print(f"\n✓ Scraping complete! Saved {len(df)} teams to {output_path}")
            return df
            
    except Exception as e:
        print(f"✗ Simple method failed: {str(e)}")
        print("\nℹ️  If this continues to fail, try the Selenium version:")
        print("   powerRank_scraper_L0.py (uses browser automation)")
        raise


if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up from L0/powerRank to project root
    
    url = "https://thepowerrank.com/college-basketball-rankings/"
    output_file = project_root / "L1" / "data" / "powerRank" / "powerRank_rawCurrent_L1.csv"
    
    # Run scraper
    df_result = scrape_power_rankings(url, str(output_file))
    
    print(f"\n✓ Scraper complete!")
    print(f"✓ Output saved to: {output_file}")
