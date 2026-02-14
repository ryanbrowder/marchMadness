#!/usr/bin/env python3
"""
NCAA Tournament Historical Game Scraper - Robust Version
Handles errors gracefully and continues scraping other years
"""

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TournamentScraper:
    """Scrapes NCAA Tournament game data from Sports-Reference.com"""
    
    def __init__(self, start_year=2008, end_year=2025, headless=True):
        self.start_year = start_year
        self.end_year = end_year
        self.base_url = "https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
        self.all_games = []
        self.failed_years = []
        
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        # Add user agent to avoid detection
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # JavaScript scraper
        self.scraper_js = """
        function scrapeTournamentGames(year) {
            const games = [];
            const regionNames = ['East', 'Midwest', 'South', 'West', 'National'];
            
            const regionalRoundMapping = {
                8: 'R64',
                4: 'R32', 
                2: 'S16',
                1: 'E8'
            };
            
            const nationalRoundMapping = {
                2: 'F4',
                1: 'Championship'
            };
            
            const brackets = document.querySelectorAll('#bracket');
            
            brackets.forEach((bracket, bracketIdx) => {
                const region = regionNames[bracketIdx] || `Unknown_${bracketIdx}`;
                const isNational = region === 'National';
                const roundMapping = isNational ? nationalRoundMapping : regionalRoundMapping;
                
                const rounds = bracket.querySelectorAll('.round');
                
                rounds.forEach((roundElem) => {
                    const gameDivs = Array.from(roundElem.querySelectorAll('div'))
                        .filter(div => div.innerHTML.includes('<!-- game -->'));
                    
                    const numGames = gameDivs.length;
                    const roundName = roundMapping[numGames] || `Unknown_${numGames}`;
                    
                    gameDivs.forEach(gameDiv => {
                        try {
                            const teamDivs = Array.from(gameDiv.children)
                                .filter(child => child.tagName === 'DIV');
                            
                            if (teamDivs.length < 2) return;
                            
                            const teamADiv = teamDivs[0];
                            const seedASpan = teamADiv.querySelector('span');
                            const seedA = seedASpan ? seedASpan.textContent.trim() : '';
                            const teamALinks = teamADiv.querySelectorAll('a');
                            const teamA = teamALinks[0] ? teamALinks[0].textContent.trim() : '';
                            const scoreA = teamALinks[1] ? teamALinks[1].textContent.trim() : '';
                            
                            const teamBDiv = teamDivs[1];
                            const seedBSpan = teamBDiv.querySelector('span');
                            const seedB = seedBSpan ? seedBSpan.textContent.trim() : '';
                            const teamBLinks = teamBDiv.querySelectorAll('a');
                            const teamB = teamBLinks[0] ? teamBLinks[0].textContent.trim() : '';
                            const scoreB = teamBLinks[1] ? teamBLinks[1].textContent.trim() : '';
                            
                            let winner = '';
                            if (teamADiv.classList.contains('winner')) {
                                winner = teamA;
                            } else if (teamBDiv.classList.contains('winner')) {
                                winner = teamB;
                            }
                            
                            let location = '';
                            const directSpans = Array.from(gameDiv.children).filter(child => child.tagName === 'SPAN');
                            if (directSpans.length > 0) {
                                const locText = directSpans[0].textContent.trim();
                                location = locText.replace('at ', '').trim();
                            }
                            
                            if (teamA && teamB) {
                                games.push({
                                    Year: year,
                                    Region: region,
                                    Round: roundName,
                                    TeamA: teamA,
                                    SeedA: seedA,
                                    ScoreA: scoreA,
                                    TeamB: teamB,
                                    SeedB: seedB,
                                    ScoreB: scoreB,
                                    Winner: winner,
                                    Location: location
                                });
                            }
                        } catch (e) {
                            console.error('Error parsing game:', e);
                        }
                    });
                });
            });
            
            return games;
        }
        
        return scrapeTournamentGames(arguments[0]);
        """
    
    def scrape_year(self, year):
        """Scrape all tournament games for a given year"""
        url = self.base_url.format(year=year)
        logger.info(f"Scraping {year}: {url}")
        
        try:
            # Navigate to the page
            self.driver.get(url)
            
            # Wait longer for bracket to load (up to 15 seconds)
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.ID, "bracket"))
                )
            except TimeoutException:
                logger.warning(f"  ⚠ Timeout waiting for bracket on {year} page")
                logger.info(f"  Checking if page exists...")
                
                # Check if we got a 404 or similar
                page_source = self.driver.page_source
                if "404" in page_source or "not found" in page_source.lower():
                    logger.error(f"  ✗ Page not found for {year}")
                    return []
                
                # Try waiting a bit more
                logger.info(f"  Waiting additional 5 seconds...")
                time.sleep(5)
                
                # Try finding bracket again
                brackets = self.driver.find_elements(By.ID, "bracket")
                if not brackets:
                    logger.error(f"  ✗ No bracket found for {year}")
                    return []
            
            # Give page time to fully render
            time.sleep(3)
            
            # Execute the scraper JavaScript
            games = self.driver.execute_script(self.scraper_js, year)
            
            if games:
                logger.info(f"  ✓ Collected {len(games)} games for {year}")
                return games
            else:
                logger.warning(f"  ⚠ No games found for {year}")
                return []
                
        except TimeoutException as e:
            logger.error(f"  ✗ Timeout error for {year}: {e}")
            self.failed_years.append(year)
            return []
        except Exception as e:
            logger.error(f"  ✗ Error scraping {year}: {e}")
            self.failed_years.append(year)
            return []
    
    def scrape_all_years(self):
        """Scrape tournament data for all years"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting scrape: {self.start_year} - {self.end_year}")
        logger.info(f"{'='*60}\n")
        
        for year in range(self.start_year, self.end_year + 1):
            year_games = self.scrape_year(year)
            self.all_games.extend(year_games)
            
            # Be polite to the server
            time.sleep(2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping complete!")
        logger.info(f"Total games collected: {len(self.all_games)}")
        if self.failed_years:
            logger.warning(f"Failed years: {self.failed_years}")
        logger.info(f"{'='*60}\n")
    
    def save_to_csv(self, output_path='../../L1/data/srcbb/srcbb_transform_L1.csv'):
        """Save scraped data to CSV"""
        if not self.all_games:
            logger.warning("No games to save!")
            return
        
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.all_games)
        
        # Reorder columns
        column_order = [
            'Year', 'Region', 'Round', 
            'TeamA', 'SeedA', 'ScoreA',
            'TeamB', 'SeedB', 'ScoreB',
            'Winner', 'Location'
        ]
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Data saved to {output_file}")
        
        # Print summary
        self._print_summary(df)
    
    def _print_summary(self, df):
        """Print summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("DATA SUMMARY")
        logger.info("="*60)
        logger.info(f"Total games: {len(df)}")
        logger.info(f"Years covered: {df['Year'].min()} - {df['Year'].max()}")
        logger.info(f"\nGames per year:")
        year_counts = df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count} games")
        
        if self.failed_years:
            logger.warning(f"\n⚠ Failed to scrape: {self.failed_years}")
            logger.warning(f"  These years may not exist or have different page structure")
        
        logger.info("="*60 + "\n")
    
    def close(self):
        """Close the browser"""
        self.driver.quit()


def main():
    """Main execution function"""
    START_YEAR = 2008
    END_YEAR = 2025
    OUTPUT_FILE = '../../L1/data/srcbb/srcbb_transform_L1.csv'
    
    scraper = TournamentScraper(
        start_year=START_YEAR, 
        end_year=END_YEAR,
        headless=True
    )
    
    try:
        scraper.scrape_all_years()
        scraper.save_to_csv(OUTPUT_FILE)
        
    except KeyboardInterrupt:
        logger.info("\n\nScraping interrupted by user")
        
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        
    finally:
        scraper.close()
        logger.info("Browser closed")


if __name__ == "__main__":
    main()