"""
Massey Composite Rankings Transform (L1)

Transforms raw Massey Composite data into modeling-ready format.

Input:  ../data/masseyComposite/masseyComposite_raw_L1.csv
Output: ../../L2/data/masseyComposite/masseyComposite_analyze_L2.csv (historical)
        ../../L2/data/masseyComposite/masseyComposite_predict_L2.csv (2026)

Transformations:
- Standardize team names via teamsIndex.csv
- Convert year columns ('01 → 2001, etc.)
- Reshape to long format
- Join with teamsIndex to get Team and Index columns
- Split into historical (analyze) and current (predict) datasets
"""

import pandas as pd
import os
from pathlib import Path

def load_teams_index():
    """Load the teams index for name standardization"""
    teams_index_path = '../../utils/teamsIndex.csv'
    
    if not os.path.exists(teams_index_path):
        raise FileNotFoundError(f"teamsIndex.csv not found at {teams_index_path}")
    
    teams_df = pd.read_csv(teams_index_path)
    
    print(f"Loaded {len(teams_df)} teams from teamsIndex.csv")
    print(f"Columns in teamsIndex: {teams_df.columns.tolist()}")
    
    return teams_df

def standardize_team_name(raw_name, teams_df):
    """
    Standardize a team name using the teams index.
    Returns tuple of (Team, Index) if found, (None, None) otherwise.
    
    Tries multiple matching strategies including handling "St" abbreviations.
    """
    # Try exact match first
    match = teams_df[teams_df['Team'] == raw_name]
    if not match.empty:
        return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    # Try case-insensitive match
    match = teams_df[teams_df['Team'].str.lower() == raw_name.lower()]
    if not match.empty:
        return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    # Try with "St" → "State" conversion
    if ' St' in raw_name:
        try_name = raw_name.replace(' St', ' State')
        match = teams_df[teams_df['Team'].str.lower() == try_name.lower()]
        if not match.empty:
            return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    # Try with "State" → "St" conversion
    if ' State' in raw_name:
        try_name = raw_name.replace(' State', ' St')
        match = teams_df[teams_df['Team'].str.lower() == try_name.lower()]
        if not match.empty:
            return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    # Try with "Univ" → "University" conversion
    if ' Univ' in raw_name:
        try_name = raw_name.replace(' Univ', ' University')
        match = teams_df[teams_df['Team'].str.lower() == try_name.lower()]
        if not match.empty:
            return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    # Try with "University" → "Univ" conversion
    if ' University' in raw_name:
        try_name = raw_name.replace(' University', ' Univ')
        match = teams_df[teams_df['Team'].str.lower() == try_name.lower()]
        if not match.empty:
            return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    # Try with "So" → "Southern" conversion
    if ' So' in raw_name:
        try_name = raw_name.replace(' So', ' Southern')
        match = teams_df[teams_df['Team'].str.lower() == try_name.lower()]
        if not match.empty:
            return match.iloc[0]['Team'], match.iloc[0]['Index']
    
    return None, None

def convert_year_column(year_str):
    """
    Convert year string from '01 format to full year (2001).
    
    Args:
        year_str: String like '01, '02, ..., '26
    
    Returns:
        Integer year (2001, 2002, ..., 2026)
    """
    # Remove leading apostrophe and convert to int
    year_num = int(year_str.replace("'", ""))
    
    # All years are 2000s (2001-2026)
    return 2000 + year_num

def reshape_to_long_format(df_wide, teams_df):
    """
    Reshape wide format (one row per team, columns per year) to long format.
    
    Args:
        df_wide: DataFrame with Team column and year columns ('01, '02, etc.)
        teams_df: Teams index for name standardization
    
    Returns:
        DataFrame in long format with columns: Year, Team, Index, masseyComposite_Rank
    """
    # Get year columns (those starting with ')
    year_columns = [col for col in df_wide.columns if col.startswith("'")]
    
    print(f"\nReshaping data from wide to long format...")
    print(f"Found {len(year_columns)} year columns")
    
    # Reshape to long format
    df_long = df_wide.melt(
        id_vars=['Team'],
        value_vars=year_columns,
        var_name='year_str',
        value_name='masseyComposite_Rank'
    )
    
    # Rename Team to raw_team for clarity
    df_long = df_long.rename(columns={'Team': 'raw_team'})
    
    # Convert year strings to integers
    df_long['Year'] = df_long['year_str'].apply(convert_year_column)
    
    # Drop the year_str column (we have Year now)
    df_long = df_long.drop(columns=['year_str'])
    
    # Drop rows where masseyComposite_Rank is None (team didn't exist that year)
    df_long = df_long[df_long['masseyComposite_Rank'].notna()].copy()
    
    # Convert masseyComposite_Rank to integer
    df_long['masseyComposite_Rank'] = df_long['masseyComposite_Rank'].astype(int)
    
    # Standardize team names
    print("\nStandardizing team names...")
    standardized = df_long['raw_team'].apply(lambda x: standardize_team_name(x, teams_df))
    df_long['Team'] = standardized.apply(lambda x: x[0])
    df_long['Index'] = standardized.apply(lambda x: x[1])
    
    # Check for unmatched teams
    unmatched = df_long[df_long['Team'].isna()]['raw_team'].unique()
    if len(unmatched) > 0:
        print(f"\nWARNING: {len(unmatched)} teams could not be matched:")
        for team in sorted(unmatched)[:20]:  # Show first 20, sorted
            print(f"  - {team}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")
        
        print("\nTo fix: Add these team name variants to teamsIndex.csv")
    
    # Remove unmatched teams
    matched_count = df_long['Team'].notna().sum()
    total_count = len(df_long)
    match_rate = (matched_count / total_count) * 100
    print(f"\nMatch rate: {matched_count}/{total_count} ({match_rate:.1f}%)")
    
    df_long = df_long[df_long['Team'].notna()].copy()
    
    # Drop raw_team column
    df_long = df_long.drop(columns=['raw_team'])
    
    # Reorder columns: Year, Team, Index, masseyComposite_Rank
    df_long = df_long[['Year', 'Team', 'Index', 'masseyComposite_Rank']]
    
    return df_long

def split_analyze_predict(df_long):
    """
    Split data into historical (analyze) and current season (predict).
    
    Args:
        df_long: Long format DataFrame
    
    Returns:
        Tuple of (analyze_df, predict_df)
    """
    # 2026 is current season (predict)
    df_predict = df_long[df_long['Year'] == 2026].copy()
    
    # All other years are historical (analyze)
    df_analyze = df_long[df_long['Year'] != 2026].copy()
    
    print(f"\nSplit into:")
    print(f"  Analyze (historical): {len(df_analyze)} records ({df_analyze['Year'].min()}-{df_analyze['Year'].max()})")
    print(f"  Predict (2026): {len(df_predict)} records")
    
    return df_analyze, df_predict

def main():
    """Main transformation function"""
    
    print("=" * 60)
    print("Massey Composite Transform (L1)")
    print("=" * 60)
    
    # Input file
    input_file = '../data/masseyComposite/masseyComposite_raw_L1.csv'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"\nLoading raw data from {input_file}")
    df_raw = pd.read_csv(input_file)
    print(f"Loaded {len(df_raw)} teams")
    
    # Load teams index
    teams_df = load_teams_index()
    
    # Reshape to long format and standardize team names
    df_long = reshape_to_long_format(df_raw, teams_df)
    
    # Split into analyze and predict
    df_analyze, df_predict = split_analyze_predict(df_long)
    
    # Create output directory
    output_dir = '../../L2/data/masseyComposite'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save analyze dataset
    analyze_file = os.path.join(output_dir, 'masseyComposite_analyze_L2.csv')
    df_analyze.to_csv(analyze_file, index=False)
    print(f"\nSaved analyze dataset: {analyze_file}")
    print(f"  Shape: {df_analyze.shape}")
    print(f"  Years: {df_analyze['Year'].min()}-{df_analyze['Year'].max()}")
    print(f"  Teams: {df_analyze['Team'].nunique()}")
    
    # Save predict dataset
    predict_file = os.path.join(output_dir, 'masseyComposite_predict_L2.csv')
    df_predict.to_csv(predict_file, index=False)
    print(f"\nSaved predict dataset: {predict_file}")
    print(f"  Shape: {df_predict.shape}")
    print(f"  Teams: {df_predict['Team'].nunique()}")
    
    # Show sample data
    print("\n" + "=" * 60)
    print("Sample from analyze dataset:")
    print("=" * 60)
    print(df_analyze.head(10))
    
    print("\n" + "=" * 60)
    print("Sample from predict dataset:")
    print("=" * 60)
    print(df_predict.head(10))
    
    # Show summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nAnalyze dataset:")
    print(f"  Records per year: {df_analyze.groupby('Year').size().mean():.0f} (avg)")
    print(f"  Rank range: {df_analyze['masseyComposite_Rank'].min()}-{df_analyze['masseyComposite_Rank'].max()}")
    
    print(f"\nPredict dataset:")
    print(f"  Rank range: {df_predict['masseyComposite_Rank'].min()}-{df_predict['masseyComposite_Rank'].max()}")
    
    print("\n" + "=" * 60)
    print("Transform complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()