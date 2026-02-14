"""
Deduplicate teamsIndex.csv while preserving manual edits
"""

import pandas as pd
import os

# Load current teamsIndex
teams_index_path = 'teamsIndex.csv'
df = pd.read_csv(teams_index_path)

print(f"Before deduplication: {len(df)} rows")
print(f"Unique teams: {df['Team'].nunique()}")

# Show some duplicate examples before removing
duplicates = df[df.duplicated(subset=['Team'], keep=False)]
if len(duplicates) > 0:
    print(f"\nDuplicates found: {len(duplicates)} rows")
    print("\nSample duplicates (keeping first occurrence):")
    print(duplicates.sort_values('Team').head(10))

# Remove duplicates, keeping first occurrence
df_clean = df.drop_duplicates(subset=['Team'], keep='first')

print(f"\nAfter deduplication: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} duplicate rows")

# Save cleaned version
df_clean.to_csv(teams_index_path, index=False)
print(f"\n✓ Saved cleaned teamsIndex.csv to: {teams_index_path}")