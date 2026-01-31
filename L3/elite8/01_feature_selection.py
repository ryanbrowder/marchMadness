"""
L3 Feature Selection Pipeline
Joins training datasets with tournament results, creates Elite 8 label, 
and performs feature selection through correlation and VIF analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional import for VIF analysis
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. VIF analysis will be skipped.")

# Configuration
DATA_DIR = Path('../data')
TRAINING_DIR = DATA_DIR / 'trainingData'
OUTPUT_DIR = Path('outputs/01_feature_selection')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ELITE8_OUTCOMES = ['CHAMPS', 'Finals', 'Final Four', 'Elite Eight']  # Fixed: "Elite Eight" not "Elite 8"
CORRELATION_THRESHOLD = 0.90  # Drop features with correlation above this
VIF_THRESHOLD = 10  # Flag features with VIF above this

print("="*80)
print("L3 FEATURE SELECTION PIPELINE")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 80)

# Load training datasets
training_long = pd.read_csv(TRAINING_DIR / 'training_set_long.csv')
training_rich = pd.read_csv(TRAINING_DIR / 'training_set_rich.csv')

print(f"training_set_long: {training_long.shape[0]} rows, {training_long.shape[1]} columns")
print(f"training_set_rich: {training_rich.shape[0]} rows, {training_rich.shape[1]} columns")

# Load tournament results
tournament_results = pd.read_csv(DATA_DIR / 'tournamentResults.csv')
print(f"tournamentResults: {tournament_results.shape[0]} rows")

# DIAGNOSTIC: Verify Elite 8+ definition
print("\nDIAGNOSTIC - Tournament outcome distribution:")
outcome_counts = tournament_results['tournamentOutcome'].value_counts()
print(outcome_counts.head(10))
elite8_count = tournament_results['tournamentOutcome'].isin(ELITE8_OUTCOMES).sum()
print(f"\nElite 8+ teams with current definition: {elite8_count}")
print(f"Expected Elite 8+ teams (8 per year): ~{len(tournament_results) / 68 * 8:.0f}")

# ============================================================================
# JOIN AND CREATE LABELS
# ============================================================================
print("\n[2] JOINING DATA AND CREATING LABELS")
print("-" * 80)

def prepare_training_data(training_df, tournament_df, dataset_name):
    """Join training data with tournament results and create Elite 8 label"""
    
    # Join on Year + Index (Team already exists in training_df)
    merged = training_df.merge(
        tournament_df[['Year', 'Index', 'tournamentSeed', 'tournamentOutcome']],
        on=['Year', 'Index'],
        how='inner'
    )
    
    print(f"\n{dataset_name}:")
    print(f"  After join: {merged.shape[0]} rows")
    
    # Create Elite 8 binary label
    merged['elite8_flag'] = merged['tournamentOutcome'].isin(ELITE8_OUTCOMES).astype(int)
    
    # Report label distribution
    elite8_count = merged['elite8_flag'].sum()
    elite8_pct = (elite8_count / len(merged)) * 100
    print(f"  Elite 8+ teams: {elite8_count} ({elite8_pct:.1f}%)")
    print(f"  Non-Elite 8 teams: {len(merged) - elite8_count} ({100-elite8_pct:.1f}%)")
    
    # Convert numeric columns - use 'coerce' to force conversion
    # Keep Year, Team, Index as-is
    print(f"  Converting columns to numeric types...")
    for col in merged.columns:
        if col not in ['Year', 'Team', 'Index', 'tournamentOutcome']:
            # Try to convert, coercing errors to NaN
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
    
    # Report how many columns are now numeric
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric columns after conversion: {len(numeric_cols)}")
    
    # Reorder columns: Year, Team, Index first, then features, seed, label
    feature_cols = [col for col in merged.columns 
                   if col not in ['Year', 'Team', 'Index', 'tournamentSeed', 
                                  'tournamentOutcome', 'elite8_flag']]
    
    final_cols = ['Year', 'Team', 'Index'] + feature_cols + ['tournamentSeed', 'elite8_flag']
    merged = merged[final_cols]
    
    return merged

# Prepare both datasets
labeled_long = prepare_training_data(training_long, tournament_results, "training_set_long")
labeled_rich = prepare_training_data(training_rich, tournament_results, "training_set_rich")

# Save labeled datasets
labeled_long.to_csv(OUTPUT_DIR / 'labeled_training_long.csv', index=False)
labeled_rich.to_csv(OUTPUT_DIR / 'labeled_training_rich.csv', index=False)
print(f"\nSaved labeled datasets to {OUTPUT_DIR}/")

# ============================================================================
# FEATURE-TO-FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n[3] FEATURE-TO-FEATURE CORRELATION ANALYSIS")
print("-" * 80)

def analyze_feature_correlations(df, dataset_name):
    """Identify highly correlated feature pairs"""
    
    # Get feature columns only (exclude Year, Team, Index, label)
    feature_cols = [col for col in df.columns 
                   if col not in ['Year', 'Team', 'Index', 'elite8_flag']]
    
    # Select only numeric features for correlation analysis
    feature_df = df[feature_cols].select_dtypes(include=[np.number])
    
    print(f"\n{dataset_name}:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numeric features for correlation: {len(feature_df.columns)}")
    
    if len(feature_df.columns) == 0:
        print("  WARNING: No numeric features found! Skipping correlation analysis.")
        return pd.DataFrame(), []
    
    # Calculate correlation matrix
    corr_matrix = feature_df.corr()
    
    # Find pairs with correlation above threshold
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    print(f"  High correlation pairs (>{CORRELATION_THRESHOLD}): {len(high_corr_pairs)}")
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
        print("\n  Top 10 highly correlated pairs:")
        print(high_corr_df.head(10).to_string(index=False))
        
        # Save full list
        output_file = OUTPUT_DIR / f'high_correlations_{dataset_name}.csv'
        high_corr_df.to_csv(output_file, index=False)
        print(f"\n  Saved full list to {output_file}")
    
    # Generate correlation heatmap (for smaller feature sets only)
    if len(feature_df.columns) <= 30:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Feature Correlation Matrix - {dataset_name}')
        plt.tight_layout()
        heatmap_file = OUTPUT_DIR / f'correlation_heatmap_{dataset_name}.png'
        plt.savefig(heatmap_file, dpi=150)
        plt.close()
        print(f"  Saved heatmap to {heatmap_file}")
    else:
        print(f"  (Heatmap skipped - too many features for visualization)")
    
    return corr_matrix, high_corr_pairs

corr_long, pairs_long = analyze_feature_correlations(labeled_long, "long")
corr_rich, pairs_rich = analyze_feature_correlations(labeled_rich, "rich")

# ============================================================================
# FEATURE-TO-LABEL CORRELATION
# ============================================================================
print("\n[4] FEATURE-TO-LABEL CORRELATION")
print("-" * 80)

def analyze_label_correlation(df, dataset_name):
    """Rank features by correlation with Elite 8 label"""
    
    feature_cols = [col for col in df.columns 
                   if col not in ['Year', 'Team', 'Index', 'elite8_flag']]
    
    # Filter to numeric features only
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n{dataset_name}:")
    print(f"  Numeric features analyzed: {len(numeric_features)}")
    
    if len(numeric_features) == 0:
        print("  WARNING: No numeric features found! Skipping label correlation analysis.")
        return pd.DataFrame()
    
    # Calculate correlation with label
    label_corrs = []
    for col in numeric_features:
        corr = df[col].corr(df['elite8_flag'])
        label_corrs.append({
            'feature': col,
            'correlation': abs(corr),
            'correlation_raw': corr
        })
    
    label_corr_df = pd.DataFrame(label_corrs).sort_values('correlation', ascending=False)
    
    print(f"  Top 15 features by correlation with elite8_flag:")
    print(label_corr_df.head(15).to_string(index=False))
    
    # Save full ranking
    output_file = OUTPUT_DIR / f'label_correlations_{dataset_name}.csv'
    label_corr_df.to_csv(output_file, index=False)
    print(f"\n  Saved full ranking to {output_file}")
    
    return label_corr_df

label_corr_long = analyze_label_correlation(labeled_long, "long")
label_corr_rich = analyze_label_correlation(labeled_rich, "rich")

# ============================================================================
# VARIANCE INFLATION FACTOR (VIF) ANALYSIS
# ============================================================================
print("\n[5] VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
print("-" * 80)

def calculate_vif(df, dataset_name):
    """Calculate VIF for each feature to detect multicollinearity"""
    
    if not HAS_STATSMODELS:
        print(f"\n{dataset_name}:")
        print(f"  VIF analysis skipped (statsmodels not available)")
        return pd.DataFrame()
    
    feature_cols = [col for col in df.columns 
                   if col not in ['Year', 'Team', 'Index', 'elite8_flag']]
    
    # Select only numeric features
    feature_df = df[feature_cols].select_dtypes(include=[np.number]).dropna()
    
    print(f"\n{dataset_name}:")
    print(f"  Calculating VIF for {len(feature_df.columns)} numeric features...")
    
    if len(feature_df.columns) == 0:
        print("  WARNING: No numeric features found! Skipping VIF analysis.")
        return pd.DataFrame()
    
    # Calculate VIF for each feature
    vif_data = []
    for i, col in enumerate(feature_df.columns):
        try:
            vif = variance_inflation_factor(feature_df.values, i)
            vif_data.append({
                'feature': col,
                'VIF': vif
            })
        except Exception as e:
            print(f"  Warning: Could not calculate VIF for {col}: {e}")
            vif_data.append({
                'feature': col,
                'VIF': np.nan
            })
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    # Flag high VIF features
    high_vif = vif_df[vif_df['VIF'] > VIF_THRESHOLD]
    print(f"  Features with VIF > {VIF_THRESHOLD}: {len(high_vif)}")
    
    if len(high_vif) > 0:
        print("\n  Top 15 features by VIF:")
        print(vif_df.head(15).to_string(index=False))
    
    # Save full VIF results
    output_file = OUTPUT_DIR / f'vif_analysis_{dataset_name}.csv'
    vif_df.to_csv(output_file, index=False)
    print(f"\n  Saved full VIF analysis to {output_file}")
    
    return vif_df

vif_long = calculate_vif(labeled_long, "long")
vif_rich = calculate_vif(labeled_rich, "rich")

# ============================================================================
# FEATURE SELECTION RECOMMENDATIONS
# ============================================================================
print("\n[6] FEATURE SELECTION RECOMMENDATIONS")
print("-" * 80)

def recommend_feature_reduction(high_corr_pairs, label_corr_df, vif_df, dataset_name):
    """Recommend which features to drop based on correlation, VIF, and label correlation"""
    
    print(f"\n{dataset_name}:")
    
    # Check if we have valid data
    if label_corr_df.empty:
        print("  No numeric features found - cannot make recommendations")
        return set()
    
    # Build drop recommendations
    features_to_drop = set()
    drop_reasons = {}
    
    # From high correlation pairs, drop the one with lower label correlation
    for pair in high_corr_pairs:
        feat1, feat2 = pair['feature1'], pair['feature2']
        
        # Get label correlations
        corr1 = label_corr_df[label_corr_df['feature'] == feat1]['correlation'].values[0]
        corr2 = label_corr_df[label_corr_df['feature'] == feat2]['correlation'].values[0]
        
        # Drop the one with lower label correlation
        to_drop = feat1 if corr1 < corr2 else feat2
        features_to_drop.add(to_drop)
        
        reason = f"Correlated with {feat2 if to_drop == feat1 else feat1} (r={pair['correlation']:.3f})"
        drop_reasons[to_drop] = drop_reasons.get(to_drop, []) + [reason]
    
    # Flag features with very high VIF (but don't auto-drop without correlation check)
    high_vif_features = []
    if not vif_df.empty:
        high_vif_features = vif_df[vif_df['VIF'] > VIF_THRESHOLD * 2]['feature'].tolist()
    
    print(f"  Recommended drops from correlation analysis: {len(features_to_drop)}")
    if HAS_STATSMODELS:
        print(f"  High VIF features (review manually): {len(high_vif_features)}")
    
    # Create recommendations dataframe
    drop_list = []
    for feat in features_to_drop:
        reasons = "; ".join(drop_reasons.get(feat, []))
        drop_list.append({
            'feature': feat,
            'reason': reasons
        })
    
    if drop_list:
        drop_df = pd.DataFrame(drop_list)
        print("\n  Features recommended for removal:")
        print(drop_df.to_string(index=False))
        
        # Save recommendations
        output_file = OUTPUT_DIR / f'drop_recommendations_{dataset_name}.csv'
        drop_df.to_csv(output_file, index=False)
        print(f"\n  Saved recommendations to {output_file}")
    else:
        print("\n  No features recommended for removal")
    
    # Features to keep
    all_features = set(label_corr_df['feature'].tolist())
    features_to_keep = all_features - features_to_drop
    
    print(f"\n  Original features: {len(all_features)}")
    print(f"  After recommended drops: {len(features_to_keep)}")
    if len(all_features) > 0:
        print(f"  Reduction: {len(features_to_drop)} features ({len(features_to_drop)/len(all_features)*100:.1f}%)")
    
    # Save reduced feature list
    keep_df = pd.DataFrame({'feature': sorted(features_to_keep)})
    output_file = OUTPUT_DIR / f'reduced_features_{dataset_name}.csv'
    keep_df.to_csv(output_file, index=False)
    print(f"  Saved reduced feature list to {output_file}")
    
    return features_to_keep

keep_long = recommend_feature_reduction(pairs_long, label_corr_long, vif_long, "long")
keep_rich = recommend_feature_reduction(pairs_rich, label_corr_rich, vif_rich, "rich")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION COMPLETE")
print("="*80)

print("\nOUTPUTS GENERATED:")
print(f"  {OUTPUT_DIR}/labeled_training_long.csv - Training data with elite8_flag label")
print(f"  {OUTPUT_DIR}/labeled_training_rich.csv - Training data with elite8_flag label")
print(f"  {OUTPUT_DIR}/high_correlations_*.csv - Highly correlated feature pairs")
print(f"  {OUTPUT_DIR}/label_correlations_*.csv - Features ranked by label correlation")
print(f"  {OUTPUT_DIR}/vif_analysis_*.csv - Variance inflation factors")
print(f"  {OUTPUT_DIR}/drop_recommendations_*.csv - Recommended features to drop")
print(f"  {OUTPUT_DIR}/reduced_features_*.csv - Recommended features to keep")

print("\nNEXT STEPS:")
print("  1. Review correlation heatmaps and drop recommendations")
print("  2. Decide which features to keep for modeling")
print("  3. Proceed to 02_exploratory_models.py")
print("\n" + "="*80)