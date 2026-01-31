"""
H2H 02_feature_correlation_analysis.py

Purpose: Analyze correlations between percentage differential features and game outcomes.
         Identify which features have predictive power and detect multicollinearity.

Inputs:
    - L3/h2h/outputs/01_build_training_matchups/training_matchups.csv

Outputs:
    - L3/h2h/outputs/02_feature_correlation/feature_target_correlations.csv
    - L3/h2h/outputs/02_feature_correlation/feature_intercorrelations.csv
    - L3/h2h/outputs/02_feature_correlation/correlation_heatmap.png
    - L3/h2h/outputs/02_feature_correlation/feature_recommendations.txt

Author: Ryan Browder
Date: 2025-01-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Input path
TRAINING_MATCHUPS_PATH = 'outputs/01_build_training_matchups/training_matchups.csv'

# Output directory
OUTPUT_DIR = 'outputs/02_feature_correlation'

# Correlation thresholds
HIGH_CORRELATION_THRESHOLD = 0.85  # Features correlated above this are redundant
MEDIUM_CORRELATION_THRESHOLD = 0.70
LOW_PREDICTIVE_POWER_THRESHOLD = 0.05  # Features below this have weak signal

# ============================================================================
# Helper Functions
# ============================================================================

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}/")

def load_training_data():
    """Load training matchups dataset."""
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    
    print(f"\nLoading training matchups from: {TRAINING_MATCHUPS_PATH}")
    df = pd.read_csv(TRAINING_MATCHUPS_PATH)
    
    print(f"  Shape: {df.shape}")
    print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")
    print(f"  Games: {len(df)}")
    
    # Identify feature columns (pct_diff_*)
    feature_cols = [col for col in df.columns if col.startswith('pct_diff_')]
    print(f"  Differential features: {len(feature_cols)}")
    
    return df, feature_cols

def analyze_target_correlations(df, feature_cols, target_col='TeamA_Won'):
    """
    Calculate correlation between each feature and the target variable.
    """
    print("\n" + "="*80)
    print("FEATURE-TARGET CORRELATIONS")
    print("="*80)
    
    correlations = []
    
    for feature in feature_cols:
        corr = df[feature].corr(df[target_col])
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    print(f"\nTop 10 features by predictive power (absolute correlation with {target_col}):")
    print(corr_df.head(10).to_string(index=False))
    
    print(f"\nBottom 10 features by predictive power:")
    print(corr_df.tail(10).to_string(index=False))
    
    # Count features by strength
    strong = len(corr_df[corr_df['abs_correlation'] >= 0.20])
    moderate = len(corr_df[(corr_df['abs_correlation'] >= 0.10) & (corr_df['abs_correlation'] < 0.20)])
    weak = len(corr_df[corr_df['abs_correlation'] < 0.10])
    
    print(f"\nFeature strength distribution:")
    print(f"  Strong (|r| >= 0.20): {strong}")
    print(f"  Moderate (0.10 <= |r| < 0.20): {moderate}")
    print(f"  Weak (|r| < 0.10): {weak}")
    
    return corr_df

def analyze_feature_intercorrelations(df, feature_cols):
    """
    Calculate correlation matrix between all features to detect multicollinearity.
    """
    print("\n" + "="*80)
    print("FEATURE INTERCORRELATIONS")
    print("="*80)
    
    # Calculate correlation matrix
    print(f"\nCalculating correlation matrix for {len(feature_cols)} features...")
    corr_matrix = df[feature_cols].corr()
    
    # Find highly correlated pairs (exclude diagonal)
    high_corr_pairs = []
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= MEDIUM_CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    'feature_1': feature_cols[i],
                    'feature_2': feature_cols[j],
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('abs_correlation', ascending=False)
        
        print(f"\n⚠ Found {len(high_corr_df)} feature pairs with |r| >= {MEDIUM_CORRELATION_THRESHOLD}:")
        print(high_corr_df.head(20).to_string(index=False))
        
        # Count by severity
        severe = len(high_corr_df[high_corr_df['abs_correlation'] >= HIGH_CORRELATION_THRESHOLD])
        moderate = len(high_corr_df[(high_corr_df['abs_correlation'] >= MEDIUM_CORRELATION_THRESHOLD) & 
                                     (high_corr_df['abs_correlation'] < HIGH_CORRELATION_THRESHOLD)])
        
        print(f"\nMulticollinearity severity:")
        print(f"  Severe (|r| >= {HIGH_CORRELATION_THRESHOLD}): {severe} pairs")
        print(f"  Moderate ({MEDIUM_CORRELATION_THRESHOLD} <= |r| < {HIGH_CORRELATION_THRESHOLD}): {moderate} pairs")
    else:
        print(f"\n✓ No feature pairs with |r| >= {MEDIUM_CORRELATION_THRESHOLD}")
        high_corr_df = pd.DataFrame()
    
    return corr_matrix, high_corr_df

def create_correlation_heatmap(corr_matrix, output_dir):
    """
    Create and save a heatmap visualization of feature correlations.
    """
    print("\n" + "="*80)
    print("CREATING CORRELATION HEATMAP")
    print("="*80)
    
    # If we have too many features, sample for visualization
    n_features = corr_matrix.shape[0]
    
    if n_features > 50:
        print(f"\n⚠ Too many features ({n_features}) for readable heatmap")
        print(f"  Creating heatmap for top 30 features by variance...")
        
        # Get top features by variance (most informative)
        variances = corr_matrix.var().sort_values(ascending=False)
        top_features = variances.head(30).index.tolist()
        corr_matrix_subset = corr_matrix.loc[top_features, top_features]
    else:
        print(f"\nCreating heatmap for all {n_features} features...")
        corr_matrix_subset = corr_matrix
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix_subset,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Intercorrelation Heatmap', fontsize=16, pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Heatmap saved to: {output_path}")

def generate_recommendations(target_corr_df, intercorr_df, output_dir):
    """
    Generate feature selection recommendations based on correlation analysis.
    """
    print("\n" + "="*80)
    print("GENERATING RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    recommendations.append("="*80)
    recommendations.append("FEATURE SELECTION RECOMMENDATIONS")
    recommendations.append("="*80)
    recommendations.append("")
    
    # Recommendation 1: Strong features to keep
    strong_features = target_corr_df[target_corr_df['abs_correlation'] >= 0.20]
    recommendations.append(f"1. KEEP - Strong predictive features (|r| >= 0.20): {len(strong_features)} features")
    recommendations.append("")
    for _, row in strong_features.head(15).iterrows():
        recommendations.append(f"   {row['feature']}: r = {row['correlation']:.3f}")
    if len(strong_features) > 15:
        recommendations.append(f"   ... and {len(strong_features) - 15} more")
    recommendations.append("")
    
    # Recommendation 2: Moderate features to consider
    moderate_features = target_corr_df[
        (target_corr_df['abs_correlation'] >= 0.10) & 
        (target_corr_df['abs_correlation'] < 0.20)
    ]
    recommendations.append(f"2. CONSIDER - Moderate predictive features (0.10 <= |r| < 0.20): {len(moderate_features)} features")
    recommendations.append("   These may add value in ensemble models")
    recommendations.append("")
    
    # Recommendation 3: Weak features to drop
    weak_features = target_corr_df[target_corr_df['abs_correlation'] < LOW_PREDICTIVE_POWER_THRESHOLD]
    recommendations.append(f"3. DROP - Weak predictive features (|r| < {LOW_PREDICTIVE_POWER_THRESHOLD}): {len(weak_features)} features")
    if len(weak_features) > 0:
        recommendations.append("   These add noise without signal:")
        recommendations.append("")
        for _, row in weak_features.head(10).iterrows():
            recommendations.append(f"   {row['feature']}: r = {row['correlation']:.3f}")
        if len(weak_features) > 10:
            recommendations.append(f"   ... and {len(weak_features) - 10} more")
    recommendations.append("")
    
    # Recommendation 4: Handle multicollinearity
    if len(intercorr_df) > 0:
        severe_pairs = intercorr_df[intercorr_df['abs_correlation'] >= HIGH_CORRELATION_THRESHOLD]
        recommendations.append(f"4. RESOLVE MULTICOLLINEARITY - {len(severe_pairs)} severe redundancies (|r| >= {HIGH_CORRELATION_THRESHOLD})")
        recommendations.append("   For each pair, keep the feature with higher target correlation:")
        recommendations.append("")
        
        for _, row in severe_pairs.head(10).iterrows():
            feat1_corr = target_corr_df[target_corr_df['feature'] == row['feature_1']]['abs_correlation'].values[0]
            feat2_corr = target_corr_df[target_corr_df['feature'] == row['feature_2']]['abs_correlation'].values[0]
            
            if feat1_corr > feat2_corr:
                keep, drop = row['feature_1'], row['feature_2']
            else:
                keep, drop = row['feature_2'], row['feature_1']
            
            recommendations.append(f"   {row['feature_1']} <-> {row['feature_2']} (r = {row['correlation']:.3f})")
            recommendations.append(f"      KEEP: {keep}, DROP: {drop}")
            recommendations.append("")
        
        if len(severe_pairs) > 10:
            recommendations.append(f"   ... and {len(severe_pairs) - 10} more pairs to resolve")
    else:
        recommendations.append("4. MULTICOLLINEARITY - No severe redundancies detected ✓")
    
    recommendations.append("")
    recommendations.append("="*80)
    recommendations.append("SUMMARY")
    recommendations.append("="*80)
    recommendations.append(f"Total features analyzed: {len(target_corr_df)}")
    recommendations.append(f"Recommended to KEEP: {len(strong_features)}")
    recommendations.append(f"Recommended to CONSIDER: {len(moderate_features)}")
    recommendations.append(f"Recommended to DROP: {len(weak_features)}")
    recommendations.append("")
    
    # Print to console
    for line in recommendations:
        print(line)
    
    # Save to file
    output_path = os.path.join(output_dir, 'feature_recommendations.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(recommendations))
    
    print(f"\n✓ Recommendations saved to: {output_path}")

def save_correlation_tables(target_corr_df, intercorr_df, corr_matrix, output_dir):
    """
    Save correlation data tables for reference.
    """
    print("\n" + "="*80)
    print("SAVING CORRELATION TABLES")
    print("="*80)
    
    # Save feature-target correlations
    target_path = os.path.join(output_dir, 'feature_target_correlations.csv')
    target_corr_df.to_csv(target_path, index=False)
    print(f"\n✓ Feature-target correlations saved to: {target_path}")
    
    # Save feature intercorrelations (high pairs only)
    if len(intercorr_df) > 0:
        intercorr_path = os.path.join(output_dir, 'feature_intercorrelations.csv')
        intercorr_df.to_csv(intercorr_path, index=False)
        print(f"✓ Feature intercorrelations saved to: {intercorr_path}")
    
    # Save full correlation matrix
    matrix_path = os.path.join(output_dir, 'full_correlation_matrix.csv')
    corr_matrix.to_csv(matrix_path)
    print(f"✓ Full correlation matrix saved to: {matrix_path}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("H2H 02_feature_correlation_analysis.py")
    print("="*80)
    print("\nAnalyzing feature correlations and predictive power...")
    
    # Create output directory
    create_output_directory(OUTPUT_DIR)
    
    # Load training data
    df, feature_cols = load_training_data()
    
    # Analyze feature-target correlations
    target_corr_df = analyze_target_correlations(df, feature_cols)
    
    # Analyze feature intercorrelations
    corr_matrix, intercorr_df = analyze_feature_intercorrelations(df, feature_cols)
    
    # Create visualization
    create_correlation_heatmap(corr_matrix, OUTPUT_DIR)
    
    # Generate recommendations
    generate_recommendations(target_corr_df, intercorr_df, OUTPUT_DIR)
    
    # Save data tables
    save_correlation_tables(target_corr_df, intercorr_df, corr_matrix, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✓ FEATURE CORRELATION ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()