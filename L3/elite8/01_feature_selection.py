"""
L3 Feature Selection Pipeline
Configure via config.py: USE_SEEDS = True/False
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Configuration
INPUT_DIR = config.RESULTS_DIR
OUTPUT_DIR = config.OUTPUT_01
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRELATION_THRESHOLD = config.CORRELATION_THRESHOLD

print("="*80)
print("L3 FEATURE SELECTION PIPELINE")
config.print_config()

# ============================================================================
# [1] LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 80)

training_set_long = pd.read_csv(INPUT_DIR / 'training_set_long.csv')
training_set_rich = pd.read_csv(INPUT_DIR / 'training_set_rich.csv')
tournament_results = pd.read_csv(config.TOURNAMENT_RESULTS_FILE)

print(f"training_set_long: {training_set_long.shape[0]} rows, {training_set_long.shape[1]} columns")
print(f"training_set_rich: {training_set_rich.shape[0]} rows, {training_set_rich.shape[1]} columns")
print(f"tournamentResults: {tournament_results.shape[0]} rows")

print("\nDIAGNOSTIC - Tournament outcome distribution:")
print(tournament_results['tournamentOutcome'].value_counts())

elite8_outcomes = ['Elite Eight', 'Final Four', 'Finals', 'CHAMPS']
total_elite8 = tournament_results['tournamentOutcome'].isin(elite8_outcomes).sum()
print(f"\nElite 8+ teams with current definition: {total_elite8}")

years_in_data = tournament_results['Year'].nunique()
expected_elite8 = years_in_data * 8
print(f"Expected Elite 8+ teams (8 per year): ~{expected_elite8}")

# ============================================================================
# [2] JOIN DATA AND CREATE LABELS
# ============================================================================
print("\n[2] JOINING DATA AND CREATING LABELS")
print("-" * 80)

datasets = {
    'long': training_set_long,
    'rich': training_set_rich
}

labeled_datasets = {}
EXCLUDE_COLS = config.get_excluded_columns()

print(f"\nColumns excluded from features: {EXCLUDE_COLS}")

for name, df in datasets.items():
    # Join with tournament results
    df_labeled = df.merge(
        tournament_results[['Year', 'Index', 'tournamentOutcome']],
        on=['Year', 'Index'],
        how='left'
    )
    
    # Create Elite 8 flag
    df_labeled['elite8_flag'] = df_labeled['tournamentOutcome'].isin(elite8_outcomes).astype(int)
    
    print(f"\ntraining_set_{name}:")
    print(f"  After join: {len(df_labeled)} rows")
    print(f"  Elite 8+ teams: {df_labeled['elite8_flag'].sum()} ({df_labeled['elite8_flag'].mean():.1%})")
    print(f"  Non-Elite 8 teams: {(df_labeled['elite8_flag']==0).sum()} ({(df_labeled['elite8_flag']==0).mean():.1%})")
    
    # Convert to numeric
    print(f"  Converting columns to numeric types...")
    for col in df_labeled.columns:
        if col not in EXCLUDE_COLS:
            df_labeled[col] = pd.to_numeric(df_labeled[col], errors='coerce')
    
    numeric_cols = df_labeled.select_dtypes(include=[np.number]).columns
    print(f"  Numeric columns after conversion: {len(numeric_cols)}")
    
    labeled_datasets[name] = df_labeled
    
    # Save labeled dataset
    output_file = OUTPUT_DIR / f'labeled_training_{name}.csv'
    df_labeled.to_csv(output_file, index=False)

print(f"\nSaved labeled datasets to {OUTPUT_DIR}/")

# ============================================================================
# [3] FEATURE-TO-FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n[3] FEATURE-TO-FEATURE CORRELATION ANALYSIS")
print("-" * 80)

for name, df in labeled_datasets.items():
    print(f"\n{name}:")
    
    # Get numeric features (using config exclusion list)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in EXCLUDE_COLS]
    
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numeric features for correlation: {len(feature_cols)}")
    
    # Correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > CORRELATION_THRESHOLD and not np.isnan(corr_val):
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    print(f"  High correlation pairs (>{CORRELATION_THRESHOLD}): {len(high_corr_df)}")
    
    if len(high_corr_df) > 0:
        print(f"\n  Top 10 highly correlated pairs:")
        print(high_corr_df.head(10).to_string(index=False))
    
    # Save full list
    high_corr_df.to_csv(OUTPUT_DIR / f'high_correlations_{name}.csv', index=False)
    print(f"  Saved full list to {OUTPUT_DIR}/high_correlations_{name}.csv")
    print(f"  (Heatmap skipped - too many features for visualization)")

# ============================================================================
# [4] FEATURE-TO-LABEL CORRELATION
# ============================================================================
print("\n[4] FEATURE-TO-LABEL CORRELATION")
print("-" * 80)

for name, df in labeled_datasets.items():
    print(f"\n{name}:")
    
    # Get numeric features (using config exclusion list)
    numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in EXCLUDE_COLS]
    
    print(f"  Numeric features analyzed: {len(numeric_features)}")
    
    # Calculate correlations with elite8_flag
    correlations = []
    for feature in numeric_features:
        valid_mask = df[[feature, 'elite8_flag']].notna().all(axis=1)
        if valid_mask.sum() > 10:
            corr, _ = pearsonr(df.loc[valid_mask, feature], df.loc[valid_mask, 'elite8_flag'])
            correlations.append({
                'feature': feature,
                'correlation': abs(corr),
                'correlation_raw': corr
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    print(f"  Top 15 features by correlation with elite8_flag:")
    print(corr_df.head(15).to_string(index=False))
    
    # Save
    corr_df.to_csv(OUTPUT_DIR / f'label_correlations_{name}.csv', index=False)
    print(f"  Saved full ranking to {OUTPUT_DIR}/label_correlations_{name}.csv")

# ============================================================================
# [5] VARIANCE INFLATION FACTOR (VIF) ANALYSIS
# ============================================================================
print("\n[5] VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
print("-" * 80)

for name, df in labeled_datasets.items():
    print(f"\n{name}:")
    
    # Get numeric features (using config exclusion list)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in EXCLUDE_COLS]
    
    print(f"  Calculating VIF for {len(feature_cols)} numeric features...")
    
    # Calculate VIF
    vif_data = []
    for i, feature in enumerate(feature_cols):
        try:
            X = df[feature_cols].fillna(0)
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({
                'feature': feature,
                'VIF': vif
            })
        except Exception as e:
            print(f"  Warning: Could not calculate VIF for {feature}: {e}")
    
    if vif_data:
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        high_vif = vif_df[vif_df['VIF'] > 10]
        print(f"  Features with VIF > 10: {len(high_vif)}")
        
        vif_df.to_csv(OUTPUT_DIR / f'vif_analysis_{name}.csv', index=False)
        print(f"\n  Saved full VIF analysis to {OUTPUT_DIR}/vif_analysis_{name}.csv")

# ============================================================================
# [6] FEATURE SELECTION RECOMMENDATIONS
# ============================================================================
print("\n[6] FEATURE SELECTION RECOMMENDATIONS")
print("-" * 80)

for name, df in labeled_datasets.items():
    print(f"\n{name}:")
    
    # Get features (using config exclusion list)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in EXCLUDE_COLS]
    
    # Load high correlations
    high_corr_df = pd.read_csv(OUTPUT_DIR / f'high_correlations_{name}.csv')
    
    # Greedy feature selection
    features_to_drop = set()
    drop_reasons = []
    
    # Load label correlations for comparison
    label_corr = pd.read_csv(OUTPUT_DIR / f'label_correlations_{name}.csv')
    
    for _, row in high_corr_df.iterrows():
        feat1, feat2, corr = row['feature1'], row['feature2'], row['correlation']
        
        if feat1 in features_to_drop or feat2 in features_to_drop:
            continue
        
        corr1 = label_corr[label_corr['feature'] == feat1]['correlation'].values
        corr2 = label_corr[label_corr['feature'] == feat2]['correlation'].values
        
        if len(corr1) > 0 and len(corr2) > 0:
            if corr1[0] >= corr2[0]:
                features_to_drop.add(feat2)
                reason = f"Correlated with {feat1} (r={corr:.3f})"
            else:
                features_to_drop.add(feat1)
                reason = f"Correlated with {feat2} (r={corr:.3f})"
            
            drop_reasons.append({
                'feature': feat2 if corr1[0] >= corr2[0] else feat1,
                'reason': reason
            })
    
    # Consolidate reasons
    consolidated_reasons = {}
    for item in drop_reasons:
        feat = item['feature']
        if feat not in consolidated_reasons:
            consolidated_reasons[feat] = []
        consolidated_reasons[feat].append(item['reason'])
    
    drop_recommendations = []
    for feat, reasons in consolidated_reasons.items():
        drop_recommendations.append({
            'feature': feat,
            'reason': '; '.join(reasons)
        })
    
    drop_df = pd.DataFrame(drop_recommendations)
    
    print(f"  Recommended drops from correlation analysis: {len(drop_df)}")
    
    # Load VIF
    try:
        vif_df = pd.read_csv(OUTPUT_DIR / f'vif_analysis_{name}.csv')
        high_vif = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
        print(f"  High VIF features (review manually): {len(high_vif)}")
    except:
        high_vif = []
    
    print(f"\n  Features recommended for removal:")
    print(drop_df.to_string(index=False))
    
    drop_df.to_csv(OUTPUT_DIR / f'drop_recommendations_{name}.csv', index=False)
    
    # Create reduced feature list
    features_to_keep = [f for f in feature_cols if f not in features_to_drop]
    
    # CRITICAL: Force keep tournamentSeed if USE_SEEDS = True
    if config.USE_SEEDS and 'tournamentSeed' in feature_cols and 'tournamentSeed' not in features_to_keep:
        features_to_keep.append('tournamentSeed')
        print(f"\n  ⚠️  tournamentSeed was auto-dropped but manually restored (bracket-aware model)")
        print(f"  Reason: tournamentSeed is essential for bracket-conditional predictions")
    
    reduced_features_df = pd.DataFrame({'feature': features_to_keep})
    reduced_features_df.to_csv(OUTPUT_DIR / f'reduced_features_{name}.csv', index=False)
    
    print(f"\n  Original features: {len(feature_cols)}")
    print(f"  After recommended drops: {len(features_to_keep)}")
    print(f"  Reduction: {len(features_to_drop)} features ({len(features_to_drop)/len(feature_cols):.1%})")
    print(f"  Saved reduced feature list to {OUTPUT_DIR}/reduced_features_{name}.csv")

# ============================================================================
# FINAL SUMMARY
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

if not config.USE_SEEDS:
    print("\n⚠️  NOTE: tournamentSeed EXCLUDED from features (pure metrics model)")
    print("This creates a model based on team strength metrics only")
else:
    print("\n✓ tournamentSeed INCLUDED in features (bracket-aware model)")
    print("This model incorporates expected seeding information")
    print("tournamentSeed will be protected from automatic removal due to correlation")

print("\nNEXT STEPS:")
print("  1. Review correlation heatmaps and drop recommendations")
print("  2. Decide which features to keep for modeling")
print("  3. Proceed to 02_exploratory_models.py")

print("\n" + "="*80)