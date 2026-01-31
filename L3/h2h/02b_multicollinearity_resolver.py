"""
H2H 02b_resolve_multicollinearity.py

Purpose: Resolve multicollinearity by selecting which features to keep from redundant pairs.
         Uses target correlation as the tiebreaker - keep the feature with stronger 
         predictive power.

Inputs:
    - L3/h2h/outputs/02_feature_correlation/feature_target_correlations.csv
    - L3/h2h/outputs/02_feature_correlation/feature_intercorrelations.csv

Outputs:
    - L3/h2h/outputs/02_feature_correlation/selected_features.csv
    - Console summary of decisions

Author: Ryan Browder
Date: 2025-01-31
"""

import pandas as pd
import os

# ============================================================================
# Configuration
# ============================================================================

# Input paths
TARGET_CORR_PATH = 'outputs/02_feature_correlation/feature_target_correlations.csv'
INTERCORR_PATH = 'outputs/02_feature_correlation/feature_intercorrelations.csv'

# Output path
OUTPUT_PATH = 'outputs/02_feature_correlation/selected_features.csv'

# Multicollinearity threshold - pairs above this are considered redundant
REDUNDANCY_THRESHOLD = 0.85

# ============================================================================
# Main Logic
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("H2H 02b_resolve_multicollinearity.py")
    print("="*80)
    print("\nResolving multicollinearity by selecting features to keep...")
    
    # Load data
    print(f"\nLoading feature correlations...")
    target_corr = pd.read_csv(TARGET_CORR_PATH)
    
    # Check if intercorrelation file exists
    if not os.path.exists(INTERCORR_PATH):
        print(f"\n✓ No intercorrelation file found - no multicollinearity detected!")
        print(f"  All features are independent enough to use.")
        
        # All features pass
        selected = target_corr[['feature', 'correlation', 'abs_correlation']].copy()
        selected['status'] = 'KEEP'
        selected['reason'] = 'No multicollinearity'
        
        selected.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✓ Selected features saved to: {OUTPUT_PATH}")
        print(f"  Total features: {len(selected)}")
        return
    
    intercorr = pd.read_csv(INTERCORR_PATH)
    
    print(f"  Target correlations: {len(target_corr)} features")
    print(f"  Intercorrelations: {len(intercorr)} pairs")
    
    # Filter to severe multicollinearity only
    severe = intercorr[intercorr['abs_correlation'] >= REDUNDANCY_THRESHOLD]
    
    if len(severe) == 0:
        print(f"\n✓ No severe multicollinearity (|r| >= {REDUNDANCY_THRESHOLD}) detected!")
        print(f"  All features are acceptable to use.")
        
        selected = target_corr[['feature', 'correlation', 'abs_correlation']].copy()
        selected['status'] = 'KEEP'
        selected['reason'] = 'No severe multicollinearity'
        
        selected.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✓ Selected features saved to: {OUTPUT_PATH}")
        print(f"  Total features: {len(selected)}")
        return
    
    print(f"\n⚠ Found {len(severe)} pairs with severe redundancy (|r| >= {REDUNDANCY_THRESHOLD})")
    
    # Track features to drop
    features_to_drop = set()
    decisions = []
    
    print("\n" + "="*80)
    print("MULTICOLLINEARITY RESOLUTION DECISIONS")
    print("="*80)
    
    for idx, row in severe.iterrows():
        feat1 = row['feature_1']
        feat2 = row['feature_2']
        pair_corr = row['correlation']
        
        # Get target correlations
        feat1_target = target_corr[target_corr['feature'] == feat1]['abs_correlation'].values[0]
        feat2_target = target_corr[target_corr['feature'] == feat2]['abs_correlation'].values[0]
        
        # Decide which to keep
        if feat1_target > feat2_target:
            keep, drop = feat1, feat2
            keep_target, drop_target = feat1_target, feat2_target
        else:
            keep, drop = feat2, feat1
            keep_target, drop_target = feat2_target, feat1_target
        
        # Only drop if not already decided to keep (resolves transitive conflicts)
        if drop not in features_to_drop and keep not in features_to_drop:
            features_to_drop.add(drop)
            
            decisions.append({
                'feature_1': feat1,
                'feature_2': feat2,
                'intercorrelation': pair_corr,
                'keep': keep,
                'keep_target_corr': keep_target,
                'drop': drop,
                'drop_target_corr': drop_target
            })
            
            print(f"\nPair {idx+1}: {feat1} <-> {feat2}")
            print(f"  Intercorrelation: {pair_corr:.3f}")
            print(f"  {feat1} target corr: {feat1_target:.3f}")
            print(f"  {feat2} target corr: {feat2_target:.3f}")
            print(f"  DECISION: KEEP {keep}, DROP {drop}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTotal features analyzed: {len(target_corr)}")
    print(f"Features to drop due to redundancy: {len(features_to_drop)}")
    print(f"Features remaining: {len(target_corr) - len(features_to_drop)}")
    
    if features_to_drop:
        print(f"\nFeatures being dropped:")
        for feat in sorted(features_to_drop):
            print(f"  - {feat}")
    
    # Create selected features dataframe
    selected = target_corr.copy()
    selected['status'] = selected['feature'].apply(
        lambda x: 'DROP' if x in features_to_drop else 'KEEP'
    )
    selected['reason'] = selected['feature'].apply(
        lambda x: 'Redundant with stronger feature' if x in features_to_drop else 'Selected'
    )
    
    # Reorder: KEEP first, then DROP
    selected = selected.sort_values(['status', 'abs_correlation'], ascending=[True, False])
    
    # Save
    selected.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Selected features saved to: {OUTPUT_PATH}")
    
    # Print breakdown by status
    print(f"\nBreakdown:")
    print(f"  KEEP: {len(selected[selected['status'] == 'KEEP'])}")
    print(f"  DROP: {len(selected[selected['status'] == 'DROP'])}")
    
    # Save decisions
    if decisions:
        decisions_df = pd.DataFrame(decisions)
        decisions_path = 'outputs/02_feature_correlation/multicollinearity_decisions.csv'
        decisions_df.to_csv(decisions_path, index=False)
        print(f"\n✓ Decisions saved to: {decisions_path}")
    
    print("\n" + "="*80)
    print("✓ MULTICOLLINEARITY RESOLUTION COMPLETE")
    print("="*80)
    print("\nNext step: Use selected_features.csv (status='KEEP') for model training")

if __name__ == "__main__":
    main()