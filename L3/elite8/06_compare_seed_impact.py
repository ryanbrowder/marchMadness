"""
L3 Seed Impact Comparison
Compares models WITH seeds vs WITHOUT seeds to identify value opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configuration
# Configuration - Build paths without config suffix
BASE_DIR = config.L3_DIR / 'elite8' / 'outputs'

PREDICTIONS_WITH_SEEDS = BASE_DIR / '04_2026_predictions' / 'elite8_predictions_2026_comparison.csv'
PREDICTIONS_NO_SEEDS = BASE_DIR / '04_2026_predictions_no_seeds' / 'elite8_predictions_2026_comparison.csv'
BACKTEST_WITH_SEEDS = BASE_DIR / '03_backtest' / 'backtest_summary.csv'
BACKTEST_NO_SEEDS = BASE_DIR / '03_backtest_no_seeds' / 'backtest_summary.csv'

OUTPUT_DIR = BASE_DIR / '06_seed_impact'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SEED IMPACT ANALYSIS - WITH SEEDS vs WITHOUT SEEDS")
print("="*80)

# ============================================================================
# [1] LOAD PREDICTIONS
# ============================================================================
print("\n[1] LOADING PREDICTIONS")
print("-" * 80)

preds_with = pd.read_csv(PREDICTIONS_WITH_SEEDS)
preds_no = pd.read_csv(PREDICTIONS_NO_SEEDS)

print(f"Loaded WITH seeds predictions: {len(preds_with)} teams")
print(f"Loaded WITHOUT seeds predictions: {len(preds_no)} teams")

# Merge on team name
comparison = preds_with[['Team', 'Avg_Probability']].merge(
    preds_no[['Team', 'Avg_Probability']],
    on='Team',
    suffixes=('_with_seeds', '_no_seeds')
)

# Calculate seed effect
comparison['seed_effect'] = comparison['Avg_Probability_with_seeds'] - comparison['Avg_Probability_no_seeds']
comparison['abs_seed_effect'] = comparison['seed_effect'].abs()

# Sort by probability with seeds
comparison = comparison.sort_values('Avg_Probability_with_seeds', ascending=False).reset_index(drop=True)
comparison['rank'] = comparison.index + 1

print(f"\nMerged comparison: {len(comparison)} teams")

# ============================================================================
# [2] SUMMARY STATISTICS
# ============================================================================
print("\n[2] SUMMARY STATISTICS")
print("-" * 80)

print(f"\nSeed Effect Distribution:")
print(f"  Mean: {comparison['seed_effect'].mean():.3%}")
print(f"  Median: {comparison['seed_effect'].median():.3%}")
print(f"  Std Dev: {comparison['seed_effect'].std():.3%}")
print(f"  Min: {comparison['seed_effect'].min():.3%}")
print(f"  Max: {comparison['seed_effect'].max():.3%}")

print(f"\nAbsolute Seed Effect:")
print(f"  Mean: {comparison['abs_seed_effect'].mean():.3%}")
print(f"  Median: {comparison['abs_seed_effect'].median():.3%}")

# Correlation
correlation = comparison['Avg_Probability_with_seeds'].corr(comparison['Avg_Probability_no_seeds'])
print(f"\nCorrelation between models: {correlation:.4f}")

# ============================================================================
# [3] IDENTIFY VALUE PICKS AND FADE CANDIDATES
# ============================================================================
print("\n[3] VALUE PICKS AND FADE CANDIDATES")
print("-" * 80)

# Value picks: metrics love them, seeds don't (negative seed effect)
# This means WITHOUT seeds gives higher probability than WITH seeds
# Interpretation: Team is UNDERSEEDED relative to metrics
value_picks = comparison[comparison['seed_effect'] < -0.02].sort_values('seed_effect')

# Fade candidates: seeds love them, metrics don't (positive seed effect)
# This means WITH seeds gives higher probability than WITHOUT seeds
# Interpretation: Team is OVERSEEDED relative to metrics
fade_candidates = comparison[comparison['seed_effect'] > 0.02].sort_values('seed_effect', ascending=False)

# Consensus picks: both models agree (small seed effect)
consensus = comparison[comparison['abs_seed_effect'] < 0.02].sort_values('Avg_Probability_with_seeds', ascending=False)

print(f"\nVALUE PICKS (Underseeded - Metrics >> Seeds):")
print(f"  Found {len(value_picks)} teams")
if len(value_picks) > 0:
    print("\n  Top 10 Value Picks:")
    print("  Rank  Team                      With Seeds  No Seeds  Seed Effect")
    for _, row in value_picks.head(10).iterrows():
        print(f"  {row['rank']:>4}  {row['Team']:<25} {row['Avg_Probability_with_seeds']:>10.1%} {row['Avg_Probability_no_seeds']:>9.1%} {row['seed_effect']:>12.1%}")

print(f"\nFADE CANDIDATES (Overseeded - Seeds >> Metrics):")
print(f"  Found {len(fade_candidates)} teams")
if len(fade_candidates) > 0:
    print("\n  Top 10 Fade Candidates:")
    print("  Rank  Team                      With Seeds  No Seeds  Seed Effect")
    for _, row in fade_candidates.head(10).iterrows():
        print(f"  {row['rank']:>4}  {row['Team']:<25} {row['Avg_Probability_with_seeds']:>10.1%} {row['Avg_Probability_no_seeds']:>9.1%} {row['seed_effect']:>12.1%}")

print(f"\nCONSENSUS PICKS (Both Models Agree):")
print(f"  Found {len(consensus)} teams")
if len(consensus) > 0:
    print("\n  Top 10 Consensus Picks:")
    print("  Rank  Team                      With Seeds  No Seeds  Seed Effect")
    for _, row in consensus.head(10).iterrows():
        print(f"  {row['rank']:>4}  {row['Team']:<25} {row['Avg_Probability_with_seeds']:>10.1%} {row['Avg_Probability_no_seeds']:>9.1%} {row['seed_effect']:>12.1%}")

# ============================================================================
# [4] HISTORICAL BACKTEST COMPARISON
# ============================================================================
print("\n[4] HISTORICAL BACKTEST COMPARISON")
print("-" * 80)

backtest_with = pd.read_csv(BACKTEST_WITH_SEEDS)
backtest_no = pd.read_csv(BACKTEST_NO_SEEDS)

# Compare long model performance
long_with = backtest_with[backtest_with['Dataset'] == 'long']
long_no = backtest_no[backtest_no['Dataset'] == 'long']

print("\nLONG Model Historical Performance:")
print(f"  WITH seeds:    {long_with['ROC-AUC'].mean():.3f} ROC-AUC, {long_with['Correct Picks'].sum()}/{long_with['Correct Picks'].count()*8} correct ({long_with['Correct Picks'].sum()/(long_with['Correct Picks'].count()*8):.1%})")
print(f"  WITHOUT seeds: {long_no['ROC-AUC'].mean():.3f} ROC-AUC, {long_no['Correct Picks'].sum()}/{long_no['Correct Picks'].count()*8} correct ({long_no['Correct Picks'].sum()/(long_no['Correct Picks'].count()*8):.1%})")
print(f"  Difference:    {long_with['ROC-AUC'].mean() - long_no['ROC-AUC'].mean():.3f} ROC-AUC")

# Compare rich model performance
rich_with = backtest_with[backtest_with['Dataset'] == 'rich']
rich_no = backtest_no[backtest_no['Dataset'] == 'rich']

if len(rich_with) > 0 and len(rich_no) > 0:
    print("\nRICH Model Historical Performance:")
    print(f"  WITH seeds:    {rich_with['ROC-AUC'].mean():.3f} ROC-AUC, {rich_with['Correct Picks'].sum()}/{rich_with['Correct Picks'].count()*8} correct ({rich_with['Correct Picks'].sum()/(rich_with['Correct Picks'].count()*8):.1%})")
    print(f"  WITHOUT seeds: {rich_no['ROC-AUC'].mean():.3f} ROC-AUC, {rich_no['Correct Picks'].sum()}/{rich_no['Correct Picks'].count()*8} correct ({rich_no['Correct Picks'].sum()/(rich_no['Correct Picks'].count()*8):.1%})")
    print(f"  Difference:    {rich_with['ROC-AUC'].mean() - rich_no['ROC-AUC'].mean():.3f} ROC-AUC")

print("\nHISTORICAL INSIGHT:")
print("  Seeds provide minimal historical predictive value (~0.005 ROC-AUC)")
print("  BUT: ESPN bracketology for 2026 may capture late-breaking info")
print("  Value picks = metrics disagree with consensus seeding")

# ============================================================================
# [5] TOP 25 COMPARISON TABLE
# ============================================================================
print("\n[5] TOP 25 TEAMS - SEED IMPACT")
print("="*80)

top_25 = comparison.head(25)
print("\n Rank  Team                      With Seeds  No Seeds  Seed Effect  Interpretation")
print("-" * 100)
for _, row in top_25.iterrows():
    if row['seed_effect'] > 0.03:
        interpretation = "OVERSEEDED"
    elif row['seed_effect'] < -0.03:
        interpretation = "UNDERSEEDED"
    else:
        interpretation = "CONSENSUS"
    
    print(f" {row['rank']:>4}  {row['Team']:<25} {row['Avg_Probability_with_seeds']:>10.1%} {row['Avg_Probability_no_seeds']:>9.1%} {row['seed_effect']:>12.1%}  {interpretation}")

# ============================================================================
# [6] STRATEGIC RECOMMENDATIONS
# ============================================================================
print("\n[6] STRATEGIC RECOMMENDATIONS")
print("="*80)

# Get high probability teams
high_prob_consensus = consensus[consensus['Avg_Probability_with_seeds'] >= 0.45].head(5)
high_prob_value = value_picks[value_picks['Avg_Probability_no_seeds'] >= 0.35].head(3)
high_prob_fade = fade_candidates[fade_candidates['Avg_Probability_with_seeds'] >= 0.45].head(3)

print("\n📊 CONSERVATIVE POOL STRATEGY:")
print("  • Trust CONSENSUS picks (both models agree)")
print("  Recommended teams:")
for _, row in high_prob_consensus.iterrows():
    print(f"    - {row['Team']}: {row['Avg_Probability_with_seeds']:.1%} (consensus)")

print("\n🎯 DIFFERENTIATION STRATEGY (Large Pools):")
print("  • Pick VALUE PICKS to differentiate from ESPN bracket consensus")
print("  Recommended value plays:")
if len(high_prob_value) > 0:
    for _, row in high_prob_value.iterrows():
        print(f"    - {row['Team']}: {row['Avg_Probability_no_seeds']:.1%} metrics (vs {row['Avg_Probability_with_seeds']:.1%} with seeds)")
else:
    print("    - No strong value picks found (minimal underseeding)")

print("\n💰 CALCUTTA AUCTION STRATEGY:")
print("  • TARGET: Value picks (underseeded teams)")
print("  • AVOID: Fade candidates (overseeded teams)")
print("  Value picks for auction:")
if len(value_picks) > 0:
    for _, row in value_picks.head(5).iterrows():
        print(f"    - {row['Team']}: Metrics say {row['Avg_Probability_no_seeds']:.1%}, market may underprice")
else:
    print("    - No clear value picks (seeds align with metrics)")

print("\n  Teams to fade in auction:")
if len(fade_candidates) > 0:
    for _, row in fade_candidates.head(5).iterrows():
        print(f"    - {row['Team']}: Seeds say {row['Avg_Probability_with_seeds']:.1%}, metrics say {row['Avg_Probability_no_seeds']:.1%}")
else:
    print("    - No clear fade candidates (seeds align with metrics)")

# ============================================================================
# [7] VISUALIZATIONS
# ============================================================================
print("\n[7] CREATING VISUALIZATIONS")
print("-" * 80)

# Scatter plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter plot with 45-degree line
ax1 = axes[0, 0]
ax1.scatter(comparison['Avg_Probability_no_seeds'], comparison['Avg_Probability_with_seeds'], alpha=0.5)
ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
ax1.set_xlabel('Probability WITHOUT Seeds (Pure Metrics)')
ax1.set_ylabel('Probability WITH Seeds (Bracket-Aware)')
ax1.set_title('Model Agreement: With Seeds vs Without Seeds')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotate top teams
for _, row in comparison.head(10).iterrows():
    ax1.annotate(row['Team'], (row['Avg_Probability_no_seeds'], row['Avg_Probability_with_seeds']),
                fontsize=8, alpha=0.7)

# Plot 2: Seed effect distribution
ax2 = axes[0, 1]
ax2.hist(comparison['seed_effect'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='No Effect')
ax2.set_xlabel('Seed Effect (With Seeds - Without Seeds)')
ax2.set_ylabel('Number of Teams')
ax2.set_title('Distribution of Seed Effect')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Top 20 comparison
ax3 = axes[1, 0]
top_20 = comparison.head(20)
x = np.arange(len(top_20))
width = 0.35
ax3.barh(x - width/2, top_20['Avg_Probability_no_seeds'], width, label='Without Seeds', alpha=0.8)
ax3.barh(x + width/2, top_20['Avg_Probability_with_seeds'], width, label='With Seeds', alpha=0.8)
ax3.set_yticks(x)
ax3.set_yticklabels(top_20['Team'], fontsize=8)
ax3.set_xlabel('Elite 8 Probability')
ax3.set_title('Top 20 Teams: With vs Without Seeds')
ax3.legend()
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Seed effect by rank
ax4 = axes[1, 1]
ax4.scatter(comparison['rank'], comparison['seed_effect'], alpha=0.5)
ax4.axhline(0, color='red', linestyle='--', label='No Effect')
ax4.axhline(0.03, color='orange', linestyle=':', label='Overseeded threshold')
ax4.axhline(-0.03, color='green', linestyle=':', label='Underseeded threshold')
ax4.set_xlabel('Team Rank (by WITH seeds probability)')
ax4.set_ylabel('Seed Effect')
ax4.set_title('Seed Effect vs Team Rank')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'seed_impact_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization: seed_impact_analysis.png")

# ============================================================================
# [8] SAVE RESULTS
# ============================================================================
print("\n[8] SAVING RESULTS")
print("-" * 80)

# Save full comparison
comparison.to_csv(OUTPUT_DIR / 'seed_impact_full_comparison.csv', index=False)
print(f"Saved: seed_impact_full_comparison.csv")

# Save value picks
if len(value_picks) > 0:
    value_picks.to_csv(OUTPUT_DIR / 'value_picks_underseeded.csv', index=False)
    print(f"Saved: value_picks_underseeded.csv ({len(value_picks)} teams)")

# Save fade candidates
if len(fade_candidates) > 0:
    fade_candidates.to_csv(OUTPUT_DIR / 'fade_candidates_overseeded.csv', index=False)
    print(f"Saved: fade_candidates_overseeded.csv ({len(fade_candidates)} teams)")

# Save consensus
consensus.to_csv(OUTPUT_DIR / 'consensus_picks.csv', index=False)
print(f"Saved: consensus_picks.csv ({len(consensus)} teams)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SEED IMPACT ANALYSIS COMPLETE")
print("="*80)

print(f"\nKEY FINDINGS:")
print(f"  • Seeds add {comparison['seed_effect'].mean():.1%} average probability boost")
print(f"  • Model correlation: {correlation:.3f} (very high agreement)")
print(f"  • Historical ROC-AUC difference: ~0.005 (minimal)")
print(f"  • Value picks (underseeded): {len(value_picks)} teams")
print(f"  • Fade candidates (overseeded): {len(fade_candidates)} teams")
print(f"  • Consensus picks: {len(consensus)} teams")

print(f"\nINTERPRETATION:")
print(f"  • Seeds provide MINIMAL historical value")
print(f"  • But ESPN bracketology may capture late-breaking info for 2026")
print(f"  • Use value picks for differentiation in pools/Calcutta")
print(f"  • Use consensus picks for conservative strategy")

print(f"\nOUTPUTS:")
print(f"  {OUTPUT_DIR}/seed_impact_full_comparison.csv")
print(f"  {OUTPUT_DIR}/value_picks_underseeded.csv")
print(f"  {OUTPUT_DIR}/fade_candidates_overseeded.csv")
print(f"  {OUTPUT_DIR}/consensus_picks.csv")
print(f"  {OUTPUT_DIR}/seed_impact_analysis.png")

print("\n" + "="*80)