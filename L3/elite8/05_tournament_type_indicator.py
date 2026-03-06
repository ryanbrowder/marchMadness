"""
L3 Tournament Type Indicator
Configure via config.py: USE_SEEDS = True/False

UNIFIED DATASET (March 2026):
- Single model predictions (no long/rich comparison)
- Analyzes prediction characteristics to forecast tournament type
- Outputs: Chalk score, tournament type probability, strategy recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configuration
PREDICTIONS_FILE = config.OUTPUT_04 / 'elite8_predictions_2026.csv'
BACKTEST_FILE = config.OUTPUT_03 / 'backtest_summary.csv'
OUTPUT_DIR = config.OUTPUT_05
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("2026 TOURNAMENT TYPE INDICATOR")
config.print_config()
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] LOADING PREDICTIONS")
print("-" * 80)

preds = pd.read_csv(PREDICTIONS_FILE)
print(f"Loaded {len(preds)} team predictions")

# Load backtest results for context
backtest = pd.read_csv(BACKTEST_FILE)
print(f"Loaded {len(backtest)} years of historical performance")

# ============================================================================
# ANALYZE PREDICTION DISTRIBUTION
# ============================================================================
print("\n[2] ANALYZING PREDICTION CHARACTERISTICS")
print("-" * 80)

chalk_score = 0

# Metric 1: Top team probability
top_prob = preds['P_E8'].iloc[0]
top_team = preds['Team'].iloc[0]
print(f"\n[1] TOP TEAM: {top_team} at {top_prob:.1%}")
if top_prob >= 0.65:
    print("  → CHALK signal (dominant favorite)")
    chalk_score += 3
    top_signal = "CHALK"
elif top_prob >= 0.55:
    print("  → NEUTRAL (no clear signal)")
    top_signal = "NEUTRAL"
else:
    print("  → CHAOS signal (no dominant team)")
    chalk_score -= 3
    top_signal = "CHAOS"

# Metric 2: High-confidence picks
high_conf = len(preds[preds['P_E8'] >= 0.50])
print(f"\n[2] HIGH-CONFIDENCE PICKS (≥50%): {high_conf}")
if high_conf >= 6:
    print("  → CHALK signal (many strong teams)")
    chalk_score += 2
    conf_signal = "CHALK"
elif high_conf >= 4:
    print("  → NEUTRAL")
    conf_signal = "NEUTRAL"
else:
    print("  → CHAOS signal (parity)")
    chalk_score -= 2
    conf_signal = "CHAOS"

# Metric 3: Probability spread (top 10)
top_10_probs = preds['P_E8'].head(10).values
spread = top_10_probs[0] - top_10_probs[9]
print(f"\n[3] TOP 10 SPREAD: {spread:.1%}")
print(f"  Range: {top_10_probs[0]:.1%} (#{1}) to {top_10_probs[9]:.1%} (#{10})")
if spread >= 0.25:
    print("  → CHALK signal (clear hierarchy)")
    chalk_score += 2
    spread_signal = "CHALK"
elif spread >= 0.15:
    print("  → NEUTRAL")
    spread_signal = "NEUTRAL"
else:
    print("  → CHAOS signal (tight clustering)")
    chalk_score -= 2
    spread_signal = "CHAOS"

# Metric 4: Std dev of top 20
top_20_std = preds['P_E8'].head(20).std()
print(f"\n[4] TOP 20 STANDARD DEVIATION: {top_20_std:.3f}")
if top_20_std >= 0.12:
    print("  → CHALK signal (clear separation)")
    chalk_score += 1
    std_signal = "CHALK"
elif top_20_std >= 0.08:
    print("  → NEUTRAL")
    std_signal = "NEUTRAL"
else:
    print("  → CHAOS signal (parity)")
    chalk_score -= 1
    std_signal = "CHAOS"

# Metric 5: Parity in 30-50% range
parity_count = len(preds[(preds['P_E8'] >= 0.30) & (preds['P_E8'] <= 0.50)])
print(f"\n[5] TEAMS IN 30-50% RANGE: {parity_count}")
if parity_count >= 20:
    print("  → CHAOS signal (many viable teams)")
    chalk_score -= 2
    parity_signal = "CHAOS"
elif parity_count >= 12:
    print("  → NEUTRAL")
    parity_signal = "NEUTRAL"
else:
    print("  → CHALK signal (few middle teams)")
    chalk_score += 2
    parity_signal = "CHALK"

# Metric 6: Concentration at top
top_5_avg = preds['P_E8'].head(5).mean()
print(f"\n[6] TOP 5 AVERAGE PROBABILITY: {top_5_avg:.1%}")
if top_5_avg >= 0.55:
    print("  → CHALK signal (strong top tier)")
    chalk_score += 1
    top5_signal = "CHALK"
elif top_5_avg >= 0.45:
    print("  → NEUTRAL")
    top5_signal = "NEUTRAL"
else:
    print("  → CHAOS signal (weak top tier)")
    chalk_score -= 1
    top5_signal = "CHAOS"

# ============================================================================
# CALCULATE TOURNAMENT TYPE PROBABILITY
# ============================================================================
print("\n[3] TOURNAMENT TYPE ESTIMATION")
print("="*80)

print(f"\nCHALK SCORE: {chalk_score} (range: -13 to +11)")

# Convert score to probability distribution - FIXED EXPECTATIONS
if chalk_score >= 7:
    # EXTREME CHALK (like 2025)
    tournament_type = "EXTREME CHALK"
    chalk_prob = 0.80
    normal_prob = 0.15
    chaos_prob = 0.05
    expected_roc = "0.93-1.00"
    expected_acc = "75-100%"
    picks = "6-8"
    color = "🟢"
elif chalk_score >= 4:
    # CHALK (like 2019, 2021) - FIXED TO BE REALISTIC
    tournament_type = "CHALK"
    chalk_prob = 0.60
    normal_prob = 0.30
    chaos_prob = 0.10
    expected_roc = "0.88-0.95"  # Based on 2015/2019/2021 average
    expected_acc = "50-75%"      # Typical 4-5, excellent 6
    picks = "4-6"                # Realistic for 60% chalk
    color = "🟢"
elif chalk_score >= 0:
    # NORMAL (chalk lean)
    tournament_type = "NORMAL (chalk lean)"
    chalk_prob = 0.40
    normal_prob = 0.45
    chaos_prob = 0.15
    expected_roc = "0.85-0.92"
    expected_acc = "50-62%"
    picks = "4-5"
    color = "🔵"
elif chalk_score >= -4:
    # NORMAL (chaos lean)
    tournament_type = "NORMAL (chaos lean)"
    chalk_prob = 0.20
    normal_prob = 0.50
    chaos_prob = 0.30
    expected_roc = "0.82-0.90"
    expected_acc = "37-55%"
    picks = "3-5"
    color = "🟡"
else:
    # CHAOS (like 2018, 2022)
    tournament_type = "CHAOS"
    chalk_prob = 0.10
    normal_prob = 0.30
    chaos_prob = 0.60
    expected_roc = "0.75-0.85"
    expected_acc = "37-50%"
    picks = "3-4"
    color = "🔴"

print(f"\n{color} PREDICTED TOURNAMENT TYPE: {tournament_type}")
print(f"\nProbability Distribution:")
print(f"  Chalk:  {chalk_prob:.0%}")
print(f"  Normal: {normal_prob:.0%}")
print(f"  Chaos:  {chaos_prob:.0%}")

print(f"\nExpected Performance:")
print(f"  ROC-AUC: {expected_roc}")
print(f"  Elite 8 Accuracy: {expected_acc}")
print(f"  Correct Picks: {picks} out of 8")

# ============================================================================
# SIGNAL SUMMARY TABLE
# ============================================================================
print("\n[4] SIGNAL BREAKDOWN")
print("-" * 80)

signals_df = pd.DataFrame({
    'Metric': [
        'Top Team Probability',
        'High-Confidence Picks',
        'Top 10 Spread',
        'Top 20 Std Dev',
        'Parity (30-50%)',
        'Top 5 Average'
    ],
    'Value': [
        f"{top_prob:.1%}",
        f"{high_conf}",
        f"{spread:.1%}",
        f"{top_20_std:.3f}",
        f"{parity_count}",
        f"{top_5_avg:.1%}"
    ],
    'Signal': [
        top_signal,
        conf_signal,
        spread_signal,
        std_signal,
        parity_signal,
        top5_signal
    ]
})

print(signals_df.to_string(index=False))

# Count signals
chalk_signals = (signals_df['Signal'] == 'CHALK').sum()
neutral_signals = (signals_df['Signal'] == 'NEUTRAL').sum()
chaos_signals = (signals_df['Signal'] == 'CHAOS').sum()

print(f"\nSignal Summary: {chalk_signals} CHALK | {neutral_signals} NEUTRAL | {chaos_signals} CHAOS")

# ============================================================================
# HISTORICAL CONTEXT
# ============================================================================
print("\n[5] HISTORICAL CONTEXT")
print("-" * 80)

print("\nPast Tournament Types (by ROC-AUC):")
print("\nCHALK YEARS:")
chalk_years = backtest[backtest['ROC-AUC'] >= 0.93].sort_values('ROC-AUC', ascending=False)
for _, row in chalk_years.iterrows():
    print(f"  {int(row['Year'])}: ROC-AUC {row['ROC-AUC']:.3f} | {row['Correct Picks']}/8 correct")

print("\nNORMAL YEARS:")
normal_years = backtest[(backtest['ROC-AUC'] >= 0.85) & (backtest['ROC-AUC'] < 0.93)].sort_values('ROC-AUC', ascending=False)
for _, row in normal_years.iterrows():
    print(f"  {int(row['Year'])}: ROC-AUC {row['ROC-AUC']:.3f} | {row['Correct Picks']}/8 correct")

print("\nCHAOS YEARS:")
chaos_years = backtest[backtest['ROC-AUC'] < 0.85].sort_values('ROC-AUC', ascending=False)
for _, row in chaos_years.iterrows():
    print(f"  {int(row['Year'])}: ROC-AUC {row['ROC-AUC']:.3f} | {row['Correct Picks']}/8 correct")

# ============================================================================
# STRATEGY RECOMMENDATIONS
# ============================================================================
print("\n[6] STRATEGY RECOMMENDATIONS")
print("="*80)

if chalk_score >= 5:
    print("\n📊 CONSERVATIVE POOL STRATEGY (Friends, Small):")
    print("  • Trust top 5-6 model predictions heavily")
    print("  • Pick teams >50% for Elite 8")
    print("  • Expect 4-5 correct (6 would be excellent)")
    print("  • Expected finish: Top 15-25%")
    
    print("\n🎯 COMPETITIVE POOL STRATEGY (ESPN, Large):")
    print("  • Trust top 4 picks as locks")
    print("  • Pick 3-4 consensus favorites")
    print("  • Add 1-2 teams from 40-50% range for differentiation")
    print("  • Don't overchase upsets - chalk years reward patience")
    print("  • Expected finish: Top 20-30% (hard to differentiate)")
    
    print("\n💰 CALCUTTA AUCTION STRATEGY:")
    print("  • Top teams will be appropriately valued")
    print("  • Target 45-55% teams (value zone)")
    print("  • Avoid longshots (<30%)")
    print("  • Budget 60-70% on top-5 seeds")

elif chalk_score >= -3:
    print("\n📊 CONSERVATIVE POOL STRATEGY (Friends, Small):")
    print("  • Trust top 4-5 predictions")
    print("  • Mix chalk (top picks) with 1-2 variance plays (35-45%)")
    print("  • Expect 3-5 correct")
    print("  • Expected finish: Top 20-35%")
    
    print("\n🎯 COMPETITIVE POOL STRATEGY (ESPN, Large):")
    print("  • Pick top 3 as safe bets")
    print("  • Fade 1 consensus favorite (pick 10-15% to differentiate)")
    print("  • Add 2 teams from 35-50% range")
    print("  • Expected finish: Top 15-30%")
    
    print("\n💰 CALCUTTA AUCTION STRATEGY:")
    print("  • Target 40-50% teams (market inefficiency)")
    print("  • Avoid overbidding on >60% teams")
    print("  • Budget split: 50% top tier, 35% middle, 15% value")

else:
    print("\n📊 CONSERVATIVE POOL STRATEGY (Friends, Small):")
    print("  • Trust top 3 picks only")
    print("  • Add 2-3 variance plays from 35-50%")
    print("  • Accept that upsets will happen")
    print("  • Expect 3-4 correct")
    print("  • Expected finish: Top 30-50%")
    
    print("\n🎯 COMPETITIVE POOL STRATEGY (ESPN, Large):")
    print("  • Pick top 2 as anchors")
    print("  • Fade 2 favorites for differentiation")
    print("  • Add 3-4 teams from 30-45% range")
    print("  • This is your year to take risks")
    print("  • Expected finish: High variance (Top 10% or Bottom 50%)")
    
    print("\n💰 CALCUTTA AUCTION STRATEGY:")
    print("  • Target undervalued 35-50% teams aggressively")
    print("  • Fade expensive favorites (they'll disappoint)")
    print("  • Budget split: 30% top tier, 50% middle, 20% value")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[7] SAVING RESULTS")
print("-" * 80)

# Save indicator summary
summary = pd.DataFrame({
    'Metric': ['Chalk Score', 'Tournament Type', 'Chalk Probability', 'Normal Probability', 'Chaos Probability',
               'Expected ROC-AUC', 'Expected Accuracy', 'Expected Correct Picks'],
    'Value': [chalk_score, tournament_type, f"{chalk_prob:.1%}", f"{normal_prob:.1%}", f"{chaos_prob:.1%}",
              expected_roc, expected_acc, picks]
})

summary.to_csv(OUTPUT_DIR / 'tournament_type_summary.csv', index=False)
print(f"Saved summary: tournament_type_summary.csv")

# Save signal breakdown
signals_df.to_csv(OUTPUT_DIR / 'signal_breakdown.csv', index=False)
print(f"Saved signals: signal_breakdown.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TOURNAMENT TYPE INDICATOR COMPLETE")
print("="*80)

print(f"\n{color} 2026 FORECAST: {tournament_type}")
print(f"\nConfidence: {max(chalk_prob, normal_prob, chaos_prob):.0%}")
print(f"Expected Elite 8 picks: {picks}")

print("\nREMINDER:")
print("  • This is probabilistic, not deterministic")
print("  • Actual tournament type unknown until it happens")
print("  • Monitor first weekend to confirm/adjust")
print("  • By Sweet 16, you'll know for sure")

print("\nNEXT STEPS:")
print("  1. Wait for Selection Sunday (March 16)")
print("  2. Re-run 05_apply_to_2026.py with real seeds")
print("  3. Re-run this script to update forecast")
print("  4. Monitor first weekend results")
print("  5. Adjust strategy if needed by Elite 8 round")

print("\n" + "="*80)