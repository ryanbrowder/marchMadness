"""
Read key L3 outputs we'll consume in L4.
Run: python debug_l3_consume.py
"""
import pandas as pd

print("=" * 70)
print("H2H features_used.txt")
print("=" * 70)
with open("../L3/h2h/outputs/03_train_models/features_used.txt", "r") as f:
    print(f.read())

print("=" * 70)
print("Elite 8 backtest_summary.csv")
print("=" * 70)
df = pd.read_csv("../L3/elite8/outputs/04_backtest/backtest_summary.csv")
print(df.to_string(index=False))

print()
print("=" * 70)
print("Elite 8 predictions_2026_long.csv (first 20 rows)")
print("=" * 70)
df = pd.read_csv("../L3/elite8/outputs/05_2026_predictions/elite8_predictions_2026_long.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head(20).to_string(index=False))

print()
print("=" * 70)
print("Elite 8 individual_model_performance_production.csv")
print("=" * 70)
df = pd.read_csv("../L3/elite8/outputs/03_ensemble_models/individual_model_performance_production.csv")
print(df.to_string(index=False))

print()
print("=" * 70)
print("H2H model_performance.csv")
print("=" * 70)
df = pd.read_csv("../L3/h2h/outputs/03_train_models/model_performance.csv")
print(df.to_string(index=False))