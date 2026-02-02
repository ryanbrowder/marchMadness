"""
Debug: Inspect L3 outputs before continuing L4 build.
Shows us what each model was trained on and what L3 produced.

Run: python debug_l3_outputs.py
"""
import json
import pickle
import os
from pathlib import Path

# ============================================================================
# 1. H2H ensemble_config.json - full contents
# ============================================================================
print("=" * 70)
print("H2H ENSEMBLE CONFIG (full contents)")
print("=" * 70)
config_path = Path("../L3/h2h/models/ensemble_config.json")
with open(config_path, 'r') as f:
    config = json.load(f)
print(json.dumps(config, indent=2))

# ============================================================================
# 2. H2H scaler - what did it fit on?
# ============================================================================
print()
print("=" * 70)
print("H2H SCALER (feature info)")
print("=" * 70)
scaler_path = Path("../L3/h2h/models/feature_scaler.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print(f"  Type: {type(scaler).__name__}")
print(f"  n_features_in_: {scaler.n_features_in_}")
if hasattr(scaler, 'feature_names_in_'):
    print(f"  feature_names_in_: {list(scaler.feature_names_in_)}")
else:
    print(f"  feature_names_in_: not stored (trained on array, not DataFrame)")
    print(f"  → We need to find the feature list elsewhere")

# ============================================================================
# 3. H2H individual model inspection - do any store feature names?
# ============================================================================
print()
print("=" * 70)
print("H2H MODELS - feature name check")
print("=" * 70)
h2h_files = {
    'GB': 'gradient_boosting.pkl',
    'RF': 'random_forest.pkl',
    'SVM': 'svm.pkl',
    'NN': 'neural_network.pkl',
    'GNB': 'gaussian_naive_bayes.pkl'
}
for name, filename in h2h_files.items():
    path = Path(f"../L3/h2h/models/{filename}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    has_features = hasattr(model, 'feature_names_in_')
    n_features = getattr(model, 'n_features_in_', 'unknown')
    print(f"  {name}: type={type(model).__name__}, n_features={n_features}, has feature_names={has_features}")
    if has_features:
        print(f"    feature_names_in_: {list(model.feature_names_in_)}")

# ============================================================================
# 4. List ALL files in L3 output directories
# ============================================================================
print()
print("=" * 70)
print("L3/elite8 DIRECTORY TREE")
print("=" * 70)
for root, dirs, files in os.walk("../L3/elite8"):
    level = root.replace("../L3/elite8", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"{subindent}{file} ({size:,} bytes)")

print()
print("=" * 70)
print("L3/h2h DIRECTORY TREE")
print("=" * 70)
for root, dirs, files in os.walk("../L3/h2h"):
    level = root.replace("../L3/h2h", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"{subindent}{file} ({size:,} bytes)")

# ============================================================================
# 5. predict_set_2026 - what columns exist, which are numeric vs string
# ============================================================================
print()
print("=" * 70)
print("predict_set_2026.csv COLUMN INVENTORY")
print("=" * 70)
import pandas as pd
df = pd.read_csv("../L3/data/predictionData/predict_set_2026.csv")
print(f"  Shape: {df.shape}")
print()
print(f"  {'Column':<35} {'dtype':<12} {'nulls':<8} {'sample'}")
print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*20}")
for col in df.columns:
    nulls = df[col].isna().sum()
    sample = str(df[col].iloc[0])[:20]
    print(f"  {col:<35} {str(df[col].dtype):<12} {nulls:<8} {sample}")