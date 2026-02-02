"""
Debug: Inspect Elite 8 pickle structure
Run: python debug_elite8_structure.py
"""
import pickle

with open('../L3/elite8/outputs/03_ensemble_models/trained_ensemble_long_production.pkl', 'rb') as f:
    obj = pickle.load(f)

print("=" * 60)
print("TOP-LEVEL KEYS")
print("=" * 60)
for k, v in obj.items():
    print(f"  '{k}' -> {type(v).__name__}: ", end="")
    if isinstance(v, list):
        print(f"[{len(v)} items] {v if len(v) < 10 else v[:5]}")
    elif isinstance(v, dict):
        print(f"{{{len(v)} keys}}")
    else:
        print(v)

print()
print("=" * 60)
print("ENSEMBLE_STRATEGIES (nested)")
print("=" * 60)
es = obj['ensemble_strategies']
if isinstance(es, dict):
    for k, v in es.items():
        print(f"  '{k}' -> {type(v).__name__}: ", end="")
        if isinstance(v, dict):
            print(f"{{{len(v)} keys}}")
            for k2, v2 in v.items():
                print(f"    '{k2}' -> {type(v2).__name__}: ", end="")
                if isinstance(v2, list):
                    print(f"[{len(v2)} items]")
                    for i, item in enumerate(v2):
                        print(f"      [{i}] {type(item).__name__}")
                elif isinstance(v2, dict):
                    print(f"{{{len(v2)} keys}}")
                else:
                    print(v2)
        elif isinstance(v, list):
            print(f"[{len(v)} items]")
            for i, item in enumerate(v):
                print(f"    [{i}] {type(item).__name__}")
        else:
            print(v)

print()
print("=" * 60)
print("MODELS key (the one that returned strings)")
print("=" * 60)
print(f"  Type: {type(obj['models'])}")
print(f"  Value: {obj['models']}")

print()
print("=" * 60)
print("BEST_STRATEGY")
print("=" * 60)
print(f"  {obj['best_strategy']}")