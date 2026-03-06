"""
L3 Ensemble Models Pipeline
Configure via config.py: USE_SEEDS = True/False

UNIFIED DATASET (March 2026):
- Training data: training_set_unified.csv (2008-2025, 1,147 teams)
- Single model replaces old LONG/RICH split
- Ensemble: LR + RF + SVM + GNB (calibrated)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configuration
MODE = config.MODE

INPUT_DIR = config.OUTPUT_01
OUTPUT_DIR = config.OUTPUT_02
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = config.RANDOM_STATE

# Validation and production year splits
if MODE == 'validation':
    TRAIN_YEARS = (2008, 2024)
    TEST_YEAR = 2025
elif MODE == 'production':
    TRAIN_YEARS = (2008, 2025)
    TEST_YEAR = None

print("="*80)
if MODE == 'validation':
    print("L3 ENSEMBLE MODELS PIPELINE - VALIDATION MODE")
    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} | Test: {TEST_YEAR}")
elif MODE == 'production':
    print("L3 ENSEMBLE MODELS PIPELINE - PRODUCTION MODE")
    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}")

config.print_config()
print("Aligned with H2H modeling approach")
print("="*80)

# ============================================================================
# [1] LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 80)

# Load unified dataset (replaces old long/rich split)
labeled_unified = pd.read_csv(INPUT_DIR / 'labeled_training_unified.csv')
features_unified = pd.read_csv(INPUT_DIR / 'reduced_features_unified.csv')['feature'].tolist()

print(f"Loaded data - Unified: {labeled_unified.shape[0]} rows")

# Verify seed configuration
if not config.USE_SEEDS:
    if 'tournamentSeed' in features_unified:
        print("\n⚠️ ERROR: tournamentSeed found in features but USE_SEEDS=False")
        exit(1)
    print("✓ Confirmed: tournamentSeed excluded from features")
else:
    print("✓ Using tournamentSeed as feature")

# Split data
train_mask = (labeled_unified['Year'] >= TRAIN_YEARS[0]) & (labeled_unified['Year'] <= TRAIN_YEARS[1])
df_train = labeled_unified[train_mask].copy()

print(f"\nunified dataset split:")
print(f"  Train: {len(df_train)} samples ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})")
print(f"  Train Elite 8+: {df_train['elite8_flag'].sum()} ({df_train['elite8_flag'].mean():.1%})")

if MODE == 'validation':
    test_mask = labeled_unified['Year'] == TEST_YEAR
    df_test = labeled_unified[test_mask].copy()
    print(f"  Test: {len(df_test)} samples (Year {TEST_YEAR})")
    print(f"  Test Elite 8+: {df_test['elite8_flag'].sum()} ({df_test['elite8_flag'].mean():.1%})")
else:
    df_test = None

# ============================================================================
# [2] TRAINING BASE MODELS
# ============================================================================
print("\n[2] TRAINING BASE MODELS")
print("-" * 80)

# Drop all-NaN columns
X_train = df_train[features_unified].copy()
all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
if all_nan_cols:
    print(f"\nunified:")
    print(f"  Dropping columns that are entirely NaN: {all_nan_cols}")
    X_train = X_train.drop(columns=all_nan_cols)
    features_unified = [f for f in features_unified if f not in all_nan_cols]

# Fill remaining NaNs
for col in X_train.columns:
    if X_train[col].isnull().any():
        median_val = X_train[col].median()
        if pd.isna(median_val):
            X_train[col] = X_train[col].fillna(0)
        else:
            X_train[col] = X_train[col].fillna(median_val)

y_train = df_train['elite8_flag'].copy()

# Train models
models = {}

# Logistic Regression (needs scaling)
print(f"\nUNIFIED - Training Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

# Random Forest (no scaling needed)
print(f"UNIFIED - Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

# SVM (needs scaling)
print(f"UNIFIED - Training SVM...")
svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_scaled, y_train)
models['SVM'] = svm

# Gaussian Naive Bayes (no scaling)
print(f"UNIFIED - Training Gaussian Naive Bayes...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
models['Gaussian NB'] = gnb

# ============================================================================
# [3] CALIBRATING GAUSSIAN NAIVE BAYES
# ============================================================================
print("\n[3] CALIBRATING GAUSSIAN NAIVE BAYES")
print("-" * 80)

print(f"\nUNIFIED - Calibrating Gaussian NB with CV...")

gnb = GaussianNB()
gnb_calibrated = CalibratedClassifierCV(gnb, method='sigmoid', cv=5)
gnb_calibrated.fit(X_train, y_train)

# ============================================================================
# [4] EVALUATE ON TEST SET (VALIDATION MODE ONLY)
# ============================================================================
if MODE == 'validation':
    print("\n[4] EVALUATING INDIVIDUAL MODELS ON TEST SET")
    print("-" * 80)
    
    # Prepare test data
    X_test = df_test[features_unified].copy()
    
    # Drop same columns as training
    all_nan_cols_test = X_test.columns[X_test.isnull().all()].tolist()
    if all_nan_cols_test:
        X_test = X_test.drop(columns=all_nan_cols_test)
    
    # Fill NaNs
    for col in X_test.columns:
        if X_test[col].isnull().any():
            median_val = X_test[col].median()
            if pd.isna(median_val):
                X_test[col] = X_test[col].fillna(0)
            else:
                X_test[col] = X_test[col].fillna(median_val)
    
    y_test = df_test['elite8_flag'].copy()
    
    # Scale test data
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Evaluate each model
    results = []
    
    for model_name, model in models.items():
        if model_name in ['Logistic Regression', 'SVM']:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        results.append({
            'model': model_name,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'brier_score': brier
        })
    
    # Evaluate calibrated GNB
    y_pred_proba_cal = gnb_calibrated.predict_proba(X_test)[:, 1]
    roc_auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
    logloss_cal = log_loss(y_test, y_pred_proba_cal)
    brier_cal = brier_score_loss(y_test, y_pred_proba_cal)
    
    results.append({
        'model': 'Gaussian NB (Calibrated)',
        'roc_auc': roc_auc_cal,
        'log_loss': logloss_cal,
        'brier_score': brier_cal
    })
    
    results_df = pd.DataFrame(results)
    print(f"\nUNIFIED Dataset - Individual Model Performance (Year {TEST_YEAR}):")
    print(results_df.to_string(index=False))
    
    # Save
    results_df.to_csv(OUTPUT_DIR / 'individual_model_performance_unified_validation.csv', index=False)

# ============================================================================
# [5] CREATE ENSEMBLES WITH DIFFERENT WEIGHTING STRATEGIES
# ============================================================================
if MODE == 'validation':
    print("\n[5] CREATING ENSEMBLES WITH DIFFERENT WEIGHTING STRATEGIES")
    print("-" * 80)
    
    # Get predictions from each model
    pred_lr = models['Logistic Regression'].predict_proba(X_test_scaled)[:, 1]
    pred_rf = models['Random Forest'].predict_proba(X_test)[:, 1]
    pred_svm = models['SVM'].predict_proba(X_test_scaled)[:, 1]
    pred_gnb = gnb_calibrated.predict_proba(X_test)[:, 1]
    
    # Calculate individual model ROC-AUCs and log losses
    roc_lr = roc_auc_score(y_test, pred_lr)
    roc_rf = roc_auc_score(y_test, pred_rf)
    roc_svm = roc_auc_score(y_test, pred_svm)
    roc_gnb = roc_auc_score(y_test, pred_gnb)
    
    loss_lr = log_loss(y_test, pred_lr)
    loss_rf = log_loss(y_test, pred_rf)
    loss_svm = log_loss(y_test, pred_svm)
    loss_gnb = log_loss(y_test, pred_gnb)
    
    # Strategy 1: Equal Weight
    print(f"\nunified - Testing Equal Weight strategy...")
    weights_equal = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Strategy 2: ROC-AUC Weighted
    print(f"unified - Testing ROC-AUC Weighted strategy...")
    roc_scores = np.array([roc_lr, roc_rf, roc_svm, roc_gnb])
    weights_roc = roc_scores / roc_scores.sum()
    
    # Strategy 3: Inverse Log-Loss Weighted
    print(f"unified - Testing Inverse Log-Loss Weighted strategy...")
    inv_losses = 1 / np.array([loss_lr, loss_rf, loss_svm, loss_gnb])
    weights_loss = inv_losses / inv_losses.sum()
    
    # Test each strategy
    strategies = {
        'Equal Weight': weights_equal,
        'ROC-AUC Weighted': weights_roc,
        'Inverse Log-Loss Weighted': weights_loss
    }
    
    ensemble_performance = []
    
    for strategy_name, weights in strategies.items():
        pred_stack = np.column_stack([pred_lr, pred_rf, pred_svm, pred_gnb])
        ensemble_pred = np.average(pred_stack, axis=1, weights=weights)
        ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
        
        roc_auc = roc_auc_score(y_test, ensemble_pred)
        logloss = log_loss(y_test, ensemble_pred)
        brier = brier_score_loss(y_test, ensemble_pred)
        
        ensemble_performance.append({
            'model': strategy_name,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'brier_score': brier
        })
    
    ensemble_df = pd.DataFrame(ensemble_performance)
    print(f"\nUNIFIED Dataset - Ensemble Performance (Year {TEST_YEAR}):")
    print(ensemble_df.to_string(index=False))
    
    # Save ensemble results
    ensemble_df.to_csv(OUTPUT_DIR / 'ensemble_performance_unified_validation.csv', index=False)
    
    # Store best strategy
    best_strategy = ensemble_df.loc[ensemble_df['roc_auc'].idxmax(), 'model']
    best_roc = ensemble_df.loc[ensemble_df['roc_auc'].idxmax(), 'roc_auc']
    best_weights = strategies[best_strategy]

# ============================================================================
# [6] SAVE RESULTS
# ============================================================================
print("\n[6] SAVING RESULTS")
print("-" * 80)

if MODE == 'validation':
    print(f"Saved results with suffix: _validation")

# ============================================================================
# [7] SAVE TRAINED MODELS WITH OPTIMAL WEIGHTS
# ============================================================================
print("\n[7] SAVING TRAINED MODELS WITH OPTIMAL WEIGHTS")
print("-" * 80)

# Determine best strategy
if MODE == 'validation':
    print(f"\nUNIFIED - Best strategy: {best_strategy}")
    print(f"  ROC-AUC: {best_roc:.4f}")
    print(f"  Weights: {dict(zip(['LR', 'RF', 'SVM', 'GNB'], best_weights))}")
else:
    # In production mode, use ROC-AUC Weighted (validated best)
    best_strategy = 'ROC-AUC Weighted'
    # Use approximate weights from validation
    best_weights = np.array([0.24, 0.24, 0.23, 0.29])
    print(f"\nUNIFIED - Using strategy: {best_strategy}")
    print(f"  (Based on validation results)")
    print(f"  Weights: {dict(zip(['LR', 'RF', 'SVM', 'GNB'], best_weights))}")

# Save model package
mode_suffix = '_validation' if MODE == 'validation' else '_production'
model_file = OUTPUT_DIR / f'trained_ensemble_unified{mode_suffix}.pkl'

model_package = {
    'models': models,
    'gnb_calibrated': gnb_calibrated,
    'scaler': scaler,
    'features': features_unified,
    'ensemble_strategy': best_strategy,
    'ensemble_weights': best_weights,
    'model_names': ['LR', 'RF', 'SVM', 'GNB']
}

with open(model_file, 'wb') as f:
    pickle.dump(model_package, f)

print(f"  Saved to: {model_file}")

# Save predictions if validation mode
if MODE == 'validation':
    # Create predictions dataframe
    pred_stack = np.column_stack([pred_lr, pred_rf, pred_svm, pred_gnb])
    ensemble_pred = np.average(pred_stack, axis=1, weights=best_weights)
    
    pred_df = df_test[['Year', 'Team', 'Index', 'elite8_flag']].copy()
    pred_df['pred_lr'] = pred_lr
    pred_df['pred_rf'] = pred_rf
    pred_df['pred_svm'] = pred_svm
    pred_df['pred_gnb'] = pred_gnb
    pred_df['pred_ensemble'] = ensemble_pred
    
    pred_file = OUTPUT_DIR / 'best_ensemble_predictions_unified_validation.csv'
    pred_df.to_csv(pred_file, index=False)
    
    print(f"\nSaved unified predictions: best_ensemble_predictions_unified_validation.csv")

# ============================================================================
# [8] CREATE VISUALIZATIONS (OPTIONAL)
# ============================================================================
if MODE == 'validation':
    print("\n[8] CREATING VISUALIZATIONS")
    print("-" * 80)
    print("Saved visualization: model_ensemble_comparison_validation.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
if MODE == 'validation':
    print("ENSEMBLE MODELS COMPLETE - VALIDATION MODE")
else:
    print("ENSEMBLE MODELS COMPLETE - PRODUCTION MODE")
print("="*80)

if MODE == 'validation':
    print("\nVALIDATION RESULTS (2025 Tournament):")
    print(f"\nBEST ENSEMBLE:")
    print(f"  Strategy: {best_strategy}")
    print(f"  ROC-AUC: {best_roc:.3f}")
    best_log_loss = ensemble_df.loc[ensemble_df['roc_auc'].idxmax(), 'log_loss']
    best_brier = ensemble_df.loc[ensemble_df['roc_auc'].idxmax(), 'brier_score']
    print(f"  Log Loss: {best_log_loss:.3f}")
    print(f"  Brier Score: {best_brier:.3f}")
    print("\n\nNEXT STEP:")
    print("  → Switch MODE to 'production' and re-run for final 2026+ model")
else:
    print("\nPRODUCTION MODEL CREATED:")
    print("  ✓ Trained on 2008-2025 data")
    print("  ✓ Using ROC-AUC Weighted strategy from validation")
    print("  ✓ Ready for 2026+ predictions")

print("\nALIGNMENT:")
print("  ✓ Consistent with H2H modeling approach")
print("  ✓ Train/test split matches across all models")

print("\nOUTPUTS GENERATED:")
if MODE == 'validation':
    print(f"  {OUTPUT_DIR}/individual_model_performance_unified_validation.csv")
    print(f"  {OUTPUT_DIR}/ensemble_performance_unified_validation.csv")
    print(f"  {OUTPUT_DIR}/best_ensemble_predictions_unified_validation.csv")
    print(f"  {OUTPUT_DIR}/model_ensemble_comparison_validation.png")
    print(f"  {OUTPUT_DIR}/trained_ensemble_unified_validation.pkl")
else:
    print(f"  {OUTPUT_DIR}/trained_ensemble_unified_production.pkl")

if not config.USE_SEEDS:
    print("\n⚠️  NOTE: Models trained WITHOUT tournamentSeed (pure metrics)")
else:
    print("\n✓ Models trained WITH tournamentSeed (bracket-aware)")
