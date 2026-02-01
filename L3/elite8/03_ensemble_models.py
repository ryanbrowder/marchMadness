"""
L3 Ensemble Models Pipeline
Combines individual models into ensembles with probability calibration.
ALIGNED with H2H modeling approach: Train through 2024, validate on 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_DIR = Path('outputs/01_feature_selection')
OUTPUT_DIR = Path('outputs/03_ensemble_models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ============================================================================
# MODE SELECTION - ALIGNED WITH H2H APPROACH
# ============================================================================
MODE = 'production'  # 'validation' or 'production'

if MODE == 'validation':
    # Validation: Train through 2024, test on 2025 (matches H2H)
    TRAIN_YEARS = (2008, 2024)
    TEST_YEAR = 2025
    OUTPUT_SUFFIX = '_validation'
    print("="*80)
    print("L3 ENSEMBLE MODELS PIPELINE - VALIDATION MODE")
    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} | Test: {TEST_YEAR}")
    print("Aligned with H2H modeling approach")
    print("="*80)
else:  # production
    # Production: Use all available data including 2025
    TRAIN_YEARS = (2008, 2025)
    TEST_YEAR = None
    OUTPUT_SUFFIX = '_production'
    PRODUCTION_BEST_STRATEGY = 'ROC-AUC Weighted'  # From validation
    print("="*80)
    print("L3 ENSEMBLE MODELS PIPELINE - PRODUCTION MODE")
    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}")
    print(f"Using strategy from validation: {PRODUCTION_BEST_STRATEGY}")
    print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 80)

labeled_long = pd.read_csv(INPUT_DIR / 'labeled_training_long.csv')
labeled_rich = pd.read_csv(INPUT_DIR / 'labeled_training_rich.csv')

features_long = pd.read_csv(INPUT_DIR / 'reduced_features_long.csv')['feature'].tolist()
features_rich = pd.read_csv(INPUT_DIR / 'reduced_features_rich.csv')['feature'].tolist()

print(f"Loaded data - Long: {labeled_long.shape[0]} rows, Rich: {labeled_rich.shape[0]} rows")

def prepare_data(df, feature_list):
    """Prepare features with train/test split (aligned with H2H)"""
    
    # Extract features and label
    X = df[feature_list].copy()
    y = df['elite8_flag'].copy()
    years = df['Year'].copy()
    teams = df['Team'].copy()
    
    # Drop entirely NaN columns
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    
    # Fill remaining NaNs
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col].fillna(0, inplace=True)
            else:
                X[col].fillna(median_val, inplace=True)
    
    # Train/test split based on mode
    if MODE == 'validation':
        train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
        test_mask = years == TEST_YEAR
    else:  # production
        train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
        test_mask = None
    
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    years_train = years[train_mask]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    result = {
        'X_train': X_train, 
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'years_train': years_train,
        'scaler': scaler
    }
    
    # Add test set in validation mode
    if MODE == 'validation':
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()
        years_test = years[test_mask]
        teams_test = teams[test_mask]
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        result.update({
            'X_test': X_test, 
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'years_test': years_test, 
            'teams_test': teams_test
        })
    
    return result

data_long = prepare_data(labeled_long, features_long)
data_rich = prepare_data(labeled_rich, features_rich)

print(f"\nLong dataset split:")
print(f"  Train: {len(data_long['y_train'])} samples ({data_long['years_train'].min()}-{data_long['years_train'].max()})")
print(f"  Train Elite 8+: {data_long['y_train'].sum()} ({data_long['y_train'].mean()*100:.1f}%)")
if MODE == 'validation':
    print(f"  Test: {len(data_long['y_test'])} samples (Year {TEST_YEAR})")
    print(f"  Test Elite 8+: {data_long['y_test'].sum()} ({data_long['y_test'].mean()*100:.1f}%)")

# ============================================================================
# TRAIN BASE MODELS
# ============================================================================
print("\n[2] TRAINING BASE MODELS")
print("-" * 80)

def train_base_models(data, dataset_name):
    """Train the 4 best models from exploratory phase"""
    
    models = {}
    predictions = {}
    
    # Logistic Regression (scaled)
    print(f"\n{dataset_name} - Training Logistic Regression...")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    lr.fit(data['X_train_scaled'], data['y_train'])
    models['Logistic Regression'] = lr
    
    preds = {'train': lr.predict_proba(data['X_train_scaled'])[:, 1]}
    if MODE == 'validation':
        preds['test'] = lr.predict_proba(data['X_test_scaled'])[:, 1]
    predictions['Logistic Regression'] = preds
    
    # Random Forest (unscaled)
    print(f"{dataset_name} - Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
    )
    rf.fit(data['X_train'], data['y_train'])
    models['Random Forest'] = rf
    
    preds = {'train': rf.predict_proba(data['X_train'])[:, 1]}
    if MODE == 'validation':
        preds['test'] = rf.predict_proba(data['X_test'])[:, 1]
    predictions['Random Forest'] = preds
    
    # SVM (scaled)
    print(f"{dataset_name} - Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
    svm.fit(data['X_train_scaled'], data['y_train'])
    models['SVM'] = svm
    
    preds = {'train': svm.predict_proba(data['X_train_scaled'])[:, 1]}
    if MODE == 'validation':
        preds['test'] = svm.predict_proba(data['X_test_scaled'])[:, 1]
    predictions['SVM'] = preds
    
    # Gaussian Naive Bayes (unscaled)
    print(f"{dataset_name} - Training Gaussian Naive Bayes...")
    gnb = GaussianNB()
    gnb.fit(data['X_train'], data['y_train'])
    models['Gaussian NB'] = gnb
    
    preds = {'train': gnb.predict_proba(data['X_train'])[:, 1]}
    if MODE == 'validation':
        preds['test'] = gnb.predict_proba(data['X_test'])[:, 1]
    predictions['Gaussian NB'] = preds
    
    return models, predictions

models_long, preds_long = train_base_models(data_long, "LONG")
models_rich, preds_rich = train_base_models(data_rich, "RICH")

# ============================================================================
# CALIBRATE GAUSSIAN NAIVE BAYES (Using Cross-Validation)
# ============================================================================
print("\n[3] CALIBRATING GAUSSIAN NAIVE BAYES")
print("-" * 80)

def calibrate_model_cv(model, X_train, y_train, X_test, dataset_name, model_name):
    """Apply Platt scaling using cross-validation on training data"""
    
    print(f"\n{dataset_name} - Calibrating {model_name} with CV...")
    
    # Use 5-fold stratified CV for calibration
    calibrated = CalibratedClassifierCV(
        model, 
        method='sigmoid', 
        cv=5  # 5-fold cross-validation
    )
    calibrated.fit(X_train, y_train)
    
    result = {}
    if MODE == 'validation':
        result['test'] = calibrated.predict_proba(X_test)[:, 1]
    
    return calibrated, result

gnb_long_cal, gnb_long_cal_preds = calibrate_model_cv(
    models_long['Gaussian NB'], 
    data_long['X_train'],
    data_long['y_train'],
    data_long['X_test'] if MODE == 'validation' else None,
    "LONG", 
    "Gaussian NB"
)

gnb_rich_cal, gnb_rich_cal_preds = calibrate_model_cv(
    models_rich['Gaussian NB'],
    data_rich['X_train'],
    data_rich['y_train'],
    data_rich['X_test'] if MODE == 'validation' else None,
    "RICH",
    "Gaussian NB"
)

# Add calibrated predictions
preds_long['Gaussian NB (Calibrated)'] = gnb_long_cal_preds
preds_rich['Gaussian NB (Calibrated)'] = gnb_rich_cal_preds

# ============================================================================
# VALIDATION MODE: EVALUATE AND SELECT BEST STRATEGY
# ============================================================================
if MODE == 'validation':
    print("\n[4] EVALUATING INDIVIDUAL MODELS ON TEST SET")
    print("-" * 80)
    
    def evaluate_predictions(y_true, y_pred_proba, model_name):
        """Calculate metrics for probability predictions"""
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        return {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba)
        }
    
    def evaluate_all_models(predictions, y_test, dataset_name):
        """Evaluate all model predictions on test set"""
        
        results = []
        for model_name, pred_dict in predictions.items():
            if 'test' in pred_dict:
                metrics = evaluate_predictions(y_test, pred_dict['test'], model_name)
                metrics['dataset'] = dataset_name
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    individual_results_long = evaluate_all_models(preds_long, data_long['y_test'], 'long')
    individual_results_rich = evaluate_all_models(preds_rich, data_rich['y_test'], 'rich')
    
    print(f"\nLONG Dataset - Individual Model Performance (Year {TEST_YEAR}):")
    print(individual_results_long[['model', 'roc_auc', 'log_loss', 'brier_score']].to_string(index=False))
    
    print(f"\nRICH Dataset - Individual Model Performance (Year {TEST_YEAR}):")
    print(individual_results_rich[['model', 'roc_auc', 'log_loss', 'brier_score']].to_string(index=False))
    
    # ========================================================================
    # ENSEMBLE STRATEGIES - VALIDATION MODE
    # ========================================================================
    print("\n[5] CREATING ENSEMBLES WITH DIFFERENT WEIGHTING STRATEGIES")
    print("-" * 80)
    
    def create_ensemble(predictions, weights, model_names):
        """Create weighted ensemble from model predictions"""
        
        pred_stack = np.column_stack([predictions[name] for name in model_names])
        ensemble_pred = np.average(pred_stack, axis=1, weights=weights)
        ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
        
        return ensemble_pred
    
    def test_ensemble_strategies(predictions, y_test, individual_results, dataset_name):
        """Test different ensemble weighting strategies"""
        
        model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'Gaussian NB (Calibrated)']
        
        strategies = {}
        
        # Strategy 1: Equal weighting
        print(f"\n{dataset_name} - Testing Equal Weight strategy...")
        strategies['Equal Weight'] = {
            'weights': np.array([0.25, 0.25, 0.25, 0.25]),
            'description': 'All models weighted equally'
        }
        
        # Strategy 2: ROC-AUC weighted
        print(f"{dataset_name} - Testing ROC-AUC Weighted strategy...")
        roc_aucs = []
        for name in model_names:
            roc = individual_results[individual_results['model'] == name]['roc_auc'].values[0]
            roc_aucs.append(roc)
        
        roc_aucs = np.array(roc_aucs)
        strategies['ROC-AUC Weighted'] = {
            'weights': roc_aucs / roc_aucs.sum(),
            'description': 'Weighted by ROC-AUC performance'
        }
        
        # Strategy 3: Inverse log-loss weighted
        print(f"{dataset_name} - Testing Inverse Log-Loss Weighted strategy...")
        log_losses = []
        for name in model_names:
            ll = individual_results[individual_results['model'] == name]['log_loss'].values[0]
            log_losses.append(ll)
        
        log_losses = np.array(log_losses)
        inv_log_loss = 1 / log_losses
        strategies['Inverse Log-Loss Weighted'] = {
            'weights': inv_log_loss / inv_log_loss.sum(),
            'description': 'Weighted by inverse of log loss'
        }
        
        # Evaluate each strategy on TEST set
        results = []
        for strategy_name, strategy_info in strategies.items():
            weights = strategy_info['weights']
            
            # Get test predictions
            predictions_test = {name: predictions[name]['test'] for name in model_names}
            
            # Create ensemble
            ensemble_pred = create_ensemble(predictions_test, weights, model_names)
            
            # Evaluate
            metrics = evaluate_predictions(y_test, ensemble_pred, strategy_name)
            metrics['dataset'] = dataset_name
            metrics['weights'] = str(dict(zip(model_names, weights)))
            metrics['description'] = strategy_info['description']
            results.append(metrics)
        
        return pd.DataFrame(results), strategies
    
    ensemble_results_long, strategies_long = test_ensemble_strategies(
        preds_long, data_long['y_test'], individual_results_long, 'long'
    )
    
    ensemble_results_rich, strategies_rich = test_ensemble_strategies(
        preds_rich, data_rich['y_test'], individual_results_rich, 'rich'
    )
    
    print(f"\nLONG Dataset - Ensemble Performance (Year {TEST_YEAR}):")
    print(ensemble_results_long[['model', 'roc_auc', 'log_loss', 'brier_score']].to_string(index=False))
    
    print(f"\nRICH Dataset - Ensemble Performance (Year {TEST_YEAR}):")
    print(ensemble_results_rich[['model', 'roc_auc', 'log_loss', 'brier_score']].to_string(index=False))

# ============================================================================
# PRODUCTION MODE: USE BEST STRATEGY FROM VALIDATION
# ============================================================================
else:  # production
    print("\n[4] PRODUCTION MODE - USING VALIDATED BEST STRATEGY")
    print("-" * 80)
    print(f"Strategy: {PRODUCTION_BEST_STRATEGY}")
    print("(No evaluation - no held-out test data)")
    
    def get_production_weights(strategy_name):
        """Get weights for production mode based on strategy name"""
        
        if strategy_name == 'Equal Weight':
            return np.array([0.25, 0.25, 0.25, 0.25])
        elif strategy_name == 'ROC-AUC Weighted':
            # Approximate weights from validation
            return np.array([0.24, 0.24, 0.23, 0.29])
        elif strategy_name == 'Inverse Log-Loss Weighted':
            return np.array([0.23, 0.25, 0.30, 0.22])
        else:
            return np.array([0.25, 0.25, 0.25, 0.25])
    
    strategies_long = {
        PRODUCTION_BEST_STRATEGY: {
            'weights': get_production_weights(PRODUCTION_BEST_STRATEGY),
            'description': 'Best strategy from validation mode'
        }
    }
    
    strategies_rich = {
        PRODUCTION_BEST_STRATEGY: {
            'weights': get_production_weights(PRODUCTION_BEST_STRATEGY),
            'description': 'Best strategy from validation mode'
        }
    }
    
    individual_results_long = None
    individual_results_rich = None
    ensemble_results_long = None
    ensemble_results_rich = None

# ============================================================================
# SAVE RESULTS (VALIDATION ONLY)
# ============================================================================
if MODE == 'validation':
    print("\n[6] SAVING RESULTS")
    print("-" * 80)
    
    all_individual = pd.concat([individual_results_long, individual_results_rich], ignore_index=True)
    all_ensemble = pd.concat([ensemble_results_long, ensemble_results_rich], ignore_index=True)
    
    all_individual.to_csv(OUTPUT_DIR / f'individual_model_performance{OUTPUT_SUFFIX}.csv', index=False)
    all_ensemble.to_csv(OUTPUT_DIR / f'ensemble_performance{OUTPUT_SUFFIX}.csv', index=False)
    
    print(f"Saved results with suffix: {OUTPUT_SUFFIX}")

# ============================================================================
# SAVE TRAINED MODELS WITH BEST WEIGHTS
# ============================================================================
section_num = 7 if MODE == 'validation' else 5
print(f"\n[{section_num}] SAVING TRAINED MODELS WITH OPTIMAL WEIGHTS")
print("-" * 80)

def save_model_package(models, scaler, features, calibrated_gnb, strategies, dataset_name):
    """Save models with best ensemble weights"""
    
    if MODE == 'validation':
        # Find best by ROC-AUC
        ensemble_results = ensemble_results_long if dataset_name == 'long' else ensemble_results_rich
        best_idx = ensemble_results['roc_auc'].idxmax()
        best_strategy_name = ensemble_results.loc[best_idx, 'model']
        best_roc_auc = ensemble_results.loc[best_idx, 'roc_auc']
        
        print(f"\n{dataset_name.upper()} - Best strategy: {best_strategy_name}")
        print(f"  ROC-AUC: {best_roc_auc:.4f}")
    else:
        best_strategy_name = PRODUCTION_BEST_STRATEGY
        print(f"\n{dataset_name.upper()} - Using strategy: {best_strategy_name}")
        print(f"  (Based on validation results)")
    
    best_weights = strategies[best_strategy_name]['weights']
    print(f"  Weights: {dict(zip(['LR', 'RF', 'SVM', 'GNB'], np.round(best_weights, 3)))}")
    
    model_package = {
        'models': models,
        'scaler': scaler,
        'features': features,
        'calibrated_gnb': calibrated_gnb,
        'ensemble_strategies': strategies,
        'best_strategy': best_strategy_name,
        'best_weights': best_weights,
        'model_names': ['Logistic Regression', 'Random Forest', 'SVM', 'Gaussian NB (Calibrated)'],
        'mode': MODE,
        'train_years': f'{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}'
    }
    
    output_file = OUTPUT_DIR / f'trained_ensemble_{dataset_name}{OUTPUT_SUFFIX}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"  Saved to: {output_file}")
    
    return best_strategy_name, best_weights

best_long_strategy, best_long_weights = save_model_package(
    models_long, data_long['scaler'], [col for col in data_long['X_train'].columns],
    gnb_long_cal, strategies_long, 'long'
)

best_rich_strategy, best_rich_weights = save_model_package(
    models_rich, data_rich['scaler'], [col for col in data_rich['X_train'].columns],
    gnb_rich_cal, strategies_rich, 'rich'
)

# Save predictions in validation mode
if MODE == 'validation':
    for dataset_name, data, predictions, best_weights, best_strategy in [
        ('long', data_long, preds_long, best_long_weights, best_long_strategy),
        ('rich', data_rich, preds_rich, best_rich_weights, best_rich_strategy)
    ]:
        model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'Gaussian NB (Calibrated)']
        predictions_test = {name: predictions[name]['test'] for name in model_names}
        
        def create_ensemble(predictions, weights, model_names):
            pred_stack = np.column_stack([predictions[name] for name in model_names])
            ensemble_pred = np.average(pred_stack, axis=1, weights=weights)
            return np.clip(ensemble_pred, 0.001, 0.999)
        
        best_ensemble_pred = create_ensemble(predictions_test, best_weights, model_names)
        
        pred_df = pd.DataFrame({
            'Team': data['teams_test'].values,
            'Year': data['years_test'].values,
            'actual': data['y_test'].values,
            'ensemble_probability': best_ensemble_pred,
            'ensemble_strategy': best_strategy
        })
        
        output_file = OUTPUT_DIR / f'best_ensemble_predictions_{dataset_name}{OUTPUT_SUFFIX}.csv'
        pred_df.to_csv(output_file, index=False)
        print(f"\nSaved {dataset_name} predictions: {output_file.name}")

# ============================================================================
# VISUALIZATIONS (VALIDATION ONLY)
# ============================================================================
if MODE == 'validation':
    section_num = 8
    print(f"\n[{section_num}] CREATING VISUALIZATIONS")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (dataset, individual, ensemble) in enumerate([
        ('Long', individual_results_long, ensemble_results_long),
        ('Rich', individual_results_rich, ensemble_results_rich)
    ]):
        ax = axes[idx]
        
        individual_copy = individual.copy()
        ensemble_copy = ensemble.copy()
        individual_copy['type'] = 'Individual'
        ensemble_copy['type'] = 'Ensemble'
        combined = pd.concat([individual_copy, ensemble_copy])
        
        x = range(len(combined))
        colors = ['skyblue' if t == 'Individual' else 'coral' for t in combined['type']]
        
        ax.barh(x, combined['roc_auc'], color=colors)
        ax.set_yticks(x)
        ax.set_yticklabels(combined['model'], fontsize=9)
        ax.set_xlabel('ROC-AUC')
        ax.set_title(f'{dataset} Dataset - {TEST_YEAR} Validation')
        ax.axvline(x=0.85, color='red', linestyle='--', alpha=0.5, label='0.85 threshold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'model_ensemble_comparison{OUTPUT_SUFFIX}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: model_ensemble_comparison{OUTPUT_SUFFIX}.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print(f"ENSEMBLE MODELS COMPLETE - {MODE.upper()} MODE")
print("="*80)

if MODE == 'validation':
    print(f"\nVALIDATION RESULTS ({TEST_YEAR} Tournament):")
    print("\nBEST ENSEMBLE BY DATASET:")
    for dataset_name, ensemble_results in [('LONG', ensemble_results_long), ('RICH', ensemble_results_rich)]:
        best_idx = ensemble_results['roc_auc'].idxmax()
        best = ensemble_results.loc[best_idx]
        print(f"\n{dataset_name}:")
        print(f"  Strategy: {best['model']}")
        print(f"  ROC-AUC: {best['roc_auc']:.3f}")
        print(f"  Log Loss: {best['log_loss']:.3f}")
        print(f"  Brier Score: {best['brier_score']:.3f}")
    
    print("\nNEXT STEP:")
    print(f"  → Switch MODE to 'production' and re-run for final 2026+ model")
else:
    print("\nPRODUCTION MODEL CREATED:")
    print(f"  ✓ Trained on {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} data")
    print(f"  ✓ Using {PRODUCTION_BEST_STRATEGY} strategy from validation")
    print(f"  ✓ Ready for 2026+ predictions")

print("\nALIGNMENT:")
print(f"  ✓ Consistent with H2H modeling approach")
print(f"  ✓ Train/test split matches across all models")

print("\nOUTPUTS GENERATED:")
if MODE == 'validation':
    print(f"  {OUTPUT_DIR}/individual_model_performance{OUTPUT_SUFFIX}.csv")
    print(f"  {OUTPUT_DIR}/ensemble_performance{OUTPUT_SUFFIX}.csv")
    print(f"  {OUTPUT_DIR}/best_ensemble_predictions_long{OUTPUT_SUFFIX}.csv")
    print(f"  {OUTPUT_DIR}/best_ensemble_predictions_rich{OUTPUT_SUFFIX}.csv")
    print(f"  {OUTPUT_DIR}/model_ensemble_comparison{OUTPUT_SUFFIX}.png")

print(f"  {OUTPUT_DIR}/trained_ensemble_long{OUTPUT_SUFFIX}.pkl")
print(f"  {OUTPUT_DIR}/trained_ensemble_rich{OUTPUT_SUFFIX}.pkl")

print("\n" + "="*80)