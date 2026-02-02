"""
H2H 03_train_models.py

Purpose: Train multiple machine learning models on head-to-head matchup data using
         selected features. Use temporal validation (train on past, validate on 2025).
         Evaluate models and build optimized ensemble.

Inputs:
    - L3/h2h/outputs/01_build_training_matchups/training_matchups.csv
    - L3/h2h/outputs/02_feature_correlation/selected_features.csv

Outputs:
    - L3/h2h/models/ (trained model artifacts)
    - L3/h2h/outputs/03_train_models/ (performance reports)

Author: Ryan Browder
Date: 2025-01-31
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# MODE: 'validation' or 'production'
# - validation: Train 2008-2024, test 2025, find best ensemble strategy
# - production: Train 2008-2025 (all data), use winning strategy from validation
MODE = 'validation'  # Switch to 'production' after validation analysis

# Input paths
TRAINING_MATCHUPS_PATH = 'outputs/01_build_training_matchups/training_matchups.csv'
SELECTED_FEATURES_PATH = 'outputs/02_feature_correlation/selected_features.csv'

# Output directories (with mode suffix)
MODELS_DIR = f'models{("_" + MODE) if MODE == "validation" else ""}'
OUTPUT_DIR = f'outputs/03_train_models{("_" + MODE) if MODE == "validation" else ""}'

# Temporal validation split based on MODE
if MODE == 'validation':
    TRAIN_YEARS = (2008, 2024)
    VALIDATION_YEAR = 2025
    PRODUCTION_STRATEGY = None  # Will be determined from validation
    print("\n" + "="*80)
    print("H2H TRAINING - VALIDATION MODE")
    print("="*80)
    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} | Validate: {VALIDATION_YEAR}")
    print("Experimenting with ensemble strategies to find best approach")
    print("="*80)
else:  # production
    TRAIN_YEARS = (2008, 2025)
    VALIDATION_YEAR = None  # No validation set in production
    # Set this based on your validation results
    PRODUCTION_STRATEGY = 'Emphasize Top 2'  # Update after running validation
    print("\n" + "="*80)
    print("H2H TRAINING - PRODUCTION MODE")
    print("="*80)
    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} (all available data)")
    print(f"Using strategy from validation: {PRODUCTION_STRATEGY}")
    print("="*80)

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Helper Functions
# ============================================================================

def create_output_directories():
    """Create output directories if they don't exist."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories ready")

def load_data():
    """Load training data and selected features."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load training matchups
    print(f"\nLoading training matchups from: {TRAINING_MATCHUPS_PATH}")
    df = pd.read_csv(TRAINING_MATCHUPS_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")
    print(f"  Games: {len(df)}")
    
    # Load selected features
    print(f"\nLoading selected features from: {SELECTED_FEATURES_PATH}")
    selected = pd.read_csv(SELECTED_FEATURES_PATH)
    
    # Filter to KEEP features only (after multicollinearity resolution)
    keep_features = selected[selected['status'] == 'KEEP']['feature'].tolist()
    print(f"  Features marked KEEP: {len(keep_features)}")
    
    return df, keep_features

def prepare_train_validation_split(df, feature_cols):
    """
    Split data temporally based on MODE.
    - Validation mode: train on 2008-2024, validate on 2025
    - Production mode: train on all data (2008-2025), no validation set
    """
    print("\n" + "="*80)
    print("TEMPORAL TRAIN/VALIDATION SPLIT")
    print("="*80)
    
    if MODE == 'validation':
        # Split by year
        train_df = df[(df['Year'] >= TRAIN_YEARS[0]) & (df['Year'] <= TRAIN_YEARS[1])].copy()
        val_df = df[df['Year'] == VALIDATION_YEAR].copy()
        
        print(f"\nTraining set: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}")
        print(f"  Games: {len(train_df)}")
        print(f"  Years: {train_df['Year'].nunique()}")
        
        print(f"\nValidation set: {VALIDATION_YEAR}")
        print(f"  Games: {len(val_df)}")
        
        # Prepare X and y
        X_train = train_df[feature_cols].values
        y_train = train_df['TeamA_Won'].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df['TeamA_Won'].values
        
        print(f"\nFeature matrix shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        
        print(f"\nTarget distribution:")
        print(f"  Train - TeamA wins: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"  Val   - TeamA wins: {y_val.sum()} ({y_val.mean()*100:.1f}%)")
        
        return X_train, X_val, y_train, y_val, train_df, val_df
    
    else:  # production mode
        # Use all available data for training
        train_df = df[(df['Year'] >= TRAIN_YEARS[0]) & (df['Year'] <= TRAIN_YEARS[1])].copy()
        
        print(f"\nProduction training set: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}")
        print(f"  Games: {len(train_df)}")
        print(f"  Years: {train_df['Year'].nunique()}")
        print(f"\n⚠ No validation set - using all data for training")
        
        # Prepare X and y
        X_train = train_df[feature_cols].values
        y_train = train_df['TeamA_Won'].values
        
        print(f"\nFeature matrix shapes:")
        print(f"  X_train: {X_train.shape}")
        
        print(f"\nTarget distribution:")
        print(f"  Train - TeamA wins: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        
        # Return None for validation sets
        return X_train, None, y_train, None, train_df, None

def scale_features(X_train, X_val):
    """
    Scale features using StandardScaler fit on training data.
    In production mode, X_val will be None.
    """
    print("\n" + "="*80)
    print("FEATURE SCALING")
    print("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    print(f"\n✓ Features scaled using StandardScaler")
    print(f"  Mean: {scaler.mean_[:5]}... (first 5)")
    print(f"  Std:  {scaler.scale_[:5]}... (first 5)")
    
    return X_train_scaled, X_val_scaled, scaler

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("\n" + "="*80)
    print("TRAINING: Random Forest")
    print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    print("\nHyperparameters:")
    print(f"  n_estimators: 200")
    print(f"  max_depth: 10")
    print(f"  min_samples_split: 20")
    print(f"  min_samples_leaf: 10")
    
    model.fit(X_train, y_train)
    print(f"\n✓ Random Forest trained")
    
    return model

def train_xgboost(X_train, y_train):
    """Train Gradient Boosting classifier (sklearn's GradientBoostingClassifier)."""
    print("\n" + "="*80)
    print("TRAINING: Gradient Boosting (XGBoost-style)")
    print("="*80)
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=RANDOM_SEED
    )
    
    print("\nHyperparameters:")
    print(f"  n_estimators: 200")
    print(f"  learning_rate: 0.05")
    print(f"  max_depth: 4")
    print(f"  subsample: 0.8")
    
    model.fit(X_train, y_train)
    print(f"\n✓ Gradient Boosting trained")
    
    return model

def train_neural_network(X_train, y_train):
    """Train Neural Network classifier."""
    print("\n" + "="*80)
    print("TRAINING: Neural Network")
    print("="*80)
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("\nHyperparameters:")
    print(f"  hidden_layers: (64, 32, 16)")
    print(f"  activation: relu")
    print(f"  alpha: 0.001")
    print(f"  max_iter: 500")
    
    model.fit(X_train, y_train)
    print(f"\n✓ Neural Network trained")
    print(f"  Iterations: {model.n_iter_}")
    
    return model

def train_naive_bayes(X_train, y_train):
    """Train Gaussian Naive Bayes classifier."""
    print("\n" + "="*80)
    print("TRAINING: Gaussian Naive Bayes")
    print("="*80)
    
    model = GaussianNB()
    
    print("\nHyperparameters:")
    print(f"  (Default Gaussian NB)")
    
    model.fit(X_train, y_train)
    print(f"\n✓ Gaussian Naive Bayes trained")
    
    return model

def train_svm(X_train, y_train):
    """Train SVM classifier with probability calibration."""
    print("\n" + "="*80)
    print("TRAINING: SVM (with calibration)")
    print("="*80)
    
    base_svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_SEED,
        probability=True
    )
    
    print("\nHyperparameters:")
    print(f"  kernel: rbf")
    print(f"  C: 1.0")
    print(f"  gamma: scale")
    
    base_svm.fit(X_train, y_train)
    print(f"\n✓ SVM trained")
    
    return base_svm

def evaluate_model(model, X_val, y_val, model_name):
    """
    Evaluate model on validation set.
    Returns dict with metrics.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)
    accuracy = (y_pred == y_val).mean()
    
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'brier_score': brier,
        'accuracy': accuracy
    }
    
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    return metrics, y_pred_proba

def optimize_ensemble_weights(models_dict, X_val, y_val, individual_aucs):
    """
    Find optimal weights for ensemble by testing multiple strategies.
    """
    print("\n" + "="*80)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*80)
    
    # Get predictions from each model
    model_preds = {}
    for name, model in models_dict.items():
        model_preds[name] = model.predict_proba(X_val)[:, 1]
    
    model_names = list(model_preds.keys())
    
    print(f"\nTesting ensemble strategies for {len(model_names)} models...")
    print(f"Models: {model_names}")
    
    # Define weight strategies to try
    weight_strategies = []
    
    # Strategy 1: Uniform weights (baseline)
    uniform_weights = {name: 1.0/len(model_names) for name in model_names}
    weight_strategies.append(('Uniform', uniform_weights))
    
    # Strategy 2: Performance-weighted (by ROC-AUC)
    total_auc = sum(individual_aucs.values())
    perf_weights = {name: auc/total_auc for name, auc in individual_aucs.items()}
    weight_strategies.append(('Performance-weighted', perf_weights))
    
    # Strategy 3: Emphasize top 2 models
    sorted_models = sorted(individual_aucs.items(), key=lambda x: x[1], reverse=True)
    top2_weights = {}
    for i, (name, _) in enumerate(sorted_models):
        if i == 0:
            top2_weights[name] = 0.35  # Best model
        elif i == 1:
            top2_weights[name] = 0.30  # Second best
        else:
            top2_weights[name] = 0.35 / (len(model_names) - 2)  # Split remainder
    weight_strategies.append(('Emphasize Top 2', top2_weights))
    
    # Strategy 4: De-emphasize worst model
    deemph_weights = {}
    worst_model = sorted_models[-1][0]
    for name in model_names:
        if name == worst_model:
            deemph_weights[name] = 0.05
        else:
            deemph_weights[name] = 0.95 / (len(model_names) - 1)
    weight_strategies.append(('De-emphasize Worst', deemph_weights))
    
    # Strategy 5: Squared performance weights (emphasize gaps)
    squared_weights = {}
    squared_sum = sum(auc**2 for auc in individual_aucs.values())
    for name, auc in individual_aucs.items():
        squared_weights[name] = (auc**2) / squared_sum
    weight_strategies.append(('Squared Performance', squared_weights))
    
    # Test each strategy
    print(f"\n{'='*80}")
    print("TESTING STRATEGIES")
    print(f"{'='*80}")
    
    best_roc_auc = 0
    best_strategy = None
    best_weights = None
    
    results = []
    
    for strategy_name, weights in weight_strategies:
        # Calculate ensemble prediction
        ensemble_pred = sum(model_preds[name] * weights[name] for name in model_names)
        
        # Evaluate
        roc_auc = roc_auc_score(y_val, ensemble_pred)
        logloss = log_loss(y_val, ensemble_pred)
        brier = brier_score_loss(y_val, ensemble_pred)
        
        results.append({
            'strategy': strategy_name,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'brier_score': brier,
            'weights': weights
        })
        
        print(f"\n{strategy_name}:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Weights: {', '.join([f'{k}: {v:.2f}' for k, v in weights.items()])}")
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_strategy = strategy_name
            best_weights = weights
    
    print(f"\n{'='*80}")
    print(f"BEST STRATEGY: {best_strategy}")
    print(f"{'='*80}")
    print(f"ROC-AUC: {best_roc_auc:.4f}")
    print(f"\nOptimal weights:")
    for name, weight in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {weight:.3f}")
    
    return best_weights, best_roc_auc, results

def identify_problematic_models(models_dict, X_val, y_val):
    """
    Identify which models produce problematic predictions on validation set.
    Returns dict with exclusion frequency for each model.
    """
    print("\n" + "="*80)
    print("IDENTIFYING PROBLEMATIC MODELS")
    print("="*80)
    
    def is_problematic(prob):
        """Check if prediction is problematic."""
        if np.isnan(prob):
            return True, "NaN"
        if np.isinf(prob):
            return True, "Inf"
        if prob < 0.001:
            return True, "< 0.1%"
        if prob > 0.999:
            return True, "> 99.9%"
        return False, None
    
    # Track problematic predictions for each model
    model_issues = {}
    
    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        problematic_count = 0
        reasons = []
        
        for prob in y_pred_proba:
            is_prob, reason = is_problematic(prob)
            if is_prob:
                problematic_count += 1
                reasons.append(reason)
        
        exclusion_rate = problematic_count / len(y_val)
        
        model_issues[model_name] = {
            'problematic_count': problematic_count,
            'total_predictions': len(y_val),
            'exclusion_rate': exclusion_rate,
            'reasons': reasons[:5]  # Sample of reasons
        }
        
        print(f"\n{model_name}:")
        print(f"  Problematic predictions: {problematic_count}/{len(y_val)} ({exclusion_rate*100:.1f}%)")
        if reasons:
            print(f"  Sample reasons: {', '.join(set(reasons[:5]))}")
    
    # Identify models that should have fallback weights computed
    # Threshold: exclude >5% of predictions OR >3 predictions total
    fallback_models = []
    for model_name, info in model_issues.items():
        if info['exclusion_rate'] > 0.05 or info['problematic_count'] > 3:
            fallback_models.append(model_name)
            print(f"\n⚠ {model_name} needs fallback weights (excluded {info['exclusion_rate']*100:.1f}% of predictions)")
    
    if not fallback_models:
        print(f"\n✓ No models require fallback weights")
    
    return model_issues, fallback_models

def optimize_fallback_weights(models_dict, X_val, y_val, exclude_model, individual_aucs):
    """
    Optimize ensemble weights with one model excluded.
    
    Args:
        models_dict: Dict of all models
        X_val: Validation features
        y_val: Validation targets
        exclude_model: Name of model to exclude
        individual_aucs: Dict of individual model ROC-AUCs
    
    Returns:
        Optimal weights for remaining models
    """
    print(f"\n  Optimizing weights excluding {exclude_model}...")
    
    # Get predictions from remaining models
    remaining_models = {k: v for k, v in models_dict.items() if k != exclude_model}
    model_preds = {}
    for name, model in remaining_models.items():
        model_preds[name] = model.predict_proba(X_val)[:, 1]
    
    model_names = list(model_preds.keys())
    
    # Try multiple strategies
    strategies = []
    
    # Uniform
    uniform = {name: 1.0/len(model_names) for name in model_names}
    strategies.append(('Uniform', uniform))
    
    # Performance-weighted
    remaining_aucs = {k: v for k, v in individual_aucs.items() if k != exclude_model}
    total_auc = sum(remaining_aucs.values())
    perf = {name: auc/total_auc for name, auc in remaining_aucs.items()}
    strategies.append(('Performance', perf))
    
    # Emphasize best
    sorted_models = sorted(remaining_aucs.items(), key=lambda x: x[1], reverse=True)
    emph = {}
    for i, (name, _) in enumerate(sorted_models):
        if i == 0:
            emph[name] = 0.40
        elif i == 1:
            emph[name] = 0.30
        else:
            emph[name] = 0.30 / (len(model_names) - 2)
    strategies.append(('Emphasize Best', emph))
    
    # Find best
    best_auc = 0
    best_weights = None
    
    for strategy_name, weights in strategies:
        ensemble_pred = sum(model_preds[name] * weights[name] for name in model_names)
        auc = roc_auc_score(y_val, ensemble_pred)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = weights
    
    print(f"    Best strategy ROC-AUC: {best_auc:.4f}")
    print(f"    Weights: {', '.join([f'{k}: {v:.2f}' for k, v in best_weights.items()])}")
    
    return best_weights

def save_models(models_dict, scaler, ensemble_weights, feature_cols, fallback_weights=None):
    """
    Save trained models and configuration including fallback weights.
    """
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    # Save each model
    for name, model in models_dict.items():
        model_path = os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_")}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: {scaler_path}")
    
    # Save ensemble configuration with fallback weights
    ensemble_config = {
        'weights': ensemble_weights,
        'models': list(models_dict.keys()),
        'feature_columns': feature_cols,
        'train_years': TRAIN_YEARS,
        'validation_year': VALIDATION_YEAR if MODE == 'validation' else None,
        'mode': MODE,
        'strategy': PRODUCTION_STRATEGY if MODE == 'production' else 'optimized_from_validation',
        'random_seed': RANDOM_SEED
    }
    
    # Add fallback weights if provided
    if fallback_weights:
        ensemble_config['fallback_weights'] = fallback_weights
        print(f"\n✓ Included {len(fallback_weights)} fallback weight configurations")
    
    config_path = os.path.join(MODELS_DIR, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"✓ Saved: {config_path}")

def save_performance_report(metrics_list, ensemble_metrics, feature_cols, ensemble_results):
    """
    Save performance metrics to CSV.
    """
    print("\n" + "="*80)
    print("SAVING PERFORMANCE REPORT")
    print("="*80)
    
    # Combine individual and ensemble metrics
    all_metrics = metrics_list + [ensemble_metrics]
    
    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values('roc_auc', ascending=False)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'model_performance.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"\n✓ Performance report saved to: {output_path}")
    
    # Print summary
    print(f"\nFinal Performance Summary (sorted by ROC-AUC):")
    print(metrics_df.to_string(index=False))
    
    # Save ensemble strategy comparison
    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_path = os.path.join(OUTPUT_DIR, 'ensemble_strategies.csv')
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"\n✓ Ensemble strategies saved to: {ensemble_path}")
    
    # Save feature list
    features_path = os.path.join(OUTPUT_DIR, 'features_used.txt')
    with open(features_path, 'w') as f:
        f.write(f"Features used for model training ({len(feature_cols)} total):\n\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"{i}. {feat}\n")
    print(f"✓ Feature list saved to: {features_path}")

def create_performance_visualizations(metrics_list, ensemble_metrics):
    """
    Create visualization of model performance comparison.
    """
    print("\n" + "="*80)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("="*80)
    
    # Combine metrics
    all_metrics = metrics_list + [ensemble_metrics]
    metrics_df = pd.DataFrame(all_metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC-AUC comparison
    ax1 = axes[0, 0]
    models_sorted = metrics_df.sort_values('roc_auc', ascending=True)
    colors = ['red' if m == 'Ensemble' else 'steelblue' for m in models_sorted['model']]
    ax1.barh(models_sorted['model'], models_sorted['roc_auc'], color=colors)
    ax1.set_xlabel('ROC-AUC Score')
    ax1.set_title('Model Performance: ROC-AUC')
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    ax1.grid(axis='x', alpha=0.3)
    
    # Log Loss comparison (lower is better)
    ax2 = axes[0, 1]
    models_sorted = metrics_df.sort_values('log_loss', ascending=True)
    colors = ['red' if m == 'Ensemble' else 'steelblue' for m in models_sorted['model']]
    ax2.barh(models_sorted['model'], models_sorted['log_loss'], color=colors)
    ax2.set_xlabel('Log Loss (lower is better)')
    ax2.set_title('Model Performance: Log Loss')
    ax2.grid(axis='x', alpha=0.3)
    
    # Brier Score comparison (lower is better)
    ax3 = axes[1, 0]
    models_sorted = metrics_df.sort_values('brier_score', ascending=True)
    colors = ['red' if m == 'Ensemble' else 'steelblue' for m in models_sorted['model']]
    ax3.barh(models_sorted['model'], models_sorted['brier_score'], color=colors)
    ax3.set_xlabel('Brier Score (lower is better)')
    ax3.set_title('Model Performance: Brier Score')
    ax3.grid(axis='x', alpha=0.3)
    
    # Accuracy comparison
    ax4 = axes[1, 1]
    models_sorted = metrics_df.sort_values('accuracy', ascending=True)
    colors = ['red' if m == 'Ensemble' else 'steelblue' for m in models_sorted['model']]
    ax4.barh(models_sorted['model'], models_sorted['accuracy'], color=colors)
    ax4.set_xlabel('Accuracy')
    ax4.set_title('Model Performance: Accuracy')
    ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'model_performance_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Performance comparison saved to: {output_path}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("H2H 03_train_models.py")
    print("="*80)
    print("\nTraining head-to-head prediction models with temporal validation...")
    
    # Create directories
    create_output_directories()
    
    # Load data
    df, feature_cols = load_data()
    
    # Prepare train/validation split
    X_train, X_val, y_train, y_val, train_df, val_df = prepare_train_validation_split(df, feature_cols)
    
    # Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    # Train models
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    models = {}
    
    # Random Forest
    models['Random Forest'] = train_random_forest(X_train, y_train)
    
    # Gradient Boosting
    models['Gradient Boosting'] = train_xgboost(X_train, y_train)
    
    # Neural Network (needs scaled features)
    models['Neural Network'] = train_neural_network(X_train_scaled, y_train)
    
    # Gaussian Naive Bayes (needs scaled features)
    models['Gaussian Naive Bayes'] = train_naive_bayes(X_train_scaled, y_train)
    
    # SVM (needs scaled features)
    models['SVM'] = train_svm(X_train_scaled, y_train)
    
    # Evaluate models
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    if MODE == 'validation':
        # Full evaluation with validation set
        metrics_list = []
        model_predictions = {}
        individual_aucs = {}
        
        for name, model in models.items():
            # Use scaled features for NN, NB, SVM
            if name in ['Neural Network', 'Gaussian Naive Bayes', 'SVM']:
                X_eval = X_val_scaled
            else:
                X_eval = X_val
            
            metrics, y_pred_proba = evaluate_model(model, X_eval, y_val, name)
            metrics_list.append(metrics)
            model_predictions[name] = y_pred_proba
            individual_aucs[name] = metrics['roc_auc']
    else:
        # Production mode - no validation metrics
        print(f"\n⚠ Production mode - skipping validation evaluation")
        print(f"  Using all data for training")
        metrics_list = []
        individual_aucs = {}
    
    # Optimize ensemble (validation) or apply production strategy
    if MODE == 'validation':
        # For ensemble, need to re-predict with appropriate feature versions
        ensemble_models = {}
        for name, model in models.items():
            ensemble_models[name] = model
        
        # Get predictions for ensemble optimization
        ensemble_preds = {}
        for name, model in ensemble_models.items():
            if name in ['Neural Network', 'Gaussian Naive Bayes', 'SVM']:
                X_eval = X_val_scaled
            else:
                X_eval = X_val
            ensemble_preds[name] = model.predict_proba(X_eval)[:, 1]
        
        # Create temporary models dict for ensemble optimization
        temp_models = {}
        for name in ensemble_models.keys():
            # Create wrapper that returns correct predictions
            class PredWrapper:
                def __init__(self, preds):
                    self.preds = preds
                def predict_proba(self, X):
                    return np.column_stack([1 - self.preds, self.preds])
            temp_models[name] = PredWrapper(ensemble_preds[name])
        
        ensemble_weights, ensemble_roc_auc, ensemble_results = optimize_ensemble_weights(
            temp_models, X_val, y_val, individual_aucs
        )
        
        # Calculate full ensemble metrics
        ensemble_pred = sum(ensemble_preds[name] * ensemble_weights[name] for name in ensemble_weights.keys())
        ensemble_metrics = {
            'model': 'Ensemble',
            'roc_auc': roc_auc_score(y_val, ensemble_pred),
            'log_loss': log_loss(y_val, ensemble_pred),
            'brier_score': brier_score_loss(y_val, ensemble_pred),
            'accuracy': ((ensemble_pred >= 0.5) == y_val).mean()
        }
        
        print(f"\n" + "="*80)
        print("ENSEMBLE PERFORMANCE")
        print("="*80)
        print(f"  ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
        print(f"  Log Loss: {ensemble_metrics['log_loss']:.4f}")
        print(f"  Brier Score: {ensemble_metrics['brier_score']:.4f}")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        
        # Identify problematic models and compute fallback weights
        model_issues, fallback_models = identify_problematic_models(temp_models, X_val, y_val)
        
        # Compute fallback weights for each problematic model
        fallback_weights = {}
        if fallback_models:
            print(f"\n{'='*80}")
            print("COMPUTING FALLBACK WEIGHTS")
            print(f"{'='*80}")
            
            for exclude_model in fallback_models:
                fallback_key = f"exclude_{exclude_model.lower().replace(' ', '_')}"
                fallback_weights[fallback_key] = optimize_fallback_weights(
                    temp_models, X_val, y_val, exclude_model, individual_aucs
                )
    
    else:  # production mode
        # Apply production strategy from validation
        print(f"\n" + "="*80)
        print("APPLYING PRODUCTION STRATEGY")
        print("="*80)
        print(f"\nStrategy: {PRODUCTION_STRATEGY}")
        print(f"  (Determined from validation mode)")
        
        # Define strategy weights based on validation winner
        model_names = list(models.keys())
        
        if PRODUCTION_STRATEGY == 'Uniform':
            ensemble_weights = {name: 1.0/len(model_names) for name in model_names}
        
        elif PRODUCTION_STRATEGY == 'Performance-weighted':
            # Can't use validation AUCs - use uniform as fallback
            print(f"\n⚠ Cannot use performance weighting without validation set")
            print(f"  Falling back to uniform weights")
            ensemble_weights = {name: 1.0/len(model_names) for name in model_names}
        
        elif PRODUCTION_STRATEGY == 'Emphasize Top 2':
            # Assume same order as validation (GB, RF, NN, SVM, GNB)
            ensemble_weights = {
                'Gradient Boosting': 0.35,
                'Random Forest': 0.30,
                'Neural Network': 0.117,
                'SVM': 0.117,
                'Gaussian Naive Bayes': 0.117
            }
        
        elif PRODUCTION_STRATEGY == 'De-emphasize Worst':
            # Assume GNB is worst based on validation
            ensemble_weights = {
                'Gradient Boosting': 0.2375,
                'Random Forest': 0.2375,
                'Neural Network': 0.2375,
                'SVM': 0.2375,
                'Gaussian Naive Bayes': 0.05
            }
        
        elif PRODUCTION_STRATEGY == 'Squared Performance':
            # Approximate based on validation results
            ensemble_weights = {
                'Gradient Boosting': 0.38,
                'Random Forest': 0.29,
                'Neural Network': 0.16,
                'SVM': 0.12,
                'Gaussian Naive Bayes': 0.05
            }
        
        else:
            # Default to uniform
            print(f"\n⚠ Unknown strategy: {PRODUCTION_STRATEGY}")
            print(f"  Falling back to uniform weights")
            ensemble_weights = {name: 1.0/len(model_names) for name in model_names}
        
        print(f"\nProduction ensemble weights:")
        for name, weight in sorted(ensemble_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {weight:.3f}")
        
        ensemble_metrics = None
        ensemble_results = []
        fallback_weights = {}  # Skip fallback computation in production
    
    # Save models and results
    save_models(models, scaler, ensemble_weights, feature_cols, fallback_weights)
    
    if MODE == 'validation':
        save_performance_report(metrics_list, ensemble_metrics, feature_cols, ensemble_results)
        create_performance_visualizations(metrics_list, ensemble_metrics)
        
        print("\n" + "="*80)
        print("✓ MODEL TRAINING COMPLETE (VALIDATION)")
        print("="*80)
        print(f"\nModels saved to: {MODELS_DIR}/")
        print(f"Results saved to: {OUTPUT_DIR}/")
        print(f"\nBest model: {max(metrics_list + [ensemble_metrics], key=lambda x: x['roc_auc'])['model']}")
        print(f"Best ROC-AUC: {max(m['roc_auc'] for m in metrics_list + [ensemble_metrics]):.4f}")
        print(f"\n⚠ NEXT STEP: Update PRODUCTION_STRATEGY in config and run in production mode")
    else:
        print("\n" + "="*80)
        print("✓ MODEL TRAINING COMPLETE (PRODUCTION)")
        print("="*80)
        print(f"\nModels saved to: {MODELS_DIR}/")
        print(f"Strategy used: {PRODUCTION_STRATEGY}")
        print(f"\n✓ Ready for 2026 predictions")

if __name__ == "__main__":
    main()