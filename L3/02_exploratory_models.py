"""
L3 Exploratory Models Pipeline
Trains and evaluates multiple model types for Elite 8+ prediction.
Tests: Logistic Regression, Random Forest, XGBoost, Neural Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Configuration
INPUT_DIR = Path('outputs/01_feature_selection')
OUTPUT_DIR = Path('outputs/02_exploratory_models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.25  # Hold out 25% for testing
TRAIN_YEARS_CUTOFF = 2022  # Train on <=2022, test on 2023-2025

print("="*80)
print("L3 EXPLORATORY MODELS PIPELINE")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 80)

# Load labeled training data
labeled_long = pd.read_csv(INPUT_DIR / 'labeled_training_long.csv')
labeled_rich = pd.read_csv(INPUT_DIR / 'labeled_training_rich.csv')

# Load reduced feature lists
features_long = pd.read_csv(INPUT_DIR / 'reduced_features_long.csv')['feature'].tolist()
features_rich = pd.read_csv(INPUT_DIR / 'reduced_features_rich.csv')['feature'].tolist()

print(f"Loaded labeled_training_long: {labeled_long.shape[0]} rows")
print(f"Loaded labeled_training_rich: {labeled_rich.shape[0]} rows")
print(f"Features for long dataset: {len(features_long)}")
print(f"Features for rich dataset: {len(features_rich)}")

# ============================================================================
# PREPARE TRAIN/TEST SPLITS
# ============================================================================
print("\n[2] PREPARING TRAIN/TEST SPLITS")
print("-" * 80)

def prepare_dataset(df, feature_list, dataset_name):
    """Prepare features and labels, create time-based split"""
    
    # Extract features and label
    X = df[feature_list].copy()
    y = df['elite8_flag'].copy()
    years = df['Year'].copy()
    teams = df['Team'].copy()
    
    # Drop columns that are entirely NaN (unusable for modeling)
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"\n{dataset_name}:")
        print(f"  Dropping columns that are entirely NaN: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
    
    # Check for missing values and fill BEFORE splitting
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n{dataset_name}:")
        print(f"  Total missing values: {missing_count}")
        
        # Fill missing with median for each column
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    # If median is NaN (column has too many NaNs), use 0
                    print(f"  WARNING: {col} has too many NaNs, filling with 0")
                    X[col].fillna(0, inplace=True)
                else:
                    X[col].fillna(median_val, inplace=True)
                    print(f"  Filled {col} missing values with median: {median_val:.3f}")
    
    # Verify no NaNs remain
    remaining_nans = X.isnull().sum().sum()
    assert remaining_nans == 0, f"NaNs still present in {dataset_name} features! Count: {remaining_nans}"
    
    # Time-based split: train on years <= 2022, test on 2023+
    train_mask = years <= TRAIN_YEARS_CUTOFF
    test_mask = years > TRAIN_YEARS_CUTOFF
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    years_train = years[train_mask]
    years_test = years[test_mask]
    teams_test = teams[test_mask]
    
    # Standardize features (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Verify no NaNs in scaled data
    assert X_train_scaled.isnull().sum().sum() == 0, f"NaNs in scaled train data for {dataset_name}!"
    assert X_test_scaled.isnull().sum().sum() == 0, f"NaNs in scaled test data for {dataset_name}!"
    
    print(f"\n{dataset_name} - Final feature set:")
    print(f"  Features: {len(X.columns)}")
    print(f"  Train: {len(X_train)} samples ({years_train.min()}-{years_train.max()})")
    print(f"  Train Elite 8+: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({years_test.min()}-{years_test.max()})")
    print(f"  Test Elite 8+: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    return {
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'years_test': years_test,
        'teams_test': teams_test,
        'scaler': scaler
    }

data_long = prepare_dataset(labeled_long, features_long, "long")
data_rich = prepare_dataset(labeled_rich, features_rich, "rich")

# ============================================================================
# DEFINE MODELS
# ============================================================================
print("\n[3] DEFINING MODELS")
print("-" * 80)

def get_models():
    """Define models to train and evaluate"""
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            ),
            'use_scaling': True  # Logistic Regression benefits from scaling
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ),
            'use_scaling': False  # Tree-based models don't need scaling
        },
        'Neural Network': {
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.15
            ),
            'use_scaling': True  # Neural networks require scaling
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(
                n_neighbors=10,
                weights='distance',  # Weight by inverse distance
                metric='euclidean',
                n_jobs=-1
            ),
            'use_scaling': True  # Distance-based models need scaling
        },
        'Support Vector Machine': {
            'model': SVC(
                kernel='rbf',  # Radial basis function for non-linearity
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,  # CRITICAL: enables predict_proba()
                random_state=RANDOM_STATE
            ),
            'use_scaling': True  # Distance-based models need scaling
        },
        'Gaussian Naive Bayes': {
            'model': GaussianNB(),
            'use_scaling': False  # Naive Bayes doesn't require scaling (but won't hurt)
        }
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = {
            'model': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                scale_pos_weight=7,  # Roughly 88/12 for class imbalance
                eval_metric='logloss',
                n_jobs=-1
            ),
            'use_scaling': False  # Tree-based models don't need scaling
        }
    
    print(f"Models to train: {list(models.keys())}")
    return models

models = get_models()

# ============================================================================
# TRAIN AND EVALUATE MODELS
# ============================================================================
print("\n[4] TRAINING AND EVALUATING MODELS")
print("-" * 80)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    """Train model and calculate evaluation metrics"""
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba)
    }
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    
    return metrics, y_pred, y_pred_proba, feature_importance

def train_all_models(data, dataset_name):
    """Train all models on a dataset"""
    
    print(f"\n{dataset_name.upper()} DATASET:")
    print("-" * 40)
    
    results = []
    predictions = {}
    feature_importances = {}
    
    for model_name, model_config in models.items():
        print(f"\nTraining {model_name}...")
        
        model = model_config['model']
        use_scaling = model_config['use_scaling']
        
        # Select scaled or unscaled data based on model requirements
        if use_scaling:
            X_train_model = data['X_train_scaled']
            X_test_model = data['X_test_scaled']
        else:
            X_train_model = data['X_train']
            X_test_model = data['X_test']
        
        try:
            metrics, y_pred, y_pred_proba, feat_imp = evaluate_model(
                model, X_train_model, X_test_model, 
                data['y_train'], data['y_test'],
                model_name, dataset_name
            )
            
            results.append(metrics)
            predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            if feat_imp is not None:
                feature_importances[model_name] = feat_imp
            
            # Print metrics
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  Log Loss: {metrics['log_loss']:.3f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return pd.DataFrame(results), predictions, feature_importances

# Train on both datasets
results_long, predictions_long, feat_imp_long = train_all_models(data_long, "long")
results_rich, predictions_rich, feat_imp_rich = train_all_models(data_rich, "rich")

# Combine results
all_results = pd.concat([results_long, results_rich], ignore_index=True)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[5] SAVING RESULTS")
print("-" * 80)

# Save metrics
all_results.to_csv(OUTPUT_DIR / 'model_metrics.csv', index=False)
print(f"Saved model metrics to {OUTPUT_DIR / 'model_metrics.csv'}")

# Save feature importances
for dataset_name, feat_imps in [('long', feat_imp_long), ('rich', feat_imp_rich)]:
    for model_name, feat_imp in feat_imps.items():
        output_file = OUTPUT_DIR / f'feature_importance_{dataset_name}_{model_name.replace(" ", "_")}.csv'
        feat_imp.to_csv(output_file, index=False)
        print(f"Saved feature importance: {output_file.name}")

# Save predictions for best model (by ROC-AUC)
for dataset_name, data, predictions in [('long', data_long, predictions_long), 
                                        ('rich', data_rich, predictions_rich)]:
    # Find best model
    dataset_results = all_results[all_results['dataset'] == dataset_name]
    best_model = dataset_results.loc[dataset_results['roc_auc'].idxmax(), 'model']
    
    # Save predictions
    pred_df = pd.DataFrame({
        'Team': data['teams_test'].values,
        'Year': data['years_test'].values,
        'actual': data['y_test'].values,
        'predicted': predictions[best_model]['y_pred'],
        'probability': predictions[best_model]['y_pred_proba']
    })
    
    output_file = OUTPUT_DIR / f'predictions_{dataset_name}_best_model.csv'
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions: {output_file.name} (best model: {best_model})")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[6] CREATING VISUALIZATIONS")
print("-" * 80)

# Model comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16)

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    # Group by model and dataset
    pivot_data = all_results.pivot(index='model', columns='dataset', values=metric)
    
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(metric.upper().replace('_', ' '))
    ax.set_xlabel('')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend(title='Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved visualization: model_comparison.png")

# Feature importance plots (top 15 features for best models)
for dataset_name, feat_imps in [('long', feat_imp_long), ('rich', feat_imp_rich)]:
    if not feat_imps:
        continue
    
    # Get Random Forest importance (usually most interpretable)
    if 'Random Forest' in feat_imps:
        feat_imp = feat_imps['Random Forest'].head(15)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feat_imp)), feat_imp['importance'])
        plt.yticks(range(len(feat_imp)), feat_imp['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importances - Random Forest ({dataset_name})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'feature_importance_{dataset_name}_rf.png', dpi=150)
        plt.close()
        print(f"Saved visualization: feature_importance_{dataset_name}_rf.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPLORATORY MODELS COMPLETE")
print("="*80)

print("\nBEST MODELS BY ROC-AUC:")
for dataset in ['long', 'rich']:
    dataset_results = all_results[all_results['dataset'] == dataset]
    best_idx = dataset_results['roc_auc'].idxmax()
    best = dataset_results.loc[best_idx]
    print(f"\n{dataset.upper()} dataset:")
    print(f"  Model: {best['model']}")
    print(f"  ROC-AUC: {best['roc_auc']:.3f}")
    print(f"  Accuracy: {best['accuracy']:.3f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")

print("\nOUTPUTS GENERATED:")
print(f"  {OUTPUT_DIR}/model_metrics.csv - All model performance metrics")
print(f"  {OUTPUT_DIR}/predictions_*_best_model.csv - Best model predictions")
print(f"  {OUTPUT_DIR}/feature_importance_*.csv - Feature importance rankings")
print(f"  {OUTPUT_DIR}/model_comparison.png - Performance visualization")
print(f"  {OUTPUT_DIR}/feature_importance_*_rf.png - Feature importance plots")

print("\nNEXT STEPS:")
print("  1. Review model performance - which performs best?")
print("  2. Examine feature importances - do they make sense?")
print("  3. Check predictions on test set (2023-2025)")
print("  4. Proceed to 03_ensemble_models.py to combine models")
print("\n" + "="*80)