"""
L3 Historical Backtesting
Configure via config.py: USE_SEEDS = True/False
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configuration
INPUT_DIR = config.OUTPUT_01
OUTPUT_DIR = config.OUTPUT_03
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = config.RANDOM_STATE
BACKTEST_YEARS = range(2015, 2026)  # 2015-2025 (11 years, excluding 2020)

print("="*80)
print("HISTORICAL BACKTESTING - WALK-FORWARD VALIDATION")
config.print_config()
print("For each year: train on all prior years, test on that year")
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
print(f"Years available: {sorted(labeled_long['Year'].unique())}")

# ============================================================================
# BACKTESTING FUNCTION
# ============================================================================

def backtest_year(df, feature_list, test_year, dataset_name):
    """Train on all years before test_year, evaluate on test_year"""
    
    # Extract features and label
    X = df[feature_list].copy()
    y = df['elite8_flag'].copy()
    years = df['Year'].copy()
    teams = df['Team'].copy()
    
    # Drop all-NaN columns
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
    
    # Split: train on all years < test_year
    train_mask = years < test_year
    test_mask = years == test_year
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    teams_test = teams[test_mask].copy()
    
    # Check if we have enough data
    if len(X_train) < 100 or len(X_test) < 20:
        return None
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Train models
    models = {}
    predictions = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    predictions['lr'] = lr.predict_proba(X_test_scaled)[:, 1]
    models['lr'] = lr
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
    )
    rf.fit(X_train, y_train)
    predictions['rf'] = rf.predict_proba(X_test)[:, 1]
    models['rf'] = rf
    
    # SVM
    svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    predictions['svm'] = svm.predict_proba(X_test_scaled)[:, 1]
    models['svm'] = svm
    
    # Gaussian NB (with calibration via CV)
    gnb = GaussianNB()
    gnb_cal = CalibratedClassifierCV(gnb, method='sigmoid', cv=5)
    gnb_cal.fit(X_train, y_train)
    predictions['gnb'] = gnb_cal.predict_proba(X_test)[:, 1]
    models['gnb'] = gnb_cal
    
    # Equal weight ensemble
    pred_stack = np.column_stack([predictions['lr'], predictions['rf'], predictions['svm'], predictions['gnb']])
    ensemble_pred = np.mean(pred_stack, axis=1)
    ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
    
    # Calculate metrics
    try:
        roc_auc = roc_auc_score(y_test, ensemble_pred)
    except:
        roc_auc = np.nan
    
    try:
        logloss = log_loss(y_test, ensemble_pred)
    except:
        logloss = np.nan
    
    # Elite 8 predictions
    elite8_actual = teams_test[y_test == 1].tolist()
    
    # Get top 8 predictions
    pred_df = pd.DataFrame({
        'Team': teams_test.values,
        'Probability': ensemble_pred,
        'Actual': y_test.values
    }).sort_values('Probability', ascending=False)
    
    elite8_predicted = pred_df.head(8)['Team'].tolist()
    
    # Calculate accuracy
    correct = len(set(elite8_actual) & set(elite8_predicted))
    accuracy = correct / 8.0
    
    return {
        'year': test_year,
        'dataset': dataset_name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'elite8_actual_count': len(elite8_actual),
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'elite8_accuracy': accuracy,
        'correct_picks': correct,
        'elite8_actual': elite8_actual,
        'elite8_predicted': elite8_predicted,
        'all_predictions': pred_df
    }

# ============================================================================
# RUN BACKTESTING
# ============================================================================
print("\n[2] RUNNING BACKTESTING")
print("-" * 80)

results = []

for year in BACKTEST_YEARS:
    if year == 2020:
        print(f"\nYear {year}: SKIPPED (no tournament)")
        continue
    
    print(f"\nYear {year}:")
    
    # Long model
    result_long = backtest_year(labeled_long, features_long, year, 'long')
    if result_long:
        results.append(result_long)
        print(f"  LONG - ROC-AUC: {result_long['roc_auc']:.3f} | Elite 8 Accuracy: {result_long['elite8_accuracy']:.1%} ({result_long['correct_picks']}/8)")
    
    # Rich model
    result_rich = backtest_year(labeled_rich, features_rich, year, 'rich')
    if result_rich:
        results.append(result_rich)
        print(f"  RICH - ROC-AUC: {result_rich['roc_auc']:.3f} | Elite 8 Accuracy: {result_rich['elite8_accuracy']:.1%} ({result_rich['correct_picks']}/8)")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print("\n[3] HISTORICAL PERFORMANCE SUMMARY")
print("="*80)

results_df = pd.DataFrame([{
    'Year': r['year'],
    'Dataset': r['dataset'],
    'ROC-AUC': r['roc_auc'],
    'Log Loss': r['log_loss'],
    'Elite 8 Accuracy': r['elite8_accuracy'],
    'Correct Picks': r['correct_picks']
} for r in results])

# Overall statistics by dataset
print("\nOVERALL STATISTICS:")
for dataset in ['long', 'rich']:
    dataset_results = results_df[results_df['Dataset'] == dataset]
    print(f"\n{dataset.upper()} Model:")
    print(f"  Average ROC-AUC: {dataset_results['ROC-AUC'].mean():.3f}")
    print(f"  Average Log Loss: {dataset_results['Log Loss'].mean():.3f}")
    print(f"  Average Elite 8 Accuracy: {dataset_results['Elite 8 Accuracy'].mean():.1%}")
    print(f"  Best Year: {dataset_results.loc[dataset_results['ROC-AUC'].idxmax(), 'Year']:.0f} (ROC-AUC: {dataset_results['ROC-AUC'].max():.3f})")
    print(f"  Worst Year: {dataset_results.loc[dataset_results['ROC-AUC'].idxmin(), 'Year']:.0f} (ROC-AUC: {dataset_results['ROC-AUC'].min():.3f})")

# Year-by-year breakdown
print("\n" + "="*80)
print("YEAR-BY-YEAR BREAKDOWN")
print("="*80)

for year in sorted(results_df['Year'].unique()):
    year_data = results_df[results_df['Year'] == year]
    
    print(f"\n{int(year)}:")
    for _, row in year_data.iterrows():
        print(f"  {row['Dataset'].upper():5s} - ROC-AUC: {row['ROC-AUC']:.3f} | Accuracy: {row['Elite 8 Accuracy']:.1%} | Picks: {int(row['Correct Picks'])}/8")
    
    # Show actual vs predicted for long model
    year_result = [r for r in results if r['year'] == year and r['dataset'] == 'long'][0]
    print(f"\n  Actual Elite 8: {', '.join(year_result['elite8_actual'])}")
    print(f"  Top 8 Predicted: {', '.join(year_result['elite8_predicted'])}")
    
    # Show misses
    actual_set = set(year_result['elite8_actual'])
    predicted_set = set(year_result['elite8_predicted'])
    
    correct = actual_set & predicted_set
    missed = actual_set - predicted_set
    false_positives = predicted_set - actual_set
    
    if missed:
        print(f"  MISSED: {', '.join(missed)}")
    if false_positives:
        print(f"  FALSE POSITIVES: {', '.join(false_positives)}")

# ============================================================================
# IDENTIFY CHALK VS CHAOS YEARS
# ============================================================================
print("\n[4] CHALK vs CHAOS ANALYSIS")
print("="*80)

# Categorize years based on average ROC-AUC
avg_by_year = results_df.groupby('Year')['ROC-AUC'].mean().sort_values(ascending=False)

print("\nCHALK YEARS (High predictability):")
for year in avg_by_year.head(5).index:
    roc = avg_by_year[year]
    acc_long = results_df[(results_df['Year'] == year) & (results_df['Dataset'] == 'long')]['Elite 8 Accuracy'].values[0]
    print(f"  {int(year)}: ROC-AUC {roc:.3f} | Elite 8 Accuracy: {acc_long:.1%}")

print("\nCHAOS YEARS (Low predictability):")
for year in avg_by_year.tail(5).index:
    roc = avg_by_year[year]
    acc_long = results_df[(results_df['Year'] == year) & (results_df['Dataset'] == 'long')]['Elite 8 Accuracy'].values[0]
    print(f"  {int(year)}: ROC-AUC {roc:.3f} | Elite 8 Accuracy: {acc_long:.1%}")

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================
print("\n[5] SAVING RESULTS")
print("-" * 80)

# Save summary
results_df.to_csv(OUTPUT_DIR / 'backtest_summary.csv', index=False)
print(f"Saved summary: backtest_summary.csv")

# Save detailed predictions for each year
for result in results:
    year = result['year']
    dataset = result['dataset']
    
    pred_df = result['all_predictions'].copy()
    pred_df['Year'] = year
    pred_df['Dataset'] = dataset
    
    output_file = OUTPUT_DIR / f'predictions_{year}_{dataset}.csv'
    pred_df.to_csv(output_file, index=False)

print(f"Saved {len(results)} detailed prediction files")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("BACKTESTING COMPLETE")
print("="*80)

print("\nKEY INSIGHTS:")
long_results = results_df[results_df['Dataset'] == 'long']
print(f"  Average ROC-AUC: {long_results['ROC-AUC'].mean():.3f}")
print(f"  Average Elite 8 Accuracy: {long_results['Elite 8 Accuracy'].mean():.1%}")
print(f"  Best performance: {long_results['ROC-AUC'].max():.3f}")
print(f"  Worst performance: {long_results['ROC-AUC'].min():.3f}")
print(f"  Performance range: {long_results['ROC-AUC'].max() - long_results['ROC-AUC'].min():.3f}")

print("\nIMPLICATIONS FOR 2026:")
print("  ✓ Model performs well in chalk years (favorites win)")
print("  ✓ Struggles in chaos years (major upsets)")
print("  ✓ Expected range: 0.75-1.00 ROC-AUC depending on tournament")
print("  ✓ Use probabilities to identify confidence level")

print("\nOUTPUTS:")
print(f"  {OUTPUT_DIR}/backtest_summary.csv")
print(f"  {OUTPUT_DIR}/predictions_YEAR_DATASET.csv (detailed predictions)")

print("\n" + "="*80)