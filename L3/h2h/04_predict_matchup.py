"""
H2H 04_predict_matchup.py

Purpose: Prediction interface for head-to-head matchup probabilities.
         Loads trained models and provides clean API for predicting any matchup.
         Automatically detects and excludes problematic model predictions.

Inputs:
    - L3/h2h/models/ (trained model artifacts)
    - L3/data/predictionData/predict_set_YYYY.csv (current season team stats)

Usage:
    As a module:
        from predict_matchup import HeadToHeadPredictor
        predictor = HeadToHeadPredictor(year=2026)
        result = predictor.predict(teamA_id=211, teamB_id=101)
        print(f"P(TeamA wins): {result['ensemble_probability']:.1%}")
    
    As a CLI:
        python 04_predict_matchup.py --teamA 211 --teamB 101 --year 2026

Author: Ryan Browder
Date: 2025-01-31
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import argparse
from pathlib import Path

# ============================================================================
# HeadToHeadPredictor Class
# ============================================================================

class HeadToHeadPredictor:
    """
    Main prediction interface for head-to-head matchups.
    """
    
    def __init__(self, year=2026, models_dir='models', data_dir='../data/predictionData'):
        """
        Initialize predictor by loading models and data.
        
        Args:
            year: Season year for prediction data
            models_dir: Directory containing trained models
            data_dir: Directory containing prediction datasets
        """
        self.year = year
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        print(f"\n{'='*80}")
        print(f"INITIALIZING HEAD-TO-HEAD PREDICTOR")
        print(f"{'='*80}")
        print(f"\nYear: {year}")
        
        # Load models and configuration
        self._load_models()
        
        # Load prediction data
        self._load_prediction_data()
        
        print(f"\n✓ Predictor ready")
    
    def _load_models(self):
        """Load trained models, scaler, and ensemble configuration."""
        print(f"\nLoading models from: {self.models_dir}/")
        
        # Load ensemble configuration
        config_path = os.path.join(self.models_dir, 'ensemble_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"  Ensemble config loaded")
        print(f"  Feature columns: {len(self.config['feature_columns'])}")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  Feature scaler loaded")
        
        # Load each model
        self.models = {}
        for model_name in self.config['models']:
            model_filename = model_name.lower().replace(' ', '_') + '.pkl'
            model_path = os.path.join(self.models_dir, model_filename)
            
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            print(f"  {model_name} loaded")
        
        # Get ensemble weights (original optimized weights from training)
        self.original_weights = self.config['weights']
        
        print(f"\n✓ {len(self.models)} models loaded")
    
    def _load_prediction_data(self):
        """Load team statistics for current season."""
        # Try to find prediction file for specified year
        predict_file = f'predict_set_{self.year}.csv'
        predict_path = os.path.join(self.data_dir, predict_file)
        
        if not os.path.exists(predict_path):
            print(f"\n⚠ Warning: Prediction file not found: {predict_path}")
            print(f"  Prediction data will need to be provided manually")
            self.prediction_data = None
            return
        
        print(f"\nLoading prediction data from: {predict_path}")
        self.prediction_data = pd.read_csv(predict_path)
        
        print(f"  Shape: {self.prediction_data.shape}")
        print(f"  Teams: {self.prediction_data['Index'].nunique()}")
        
        # Verify we have all required features
        missing_features = []
        for feature in self.config['feature_columns']:
            # Feature columns are pct_diff_*, but prediction data has raw features
            # Extract the base feature name
            if feature.startswith('pct_diff_'):
                base_feature = feature.replace('pct_diff_', '')
                if base_feature not in self.prediction_data.columns:
                    missing_features.append(base_feature)
        
        if missing_features:
            print(f"\n⚠ Warning: Missing features in prediction data:")
            for feat in missing_features[:5]:
                print(f"    - {feat}")
            if len(missing_features) > 5:
                print(f"    ... and {len(missing_features) - 5} more")
    
    def _calculate_pct_diff(self, val_a, val_b):
        """
        Calculate symmetric percentage difference: (A - B) / ((A + B) / 2)
        """
        avg = (val_a + val_b) / 2.0
        
        if avg == 0 or np.isnan(avg):
            return 0.0
        
        return (val_a - val_b) / avg
    
    def _get_team_stats(self, team_id):
        """
        Get statistics for a team.
        
        Args:
            team_id: Team index/ID
        
        Returns:
            Series with team statistics, or None if not found
        """
        if self.prediction_data is None:
            return None
        
        team_data = self.prediction_data[self.prediction_data['Index'] == team_id]
        
        if len(team_data) == 0:
            return None
        
        return team_data.iloc[0]
    
    def _build_matchup_features(self, teamA_stats, teamB_stats):
        """
        Build percentage differential features for a matchup.
        
        Args:
            teamA_stats: Series with TeamA statistics
            teamB_stats: Series with TeamB statistics
        
        Returns:
            numpy array of features in correct order
        """
        features = []
        
        for feature_col in self.config['feature_columns']:
            # Feature column is like 'pct_diff_kenpom_Rk'
            # Extract base feature name
            base_feature = feature_col.replace('pct_diff_', '')
            
            # Get values for both teams
            val_a = teamA_stats.get(base_feature, 0)
            val_b = teamB_stats.get(base_feature, 0)
            
            # Calculate percentage difference
            pct_diff = self._calculate_pct_diff(val_a, val_b)
            features.append(pct_diff)
        
        return np.array(features).reshape(1, -1)
    
    def _is_problematic_prediction(self, prob):
        """
        Detect if a prediction is problematic and should be excluded.
        
        Args:
            prob: Predicted probability
        
        Returns:
            (is_problematic, reason)
        """
        if np.isnan(prob):
            return True, "NaN prediction"
        if np.isinf(prob):
            return True, "Infinite prediction"
        if prob < 0.001:
            return True, "Extreme confidence (< 0.1%)"
        if prob > 0.999:
            return True, "Extreme confidence (> 99.9%)"
        return False, None
    
    def predict(self, teamA_id, teamB_id, teamA_stats=None, teamB_stats=None, return_details=True):
        """
        Predict probability that TeamA beats TeamB.
        
        Args:
            teamA_id: Team A index/ID
            teamB_id: Team B index/ID
            teamA_stats: Optional - provide stats directly (if not loading from file)
            teamB_stats: Optional - provide stats directly (if not loading from file)
            return_details: If True, return detailed breakdown by model
        
        Returns:
            Dictionary with prediction results
        """
        # Get team statistics
        if teamA_stats is None:
            teamA_stats = self._get_team_stats(teamA_id)
            if teamA_stats is None:
                return {
                    'error': f'TeamA (ID {teamA_id}) not found in prediction data',
                    'success': False
                }
        
        if teamB_stats is None:
            teamB_stats = self._get_team_stats(teamB_id)
            if teamB_stats is None:
                return {
                    'error': f'TeamB (ID {teamB_id}) not found in prediction data',
                    'success': False
                }
        
        # Build feature vector
        X = self._build_matchup_features(teamA_stats, teamB_stats)
        
        # Get predictions from each model
        model_predictions = {}
        excluded_models = {}
        
        for model_name, model in self.models.items():
            # Use scaled features for models that need them
            if model_name in ['Neural Network', 'Gaussian Naive Bayes', 'SVM']:
                X_model = self.scaler.transform(X)
            else:
                X_model = X
            
            # Get probability
            prob = model.predict_proba(X_model)[0, 1]
            
            # Check if prediction is problematic
            is_problematic, reason = self._is_problematic_prediction(prob)
            
            if is_problematic:
                excluded_models[model_name] = {
                    'raw_probability': float(prob),
                    'reason': reason,
                    'original_weight': self.original_weights.get(model_name, 0.0)
                }
            else:
                model_predictions[model_name] = float(prob)
        
        # Calculate ensemble prediction with dynamic re-weighting
        if len(model_predictions) == 0:
            # All models excluded - return error
            return {
                'error': 'All models produced problematic predictions',
                'success': False,
                'excluded_models': excluded_models
            }
        
        # Determine which models are excluded
        excluded_model_names = set(excluded_models.keys())
        
        # Check if we have pre-computed fallback weights for this exclusion pattern
        fallback_weights = self.config.get('fallback_weights', {})
        adjusted_weights = None
        weight_source = None
        
        # Try to find matching fallback weight configuration
        if len(excluded_model_names) == 1:
            # Single model excluded - check for pre-computed weights
            excluded_model = list(excluded_model_names)[0]
            fallback_key = f"exclude_{excluded_model.lower().replace(' ', '_')}"
            
            if fallback_key in fallback_weights:
                # Use pre-computed optimal weights
                adjusted_weights = fallback_weights[fallback_key]
                weight_source = f"Pre-computed (excluding {excluded_model})"
        
        # Fallback to re-normalization if no pre-computed weights available
        if adjusted_weights is None:
            # Re-normalize ORIGINAL weights for non-excluded models
            total_weight = sum(
                self.original_weights[name] 
                for name in model_predictions.keys()
            )
            
            adjusted_weights = {
                name: self.original_weights[name] / total_weight
                for name in model_predictions.keys()
            }
            weight_source = "Re-normalized original weights"
        
        # Calculate ensemble prediction
        ensemble_prob = sum(
            model_predictions[name] * adjusted_weights[name]
            for name in model_predictions.keys()
        )
        
        # Build result
        result = {
            'success': True,
            'teamA_id': teamA_id,
            'teamB_id': teamB_id,
            'teamA_name': teamA_stats.get('Team', 'Unknown') if isinstance(teamA_stats, pd.Series) else 'Unknown',
            'teamB_name': teamB_stats.get('Team', 'Unknown') if isinstance(teamB_stats, pd.Series) else 'Unknown',
            'ensemble_probability': float(ensemble_prob),
            'ensemble_weights': adjusted_weights,
            'original_weights': self.original_weights,
            'weight_source': weight_source,
            'models_used': len(model_predictions),
            'models_excluded': len(excluded_models)
        }
        
        if return_details:
            result['model_predictions'] = model_predictions
            if excluded_models:
                result['excluded_models'] = excluded_models
        
        return result
    
    def predict_and_print(self, teamA_id, teamB_id):
        """
        Predict and print results in human-readable format.
        """
        result = self.predict(teamA_id, teamB_id)
        
        if not result['success']:
            print(f"\n❌ Error: {result['error']}")
            if 'excluded_models' in result:
                print(f"\nExcluded models:")
                for model_name, info in result['excluded_models'].items():
                    print(f"  - {model_name}: {info['reason']}")
            return result
        
        print(f"\n{'='*80}")
        print(f"MATCHUP PREDICTION")
        print(f"{'='*80}")
        print(f"\nTeamA: {result['teamA_name']} (ID: {result['teamA_id']})")
        print(f"TeamB: {result['teamB_name']} (ID: {result['teamB_id']})")
        print(f"\n{'='*80}")
        print(f"ENSEMBLE PREDICTION")
        print(f"{'='*80}")
        print(f"\nP(TeamA wins): {result['ensemble_probability']:.1%}")
        print(f"P(TeamB wins): {(1 - result['ensemble_probability']):.1%}")
        print(f"\nModels used: {result['models_used']}/{result['models_used'] + result['models_excluded']}")
        print(f"Weight source: {result['weight_source']}")
        
        if result['models_excluded'] > 0:
            print(f"⚠ {result['models_excluded']} model(s) excluded for problematic predictions")
        
        if 'model_predictions' in result:
            print(f"\n{'='*80}")
            print(f"INDIVIDUAL MODEL PREDICTIONS")
            print(f"{'='*80}")
            
            # Sort by probability
            sorted_preds = sorted(
                result['model_predictions'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for model_name, prob in sorted_preds:
                adjusted_weight = result['ensemble_weights'].get(model_name, 0)
                original_weight = result['original_weights'].get(model_name, 0)
                print(f"\n{model_name}:")
                print(f"  P(TeamA wins): {prob:.1%}")
                print(f"  Weight: {adjusted_weight:.3f} (original: {original_weight:.3f})")
        
        # Show excluded models
        if 'excluded_models' in result:
            print(f"\n{'='*80}")
            print(f"EXCLUDED MODELS")
            print(f"{'='*80}")
            
            for model_name, info in result['excluded_models'].items():
                print(f"\n{model_name}:")
                print(f"  Raw probability: {info['raw_probability']:.4f}")
                print(f"  Reason: {info['reason']}")
                print(f"  Original weight: {info['original_weight']:.3f}")
        
        return result

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Predict head-to-head matchup probabilities'
    )
    parser.add_argument(
        '--teamA',
        type=int,
        required=True,
        help='Team A ID/Index'
    )
    parser.add_argument(
        '--teamB',
        type=int,
        required=True,
        help='Team B ID/Index'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2026,
        help='Season year (default: 2026)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/predictionData',
        help='Directory containing prediction data'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HeadToHeadPredictor(
        year=args.year,
        models_dir=args.models_dir,
        data_dir=args.data_dir
    )
    
    # Make prediction and print
    predictor.predict_and_print(args.teamA, args.teamB)

if __name__ == "__main__":
    main()
