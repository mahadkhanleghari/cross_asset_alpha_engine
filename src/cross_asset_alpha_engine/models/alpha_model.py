"""Alpha model wrapper for scikit-learn models.

This module provides a unified interface for alpha generation models,
supporting both regime-aware and standard approaches with comprehensive
evaluation and prediction capabilities.
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.base import BaseEstimator, clone
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise ImportError(f"Required scikit-learn not installed: {e}")

from .model_utils import ModelUtils, ModelConfig
from ..regimes.hmm_regime_model import RegimeHMM
from ..utils.logging_utils import LoggerMixin


@dataclass
class AlphaModelConfig:
    """Configuration for alpha model."""
    # Model selection
    model_type: str = "random_forest"  # "linear", "ridge", "lasso", "elastic_net", "random_forest", "gbm", "svm", "mlp"
    model_params: Dict[str, Any] = None
    
    # Regime awareness
    regime_aware: bool = False
    regime_model: Optional[RegimeHMM] = None
    regime_features: List[str] = None
    
    # Training
    test_size: float = 0.2
    random_state: int = 42
    
    # Prediction
    return_probabilities: bool = False
    confidence_intervals: bool = False
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = self._get_default_params()
        
        if self.regime_features is None:
            self.regime_features = []
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for different model types."""
        defaults = {
            "linear": {},
            "ridge": {"alpha": 1.0},
            "lasso": {"alpha": 1.0},
            "elastic_net": {"alpha": 1.0, "l1_ratio": 0.5},
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state
            },
            "gbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_samples_split": 5,
                "random_state": self.random_state
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale"
            },
            "mlp": {
                "hidden_layer_sizes": (100, 50),
                "max_iter": 500,
                "random_state": self.random_state
            }
        }
        
        return defaults.get(self.model_type, {})


class AlphaModel(LoggerMixin):
    """Wrapper for scikit-learn models with alpha-specific functionality."""
    
    def __init__(self, config: Optional[AlphaModelConfig] = None):
        """Initialize alpha model.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or AlphaModelConfig()
        self.model = None
        self.regime_models = {}  # For regime-specific models
        self.model_utils = ModelUtils()
        self.is_fitted = False
        self.training_history = []
        
        self.logger.info(f"Initialized AlphaModel with type: {self.config.model_type}")
    
    def _create_base_model(self) -> BaseEstimator:
        """Create base scikit-learn model.
        
        Returns:
            Configured scikit-learn model
        """
        model_map = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "elastic_net": ElasticNet,
            "random_forest": RandomForestRegressor,
            "gbm": GradientBoostingRegressor,
            "svm": SVR,
            "mlp": MLPRegressor
        }
        
        if self.config.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        model_class = model_map[self.config.model_type]
        return model_class(**self.config.model_params)
    
    def _prepare_regime_features(
        self,
        X: pd.DataFrame,
        regimes: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Add regime information to features.
        
        Args:
            X: Feature matrix
            regimes: Regime predictions (optional)
            
        Returns:
            Feature matrix with regime information
        """
        X_with_regimes = X.copy()
        
        if regimes is not None:
            # Add regime as categorical feature
            X_with_regimes["regime"] = regimes
            
            # Add regime dummy variables
            regime_dummies = pd.get_dummies(regimes, prefix="regime")
            X_with_regimes = pd.concat([X_with_regimes, regime_dummies], axis=1)
        
        elif self.config.regime_model is not None:
            # Predict regimes using regime model
            try:
                regime_features = X[self.config.regime_features] if self.config.regime_features else X
                predicted_regimes = self.config.regime_model.predict_regimes(regime_features)
                regime_probs = self.config.regime_model.predict_proba(regime_features)
                
                X_with_regimes["regime"] = predicted_regimes
                
                # Add regime probabilities as features
                for i in range(regime_probs.shape[1]):
                    X_with_regimes[f"regime_prob_{i}"] = regime_probs[:, i]
                
            except Exception as e:
                self.logger.warning(f"Could not add regime features: {e}")
        
        return X_with_regimes
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'AlphaModel':
        """Fit alpha model to training data.
        
        Args:
            X: Feature matrix
            y: Target variable (returns or labels)
            regimes: Optional regime labels for regime-aware training
            sample_weight: Optional sample weights
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting alpha model with {len(X)} samples and {X.shape[1]} features")
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if sample_weight is not None:
            sample_weight = sample_weight[X.index.get_indexer(common_idx)]
        
        # Prepare features
        X_processed = self.model_utils.prepare_features(X_aligned, y_aligned, fit_preprocessors=True)
        
        # Add regime information if available
        if self.config.regime_aware:
            if regimes is not None:
                regimes_aligned = regimes[X.index.get_indexer(common_idx)]
            else:
                regimes_aligned = None
            
            X_processed = self._prepare_regime_features(X_processed, regimes_aligned)
        
        # Train model
        start_time = datetime.now()
        
        if self.config.regime_aware and regimes is not None:
            # Train separate models for each regime
            unique_regimes = np.unique(regimes_aligned)
            self.regime_models = {}
            
            for regime in unique_regimes:
                regime_mask = regimes_aligned == regime
                if regime_mask.sum() < 10:  # Skip regimes with too few samples
                    self.logger.warning(f"Skipping regime {regime} with only {regime_mask.sum()} samples")
                    continue
                
                X_regime = X_processed[regime_mask]
                y_regime = y_aligned[regime_mask]
                weight_regime = sample_weight[regime_mask] if sample_weight is not None else None
                
                regime_model = self._create_base_model()
                
                # Fit regime-specific model
                if weight_regime is not None and hasattr(regime_model, 'fit') and 'sample_weight' in regime_model.fit.__code__.co_varnames:
                    regime_model.fit(X_regime, y_regime, sample_weight=weight_regime)
                else:
                    regime_model.fit(X_regime, y_regime)
                
                self.regime_models[regime] = regime_model
                self.logger.info(f"Fitted model for regime {regime} with {len(X_regime)} samples")
            
            # Also train a global model as fallback
            self.model = self._create_base_model()
            if sample_weight is not None and hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
                self.model.fit(X_processed, y_aligned, sample_weight=sample_weight)
            else:
                self.model.fit(X_processed, y_aligned)
        
        else:
            # Train single global model
            self.model = self._create_base_model()
            if sample_weight is not None and hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
                self.model.fit(X_processed, y_aligned, sample_weight=sample_weight)
            else:
                self.model.fit(X_processed, y_aligned)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Record training history
        training_record = {
            "timestamp": datetime.now(),
            "n_samples": len(X_processed),
            "n_features": X_processed.shape[1],
            "training_time": training_time,
            "regime_aware": self.config.regime_aware,
            "n_regimes": len(self.regime_models) if self.regime_models else 1
        }
        self.training_history.append(training_record)
        
        self.is_fitted = True
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        regimes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate predictions from fitted model.
        
        Args:
            X: Feature matrix
            regimes: Optional regime labels for regime-aware prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        X_processed = self.model_utils.prepare_features(X, pd.Series(index=X.index), fit_preprocessors=False)
        
        # Add regime information if available
        if self.config.regime_aware:
            X_processed = self._prepare_regime_features(X_processed, regimes)
        
        # Generate predictions
        if self.config.regime_aware and self.regime_models and regimes is not None:
            # Use regime-specific models
            predictions = np.full(len(X), np.nan)
            
            for regime, regime_model in self.regime_models.items():
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    X_regime = X_processed[regime_mask]
                    predictions[regime_mask] = regime_model.predict(X_regime)
            
            # Use global model for missing predictions
            missing_mask = np.isnan(predictions)
            if missing_mask.sum() > 0:
                predictions[missing_mask] = self.model.predict(X_processed[missing_mask])
        
        else:
            # Use global model
            predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        regimes: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Generate prediction probabilities (for classification models).
        
        Args:
            X: Feature matrix
            regimes: Optional regime labels
            
        Returns:
            Array of prediction probabilities or None if not supported
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            self.logger.warning("Model does not support probability predictions")
            return None
        
        # Prepare features
        X_processed = self.model_utils.prepare_features(X, pd.Series(index=X.index), fit_preprocessors=False)
        
        # Add regime information if available
        if self.config.regime_aware:
            X_processed = self._prepare_regime_features(X_processed, regimes)
        
        # Generate probability predictions
        if self.config.regime_aware and self.regime_models and regimes is not None:
            # Use regime-specific models
            n_classes = self.model.classes_.shape[0] if hasattr(self.model, 'classes_') else 2
            probabilities = np.full((len(X), n_classes), np.nan)
            
            for regime, regime_model in self.regime_models.items():
                if hasattr(regime_model, 'predict_proba'):
                    regime_mask = regimes == regime
                    if regime_mask.sum() > 0:
                        X_regime = X_processed[regime_mask]
                        probabilities[regime_mask] = regime_model.predict_proba(X_regime)
            
            # Use global model for missing predictions
            missing_mask = np.isnan(probabilities).any(axis=1)
            if missing_mask.sum() > 0:
                probabilities[missing_mask] = self.model.predict_proba(X_processed[missing_mask])
        
        else:
            # Use global model
            probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from fitted model.
        
        Returns:
            Series with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get feature names from model utils
        if self.model_utils.selected_features is not None:
            feature_names = list(self.model_utils.selected_features)
        else:
            feature_names = []
        
        # Get importance from global model
        importance = self.model_utils.calculate_feature_importance(self.model, feature_names)
        
        return importance
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            regimes: Optional regime labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate predictions
        y_pred = self.predict(X, regimes)
        
        # Align predictions with true values
        common_idx = X.index.intersection(y.index)
        y_true_aligned = y.loc[common_idx].values
        y_pred_aligned = y_pred[X.index.get_indexer(common_idx)]
        
        # Calculate metrics
        if pd.api.types.is_numeric_dtype(y):
            metrics = self.model_utils.evaluate_regression_model(y_true_aligned, y_pred_aligned)
        else:
            y_proba = self.predict_proba(X, regimes)
            metrics = self.model_utils.evaluate_classification_model(
                y_true_aligned, y_pred_aligned, y_proba
            )
        
        # Add model-specific information
        evaluation = {
            "metrics": metrics,
            "model_type": self.config.model_type,
            "regime_aware": self.config.regime_aware,
            "n_samples": len(y_true_aligned),
            "n_features": X.shape[1]
        }
        
        # Regime-specific evaluation
        if self.config.regime_aware and regimes is not None:
            regime_metrics = {}
            regimes_aligned = regimes[X.index.get_indexer(common_idx)]
            
            for regime in np.unique(regimes_aligned):
                regime_mask = regimes_aligned == regime
                if regime_mask.sum() > 5:  # Only evaluate if enough samples
                    y_true_regime = y_true_aligned[regime_mask]
                    y_pred_regime = y_pred_aligned[regime_mask]
                    
                    if pd.api.types.is_numeric_dtype(y):
                        regime_metrics[f"regime_{regime}"] = self.model_utils.evaluate_regression_model(
                            y_true_regime, y_pred_regime
                        )
                    else:
                        regime_metrics[f"regime_{regime}"] = self.model_utils.evaluate_classification_model(
                            y_true_regime, y_pred_regime
                        )
            
            evaluation["regime_metrics"] = regime_metrics
        
        return evaluation
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error"
    ) -> Dict[str, Any]:
        """Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            self.model = self._create_base_model()
        
        return self.model_utils.cross_validate_model(self.model, X, y, scoring, cv)
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save fitted model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "regime_models": self.regime_models,
            "config": self.config,
            "model_utils": self.model_utils,
            "is_fitted": self.is_fitted,
            "training_history": self.training_history
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'AlphaModel':
        """Load fitted model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.regime_models = model_data["regime_models"]
        self.config = model_data["config"]
        self.model_utils = model_data["model_utils"]
        self.is_fitted = model_data["is_fitted"]
        self.training_history = model_data.get("training_history", [])
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary.
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        summary = {
            "status": "fitted",
            "model_type": self.config.model_type,
            "regime_aware": self.config.regime_aware,
            "n_regime_models": len(self.regime_models),
            "training_history": self.training_history
        }
        
        # Model parameters
        if self.model is not None:
            summary["model_params"] = self.model.get_params()
        
        # Feature importance
        try:
            importance = self.get_feature_importance()
            summary["top_features"] = importance.head(10).to_dict()
        except Exception as e:
            self.logger.warning(f"Could not get feature importance: {e}")
        
        return summary
