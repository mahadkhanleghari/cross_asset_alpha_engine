"""Utilities for model training, evaluation, and selection.

This module provides common utilities for machine learning models used in
alpha generation, including feature selection, cross-validation, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.base import BaseEstimator
except ImportError as e:
    raise ImportError(f"Required scikit-learn not installed: {e}")

from ..utils.logging_utils import LoggerMixin


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    gap: int = 0  # Gap between train and test for time series
    
    # Feature selection
    feature_selection: bool = True
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"  # "f_test", "mutual_info"
    
    # Preprocessing
    scale_features: bool = True
    scaler_type: str = "standard"  # "standard", "robust"
    handle_outliers: bool = True
    outlier_threshold: float = 3.0
    
    # Model evaluation
    classification_metrics: List[str] = None
    regression_metrics: List[str] = None
    
    def __post_init__(self):
        if self.classification_metrics is None:
            self.classification_metrics = ["accuracy", "precision", "recall", "f1"]
        
        if self.regression_metrics is None:
            self.regression_metrics = ["mse", "mae", "r2"]


class ModelUtils(LoggerMixin):
    """Utilities for model training and evaluation."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model utilities.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or ModelConfig()
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
        self.logger.info(f"Initialized ModelUtils with config: {self.config}")
    
    def prepare_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_preprocessors: bool = False
    ) -> pd.DataFrame:
        """Prepare features for model training.
        
        Args:
            X: Feature matrix
            y: Target variable
            fit_preprocessors: Whether to fit preprocessing transformers
            
        Returns:
            Preprocessed feature matrix
        """
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(method='ffill').fillna(method='bfill')
        X_processed = X_processed.fillna(0)
        
        # Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(0)
        
        # Remove constant features
        if fit_preprocessors:
            constant_features = X_processed.columns[X_processed.std() == 0]
            if len(constant_features) > 0:
                self.logger.info(f"Removing {len(constant_features)} constant features")
                X_processed = X_processed.drop(columns=constant_features)
        
        # Handle outliers
        if self.config.handle_outliers and fit_preprocessors:
            for col in X_processed.select_dtypes(include=[np.number]).columns:
                mean_val = X_processed[col].mean()
                std_val = X_processed[col].std()
                threshold = self.config.outlier_threshold
                
                outlier_mask = np.abs(X_processed[col] - mean_val) > threshold * std_val
                n_outliers = outlier_mask.sum()
                
                if n_outliers > 0:
                    # Cap outliers at threshold
                    X_processed.loc[outlier_mask & (X_processed[col] > mean_val), col] = (
                        mean_val + threshold * std_val
                    )
                    X_processed.loc[outlier_mask & (X_processed[col] < mean_val), col] = (
                        mean_val - threshold * std_val
                    )
        
        # Feature scaling
        if self.config.scale_features:
            if fit_preprocessors:
                if self.config.scaler_type == "standard":
                    self.scaler = StandardScaler()
                elif self.config.scaler_type == "robust":
                    self.scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
                
                X_processed = pd.DataFrame(
                    self.scaler.fit_transform(X_processed),
                    index=X_processed.index,
                    columns=X_processed.columns
                )
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Set fit_preprocessors=True first.")
                
                X_processed = pd.DataFrame(
                    self.scaler.transform(X_processed),
                    index=X_processed.index,
                    columns=X_processed.columns
                )
        
        # Feature selection
        if self.config.feature_selection:
            if fit_preprocessors:
                n_features = self.config.max_features or min(50, X_processed.shape[1])
                n_features = min(n_features, X_processed.shape[1])
                
                if self.config.feature_selection_method == "f_test":
                    self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
                elif self.config.feature_selection_method == "mutual_info":
                    self.feature_selector = SelectKBest(
                        score_func=mutual_info_regression, k=n_features
                    )
                else:
                    raise ValueError(f"Unknown feature selection method: {self.config.feature_selection_method}")
                
                # Align X and y indices
                common_idx = X_processed.index.intersection(y.index)
                X_aligned = X_processed.loc[common_idx]
                y_aligned = y.loc[common_idx]
                
                X_selected = self.feature_selector.fit_transform(X_aligned, y_aligned)
                self.selected_features = X_processed.columns[self.feature_selector.get_support()]
                
                X_processed = pd.DataFrame(
                    X_selected,
                    index=X_aligned.index,
                    columns=self.selected_features
                )
                
                self.logger.info(f"Selected {len(self.selected_features)} features out of {X.shape[1]}")
            else:
                if self.feature_selector is None or self.selected_features is None:
                    raise ValueError("Feature selector not fitted. Set fit_preprocessors=True first.")
                
                X_processed = X_processed[self.selected_features]
        
        return X_processed
    
    def create_time_series_splits(
        self,
        X: pd.DataFrame,
        n_splits: Optional[int] = None,
        test_size: Optional[float] = None,
        gap: Optional[int] = None
    ) -> TimeSeriesSplit:
        """Create time series cross-validation splits.
        
        Args:
            X: Feature matrix with datetime index
            n_splits: Number of splits (uses config default if None)
            test_size: Test size as fraction (uses config default if None)
            gap: Gap between train and test (uses config default if None)
            
        Returns:
            TimeSeriesSplit object
        """
        n_splits = n_splits or self.config.cv_folds
        test_size = test_size or self.config.test_size
        gap = gap or self.config.gap
        
        # Calculate test size in samples
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_samples,
            gap=gap
        )
        
        return tscv
    
    def evaluate_classification_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        if "accuracy" in self.config.classification_metrics:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        if "precision" in self.config.classification_metrics:
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        
        if "recall" in self.config.classification_metrics:
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        
        if "f1" in self.config.classification_metrics:
            metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # Additional metrics if probabilities are available
        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score, log_loss
                
                # Handle multiclass case
                if y_proba.shape[1] > 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                
                metrics["log_loss"] = log_loss(y_true, y_proba)
                
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC or log loss: {e}")
        
        return metrics
    
    def evaluate_regression_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        if "mse" in self.config.regression_metrics:
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
        
        if "mae" in self.config.regression_metrics:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
        
        if "r2" in self.config.regression_metrics:
            metrics["r2"] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        residuals = y_true - y_pred
        metrics["mean_residual"] = np.mean(residuals)
        metrics["std_residual"] = np.std(residuals)
        
        # Information Coefficient (IC) for alpha models
        if len(y_true) > 1:
            ic = np.corrcoef(y_true, y_pred)[0, 1]
            metrics["information_coefficient"] = ic if not np.isnan(ic) else 0.0
        
        return metrics
    
    def cross_validate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "neg_mean_squared_error",
        cv: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation on model.
        
        Args:
            model: Scikit-learn model
            X: Feature matrix
            y: Target variable
            scoring: Scoring metric
            cv: Number of CV folds (uses config default if None)
            
        Returns:
            Dictionary with cross-validation results
        """
        cv = cv or self.config.cv_folds
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Create time series splits
        tscv = self.create_time_series_splits(X_aligned)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_aligned, y_aligned,
            cv=tscv, scoring=scoring, n_jobs=-1
        )
        
        results = {
            "cv_scores": cv_scores,
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "scoring_metric": scoring
        }
        
        return results
    
    def grid_search_model(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "neg_mean_squared_error",
        cv: Optional[int] = None
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform grid search for hyperparameter optimization.
        
        Args:
            model: Scikit-learn model
            param_grid: Parameter grid for search
            X: Feature matrix
            y: Target variable
            scoring: Scoring metric
            cv: Number of CV folds (uses config default if None)
            
        Returns:
            Tuple of (best_model, search_results)
        """
        cv = cv or self.config.cv_folds
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Create time series splits
        tscv = self.create_time_series_splits(X_aligned)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid,
            cv=tscv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_aligned, y_aligned)
        
        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
        
        self.logger.info(f"Grid search completed. Best score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, results
    
    def calculate_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        method: str = "auto"
    ) -> pd.Series:
        """Calculate feature importance from trained model.
        
        Args:
            model: Trained scikit-learn model
            feature_names: List of feature names
            method: Method to extract importance ("auto", "coef", "importance", "permutation")
            
        Returns:
            Series with feature importance scores
        """
        if method == "auto":
            if hasattr(model, "feature_importances_"):
                method = "importance"
            elif hasattr(model, "coef_"):
                method = "coef"
            else:
                self.logger.warning("Model does not have feature_importances_ or coef_ attributes")
                return pd.Series(index=feature_names, dtype=float)
        
        if method == "importance" and hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif method == "coef" and hasattr(model, "coef_"):
            importance = np.abs(model.coef_).flatten()
        else:
            self.logger.warning(f"Cannot extract feature importance using method: {method}")
            return pd.Series(index=feature_names, dtype=float)
        
        # Ensure we have the right number of features
        if len(importance) != len(feature_names):
            self.logger.warning(f"Feature importance length ({len(importance)}) doesn't match feature names ({len(feature_names)})")
            min_len = min(len(importance), len(feature_names))
            importance = importance[:min_len]
            feature_names = feature_names[:min_len]
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    def analyze_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """Analyze model residuals for diagnostics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Optional timestamps for time series analysis
            
        Returns:
            Dictionary with residual analysis
        """
        residuals = y_true - y_pred
        
        analysis = {
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "min_residual": np.min(residuals),
            "max_residual": np.max(residuals),
            "residual_skewness": pd.Series(residuals).skew(),
            "residual_kurtosis": pd.Series(residuals).kurtosis()
        }
        
        # Normality test
        try:
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(residuals)
            analysis["jarque_bera_stat"] = jb_stat
            analysis["jarque_bera_pvalue"] = jb_pvalue
            analysis["residuals_normal"] = jb_pvalue > 0.05
        except ImportError:
            pass
        
        # Autocorrelation of residuals (if timestamps provided)
        if timestamps is not None and len(timestamps) == len(residuals):
            residual_series = pd.Series(residuals, index=timestamps)
            
            # Lag-1 autocorrelation
            autocorr_1 = residual_series.autocorr(lag=1)
            analysis["residual_autocorr_lag1"] = autocorr_1 if not np.isnan(autocorr_1) else 0.0
            
            # Durbin-Watson statistic (approximate)
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
            analysis["durbin_watson"] = dw_stat
        
        return analysis
    
    def get_model_summary(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Get comprehensive model summary.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with model summary
        """
        summary = {
            "model_type": type(model).__name__,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": list(X.columns)
        }
        
        # Model parameters
        if hasattr(model, "get_params"):
            summary["model_params"] = model.get_params()
        
        # Performance metrics
        if pd.api.types.is_numeric_dtype(y):
            # Regression metrics
            summary["performance"] = self.evaluate_regression_model(y.values, y_pred)
        else:
            # Classification metrics
            summary["performance"] = self.evaluate_classification_model(y.values, y_pred)
        
        # Feature importance
        try:
            feature_importance = self.calculate_feature_importance(model, list(X.columns))
            summary["feature_importance"] = feature_importance.to_dict()
            summary["top_features"] = feature_importance.head(10).index.tolist()
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {e}")
        
        # Residual analysis (for regression)
        if pd.api.types.is_numeric_dtype(y):
            summary["residual_analysis"] = self.analyze_residuals(
                y.values, y_pred, X.index if isinstance(X.index, pd.DatetimeIndex) else None
            )
        
        return summary
