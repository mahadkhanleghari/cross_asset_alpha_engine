"""Hidden Markov Model for regime detection.

This module implements a Hidden Markov Model using hmmlearn for detecting
market regimes based on multiple features.
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Please install hmmlearn and scikit-learn.")

from ..config import DEFAULT_N_REGIMES
from ..utils.logging_utils import LoggerMixin


@dataclass
class HMMConfig:
    """Configuration for Hidden Markov Model regime detection."""
    n_regimes: int = DEFAULT_N_REGIMES
    covariance_type: str = "full"  # "full", "diag", "tied", "spherical"
    n_iter: int = 100
    random_state: int = 42
    tol: float = 1e-2
    algorithm: str = "viterbi"  # "viterbi" or "map"
    
    # Feature preprocessing
    standardize_features: bool = True
    use_pca: bool = False
    pca_components: Optional[int] = None
    pca_variance_threshold: float = 0.95
    
    # Model selection
    min_regimes: int = 2
    max_regimes: int = 5
    selection_criterion: str = "aic"  # "aic", "bic", "log_likelihood"


class RegimeHMM(LoggerMixin):
    """Hidden Markov Model for market regime detection."""
    
    def __init__(self, config: Optional[HMMConfig] = None):
        """Initialize HMM regime model.
        
        Args:
            config: HMM configuration (uses defaults if None)
        """
        self.config = config or HMMConfig()
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.is_fitted = False
        
        self.logger.info(f"Initialized RegimeHMM with {self.config.n_regimes} regimes")
    
    def _preprocess_features(
        self,
        X: pd.DataFrame,
        fit_preprocessors: bool = False
    ) -> np.ndarray:
        """Preprocess features for HMM training.
        
        Args:
            X: Feature matrix
            fit_preprocessors: Whether to fit preprocessing transformers
            
        Returns:
            Preprocessed feature matrix
        """
        # Handle missing values
        X_processed = X.fillna(method='ffill').fillna(method='bfill')
        
        # Remove infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(0)
        
        # Standardization
        if self.config.standardize_features:
            if fit_preprocessors:
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X_processed)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call fit() first.")
                X_processed = self.scaler.transform(X_processed)
        else:
            X_processed = X_processed.values
        
        # PCA dimensionality reduction
        if self.config.use_pca:
            if fit_preprocessors:
                n_components = self.config.pca_components
                if n_components is None:
                    # Determine components based on variance threshold
                    pca_temp = PCA()
                    pca_temp.fit(X_processed)
                    cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
                    n_components = np.argmax(cumsum_var >= self.config.pca_variance_threshold) + 1
                
                self.pca = PCA(n_components=n_components, random_state=self.config.random_state)
                X_processed = self.pca.fit_transform(X_processed)
                
                self.logger.info(f"PCA reduced features from {X.shape[1]} to {n_components}")
                self.logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
            else:
                if self.pca is None:
                    raise ValueError("PCA not fitted. Call fit() first.")
                X_processed = self.pca.transform(X_processed)
        
        return X_processed
    
    def _create_hmm_model(self, n_regimes: int) -> hmm.GaussianHMM:
        """Create HMM model with specified configuration.
        
        Args:
            n_regimes: Number of regimes
            
        Returns:
            Configured HMM model
        """
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            tol=self.config.tol,
            algorithm=self.config.algorithm
        )
        
        return model
    
    def _calculate_information_criterion(
        self,
        model: hmm.GaussianHMM,
        X: np.ndarray,
        criterion: str = "aic"
    ) -> float:
        """Calculate information criterion for model selection.
        
        Args:
            model: Fitted HMM model
            X: Feature matrix
            criterion: Information criterion ("aic" or "bic")
            
        Returns:
            Information criterion value
        """
        log_likelihood = model.score(X)
        n_params = self._count_parameters(model)
        n_samples = X.shape[0]
        
        if criterion == "aic":
            return -2 * log_likelihood + 2 * n_params
        elif criterion == "bic":
            return -2 * log_likelihood + n_params * np.log(n_samples)
        elif criterion == "log_likelihood":
            return -log_likelihood  # Return negative for minimization
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def _count_parameters(self, model: hmm.GaussianHMM) -> int:
        """Count number of parameters in HMM model.
        
        Args:
            model: HMM model
            
        Returns:
            Number of parameters
        """
        n_states = model.n_components
        n_features = model.means_.shape[1]
        
        # Transition matrix parameters (n_states * (n_states - 1))
        transition_params = n_states * (n_states - 1)
        
        # Initial state distribution parameters (n_states - 1)
        initial_params = n_states - 1
        
        # Emission parameters (means and covariances)
        means_params = n_states * n_features
        
        if model.covariance_type == "full":
            # Full covariance matrix for each state
            cov_params = n_states * n_features * (n_features + 1) // 2
        elif model.covariance_type == "diag":
            # Diagonal covariance matrix for each state
            cov_params = n_states * n_features
        elif model.covariance_type == "tied":
            # Single covariance matrix for all states
            cov_params = n_features * (n_features + 1) // 2
        elif model.covariance_type == "spherical":
            # Single variance parameter for each state
            cov_params = n_states
        else:
            cov_params = 0
        
        return transition_params + initial_params + means_params + cov_params
    
    def fit(
        self,
        X: pd.DataFrame,
        auto_select_regimes: bool = False
    ) -> 'RegimeHMM':
        """Fit HMM model to feature data.
        
        Args:
            X: Feature matrix with datetime index
            auto_select_regimes: Whether to automatically select number of regimes
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting HMM model with {len(X)} observations and {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Preprocess features
        X_processed = self._preprocess_features(X, fit_preprocessors=True)
        
        if auto_select_regimes:
            self.logger.info("Auto-selecting optimal number of regimes")
            best_score = np.inf
            best_model = None
            best_n_regimes = self.config.n_regimes
            
            scores = {}
            
            for n_regimes in range(self.config.min_regimes, self.config.max_regimes + 1):
                try:
                    model = self._create_hmm_model(n_regimes)
                    model.fit(X_processed)
                    
                    score = self._calculate_information_criterion(
                        model, X_processed, self.config.selection_criterion
                    )
                    scores[n_regimes] = score
                    
                    self.logger.info(f"Regimes: {n_regimes}, {self.config.selection_criterion.upper()}: {score:.2f}")
                    
                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_n_regimes = n_regimes
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fit model with {n_regimes} regimes: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("Failed to fit any HMM model")
            
            self.model = best_model
            self.config.n_regimes = best_n_regimes
            self.logger.info(f"Selected {best_n_regimes} regimes with {self.config.selection_criterion.upper()}: {best_score:.2f}")
            
        else:
            # Fit model with specified number of regimes
            self.model = self._create_hmm_model(self.config.n_regimes)
            self.model.fit(X_processed)
            
            score = self._calculate_information_criterion(
                self.model, X_processed, self.config.selection_criterion
            )
            self.logger.info(f"Fitted HMM with {self.config.n_regimes} regimes, {self.config.selection_criterion.upper()}: {score:.2f}")
        
        self.is_fitted = True
        return self
    
    def predict_regimes(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime sequence for given features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted regimes
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_processed = self._preprocess_features(X, fit_preprocessors=False)
        regimes = self.model.predict(X_processed)
        
        return regimes
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for given features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of regime probabilities (n_samples, n_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_processed = self._preprocess_features(X, fit_preprocessors=False)
        
        # Use forward-backward algorithm to get state probabilities
        log_proba = self.model.predict_proba(X_processed)
        
        return log_proba
    
    def get_regime_statistics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about detected regimes.
        
        Args:
            X: Feature matrix used for regime detection
            
        Returns:
            Dictionary with regime statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        regimes = self.predict_regimes(X)
        regime_probs = self.predict_proba(X)
        
        stats = {
            "n_regimes": self.config.n_regimes,
            "regime_counts": pd.Series(regimes).value_counts().sort_index().to_dict(),
            "regime_frequencies": pd.Series(regimes).value_counts(normalize=True).sort_index().to_dict(),
            "avg_regime_probabilities": regime_probs.mean(axis=0).tolist(),
            "transition_matrix": self.model.transmat_.tolist(),
            "initial_distribution": self.model.startprob_.tolist()
        }
        
        # Regime characteristics (means of features in each regime)
        if hasattr(self.model, 'means_'):
            regime_means = {}
            X_processed = self._preprocess_features(X, fit_preprocessors=False)
            
            for regime in range(self.config.n_regimes):
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    if self.config.use_pca:
                        # Transform back from PCA space (approximate)
                        regime_mean_pca = X_processed[regime_mask].mean(axis=0)
                        if self.pca is not None:
                            regime_mean = self.pca.inverse_transform(regime_mean_pca.reshape(1, -1))[0]
                        else:
                            regime_mean = regime_mean_pca
                    else:
                        regime_mean = X_processed[regime_mask].mean(axis=0)
                    
                    if self.scaler is not None and self.config.standardize_features:
                        regime_mean = self.scaler.inverse_transform(regime_mean.reshape(1, -1))[0]
                    
                    regime_means[f"regime_{regime}"] = regime_mean.tolist()
            
            stats["regime_feature_means"] = regime_means
        
        return stats
    
    def get_regime_transitions(self, regimes: np.ndarray) -> pd.DataFrame:
        """Analyze regime transitions.
        
        Args:
            regimes: Array of regime predictions
            
        Returns:
            DataFrame with transition analysis
        """
        transitions = []
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions.append({
                    "from_regime": regimes[i-1],
                    "to_regime": regimes[i],
                    "position": i
                })
        
        if not transitions:
            return pd.DataFrame()
        
        df_transitions = pd.DataFrame(transitions)
        
        # Add transition statistics
        transition_counts = df_transitions.groupby(["from_regime", "to_regime"]).size().reset_index(name="count")
        
        return transition_counts
    
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
            "scaler": self.scaler,
            "pca": self.pca,
            "config": self.config,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'RegimeHMM':
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
        self.scaler = model_data["scaler"]
        self.pca = model_data["pca"]
        self.config = model_data["config"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = model_data["is_fitted"]
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of fitted model.
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        summary = {
            "status": "fitted",
            "n_regimes": self.config.n_regimes,
            "covariance_type": self.config.covariance_type,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "feature_names": self.feature_names,
            "standardized": self.config.standardize_features,
            "pca_used": self.config.use_pca,
            "n_parameters": self._count_parameters(self.model) if self.model else None
        }
        
        if self.config.use_pca and self.pca is not None:
            summary["pca_components"] = self.pca.n_components_
            summary["pca_explained_variance"] = self.pca.explained_variance_ratio_.sum()
        
        return summary
