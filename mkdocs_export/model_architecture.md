# Model Architecture and Machine Learning Framework

## Overview

The Cross-Asset Alpha Engine employs a sophisticated machine learning architecture that combines regime detection with adaptive alpha generation. The system uses multiple model types, each optimized for different aspects of market behavior and regime conditions.

## Regime Detection Models

### Hidden Markov Model (HMM) Implementation

#### Mathematical Foundation
The HMM assumes markets exist in K unobservable states (regimes) with transition dynamics:

```python
from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import StandardScaler

class RegimeHMM:
    """Advanced Hidden Markov Model for regime detection."""
    
    def __init__(self, n_components=3, covariance_type="full", n_iter=100):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.regime_labels = {
            0: "Low Volatility",
            1: "High Volatility", 
            2: "Transition"
        }
    
    def fit(self, features):
        """Fit HMM to regime detection features."""
        # Standardize features for numerical stability
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM with multiple random initializations
        best_score = -np.inf
        best_model = None
        
        for seed in range(5):
            temp_model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.model.covariance_type,
                n_iter=self.model.n_iter,
                random_state=seed
            )
            
            try:
                temp_model.fit(features_scaled)
                score = temp_model.score(features_scaled)
                
                if score > best_score:
                    best_score = score
                    best_model = temp_model
            except:
                continue
        
        self.model = best_model if best_model else self.model
        return self
    
    def predict_regimes(self, features):
        """Predict most likely regime sequence using Viterbi algorithm."""
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def predict_proba(self, features):
        """Predict regime probabilities using forward-backward algorithm."""
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def get_regime_characteristics(self, features, regimes):
        """Analyze characteristics of each regime."""
        characteristics = {}
        
        for regime in range(self.n_components):
            regime_mask = regimes == regime
            regime_data = features[regime_mask]
            
            characteristics[self.regime_labels[regime]] = {
                'observations': len(regime_data),
                'percentage': len(regime_data) / len(features) * 100,
                'mean_volatility': regime_data['volatility_20d'].mean() if 'volatility_20d' in regime_data.columns else None,
                'mean_vix': regime_data['vix_level'].mean() if 'vix_level' in regime_data.columns else None,
                'mean_correlation': regime_data['equity_bond_corr'].mean() if 'equity_bond_corr' in regime_data.columns else None
            }
        
        return characteristics
```

#### Regime Feature Selection
Key features for regime detection:

```python
def prepare_regime_features(data):
    """Prepare features specifically for regime detection."""
    
    regime_features = pd.DataFrame(index=data.index)
    
    # Volatility measures
    regime_features['volatility_20d'] = data.groupby('symbol')['returns_1d'].rolling(20).std().reset_index(0, drop=True) * np.sqrt(252)
    regime_features['vol_ratio_5_20'] = (
        data.groupby('symbol')['returns_1d'].rolling(5).std().reset_index(0, drop=True) /
        data.groupby('symbol')['returns_1d'].rolling(20).std().reset_index(0, drop=True)
    )
    
    # Cross-asset correlations
    equity_returns = data[data['symbol'].isin(['SPY', 'QQQ'])]['returns_1d']
    bond_returns = data[data['symbol'] == 'TLT']['returns_1d']
    
    if len(equity_returns) > 0 and len(bond_returns) > 0:
        regime_features['equity_bond_corr'] = equity_returns.rolling(20).corr(bond_returns)
    
    # VIX level (if available)
    vix_data = data[data['symbol'] == 'VIX']
    if len(vix_data) > 0:
        regime_features['vix_level'] = vix_data['close']
        regime_features['vix_change'] = vix_data['returns_1d']
    
    # Volume patterns
    regime_features['volume_zscore'] = data.groupby('symbol').apply(
        lambda x: (x['volume'] - x['volume'].rolling(20).mean()) / x['volume'].rolling(20).std()
    ).reset_index(0, drop=True)
    
    return regime_features.dropna()
```

### Statistical Regime Detection

#### Threshold-Based Models
```python
class ThresholdRegimeDetector:
    """Statistical threshold-based regime detection."""
    
    def __init__(self, volatility_thresholds=[0.15, 0.25], vix_thresholds=[20, 30]):
        self.vol_thresholds = volatility_thresholds
        self.vix_thresholds = vix_thresholds
    
    def detect_regimes(self, features):
        """Detect regimes using statistical thresholds."""
        regimes = np.zeros(len(features))
        
        # Volatility-based regime detection
        vol_regimes = pd.cut(
            features['volatility_20d'], 
            bins=[0] + self.vol_thresholds + [np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        # VIX-based regime detection (if available)
        if 'vix_level' in features.columns:
            vix_regimes = pd.cut(
                features['vix_level'],
                bins=[0] + self.vix_thresholds + [np.inf],
                labels=[0, 1, 2]
            ).astype(int)
            
            # Combined regime (average of indicators)
            regimes = np.round((vol_regimes + vix_regimes) / 2).astype(int)
        else:
            regimes = vol_regimes
        
        return regimes
```

## Alpha Generation Models

### Random Forest Architecture

#### Core Implementation
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import shap

class AlphaModel:
    """Regime-aware Random Forest alpha generation model."""
    
    def __init__(self, model_config=None):
        self.config = model_config or {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.models = {}
        self.feature_importance = {}
        self.shap_explainers = {}
    
    def fit(self, features, targets, regimes=None, regime_probs=None):
        """Train regime-specific and overall models."""
        
        # Overall model (baseline)
        self.models['overall'] = RandomForestRegressor(**self.config)
        self.models['overall'].fit(features, targets)
        
        # Store overall feature importance
        self.feature_importance['overall'] = pd.DataFrame({
            'feature': features.columns,
            'importance': self.models['overall'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Regime-specific models
        if regimes is not None:
            for regime in np.unique(regimes):
                regime_mask = regimes == regime
                
                # Require minimum samples for regime-specific model
                if regime_mask.sum() > 100:
                    regime_features = features[regime_mask]
                    regime_targets = targets[regime_mask]
                    
                    # Train regime-specific model
                    model_key = f'regime_{regime}'
                    self.models[model_key] = RandomForestRegressor(**self.config)
                    self.models[model_key].fit(regime_features, regime_targets)
                    
                    # Store feature importance
                    self.feature_importance[model_key] = pd.DataFrame({
                        'feature': features.columns,
                        'importance': self.models[model_key].feature_importances_
                    }).sort_values('importance', ascending=False)
        
        # Initialize SHAP explainers
        self._initialize_shap_explainers(features.sample(min(1000, len(features))))
        
        return self
    
    def predict(self, features, regimes=None, regime_probs=None):
        """Generate alpha predictions with regime awareness."""
        
        if regime_probs is not None and regime_probs.shape[1] > 1:
            # Ensemble prediction weighted by regime probabilities
            predictions = np.zeros(len(features))
            
            for regime in range(regime_probs.shape[1]):
                model_key = f'regime_{regime}'
                if model_key in self.models:
                    regime_preds = self.models[model_key].predict(features)
                    predictions += regime_probs[:, regime] * regime_preds
                else:
                    # Fallback to overall model
                    regime_preds = self.models['overall'].predict(features)
                    predictions += regime_probs[:, regime] * regime_preds
            
            return predictions
            
        elif regimes is not None:
            # Use most likely regime for each prediction
            predictions = np.zeros(len(features))
            
            for i, regime in enumerate(regimes):
                model_key = f'regime_{regime}'
                if model_key in self.models:
                    predictions[i] = self.models[model_key].predict(features.iloc[[i]])[0]
                else:
                    predictions[i] = self.models['overall'].predict(features.iloc[[i]])[0]
            
            return predictions
        
        else:
            # Use overall model
            return self.models['overall'].predict(features)
    
    def _initialize_shap_explainers(self, sample_features):
        """Initialize SHAP explainers for model interpretability."""
        for model_name, model in self.models.items():
            try:
                self.shap_explainers[model_name] = shap.TreeExplainer(model)
            except:
                pass  # Skip if SHAP fails
    
    def explain_predictions(self, features, model_name='overall'):
        """Generate SHAP explanations for predictions."""
        if model_name in self.shap_explainers:
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(features)
            return shap_values
        else:
            return None
    
    def get_feature_importance(self, model_name='overall', top_n=20):
        """Get top feature importance for specified model."""
        if model_name in self.feature_importance:
            return self.feature_importance[model_name].head(top_n)
        else:
            return None
```

#### Model Validation and Selection
```python
class ModelValidator:
    """Comprehensive model validation for time series data."""
    
    def __init__(self, n_splits=5, test_size=63):  # ~3 months test
        self.n_splits = n_splits
        self.test_size = test_size
    
    def walk_forward_validation(self, features, targets, model_class, model_config):
        """Perform walk-forward validation."""
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        validation_results = {
            'train_scores': [],
            'test_scores': [],
            'predictions': [],
            'feature_importance': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            # Split data
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
            
            # Train model
            model = model_class(**model_config)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Predictions
            test_predictions = model.predict(X_test)
            
            # Store results
            validation_results['train_scores'].append(train_score)
            validation_results['test_scores'].append(test_score)
            validation_results['predictions'].append({
                'fold': fold,
                'actual': y_test.values,
                'predicted': test_predictions,
                'dates': X_test.index
            })
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': features.columns,
                    'importance': model.feature_importances_,
                    'fold': fold
                })
                validation_results['feature_importance'].append(importance_df)
        
        return validation_results
    
    def calculate_validation_metrics(self, validation_results):
        """Calculate comprehensive validation metrics."""
        
        # Aggregate predictions
        all_actual = []
        all_predicted = []
        
        for pred_result in validation_results['predictions']:
            all_actual.extend(pred_result['actual'])
            all_predicted.extend(pred_result['predicted'])
        
        all_actual = np.array(all_actual)
        all_predicted = np.array(all_predicted)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mean_train_score': np.mean(validation_results['train_scores']),
            'std_train_score': np.std(validation_results['train_scores']),
            'mean_test_score': np.mean(validation_results['test_scores']),
            'std_test_score': np.std(validation_results['test_scores']),
            'mse': mean_squared_error(all_actual, all_predicted),
            'mae': mean_absolute_error(all_actual, all_predicted),
            'r2': r2_score(all_actual, all_predicted),
            'information_coefficient': np.corrcoef(all_actual, all_predicted)[0, 1]
        }
        
        return metrics
```

### Alternative Model Architectures

#### Logistic Regression (Interpretable Model)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class InterpretableAlphaModel:
    """Logistic regression model for interpretable alpha generation."""
    
    def __init__(self, penalty='l1', C=0.1, solver='liblinear'):
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_coefficients = None
    
    def fit(self, features, targets):
        """Fit logistic regression model."""
        # Convert regression targets to classification (directional prediction)
        binary_targets = (targets > targets.median()).astype(int)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(features_scaled, binary_targets)
        
        # Store coefficients for interpretation
        self.feature_coefficients = pd.DataFrame({
            'feature': features.columns,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return self
    
    def predict_proba(self, features):
        """Predict probability of positive return."""
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]
    
    def get_feature_interpretation(self):
        """Get interpretable feature coefficients."""
        return self.feature_coefficients
```

#### Support Vector Machine (Non-Linear Patterns)
```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

class NonLinearAlphaModel:
    """Support Vector Machine for non-linear pattern recognition."""
    
    def __init__(self, kernel='rbf', param_grid=None):
        self.kernel = kernel
        self.param_grid = param_grid or {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'epsilon': [0.01, 0.1, 0.2]
        }
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, features, targets):
        """Fit SVM with hyperparameter optimization."""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Grid search for optimal parameters
        svm = SVR(kernel=self.kernel)
        grid_search = GridSearchCV(
            svm, 
            self.param_grid, 
            cv=3, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(features_scaled, targets)
        
        # Store best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        return self
    
    def predict(self, features):
        """Generate predictions using fitted SVM."""
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
```

## Model Ensemble and Meta-Learning

### Ensemble Architecture
```python
class AlphaEnsemble:
    """Ensemble of multiple alpha models with dynamic weighting."""
    
    def __init__(self, models_config):
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        
        # Initialize individual models
        for name, config in models_config.items():
            if config['type'] == 'random_forest':
                self.models[name] = AlphaModel(config['params'])
            elif config['type'] == 'logistic':
                self.models[name] = InterpretableAlphaModel(**config['params'])
            elif config['type'] == 'svm':
                self.models[name] = NonLinearAlphaModel(**config['params'])
    
    def fit(self, features, targets, regimes=None):
        """Train all models in the ensemble."""
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit'):
                    if 'regime' in name.lower() and regimes is not None:
                        model.fit(features, targets, regimes)
                    else:
                        model.fit(features, targets)
                
                # Initialize equal weights
                self.model_weights[name] = 1.0 / len(self.models)
                
            except Exception as e:
                print(f"Failed to train model {name}: {e}")
                self.model_weights[name] = 0.0
        
        return self
    
    def predict(self, features, regimes=None, regime_probs=None):
        """Generate ensemble predictions with dynamic weighting."""
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    if 'regime' in name.lower() and regimes is not None:
                        pred = model.predict(features, regimes, regime_probs)
                    else:
                        pred = model.predict(features)
                    predictions[name] = pred
                elif hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(features)
                    predictions[name] = pred
            except:
                predictions[name] = np.zeros(len(features))
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(features))
        total_weight = sum(self.model_weights.values())
        
        for name, pred in predictions.items():
            weight = self.model_weights[name] / total_weight
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def update_weights(self, recent_performance):
        """Update model weights based on recent performance."""
        
        # Softmax weighting based on performance
        performance_scores = np.array(list(recent_performance.values()))
        exp_scores = np.exp(performance_scores - np.max(performance_scores))
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Update weights
        for i, name in enumerate(recent_performance.keys()):
            self.model_weights[name] = softmax_weights[i]
```

## Model Performance Monitoring

### Real-Time Performance Tracking
```python
class ModelPerformanceMonitor:
    """Monitor model performance in real-time."""
    
    def __init__(self, lookback_window=63):  # ~3 months
        self.lookback_window = lookback_window
        self.performance_history = {}
        self.alerts = []
    
    def update_performance(self, model_name, predictions, actual_returns, dates):
        """Update performance metrics for a model."""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                'predictions': [],
                'returns': [],
                'dates': [],
                'ic_history': [],
                'sharpe_history': []
            }
        
        # Store recent data
        history = self.performance_history[model_name]
        history['predictions'].extend(predictions)
        history['returns'].extend(actual_returns)
        history['dates'].extend(dates)
        
        # Keep only recent data
        if len(history['predictions']) > self.lookback_window:
            history['predictions'] = history['predictions'][-self.lookback_window:]
            history['returns'] = history['returns'][-self.lookback_window:]
            history['dates'] = history['dates'][-self.lookback_window:]
        
        # Calculate rolling metrics
        if len(history['predictions']) >= 20:  # Minimum for meaningful metrics
            ic = np.corrcoef(history['predictions'], history['returns'])[0, 1]
            
            # Convert predictions to portfolio returns (simplified)
            portfolio_returns = np.array(history['predictions']) * np.array(history['returns'])
            sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
            
            history['ic_history'].append(ic)
            history['sharpe_history'].append(sharpe)
            
            # Check for performance degradation
            self._check_performance_alerts(model_name, ic, sharpe)
    
    def _check_performance_alerts(self, model_name, current_ic, current_sharpe):
        """Check for performance degradation alerts."""
        
        # Alert thresholds
        min_ic_threshold = 0.02
        min_sharpe_threshold = 0.5
        
        if current_ic < min_ic_threshold:
            alert = {
                'timestamp': pd.Timestamp.now(),
                'model': model_name,
                'type': 'low_ic',
                'value': current_ic,
                'threshold': min_ic_threshold
            }
            self.alerts.append(alert)
        
        if current_sharpe < min_sharpe_threshold:
            alert = {
                'timestamp': pd.Timestamp.now(),
                'model': model_name,
                'type': 'low_sharpe',
                'value': current_sharpe,
                'threshold': min_sharpe_threshold
            }
            self.alerts.append(alert)
    
    def get_current_performance(self, model_name):
        """Get current performance metrics for a model."""
        
        if model_name not in self.performance_history:
            return None
        
        history = self.performance_history[model_name]
        
        if len(history['ic_history']) == 0:
            return None
        
        return {
            'current_ic': history['ic_history'][-1],
            'current_sharpe': history['sharpe_history'][-1],
            'avg_ic_30d': np.mean(history['ic_history'][-30:]) if len(history['ic_history']) >= 30 else None,
            'avg_sharpe_30d': np.mean(history['sharpe_history'][-30:]) if len(history['sharpe_history']) >= 30 else None
        }
```

This comprehensive model architecture ensures the Cross-Asset Alpha Engine can adapt to changing market conditions while maintaining robust performance monitoring and interpretability.
