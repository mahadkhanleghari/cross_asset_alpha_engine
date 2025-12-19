# Feature Engineering Framework

**IMPORTANT: All features are computed from daily OHLCV bars. No intraday, tick, or order-book data is used in the current experiment.**

## Overview

The Cross-Asset Alpha Engine generates over 40 sophisticated features designed to capture different aspects of market behavior across technical analysis, daily microstructure-inspired patterns, and cross-asset relationships. All features are computed from daily OHLCV (Open, High, Low, Close, Volume) bars from Polygon.io.

## Feature Categories

### Technical Analysis Features (Price-Based Signals)

#### Momentum Indicators
Multi-timeframe momentum captures trends across different horizons:

```python
# Multi-horizon returns
for period in [1, 5, 20, 60]:
    features[f'returns_{period}d'] = data['close'].pct_change(period)

# Momentum acceleration
features['momentum_accel'] = features['returns_5d'] - features['returns_20d']

# Relative strength
features['relative_strength'] = features['returns_20d'] / features['returns_60d']
```

**Key Momentum Features**:
- **1d, 5d, 20d, 60d Returns**: Capturing short to medium-term trends
- **Momentum Ratios**: Price/SMA ratios identifying trend strength
- **Rate of Change**: Acceleration/deceleration in price movements
- **Momentum Oscillators**: Custom indicators with regime-dependent thresholds

#### Volatility Analysis
Volatility clustering and regime identification:

```python
# Multi-horizon volatility
for window in [5, 20, 60]:
    vol = returns.rolling(window).std() * np.sqrt(252)
    features[f'volatility_{window}d'] = vol

# Volatility ratios for regime detection
features['vol_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']

# Volatility persistence
features['vol_persistence'] = features['volatility_5d'].rolling(5).mean()
```

**Volatility Features**:
- **Realized Volatility**: Rolling standard deviations across multiple windows
- **Volatility Ratios**: Short-term vs long-term volatility relationships
- **GARCH Effects**: Volatility clustering and persistence modeling
- **Volatility Breakouts**: Statistical significance of volatility changes

#### Mean Reversion Signals
Statistical measures of price extremes:

```python
# Bollinger Band position
sma_20 = data['close'].rolling(20).mean()
bb_std = data['close'].rolling(20).std()
features['bb_position'] = (data['close'] - sma_20) / (2 * bb_std)

# RSI calculation
gains = returns.where(returns > 0, 0)
losses = -returns.where(returns < 0, 0)
rs = gains.rolling(14).mean() / losses.rolling(14).mean()
features['rsi'] = 100 - (100 / (1 + rs))

# Z-score analysis
features['price_zscore'] = (data['close'] - data['close'].rolling(60).mean()) / data['close'].rolling(60).std()
```

**Mean Reversion Features**:
- **Bollinger Band Position**: Standardized position within volatility bands
- **RSI Variations**: Multiple RSI calculations with different lookback periods
- **Z-Score Analysis**: Price deviations from historical means
- **Reversion Strength**: Magnitude and persistence of mean-reverting moves

### Daily Microstructure-Inspired Features (Volume and Daily Patterns)

**All features in this section are computed from daily OHLCV bars. No intraday or tick data is used.**

#### Volume Analysis
Daily trading activity patterns computed from daily volume data:

```python
# Volume z-score
vol_mean = data['volume'].rolling(20).mean()
vol_std = data['volume'].rolling(20).std()
features['volume_zscore'] = (data['volume'] - vol_mean) / vol_std

# Volume-price correlation
features['vol_price_corr'] = data['volume'].rolling(20).corr(returns)

# Accumulation/distribution
features['acc_dist'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']
```

**Volume Features**:
- **Volume Z-Scores**: Standardized volume relative to historical patterns
- **Volume-Price Relationship**: Correlation between volume and price changes
- **Accumulation/Distribution**: Net buying/selling pressure indicators
- **Volume Clustering**: Persistence in high/low volume periods

#### VWAP Analysis
Institutional trading patterns and execution quality:

```python
# VWAP deviation
features['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']

# VWAP momentum
features['vwap_momentum'] = data['vwap'].pct_change(5)

# Price improvement vs VWAP
features['price_improvement'] = np.where(
    data['close'] > data['vwap'], 1, -1
) * features['vwap_deviation'].abs()
```

**VWAP Features**:
- **VWAP Deviations**: Price distance from volume-weighted average price
- **VWAP Momentum**: Rate of change in VWAP relative to price
- **Institutional Activity**: Large block trading detection through VWAP analysis
- **Execution Quality**: Price improvement/deterioration vs VWAP

#### Daily Price Patterns (Computed from OHLCV)
Daily price dynamics computed from daily OHLCV bars:

```python
# Gap analysis (computed from daily bars)
features['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)

# Daily range analysis (from daily high-low)
features['daily_range'] = (data['high'] - data['low']) / data['close']
features['range_zscore'] = (features['daily_range'] - features['daily_range'].rolling(20).mean()) / features['daily_range'].rolling(20).std()

# Daily open-to-close returns (computed from daily bars)
features['intraday_return'] = (data['close'] - data['open']) / data['open']
```

**Daily Price Pattern Features** (computed from daily OHLCV):
- **Gap Analysis**: Overnight price gaps (daily open vs previous close) and their subsequent behavior
- **Range Analysis**: Daily high-low ranges relative to historical norms
- **Daily Open-to-Close Returns**: Computed from daily OHLCV bars (not true intraday data)
- **Note**: True intraday patterns and time-of-day effects require intraday data, which is not used in the current experiment

### Cross-Asset Features (Inter-Market Relationships)

#### Correlation Dynamics
Inter-market relationships and regime detection:

```python
# Rolling correlations between asset classes
equity_returns = equity_data['close'].pct_change()
bond_returns = bond_data['close'].pct_change()

features['equity_bond_corr'] = equity_returns.rolling(20).corr(bond_returns)
features['equity_vix_corr'] = equity_returns.rolling(20).corr(vix_changes)

# Correlation breakdown detection
features['corr_change'] = features['equity_bond_corr'] - features['equity_bond_corr'].rolling(60).mean()
```

**Correlation Features**:
- **Rolling Correlations**: Dynamic correlation tracking between asset classes
- **Correlation Breakdowns**: Statistical significance of correlation changes
- **Lead-Lag Relationships**: Which assets lead/follow in price movements
- **Correlation Clustering**: Periods of high/low correlation across markets

#### Risk Sentiment Indicators
Market-wide risk appetite and flight-to-quality:

```python
# Risk-on/risk-off sentiment
features['risk_sentiment'] = -vix_changes + bond_returns

# Flight-to-quality indicator
features['flight_to_quality'] = (
    (bond_returns > bond_returns.quantile(0.8)) & 
    (equity_returns < 0)
).astype(int)

# Currency strength impact
features['dollar_strength'] = dxy_returns.rolling(5).mean()
```

**Risk Sentiment Features**:
- **VIX-Equity Relationship**: Fear gauge vs equity market behavior
- **Flight-to-Quality**: Treasury bond performance during equity stress
- **Risk Parity Signals**: Balanced risk allocation across asset classes
- **Currency Strength**: Dollar impact on multinational corporations

#### Regime-Dependent Features
Features that change behavior across market regimes:

```python
# Conditional correlations by volatility regime
high_vol_mask = volatility > volatility.quantile(0.7)
features['corr_high_vol'] = equity_bond_corr.where(high_vol_mask, np.nan)
features['corr_low_vol'] = equity_bond_corr.where(~high_vol_mask, np.nan)

# Volatility spillovers
features['vol_spillover'] = equity_vol.rolling(5).corr(bond_vol.rolling(5))

# Crisis indicators
features['crisis_indicator'] = (
    (vix_level > vix_level.quantile(0.9)) & 
    (equity_returns < equity_returns.quantile(0.1))
).astype(int)
```

**Regime Features**:
- **Conditional Correlations**: Asset relationships that change by regime
- **Volatility Spillovers**: How volatility transmits across asset classes
- **Crisis Indicators**: Early warning signals for market stress
- **Recovery Patterns**: Post-crisis market behavior characteristics

## Advanced Feature Engineering Techniques

### Feature Transformation

#### Normalization and Scaling
```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# Z-score normalization for stationarity
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Robust scaling for outlier resistance
robust_scaler = RobustScaler()
features_robust = robust_scaler.fit_transform(features)

# Rank transformation for non-parametric relationships
features_ranked = features.rank(pct=True)
```

#### Handling Non-Stationarity
```python
# Differencing for trend removal
features_diff = features.diff()

# Log transformation for skewed distributions
features_log = np.log(features.abs() + 1e-8) * np.sign(features)

# Detrending using linear regression
from scipy import signal
features_detrended = signal.detrend(features, axis=0)
```

### Feature Interaction and Engineering

#### Polynomial and Interaction Features
```python
from sklearn.preprocessing import PolynomialFeatures

# Create interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
features_poly = poly.fit_transform(features[['momentum_5d', 'volatility_20d', 'volume_zscore']])

# Custom interaction features
features['momentum_vol_interaction'] = features['momentum_5d'] * features['volatility_20d']
features['volume_price_interaction'] = features['volume_zscore'] * features['returns_1d']
```

#### Regime-Conditional Features
```python
# Features that activate only in specific regimes
def create_regime_features(features, regimes):
    regime_features = features.copy()
    
    for regime in np.unique(regimes):
        regime_mask = regimes == regime
        
        # Regime-specific momentum
        regime_features[f'momentum_regime_{regime}'] = (
            features['momentum_5d'].where(regime_mask, 0)
        )
        
        # Regime-specific volatility
        regime_features[f'volatility_regime_{regime}'] = (
            features['volatility_20d'].where(regime_mask, 0)
        )
    
    return regime_features
```

### Feature Selection and Validation

#### Statistical Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Univariate feature selection
selector_f = SelectKBest(score_func=f_regression, k=20)
features_selected_f = selector_f.fit_transform(features, targets)

# Mutual information based selection
selector_mi = SelectKBest(score_func=mutual_info_regression, k=20)
features_selected_mi = selector_mi.fit_transform(features, targets)

# Get feature scores
feature_scores = pd.DataFrame({
    'feature': features.columns,
    'f_score': selector_f.scores_,
    'mi_score': selector_mi.scores_
}).sort_values('f_score', ascending=False)
```

#### Recursive Feature Elimination
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Recursive feature elimination
estimator = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=25, step=1)
features_rfe = rfe.fit_transform(features, targets)

# Get selected features
selected_features = features.columns[rfe.support_]
print(f"Selected features: {list(selected_features)}")
```

#### Feature Stability Testing
```python
def test_feature_stability(features, targets, n_splits=5):
    """Test feature importance stability across different time periods."""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    feature_importance_scores = []
    
    for train_idx, test_idx in tscv.split(features):
        X_train, y_train = features.iloc[train_idx], targets.iloc[train_idx]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        importance_df = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        })
        feature_importance_scores.append(importance_df)
    
    # Calculate stability metrics
    importance_matrix = pd.concat(feature_importance_scores, axis=1)
    stability_metrics = {
        'mean_importance': importance_matrix.groupby('feature')['importance'].mean(),
        'std_importance': importance_matrix.groupby('feature')['importance'].std(),
        'cv_importance': importance_matrix.groupby('feature')['importance'].std() / importance_matrix.groupby('feature')['importance'].mean()
    }
    
    return stability_metrics
```

## Feature Engineering Pipeline

### Automated Feature Generation
```python
class FeatureEngineeringPipeline:
    """Comprehensive feature engineering pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.technical_engine = TechnicalFeatureEngine(config.technical)
        self.microstructure_engine = MicrostructureFeatureEngine(config.microstructure)
        self.cross_asset_engine = CrossAssetFeatureEngine(config.cross_asset)
    
    def generate_all_features(self, equity_data, regime_data):
        """Generate all feature categories."""
        
        # Technical features
        tech_features = self.technical_engine.generate_all_features(equity_data)
        
        # Microstructure features
        micro_features = self.microstructure_engine.generate_all_features(equity_data)
        
        # Cross-asset features
        cross_features = self.cross_asset_engine.generate_all_features(
            equity_data, regime_data
        )
        
        # Combine all features
        all_features = pd.concat([
            tech_features, 
            micro_features, 
            cross_features
        ], axis=1)
        
        # Apply transformations
        all_features = self._apply_transformations(all_features)
        
        # Feature selection
        if self.config.feature_selection.enabled:
            all_features = self._select_features(all_features)
        
        return all_features
    
    def _apply_transformations(self, features):
        """Apply feature transformations."""
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Winsorize outliers
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                q01, q99 = features[col].quantile([0.01, 0.99])
                features[col] = features[col].clip(q01, q99)
        
        # Normalize features
        if self.config.normalization.method == 'zscore':
            features = (features - features.rolling(252).mean()) / features.rolling(252).std()
        elif self.config.normalization.method == 'rank':
            features = features.rolling(252).rank(pct=True)
        
        return features
```

This comprehensive feature engineering framework ensures the Cross-Asset Alpha Engine captures all relevant market signals while maintaining statistical rigor and economic interpretability.
