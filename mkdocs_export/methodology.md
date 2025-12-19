# Cross-Asset Alpha Engine Methodology

**IMPORTANT: All empirical analysis in this project is conducted at daily frequency using daily OHLCV bars from Polygon.io. No intraday, tick, or order-book data is used in the current experiment.**

## Theoretical Foundation

### Information Flow Theory
The Cross-Asset Alpha Engine is built on the principle that financial markets are interconnected systems where information flows between asset classes create predictable patterns. Price movements in one asset class often precede movements in related assets, creating lead-lag relationships that can be systematically exploited.

### Regime-Dependent Market Behavior
Asset correlations and volatility patterns change dramatically during different market conditions:
- **Bull Markets**: Low volatility, positive momentum, high correlations
- **Bear Markets**: High volatility, negative momentum, flight-to-quality
- **Crisis Periods**: Correlation breakdown, extreme volatility, liquidity constraints

## Multi-Asset Universe Construction

### Core Asset Classes

#### Equity Universe
**Broad Market ETFs**:
- **SPY**: S&P 500 ETF representing large-cap US equities
- **QQQ**: NASDAQ-100 ETF capturing technology sector dynamics
- **IWM**: Russell 2000 ETF representing small-cap equity exposure

**Individual Mega-Cap Stocks**:
- **AAPL, MSFT, GOOGL**: Technology sector leaders
- **AMZN**: E-commerce and cloud computing
- **TSLA**: Electric vehicle and clean energy
- **NVDA**: Semiconductor and AI infrastructure

#### Cross-Asset Regime Indicators
**Volatility Measures**:
- **VIX**: CBOE Volatility Index measuring implied volatility and market fear

**Interest Rate Environment**:
- **TLT**: iShares 20+ Year Treasury Bond ETF reflecting long-term rates

**Safe Haven Assets**:
- **GLD**: SPDR Gold Trust capturing precious metals demand

**Currency Strength**:
- **DXY**: US Dollar Index measuring dollar strength vs major currencies

**Commodity Exposure**:
- **USO**: United States Oil Fund tracking crude oil prices

## Advanced Feature Engineering Framework

### Technical Analysis Features

#### Multi-Horizon Momentum
```python
# Momentum across different timeframes
returns_1d = price.pct_change(1)
returns_5d = price.pct_change(5)
returns_20d = price.pct_change(20)
returns_60d = price.pct_change(60)

# Momentum acceleration
momentum_acceleration = returns_5d - returns_20d
```

#### Volatility Analysis
```python
# Realized volatility measures
vol_5d = returns.rolling(5).std() * np.sqrt(252)
vol_20d = returns.rolling(20).std() * np.sqrt(252)
vol_ratio = vol_5d / vol_20d

# GARCH-style volatility clustering
vol_persistence = vol_5d.rolling(5).mean()
```

#### Mean Reversion Indicators
```python
# Bollinger Band position
sma_20 = price.rolling(20).mean()
bb_std = price.rolling(20).std()
bb_position = (price - sma_20) / (2 * bb_std)

# RSI calculation
gains = returns.where(returns > 0, 0)
losses = -returns.where(returns < 0, 0)
rs = gains.rolling(14).mean() / losses.rolling(14).mean()
rsi = 100 - (100 / (1 + rs))
```

### Daily Microstructure-Inspired Features

**Note: All features are computed from daily OHLCV bars; no intraday or tick data is used in the current experiment.**

#### Volume-Weighted Average Price (VWAP) Analysis
```python
# VWAP deviation
vwap_deviation = (price - vwap) / vwap

# Volume-price relationship
volume_zscore = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
```

#### Daily Price Patterns (Computed from OHLCV)
```python
# Gap analysis (computed from daily bars)
overnight_gap = (open_price - close_price.shift(1)) / close_price.shift(1)

# Daily range analysis
daily_range = (high - low) / close
range_zscore = (daily_range - daily_range.rolling(20).mean()) / daily_range.rolling(20).std()
```

### Cross-Asset Feature Engineering

#### Dynamic Correlation Analysis
```python
# Rolling correlations between asset classes
equity_bond_corr = equity_returns.rolling(20).corr(bond_returns)
equity_vix_corr = equity_returns.rolling(20).corr(vix_changes)

# Correlation breakdown detection
corr_change = equity_bond_corr - equity_bond_corr.rolling(60).mean()
```

#### Risk Sentiment Indicators
```python
# Risk-on/risk-off sentiment
risk_sentiment = -vix_changes + bond_returns

# Flight-to-quality indicator
flight_to_quality = (bond_returns > bond_returns.quantile(0.8)) & (equity_returns < 0)
```

## Regime Detection Methodology

**IMPORTANT**: The current experiment uses volatility/VIX quantile regimes, NOT Hidden Markov Models. HMM-based detection is available as an optional extension but is not used in the reported results.

### Current Implementation: Volatility/VIX Quantile Regimes

The reported results use a simple but effective quantile-based regime detection method:

#### Methodology
```python
from cross_asset_alpha_engine.regimes import assign_regimes

# Current experiment uses this method
regimes = assign_regimes(data, method="vol_vix_quantiles", n_regimes=3)
```

#### Regime Construction
1. **Volatility Regime**: 20-day realized volatility divided into 3 quantiles
   - Low_Vol: Bottom 33% of volatility observations
   - Med_Vol: Middle 33% of volatility observations  
   - High_Vol: Top 33% of volatility observations

2. **VIX Regime**: VIX level divided into 3 quantiles (when available)
   - Low_VIX: Bottom 33% of VIX levels
   - Med_VIX: Middle 33% of VIX levels
   - High_VIX: Top 33% of VIX levels

3. **Combined Regime**: Cartesian product of volatility and VIX regimes
   - Results in up to 9 possible regime combinations
   - Example: "Med_Vol_High_VIX" = medium volatility with high VIX

#### Regime Descriptions
- **Low_Vol_Low_VIX**: Calm, stable market conditions
- **Med_Vol_Med_VIX**: Typical market environment  
- **High_Vol_High_VIX**: Crisis or extreme stress conditions
- **Low_Vol_High_VIX**: Unusual combination, potential turning point
- **High_Vol_Low_VIX**: Unusual market dynamics

### Planned Extension: Hidden Markov Model Implementation

**Note**: HMM-based regimes are available in the framework but require proper train/test implementation to avoid look-ahead bias.

#### Mathematical Framework
The HMM assumes markets exist in K unobservable states with transition probabilities:

$$P(S_t = j | S_{t-1} = i) = a_{ij}$$

#### TODO for HMM Implementation
To properly implement HMM-based regimes:
1. Fit HMM only on training data
2. Predict regimes on both train and test sets using fitted model
3. Ensure no look-ahead bias in regime assignment
4. Compare performance vs quantile-based method

```python
# Future HMM implementation (not currently used)
from cross_asset_alpha_engine.regimes import RegimeHMM

# This would require proper train/test split
hmm_model = RegimeHMM(n_components=3)
hmm_model.fit(train_features)  # Only training data
regimes = hmm_model.predict_regimes(all_features)  # Predict on all data
```

## Machine Learning Architecture

### Random Forest Implementation

#### Model Configuration
```python
from sklearn.ensemble import RandomForestRegressor

# Regime-specific model training
models = {}
for regime in ['low_vol', 'high_vol', 'transition']:
    regime_data = data[data['regime'] == regime]
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42
    )
    
    model.fit(regime_data[features], regime_data['target'])
    models[regime] = model
```

#### Feature Importance Analysis
```python
# Extract feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# SHAP values for individual predictions
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

### Alternative Model Architectures

#### Logistic Regression (Interpretable)
```python
from sklearn.linear_model import LogisticRegression

# For directional prediction
direction_model = LogisticRegression(
    penalty='l1',
    C=0.1,
    solver='liblinear'
)
```

#### Support Vector Machines (Non-Linear)
```python
from sklearn.svm import SVR

# Non-linear pattern recognition
svm_model = SVR(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
```

## Portfolio Construction Methodology

### Position Sizing Algorithms

#### Alpha-Proportional Sizing
```python
# Position size based on alpha strength
def calculate_position_size(alpha_score, volatility, max_position=0.1):
    vol_adjusted_alpha = alpha_score / volatility
    position = np.clip(vol_adjusted_alpha * 0.05, -max_position, max_position)
    return position
```

#### Kelly Criterion Optimization
```python
# Optimal position sizing
def kelly_position(win_prob, avg_win, avg_loss):
    if avg_loss == 0:
        return 0
    
    odds = avg_win / avg_loss
    kelly_fraction = (win_prob * odds - (1 - win_prob)) / odds
    return np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
```

#### Risk Parity Allocation
```python
# Equal risk contribution
def risk_parity_weights(volatilities):
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    return weights
```

### Risk Management Framework

#### Portfolio-Level Constraints
```python
# Risk control implementation
def apply_risk_controls(positions, constraints):
    # Gross exposure limit
    gross_exposure = np.abs(positions).sum()
    if gross_exposure > constraints['max_gross']:
        positions *= constraints['max_gross'] / gross_exposure
    
    # Net exposure (market neutrality)
    net_exposure = positions.sum()
    positions -= net_exposure / len(positions)
    
    # Individual position limits
    positions = np.clip(positions, -constraints['max_individual'], 
                       constraints['max_individual'])
    
    return positions
```

## Validation Framework

### Walk-Forward Validation
```python
# Time series cross-validation
def walk_forward_validation(data, model, train_window=252, test_window=63):
    results = []
    
    for start in range(train_window, len(data) - test_window):
        # Training data
        train_data = data.iloc[start-train_window:start]
        
        # Test data
        test_data = data.iloc[start:start+test_window]
        
        # Train model
        model.fit(train_data[features], train_data['target'])
        
        # Generate predictions
        predictions = model.predict(test_data[features])
        
        # Store results
        results.append({
            'period': test_data.index,
            'predictions': predictions,
            'actual': test_data['target'].values
        })
    
    return results
```

### Performance Metrics
```python
# Comprehensive performance analysis
def calculate_performance_metrics(returns):
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': returns.mean() * 252,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
        'win_rate': (returns > 0).mean(),
        'avg_win': returns[returns > 0].mean(),
        'avg_loss': returns[returns < 0].mean()
    }
    
    return metrics
```

This comprehensive methodology ensures systematic alpha generation while maintaining robust risk controls and realistic execution assumptions throughout the investment process.
