# Implementation Guide

**IMPORTANT: All empirical analysis in this project is conducted at daily frequency using daily OHLCV bars from Polygon.io. No intraday, tick, or order-book data is used in the current experiment.**

## Getting Started

This guide provides step-by-step instructions for implementing and deploying the Cross-Asset Alpha Engine in your research or production environment. All features are computed from daily OHLCV bars, and execution is modeled at daily close-to-close with simple costs.

## Prerequisites

### System Requirements
- **Python**: 3.7 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 5GB free space for data and cache
- **CPU**: Multi-core processor recommended for parallel processing

### API Access
- **Polygon.io API Key**: Required for real market data
- **Alternative**: System can run with synthetic data for testing

## Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd cross_asset_alpha_engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Install Jupyter dependencies (optional)
pip install -r requirements-jupyter.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API key
POLYGON_API_KEY=your_polygon_api_key_here
```

### 4. Verify Installation

```bash
# Run tests to verify installation
python -m pytest tests/

# Test basic imports
python -c "from cross_asset_alpha_engine import config; print('Installation successful')"
```

## Data Collection

### Automated Data Collection

```bash
# Run comprehensive data collection
python scripts/comprehensive_data_collection.py
```

This script will:
- Fetch real market data from Polygon API
- Generate synthetic data as fallback
- Validate data quality
- Save data in efficient Parquet format

### Manual Data Loading

```python
from cross_asset_alpha_engine.data import load_daily_bars
from datetime import date

# Load specific symbols and date range
data = load_daily_bars(
    symbols=['SPY', 'QQQ', 'AAPL'],
    start_date=date(2023, 1, 1),
    end_date=date(2025, 12, 6),
    use_cache=True
)
```

## Basic Usage Examples

### 1. Simple Alpha Generation

```python
from cross_asset_alpha_engine import AlphaEngine
from datetime import date

# Initialize engine
engine = AlphaEngine()

# Load data
engine.load_data(
    symbols=['SPY', 'QQQ', 'IWM'],
    start_date=date(2023, 1, 1),
    end_date=date(2025, 12, 6)
)

# Generate features
features = engine.create_features()

# Detect regimes
regimes = engine.detect_regimes(features)

# Train models
models = engine.train_models(features, regimes)

# Generate signals
signals = engine.generate_signals(models)

# Construct portfolio
portfolio = engine.construct_portfolio(signals)

# Run backtest
results = engine.backtest(portfolio)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### 2. Custom Feature Engineering

```python
from cross_asset_alpha_engine.features import (
    TechnicalFeatureEngine,
    MicrostructureFeatureEngine,
    CrossAssetFeatureEngine
)

# Initialize feature engines
tech_engine = TechnicalFeatureEngine()
micro_engine = MicrostructureFeatureEngine()
cross_engine = CrossAssetFeatureEngine()

# Generate features
tech_features = tech_engine.generate_all_features(data)
micro_features = micro_engine.generate_all_features(data)
cross_features = cross_engine.generate_all_features(equity_data, regime_data)

# Combine features
all_features = pd.concat([tech_features, micro_features, cross_features], axis=1)
```

### 3. Regime Detection

```python
from cross_asset_alpha_engine.regimes import RegimeHMM

# Initialize HMM
regime_model = RegimeHMM(n_components=3)

# Prepare regime features
regime_features = data[['volatility_20d', 'vix_level', 'equity_bond_corr']]

# Fit model
regime_model.fit(regime_features.values)

# Predict regimes
regimes = regime_model.predict_regimes(regime_features.values)
regime_probs = regime_model.predict_proba(regime_features.values)

# Analyze regime characteristics
for regime in range(3):
    regime_data = data[regimes == regime]
    print(f"Regime {regime}: {len(regime_data)} observations")
    print(f"  Avg Volatility: {regime_data['volatility_20d'].mean():.2%}")
    print(f"  Avg VIX: {regime_data['vix_level'].mean():.1f}")
```

### 4. Portfolio Construction

```python
from cross_asset_alpha_engine.portfolio import PortfolioConstructor

# Initialize portfolio constructor
portfolio_config = {
    'max_position': 0.1,
    'max_gross_exposure': 1.0,
    'risk_parity': False
}

constructor = PortfolioConstructor(portfolio_config)

# Generate alpha scores (from your model)
alpha_scores = model.predict(features)

# Calculate volatilities
volatilities = data.groupby('symbol')['returns_1d'].rolling(20).std().groupby('symbol').last()

# Construct portfolio
positions = constructor.construct_portfolio(alpha_scores, volatilities)

print("Portfolio Positions:")
for symbol, position in positions.items():
    print(f"  {symbol}: {position:.2%}")
```

## Advanced Configuration

### 1. Custom Model Configuration

```python
# Random Forest configuration
rf_config = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

# Alternative models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

models = {
    'random_forest': RandomForestRegressor(**rf_config),
    'logistic': LogisticRegression(penalty='l1', C=0.1),
    'svm': SVR(kernel='rbf', C=1.0)
}
```

### 2. Risk Management Configuration

```python
risk_config = {
    'max_individual_position': 0.10,  # 10% max per position
    'max_gross_exposure': 1.0,        # 100% gross exposure
    'max_net_exposure': 0.05,         # 5% net exposure (market neutral)
    'max_sector_exposure': 0.30,      # 30% max per sector
    'max_drawdown_stop': 0.15,        # 15% drawdown stop
    'volatility_target': 0.12,        # 12% annual volatility target
    'rebalance_frequency': 'daily'    # Daily rebalancing
}
```

### 3. Backtesting Configuration

```python
backtest_config = {
    'initial_capital': 1_000_000,
    'commission_rate': 0.001,      # 10 bps
    'spread_cost': 0.0005,         # 5 bps
    'impact_coefficient': 0.1,     # Market impact
    'slippage_rate': 0.0005,       # 5 bps slippage
    'benchmark': 'SPY'             # Benchmark for comparison
}
```

## Jupyter Notebook Usage

### 1. Setup Jupyter Kernel

```bash
# Install Jupyter kernel for the project
python -m ipykernel install --user --name=cross_asset_alpha_engine --display-name="Cross-Asset Alpha Engine"
```

### 2. Launch Jupyter

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### 3. Available Notebooks

- **Complete System Analysis**: `notebooks/Complete_System_Analysis.ipynb`
- **Data Exploration**: `notebooks/01_data_sanity_check.ipynb`
- **Feature Engineering**: `notebooks/02_feature_exploration.ipynb`
- **Regime Detection**: `notebooks/03_regime_detection_demo.ipynb`
- **Backtesting**: `notebooks/04_alpha_backtest_demo.ipynb`
- **Execution Simulation**: `notebooks/05_execution_simulation_demo.ipynb`

## Production Deployment

### 1. Environment Variables

```bash
# Production environment settings
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export DATA_SOURCE=real_time
export RISK_CHECKS_ENABLED=true
export POSITION_LIMITS_ENFORCED=true
```

### 2. Monitoring Setup

```python
# Performance monitoring
import logging
from cross_asset_alpha_engine.utils import setup_logger

# Setup production logging
logger = setup_logger(
    name="alpha_engine_prod",
    level="INFO",
    file_output=True,
    log_file="logs/production.log"
)

# Monitor key metrics
def monitor_performance(portfolio_returns):
    current_sharpe = calculate_sharpe_ratio(portfolio_returns)
    current_drawdown = calculate_max_drawdown(portfolio_returns)
    
    if current_sharpe < 1.0:
        logger.warning(f"Sharpe ratio below threshold: {current_sharpe:.2f}")
    
    if current_drawdown > 0.10:
        logger.error(f"Drawdown exceeds limit: {current_drawdown:.2%}")
```

### 3. Automated Execution

```python
# Daily execution script
def daily_execution():
    # Load latest data
    data = load_latest_data()
    
    # Generate signals
    signals = generate_signals(data)
    
    # Construct portfolio
    portfolio = construct_portfolio(signals)
    
    # Apply risk controls
    portfolio = apply_risk_controls(portfolio)
    
    # Execute trades
    execute_trades(portfolio)
    
    # Log performance
    log_performance(portfolio)

# Schedule daily execution
import schedule
schedule.every().day.at("09:00").do(daily_execution)
```

## Troubleshooting

### Common Issues

**1. API Rate Limiting**
```python
# Solution: Implement exponential backoff
import time

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

**2. Memory Issues with Large Datasets**
```python
# Solution: Process data in chunks
def process_large_dataset(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        chunk_result = process_chunk(chunk)
        results.append(chunk_result)
    return pd.concat(results)
```

**3. Model Performance Degradation**
```python
# Solution: Implement model monitoring
def monitor_model_performance(model, recent_data, threshold=0.05):
    recent_performance = evaluate_model(model, recent_data)
    baseline_performance = model.baseline_performance
    
    if recent_performance < baseline_performance - threshold:
        logger.warning("Model performance degraded, consider retraining")
        return False
    return True
```

## Performance Optimization

### 1. Data Processing Optimization

```python
# Use vectorized operations
import numpy as np

# Efficient feature calculation
def calculate_returns_vectorized(prices):
    return np.log(prices / prices.shift(1))

# Parallel processing
from multiprocessing import Pool

def parallel_feature_generation(symbols):
    with Pool() as pool:
        results = pool.map(generate_features_for_symbol, symbols)
    return pd.concat(results)
```

### 2. Model Training Optimization

```python
# Use early stopping
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=1000,
    warm_start=True,
    oob_score=True
)

# Incremental training
for n_est in [100, 200, 300, 400, 500]:
    model.n_estimators = n_est
    model.fit(X_train, y_train)
    
    if model.oob_score_ > best_score:
        best_score = model.oob_score_
        best_n_estimators = n_est
    else:
        break  # Early stopping
```

## Support and Resources

### Documentation
- **API Reference**: Complete function and class documentation
- **Examples**: Comprehensive usage examples
- **Tutorials**: Step-by-step guides for common tasks

### Community
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share insights
- **Contributions**: Guidelines for contributing to the project

### Professional Support
- **Consulting**: Custom implementation and optimization
- **Training**: Workshops and educational programs
- **Maintenance**: Ongoing support and updates

This implementation guide provides everything needed to successfully deploy and operate the Cross-Asset Alpha Engine in both research and production environments.
