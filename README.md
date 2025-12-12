# Cross-Asset Alpha Engine

A comprehensive quantitative trading system that generates alpha through regime-aware feature engineering and machine learning techniques across multiple asset classes.

## Overview

This system combines traditional quantitative finance methods with modern machine learning to identify trading opportunities across equities, bonds, commodities, and currencies. The engine employs Hidden Markov Models for regime detection and uses cross-asset relationships to generate robust alpha signals.

## Key Features

- **Multi-Asset Coverage**: Equities, bonds, commodities, and currency instruments
- **Regime Detection**: Automatic identification of market regimes using statistical methods
- **Advanced Feature Engineering**: 40+ technical, microstructure, and cross-asset features
- **Machine Learning Integration**: Random Forest models with regime-specific training
- **Risk Management**: Built-in position sizing and portfolio risk controls
- **Professional Data Pipeline**: Real-time data ingestion with caching and validation
- **Comprehensive Backtesting**: Walk-forward validation and performance analytics

## System Architecture

```
├── Data Layer
│   ├── Polygon API Integration
│   ├── Data Caching System
│   └── Quality Validation
├── Feature Engineering
│   ├── Technical Indicators
│   ├── Microstructure Features
│   └── Cross-Asset Signals
├── Regime Detection
│   ├── Statistical Methods
│   └── HMM Implementation
├── Alpha Generation
│   ├── ML Model Training
│   └── Signal Generation
├── Portfolio Construction
│   ├── Position Sizing
│   └── Risk Controls
└── Performance Analytics
    ├── Backtesting Engine
    └── Risk Metrics
```

## Installation

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)
- Polygon.io API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cross_asset_alpha_engine
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your POLYGON_API_KEY
```

5. Install Jupyter kernel (for notebooks):
```bash
pip install -r requirements-jupyter.txt
python -m ipykernel install --user --name=cross_asset_alpha_engine
```

## Configuration

### API Setup

1. Sign up for a Polygon.io account
2. Obtain your API key
3. Add the key to your `.env` file:
```
POLYGON_API_KEY=your_api_key_here
```

### Data Collection

The system supports both real-time data collection and cached data usage:

```python
from cross_asset_alpha_engine.data import load_daily_bars
from datetime import date

# Load data for analysis
data = load_daily_bars(
    symbols=['SPY', 'QQQ', 'AAPL'],
    start_date=date(2023, 1, 1),
    end_date=date(2025, 12, 6),
    use_cache=True
)
```

## Usage

### Basic Example

```python
from cross_asset_alpha_engine import AlphaEngine
from datetime import date

# Initialize the engine
engine = AlphaEngine()

# Load and process data
engine.load_data(
    symbols=['SPY', 'QQQ', 'IWM'],
    start_date=date(2023, 1, 1),
    end_date=date(2025, 12, 6)
)

# Generate features
features = engine.create_features()

# Detect market regimes
regimes = engine.detect_regimes(features)

# Train alpha models
models = engine.train_models(features, regimes)

# Generate signals
signals = engine.generate_signals(models)

# Construct portfolio
portfolio = engine.construct_portfolio(signals)

# Run backtest
results = engine.backtest(portfolio)
```

### Jupyter Notebooks

The system includes comprehensive Jupyter notebooks for analysis:

1. **Complete System Analysis** (`notebooks/Complete_System_Analysis.ipynb`)
   - End-to-end system demonstration
   - Professional analysis with real market data
   - Publication-ready results and visualizations

2. **Data Exploration** (`notebooks/01_data_sanity_check.ipynb`)
   - Data quality analysis
   - Market data validation
   - Universe construction

3. **Feature Engineering** (`notebooks/02_feature_exploration.ipynb`)
   - Feature generation and analysis
   - Cross-asset relationship exploration
   - Feature importance evaluation

4. **Regime Detection** (`notebooks/03_regime_detection_demo.ipynb`)
   - Market regime identification
   - HMM implementation
   - Regime transition analysis

5. **Backtesting** (`notebooks/04_alpha_backtest_demo.ipynb`)
   - Portfolio construction
   - Performance analysis
   - Risk metrics calculation

## Data Requirements

### Supported Instruments

**Equity Universe:**
- SPY (S&P 500 ETF)
- QQQ (NASDAQ-100 ETF)
- IWM (Russell 2000 ETF)
- Major individual stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META)

**Regime Indicators:**
- VIX (Volatility Index)
- TLT (20+ Year Treasury Bond ETF)
- GLD (Gold ETF)
- DXY (US Dollar Index)
- USO (Oil ETF)

### Data Quality Standards

- Minimum 500 trading days per instrument
- Daily OHLCV data with VWAP
- Maximum 5% missing data tolerance
- Professional data validation and cleaning

## Feature Engineering

The system generates comprehensive features across multiple categories:

### Technical Features
- Multi-timeframe returns (1d, 5d, 20d, 60d)
- Volatility measures and ratios
- Momentum indicators
- Mean reversion signals
- Bollinger Bands and RSI

### Microstructure Features
- VWAP deviations
- Volume patterns and z-scores
- Intraday range analysis
- Gap and overnight return analysis

### Cross-Asset Features
- Inter-market correlations
- Volatility regime indicators
- Risk-on/risk-off sentiment
- Currency and commodity exposure

## Model Architecture

### Regime Detection
- Statistical regime identification
- Hidden Markov Model implementation
- Volatility and correlation-based features
- Dynamic regime probability estimation

### Alpha Generation
- Random Forest ensemble models
- Regime-specific model training
- Feature importance analysis
- Cross-validation and robustness testing

### Portfolio Construction
- Risk-controlled position sizing
- Market neutral construction
- Volatility targeting
- Dynamic rebalancing

## Performance Analytics

### Backtesting Framework
- Walk-forward validation
- Out-of-sample testing
- Transaction cost modeling
- Realistic execution simulation

### Risk Metrics
- Sharpe ratio and information ratio
- Maximum drawdown analysis
- Value-at-Risk (VaR) calculation
- Tail risk measures

### Performance Reporting
- Comprehensive performance attribution
- Risk factor decomposition
- Regime-specific performance analysis
- Publication-quality visualizations

## Project Structure

```
cross_asset_alpha_engine/
├── src/cross_asset_alpha_engine/
│   ├── data/                 # Data ingestion and management
│   ├── features/             # Feature engineering modules
│   ├── regimes/              # Regime detection algorithms
│   ├── models/               # Alpha generation models
│   ├── execution/            # Execution simulation
│   ├── backtest/             # Backtesting framework
│   └── utils/                # Utility functions
├── notebooks/                # Jupyter analysis notebooks
├── scripts/                  # Data collection and utility scripts
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## Testing

Run the test suite to ensure system integrity:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_basic_imports.py

# Run with coverage
python -m pytest tests/ --cov=cross_asset_alpha_engine
```

## Contributing

This project follows standard software development practices:

1. Code should be well-documented and tested
2. Follow PEP 8 style guidelines
3. Include unit tests for new functionality
4. Update documentation for API changes

## Research Applications

This system is designed for academic and professional research in quantitative finance:

- **Academic Research**: Suitable for journal publication with comprehensive documentation
- **Institutional Trading**: Professional-grade risk management and execution modeling
- **Strategy Development**: Extensible framework for new alpha factors
- **Risk Management**: Advanced portfolio construction and risk analytics

## Performance Considerations

- **Data Caching**: Automatic caching reduces API calls and improves performance
- **Vectorized Operations**: NumPy and Pandas optimization for large datasets
- **Memory Management**: Efficient data structures for large-scale analysis
- **Parallel Processing**: Multi-core support for model training and backtesting

## Limitations and Disclaimers

- This system is for research and educational purposes
- Past performance does not guarantee future results
- All trading involves risk of loss
- Proper risk management is essential for live trading
- Market conditions may differ from historical patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or contributions, please refer to the project's issue tracker or documentation.

---

**Note**: This system demonstrates advanced quantitative finance techniques and should be used by individuals with appropriate knowledge of financial markets and risk management.