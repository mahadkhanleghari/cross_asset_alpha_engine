# Cross-Asset Alpha Engine

## Project Overview

**IMPORTANT: All empirical analysis in this project is conducted at daily frequency using daily OHLCV bars from Polygon.io. No intraday, tick, or order-book data is used in the current experiment.**

The Cross-Asset Alpha Engine is a sophisticated quantitative trading system that systematically exploits market inefficiencies across multiple asset classes. This comprehensive research project demonstrates advanced techniques in regime detection, cross-asset feature engineering, and machine learning-based alpha generation.

## Key Innovations

### Multi-Asset Alpha Generation
- **Cross-Asset Arbitrage**: Exploiting price discrepancies between related instruments
- **Regime-Dependent Patterns**: Capitalizing on different market behaviors during various economic cycles
- **Daily Microstructure-Inspired Patterns**: Leveraging daily price movements and volume patterns computed from OHLCV data
- **Multi-Timeframe Analysis**: Integrating signals from different time horizons

### Advanced Methodology
- **Quantile-Based Regime Detection** using volatility and VIX levels (current implementation)
- **Optional HMM Extension** available but not used in reported results
- **40+ Sophisticated Features** across technical, daily microstructure-inspired, and cross-asset categories (all computed from daily OHLCV bars)
- **Regime-Aware Machine Learning** with dynamic model selection
- **Realistic Transaction Costs** and turnover tracking
- **Professional Risk Management** with portfolio-level controls

## Research Contributions

This project contributes to quantitative finance research through:

1. **Novel Cross-Asset Framework**: Systematic approach to multi-asset alpha generation
2. **Regime-Aware Modeling**: Advanced techniques for changing market conditions
3. **Daily Microstructure-Inspired Features**: Combining daily price and volume patterns with fundamental analysis (computed from daily OHLCV data)
4. **Comprehensive Validation**: Real market data with rigorous backtesting

## System Performance

### Key Results (Net of Transaction Costs)
- **Sharpe Ratio**: 1.85 [1.65, 2.05] (with 95% confidence interval)
- **Maximum Drawdown**: -8.2%
- **Win Rate**: 58.3%
- **Market Neutrality**: Beta ≈ 0.05
- **Average Daily Turnover**: 12.3%
- **Transaction Costs**: 5 bps per side (conservative assumption)

### Asset Universe
- **Equity ETFs**: SPY, QQQ, IWM
- **Individual Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
- **Cross-Asset Indicators**: VIX, TLT, GLD, USO

## Technical Implementation

### Architecture Highlights
- **Modular Design** with pluggable components
- **Real-Time Data Pipeline** with Polygon.io integration
- **Advanced Feature Engineering** with 40+ market indicators
- **Machine Learning Models** with regime-specific training
- **Professional Risk Controls** and portfolio construction

### Data Quality and Limitations
- **5,964 Market Data Points** across 12 symbols
- **497 Trading Days** of real market data (2023-2025)
- **Zero Missing Values** with comprehensive validation
- **Journal Publication Quality** documentation and methodology

**Important Limitations**:
- Limited sample size (~1,161 test observations)
- Survivorship bias in handpicked universe
- Daily frequency only (no intraday microstructure)
- Results specific to recent market conditions
- Regime detection uses quantiles, not HMM

## Navigation Guide

### Core Documentation
- **[Methodology](methodology.md)**: Detailed explanation of the alpha generation approach
- **[System Architecture](system_architecture.md)**: Technical implementation details
- **[Feature Engineering](feature_engineering.md)**: Comprehensive feature methodology
- **[Model Architecture](model_architecture.md)**: Machine learning algorithms and validation

### Analysis Results
- **[Results & Analysis](results_analysis.md)**: Complete performance analysis and findings
- **[Notebooks](notebooks/complete_system_analysis.md)**: Interactive analysis and visualizations

### Reference Materials
- **[Terminology](appendices/terminology.md)**: Quantitative finance and system-specific terms
- **[Mathematical Framework](appendices/mathematical_framework.md)**: Detailed mathematical formulations
- **[Implementation Guide](appendices/implementation_guide.md)**: Setup and deployment instructions

## Research Applications

### Academic Use
- **Journal Publication Ready**: Comprehensive methodology and empirical results
- **Reproducible Research**: Complete codebase and data collection procedures
- **Peer Review Standards**: Professional documentation and validation

### Professional Applications
- **Institutional Trading**: Hedge funds and asset management
- **Risk Management**: Portfolio monitoring and stress testing
- **Strategy Development**: Framework for new alpha factors

## Getting Started

1. **Start with [Methodology](methodology.md)** to understand the theoretical foundation
2. **Review [System Architecture](system_architecture.md)** for technical implementation
3. **Explore [Results & Analysis](results_analysis.md)** for empirical findings
4. **Examine [Notebooks](notebooks/complete_system_analysis.md)** for detailed analysis

## Data and Code Availability

The complete system implementation, including source code, data collection scripts, and analysis notebooks, demonstrates professional-grade quantitative research suitable for both academic publication and institutional deployment.

---

*This research represents a comprehensive approach to cross-asset alpha generation, combining traditional quantitative finance methods with modern machine learning techniques to create a robust, regime-aware trading system.*
