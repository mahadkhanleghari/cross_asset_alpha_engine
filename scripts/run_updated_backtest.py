#!/usr/bin/env python3
"""
Updated backtest script demonstrating the honest regime detection and transaction cost modeling.

This script shows the corrected implementation:
1. Uses quantile-based regime detection (not HMM)
2. Includes realistic transaction costs and turnover tracking
3. Provides Sharpe ratio confidence intervals
4. Clearly labels benchmark as equal-weight universe average
5. Documents limitations explicitly
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cross_asset_alpha_engine.data import load_from_parquet
from cross_asset_alpha_engine.regimes import assign_regimes, get_regime_descriptions
from cross_asset_alpha_engine.backtest import BacktestEngine, BacktestConfig
from cross_asset_alpha_engine.utils.logging_utils import setup_logger
from sklearn.ensemble import RandomForestRegressor


def main():
    """Run the updated backtest with honest methodology."""
    
    # Setup logging
    logger = setup_logger("updated_backtest")
    logger.info("Starting updated backtest with honest regime detection and transaction costs")
    
    # Load data
    data_dir = Path("data")
    equity_file = data_dir / "equity_universe_comprehensive.parquet"
    regime_file = data_dir / "regime_indicators_comprehensive.parquet"
    
    if not equity_file.exists():
        logger.error(f"Data file not found: {equity_file}")
        logger.info("Please run scripts/comprehensive_data_collection.py first")
        return
    
    logger.info("Loading market data...")
    equity_data = load_from_parquet(str(equity_file))
    regime_data = load_from_parquet(str(regime_file))
    
    # Combine data
    all_data = pd.concat([equity_data, regime_data], ignore_index=True)
    logger.info(f"Loaded {len(all_data)} total observations")
    
    # Create basic features
    logger.info("Creating features...")
    feature_data = create_features(all_data)
    
    # Split data chronologically (70/30 split)
    split_date = feature_data['timestamp'].quantile(0.7)
    train_data = feature_data[feature_data['timestamp'] <= split_date]
    test_data = feature_data[feature_data['timestamp'] > split_date]
    
    logger.info(f"Training data: {len(train_data)} observations (through {split_date.date()})")
    logger.info(f"Testing data: {len(test_data)} observations (from {test_data['timestamp'].min().date()})")
    
    # HONEST REGIME DETECTION: Use quantile-based method, not HMM
    logger.info("Applying quantile-based regime detection (NOT HMM)")
    logger.info("Current experiment uses volatility/VIX quantile regimes")
    
    train_regimes = assign_regimes(train_data, method="vol_vix_quantiles", n_regimes=3)
    test_regimes = assign_regimes(test_data, method="vol_vix_quantiles", n_regimes=3)
    
    train_data = train_data.copy()
    train_data['market_regime'] = train_regimes
    test_data = test_data.copy()
    test_data['market_regime'] = test_regimes
    
    # Print regime descriptions
    regime_descriptions = get_regime_descriptions()
    logger.info("Regime definitions:")
    for regime, description in regime_descriptions.items():
        logger.info(f"  {regime}: {description}")
    
    # Train alpha models
    logger.info("Training alpha models...")
    feature_cols = [col for col in train_data.columns if col.startswith(('returns_', 'volatility_', 'volume_', 'momentum_'))]
    feature_cols = feature_cols[:20]  # Use top 20 features for simplicity
    
    models = train_alpha_models(train_data, feature_cols)
    
    # Generate predictions on test set
    logger.info("Generating predictions...")
    test_predictions = generate_predictions(test_data, models, feature_cols)
    
    # REALISTIC BACKTESTING with transaction costs
    logger.info("Running backtest with transaction costs...")
    
    backtest_config = BacktestConfig(
        transaction_cost_bps_per_side=5.0,  # 5 bps per side (conservative)
        max_position=0.10,  # 10% max position
        max_gross_exposure=1.0,  # 100% gross exposure
        target_net_exposure=0.0,  # Market neutral
        risk_free_rate=0.02,  # 2% risk-free rate
        save_portfolio_performance=True
    )
    
    backtest_engine = BacktestEngine(config=backtest_config)
    
    results = backtest_engine.run_backtest(
        predictions_df=test_predictions,
        alpha_col='alpha_regime',
        target_col='target_1d'
    )
    
    # Print comprehensive results
    backtest_engine.print_performance_summary(results)
    
    # Document limitations
    print_limitations(results, train_data, test_data)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    portfolio_performance = results['portfolio_performance']
    metrics = results['metrics']
    
    # Save performance data
    portfolio_performance.to_parquet(results_dir / "updated_portfolio_performance.parquet")
    
    # Save comprehensive summary
    summary = {
        'methodology': {
            'regime_detection': 'vol_vix_quantiles',
            'regime_descriptions': regime_descriptions,
            'hmm_status': 'Available as optional extension but NOT used in reported results'
        },
        'performance_metrics': metrics,
        'limitations': {
            'sample_length': f"{len(test_data)} test observations",
            'survivorship_bias': 'Handpicked large-cap universe',
            'frequency': 'Daily (no intraday microstructure)',
            'time_period': f"{test_data['timestamp'].min().date()} to {test_data['timestamp'].max().date()}",
            'transaction_costs': f"Assumed {backtest_config.transaction_cost_bps_per_side} bps per side"
        },
        'data_summary': {
            'n_symbols': test_data['symbol'].nunique(),
            'n_test_observations': len(test_data),
            'avg_observations_per_symbol': len(test_data) / test_data['symbol'].nunique()
        }
    }
    
    with open(results_dir / "updated_results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_dir}")
    logger.info("Updated backtest completed successfully")


def create_features(data):
    """Create basic features for the model."""
    
    feature_data = []
    
    for symbol, group in data.groupby('symbol'):
        df = group.copy().sort_values('timestamp')
        
        # Basic return features
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Volatility features
        df['volatility_5d'] = df['returns_1d'].rolling(5).std() * np.sqrt(252)
        df['volatility_20d'] = df['returns_1d'].rolling(20).std() * np.sqrt(252)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_ma_20d'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20d']
        
        # Momentum features
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # Target variable (next day return)
        df['target_1d'] = df['returns_1d'].shift(-1)
        
        # Add VIX level for regime detection
        if symbol == 'VIX':
            df['vix_level'] = df['close']
        
        feature_data.append(df)
    
    return pd.concat(feature_data, ignore_index=True).dropna()


def train_alpha_models(data, feature_cols):
    """Train regime-specific alpha models."""
    
    models = {}
    
    # Overall model
    X = data[feature_cols].fillna(0)
    y = data['target_1d'].fillna(0)
    
    overall_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    overall_model.fit(X, y)
    models['overall'] = overall_model
    
    # Regime-specific models
    for regime in data['market_regime'].unique():
        if pd.isna(regime):
            continue
            
        regime_data = data[data['market_regime'] == regime]
        if len(regime_data) < 50:  # Skip regimes with insufficient data
            continue
        
        X_regime = regime_data[feature_cols].fillna(0)
        y_regime = regime_data['target_1d'].fillna(0)
        
        regime_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        regime_model.fit(X_regime, y_regime)
        models[regime] = regime_model
    
    return models


def generate_predictions(data, models, feature_cols):
    """Generate alpha predictions using trained models."""
    
    predictions = []
    
    for symbol, group in data.groupby('symbol'):
        df = group.copy().sort_values('timestamp')
        
        # Overall predictions
        X = df[feature_cols].fillna(0)
        df['alpha_overall'] = models['overall'].predict(X)
        
        # Regime-specific predictions
        df['alpha_regime'] = df['alpha_overall']  # Default to overall
        
        for regime in df['market_regime'].unique():
            if pd.isna(regime) or regime not in models:
                continue
                
            regime_mask = df['market_regime'] == regime
            if regime_mask.sum() > 0:
                X_regime = df.loc[regime_mask, feature_cols].fillna(0)
                df.loc[regime_mask, 'alpha_regime'] = models[regime].predict(X_regime)
        
        predictions.append(df)
    
    return pd.concat(predictions, ignore_index=True)


def print_limitations(results, train_data, test_data):
    """Print explicit limitations of the analysis."""
    
    print("\n" + "="*60)
    print("IMPORTANT LIMITATIONS AND ASSUMPTIONS")
    print("="*60)
    
    print(f"\n1. LIMITED SAMPLE SIZE:")
    print(f"   - Test period: {len(test_data)} observations")
    print(f"   - Approximately {len(test_data) / test_data['symbol'].nunique():.0f} observations per symbol")
    print(f"   - Time period: {test_data['timestamp'].min().date()} to {test_data['timestamp'].max().date()}")
    
    print(f"\n2. SURVIVORSHIP BIAS:")
    print(f"   - Handpicked universe of {test_data['symbol'].nunique()} large-cap symbols")
    print(f"   - No delisted or failed companies included")
    print(f"   - Results may not generalize to broader universe")
    
    print(f"\n3. FREQUENCY LIMITATIONS:")
    print(f"   - Daily OHLCV bars only (no intraday, tick, or order-book data)")
    print(f"   - Limited to end-of-day pricing")
    print(f"   - No true high-frequency trading patterns or intraday microstructure")
    
    print(f"\n4. TRANSACTION COST ASSUMPTIONS:")
    config = results['config']
    print(f"   - Assumed {config.transaction_cost_bps_per_side} bps per trade side")
    print(f"   - Simplified market impact model")
    print(f"   - No capacity constraints modeled")
    
    print(f"\n5. REGIME DETECTION:")
    print(f"   - Uses simple volatility/VIX quantiles")
    print(f"   - HMM available but NOT used in these results")
    print(f"   - No forward-looking bias in regime assignment")
    
    print(f"\n6. TIME PERIOD SPECIFICITY:")
    print(f"   - Results specific to 2023-2025 market conditions")
    print(f"   - May not generalize to different market cycles")
    print(f"   - Limited stress testing scenarios")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
