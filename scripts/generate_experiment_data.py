#!/usr/bin/env python3
"""
Data Generation Script for Cross-Asset Alpha Engine Experiment

This script generates comprehensive market data for the end-to-end system demonstration.
It fetches real market data where possible and generates realistic synthetic data as fallback.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_asset_alpha_engine.data import load_daily_bars
from cross_asset_alpha_engine.data.cache import save_to_parquet
from cross_asset_alpha_engine.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger("data_generation")

# Experiment configuration
EXPERIMENT_CONFIG = {
    "equity_universe": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
    "regime_indicators": ["VIX", "TLT", "GLD", "DXY", "USO"],
    "start_date": date(2023, 1, 1),
    "end_date": date(2025, 12, 6),
    "data_dir": Path("data"),
    "min_observations": 500  # Minimum bars needed for meaningful analysis
}

def generate_realistic_price_series(symbol: str, start_date: date, end_date: date, 
                                  base_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic synthetic price data with proper OHLCV structure."""
    
    # Calculate business days between dates
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    # Symbol-specific parameters
    symbol_params = {
        "SPY": {"volatility": 0.15, "drift": 0.08, "base_price": 400},
        "QQQ": {"volatility": 0.20, "drift": 0.10, "base_price": 350},
        "IWM": {"volatility": 0.25, "drift": 0.06, "base_price": 200},
        "AAPL": {"volatility": 0.25, "drift": 0.12, "base_price": 150},
        "MSFT": {"volatility": 0.22, "drift": 0.11, "base_price": 300},
        "GOOGL": {"volatility": 0.24, "drift": 0.09, "base_price": 2500},
        "AMZN": {"volatility": 0.28, "drift": 0.08, "base_price": 3000},
        "TSLA": {"volatility": 0.45, "drift": 0.15, "base_price": 200},
        "NVDA": {"volatility": 0.35, "drift": 0.20, "base_price": 400},
        "META": {"volatility": 0.30, "drift": 0.10, "base_price": 250},
        "VIX": {"volatility": 0.80, "drift": -0.05, "base_price": 20},
        "TLT": {"volatility": 0.12, "drift": 0.02, "base_price": 120},
        "GLD": {"volatility": 0.16, "drift": 0.04, "base_price": 180},
        "DXY": {"volatility": 0.08, "drift": 0.01, "base_price": 100},
        "USO": {"volatility": 0.35, "drift": 0.03, "base_price": 70}
    }
    
    params = symbol_params.get(symbol, {"volatility": 0.20, "drift": 0.08, "base_price": base_price})
    
    # Generate price path using geometric Brownian motion
    dt = 1/252  # Daily time step
    returns = np.random.normal(
        params["drift"] * dt, 
        params["volatility"] * np.sqrt(dt), 
        n_days
    )
    
    # Add some regime changes and volatility clustering
    regime_changes = np.random.choice([0, 1], size=n_days, p=[0.98, 0.02])
    volatility_multiplier = np.where(regime_changes, 2.0, 1.0)
    returns *= volatility_multiplier
    
    # Calculate cumulative prices
    price_path = params["base_price"] * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (date_val, close_price) in enumerate(zip(dates, price_path)):
        # Generate intraday volatility
        intraday_vol = params["volatility"] * 0.3 * np.random.uniform(0.5, 1.5)
        
        # Open price (with gap from previous close)
        if i == 0:
            open_price = close_price * (1 + np.random.normal(0, 0.005))
        else:
            gap = np.random.normal(0, 0.01)  # Overnight gap
            open_price = price_path[i-1] * (1 + gap)
        
        # High and low based on intraday volatility
        high_factor = 1 + abs(np.random.normal(0, intraday_vol))
        low_factor = 1 - abs(np.random.normal(0, intraday_vol))
        
        high_price = max(open_price, close_price) * high_factor
        low_price = min(open_price, close_price) * low_factor
        
        # Volume (with realistic patterns)
        base_volume = 50_000_000 if symbol in ["SPY", "QQQ"] else 20_000_000
        volume_factor = np.random.lognormal(0, 0.5)  # Log-normal distribution
        volume = int(base_volume * volume_factor)
        
        # VWAP approximation
        vwap = (open_price + high_price + low_price + close_price) / 4
        
        data.append({
            "symbol": symbol,
            "timestamp": date_val,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume,
            "vwap": round(vwap, 2)
        })
    
    return pd.DataFrame(data)

def fetch_or_generate_data(symbols: list, start_date: date, end_date: date) -> pd.DataFrame:
    """Attempt to fetch real data, fallback to synthetic generation."""
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        try:
            # Attempt to fetch real data
            real_data = load_daily_bars([symbol], start_date, end_date, use_cache=True)
            
            if not real_data.empty and len(real_data) >= EXPERIMENT_CONFIG["min_observations"]:
                logger.info(f"Successfully fetched {len(real_data)} real bars for {symbol}")
                all_data.append(real_data)
            else:
                raise ValueError("Insufficient real data")
                
        except Exception as e:
            logger.warning(f"Failed to fetch real data for {symbol}: {e}")
            logger.info(f"Generating synthetic data for {symbol}")
            
            # Generate synthetic data
            synthetic_data = generate_realistic_price_series(symbol, start_date, end_date)
            all_data.append(synthetic_data)
            
        # Small delay to avoid rate limits
        import time
        time.sleep(0.2)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total dataset: {len(combined_data)} bars across {len(symbols)} symbols")
    
    return combined_data

def main():
    """Generate all required data for the experiment."""
    
    logger.info("Starting data generation for Cross-Asset Alpha Engine experiment")
    
    # Create data directory
    EXPERIMENT_CONFIG["data_dir"].mkdir(exist_ok=True)
    
    # Generate equity universe data
    logger.info("Generating equity universe data...")
    equity_data = fetch_or_generate_data(
        EXPERIMENT_CONFIG["equity_universe"],
        EXPERIMENT_CONFIG["start_date"],
        EXPERIMENT_CONFIG["end_date"]
    )
    
    # Save equity data
    equity_file = EXPERIMENT_CONFIG["data_dir"] / "equity_universe.parquet"
    save_to_parquet(equity_data, str(equity_file))
    logger.info(f"Saved equity data to {equity_file}")
    
    # Generate regime indicator data
    logger.info("Generating regime indicator data...")
    regime_data = fetch_or_generate_data(
        EXPERIMENT_CONFIG["regime_indicators"],
        EXPERIMENT_CONFIG["start_date"],
        EXPERIMENT_CONFIG["end_date"]
    )
    
    # Save regime data
    regime_file = EXPERIMENT_CONFIG["data_dir"] / "regime_indicators.parquet"
    save_to_parquet(regime_data, str(regime_file))
    logger.info(f"Saved regime data to {regime_file}")
    
    # Generate summary statistics
    summary_stats = {
        "generation_date": datetime.now().isoformat(),
        "date_range": {
            "start": EXPERIMENT_CONFIG["start_date"].isoformat(),
            "end": EXPERIMENT_CONFIG["end_date"].isoformat()
        },
        "equity_universe": {
            "symbols": EXPERIMENT_CONFIG["equity_universe"],
            "total_bars": len(equity_data),
            "date_range_actual": {
                "start": equity_data["timestamp"].min().isoformat(),
                "end": equity_data["timestamp"].max().isoformat()
            }
        },
        "regime_indicators": {
            "symbols": EXPERIMENT_CONFIG["regime_indicators"],
            "total_bars": len(regime_data),
            "date_range_actual": {
                "start": regime_data["timestamp"].min().isoformat(),
                "end": regime_data["timestamp"].max().isoformat()
            }
        }
    }
    
    # Save summary
    import json
    summary_file = EXPERIMENT_CONFIG["data_dir"] / "data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info(f"Data generation complete. Summary saved to {summary_file}")
    logger.info(f"Total equity bars: {len(equity_data)}")
    logger.info(f"Total regime bars: {len(regime_data)}")

if __name__ == "__main__":
    main()
