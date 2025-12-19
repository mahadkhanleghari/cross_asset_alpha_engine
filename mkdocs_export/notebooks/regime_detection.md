# Regime Detection Demo

**IMPORTANT**: The current experiment uses volatility/VIX quantile regimes, NOT Hidden Markov Models. HMM-based detection is available as an optional extension but is not used in the reported results.

## Overview

We'll explore:
1. Current implementation: Volatility/VIX quantile regimes
2. Regime feature engineering
3. Optional: HMM model training (not used in reported results)
4. Regime prediction and analysis
5. Regime visualization and interpretation



```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Cross-Asset Alpha Engine components
from cross_asset_alpha_engine.data import load_daily_bars, AssetUniverse
from cross_asset_alpha_engine.regimes import RegimeHMM, RegimeFeatureEngine
from cross_asset_alpha_engine.utils import setup_logger, plot_regime_overlay

# Setup
logger = setup_logger("regime_demo", console_output=True)
print("✅ All imports successful!")

```

## 1. Load Multi-Asset Data for Regime Detection



```python
# Load regime detection symbols
universe = AssetUniverse()
regime_symbols = universe.get_market_regime_symbols()
print(f"Regime detection symbols: {regime_symbols}")

# Load data
end_date = date.today()
start_date = end_date - timedelta(days=365)  # 1 year of data

print(f"\nLoading data from {start_date} to {end_date}")

try:
    regime_data = load_daily_bars(regime_symbols, start_date, end_date, use_cache=True)
    if regime_data.empty:
        raise ValueError("No data returned")
    print(f"✅ Loaded {len(regime_data)} bars from API")
except Exception as e:
    print(f"⚠️ API error: {e}")
    print("📊 Creating sample regime data...")
    
    # Create sample multi-asset data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = []
    
    # Define different regime periods with distinct characteristics
    regime_periods = [
        (0, len(dates)//3, 'low_vol'),      # Low volatility regime
        (len(dates)//3, 2*len(dates)//3, 'high_vol'),  # High volatility regime  
        (2*len(dates)//3, len(dates), 'crisis')        # Crisis regime
    ]
    
    for symbol in regime_symbols:
        base_price = {'SPY': 400, 'QQQ': 300, 'IWM': 200, 'VIX': 20, 'TLT': 120, 'GLD': 180}.get(symbol, 100)
        prices = [base_price]
        
        for i in range(1, len(dates)):
            # Determine current regime
            current_regime = None
            for start_idx, end_idx, regime_type in regime_periods:
                if start_idx <= i < end_idx:
                    current_regime = regime_type
                    break
            
            # Set volatility based on regime
            if current_regime == 'low_vol':
                vol = 0.008 if symbol != 'VIX' else 0.15
            elif current_regime == 'high_vol':
                vol = 0.020 if symbol != 'VIX' else 0.25
            else:  # crisis
                vol = 0.035 if symbol != 'VIX' else 0.40
            
            # VIX tends to be negatively correlated with equities
            if symbol == 'VIX' and current_regime == 'crisis':
                daily_return = abs(np.random.randn()) * vol  # VIX spikes in crisis
            elif symbol == 'VIX':
                daily_return = np.random.randn() * vol
            else:
                daily_return = np.random.randn() * vol
                if current_regime == 'crisis':
                    daily_return -= 0.001  # Slight negative drift in crisis
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        for i, (date_val, price) in enumerate(zip(dates, prices)):
            daily_return = np.random.randn() * 0.005
            open_price = price * (1 + np.random.randn() * 0.002)
            close_price = price
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.005)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.005)
            
            sample_data.append({
                'symbol': symbol,
                'timestamp': date_val,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000000, 50000000),
                'vwap': (open_price + high_price + low_price + close_price) / 4
            })
    
    regime_data = pd.DataFrame(sample_data)
    print(f"✅ Created sample regime dataset with {len(regime_data)} bars")

print(f"\nData shape: {regime_data.shape}")
print(f"Symbols: {regime_data['symbol'].unique()}")
print(f"Date range: {regime_data['timestamp'].min()} to {regime_data['timestamp'].max()}")

```
