# Data Sanity Check

This notebook demonstrates basic data loading and validation functionality of the Cross-Asset Alpha Engine.

## Prerequisites

1. Ensure you have set your Polygon API key in the `.env` file
2. Install the package: `pip install -e .`
3. Activate the virtual environment: `source .venv/bin/activate`



```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Cross-Asset Alpha Engine components
from cross_asset_alpha_engine.data import (
    PolygonClient, 
    DataCache, 
    AssetUniverse,
    load_daily_bars
)
# Note: Only daily OHLCV bars are used. No intraday data is loaded.
from cross_asset_alpha_engine.utils import setup_logger

# Setup logging
logger = setup_logger("data_check", console_output=True)
print("✅ All imports successful!")

```

    ✅ All imports successful!


## 1. Asset Universe Exploration



```python
# Initialize asset universe
universe = AssetUniverse()

# Get universe statistics
stats = universe.get_universe_stats()
print("Asset Universe Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Get equity symbols
equity_symbols = universe.get_equity_symbols()
print(f"\nAvailable equity symbols ({len(equity_symbols)}):")
print(equity_symbols[:10])  # Show first 10

# Get regime detection symbols
regime_symbols = universe.get_market_regime_symbols()
print(f"\nRegime detection symbols: {regime_symbols}")

# Get cross-asset symbols by class
cross_asset = universe.get_cross_asset_symbols()
print(f"\nCross-asset symbols by class:")
for asset_class, symbols in cross_asset.items():
    print(f"  {asset_class}: {symbols[:5]}")  # Show first 5 per class

```

    Asset Universe Statistics:
      total_assets: 32
      active_assets: 32
      asset_class_counts: {'equity': 26, 'bond': 3, 'commodity': 3}
      exchange_counts: {'NYSE': 23, 'NASDAQ': 8, 'CBOE': 1}
    
    Available equity symbols (26):
    ['AAPL', 'AMZN', 'BRK.B', 'GOOGL', 'IWM', 'JNJ', 'JPM', 'META', 'MSFT', 'NVDA']
    
    Regime detection symbols: ['SPY', 'QQQ', 'IWM', 'VIX', 'TLT', 'GLD']
    
    Cross-asset symbols by class:
      equity: ['AAPL', 'AMZN', 'BRK.B', 'GOOGL', 'IWM']
      commodity: ['GLD', 'SLV', 'USO']
      bond: ['HYG', 'IEF', 'TLT']


## 2. Data Loading Test



```python
# Define test parameters
test_symbols = ["SPY", "QQQ", "AAPL"]
end_date = date.today()
start_date = end_date - timedelta(days=30)  # Last 30 days

print(f"Loading data for {test_symbols} from {start_date} to {end_date}")

# Load daily data
try:
    daily_data = load_daily_bars(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    if not daily_data.empty:
        print(f"✅ Successfully loaded {len(daily_data)} daily bars")
        print(f"Date range: {daily_data['timestamp'].min()} to {daily_data['timestamp'].max()}")
        print(f"Symbols: {daily_data['symbol'].unique()}")
        
        # Display sample data
        print("\nSample data:")
        print(daily_data.head())
    else:
        print("⚠️ No data returned - this may be due to API key issues or market hours")
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Note: This requires a valid Polygon API key in the .env file")
    
    # Create sample data for demonstration if API fails
    print("\n📊 Creating sample data for demonstration...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = []
    
    for symbol in test_symbols:
        base_price = 100 if symbol == "SPY" else 200 if symbol == "QQQ" else 150
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.02)
        
        for i, date_val in enumerate(dates):
            sample_data.append({
                'symbol': symbol,
                'timestamp': date_val,
                'open': prices[i] * (1 + np.random.randn() * 0.001),
                'high': prices[i] * (1 + abs(np.random.randn()) * 0.005),
                'low': prices[i] * (1 - abs(np.random.randn()) * 0.005),
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000),
                'vwap': prices[i] * (1 + np.random.randn() * 0.0005)
            })
    
    daily_data = pd.DataFrame(sample_data)
    print(f"✅ Created sample dataset with {len(daily_data)} bars")
    print(daily_data.head())

```

    Loading data for ['SPY', 'QQQ', 'AAPL'] from 2025-11-11 to 2025-12-11
    Fetching SPY daily data from API...
    No data returned for SPY
    Fetching QQQ daily data from API...
    No data returned for QQQ
    Fetching AAPL daily data from API...
    Rate limited. Waiting 1.0s before retry...
    Rate limited. Waiting 2.0s before retry...
    Rate limited. Waiting 4.0s before retry...
    Error fetching data for AAPL: Rate limited after 3 retries
    ⚠️ No data returned - this may be due to API key issues or market hours

