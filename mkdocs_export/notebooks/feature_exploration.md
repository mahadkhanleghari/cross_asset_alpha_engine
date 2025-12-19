# Feature Exploration

This notebook demonstrates the feature engineering capabilities of the Cross-Asset Alpha Engine.

## Overview

We'll explore:
1. Daily feature engineering
2. Intraday feature engineering  
3. Cross-asset feature engineering
4. Feature analysis and visualization



```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Cross-Asset Alpha Engine components
from cross_asset_alpha_engine.data import load_daily_bars, AssetUniverse
from cross_asset_alpha_engine.features import (
    DailyFeatureEngine, 
    IntradayFeatureEngine,
    CrossAssetFeatureEngine
)
from cross_asset_alpha_engine.utils import setup_logger

# Setup
logger = setup_logger("feature_exploration", console_output=True)
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
print("✅ All imports successful!")

```

    ✅ All imports successful!


## 🔍 API Key Diagnostic Test



```python
# 🔍 JUPYTER API KEY DIAGNOSTIC
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("🔍 Jupyter Kernel API Key Diagnostic")
print("=" * 50)

# Check current working directory
print(f"Current directory: {os.getcwd()}")

# Check if .env file exists in current directory
env_file = Path('.env')
print(f".env file exists: {env_file.exists()}")

if env_file.exists():
    print(f".env file path: {env_file.absolute()}")
    # Read and show first few chars of API key from file
    with open('.env', 'r') as f:
        content = f.read()
        if 'POLYGON_API_KEY=' in content:
            key_line = [line for line in content.split('\n') if line.startswith('POLYGON_API_KEY=')][0]
            key_value = key_line.split('=', 1)[1].strip()
            if key_value and key_value != 'YOUR_KEY_HERE':
                masked = key_value[:4] + '*' * (len(key_value) - 8) + key_value[-4:]
                print(f"✅ .env file contains key: {masked}")
            else:
                print("❌ .env file has placeholder or empty key")
        else:
            print("❌ POLYGON_API_KEY not found in .env file")

# Force reload .env
print("\n🔄 Force loading .env file...")
load_dotenv(override=True)

# Check environment variable
api_key = os.getenv('POLYGON_API_KEY')
if api_key and api_key != 'YOUR_KEY_HERE':
    masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
    print(f"✅ Environment variable: {masked_key}")
else:
    print("❌ Environment variable not set or is placeholder")

# Test the cross_asset_alpha_engine config
try:
    from cross_asset_alpha_engine.config import POLYGON_API_KEY
    if POLYGON_API_KEY and POLYGON_API_KEY != 'YOUR_KEY_HERE':
        masked = POLYGON_API_KEY[:4] + '*' * (len(POLYGON_API_KEY) - 8) + POLYGON_API_KEY[-4:]
        print(f"✅ Config module: {masked}")
    else:
        print("❌ Config module: No key or placeholder")
except Exception as e:
    print(f"❌ Config module error: {e}")

print("\n" + "=" * 50)

```

    🔍 Jupyter Kernel API Key Diagnostic
    ==================================================
    Current directory: /Users/mahadafzal/Projects/cross_asset_alpha_engine/notebooks
    .env file exists: False
    
    🔄 Force loading .env file...
    ✅ Environment variable: 1qhW************************Kvpf
    ✅ Config module: 1qhW************************Kvpf
    
    ==================================================


## 🧪 Test API Connection



```python
# 🧪 Test actual API call with corrected date range
print("🧪 Testing actual API call...")

try:
    from cross_asset_alpha_engine.data import load_daily_bars
    from datetime import date
    
    # Use working date range
    end_date = date(2025, 12, 6)
    start_date = date(2025, 11, 25)
    
    print(f"Loading SPY data from {start_date} to {end_date}")
    data = load_daily_bars(['SPY'], start_date, end_date)
    
    if not data.empty:
        print(f"✅ API call successful: {len(data)} bars loaded")
        print(f"Latest SPY price: ${data['close'].iloc[-1]:.2f}")
        print("🎉 API key is working in Jupyter!")
    else:
        print("❌ API call returned no data")
        
except Exception as e:
    print(f"❌ API call failed: {e}")
    print("\n🔧 Try this fix:")
    print("1. Restart the kernel (Kernel → Restart)")
    print("2. Re-run all cells")
    print("3. Or add this at the top of your notebook:")
    print("   from dotenv import load_dotenv")
    print("   load_dotenv(override=True)")

```

    🧪 Testing actual API call...
    Loading SPY data from 2025-11-25 to 2025-12-06
    Loaded SPY daily data from cache
    ✅ API call successful: 8 bars loaded
    Latest SPY price: $685.69
    🎉 API key is working in Jupyter!


## 🔧 Fix: Load Data with Corrected Parameters



```python
# Load data for feature engineering with CORRECTED date range
symbols = ["AAPL", "SPY", "QQQ"]

# 🔧 FIXED: Use specific dates that work with the API
end_date = date(2025, 12, 6)    # Recent Friday
start_date = date(2025, 11, 15)  # 3 weeks back (shorter range = more reliable)

print(f"📊 Loading data for {symbols}")
print(f"📅 Date range: {start_date} to {end_date}")

# Load symbols one at a time to avoid rate limits
all_data = []
for symbol in symbols:
    print(f"\n🔄 Loading {symbol}...")
    try:
        symbol_data = load_daily_bars([symbol], start_date, end_date, use_cache=True)
        if not symbol_data.empty:
            all_data.append(symbol_data)
            latest_price = symbol_data['close'].iloc[-1]
            print(f"✅ {symbol}: {len(symbol_data)} bars, latest: ${latest_price:.2f}")
        else:
            print(f"⚠️ {symbol}: No data returned")
    except Exception as e:
        print(f"❌ {symbol}: Error - {e}")
    
    # Small delay to avoid rate limits
    import time
    time.sleep(0.3)

# Combine all data
if all_data:
    data = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ SUCCESS: Loaded {len(data)} total bars from API")
    print(f"📈 Symbols: {data['symbol'].unique()}")
    print(f"📅 Actual date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    print("\n📋 Sample real data:")
    print(data.head())
else:
    print("\n⚠️ No real data loaded, creating sample data...")
    # Fallback to sample data creation
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = []
    
    for symbol in symbols:
        base_price = 150 if symbol == "AAPL" else 400 if symbol == "SPY" else 300
        prices = base_price * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015))
        
        for i, date_val in enumerate(dates):
            daily_return = np.random.randn() * 0.02
            open_price = prices[i] * (1 + np.random.randn() * 0.005)
            close_price = open_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.01)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.01)
            
            sample_data.append({
                'symbol': symbol,
                'timestamp': date_val,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(10000000, 100000000),
                'vwap': (open_price + high_price + low_price + close_price) / 4
            })
    
    data = pd.DataFrame(sample_data)
    print(f"📊 Created sample dataset with {len(data)} bars")

print(f"\n📊 Final data shape: {data.shape}")
print(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print(f"🎯 Ready for feature engineering!")

```

    📊 Loading data for ['AAPL', 'SPY', 'QQQ']
    📅 Date range: 2025-11-15 to 2025-12-06
    
    🔄 Loading AAPL...
    Loaded AAPL daily data from cache
    ✅ AAPL: 14 bars, latest: $278.78
    
    🔄 Loading SPY...
    Loaded SPY daily data from cache
    ✅ SPY: 14 bars, latest: $685.69
    
    🔄 Loading QQQ...
    Loaded QQQ daily data from cache
    ✅ QQQ: 14 bars, latest: $625.48
    
    ✅ SUCCESS: Loaded 42 total bars from API
    📈 Symbols: ['AAPL' 'SPY' 'QQQ']
    📅 Actual date range: 2025-11-17 05:00:00 to 2025-12-05 05:00:00
    
    📋 Sample real data:
      symbol           timestamp     open    high     low   close      volume  \
    0   AAPL 2025-11-17 05:00:00  268.815  270.49  265.73  267.46  44958759.0   
    1   AAPL 2025-11-18 05:00:00  269.990  270.71  265.32  267.44  45677270.0   
    2   AAPL 2025-11-19 05:00:00  265.525  272.21  265.50  268.56  40334193.0   
    3   AAPL 2025-11-20 05:00:00  270.830  275.43  265.92  266.25  45728132.0   
    4   AAPL 2025-11-21 05:00:00  265.950  273.33  265.67  271.49  58923249.0   
    
           vwap  
    0  267.9843  
    1  267.7250  
    2  269.3236  
    3  269.4688  
    4  270.5143  
    
    📊 Final data shape: (42, 8)
    📅 Date range: 2025-11-17 05:00:00 to 2025-12-05 05:00:00
    🎯 Ready for feature engineering!


## 1. Load Sample Data



```python
# Load data for feature engineering
symbols = ["AAPL", "SPY", "QQQ"]
end_date = date.today()
start_date = end_date - timedelta(days=90)  # 3 months of data

print(f"Loading data for {symbols} from {start_date} to {end_date}")

try:
    data = load_daily_bars(symbols, start_date, end_date, use_cache=True)
    if data.empty:
        raise ValueError("No data returned from API")
    print(f"✅ Loaded {len(data)} bars from API")
except Exception as e:
    print(f"⚠️ API error: {e}")
    print("📊 Creating sample data for demonstration...")
    
    # Create realistic sample data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = []
    
    for symbol in symbols:
        base_price = 150 if symbol == "AAPL" else 400 if symbol == "SPY" else 300
        prices = base_price * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015))
        
        for i, date_val in enumerate(dates):
            daily_return = np.random.randn() * 0.02
            open_price = prices[i] * (1 + np.random.randn() * 0.005)
            close_price = open_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.01)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.01)
            
            sample_data.append({
                'symbol': symbol,
                'timestamp': date_val,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(10000000, 100000000),
                'vwap': (open_price + high_price + low_price + close_price) / 4
            })
    
    data = pd.DataFrame(sample_data)
    print(f"✅ Created sample dataset with {len(data)} bars")

print(f"\nData shape: {data.shape}")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print(f"Symbols: {data['symbol'].unique()}")
print("\nSample data:")
print(data.head())

```

    Loading data for ['AAPL', 'SPY', 'QQQ'] from 2025-09-13 to 2025-12-12
    Fetching AAPL daily data from API...
    No data returned for AAPL
    Fetching SPY daily data from API...
    Rate limited. Waiting 1.0s before retry...
    Rate limited. Waiting 2.0s before retry...
    Rate limited. Waiting 4.0s before retry...
    Error fetching data for SPY: Rate limited after 3 retries
    Fetching QQQ daily data from API...
    Rate limited. Waiting 1.0s before retry...
    Rate limited. Waiting 2.0s before retry...
    Rate limited. Waiting 4.0s before retry...
    No data returned for QQQ
    ⚠️ API error: No data returned from API
    📊 Creating sample data for demonstration...
    ✅ Created sample dataset with 273 bars
    
    Data shape: (273, 8)
    Date range: 2025-09-13 00:00:00 to 2025-12-12 00:00:00
    Symbols: ['AAPL' 'SPY' 'QQQ']
    
    Sample data:
      symbol  timestamp        open        high         low       close    volume  \
    0   AAPL 2025-09-13  151.364569  154.804483  147.882720  152.230457  85372076   
    1   AAPL 2025-09-14  150.071880  151.311341  147.779352  148.577250  90238489   
    2   AAPL 2025-09-15  152.153169  157.131092  149.310156  154.126730  89611686   
    3   AAPL 2025-09-16  151.840672  153.141383  147.157646  151.891077  97127284   
    4   AAPL 2025-09-17  147.221806  154.995306  146.284541  152.398097  86809828   
    
             vwap  
    0  151.570557  
    1  149.434956  
    2  153.180287  
    3  151.007694  
    4  150.224937  

