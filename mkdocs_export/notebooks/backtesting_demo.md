# Alpha Model Backtest Demo

This notebook demonstrates alpha model training and backtesting capabilities.

## Overview

We'll explore:
1. Feature engineering for alpha models
2. Alpha model training with regime awareness
3. Model evaluation and performance metrics
4. Basic backtesting framework



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
from cross_asset_alpha_engine.features import DailyFeatureEngine, CrossAssetFeatureEngine
from cross_asset_alpha_engine.regimes import RegimeHMM, RegimeFeatureEngine
from cross_asset_alpha_engine.models import AlphaModel
from cross_asset_alpha_engine.models.alpha_model import AlphaModelConfig
from cross_asset_alpha_engine.utils import setup_logger, plot_equity_curve

# Setup
logger = setup_logger("alpha_demo", console_output=True)
print("✅ All imports successful!")

```
