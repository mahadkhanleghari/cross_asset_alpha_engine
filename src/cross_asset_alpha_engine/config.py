"""Configuration module for Cross-Asset Alpha Engine.

This module loads environment variables and provides configuration settings
for the entire application.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
POLYGON_API_KEY: Optional[str] = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_KEY_HERE":
    import warnings
    warnings.warn(
        "POLYGON_API_KEY is not set in .env file. "
        "Some functionality will be limited without a valid API key.",
        UserWarning
    )

# Data Configuration
POLYGON_BASE_URL = "https://api.polygon.io"
DEFAULT_CACHE_DIR = "cache"
DEFAULT_DATA_DIR = "data"

# Feature Engineering Configuration
DEFAULT_LOOKBACK_PERIODS = [1, 3, 5, 10, 20]
DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_VOLUME_ZSCORE_WINDOW = 20

# Regime Detection Configuration
DEFAULT_N_REGIMES = 3
DEFAULT_REGIME_FEATURES = ["volatility", "volume", "cross_asset_vol_ratio"]

# Backtest Configuration
DEFAULT_INITIAL_CAPITAL = 1_000_000
DEFAULT_COMMISSION_RATE = 0.001  # 10 bps
DEFAULT_SLIPPAGE_RATE = 0.0005   # 5 bps

# Execution Simulation Configuration
DEFAULT_PARTICIPATION_RATE = 0.1  # 10% of volume
DEFAULT_SLIPPAGE_COEFFICIENT = 0.1

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Trading Configuration
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
TIMEZONE = "America/New_York"
