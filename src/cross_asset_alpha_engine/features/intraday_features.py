"""Daily microstructure-inspired feature engineering.

IMPORTANT: This module computes features from daily OHLCV bars, not true intraday or tick data.
All features are inspired by microstructure concepts but computed from daily bars only.
No intraday, tick, or order-book data is used in the current experiment.

This module provides feature extraction capabilities for daily market data,
including rolling returns, volatility, VWAP deviations, and momentum indicators
computed from daily OHLCV bars.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from ..config import DEFAULT_LOOKBACK_PERIODS, DEFAULT_VOLATILITY_WINDOW, DEFAULT_VOLUME_ZSCORE_WINDOW
from ..utils.logging_utils import LoggerMixin


@dataclass
class IntradayFeatureConfig:
    """Configuration for intraday feature generation."""
    lookback_periods: List[int] = None
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW
    volume_zscore_window: int = DEFAULT_VOLUME_ZSCORE_WINDOW
    vwap_deviation_windows: List[int] = None
    momentum_windows: List[int] = None
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = DEFAULT_LOOKBACK_PERIODS.copy()
        
        if self.vwap_deviation_windows is None:
            self.vwap_deviation_windows = [5, 10, 20]
        
        if self.momentum_windows is None:
            self.momentum_windows = [3, 5, 10, 20]


class IntradayFeatureEngine(LoggerMixin):
    """Engine for generating daily microstructure-inspired features from daily OHLCV bars.
    
    Note: Despite the name, this engine works with daily OHLCV bars, not true intraday data.
    Features are inspired by microstructure concepts but computed from daily bars only.
    """
    
    def __init__(self, config: Optional[IntradayFeatureConfig] = None):
        """Initialize daily microstructure-inspired feature engine.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or IntradayFeatureConfig()
        self.logger.info(f"Initialized IntradayFeatureEngine with config: {self.config}")
    
    def calculate_rolling_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate rolling returns for multiple periods from daily OHLCV bars.
        
        Note: Despite the "m" suffix in feature names, these are daily periods, not minutes.
        All calculations use daily OHLCV bars only.
        
        Args:
            df: DataFrame with daily OHLCV data
            price_col: Column name for price data
            periods: List of periods (in days) for rolling returns
            
        Returns:
            DataFrame with rolling return features
        """
        if periods is None:
            periods = self.config.lookback_periods
        
        result = df.copy()
        
        for period in periods:
            # Simple return (daily bars, period in days)
            result[f"return_{period}m"] = df[price_col].pct_change(period)
            
            # Log return (daily bars, period in days)
            result[f"log_return_{period}m"] = np.log(df[price_col] / df[price_col].shift(period))
        
        return result
    
    def calculate_rolling_volatility(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate rolling volatility measures from daily OHLCV bars.
        
        Args:
            df: DataFrame with daily OHLCV data
            price_col: Column name for price data
            windows: List of windows (in days) for volatility calculation
            
        Returns:
            DataFrame with volatility features computed from daily bars
        """
        if windows is None:
            windows = [5, 10, 20, 60]  # 5-day, 10-day, 20-day, 60-day (daily bars)
        
        result = df.copy()
        
        # Calculate returns first
        returns = df[price_col].pct_change()
        
        for window in windows:
            # Rolling standard deviation
            result[f"volatility_{window}m"] = returns.rolling(window).std()
            
            # Realized volatility (sum of squared returns)
            result[f"realized_vol_{window}m"] = (returns ** 2).rolling(window).sum()
            
            # Parkinson volatility (using high-low range)
            if "high" in df.columns and "low" in df.columns:
                hl_ratio = np.log(df["high"] / df["low"])
                result[f"parkinson_vol_{window}m"] = np.sqrt(
                    (hl_ratio ** 2).rolling(window).mean() / (4 * np.log(2))
                )
        
        return result
    
    def calculate_vwap_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate VWAP-based features.
        
        Args:
            df: DataFrame with OHLCV data including vwap column
            windows: List of windows for VWAP calculations
            
        Returns:
            DataFrame with VWAP features
        """
        if windows is None:
            windows = self.config.vwap_deviation_windows
        
        result = df.copy()
        
        if "vwap" not in df.columns:
            self.logger.warning("VWAP column not found, calculating from price/volume")
            # Calculate VWAP if not present
            result["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        
        # Current price vs VWAP
        result["price_vwap_ratio"] = df["close"] / result["vwap"]
        result["price_vwap_deviation"] = (df["close"] - result["vwap"]) / result["vwap"]
        
        for window in windows:
            # Rolling VWAP
            rolling_vwap = (
                (df["close"] * df["volume"]).rolling(window).sum() / 
                df["volume"].rolling(window).sum()
            )
            result[f"rolling_vwap_{window}m"] = rolling_vwap
            
            # Price vs rolling VWAP
            result[f"price_rvwap_ratio_{window}m"] = df["close"] / rolling_vwap
            result[f"price_rvwap_deviation_{window}m"] = (df["close"] - rolling_vwap) / rolling_vwap
        
        return result
    
    def calculate_volume_features(
        self,
        df: pd.DataFrame,
        zscore_window: Optional[int] = None
    ) -> pd.DataFrame:
        """Calculate volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            zscore_window: Window for volume z-score calculation
            
        Returns:
            DataFrame with volume features
        """
        if zscore_window is None:
            zscore_window = self.config.volume_zscore_window
        
        result = df.copy()
        
        # Volume z-score
        vol_mean = df["volume"].rolling(zscore_window).mean()
        vol_std = df["volume"].rolling(zscore_window).std()
        result["volume_zscore"] = (df["volume"] - vol_mean) / vol_std
        
        # Volume rate of change
        for period in [1, 3, 5, 10]:
            result[f"volume_roc_{period}m"] = df["volume"].pct_change(period)
        
        # Volume moving averages
        for window in [5, 10, 20]:
            result[f"volume_ma_{window}m"] = df["volume"].rolling(window).mean()
            result[f"volume_ratio_{window}m"] = df["volume"] / result[f"volume_ma_{window}m"]
        
        # Dollar volume
        result["dollar_volume"] = df["close"] * df["volume"]
        result["dollar_volume_ma_20m"] = result["dollar_volume"].rolling(20).mean()
        result["dollar_volume_ratio"] = result["dollar_volume"] / result["dollar_volume_ma_20m"]
        
        return result
    
    def calculate_bar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bar-specific features (range, gaps, etc.).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with bar features
        """
        result = df.copy()
        
        # Bar ranges
        result["bar_range"] = (df["high"] - df["low"]) / df["close"]
        result["upper_shadow"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["close"]
        result["lower_shadow"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["close"]
        result["body_size"] = np.abs(df["close"] - df["open"]) / df["close"]
        
        # Gap features
        result["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        result["gap_filled"] = (
            (result["gap"] > 0) & (df["low"] <= df["close"].shift(1)) |
            (result["gap"] < 0) & (df["high"] >= df["close"].shift(1))
        ).astype(int)
        
        # Price position within bar
        result["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        result["price_position"] = result["price_position"].fillna(0.5)  # Handle zero-range bars
        
        # Rolling bar statistics
        for window in [5, 10, 20]:
            result[f"avg_bar_range_{window}m"] = result["bar_range"].rolling(window).mean()
            result[f"bar_range_zscore_{window}m"] = (
                (result["bar_range"] - result[f"avg_bar_range_{window}m"]) /
                result["bar_range"].rolling(window).std()
            )
        
        return result
    
    def calculate_momentum_features(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate momentum and trend features.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            windows: List of windows for momentum calculations
            
        Returns:
            DataFrame with momentum features
        """
        if windows is None:
            windows = self.config.momentum_windows
        
        result = df.copy()
        
        for window in windows:
            # Simple moving average
            ma = df[price_col].rolling(window).mean()
            result[f"ma_{window}m"] = ma
            result[f"price_ma_ratio_{window}m"] = df[price_col] / ma
            result[f"price_ma_deviation_{window}m"] = (df[price_col] - ma) / ma
            
            # Momentum (rate of change)
            result[f"momentum_{window}m"] = df[price_col] / df[price_col].shift(window) - 1
            
            # RSI-like indicator
            returns = df[price_col].pct_change()
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            
            rs = avg_gains / avg_losses
            result[f"rsi_{window}m"] = 100 - (100 / (1 + rs))
        
        # Trend strength
        for window in [10, 20]:
            # Linear regression slope
            x = np.arange(window)
            slopes = []
            
            for i in range(len(df)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = df[price_col].iloc[i-window+1:i+1].values
                    if len(y) == window:
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope / df[price_col].iloc[i])  # Normalize by price
                    else:
                        slopes.append(np.nan)
            
            result[f"trend_slope_{window}m"] = slopes
        
        return result
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        forward_periods: List[int] = None,
        threshold: float = 0.001
    ) -> pd.DataFrame:
        """Generate forward-looking labels for supervised learning.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            forward_periods: List of forward-looking periods
            threshold: Threshold for binary classification
            
        Returns:
            DataFrame with label features
        """
        if forward_periods is None:
            forward_periods = [1, 3, 5, 10]
        
        result = df.copy()
        
        for period in forward_periods:
            # Forward returns
            forward_return = df[price_col].shift(-period) / df[price_col] - 1
            result[f"forward_return_{period}m"] = forward_return
            
            # Binary labels
            result[f"forward_up_{period}m"] = (forward_return > threshold).astype(int)
            result[f"forward_down_{period}m"] = (forward_return < -threshold).astype(int)
            
            # Categorical labels
            result[f"forward_direction_{period}m"] = np.select(
                [forward_return > threshold, forward_return < -threshold],
                [1, -1],
                default=0
            )
            
            # Forward high/low within period
            if period > 1:
                forward_high = df["high"].rolling(period).max().shift(-period+1)
                forward_low = df["low"].rolling(period).min().shift(-period+1)
                
                result[f"forward_high_return_{period}m"] = forward_high / df[price_col] - 1
                result[f"forward_low_return_{period}m"] = forward_low / df[price_col] - 1
        
        return result
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_labels: bool = False
    ) -> pd.DataFrame:
        """Generate all intraday features.
        
        Args:
            df: DataFrame with OHLCV data
            include_labels: Whether to include forward-looking labels
            
        Returns:
            DataFrame with all intraday features
        """
        self.logger.info(f"Generating intraday features for {len(df)} bars")
        
        # Start with original data
        result = df.copy()
        
        # Generate all feature types
        result = self.calculate_rolling_returns(result)
        result = self.calculate_rolling_volatility(result)
        result = self.calculate_vwap_features(result)
        result = self.calculate_volume_features(result)
        result = self.calculate_bar_features(result)
        result = self.calculate_momentum_features(result)
        
        if include_labels:
            result = self.generate_labels(result)
        
        # Log feature generation results
        feature_cols = [col for col in result.columns if col not in df.columns]
        self.logger.info(f"Generated {len(feature_cols)} intraday features")
        
        return result
    
    def get_feature_names(self, include_labels: bool = False) -> List[str]:
        """Get list of all feature names that would be generated.
        
        Args:
            include_labels: Whether to include label feature names
            
        Returns:
            List of feature names
        """
        features = []
        
        # Rolling returns
        for period in self.config.lookback_periods:
            features.extend([f"return_{period}m", f"log_return_{period}m"])
        
        # Volatility features
        for window in [5, 10, 20, 60]:
            features.extend([
                f"volatility_{window}m",
                f"realized_vol_{window}m",
                f"parkinson_vol_{window}m"
            ])
        
        # VWAP features
        features.extend(["price_vwap_ratio", "price_vwap_deviation"])
        for window in self.config.vwap_deviation_windows:
            features.extend([
                f"rolling_vwap_{window}m",
                f"price_rvwap_ratio_{window}m",
                f"price_rvwap_deviation_{window}m"
            ])
        
        # Volume features
        features.extend(["volume_zscore", "dollar_volume", "dollar_volume_ratio"])
        for period in [1, 3, 5, 10]:
            features.append(f"volume_roc_{period}m")
        for window in [5, 10, 20]:
            features.extend([
                f"volume_ma_{window}m",
                f"volume_ratio_{window}m"
            ])
        
        # Bar features
        features.extend([
            "bar_range", "upper_shadow", "lower_shadow", "body_size",
            "gap", "gap_filled", "price_position"
        ])
        for window in [5, 10, 20]:
            features.extend([
                f"avg_bar_range_{window}m",
                f"bar_range_zscore_{window}m"
            ])
        
        # Momentum features
        for window in self.config.momentum_windows:
            features.extend([
                f"ma_{window}m",
                f"price_ma_ratio_{window}m",
                f"price_ma_deviation_{window}m",
                f"momentum_{window}m",
                f"rsi_{window}m"
            ])
        for window in [10, 20]:
            features.append(f"trend_slope_{window}m")
        
        # Labels
        if include_labels:
            for period in [1, 3, 5, 10]:
                features.extend([
                    f"forward_return_{period}m",
                    f"forward_up_{period}m",
                    f"forward_down_{period}m",
                    f"forward_direction_{period}m"
                ])
                if period > 1:
                    features.extend([
                        f"forward_high_return_{period}m",
                        f"forward_low_return_{period}m"
                    ])
        
        return features
