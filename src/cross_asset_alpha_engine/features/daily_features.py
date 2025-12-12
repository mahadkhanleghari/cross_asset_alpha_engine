"""Daily feature engineering for longer-term alpha generation.

This module provides feature extraction capabilities for daily market data,
including multi-day returns, realized volatility, gap analysis, and turnover metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from ..utils.logging_utils import LoggerMixin


@dataclass
class DailyFeatureConfig:
    """Configuration for daily feature generation."""
    return_periods: List[int] = None
    volatility_windows: List[int] = None
    momentum_windows: List[int] = None
    gap_analysis_periods: List[int] = None
    
    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = [1, 2, 3, 5, 10, 20, 60]
        
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60, 120]
        
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20, 50, 200]
        
        if self.gap_analysis_periods is None:
            self.gap_analysis_periods = [5, 10, 20]


class DailyFeatureEngine(LoggerMixin):
    """Engine for generating daily features from daily market data."""
    
    def __init__(self, config: Optional[DailyFeatureConfig] = None):
        """Initialize daily feature engine.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or DailyFeatureConfig()
        self.logger.info(f"Initialized DailyFeatureEngine with config: {self.config}")
    
    def calculate_multi_day_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate multi-day returns.
        
        Args:
            df: DataFrame with daily OHLCV data
            price_col: Column name for price data
            periods: List of periods for return calculation
            
        Returns:
            DataFrame with return features
        """
        if periods is None:
            periods = self.config.return_periods
        
        result = df.copy()
        
        for period in periods:
            # Simple returns
            result[f"return_{period}d"] = df[price_col].pct_change(period)
            
            # Log returns
            result[f"log_return_{period}d"] = np.log(df[price_col] / df[price_col].shift(period))
            
            # Cumulative returns
            result[f"cum_return_{period}d"] = (1 + df[price_col].pct_change()).rolling(period).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
            
            # Forward returns (for labels)
            result[f"forward_return_{period}d"] = df[price_col].shift(-period) / df[price_col] - 1
        
        return result
    
    def calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate realized volatility measures.
        
        Args:
            df: DataFrame with daily OHLCV data
            price_col: Column name for price data
            windows: List of windows for volatility calculation
            
        Returns:
            DataFrame with volatility features
        """
        if windows is None:
            windows = self.config.volatility_windows
        
        result = df.copy()
        
        # Daily returns
        returns = df[price_col].pct_change()
        log_returns = np.log(df[price_col] / df[price_col].shift(1))
        
        for window in windows:
            # Standard volatility (annualized)
            result[f"volatility_{window}d"] = returns.rolling(window).std() * np.sqrt(252)
            
            # Realized volatility (sum of squared returns, annualized)
            result[f"realized_vol_{window}d"] = np.sqrt(
                (returns ** 2).rolling(window).sum() * 252 / window
            )
            
            # Log return volatility
            result[f"log_vol_{window}d"] = log_returns.rolling(window).std() * np.sqrt(252)
            
            # Parkinson volatility (using high-low range)
            if "high" in df.columns and "low" in df.columns:
                hl_ratio = np.log(df["high"] / df["low"])
                result[f"parkinson_vol_{window}d"] = np.sqrt(
                    (hl_ratio ** 2).rolling(window).mean() * 252 / (4 * np.log(2))
                )
            
            # Garman-Klass volatility
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                # GK estimator components
                hl_component = np.log(df["high"] / df["low"]) ** 2
                co_component = np.log(df["close"] / df["open"]) ** 2
                
                gk_vol = np.sqrt(
                    (0.5 * hl_component - (2 * np.log(2) - 1) * co_component).rolling(window).mean() * 252
                )
                result[f"gk_vol_{window}d"] = gk_vol
            
            # Volatility of volatility
            if window >= 10:
                vol_series = returns.rolling(10).std()
                result[f"vol_of_vol_{window}d"] = vol_series.rolling(window).std()
        
        return result
    
    def calculate_gap_features(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate gap analysis features.
        
        Args:
            df: DataFrame with daily OHLCV data
            periods: List of periods for gap analysis
            
        Returns:
            DataFrame with gap features
        """
        if periods is None:
            periods = self.config.gap_analysis_periods
        
        result = df.copy()
        
        if "open" not in df.columns:
            self.logger.warning("Open prices not available for gap analysis")
            return result
        
        # Daily gap
        result["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        result["gap_abs"] = np.abs(result["gap"])
        
        # Gap direction
        result["gap_up"] = (result["gap"] > 0).astype(int)
        result["gap_down"] = (result["gap"] < 0).astype(int)
        
        # Gap magnitude categories
        gap_percentiles = result["gap_abs"].quantile([0.5, 0.75, 0.9, 0.95])
        result["gap_size_category"] = pd.cut(
            result["gap_abs"],
            bins=[0] + gap_percentiles.tolist() + [np.inf],
            labels=["small", "medium", "large", "very_large", "extreme"]
        )
        
        # Gap fill analysis
        result["gap_filled_same_day"] = (
            ((result["gap"] > 0) & (df["low"] <= df["close"].shift(1))) |
            ((result["gap"] < 0) & (df["high"] >= df["close"].shift(1)))
        ).astype(int)
        
        # Rolling gap statistics
        for period in periods:
            result[f"avg_gap_{period}d"] = result["gap"].rolling(period).mean()
            result[f"gap_volatility_{period}d"] = result["gap"].rolling(period).std()
            result[f"gap_up_freq_{period}d"] = result["gap_up"].rolling(period).mean()
            result[f"gap_fill_freq_{period}d"] = result["gap_filled_same_day"].rolling(period).mean()
        
        return result
    
    def calculate_turnover_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate turnover and liquidity proxy features.
        
        Args:
            df: DataFrame with daily OHLCV data
            windows: List of windows for turnover calculations
            
        Returns:
            DataFrame with turnover features
        """
        if windows is None:
            windows = [5, 10, 20, 60]
        
        result = df.copy()
        
        if "volume" not in df.columns:
            self.logger.warning("Volume data not available for turnover analysis")
            return result
        
        # Dollar volume
        result["dollar_volume"] = df["close"] * df["volume"]
        
        # Volume statistics
        for window in windows:
            # Average volume
            result[f"avg_volume_{window}d"] = df["volume"].rolling(window).mean()
            result[f"volume_ratio_{window}d"] = df["volume"] / result[f"avg_volume_{window}d"]
            
            # Volume volatility
            result[f"volume_volatility_{window}d"] = df["volume"].rolling(window).std()
            result[f"volume_zscore_{window}d"] = (
                (df["volume"] - result[f"avg_volume_{window}d"]) / result[f"volume_volatility_{window}d"]
            )
            
            # Dollar volume statistics
            result[f"avg_dollar_volume_{window}d"] = result["dollar_volume"].rolling(window).mean()
            result[f"dollar_volume_ratio_{window}d"] = (
                result["dollar_volume"] / result[f"avg_dollar_volume_{window}d"]
            )
        
        # Volume-price relationship
        returns = df["close"].pct_change()
        result["volume_return_corr_20d"] = returns.rolling(20).corr(df["volume"])
        
        # Turnover rate (if shares outstanding available)
        # This would require additional data, so we'll create a proxy
        result["relative_volume"] = df["volume"] / df["volume"].rolling(252).mean()
        
        return result
    
    def calculate_momentum_features(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate momentum and trend features.
        
        Args:
            df: DataFrame with daily OHLCV data
            price_col: Column name for price data
            windows: List of windows for momentum calculations
            
        Returns:
            DataFrame with momentum features
        """
        if windows is None:
            windows = self.config.momentum_windows
        
        result = df.copy()
        
        for window in windows:
            # Moving averages
            ma = df[price_col].rolling(window).mean()
            result[f"ma_{window}d"] = ma
            result[f"price_ma_ratio_{window}d"] = df[price_col] / ma
            result[f"price_ma_deviation_{window}d"] = (df[price_col] - ma) / ma
            
            # Momentum
            result[f"momentum_{window}d"] = df[price_col] / df[price_col].shift(window) - 1
            
            # RSI
            returns = df[price_col].pct_change()
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            
            rs = avg_gains / avg_losses
            result[f"rsi_{window}d"] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            if window >= 20:
                bb_std = df[price_col].rolling(window).std()
                result[f"bb_upper_{window}d"] = ma + 2 * bb_std
                result[f"bb_lower_{window}d"] = ma - 2 * bb_std
                result[f"bb_position_{window}d"] = (
                    (df[price_col] - result[f"bb_lower_{window}d"]) /
                    (result[f"bb_upper_{window}d"] - result[f"bb_lower_{window}d"])
                )
                result[f"bb_width_{window}d"] = (
                    (result[f"bb_upper_{window}d"] - result[f"bb_lower_{window}d"]) / ma
                )
        
        # MACD
        if 12 in windows and 26 in windows:
            ema12 = df[price_col].ewm(span=12).mean()
            ema26 = df[price_col].ewm(span=26).mean()
            result["macd"] = ema12 - ema26
            result["macd_signal"] = result["macd"].ewm(span=9).mean()
            result["macd_histogram"] = result["macd"] - result["macd_signal"]
        
        # Trend strength
        for window in [20, 50]:
            if window in windows:
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
                            slopes.append(slope / df[price_col].iloc[i])  # Normalize
                        else:
                            slopes.append(np.nan)
                
                result[f"trend_slope_{window}d"] = slopes
                
                # R-squared of trend
                r_squared = []
                for i in range(len(df)):
                    if i < window - 1:
                        r_squared.append(np.nan)
                    else:
                        y = df[price_col].iloc[i-window+1:i+1].values
                        if len(y) == window:
                            correlation_matrix = np.corrcoef(x, y)
                            r_sq = correlation_matrix[0, 1] ** 2
                            r_squared.append(r_sq)
                        else:
                            r_squared.append(np.nan)
                
                result[f"trend_strength_{window}d"] = r_squared
        
        return result
    
    def calculate_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate seasonal and calendar features.
        
        Args:
            df: DataFrame with daily OHLCV data and datetime index
            
        Returns:
            DataFrame with seasonal features
        """
        result = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                result = result.set_index("timestamp")
            else:
                self.logger.warning("No datetime index or timestamp column found")
                return result
        
        # Day of week (Monday=0, Sunday=6)
        result["day_of_week"] = result.index.dayofweek
        result["is_monday"] = (result["day_of_week"] == 0).astype(int)
        result["is_friday"] = (result["day_of_week"] == 4).astype(int)
        
        # Month
        result["month"] = result.index.month
        result["quarter"] = result.index.quarter
        
        # Day of month
        result["day_of_month"] = result.index.day
        result["is_month_end"] = (result.index.day >= 28).astype(int)  # Approximate
        result["is_month_start"] = (result.index.day <= 3).astype(int)
        
        # Year
        result["year"] = result.index.year
        
        # Holiday proximity (simplified - would need holiday calendar for accuracy)
        # This is a basic implementation
        result["days_since_year_start"] = result.index.dayofyear
        result["days_until_year_end"] = 365 - result["days_since_year_start"]
        
        return result
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_labels: bool = False
    ) -> pd.DataFrame:
        """Generate all daily features.
        
        Args:
            df: DataFrame with daily OHLCV data
            include_labels: Whether to include forward-looking labels
            
        Returns:
            DataFrame with all daily features
        """
        self.logger.info(f"Generating daily features for {len(df)} days")
        
        # Start with original data
        result = df.copy()
        
        # Generate all feature types
        result = self.calculate_multi_day_returns(result)
        result = self.calculate_realized_volatility(result)
        result = self.calculate_gap_features(result)
        result = self.calculate_turnover_features(result)
        result = self.calculate_momentum_features(result)
        result = self.calculate_seasonal_features(result)
        
        # Log feature generation results
        feature_cols = [col for col in result.columns if col not in df.columns]
        self.logger.info(f"Generated {len(feature_cols)} daily features")
        
        return result
    
    def get_feature_names(self, include_labels: bool = False) -> List[str]:
        """Get list of all feature names that would be generated.
        
        Args:
            include_labels: Whether to include label feature names
            
        Returns:
            List of feature names
        """
        features = []
        
        # Multi-day returns
        for period in self.config.return_periods:
            features.extend([
                f"return_{period}d",
                f"log_return_{period}d",
                f"cum_return_{period}d"
            ])
            if include_labels:
                features.append(f"forward_return_{period}d")
        
        # Volatility features
        for window in self.config.volatility_windows:
            features.extend([
                f"volatility_{window}d",
                f"realized_vol_{window}d",
                f"log_vol_{window}d",
                f"parkinson_vol_{window}d",
                f"gk_vol_{window}d"
            ])
            if window >= 10:
                features.append(f"vol_of_vol_{window}d")
        
        # Gap features
        features.extend([
            "gap", "gap_abs", "gap_up", "gap_down", "gap_size_category", "gap_filled_same_day"
        ])
        for period in self.config.gap_analysis_periods:
            features.extend([
                f"avg_gap_{period}d",
                f"gap_volatility_{period}d",
                f"gap_up_freq_{period}d",
                f"gap_fill_freq_{period}d"
            ])
        
        # Turnover features
        features.extend(["dollar_volume", "relative_volume", "volume_return_corr_20d"])
        for window in [5, 10, 20, 60]:
            features.extend([
                f"avg_volume_{window}d",
                f"volume_ratio_{window}d",
                f"volume_volatility_{window}d",
                f"volume_zscore_{window}d",
                f"avg_dollar_volume_{window}d",
                f"dollar_volume_ratio_{window}d"
            ])
        
        # Momentum features
        for window in self.config.momentum_windows:
            features.extend([
                f"ma_{window}d",
                f"price_ma_ratio_{window}d",
                f"price_ma_deviation_{window}d",
                f"momentum_{window}d",
                f"rsi_{window}d"
            ])
            if window >= 20:
                features.extend([
                    f"bb_upper_{window}d",
                    f"bb_lower_{window}d",
                    f"bb_position_{window}d",
                    f"bb_width_{window}d"
                ])
        
        # MACD
        features.extend(["macd", "macd_signal", "macd_histogram"])
        
        # Trend features
        for window in [20, 50]:
            features.extend([
                f"trend_slope_{window}d",
                f"trend_strength_{window}d"
            ])
        
        # Seasonal features
        features.extend([
            "day_of_week", "is_monday", "is_friday", "month", "quarter",
            "day_of_month", "is_month_end", "is_month_start", "year",
            "days_since_year_start", "days_until_year_end"
        ])
        
        return features
