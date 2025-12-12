"""Cross-asset feature engineering for multi-asset alpha generation.

This module provides feature extraction capabilities that analyze relationships
between different asset classes, including regime signals, volatility ratios,
and cross-asset correlations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from ..data.universe import AssetUniverse, AssetClass
from ..utils.logging_utils import LoggerMixin


@dataclass
class CrossAssetFeatureConfig:
    """Configuration for cross-asset feature generation."""
    correlation_windows: List[int] = None
    volatility_windows: List[int] = None
    regime_symbols: List[str] = None
    benchmark_symbol: str = "SPY"
    
    def __post_init__(self):
        if self.correlation_windows is None:
            self.correlation_windows = [5, 10, 20, 60]
        
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 60]
        
        if self.regime_symbols is None:
            self.regime_symbols = ["SPY", "QQQ", "IWM", "VIX", "TLT", "GLD"]


class CrossAssetFeatureEngine(LoggerMixin):
    """Engine for generating cross-asset features from multi-asset market data."""
    
    def __init__(
        self,
        config: Optional[CrossAssetFeatureConfig] = None,
        universe: Optional[AssetUniverse] = None
    ):
        """Initialize cross-asset feature engine.
        
        Args:
            config: Feature configuration (uses defaults if None)
            universe: Asset universe for symbol information
        """
        self.config = config or CrossAssetFeatureConfig()
        self.universe = universe or AssetUniverse()
        self.logger.info(f"Initialized CrossAssetFeatureEngine with config: {self.config}")
    
    def calculate_market_regime_signals(
        self,
        data: pd.DataFrame,
        target_symbol: str,
        regime_symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate market regime signals using key market indicators.
        
        Args:
            data: DataFrame with multi-asset price data
            target_symbol: Symbol to generate features for
            regime_symbols: List of symbols to use for regime detection
            
        Returns:
            DataFrame with regime signal features
        """
        if regime_symbols is None:
            regime_symbols = self.config.regime_symbols
        
        result = data.copy()
        
        # Filter to available symbols
        available_symbols = [s for s in regime_symbols if s in data["symbol"].unique()]
        if not available_symbols:
            self.logger.warning("No regime symbols found in data")
            return result
        
        # Pivot data for easier cross-asset calculations
        price_data = data.pivot(index="timestamp", columns="symbol", values="close")
        
        # Market regime features
        regime_features = pd.DataFrame(index=price_data.index)
        
        # SPY-based features (broad market)
        if "SPY" in price_data.columns:
            spy_returns = price_data["SPY"].pct_change()
            
            # SPY momentum
            for window in [5, 10, 20]:
                regime_features[f"spy_momentum_{window}d"] = (
                    price_data["SPY"] / price_data["SPY"].shift(window) - 1
                )
                regime_features[f"spy_volatility_{window}d"] = (
                    spy_returns.rolling(window).std() * np.sqrt(252)
                )
            
            # SPY trend strength
            regime_features["spy_above_ma20"] = (
                price_data["SPY"] > price_data["SPY"].rolling(20).mean()
            ).astype(int)
            regime_features["spy_above_ma50"] = (
                price_data["SPY"] > price_data["SPY"].rolling(50).mean()
            ).astype(int)
        
        # VIX-based features (volatility regime)
        if "VIX" in price_data.columns:
            # VIX level categories
            vix_percentiles = price_data["VIX"].quantile([0.2, 0.4, 0.6, 0.8])
            regime_features["vix_regime"] = pd.cut(
                price_data["VIX"],
                bins=[0] + vix_percentiles.tolist() + [np.inf],
                labels=["very_low", "low", "medium", "high", "very_high"]
            )
            
            # VIX momentum
            vix_returns = price_data["VIX"].pct_change()
            regime_features["vix_momentum_5d"] = (
                price_data["VIX"] / price_data["VIX"].shift(5) - 1
            )
            regime_features["vix_spike"] = (vix_returns > vix_returns.quantile(0.95)).astype(int)
        
        # Bond-equity relationship (TLT vs SPY)
        if "TLT" in price_data.columns and "SPY" in price_data.columns:
            tlt_returns = price_data["TLT"].pct_change()
            spy_returns = price_data["SPY"].pct_change()
            
            # Rolling correlation
            for window in [10, 20, 60]:
                regime_features[f"bond_equity_corr_{window}d"] = (
                    tlt_returns.rolling(window).corr(spy_returns)
                )
            
            # Relative performance
            regime_features["tlt_spy_ratio"] = price_data["TLT"] / price_data["SPY"]
            regime_features["tlt_spy_momentum_20d"] = (
                regime_features["tlt_spy_ratio"] / regime_features["tlt_spy_ratio"].shift(20) - 1
            )
        
        # Growth vs Value (QQQ vs IWM)
        if "QQQ" in price_data.columns and "IWM" in price_data.columns:
            regime_features["growth_value_ratio"] = price_data["QQQ"] / price_data["IWM"]
            regime_features["growth_value_momentum_20d"] = (
                regime_features["growth_value_ratio"] / 
                regime_features["growth_value_ratio"].shift(20) - 1
            )
        
        # Risk-on/Risk-off sentiment
        if all(s in price_data.columns for s in ["SPY", "TLT", "GLD"]):
            # Simple risk sentiment score
            spy_mom = price_data["SPY"] / price_data["SPY"].shift(10) - 1
            tlt_mom = price_data["TLT"] / price_data["TLT"].shift(10) - 1
            gld_mom = price_data["GLD"] / price_data["GLD"].shift(10) - 1
            
            # Risk-on when stocks up, bonds/gold down
            regime_features["risk_sentiment"] = spy_mom - 0.5 * (tlt_mom + gld_mom)
            regime_features["risk_on"] = (regime_features["risk_sentiment"] > 0).astype(int)
        
        # Merge back with target symbol data
        target_data = data[data["symbol"] == target_symbol].copy()
        if not target_data.empty:
            target_data = target_data.set_index("timestamp")
            target_data = target_data.join(regime_features, how="left")
            target_data = target_data.reset_index()
        
        return target_data
    
    def calculate_cross_asset_volatility(
        self,
        data: pd.DataFrame,
        target_symbol: str,
        reference_symbols: Optional[List[str]] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate cross-asset volatility ratios and relationships.
        
        Args:
            data: DataFrame with multi-asset price data
            target_symbol: Symbol to generate features for
            reference_symbols: Symbols to compare volatility against
            windows: Windows for volatility calculations
            
        Returns:
            DataFrame with cross-asset volatility features
        """
        if reference_symbols is None:
            reference_symbols = ["SPY", "QQQ", "TLT", "GLD"]
        
        if windows is None:
            windows = self.config.volatility_windows
        
        # Pivot data for easier calculations
        price_data = data.pivot(index="timestamp", columns="symbol", values="close")
        
        # Calculate returns
        returns_data = price_data.pct_change()
        
        # Volatility features
        vol_features = pd.DataFrame(index=price_data.index)
        
        if target_symbol not in returns_data.columns:
            self.logger.warning(f"Target symbol {target_symbol} not found in data")
            return data[data["symbol"] == target_symbol].copy()
        
        target_returns = returns_data[target_symbol]
        
        for window in windows:
            target_vol = target_returns.rolling(window).std() * np.sqrt(252)
            
            for ref_symbol in reference_symbols:
                if ref_symbol in returns_data.columns and ref_symbol != target_symbol:
                    ref_vol = returns_data[ref_symbol].rolling(window).std() * np.sqrt(252)
                    
                    # Volatility ratio
                    vol_features[f"vol_ratio_{ref_symbol}_{window}d"] = target_vol / ref_vol
                    
                    # Volatility spread
                    vol_features[f"vol_spread_{ref_symbol}_{window}d"] = target_vol - ref_vol
                    
                    # Relative volatility rank
                    combined_vol = pd.concat([target_vol, ref_vol], axis=1)
                    vol_features[f"vol_rank_{ref_symbol}_{window}d"] = (
                        target_vol.rank(pct=True, axis=0)
                    )
        
        # Cross-asset volatility regime
        if "SPY" in returns_data.columns:
            spy_vol = returns_data["SPY"].rolling(20).std() * np.sqrt(252)
            target_vol_20d = target_returns.rolling(20).std() * np.sqrt(252)
            
            # Volatility regime based on SPY
            spy_vol_percentiles = spy_vol.quantile([0.25, 0.5, 0.75])
            vol_features["market_vol_regime"] = pd.cut(
                spy_vol,
                bins=[0] + spy_vol_percentiles.tolist() + [np.inf],
                labels=["low_vol", "medium_vol", "high_vol", "very_high_vol"]
            )
            
            # Target volatility relative to market regime
            vol_features["vol_vs_market_regime"] = target_vol_20d / spy_vol
        
        # Merge back with target symbol data
        target_data = data[data["symbol"] == target_symbol].copy()
        if not target_data.empty:
            target_data = target_data.set_index("timestamp")
            target_data = target_data.join(vol_features, how="left")
            target_data = target_data.reset_index()
        
        return target_data
    
    def calculate_cross_asset_correlations(
        self,
        data: pd.DataFrame,
        target_symbol: str,
        reference_symbols: Optional[List[str]] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate rolling correlations with other assets.
        
        Args:
            data: DataFrame with multi-asset price data
            target_symbol: Symbol to generate features for
            reference_symbols: Symbols to calculate correlations with
            windows: Windows for correlation calculations
            
        Returns:
            DataFrame with correlation features
        """
        if reference_symbols is None:
            reference_symbols = ["SPY", "QQQ", "TLT", "GLD", "VIX"]
        
        if windows is None:
            windows = self.config.correlation_windows
        
        # Pivot data for easier calculations
        price_data = data.pivot(index="timestamp", columns="symbol", values="close")
        returns_data = price_data.pct_change()
        
        # Correlation features
        corr_features = pd.DataFrame(index=price_data.index)
        
        if target_symbol not in returns_data.columns:
            self.logger.warning(f"Target symbol {target_symbol} not found in data")
            return data[data["symbol"] == target_symbol].copy()
        
        target_returns = returns_data[target_symbol]
        
        for window in windows:
            for ref_symbol in reference_symbols:
                if ref_symbol in returns_data.columns and ref_symbol != target_symbol:
                    # Rolling correlation
                    corr = target_returns.rolling(window).corr(returns_data[ref_symbol])
                    corr_features[f"corr_{ref_symbol}_{window}d"] = corr
                    
                    # Correlation change
                    corr_features[f"corr_change_{ref_symbol}_{window}d"] = corr.diff(5)
                    
                    # Correlation regime (high/medium/low)
                    corr_abs = np.abs(corr)
                    corr_percentiles = corr_abs.quantile([0.33, 0.67])
                    corr_features[f"corr_regime_{ref_symbol}_{window}d"] = pd.cut(
                        corr_abs,
                        bins=[0] + corr_percentiles.tolist() + [1],
                        labels=["low_corr", "medium_corr", "high_corr"]
                    )
        
        # Beta calculations (vs SPY)
        if "SPY" in returns_data.columns:
            spy_returns = returns_data["SPY"]
            
            for window in [20, 60, 120]:
                # Rolling beta
                covariance = target_returns.rolling(window).cov(spy_returns)
                spy_variance = spy_returns.rolling(window).var()
                beta = covariance / spy_variance
                corr_features[f"beta_spy_{window}d"] = beta
                
                # Beta stability (rolling std of beta)
                corr_features[f"beta_stability_{window}d"] = beta.rolling(20).std()
        
        # Merge back with target symbol data
        target_data = data[data["symbol"] == target_symbol].copy()
        if not target_data.empty:
            target_data = target_data.set_index("timestamp")
            target_data = target_data.join(corr_features, how="left")
            target_data = target_data.reset_index()
        
        return target_data
    
    def calculate_sector_relative_features(
        self,
        data: pd.DataFrame,
        target_symbol: str
    ) -> pd.DataFrame:
        """Calculate features relative to sector performance.
        
        Args:
            data: DataFrame with multi-asset price data
            target_symbol: Symbol to generate features for
            
        Returns:
            DataFrame with sector relative features
        """
        try:
            asset_info = self.universe.get_asset(target_symbol)
            sector = asset_info.sector
        except KeyError:
            self.logger.warning(f"Asset {target_symbol} not found in universe")
            return data[data["symbol"] == target_symbol].copy()
        
        # Get sector symbols
        sector_symbols = self.universe.get_symbols(sector=sector)
        sector_symbols = [s for s in sector_symbols if s != target_symbol]
        
        if not sector_symbols:
            self.logger.warning(f"No other symbols found for sector {sector}")
            return data[data["symbol"] == target_symbol].copy()
        
        # Filter to available symbols in data
        available_sector_symbols = [
            s for s in sector_symbols if s in data["symbol"].unique()
        ]
        
        if not available_sector_symbols:
            return data[data["symbol"] == target_symbol].copy()
        
        # Pivot data
        price_data = data.pivot(index="timestamp", columns="symbol", values="close")
        returns_data = price_data.pct_change()
        
        # Calculate sector average (equal-weighted)
        sector_prices = price_data[available_sector_symbols].mean(axis=1)
        sector_returns = sector_prices.pct_change()
        
        # Sector relative features
        sector_features = pd.DataFrame(index=price_data.index)
        
        if target_symbol in price_data.columns:
            target_price = price_data[target_symbol]
            target_returns = returns_data[target_symbol]
            
            # Relative price performance
            sector_features["sector_relative_price"] = target_price / sector_prices
            
            # Relative momentum
            for window in [5, 10, 20]:
                target_mom = target_price / target_price.shift(window) - 1
                sector_mom = sector_prices / sector_prices.shift(window) - 1
                sector_features[f"sector_relative_momentum_{window}d"] = target_mom - sector_mom
            
            # Relative volatility
            for window in [10, 20]:
                target_vol = target_returns.rolling(window).std()
                sector_vol = sector_returns.rolling(window).std()
                sector_features[f"sector_relative_volatility_{window}d"] = target_vol / sector_vol
            
            # Correlation with sector
            for window in [10, 20, 60]:
                sector_features[f"sector_correlation_{window}d"] = (
                    target_returns.rolling(window).corr(sector_returns)
                )
        
        # Merge back with target symbol data
        target_data = data[data["symbol"] == target_symbol].copy()
        if not target_data.empty:
            target_data = target_data.set_index("timestamp")
            target_data = target_data.join(sector_features, how="left")
            target_data = target_data.reset_index()
        
        return target_data
    
    def generate_all_features(
        self,
        data: pd.DataFrame,
        target_symbol: str
    ) -> pd.DataFrame:
        """Generate all cross-asset features for a target symbol.
        
        Args:
            data: DataFrame with multi-asset price data
            target_symbol: Symbol to generate features for
            
        Returns:
            DataFrame with all cross-asset features for target symbol
        """
        self.logger.info(f"Generating cross-asset features for {target_symbol}")
        
        # Start with target symbol data
        result = data[data["symbol"] == target_symbol].copy()
        
        # Generate all feature types
        result = self.calculate_market_regime_signals(data, target_symbol)
        result = self.calculate_cross_asset_volatility(data, target_symbol)
        result = self.calculate_cross_asset_correlations(data, target_symbol)
        result = self.calculate_sector_relative_features(data, target_symbol)
        
        # Log feature generation results
        original_cols = data[data["symbol"] == target_symbol].columns
        feature_cols = [col for col in result.columns if col not in original_cols]
        self.logger.info(f"Generated {len(feature_cols)} cross-asset features for {target_symbol}")
        
        return result
    
    def get_feature_names(self, target_symbol: str = "SYMBOL") -> List[str]:
        """Get list of all feature names that would be generated.
        
        Args:
            target_symbol: Symbol name to use in feature names (for template)
            
        Returns:
            List of feature names
        """
        features = []
        
        # Market regime features
        features.extend([
            "spy_momentum_5d", "spy_momentum_10d", "spy_momentum_20d",
            "spy_volatility_5d", "spy_volatility_10d", "spy_volatility_20d",
            "spy_above_ma20", "spy_above_ma50",
            "vix_regime", "vix_momentum_5d", "vix_spike",
            "bond_equity_corr_10d", "bond_equity_corr_20d", "bond_equity_corr_60d",
            "tlt_spy_ratio", "tlt_spy_momentum_20d",
            "growth_value_ratio", "growth_value_momentum_20d",
            "risk_sentiment", "risk_on"
        ])
        
        # Cross-asset volatility features
        reference_symbols = ["SPY", "QQQ", "TLT", "GLD"]
        for window in self.config.volatility_windows:
            for ref_symbol in reference_symbols:
                features.extend([
                    f"vol_ratio_{ref_symbol}_{window}d",
                    f"vol_spread_{ref_symbol}_{window}d",
                    f"vol_rank_{ref_symbol}_{window}d"
                ])
        
        features.extend(["market_vol_regime", "vol_vs_market_regime"])
        
        # Correlation features
        reference_symbols = ["SPY", "QQQ", "TLT", "GLD", "VIX"]
        for window in self.config.correlation_windows:
            for ref_symbol in reference_symbols:
                features.extend([
                    f"corr_{ref_symbol}_{window}d",
                    f"corr_change_{ref_symbol}_{window}d",
                    f"corr_regime_{ref_symbol}_{window}d"
                ])
        
        # Beta features
        for window in [20, 60, 120]:
            features.extend([
                f"beta_spy_{window}d",
                f"beta_stability_{window}d"
            ])
        
        # Sector relative features
        features.extend([
            "sector_relative_price",
            "sector_correlation_10d", "sector_correlation_20d", "sector_correlation_60d"
        ])
        for window in [5, 10, 20]:
            features.append(f"sector_relative_momentum_{window}d")
        for window in [10, 20]:
            features.append(f"sector_relative_volatility_{window}d")
        
        return features
