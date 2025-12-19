"""Regime detection for market regime identification.

This module provides both quantile-based (current implementation) and HMM-based 
(optional/future) regime detection methods. The current experiment uses 
volatility/VIX quantile regimes, not HMM.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from ..config import DEFAULT_N_REGIMES, DEFAULT_REGIME_FEATURES
from ..utils.logging_utils import LoggerMixin


@dataclass
class RegimeFeatureConfig:
    """Configuration for regime feature generation."""
    volatility_windows: List[int] = None
    volume_windows: List[int] = None
    cross_asset_symbols: List[str] = None
    feature_names: List[str] = None
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60]
        
        if self.volume_windows is None:
            self.volume_windows = [5, 10, 20]
        
        if self.cross_asset_symbols is None:
            self.cross_asset_symbols = ["SPY", "QQQ", "TLT", "VIX", "GLD"]
        
        if self.feature_names is None:
            self.feature_names = DEFAULT_REGIME_FEATURES.copy()


class RegimeFeatureEngine(LoggerMixin):
    """Engine for generating features specifically for regime detection."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        """Initialize regime feature engine.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or RegimeFeatureConfig()
        self.logger.info(f"Initialized RegimeFeatureEngine with config: {self.config}")
    
    def calculate_volatility_features(
        self,
        data: pd.DataFrame,
        price_col: str = "close"
    ) -> pd.DataFrame:
        """Calculate volatility-based regime features.
        
        Args:
            data: DataFrame with price data
            price_col: Column name for price data
            
        Returns:
            DataFrame with volatility regime features
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate returns
        returns = data[price_col].pct_change()
        
        for window in self.config.volatility_windows:
            # Rolling volatility (annualized)
            vol = returns.rolling(window).std() * np.sqrt(252)
            result[f"volatility_{window}d"] = vol
            
            # Volatility of volatility
            if window >= 10:
                vol_of_vol = vol.rolling(10).std()
                result[f"vol_of_vol_{window}d"] = vol_of_vol
            
            # Realized volatility (sum of squared returns)
            realized_vol = np.sqrt((returns ** 2).rolling(window).sum() * 252 / window)
            result[f"realized_vol_{window}d"] = realized_vol
            
            # Volatility regime indicator (high/medium/low)
            vol_percentiles = vol.quantile([0.33, 0.67])
            result[f"vol_regime_{window}d"] = pd.cut(
                vol,
                bins=[0] + vol_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2]  # Low, Medium, High
            ).astype(float)
        
        # Volatility clustering (GARCH-like effect)
        abs_returns = np.abs(returns)
        for window in [5, 10, 20]:
            # Autocorrelation of squared returns (volatility clustering)
            squared_returns = returns ** 2
            result[f"vol_clustering_{window}d"] = (
                squared_returns.rolling(window).corr(squared_returns.shift(1))
            )
            
            # Volatility persistence
            result[f"vol_persistence_{window}d"] = (
                abs_returns.rolling(window).corr(abs_returns.shift(1))
            )
        
        return result
    
    def calculate_volume_features(
        self,
        data: pd.DataFrame,
        volume_col: str = "volume"
    ) -> pd.DataFrame:
        """Calculate volume-based regime features.
        
        Args:
            data: DataFrame with volume data
            volume_col: Column name for volume data
            
        Returns:
            DataFrame with volume regime features
        """
        result = pd.DataFrame(index=data.index)
        
        if volume_col not in data.columns:
            self.logger.warning(f"Volume column {volume_col} not found")
            return result
        
        volume = data[volume_col]
        
        for window in self.config.volume_windows:
            # Volume moving average and ratio
            vol_ma = volume.rolling(window).mean()
            result[f"volume_ratio_{window}d"] = volume / vol_ma
            
            # Volume volatility
            vol_volatility = volume.rolling(window).std()
            result[f"volume_volatility_{window}d"] = vol_volatility
            
            # Volume z-score
            result[f"volume_zscore_{window}d"] = (volume - vol_ma) / vol_volatility
            
            # Volume regime (high/medium/low)
            vol_ratio = result[f"volume_ratio_{window}d"]
            vol_percentiles = vol_ratio.quantile([0.33, 0.67])
            result[f"volume_regime_{window}d"] = pd.cut(
                vol_ratio,
                bins=[0] + vol_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2]  # Low, Medium, High
            ).astype(float)
        
        # Volume trend
        for window in [5, 10, 20]:
            # Volume momentum
            result[f"volume_momentum_{window}d"] = volume / volume.shift(window) - 1
            
            # Volume trend strength (linear regression slope)
            x = np.arange(window)
            slopes = []
            
            for i in range(len(volume)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = volume.iloc[i-window+1:i+1].values
                    if len(y) == window and not np.isnan(y).all():
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope / volume.iloc[i])  # Normalize
                    else:
                        slopes.append(np.nan)
            
            result[f"volume_trend_{window}d"] = slopes
        
        return result
    
    def calculate_cross_asset_regime_features(
        self,
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate cross-asset features for regime detection.
        
        Args:
            data: DataFrame with multi-asset price data
            symbols: List of symbols to use for cross-asset features
            
        Returns:
            DataFrame with cross-asset regime features
        """
        if symbols is None:
            symbols = self.config.cross_asset_symbols
        
        # Pivot data to have symbols as columns
        if "symbol" in data.columns:
            price_data = data.pivot(index="timestamp", columns="symbol", values="close")
        else:
            price_data = data
        
        # Filter to available symbols
        available_symbols = [s for s in symbols if s in price_data.columns]
        if len(available_symbols) < 2:
            self.logger.warning("Not enough symbols available for cross-asset features")
            return pd.DataFrame(index=price_data.index)
        
        result = pd.DataFrame(index=price_data.index)
        
        # Calculate returns for all symbols
        returns_data = price_data[available_symbols].pct_change()
        
        # Cross-asset volatility ratios
        if "SPY" in available_symbols:
            spy_vol = returns_data["SPY"].rolling(20).std() * np.sqrt(252)
            
            for symbol in available_symbols:
                if symbol != "SPY":
                    symbol_vol = returns_data[symbol].rolling(20).std() * np.sqrt(252)
                    result[f"vol_ratio_{symbol}_spy"] = symbol_vol / spy_vol
        
        # VIX-based regime features
        if "VIX" in available_symbols:
            vix = price_data["VIX"]
            
            # VIX level regime
            vix_percentiles = vix.quantile([0.2, 0.4, 0.6, 0.8])
            result["vix_regime"] = pd.cut(
                vix,
                bins=[0] + vix_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2, 3, 4]  # Very Low to Very High
            ).astype(float)
            
            # VIX momentum
            result["vix_momentum_5d"] = vix / vix.shift(5) - 1
            result["vix_momentum_20d"] = vix / vix.shift(20) - 1
        
        # Bond-equity relationship
        if "SPY" in available_symbols and "TLT" in available_symbols:
            spy_returns = returns_data["SPY"]
            tlt_returns = returns_data["TLT"]
            
            # Rolling correlation
            for window in [10, 20, 60]:
                corr = spy_returns.rolling(window).corr(tlt_returns)
                result[f"bond_equity_corr_{window}d"] = corr
                
                # Correlation regime
                corr_abs = np.abs(corr)
                corr_percentiles = corr_abs.quantile([0.33, 0.67])
                result[f"bond_equity_corr_regime_{window}d"] = pd.cut(
                    corr_abs,
                    bins=[0] + corr_percentiles.tolist() + [1],
                    labels=[0, 1, 2]  # Low, Medium, High correlation
                ).astype(float)
        
        # Risk-on/Risk-off indicators
        if all(s in available_symbols for s in ["SPY", "TLT", "GLD"]):
            # Simple risk sentiment
            spy_mom = returns_data["SPY"].rolling(10).mean()
            tlt_mom = returns_data["TLT"].rolling(10).mean()
            gld_mom = returns_data["GLD"].rolling(10).mean()
            
            # Risk-on when equities outperform bonds and gold
            risk_sentiment = spy_mom - 0.5 * (tlt_mom + gld_mom)
            result["risk_sentiment"] = risk_sentiment
            
            # Risk regime
            risk_percentiles = risk_sentiment.quantile([0.33, 0.67])
            result["risk_regime"] = pd.cut(
                risk_sentiment,
                bins=[-np.inf] + risk_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2]  # Risk-off, Neutral, Risk-on
            ).astype(float)
        
        # Growth vs Value (QQQ vs IWM)
        if "QQQ" in available_symbols and "IWM" in available_symbols:
            growth_value_ratio = price_data["QQQ"] / price_data["IWM"]
            result["growth_value_ratio"] = growth_value_ratio
            
            # Growth/Value momentum
            result["growth_value_momentum_20d"] = (
                growth_value_ratio / growth_value_ratio.shift(20) - 1
            )
            
            # Growth/Value regime
            gv_percentiles = result["growth_value_momentum_20d"].quantile([0.33, 0.67])
            result["growth_value_regime"] = pd.cut(
                result["growth_value_momentum_20d"],
                bins=[-np.inf] + gv_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2]  # Value outperform, Neutral, Growth outperform
            ).astype(float)
        
        return result
    
    def calculate_market_stress_features(
        self,
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate market stress and crisis indicators.
        
        Args:
            data: DataFrame with multi-asset price data
            symbols: List of symbols to use for stress indicators
            
        Returns:
            DataFrame with market stress features
        """
        if symbols is None:
            symbols = self.config.cross_asset_symbols
        
        # Pivot data to have symbols as columns
        if "symbol" in data.columns:
            price_data = data.pivot(index="timestamp", columns="symbol", values="close")
        else:
            price_data = data
        
        # Filter to available symbols
        available_symbols = [s for s in symbols if s in price_data.columns]
        
        result = pd.DataFrame(index=price_data.index)
        
        if len(available_symbols) < 2:
            return result
        
        # Calculate returns
        returns_data = price_data[available_symbols].pct_change()
        
        # Market stress indicators
        if "SPY" in available_symbols:
            spy_returns = returns_data["SPY"]
            
            # Drawdown
            spy_cumulative = (1 + spy_returns).cumprod()
            spy_running_max = spy_cumulative.expanding().max()
            drawdown = (spy_cumulative - spy_running_max) / spy_running_max
            result["market_drawdown"] = drawdown
            
            # Stress level based on drawdown
            dd_percentiles = drawdown.quantile([0.1, 0.25])  # More negative values
            result["stress_regime"] = pd.cut(
                drawdown,
                bins=[-np.inf] + dd_percentiles.tolist() + [0],
                labels=[2, 1, 0]  # High stress, Medium stress, Low stress
            ).astype(float)
            
            # Tail risk (extreme negative returns)
            spy_vol = spy_returns.rolling(20).std()
            result["tail_risk"] = (spy_returns < -2 * spy_vol).astype(int)
            
            # Market crash indicator (large negative returns)
            result["market_crash"] = (spy_returns < -0.05).astype(int)  # 5% daily drop
        
        # Cross-asset correlation breakdown (flight to quality)
        if len(available_symbols) >= 3:
            # Average pairwise correlation
            corr_matrix = returns_data.rolling(20).corr()
            
            # Extract upper triangle correlations for each date
            avg_correlations = []
            for date in returns_data.index:
                if pd.isna(date):
                    avg_correlations.append(np.nan)
                    continue
                
                try:
                    date_corr = corr_matrix.loc[date]
                    if isinstance(date_corr, pd.DataFrame):
                        # Get upper triangle (excluding diagonal)
                        mask = np.triu(np.ones_like(date_corr, dtype=bool), k=1)
                        correlations = date_corr.values[mask]
                        avg_corr = np.nanmean(correlations)
                        avg_correlations.append(avg_corr)
                    else:
                        avg_correlations.append(np.nan)
                except (KeyError, IndexError):
                    avg_correlations.append(np.nan)
            
            result["avg_cross_correlation"] = avg_correlations
            
            # Correlation breakdown (low correlation = potential crisis)
            result["correlation_breakdown"] = (
                pd.Series(avg_correlations, index=returns_data.index) < 0.3
            ).astype(int)
        
        # Flight to quality indicators
        if "TLT" in available_symbols and "SPY" in available_symbols:
            # Bond outperformance during stress
            bond_equity_spread = returns_data["TLT"] - returns_data["SPY"]
            result["flight_to_quality"] = bond_equity_spread.rolling(5).mean()
            
            # Flight to quality regime
            ftq_percentiles = result["flight_to_quality"].quantile([0.33, 0.67])
            result["flight_to_quality_regime"] = pd.cut(
                result["flight_to_quality"],
                bins=[-np.inf] + ftq_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2]  # No flight, Moderate, Strong flight to quality
            ).astype(float)
        
        return result
    
    def build_regime_feature_matrix(
        self,
        data: pd.DataFrame,
        target_symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Build complete feature matrix for regime detection.
        
        Args:
            data: DataFrame with market data
            target_symbol: Optional target symbol to focus on
            
        Returns:
            DataFrame with all regime detection features
        """
        self.logger.info("Building regime feature matrix")
        
        if target_symbol and "symbol" in data.columns:
            # Focus on specific symbol but use all data for cross-asset features
            target_data = data[data["symbol"] == target_symbol].copy()
            if target_data.empty:
                self.logger.error(f"No data found for target symbol {target_symbol}")
                return pd.DataFrame()
            
            # Calculate single-asset features
            vol_features = self.calculate_volatility_features(target_data)
            volume_features = self.calculate_volume_features(target_data)
            
            # Calculate cross-asset features using all data
            cross_asset_features = self.calculate_cross_asset_regime_features(data)
            stress_features = self.calculate_market_stress_features(data)
            
            # Align features with target symbol timestamps
            target_data = target_data.set_index("timestamp")
            result = target_data.join([vol_features, volume_features], how="left")
            result = result.join([cross_asset_features, stress_features], how="left")
            result = result.reset_index()
            
        else:
            # Use all data (assuming single symbol or already pivoted)
            vol_features = self.calculate_volatility_features(data)
            volume_features = self.calculate_volume_features(data)
            cross_asset_features = self.calculate_cross_asset_regime_features(data)
            stress_features = self.calculate_market_stress_features(data)
            
            # Combine all features
            result = data.copy()
            for features in [vol_features, volume_features, cross_asset_features, stress_features]:
                result = result.join(features, how="left")
        
        # Log results
        original_cols = data.columns
        feature_cols = [col for col in result.columns if col not in original_cols]
        self.logger.info(f"Built regime feature matrix with {len(feature_cols)} features")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of all regime feature names.
        
        Returns:
            List of regime feature names
        """
        features = []
        
        # Volatility features
        for window in self.config.volatility_windows:
            features.extend([
                f"volatility_{window}d",
                f"realized_vol_{window}d",
                f"vol_regime_{window}d"
            ])
            if window >= 10:
                features.append(f"vol_of_vol_{window}d")
        
        # Volume features
        for window in self.config.volume_windows:
            features.extend([
                f"volume_ratio_{window}d",
                f"volume_volatility_{window}d",
                f"volume_zscore_{window}d",
                f"volume_regime_{window}d",
                f"volume_momentum_{window}d",
                f"volume_trend_{window}d"
            ])
        
        # Volatility clustering
        for window in [5, 10, 20]:
            features.extend([
                f"vol_clustering_{window}d",
                f"vol_persistence_{window}d"
            ])
        
        # Cross-asset features
        cross_asset_symbols = ["QQQ", "TLT", "VIX", "GLD"]
        for symbol in cross_asset_symbols:
            features.append(f"vol_ratio_{symbol}_spy")
        
        features.extend([
            "vix_regime", "vix_momentum_5d", "vix_momentum_20d",
            "risk_sentiment", "risk_regime",
            "growth_value_ratio", "growth_value_momentum_20d", "growth_value_regime"
        ])
        
        # Bond-equity correlation
        for window in [10, 20, 60]:
            features.extend([
                f"bond_equity_corr_{window}d",
                f"bond_equity_corr_regime_{window}d"
            ])
        
        # Market stress features
        features.extend([
            "market_drawdown", "stress_regime", "tail_risk", "market_crash",
            "avg_cross_correlation", "correlation_breakdown",
            "flight_to_quality", "flight_to_quality_regime"
        ])
        
        return features


def assign_regimes(
    df: pd.DataFrame,
    method: str = "vol_vix_quantiles",
    regime_features: Optional[List[str]] = None,
    n_regimes: int = 3,
    hmm_model: Optional['RegimeHMM'] = None
) -> pd.Series:
    """
    Assign market regime labels per timestamp.
    
    The current experiment uses volatility/VIX quantile regimes (method="vol_vix_quantiles").
    HMM-based regimes are available as an optional extension but are NOT used in the 
    reported results.
    
    Args:
        df: DataFrame with market data including volatility_20d and optionally vix_level
        method: Regime detection method
            - "vol_vix_quantiles": Current implementation using volatility/VIX 3x3 grid
            - "hmm": Optional HMM-based states (requires hmm_model parameter)
        regime_features: List of feature columns for HMM (ignored for quantile method)
        n_regimes: Number of regimes for quantile method (default: 3)
        hmm_model: Fitted HMM model (required if method="hmm")
        
    Returns:
        Series with regime labels for each timestamp
        
    Regime Descriptions (for vol_vix_quantiles method):
        - Low_Vol_Low_VIX: Low volatility, low VIX (calm markets)
        - Low_Vol_Med_VIX: Low volatility, medium VIX (mixed signals)
        - Low_Vol_High_VIX: Low volatility, high VIX (potential turning point)
        - Med_Vol_Low_VIX: Medium volatility, low VIX (normal markets)
        - Med_Vol_Med_VIX: Medium volatility, medium VIX (typical conditions)
        - Med_Vol_High_VIX: Medium volatility, high VIX (elevated uncertainty)
        - High_Vol_Low_VIX: High volatility, low VIX (unusual combination)
        - High_Vol_Med_VIX: High volatility, medium VIX (stressed markets)
        - High_Vol_High_VIX: High volatility, high VIX (crisis conditions)
    """
    
    if method == "vol_vix_quantiles":
        # Current implementation: volatility/VIX quantile-based regimes
        # Create a copy to avoid modifying original
        df_work = df.copy()
        df_work['_original_index'] = df_work.index
        
        # Process each symbol group and add regime directly
        for symbol, group in df_work.groupby('symbol'):
            group_indices = group.index
            
            # Volatility regime (3 quantiles)
            vol_col = 'volatility_20d'
            if vol_col not in group.columns:
                raise ValueError(f"Required column '{vol_col}' not found for quantile regime detection")
                
            vol_data = group[vol_col].fillna(group[vol_col].median())
            vol_regime = pd.qcut(
                vol_data, 
                q=n_regimes, 
                labels=['Low_Vol', 'Med_Vol', 'High_Vol'],
                duplicates='drop'
            )
            
            # VIX regime (3 quantiles) - use VIX if available, otherwise default to Med_VIX
            if 'vix_level' in group.columns:
                vix_data = group['vix_level'].fillna(group['vix_level'].median())
                vix_regime = pd.qcut(
                    vix_data,
                    q=n_regimes,
                    labels=['Low_VIX', 'Med_VIX', 'High_VIX'],
                    duplicates='drop'
                )
            else:
                # Default to medium VIX if VIX data not available
                vix_regime = pd.Series('Med_VIX', index=group.index)
            
            # Combined regime label
            market_regime = vol_regime.astype(str) + '_' + vix_regime.astype(str)
            
            # Assign back to original dataframe using loc
            df_work.loc[group_indices, 'market_regime'] = market_regime.values
        
        # Return series with original index
        return df_work['market_regime']
        
    elif method == "hmm":
        # Optional HMM-based regime detection
        # TODO: This requires proper train/test split to avoid look-ahead bias
        # Currently not used in reported results
        
        if hmm_model is None:
            raise ValueError("hmm_model parameter required for HMM regime detection")
            
        if regime_features is None:
            regime_features = ['volatility_20d', 'vix_level', 'volume_zscore_20d']
            
        # Check if required features are available
        missing_features = [f for f in regime_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for HMM: {missing_features}")
        
        # Extract features for HMM
        feature_data = df[regime_features].fillna(0)
        
        # Predict regimes using fitted HMM
        try:
            regimes = hmm_model.predict_regimes(feature_data)
            return pd.Series(regimes, index=df.index, name='market_regime')
        except Exception as e:
            raise ValueError(f"HMM regime prediction failed: {e}")
    
    else:
        raise ValueError(f"Unknown regime detection method: {method}")


def get_regime_descriptions() -> Dict[str, str]:
    """
    Get descriptions of regime labels used in the current experiment.
    
    Returns:
        Dictionary mapping regime labels to descriptions
    """
    return {
        'Low_Vol_Low_VIX': 'Low volatility, low VIX - Calm, stable market conditions',
        'Low_Vol_Med_VIX': 'Low volatility, medium VIX - Mixed signals, potential transition',
        'Low_Vol_High_VIX': 'Low volatility, high VIX - Unusual combination, potential turning point',
        'Med_Vol_Low_VIX': 'Medium volatility, low VIX - Normal market conditions',
        'Med_Vol_Med_VIX': 'Medium volatility, medium VIX - Typical market environment',
        'Med_Vol_High_VIX': 'Medium volatility, high VIX - Elevated uncertainty and stress',
        'High_Vol_Low_VIX': 'High volatility, low VIX - Unusual market dynamics',
        'High_Vol_Med_VIX': 'High volatility, medium VIX - Stressed market conditions',
        'High_Vol_High_VIX': 'High volatility, high VIX - Crisis or extreme stress conditions'
    }
