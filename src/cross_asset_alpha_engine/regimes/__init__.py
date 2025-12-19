"""Regime detection modules for Cross-Asset Alpha Engine.

This package provides regime detection capabilities. The current experiment uses 
quantile-based volatility/VIX regimes. HMM-based detection is available as an 
optional extension but is not used in the reported results.
"""

from .hmm_regime_model import RegimeHMM
from .regime_features import RegimeFeatureEngine, assign_regimes, get_regime_descriptions

__all__ = [
    "RegimeHMM",
    "RegimeFeatureEngine", 
    "assign_regimes",
    "get_regime_descriptions",
]
