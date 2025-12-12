"""Regime detection modules for Cross-Asset Alpha Engine.

This package provides regime detection capabilities using Hidden Markov Models
and other statistical techniques to identify market regimes.
"""

from .hmm_regime_model import RegimeHMM
from .regime_features import RegimeFeatureEngine

__all__ = [
    "RegimeHMM",
    "RegimeFeatureEngine",
]
