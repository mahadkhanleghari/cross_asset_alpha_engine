"""Backtesting modules for Cross-Asset Alpha Engine.

This package provides comprehensive backtesting capabilities with realistic
transaction costs, turnover tracking, and performance analytics.
"""

from .backtest_engine import BacktestEngine, BacktestConfig
from .performance_metrics import calculate_performance_metrics, calculate_sharpe_confidence_interval

__all__ = [
    "BacktestEngine",
    "BacktestConfig", 
    "calculate_performance_metrics",
    "calculate_sharpe_confidence_interval",
]
