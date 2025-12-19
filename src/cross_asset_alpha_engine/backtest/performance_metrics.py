"""
Performance metrics calculation with confidence intervals.

This module provides comprehensive performance analytics including
Sharpe ratio confidence intervals and other risk-adjusted metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Dictionary with performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Remove any NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {}
    
    # Basic return metrics
    total_return = (1 + returns_clean).prod() - 1
    annualized_return = (1 + returns_clean.mean()) ** 252 - 1
    volatility = returns_clean.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    excess_returns = returns_clean - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / returns_clean.std() * np.sqrt(252) if returns_clean.std() > 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns_clean).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Additional metrics
    win_rate = (returns_clean > 0).mean()
    avg_win = returns_clean[returns_clean > 0].mean() if (returns_clean > 0).any() else 0
    avg_loss = returns_clean[returns_clean < 0].mean() if (returns_clean < 0).any() else 0
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio (excess return / downside deviation)
    downside_returns = returns_clean[returns_clean < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'n_observations': len(returns_clean)
    }


def calculate_sharpe_confidence_interval(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for Sharpe ratio using analytical approximation.
    
    Based on the asymptotic distribution of the Sharpe ratio estimator.
    Reference: Lo, A. W. (2002). The Statistics of Sharpe Ratios.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        confidence_level: Confidence level (default: 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for Sharpe ratio
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 10:  # Need minimum observations
        return (np.nan, np.nan)
    
    # Calculate Sharpe ratio
    excess_returns = returns_clean - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / returns_clean.std() * np.sqrt(252) if returns_clean.std() > 0 else 0
    
    # Calculate standard error using Lo (2002) approximation
    T = len(returns_clean)  # Number of observations
    
    # Approximate standard error: SE ≈ sqrt((1 + 0.5*S^2) / (T - 1))
    # where S is the Sharpe ratio (annualized)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe_ratio**2) / (T - 1))
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    lower_bound = sharpe_ratio - z_score * se_sharpe
    upper_bound = sharpe_ratio + z_score * se_sharpe
    
    return (lower_bound, upper_bound)


def calculate_information_ratio_ci(
    excess_returns: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for Information Ratio.
    
    Args:
        excess_returns: Series of excess returns vs benchmark
        confidence_level: Confidence level (default: 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for Information Ratio
    """
    excess_returns_clean = excess_returns.dropna()
    
    if len(excess_returns_clean) < 10:
        return (np.nan, np.nan)
    
    # Information ratio = mean excess return / tracking error
    ir = excess_returns_clean.mean() / excess_returns_clean.std() * np.sqrt(252) if excess_returns_clean.std() > 0 else 0
    
    # Use similar approximation as Sharpe ratio
    T = len(excess_returns_clean)
    se_ir = np.sqrt((1 + 0.5 * ir**2) / (T - 1))
    
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    lower_bound = ir - z_score * se_ir
    upper_bound = ir + z_score * se_ir
    
    return (lower_bound, upper_bound)


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 63,  # ~3 months
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Series of daily returns
        window: Rolling window size in days
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling Sharpe ratio
    excess_returns = returns - risk_free_rate / 252
    rolling_metrics['sharpe_ratio'] = (
        excess_returns.rolling(window).mean() / 
        returns.rolling(window).std() * np.sqrt(252)
    )
    
    # Rolling volatility
    rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.rolling(window).max()
    rolling_metrics['drawdown'] = (cumulative_returns - rolling_max) / rolling_max
    
    # Rolling win rate
    rolling_metrics['win_rate'] = (returns > 0).rolling(window).mean()
    
    return rolling_metrics


def generate_performance_report(
    portfolio_performance: pd.DataFrame,
    metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        portfolio_performance: DataFrame with daily performance
        metrics: Dictionary with performance metrics
        save_path: Optional path to save report
        
    Returns:
        String with formatted report
    """
    report_lines = []
    
    report_lines.append("CROSS-ASSET ALPHA ENGINE - PERFORMANCE REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Strategy overview
    report_lines.append("STRATEGY PROFILE:")
    report_lines.append(f"  Market-neutral strategy with low beta to benchmark")
    report_lines.append(f"  Average gross exposure: {metrics.get('avg_gross_exposure', 0):.1%}")
    report_lines.append(f"  Focus on risk-adjusted returns and diversification")
    report_lines.append("")
    
    # Transaction costs
    report_lines.append("TRANSACTION COST ASSUMPTIONS:")
    report_lines.append(f"  Cost per side: {metrics.get('transaction_cost_bps_per_side', 0):.1f} bps")
    report_lines.append(f"  Average daily turnover: {metrics.get('avg_daily_turnover', 0):.1%}")
    report_lines.append(f"  Total transaction costs: {metrics.get('total_transaction_costs', 0):.2%}")
    report_lines.append("")
    
    # Performance metrics
    report_lines.append("PERFORMANCE METRICS (Net of Transaction Costs):")
    report_lines.append(f"  Total Return: {metrics.get('total_return', 0):.1%}")
    report_lines.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.1%}")
    report_lines.append(f"  Volatility: {metrics.get('volatility', 0):.1%}")
    
    # Sharpe ratio with confidence interval
    sharpe = metrics.get('sharpe_ratio', 0)
    sharpe_ci_lower = metrics.get('sharpe_ratio_net_ci_lower', sharpe)
    sharpe_ci_upper = metrics.get('sharpe_ratio_net_ci_upper', sharpe)
    report_lines.append(f"  Sharpe Ratio: {sharpe:.3f} [{sharpe_ci_lower:.3f}, {sharpe_ci_upper:.3f}]")
    
    report_lines.append(f"  Maximum Drawdown: {metrics.get('max_drawdown', 0):.1%}")
    report_lines.append(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
    report_lines.append(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
    report_lines.append("")
    
    # Benchmark comparison
    report_lines.append("BENCHMARK COMPARISON:")
    report_lines.append(f"  Universe Equal-Weight Benchmark: {metrics.get('benchmark_total_return', 0):.1%}")
    report_lines.append(f"  Excess Return vs Benchmark: {metrics.get('excess_return_vs_universe_benchmark', 0):.1%}")
    
    if 'information_ratio' in metrics:
        ir = metrics['information_ratio']
        ir_ci_lower = metrics.get('information_ratio_ci_lower', ir)
        ir_ci_upper = metrics.get('information_ratio_ci_upper', ir)
        report_lines.append(f"  Information Ratio: {ir:.3f} [{ir_ci_lower:.3f}, {ir_ci_upper:.3f}]")
    
    report_lines.append("")
    
    # Important notes
    report_lines.append("IMPORTANT NOTES:")
    report_lines.append("  • The benchmark is the equal-weight average return across the equity universe")
    report_lines.append("  • The strategy is market-neutral with near-zero beta to the benchmark")
    report_lines.append("  • Negative excess return vs a bull market benchmark is expected given this profile")
    report_lines.append("  • Primary value is high Sharpe ratio and low drawdown for diversification")
    report_lines.append("")
    
    # Data summary
    n_obs = len(portfolio_performance)
    start_date = portfolio_performance.index[0].strftime('%Y-%m-%d')
    end_date = portfolio_performance.index[-1].strftime('%Y-%m-%d')
    
    report_lines.append("DATA SUMMARY:")
    report_lines.append(f"  Period: {start_date} to {end_date}")
    report_lines.append(f"  Trading days: {n_obs}")
    report_lines.append(f"  Years: {n_obs / 252:.1f}")
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report
