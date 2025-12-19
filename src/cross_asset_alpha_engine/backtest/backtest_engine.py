"""
Backtesting engine with daily execution and transaction costs.

IMPORTANT: Execution is modeled at daily close-to-close with simple costs,
not an intraday microstructure model. All analysis uses daily OHLCV bars only.

This module provides comprehensive backtesting capabilities that include:
- Transaction cost modeling (simplified for daily rebalancing)
- Daily turnover calculation  
- Gross and net return tracking
- Market-neutral portfolio construction
- Note: True intraday execution simulation (order books, tick data) is not used
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging_utils import LoggerMixin


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    
    # Transaction cost parameters (in basis points)
    transaction_cost_bps_per_side: float = 5.0  # 0.05% per trade side (conservative)
    
    # Portfolio construction parameters
    max_position: float = 0.10  # 10% max position per asset
    max_gross_exposure: float = 1.0  # 100% max gross exposure
    target_net_exposure: float = 0.0  # Market neutral target
    
    # Rebalancing parameters
    rebalance_frequency: str = "daily"  # Currently only daily supported
    
    # Performance calculation
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Output configuration
    save_portfolio_performance: bool = True
    save_detailed_positions: bool = False


class BacktestEngine(LoggerMixin):
    """
    Comprehensive backtesting engine with daily execution and transaction costs.
    
    IMPORTANT: Execution is modeled at daily close-to-close with simple costs,
    not an intraday microstructure model. All analysis uses daily OHLCV bars only.
    
    The engine tracks both gross and net returns, calculates daily turnover,
    and applies transaction costs to provide realistic performance estimates
    for daily rebalancing strategies.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize backtest engine.
        
        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        self.logger.info(f"Initialized BacktestEngine with transaction costs: {self.config.transaction_cost_bps_per_side} bps per side")
    
    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        alpha_col: str = 'alpha_regime',
        target_col: str = 'target_1d'
    ) -> Dict[str, Any]:
        """
        Run complete backtest with transaction costs and turnover tracking.
        
        Args:
            predictions_df: DataFrame with predictions, must have columns:
                - timestamp, symbol, alpha_col, target_col
            alpha_col: Column name for alpha predictions
            target_col: Column name for target returns
            
        Returns:
            Dictionary containing:
                - portfolio_performance: DataFrame with daily performance
                - metrics: Performance metrics dictionary
                - positions: Daily positions (if save_detailed_positions=True)
        """
        self.logger.info("Starting backtest execution")
        
        # Validate input data
        required_cols = ['timestamp', 'symbol', alpha_col, target_col]
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by timestamp for proper chronological processing
        data = predictions_df.sort_values(['timestamp', 'symbol']).copy()
        
        # Build portfolio positions
        portfolio_data = self._construct_portfolio(data, alpha_col)
        
        # Calculate returns with transaction costs
        portfolio_performance = self._calculate_portfolio_returns(portfolio_data, target_col)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(portfolio_performance)
        
        # Prepare results
        results = {
            'portfolio_performance': portfolio_performance,
            'metrics': metrics,
            'config': self.config
        }
        
        if self.config.save_detailed_positions:
            results['positions'] = portfolio_data[['timestamp', 'symbol', 'position']].copy()
        
        self.logger.info("Backtest execution completed")
        
        # Log metrics with proper formatting (handle string 'N/A' case)
        sharpe = metrics.get('sharpe_ratio', 'N/A')
        sharpe_str = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        self.logger.info(f"Net Sharpe Ratio: {sharpe_str}")
        
        turnover = metrics.get('avg_daily_turnover', 'N/A')
        turnover_str = f"{turnover:.3f}" if isinstance(turnover, (int, float)) else str(turnover)
        self.logger.info(f"Average Daily Turnover: {turnover_str}")
        
        return results
    
    def _construct_portfolio(
        self,
        data: pd.DataFrame,
        alpha_col: str
    ) -> pd.DataFrame:
        """
        Construct market-neutral portfolio based on alpha predictions.
        
        Args:
            data: DataFrame with alpha predictions
            alpha_col: Column name for alpha scores
            
        Returns:
            DataFrame with position allocations
        """
        portfolio_data = []
        
        for date, group in data.groupby('timestamp'):
            group = group.copy()
            
            # Rank alpha predictions and convert to z-scores
            group['alpha_rank'] = group[alpha_col].rank(ascending=False)
            alpha_mean = group[alpha_col].mean()
            alpha_std = group[alpha_col].std()
            
            if alpha_std > 0:
                group['alpha_zscore'] = (group[alpha_col] - alpha_mean) / alpha_std
            else:
                group['alpha_zscore'] = 0.0
            
            # Position sizing based on alpha z-score
            # Scale by 0.05 to get reasonable position sizes
            group['position'] = np.clip(
                group['alpha_zscore'] * 0.05, 
                -self.config.max_position, 
                self.config.max_position
            )
            
            # Ensure market neutrality (positions sum to approximately zero)
            position_sum = group['position'].sum()
            group['position'] = group['position'] - position_sum / len(group)
            
            # Apply gross exposure limit
            gross_exposure = group['position'].abs().sum()
            if gross_exposure > self.config.max_gross_exposure:
                scaling_factor = self.config.max_gross_exposure / gross_exposure
                group['position'] = group['position'] * scaling_factor
            
            portfolio_data.append(group)
        
        return pd.concat(portfolio_data, ignore_index=True)
    
    def _calculate_portfolio_returns(
        self,
        portfolio_data: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Calculate portfolio returns with transaction costs and turnover tracking.
        
        Args:
            portfolio_data: DataFrame with positions and target returns
            target_col: Column name for target returns
            
        Returns:
            DataFrame with portfolio performance metrics
        """
        # Calculate position returns for each asset
        portfolio_data['position_return'] = portfolio_data['position'] * portfolio_data[target_col]
        
        # Aggregate to portfolio level by date
        daily_performance = portfolio_data.groupby('timestamp').agg({
            'position_return': 'sum',  # Portfolio return (gross)
            'position': lambda x: x.abs().sum(),  # Gross exposure
            target_col: 'mean'  # Universe benchmark return (equal-weight average)
        }).rename(columns={
            'position_return': 'portfolio_return_gross',
            'position': 'gross_exposure',
            target_col: 'universe_benchmark_return'
        })
        
        # Calculate daily turnover and transaction costs
        daily_performance['daily_turnover'] = 0.0
        daily_performance['transaction_costs'] = 0.0
        daily_performance['portfolio_return_net'] = daily_performance['portfolio_return_gross']
        
        # Track previous positions for turnover calculation
        prev_positions = {}
        
        for date in daily_performance.index:
            current_positions = portfolio_data[portfolio_data['timestamp'] == date].set_index('symbol')['position']
            
            if prev_positions:
                # Calculate turnover as sum of absolute position changes
                common_symbols = set(current_positions.index) & set(prev_positions.keys())
                
                turnover = 0.0
                for symbol in common_symbols:
                    position_change = abs(current_positions[symbol] - prev_positions[symbol])
                    turnover += position_change
                
                # Add new positions (full position size counts as turnover)
                new_symbols = set(current_positions.index) - set(prev_positions.keys())
                for symbol in new_symbols:
                    turnover += abs(current_positions[symbol])
                
                # Add closed positions (full position size counts as turnover)  
                closed_symbols = set(prev_positions.keys()) - set(current_positions.index)
                for symbol in closed_symbols:
                    turnover += abs(prev_positions[symbol])
                
                daily_performance.loc[date, 'daily_turnover'] = turnover
                
                # Calculate transaction costs
                transaction_cost = turnover * (self.config.transaction_cost_bps_per_side / 10000.0)
                daily_performance.loc[date, 'transaction_costs'] = transaction_cost
                
                # Net return = gross return - transaction costs
                daily_performance.loc[date, 'portfolio_return_net'] = (
                    daily_performance.loc[date, 'portfolio_return_gross'] - transaction_cost
                )
            
            # Update previous positions
            prev_positions = current_positions.to_dict()
        
        # Calculate cumulative returns
        daily_performance['cumulative_return_gross'] = (1 + daily_performance['portfolio_return_gross']).cumprod()
        daily_performance['cumulative_return_net'] = (1 + daily_performance['portfolio_return_net']).cumprod()
        daily_performance['cumulative_benchmark'] = (1 + daily_performance['universe_benchmark_return']).cumprod()
        
        # Calculate excess returns vs benchmark
        daily_performance['excess_return_gross'] = (
            daily_performance['portfolio_return_gross'] - daily_performance['universe_benchmark_return']
        )
        daily_performance['excess_return_net'] = (
            daily_performance['portfolio_return_net'] - daily_performance['universe_benchmark_return']
        )
        
        return daily_performance
    
    def _calculate_metrics(self, portfolio_performance: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_performance: DataFrame with daily performance
            
        Returns:
            Dictionary with performance metrics
        """
        from .performance_metrics import calculate_performance_metrics, calculate_sharpe_confidence_interval
        
        # Calculate metrics for both gross and net returns
        gross_returns = portfolio_performance['portfolio_return_gross']
        net_returns = portfolio_performance['portfolio_return_net']
        benchmark_returns = portfolio_performance['universe_benchmark_return']
        excess_returns_net = portfolio_performance['excess_return_net']
        
        # Basic performance metrics
        metrics = {}
        
        # Gross return metrics
        gross_metrics = calculate_performance_metrics(gross_returns, self.config.risk_free_rate)
        for key, value in gross_metrics.items():
            metrics[f"{key}_gross"] = value
        
        # Net return metrics (primary metrics)
        net_metrics = calculate_performance_metrics(net_returns, self.config.risk_free_rate)
        for key, value in net_metrics.items():
            if key.endswith('_gross'):
                metrics[key.replace('_gross', '_net')] = value
            else:
                metrics[key] = value
        
        # Benchmark metrics
        benchmark_metrics = calculate_performance_metrics(benchmark_returns, self.config.risk_free_rate)
        for key, value in benchmark_metrics.items():
            metrics[f"benchmark_{key}"] = value
        
        # Excess return metrics (vs benchmark)
        if len(excess_returns_net) > 0 and excess_returns_net.std() > 0:
            metrics['information_ratio'] = excess_returns_net.mean() / excess_returns_net.std() * np.sqrt(252)
            
            # Sharpe confidence interval for excess returns
            ir_ci = calculate_sharpe_confidence_interval(excess_returns_net)
            metrics['information_ratio_ci_lower'] = ir_ci[0]
            metrics['information_ratio_ci_upper'] = ir_ci[1]
        
        # Sharpe confidence intervals
        sharpe_ci_gross = calculate_sharpe_confidence_interval(gross_returns, self.config.risk_free_rate)
        sharpe_ci_net = calculate_sharpe_confidence_interval(net_returns, self.config.risk_free_rate)
        
        metrics['sharpe_ratio_gross_ci_lower'] = sharpe_ci_gross[0]
        metrics['sharpe_ratio_gross_ci_upper'] = sharpe_ci_gross[1]
        metrics['sharpe_ratio_net_ci_lower'] = sharpe_ci_net[0]
        metrics['sharpe_ratio_net_ci_upper'] = sharpe_ci_net[1]
        
        # Transaction cost and turnover metrics
        metrics['avg_daily_turnover'] = portfolio_performance['daily_turnover'].mean()
        metrics['total_transaction_costs'] = portfolio_performance['transaction_costs'].sum()
        metrics['avg_daily_transaction_costs'] = portfolio_performance['transaction_costs'].mean()
        metrics['transaction_cost_bps_per_side'] = self.config.transaction_cost_bps_per_side
        
        # Portfolio characteristics
        metrics['avg_gross_exposure'] = portfolio_performance['gross_exposure'].mean()
        metrics['avg_net_exposure'] = portfolio_performance['portfolio_return_net'].mean()  # Should be near zero
        
        # Excess return vs benchmark
        total_excess_return = (
            portfolio_performance['cumulative_return_net'].iloc[-1] - 
            portfolio_performance['cumulative_benchmark'].iloc[-1]
        )
        metrics['excess_return_vs_universe_benchmark'] = total_excess_return
        
        # Add descriptive labels
        metrics['_labels'] = {
            'portfolio_return_net': 'Portfolio Total Return (Net)',
            'benchmark_total_return': 'Universe Equal-Weight Benchmark Return',
            'excess_return_vs_universe_benchmark': 'Excess Return vs Universe Benchmark',
            'sharpe_ratio': 'Sharpe Ratio (Net of Costs)',
            'information_ratio': 'Information Ratio vs Universe Benchmark',
            'avg_daily_turnover': 'Average Daily Turnover (as fraction of capital)',
            'avg_gross_exposure': 'Average Gross Exposure'
        }
        
        return metrics
    
    def print_performance_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted performance summary.
        
        Args:
            results: Results dictionary from run_backtest()
        """
        metrics = results['metrics']
        config = results['config']
        
        # Helper function to safely format values
        def safe_format(value, format_str, default='N/A'):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            if isinstance(value, (int, float)):
                return format_str.format(value)
            return str(value)
        
        print("\n" + "="*60)
        print("CROSS-ASSET ALPHA ENGINE - BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nStrategy Profile:")
        avg_gross = safe_format(metrics.get('avg_gross_exposure'), '{:.1%}')
        print(f"  Market-neutral with avg gross exposure ~{avg_gross}")
        print(f"  Low beta to universe benchmark (near-zero net exposure)")
        print(f"  Focus on risk-adjusted returns and diversification")
        
        print(f"\nTransaction Cost Assumptions:")
        print(f"  Cost per side: {config.transaction_cost_bps_per_side:.1f} bps")
        turnover = safe_format(metrics.get('avg_daily_turnover'), '{:.1%}')
        total_costs = safe_format(metrics.get('total_transaction_costs'), '{:.2%}')
        print(f"  Average daily turnover: {turnover}")
        print(f"  Total transaction costs: {total_costs}")
        
        print(f"\nPerformance Metrics (Net of Transaction Costs):")
        total_ret = safe_format(metrics.get('total_return'), '{:.1%}')
        ann_ret = safe_format(metrics.get('annualized_return'), '{:.1%}')
        vol = safe_format(metrics.get('volatility'), '{:.1%}')
        print(f"  Total Return: {total_ret}")
        print(f"  Annualized Return: {ann_ret}")
        print(f"  Volatility: {vol}")
        
        sharpe = metrics.get('sharpe_ratio')
        sharpe_lower = metrics.get('sharpe_ratio_net_ci_lower')
        sharpe_upper = metrics.get('sharpe_ratio_net_ci_upper')
        if sharpe is not None and not (isinstance(sharpe, float) and np.isnan(sharpe)):
            sharpe_str = f"{sharpe:.3f}"
            if sharpe_lower is not None and sharpe_upper is not None:
                if not (isinstance(sharpe_lower, float) and np.isnan(sharpe_lower)):
                    sharpe_str += f" [{sharpe_lower:.3f}, {sharpe_upper:.3f}]"
            print(f"  Sharpe Ratio: {sharpe_str}")
        else:
            print(f"  Sharpe Ratio: N/A")
        
        max_dd = safe_format(metrics.get('max_drawdown'), '{:.1%}')
        win_rate = safe_format(metrics.get('win_rate'), '{:.1%}')
        print(f"  Maximum Drawdown: {max_dd}")
        print(f"  Win Rate: {win_rate}")
        
        print(f"\nBenchmark Comparison:")
        bench_ret = safe_format(metrics.get('benchmark_total_return'), '{:.1%}')
        excess_ret = safe_format(metrics.get('excess_return_vs_universe_benchmark'), '{:.1%}')
        print(f"  Universe Equal-Weight Benchmark Return: {bench_ret}")
        print(f"  Excess Return vs Benchmark: {excess_ret}")
        
        if 'information_ratio' in metrics:
            ir = metrics['information_ratio']
            ir_lower = metrics.get('information_ratio_ci_lower')
            ir_upper = metrics.get('information_ratio_ci_upper')
            if ir is not None and not (isinstance(ir, float) and np.isnan(ir)):
                ir_str = f"{ir:.3f}"
                if ir_lower is not None and ir_upper is not None:
                    if not (isinstance(ir_lower, float) and np.isnan(ir_lower)):
                        ir_str += f" [{ir_lower:.3f}, {ir_upper:.3f}]"
                print(f"  Information Ratio: {ir_str}")
        
        print(f"\nNote: The benchmark is the equal-weight average return across the equity")
        print(f"universe. The strategy is constructed to be nearly market-neutral; therefore,")
        print(f"we expect it to have low beta to this benchmark and to prioritize risk-adjusted")
        print(f"metrics (Sharpe, max drawdown) over raw excess return, especially in strong bull markets.")
        
        print("\n" + "="*60)
