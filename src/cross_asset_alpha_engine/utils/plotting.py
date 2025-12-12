"""Plotting utilities for Cross-Asset Alpha Engine.

This module provides visualization functions for equity curves, regime overlays,
feature analysis, and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Union, Tuple

from .logging_utils import get_logger

logger = get_logger(__name__)


def plot_equity_curve(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """Plot equity curve from returns series.
    
    Args:
        returns: Series of returns with datetime index
        benchmark_returns: Optional benchmark returns for comparison
        title: Plot title
        figsize: Figure size for matplotlib
        interactive: Whether to create interactive plotly chart
        
    Returns:
        Matplotlib or Plotly figure object
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    if interactive:
        # Create interactive Plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))
        
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    else:
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(cum_returns.index, cum_returns.values, 
               label='Strategy', color='blue', linewidth=2)
        
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cum.index, benchmark_cum.values,
                   label='Benchmark', color='red', linewidth=2, linestyle='--')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown Analysis",
    figsize: Tuple[int, int] = (12, 6),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """Plot drawdown analysis.
    
    Args:
        returns: Series of returns with datetime index
        title: Plot title
        figsize: Figure size for matplotlib
        interactive: Whether to create interactive plotly chart
        
    Returns:
        Matplotlib or Plotly figure object
    """
    # Calculate cumulative returns and drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    if interactive:
        # Create interactive Plotly chart with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values,
                      mode='lines', name='Equity', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      mode='lines', name='Drawdown', 
                      line=dict(color='red'), fill='tozeroy'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=600
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        
        return fig
    
    else:
        # Create matplotlib chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Equity curve
        ax1.plot(cum_returns.index, cum_returns.values, color='blue', linewidth=2)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color='red', alpha=0.7, label='Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig


def plot_regime_overlay(
    prices: pd.Series,
    regimes: pd.Series,
    regime_names: Optional[Dict[int, str]] = None,
    title: str = "Price with Regime Overlay",
    figsize: Tuple[int, int] = (14, 8),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """Plot price series with regime overlay.
    
    Args:
        prices: Price series with datetime index
        regimes: Regime series with same index as prices
        regime_names: Optional mapping of regime numbers to names
        title: Plot title
        figsize: Figure size for matplotlib
        interactive: Whether to create interactive plotly chart
        
    Returns:
        Matplotlib or Plotly figure object
    """
    if regime_names is None:
        unique_regimes = sorted(regimes.unique())
        regime_names = {i: f"Regime {i}" for i in unique_regimes}
    
    # Define colors for regimes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    regime_colors = {regime: colors[i % len(colors)] 
                    for i, regime in enumerate(sorted(regimes.unique()))}
    
    if interactive:
        # Create interactive Plotly chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price', 'Regimes'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price series
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices.values,
                      mode='lines', name='Price', line=dict(color='black')),
            row=1, col=1
        )
        
        # Regime overlay on price chart
        for regime in sorted(regimes.unique()):
            regime_mask = regimes == regime
            regime_periods = prices[regime_mask]
            
            if not regime_periods.empty:
                fig.add_trace(
                    go.Scatter(x=regime_periods.index, y=regime_periods.values,
                              mode='markers', name=regime_names[regime],
                              marker=dict(color=regime_colors[regime], size=3)),
                    row=1, col=1
                )
        
        # Regime time series
        fig.add_trace(
            go.Scatter(x=regimes.index, y=regimes.values,
                      mode='lines+markers', name='Regime',
                      line=dict(color='gray'), showlegend=False),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=600
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=1)
        
        return fig
    
    else:
        # Create matplotlib chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Price chart with regime background
        ax1.plot(prices.index, prices.values, color='black', linewidth=1.5, label='Price')
        
        # Add regime background colors
        for regime in sorted(regimes.unique()):
            regime_periods = regimes == regime
            
            # Find continuous periods of this regime
            regime_changes = regime_periods.astype(int).diff().fillna(0)
            starts = regime_changes[regime_changes == 1].index
            ends = regime_changes[regime_changes == -1].index
            
            # Handle case where regime continues to the end
            if len(starts) > len(ends):
                ends = ends.tolist() + [regimes.index[-1]]
            
            for start, end in zip(starts, ends):
                ax1.axvspan(start, end, alpha=0.3, color=regime_colors[regime],
                           label=regime_names[regime] if start == starts[0] else "")
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regime time series
        ax2.plot(regimes.index, regimes.values, color='gray', linewidth=2)
        ax2.set_ylabel('Regime')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig


def plot_feature_correlation(
    features_df: pd.DataFrame,
    method: str = "pearson",
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """Plot correlation matrix of features.
    
    Args:
        features_df: DataFrame with features as columns
        method: Correlation method ('pearson', 'spearman', 'kendall')
        title: Plot title
        figsize: Figure size for matplotlib
        interactive: Whether to create interactive plotly chart
        
    Returns:
        Matplotlib or Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = features_df.corr(method=method)
    
    if interactive:
        # Create interactive Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=600,
            height=600
        )
        
        return fig
    
    else:
        # Create matplotlib heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(corr_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.index)
        
        # Add correlation values as text
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


def plot_returns_distribution(
    returns: pd.Series,
    bins: int = 50,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (10, 6),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """Plot distribution of returns.
    
    Args:
        returns: Series of returns
        bins: Number of histogram bins
        title: Plot title
        figsize: Figure size for matplotlib
        interactive: Whether to create interactive plotly chart
        
    Returns:
        Matplotlib or Plotly figure object
    """
    if interactive:
        # Create interactive Plotly histogram
        fig = go.Figure(data=[go.Histogram(x=returns.values, nbinsx=bins)])
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        # Add statistics as annotations
        mean_ret = returns.mean()
        std_ret = returns.std()
        skew_ret = returns.skew()
        kurt_ret = returns.kurtosis()
        
        fig.add_annotation(
            x=0.7, y=0.9,
            xref="paper", yref="paper",
            text=f"Mean: {mean_ret:.4f}<br>"
                 f"Std: {std_ret:.4f}<br>"
                 f"Skew: {skew_ret:.2f}<br>"
                 f"Kurt: {kurt_ret:.2f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    else:
        # Create matplotlib histogram
        fig, ax = plt.subplots(figsize=figsize)
        
        n, bins_edges, patches = ax.hist(returns.values, bins=bins, 
                                        alpha=0.7, color='blue', edgecolor='black')
        
        # Add vertical line for mean
        ax.axvline(returns.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {returns.mean():.4f}')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'Mean: {returns.mean():.4f}\n' \
                    f'Std: {returns.std():.4f}\n' \
                    f'Skew: {returns.skew():.2f}\n' \
                    f'Kurt: {returns.kurtosis():.2f}'
        
        ax.text(0.7, 0.9, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: List[str] = None,
    title: str = "Rolling Performance Metrics",
    figsize: Tuple[int, int] = (14, 10),
    interactive: bool = False
) -> Union[plt.Figure, go.Figure]:
    """Plot rolling performance metrics.
    
    Args:
        returns: Series of returns with datetime index
        window: Rolling window size
        metrics: List of metrics to plot ('sharpe', 'volatility', 'max_dd')
        title: Plot title
        figsize: Figure size for matplotlib
        interactive: Whether to create interactive plotly chart
        
    Returns:
        Matplotlib or Plotly figure object
    """
    if metrics is None:
        metrics = ['sharpe', 'volatility', 'max_dd']
    
    # Calculate rolling metrics
    rolling_data = {}
    
    if 'sharpe' in metrics:
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        rolling_data['Sharpe Ratio'] = rolling_sharpe
    
    if 'volatility' in metrics:
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_data['Volatility'] = rolling_vol
    
    if 'max_dd' in metrics:
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.rolling(window).max()
        rolling_dd = (cum_returns - rolling_max) / rolling_max
        rolling_data['Max Drawdown'] = rolling_dd.rolling(window).min()
    
    if interactive:
        # Create interactive Plotly chart with subplots
        n_metrics = len(rolling_data)
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=list(rolling_data.keys()),
            vertical_spacing=0.1
        )
        
        for i, (metric_name, metric_data) in enumerate(rolling_data.items(), 1):
            fig.add_trace(
                go.Scatter(x=metric_data.index, y=metric_data.values,
                          mode='lines', name=metric_name, showlegend=False),
                row=i, col=1
            )
        
        fig.update_layout(
            title=title,
            height=200 * n_metrics + 100
        )
        
        return fig
    
    else:
        # Create matplotlib chart
        n_metrics = len(rolling_data)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, metric_data) in zip(axes, rolling_data.items()):
            ax.plot(metric_data.index, metric_data.values, linewidth=2)
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            if metric_name == 'Sharpe Ratio':
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date')
        
        # Format x-axis dates
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
