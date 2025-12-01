"""
Integration adapter for deepentropy/lwcharts (TradingView Lightweight Charts for Jupyter).

This module provides functions to visualize OHLCV data and patterns using lwcharts.
If lwcharts is not installed, graceful degradation is provided.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import lwcharts
    HAS_LWCHARTS = True
except ImportError:
    HAS_LWCHARTS = False
    lwcharts = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def _check_lwcharts() -> None:
    """Check if lwcharts is available."""
    if not HAS_LWCHARTS:
        raise ImportError(
            "lwcharts is required for visualization. "
            "Install it with: pip install lwcharts"
        )


def _prepare_dataframe(df: Any) -> Any:
    """Prepare DataFrame for lwcharts."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for visualization")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'o']:
            column_mapping[col] = 'open'
        elif col_lower in ['high', 'h']:
            column_mapping[col] = 'high'
        elif col_lower in ['low', 'l']:
            column_mapping[col] = 'low'
        elif col_lower in ['close', 'c', 'adj close', 'adj_close']:
            column_mapping[col] = 'close'
        elif col_lower in ['volume', 'vol', 'v']:
            column_mapping[col] = 'volume'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df


def plot_chart(
    df: Any,
    indicators: Optional[Dict[str, Any]] = None,
    volume: bool = True,
    height: int = 400,
    width: int = 800,
    title: Optional[str] = None
) -> Any:
    """
    Plot OHLCV candlestick chart with optional indicators.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    indicators : dict, optional
        Dictionary of indicator data to overlay. Keys are names, values are
        either Series/arrays or dicts with 'data' and optional 'color', 'type'.
        Example: {'SMA_20': sma_data, 'EMA_10': {'data': ema_data, 'color': 'blue'}}
    volume : bool, optional
        Whether to show volume bars (default: True)
    height : int, optional
        Chart height in pixels (default: 400)
    width : int, optional
        Chart width in pixels (default: 800)
    title : str, optional
        Chart title
    
    Returns
    -------
    Chart object or None if lwcharts is not available
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numta
    >>> df = pd.DataFrame({...})  # Your OHLCV data
    >>> from numta.viz import plot_chart
    >>> chart = plot_chart(df, indicators={'SMA_20': df.ta.sma(20)})
    """
    if not HAS_LWCHARTS:
        print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
        print("Returning None instead of chart.")
        return None
    
    df = _prepare_dataframe(df)
    
    # Create the chart
    chart = lwcharts.Chart(width=width, height=height)
    
    if title:
        chart.title(title)
    
    # Add candlestick series
    candlestick = chart.candlestick_series()
    
    # Prepare OHLCV data for lwcharts
    ohlcv_data = []
    for idx, row in df.iterrows():
        data_point = {
            'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        }
        ohlcv_data.append(data_point)
    
    candlestick.set_data(ohlcv_data)
    
    # Add volume if requested
    if volume and 'volume' in df.columns:
        volume_series = chart.histogram_series(
            pane=1,  # Separate pane for volume
            color='rgba(38, 166, 154, 0.5)'
        )
        
        volume_data = []
        for idx, row in df.iterrows():
            volume_data.append({
                'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
                'value': float(row['volume']),
                'color': 'rgba(38, 166, 154, 0.5)' if row['close'] >= row['open'] else 'rgba(239, 83, 80, 0.5)'
            })
        
        volume_series.set_data(volume_data)
    
    # Add indicators
    if indicators:
        colors = ['blue', 'orange', 'green', 'purple', 'red', 'cyan', 'magenta']
        color_idx = 0
        
        for name, indicator in indicators.items():
            if isinstance(indicator, dict):
                data = indicator.get('data')
                color = indicator.get('color', colors[color_idx % len(colors)])
                line_type = indicator.get('type', 'line')
            else:
                data = indicator
                color = colors[color_idx % len(colors)]
                line_type = 'line'
            
            line_series = chart.line_series(color=color, width=1)
            
            line_data = []
            if HAS_PANDAS and hasattr(data, 'index'):
                # pandas Series
                for idx, val in data.items():
                    if not np.isnan(val):
                        line_data.append({
                            'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
                            'value': float(val)
                        })
            else:
                # numpy array
                for i, val in enumerate(data):
                    if not np.isnan(val):
                        line_data.append({
                            'time': i,
                            'value': float(val)
                        })
            
            line_series.set_data(line_data)
            color_idx += 1
    
    return chart


def plot_pattern(
    df: Any,
    patterns: List[Any],
    show_annotations: bool = True,
    show_trendlines: bool = True,
    height: int = 400,
    width: int = 800
) -> Any:
    """
    Plot OHLCV chart with pattern overlays.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    patterns : list
        List of pattern objects (HeadShouldersPattern, DoublePattern, etc.)
    show_annotations : bool, optional
        Whether to show pattern annotations (default: True)
    show_trendlines : bool, optional
        Whether to show trendlines (default: True)
    height : int, optional
        Chart height in pixels (default: 400)
    width : int, optional
        Chart width in pixels (default: 800)
    
    Returns
    -------
    Chart object or None if lwcharts is not available
    """
    if not HAS_LWCHARTS:
        print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
        return None
    
    # Start with base chart
    chart = plot_chart(df, height=height, width=width, volume=False)
    
    if chart is None:
        return None
    
    # Add pattern markers
    for pattern in patterns:
        pattern_type = getattr(pattern, 'pattern_type', 'unknown')
        
        if hasattr(pattern, 'neckline') and show_trendlines:
            # Draw neckline for head and shoulders patterns
            slope, intercept = pattern.neckline
            start_idx = pattern.start_index
            end_idx = pattern.end_index
            
            line_series = chart.line_series(
                color='red' if 'inverse' not in pattern_type else 'green',
                width=2,
                style='dashed'
            )
            line_series.set_data([
                {'time': start_idx, 'value': slope * start_idx + intercept},
                {'time': end_idx, 'value': slope * end_idx + intercept}
            ])
        
        if hasattr(pattern, 'upper_trendline') and show_trendlines:
            # Draw triangle/wedge trendlines
            upper_slope, upper_intercept = pattern.upper_trendline
            lower_slope, lower_intercept = pattern.lower_trendline
            start_idx = pattern.start_index
            end_idx = pattern.end_index
            
            upper_line = chart.line_series(color='blue', width=1, style='solid')
            upper_line.set_data([
                {'time': start_idx, 'value': upper_slope * start_idx + upper_intercept},
                {'time': end_idx, 'value': upper_slope * end_idx + upper_intercept}
            ])
            
            lower_line = chart.line_series(color='blue', width=1, style='solid')
            lower_line.set_data([
                {'time': start_idx, 'value': lower_slope * start_idx + lower_intercept},
                {'time': end_idx, 'value': lower_slope * end_idx + lower_intercept}
            ])
    
    return chart


def plot_harmonic(
    df: Any,
    patterns: List[Any],
    show_fibonacci: bool = True,
    show_prz: bool = True,
    height: int = 400,
    width: int = 800
) -> Any:
    """
    Plot OHLCV chart with harmonic pattern overlays.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    patterns : list
        List of HarmonicPattern objects
    show_fibonacci : bool, optional
        Whether to show Fibonacci ratios (default: True)
    show_prz : bool, optional
        Whether to show Potential Reversal Zone (default: True)
    height : int, optional
        Chart height in pixels (default: 400)
    width : int, optional
        Chart width in pixels (default: 800)
    
    Returns
    -------
    Chart object or None if lwcharts is not available
    """
    if not HAS_LWCHARTS:
        print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
        return None
    
    # Start with base chart
    chart = plot_chart(df, height=height, width=width, volume=False)
    
    if chart is None:
        return None
    
    # Add harmonic pattern lines
    for pattern in patterns:
        color = 'green' if pattern.direction == 'bullish' else 'red'
        
        # Draw XABCD lines
        points = [pattern.X, pattern.A, pattern.B, pattern.C, pattern.D]
        
        for i in range(len(points) - 1):
            line_series = chart.line_series(color=color, width=2)
            line_series.set_data([
                {'time': points[i][0], 'value': points[i][1]},
                {'time': points[i + 1][0], 'value': points[i + 1][1]}
            ])
        
        # Show PRZ zone if requested
        if show_prz:
            prz_lower, prz_upper = pattern.prz
            # Note: lwcharts doesn't support zones natively, 
            # we'd need to use line markers or custom rendering
    
    return chart


# Export availability flag
__all__ = [
    'HAS_LWCHARTS',
    'plot_chart',
    'plot_pattern',
    'plot_harmonic',
]
