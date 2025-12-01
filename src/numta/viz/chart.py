"""
Chart plotting functions for numta visualization.

This module provides functions to create OHLC candlestick charts
and line charts using lwcharts.
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


def _prepare_ohlcv_data(df: Any) -> list:
    """Prepare OHLCV data for lwcharts from DataFrame."""
    ohlcv_data = []
    
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
    
    df = df.copy()
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    for idx, row in df.iterrows():
        data_point = {
            'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        }
        ohlcv_data.append(data_point)
    
    return ohlcv_data


def plot_ohlc(
    df: Any,
    volume: bool = True,
    height: int = 400,
    width: int = 800,
    title: Optional[str] = None
) -> Any:
    """
    Plot OHLCV candlestick chart.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with OHLCV data. Expected columns (case-insensitive):
        - open (or o)
        - high (or h)
        - low (or l)
        - close (or c)
        - volume (optional, or vol, v)
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
    >>> from numta.viz import plot_ohlc
    >>> df = pd.DataFrame({
    ...     'open': [100, 101, 102],
    ...     'high': [105, 106, 107],
    ...     'low': [99, 100, 101],
    ...     'close': [104, 105, 106],
    ...     'volume': [1000, 1100, 1200]
    ... })
    >>> chart = plot_ohlc(df)
    """
    if not HAS_LWCHARTS:
        print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
        return None
    
    if not HAS_PANDAS:
        raise ImportError("pandas is required for visualization")
    
    # Create the chart
    chart = lwcharts.Chart(width=width, height=height)
    
    if title:
        chart.title(title)
    
    # Add candlestick series
    candlestick = chart.candlestick_series()
    ohlcv_data = _prepare_ohlcv_data(df)
    candlestick.set_data(ohlcv_data)
    
    # Add volume if requested and available
    if volume:
        # Check for volume column
        vol_col = None
        for col in df.columns:
            if col.lower() in ['volume', 'vol', 'v']:
                vol_col = col
                break
        
        if vol_col is not None:
            volume_series = chart.histogram_series(
                pane=1,
                color='rgba(38, 166, 154, 0.5)'
            )
            
            volume_data = []
            for idx, row in df.iterrows():
                # Determine color based on candle direction
                open_col = None
                close_col = None
                for col in df.columns:
                    if col.lower() in ['open', 'o']:
                        open_col = col
                    elif col.lower() in ['close', 'c']:
                        close_col = col
                
                color = 'rgba(38, 166, 154, 0.5)'  # green
                if open_col and close_col and row[close_col] < row[open_col]:
                    color = 'rgba(239, 83, 80, 0.5)'  # red
                
                volume_data.append({
                    'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
                    'value': float(row[vol_col]),
                    'color': color
                })
            
            volume_series.set_data(volume_data)
    
    return chart


def plot_line(
    df: Any,
    column: str,
    color: str = 'blue',
    width: int = 1,
    height: int = 400,
    chart_width: int = 800,
    title: Optional[str] = None
) -> Any:
    """
    Plot a line chart from DataFrame column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with data
    column : str
        Column name to plot
    color : str, optional
        Line color (default: 'blue')
    width : int, optional
        Line width (default: 1)
    height : int, optional
        Chart height in pixels (default: 400)
    chart_width : int, optional
        Chart width in pixels (default: 800)
    title : str, optional
        Chart title
    
    Returns
    -------
    Chart object or None if lwcharts is not available
    
    Examples
    --------
    >>> import pandas as pd
    >>> from numta.viz import plot_line
    >>> df = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
    >>> chart = plot_line(df, 'close')
    """
    if not HAS_LWCHARTS:
        print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
        return None
    
    if not HAS_PANDAS:
        raise ImportError("pandas is required for visualization")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Create the chart
    chart = lwcharts.Chart(width=chart_width, height=height)
    
    if title:
        chart.title(title)
    
    # Add line series
    line_series = chart.line_series(color=color, width=width)
    
    line_data = []
    for idx, val in df[column].items():
        if not np.isnan(val):
            line_data.append({
                'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
                'value': float(val)
            })
    
    line_series.set_data(line_data)
    
    return chart


def plot_with_indicators(
    df: Any,
    indicators: Dict[str, Any],
    volume: bool = True,
    height: int = 400,
    width: int = 800,
    title: Optional[str] = None
) -> Any:
    """
    Plot OHLCV chart with indicator overlays.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    indicators : dict
        Dictionary of indicator data to overlay. Keys are names, values are
        either Series/arrays or dicts with 'data' and optional 'color', 'width'.
        
        Examples:
        - {'SMA_20': sma_series}
        - {'EMA_10': {'data': ema_series, 'color': 'orange', 'width': 2}}
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
    >>> from numta.viz import plot_with_indicators
    >>> chart = plot_with_indicators(
    ...     df,
    ...     indicators={
    ...         'SMA_20': df.ta.sma(20),
    ...         'EMA_10': {'data': df.ta.ema(10), 'color': 'orange'}
    ...     }
    ... )
    """
    if not HAS_LWCHARTS:
        print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
        return None
    
    # Start with base OHLC chart
    chart = plot_ohlc(df, volume=volume, height=height, width=width, title=title)
    
    if chart is None:
        return None
    
    # Default colors for indicators
    colors = ['blue', 'orange', 'green', 'purple', 'red', 'cyan', 'magenta', 'yellow']
    color_idx = 0
    
    for name, indicator in indicators.items():
        if isinstance(indicator, dict):
            data = indicator.get('data')
            color = indicator.get('color', colors[color_idx % len(colors)])
            line_width = indicator.get('width', 1)
        else:
            data = indicator
            color = colors[color_idx % len(colors)]
            line_width = 1
        
        line_series = chart.line_series(color=color, width=line_width)
        
        line_data = []
        if HAS_PANDAS and hasattr(data, 'items'):
            # pandas Series
            for idx, val in data.items():
                if not np.isnan(val):
                    line_data.append({
                        'time': str(idx) if hasattr(idx, 'strftime') else int(idx),
                        'value': float(val)
                    })
        else:
            # numpy array or list
            for i, val in enumerate(data):
                if not np.isnan(val):
                    line_data.append({
                        'time': i,
                        'value': float(val)
                    })
        
        line_series.set_data(line_data)
        color_idx += 1
    
    return chart
