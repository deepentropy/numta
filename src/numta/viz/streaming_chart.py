"""
Streaming chart for real-time data visualization.

This module provides a StreamingChart class that accumulates streaming data
and displays it in a Jupyter notebook using lwcharts.
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


class StreamingChart:
    """
    Chart that accumulates streaming OHLCV data.
    
    StreamingChart provides a buffer for streaming data that can be
    visualized in Jupyter notebooks using lwcharts.
    
    Parameters
    ----------
    max_points : int, optional
        Maximum number of points to keep in buffer (default: 1000)
    height : int, optional
        Chart height in pixels (default: 400)
    width : int, optional
        Chart width in pixels (default: 800)
    title : str, optional
        Chart title
    
    Examples
    --------
    >>> from numta.viz import StreamingChart
    >>> from numta.streaming import StreamingSMA, StreamingRSI
    >>> 
    >>> # Create chart and indicators
    >>> chart = StreamingChart(max_points=100)
    >>> sma = StreamingSMA(timeperiod=20)
    >>> 
    >>> # Add streaming data
    >>> for bar in price_stream:
    ...     chart.add_bar(bar['time'], bar['open'], bar['high'],
    ...                   bar['low'], bar['close'], bar['volume'])
    ...     sma_value = sma.update(bar['close'])
    ...     if sma_value is not None:
    ...         chart.add_indicator('SMA_20', bar['time'], sma_value)
    >>> 
    >>> # Display chart
    >>> chart.show()
    """
    
    def __init__(
        self,
        max_points: int = 1000,
        height: int = 400,
        width: int = 800,
        title: Optional[str] = None
    ):
        self._max_points = max_points
        self._height = height
        self._width = width
        self._title = title
        
        # Data buffers
        self._ohlcv_data: List[Dict] = []
        self._indicators: Dict[str, List[Dict]] = {}
        self._indicator_colors: Dict[str, str] = {}
        
        # Default colors for indicators
        self._default_colors = [
            'blue', 'orange', 'green', 'purple', 
            'red', 'cyan', 'magenta', 'yellow'
        ]
        self._color_idx = 0
    
    @property
    def max_points(self) -> int:
        """Maximum number of data points to keep."""
        return self._max_points
    
    @property
    def num_points(self) -> int:
        """Current number of OHLCV data points."""
        return len(self._ohlcv_data)
    
    def add_bar(
        self,
        time: Union[int, str],
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> None:
        """
        Add a new OHLCV bar.
        
        Parameters
        ----------
        time : int or str
            Bar timestamp (index, timestamp, or datetime string)
        open_ : float
            Open price
        high : float
            High price
        low : float
            Low price
        close : float
            Close price
        volume : float, optional
            Volume (default: 0.0)
        """
        bar = {
            'time': time if isinstance(time, str) else int(time),
            'open': float(open_),
            'high': float(high),
            'low': float(low),
            'close': float(close),
            'volume': float(volume)
        }
        
        self._ohlcv_data.append(bar)
        
        # Trim if exceeds max points
        if len(self._ohlcv_data) > self._max_points:
            self._ohlcv_data = self._ohlcv_data[-self._max_points:]
    
    def update_last_bar(
        self,
        open_: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        close: Optional[float] = None,
        volume: Optional[float] = None
    ) -> None:
        """
        Update the last OHLCV bar (useful for real-time updates).
        
        Parameters
        ----------
        open_ : float, optional
            New open price
        high : float, optional
            New high price
        low : float, optional
            New low price
        close : float, optional
            New close price
        volume : float, optional
            New volume
        """
        if not self._ohlcv_data:
            return
        
        last_bar = self._ohlcv_data[-1]
        
        if open_ is not None:
            last_bar['open'] = float(open_)
        if high is not None:
            last_bar['high'] = float(high)
        if low is not None:
            last_bar['low'] = float(low)
        if close is not None:
            last_bar['close'] = float(close)
        if volume is not None:
            last_bar['volume'] = float(volume)
    
    def add_indicator(
        self,
        name: str,
        time: Union[int, str],
        value: float,
        color: Optional[str] = None
    ) -> None:
        """
        Add an indicator data point.
        
        Parameters
        ----------
        name : str
            Indicator name (e.g., 'SMA_20')
        time : int or str
            Data point timestamp
        value : float
            Indicator value
        color : str, optional
            Line color for this indicator
        """
        if np.isnan(value):
            return
        
        if name not in self._indicators:
            self._indicators[name] = []
            # Assign color
            if color is not None:
                self._indicator_colors[name] = color
            else:
                self._indicator_colors[name] = self._default_colors[
                    self._color_idx % len(self._default_colors)
                ]
                self._color_idx += 1
        
        data_point = {
            'time': time if isinstance(time, str) else int(time),
            'value': float(value)
        }
        
        self._indicators[name].append(data_point)
        
        # Trim if exceeds max points
        if len(self._indicators[name]) > self._max_points:
            self._indicators[name] = self._indicators[name][-self._max_points:]
    
    def clear(self) -> None:
        """Clear all data."""
        self._ohlcv_data = []
        self._indicators = {}
        self._indicator_colors = {}
        self._color_idx = 0
    
    def clear_indicators(self) -> None:
        """Clear only indicator data."""
        self._indicators = {}
        self._indicator_colors = {}
        self._color_idx = 0
    
    def to_dataframe(self) -> Any:
        """
        Convert OHLCV data to pandas DataFrame.
        
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with OHLCV data, or None if pandas not available
        """
        if not HAS_PANDAS:
            return None
        
        if not self._ohlcv_data:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        return pd.DataFrame(self._ohlcv_data)
    
    def show(self, volume: bool = True) -> Any:
        """
        Display the chart.
        
        Parameters
        ----------
        volume : bool, optional
            Whether to show volume bars (default: True)
        
        Returns
        -------
        Chart object or None if lwcharts is not available
        """
        if not HAS_LWCHARTS:
            print("Warning: lwcharts is not installed. Install with: pip install lwcharts")
            return None
        
        if not self._ohlcv_data:
            print("No data to display")
            return None
        
        # Create the chart
        chart = lwcharts.Chart(width=self._width, height=self._height)
        
        if self._title:
            chart.title(self._title)
        
        # Add candlestick series
        candlestick = chart.candlestick_series()
        
        # Prepare OHLCV data (without volume for candlestick)
        candle_data = [
            {
                'time': bar['time'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close']
            }
            for bar in self._ohlcv_data
        ]
        candlestick.set_data(candle_data)
        
        # Add volume if requested
        if volume:
            volume_series = chart.histogram_series(
                pane=1,
                color='rgba(38, 166, 154, 0.5)'
            )
            
            volume_data = [
                {
                    'time': bar['time'],
                    'value': bar['volume'],
                    'color': 'rgba(38, 166, 154, 0.5)' if bar['close'] >= bar['open']
                             else 'rgba(239, 83, 80, 0.5)'
                }
                for bar in self._ohlcv_data
            ]
            volume_series.set_data(volume_data)
        
        # Add indicators
        for name, data in self._indicators.items():
            color = self._indicator_colors.get(name, 'blue')
            line_series = chart.line_series(color=color, width=1)
            line_series.set_data(data)
        
        return chart
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StreamingChart(points={self.num_points}, "
            f"indicators={list(self._indicators.keys())})"
        )
