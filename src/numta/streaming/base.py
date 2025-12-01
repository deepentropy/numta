"""
Base class for streaming indicators.

Streaming indicators support O(1) updates with new price data,
suitable for real-time data processing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class CircularBuffer:
    """
    Fixed-size circular buffer for efficient rolling window operations.
    
    Uses O(1) space and O(1) append/access operations.
    """
    
    def __init__(self, size: int):
        """
        Initialize circular buffer.
        
        Parameters
        ----------
        size : int
            Maximum buffer size
        """
        self._size = size
        self._buffer = np.zeros(size, dtype=np.float64)
        self._index = 0
        self._count = 0
    
    def append(self, value: float) -> None:
        """Add a value to the buffer."""
        self._buffer[self._index] = value
        self._index = (self._index + 1) % self._size
        if self._count < self._size:
            self._count += 1
    
    def __getitem__(self, idx: int) -> float:
        """Get value at relative index (0 = oldest, -1 = newest)."""
        if idx < 0:
            idx = self._count + idx
        if idx < 0 or idx >= self._count:
            raise IndexError("Index out of range")
        actual_idx = (self._index - self._count + idx) % self._size
        return self._buffer[actual_idx]
    
    def __len__(self) -> int:
        """Return current number of elements."""
        return self._count
    
    @property
    def full(self) -> bool:
        """Return True if buffer is full."""
        return self._count == self._size
    
    @property
    def sum(self) -> float:
        """Return sum of all elements in buffer."""
        if self._count == 0:
            return 0.0
        if self._count == self._size:
            return np.sum(self._buffer)
        # Only sum the valid portion
        if self._index >= self._count:
            return np.sum(self._buffer[self._index - self._count:self._index])
        else:
            return np.sum(self._buffer[:self._index]) + np.sum(self._buffer[self._size - self._count + self._index:])
    
    @property
    def values(self) -> np.ndarray:
        """Return values in order from oldest to newest."""
        if self._count == 0:
            return np.array([], dtype=np.float64)
        if self._count == self._size:
            # Full buffer - reorder from start index
            start = self._index
            return np.concatenate([self._buffer[start:], self._buffer[:start]])
        else:
            # Not full yet - return from start
            return self._buffer[:self._count].copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.fill(0.0)
        self._index = 0
        self._count = 0


class StreamingIndicator(ABC):
    """
    Abstract base class for streaming indicators.
    
    Streaming indicators support O(1) updates with new price data,
    making them suitable for real-time processing.
    
    Attributes
    ----------
    timeperiod : int
        Lookback period for the indicator
    
    Properties
    ----------
    value : float or None
        Current indicator value (None if not ready)
    ready : bool
        True when enough data has been accumulated
    """
    
    def __init__(self, timeperiod: int):
        """
        Initialize streaming indicator.
        
        Parameters
        ----------
        timeperiod : int
            Lookback period for the indicator
        """
        if timeperiod < 1:
            raise ValueError("timeperiod must be >= 1")
        self._timeperiod = timeperiod
        self._value: Optional[float] = None
        self._count = 0
    
    @property
    def timeperiod(self) -> int:
        """Get the timeperiod."""
        return self._timeperiod
    
    @property
    def value(self) -> Optional[float]:
        """Get current indicator value or None if not ready."""
        return self._value
    
    @property
    def ready(self) -> bool:
        """Return True when enough data has been accumulated."""
        return self._value is not None
    
    @abstractmethod
    def update(self, value: float) -> Optional[float]:
        """
        Update indicator with a new price value.
        
        Parameters
        ----------
        value : float
            New price value (typically close price)
        
        Returns
        -------
        float or None
            Current indicator value, or None if not enough data
        """
        pass
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[float]:
        """
        Update indicator with an OHLCV bar.
        
        By default, uses the close price. Subclasses can override
        for indicators that require OHLCV data.
        
        Parameters
        ----------
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
        
        Returns
        -------
        float or None
            Current indicator value, or None if not enough data
        """
        return self.update(close)
    
    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state to initial values."""
        pass


class StreamingOHLCVIndicator(StreamingIndicator):
    """
    Base class for streaming indicators that require OHLCV data.
    
    These indicators need high, low, close, and optionally volume data
    for each bar.
    """
    
    @abstractmethod
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[float]:
        """
        Update indicator with an OHLCV bar.
        
        Parameters
        ----------
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
        
        Returns
        -------
        float or None
            Current indicator value, or None if not enough data
        """
        pass
    
    def update(self, value: float) -> Optional[float]:
        """
        Update with a single value (treated as close price with same OHLC).
        
        For OHLCV indicators, this is a convenience method that
        sets open = high = low = close = value.
        """
        return self.update_bar(value, value, value, value, 0.0)
