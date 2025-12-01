"""
Streaming Overlap Indicators.

Streaming versions of overlap indicators that support O(1) updates.
"""

from typing import Optional, Tuple
import numpy as np

from .base import StreamingIndicator, CircularBuffer


class StreamingSMA(StreamingIndicator):
    """
    Streaming Simple Moving Average.
    
    Uses a running sum for O(1) updates.
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for the moving average (default: 30)
    
    Examples
    --------
    >>> sma = StreamingSMA(timeperiod=3)
    >>> sma.update(1.0)  # Not ready yet
    >>> sma.update(2.0)  # Not ready yet  
    >>> sma.update(3.0)  # Returns 2.0
    2.0
    >>> sma.update(4.0)  # Returns 3.0
    3.0
    """
    
    def __init__(self, timeperiod: int = 30):
        super().__init__(timeperiod)
        self._buffer = CircularBuffer(timeperiod)
        self._running_sum = 0.0
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current SMA or None."""
        if self._buffer.full:
            # Remove oldest value from sum
            self._running_sum -= self._buffer[0]
        
        self._buffer.append(value)
        self._running_sum += value
        self._count += 1
        
        if self._buffer.full:
            self._value = self._running_sum / self._timeperiod
        else:
            self._value = None
        
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._buffer.clear()
        self._running_sum = 0.0
        self._value = None
        self._count = 0


class StreamingEMA(StreamingIndicator):
    """
    Streaming Exponential Moving Average.
    
    Uses standard EMA formula for O(1) updates.
    Initial value is seeded with SMA of first timeperiod values.
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for the moving average (default: 30)
    
    Examples
    --------
    >>> ema = StreamingEMA(timeperiod=3)
    >>> for price in [1.0, 2.0, 3.0, 4.0, 5.0]:
    ...     result = ema.update(price)
    ...     if result is not None:
    ...         print(f"{result:.4f}")
    """
    
    def __init__(self, timeperiod: int = 30):
        super().__init__(timeperiod)
        self._multiplier = 2.0 / (timeperiod + 1)
        self._warmup_sum = 0.0
        self._warmup_count = 0
        self._initialized = False
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current EMA or None."""
        if not self._initialized:
            self._warmup_sum += value
            self._warmup_count += 1
            
            if self._warmup_count >= self._timeperiod:
                # Initialize with SMA
                self._value = self._warmup_sum / self._timeperiod
                self._initialized = True
        else:
            # EMA update
            self._value = (value - self._value) * self._multiplier + self._value
        
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._value = None
        self._count = 0
        self._warmup_sum = 0.0
        self._warmup_count = 0
        self._initialized = False


class StreamingBBANDS:
    """
    Streaming Bollinger Bands.
    
    Returns upper, middle, lower bands using O(n) computation per update
    where n is the timeperiod. For truly O(1) variance updates, a more
    complex algorithm would be needed.
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for the moving average (default: 5)
    nbdevup : float
        Number of standard deviations for upper band (default: 2.0)
    nbdevdn : float
        Number of standard deviations for lower band (default: 2.0)
    
    Properties
    ----------
    upper : float or None
        Upper band value
    middle : float or None
        Middle band value (SMA)
    lower : float or None
        Lower band value
    ready : bool
        True when enough data has been accumulated
    """
    
    def __init__(
        self,
        timeperiod: int = 5,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0
    ):
        if timeperiod < 2:
            raise ValueError("timeperiod must be >= 2")
        self._timeperiod = timeperiod
        self._nbdevup = nbdevup
        self._nbdevdn = nbdevdn
        self._buffer = CircularBuffer(timeperiod)
        self._running_sum = 0.0
        self._upper: Optional[float] = None
        self._middle: Optional[float] = None
        self._lower: Optional[float] = None
    
    @property
    def timeperiod(self) -> int:
        return self._timeperiod
    
    @property
    def upper(self) -> Optional[float]:
        return self._upper
    
    @property
    def middle(self) -> Optional[float]:
        return self._middle
    
    @property
    def lower(self) -> Optional[float]:
        return self._lower
    
    @property
    def value(self) -> Optional[Tuple[float, float, float]]:
        """Return (upper, middle, lower) tuple or None."""
        if self.ready:
            return (self._upper, self._middle, self._lower)
        return None
    
    @property
    def ready(self) -> bool:
        return self._middle is not None
    
    def update(self, value: float) -> Optional[Tuple[float, float, float]]:
        """
        Update with new value and return bands.
        
        Returns
        -------
        tuple or None
            (upper, middle, lower) bands or None if not ready
        """
        if self._buffer.full:
            self._running_sum -= self._buffer[0]
        
        self._buffer.append(value)
        self._running_sum += value
        
        if self._buffer.full:
            # Calculate SMA
            self._middle = self._running_sum / self._timeperiod
            
            # Calculate standard deviation
            values = self._buffer.values
            variance = np.sum((values - self._middle) ** 2) / self._timeperiod
            std = np.sqrt(variance)
            
            self._upper = self._middle + self._nbdevup * std
            self._lower = self._middle - self._nbdevdn * std
            
            return (self._upper, self._middle, self._lower)
        
        return None
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[Tuple[float, float, float]]:
        """Update with OHLCV bar (uses close price)."""
        return self.update(close)
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._buffer.clear()
        self._running_sum = 0.0
        self._upper = None
        self._middle = None
        self._lower = None


class StreamingDEMA(StreamingIndicator):
    """
    Streaming Double Exponential Moving Average.
    
    DEMA = 2 * EMA - EMA(EMA)
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for the calculation (default: 30)
    """
    
    def __init__(self, timeperiod: int = 30):
        super().__init__(timeperiod)
        self._ema1 = StreamingEMA(timeperiod)
        self._ema2 = StreamingEMA(timeperiod)
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current DEMA or None."""
        ema1_val = self._ema1.update(value)
        
        if ema1_val is not None:
            ema2_val = self._ema2.update(ema1_val)
            if ema2_val is not None:
                self._value = 2.0 * ema1_val - ema2_val
            else:
                self._value = None
        else:
            self._value = None
        
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._ema1.reset()
        self._ema2.reset()
        self._value = None
        self._count = 0


class StreamingTEMA(StreamingIndicator):
    """
    Streaming Triple Exponential Moving Average.
    
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for the calculation (default: 30)
    """
    
    def __init__(self, timeperiod: int = 30):
        super().__init__(timeperiod)
        self._ema1 = StreamingEMA(timeperiod)
        self._ema2 = StreamingEMA(timeperiod)
        self._ema3 = StreamingEMA(timeperiod)
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current TEMA or None."""
        ema1_val = self._ema1.update(value)
        
        if ema1_val is not None:
            ema2_val = self._ema2.update(ema1_val)
            if ema2_val is not None:
                ema3_val = self._ema3.update(ema2_val)
                if ema3_val is not None:
                    self._value = 3.0 * ema1_val - 3.0 * ema2_val + ema3_val
                else:
                    self._value = None
            else:
                self._value = None
        else:
            self._value = None
        
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._ema1.reset()
        self._ema2.reset()
        self._ema3.reset()
        self._value = None
        self._count = 0


class StreamingWMA(StreamingIndicator):
    """
    Streaming Weighted Moving Average.
    
    WMA assigns linearly decreasing weights to older data.
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for the moving average (default: 30)
    """
    
    def __init__(self, timeperiod: int = 30):
        super().__init__(timeperiod)
        self._buffer = CircularBuffer(timeperiod)
        # Pre-compute weights: [1, 2, 3, ..., n]
        self._weights = np.arange(1, timeperiod + 1, dtype=np.float64)
        self._weight_sum = np.sum(self._weights)
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current WMA or None."""
        self._buffer.append(value)
        self._count += 1
        
        if self._buffer.full:
            values = self._buffer.values
            self._value = np.sum(values * self._weights) / self._weight_sum
        else:
            self._value = None
        
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._buffer.clear()
        self._value = None
        self._count = 0
