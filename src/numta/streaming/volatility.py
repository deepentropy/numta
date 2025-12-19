"""
Streaming Volatility Indicators.

Streaming versions of volatility indicators that support O(1) updates.
"""

from typing import Optional
import numpy as np

from .base import StreamingOHLCVIndicator, CircularBuffer


class StreamingTRANGE(StreamingOHLCVIndicator):
    """
    Streaming True Range.
    
    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    
    Examples
    --------
    >>> tr = StreamingTRANGE()
    >>> tr.update_bar(100, 105, 95, 102)  # First bar - returns high-low
    >>> tr.update_bar(102, 108, 100, 106)  # Subsequent bars use prev close
    """
    
    def __init__(self):
        # TRANGE doesn't have a period parameter
        super().__init__(timeperiod=1)
        self._prev_close: Optional[float] = None
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[float]:
        """Update with OHLCV bar and return true range."""
        if self._prev_close is None:
            # First bar - use high-low range
            self._value = high - low
        else:
            # Calculate true range
            hl = high - low
            hpc = abs(high - self._prev_close)
            lpc = abs(low - self._prev_close)
            self._value = max(hl, hpc, lpc)
        
        self._prev_close = close
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._value = None
        self._count = 0
        self._prev_close = None


class StreamingATR(StreamingOHLCVIndicator):
    """
    Streaming Average True Range.
    
    Uses Wilder's smoothing method for O(1) updates.
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for ATR calculation (default: 14)
    
    Examples
    --------
    >>> atr = StreamingATR(timeperiod=14)
    >>> for bar in ohlcv_data:
    ...     result = atr.update_bar(*bar)
    ...     if result is not None:
    ...         print(f"ATR: {result:.4f}")
    """
    
    def __init__(self, timeperiod: int = 14):
        super().__init__(timeperiod)
        self._trange = StreamingTRANGE()
        self._warmup_sum = 0.0
        self._warmup_count = 0
        self._initialized = False
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[float]:
        """Update with OHLCV bar and return current ATR or None."""
        tr = self._trange.update_bar(open_, high, low, close, volume)
        
        if tr is None:
            return None
        
        if not self._initialized:
            self._warmup_sum += tr
            self._warmup_count += 1
            
            if self._warmup_count >= self._timeperiod:
                # Initialize with average of first timeperiod TRs
                self._value = self._warmup_sum / self._timeperiod
                self._initialized = True
        else:
            # Wilder's smoothing
            self._value = (self._value * (self._timeperiod - 1) + tr) / self._timeperiod
        
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._trange.reset()
        self._value = None
        self._count = 0
        self._warmup_sum = 0.0
        self._warmup_count = 0
        self._initialized = False
