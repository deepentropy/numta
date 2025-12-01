"""
Streaming Volume Indicators.

Streaming versions of volume indicators that support O(1) updates.
"""

from typing import Optional

from .base import StreamingOHLCVIndicator


class StreamingOBV(StreamingOHLCVIndicator):
    """
    Streaming On Balance Volume.
    
    OBV is a cumulative indicator that adds volume on up days
    and subtracts volume on down days.
    
    Examples
    --------
    >>> obv = StreamingOBV()
    >>> obv.update_bar(100, 105, 95, 102, 1000)
    >>> obv.update_bar(102, 108, 100, 106, 1500)  # Up day, adds volume
    """
    
    def __init__(self):
        super().__init__(timeperiod=1)
        self._prev_close: Optional[float] = None
        self._value = 0.0  # OBV starts at 0
    
    @property
    def ready(self) -> bool:
        """OBV is always ready after first update."""
        return self._count > 0
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[float]:
        """Update with OHLCV bar and return current OBV."""
        if self._prev_close is None:
            # First bar - OBV starts at 0
            self._value = 0.0
        else:
            if close > self._prev_close:
                self._value += volume
            elif close < self._prev_close:
                self._value -= volume
            # If equal, OBV unchanged
        
        self._prev_close = close
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._value = 0.0
        self._count = 0
        self._prev_close = None


class StreamingAD(StreamingOHLCVIndicator):
    """
    Streaming Accumulation/Distribution Line.
    
    AD Line measures the cumulative flow of money into and out of a security.
    
    Formula:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = MFM * Volume
        AD = Previous AD + MFV
    
    Examples
    --------
    >>> ad = StreamingAD()
    >>> ad.update_bar(100, 105, 95, 102, 1000)
    >>> ad.update_bar(102, 108, 100, 107, 1500)
    """
    
    def __init__(self):
        super().__init__(timeperiod=1)
        self._value = 0.0  # AD starts at 0
    
    @property
    def ready(self) -> bool:
        """AD is always ready after first update."""
        return self._count > 0
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[float]:
        """Update with OHLCV bar and return current AD."""
        if high == low:
            # Avoid division by zero - no price movement
            mfm = 0.0
        else:
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # Accumulate
        self._value += mfv
        self._count += 1
        
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._value = 0.0
        self._count = 0
