"""
Streaming Momentum Indicators.

Streaming versions of momentum indicators that support O(1) updates.
"""

from typing import Optional, Tuple
import numpy as np

from .base import StreamingIndicator, StreamingOHLCVIndicator, CircularBuffer
from .overlap import StreamingEMA, StreamingSMA


class StreamingRSI(StreamingIndicator):
    """
    Streaming Relative Strength Index.
    
    Uses Wilder's smoothing method for O(1) updates.
    
    Parameters
    ----------
    timeperiod : int
        Number of periods for RSI calculation (default: 14)
    
    Examples
    --------
    >>> rsi = StreamingRSI(timeperiod=14)
    >>> for price in prices:
    ...     result = rsi.update(price)
    ...     if result is not None:
    ...         print(f"RSI: {result:.2f}")
    """
    
    def __init__(self, timeperiod: int = 14):
        super().__init__(timeperiod)
        self._prev_value: Optional[float] = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._warmup_gains = []
        self._warmup_losses = []
        self._initialized = False
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current RSI or None."""
        if self._prev_value is None:
            self._prev_value = value
            self._count += 1
            return None
        
        change = value - self._prev_value
        self._prev_value = value
        
        gain = max(0.0, change)
        loss = max(0.0, -change)
        
        if not self._initialized:
            self._warmup_gains.append(gain)
            self._warmup_losses.append(loss)
            
            if len(self._warmup_gains) >= self._timeperiod:
                # Initialize with average of first timeperiod values
                self._avg_gain = sum(self._warmup_gains) / self._timeperiod
                self._avg_loss = sum(self._warmup_losses) / self._timeperiod
                self._initialized = True
                self._warmup_gains = []
                self._warmup_losses = []
                
                # Calculate initial RSI
                if self._avg_loss == 0:
                    self._value = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    self._value = 100.0 - (100.0 / (1.0 + rs))
        else:
            # Wilder's smoothing
            self._avg_gain = (self._avg_gain * (self._timeperiod - 1) + gain) / self._timeperiod
            self._avg_loss = (self._avg_loss * (self._timeperiod - 1) + loss) / self._timeperiod
            
            if self._avg_loss == 0:
                self._value = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                self._value = 100.0 - (100.0 / (1.0 + rs))
        
        self._count += 1
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._value = None
        self._count = 0
        self._prev_value = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._warmup_gains = []
        self._warmup_losses = []
        self._initialized = False


class StreamingMACD:
    """
    Streaming Moving Average Convergence Divergence.
    
    Returns MACD line, signal line, and histogram.
    
    Parameters
    ----------
    fastperiod : int
        Fast EMA period (default: 12)
    slowperiod : int
        Slow EMA period (default: 26)
    signalperiod : int
        Signal line EMA period (default: 9)
    
    Properties
    ----------
    macd : float or None
        MACD line (fast EMA - slow EMA)
    signal : float or None
        Signal line (EMA of MACD)
    histogram : float or None
        MACD - Signal
    ready : bool
        True when all three values are available
    """
    
    def __init__(
        self,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ):
        if fastperiod >= slowperiod:
            raise ValueError("fastperiod must be less than slowperiod")
        
        self._fastperiod = fastperiod
        self._slowperiod = slowperiod
        self._signalperiod = signalperiod
        
        self._fast_ema = StreamingEMA(fastperiod)
        self._slow_ema = StreamingEMA(slowperiod)
        self._signal_ema = StreamingEMA(signalperiod)
        
        self._macd: Optional[float] = None
        self._signal: Optional[float] = None
        self._histogram: Optional[float] = None
    
    @property
    def macd(self) -> Optional[float]:
        return self._macd
    
    @property
    def signal(self) -> Optional[float]:
        return self._signal
    
    @property
    def histogram(self) -> Optional[float]:
        return self._histogram
    
    @property
    def value(self) -> Optional[Tuple[float, float, float]]:
        """Return (macd, signal, histogram) tuple or None."""
        if self.ready:
            return (self._macd, self._signal, self._histogram)
        return None
    
    @property
    def ready(self) -> bool:
        return self._histogram is not None
    
    def update(self, value: float) -> Optional[Tuple[float, float, float]]:
        """
        Update with new value.
        
        Returns
        -------
        tuple or None
            (macd, signal, histogram) or None if not ready
        """
        fast = self._fast_ema.update(value)
        slow = self._slow_ema.update(value)
        
        if fast is not None and slow is not None:
            self._macd = fast - slow
            sig = self._signal_ema.update(self._macd)
            if sig is not None:
                self._signal = sig
                self._histogram = self._macd - self._signal
                return (self._macd, self._signal, self._histogram)
        
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
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema.reset()
        self._macd = None
        self._signal = None
        self._histogram = None


class StreamingSTOCH:
    """
    Streaming Stochastic Oscillator.
    
    Returns slowK and slowD values.
    
    Parameters
    ----------
    fastk_period : int
        Fast %K period (default: 5)
    slowk_period : int
        Slow %K smoothing period (default: 3)
    slowd_period : int
        Slow %D smoothing period (default: 3)
    
    Properties
    ----------
    slowk : float or None
        Slow %K value
    slowd : float or None
        Slow %D value (SMA of slow %K)
    ready : bool
        True when both values are available
    """
    
    def __init__(
        self,
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowd_period: int = 3
    ):
        self._fastk_period = fastk_period
        self._slowk_period = slowk_period
        self._slowd_period = slowd_period
        
        self._high_buffer = CircularBuffer(fastk_period)
        self._low_buffer = CircularBuffer(fastk_period)
        
        self._fastk_sma = StreamingSMA(slowk_period)
        self._slowd_sma = StreamingSMA(slowd_period)
        
        self._slowk: Optional[float] = None
        self._slowd: Optional[float] = None
    
    @property
    def slowk(self) -> Optional[float]:
        return self._slowk
    
    @property
    def slowd(self) -> Optional[float]:
        return self._slowd
    
    @property
    def value(self) -> Optional[Tuple[float, float]]:
        """Return (slowk, slowd) tuple or None."""
        if self.ready:
            return (self._slowk, self._slowd)
        return None
    
    @property
    def ready(self) -> bool:
        return self._slowd is not None
    
    def update_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        """
        Update with OHLCV bar.
        
        Returns
        -------
        tuple or None
            (slowk, slowd) or None if not ready
        """
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if self._high_buffer.full:
            highest = np.max(self._high_buffer.values)
            lowest = np.min(self._low_buffer.values)
            
            if highest - lowest == 0:
                fastk = 50.0  # Neutral when no range
            else:
                fastk = 100.0 * (close - lowest) / (highest - lowest)
            
            slowk = self._fastk_sma.update(fastk)
            if slowk is not None:
                self._slowk = slowk
                slowd = self._slowd_sma.update(slowk)
                if slowd is not None:
                    self._slowd = slowd
                    return (self._slowk, self._slowd)
        
        return None
    
    def update(self, value: float) -> Optional[Tuple[float, float]]:
        """Update with single value (uses value as close, high, and low)."""
        return self.update_bar(value, value, value, value, 0.0)
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._fastk_sma.reset()
        self._slowd_sma.reset()
        self._slowk = None
        self._slowd = None


class StreamingMOM(StreamingIndicator):
    """
    Streaming Momentum.
    
    MOM = Close - Close[n periods ago]
    
    Parameters
    ----------
    timeperiod : int
        Number of periods to look back (default: 10)
    """
    
    def __init__(self, timeperiod: int = 10):
        super().__init__(timeperiod)
        self._buffer = CircularBuffer(timeperiod + 1)
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current momentum or None."""
        self._buffer.append(value)
        self._count += 1
        
        if len(self._buffer) > self._timeperiod:
            self._value = value - self._buffer[0]
        else:
            self._value = None
        
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._buffer.clear()
        self._value = None
        self._count = 0


class StreamingROC(StreamingIndicator):
    """
    Streaming Rate of Change.
    
    ROC = ((Close - Close[n]) / Close[n]) * 100
    
    Parameters
    ----------
    timeperiod : int
        Number of periods to look back (default: 10)
    """
    
    def __init__(self, timeperiod: int = 10):
        super().__init__(timeperiod)
        self._buffer = CircularBuffer(timeperiod + 1)
    
    def update(self, value: float) -> Optional[float]:
        """Update with new value and return current ROC or None."""
        self._buffer.append(value)
        self._count += 1
        
        if len(self._buffer) > self._timeperiod:
            prev_value = self._buffer[0]
            if prev_value != 0:
                self._value = ((value - prev_value) / prev_value) * 100.0
            else:
                self._value = 0.0
        else:
            self._value = None
        
        return self._value
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._buffer.clear()
        self._value = None
        self._count = 0
