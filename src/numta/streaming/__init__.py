"""
Streaming indicators for real-time data processing.

This module provides streaming versions of technical analysis indicators
that support O(1) updates with new price data.

Examples
--------
>>> from numta.streaming import StreamingSMA, StreamingRSI
>>> 
>>> # Create streaming indicators
>>> sma = StreamingSMA(timeperiod=20)
>>> rsi = StreamingRSI(timeperiod=14)
>>> 
>>> # Update with streaming data
>>> for price in price_stream:
...     sma_value = sma.update(price)
...     rsi_value = rsi.update(price)
...     if sma_value is not None and rsi_value is not None:
...         print(f"SMA: {sma_value:.2f}, RSI: {rsi_value:.2f}")
"""

from .base import (
    StreamingIndicator,
    StreamingOHLCVIndicator,
    CircularBuffer,
)

from .overlap import (
    StreamingSMA,
    StreamingEMA,
    StreamingBBANDS,
    StreamingDEMA,
    StreamingTEMA,
    StreamingWMA,
)

from .momentum import (
    StreamingRSI,
    StreamingMACD,
    StreamingSTOCH,
    StreamingMOM,
    StreamingROC,
)

from .volatility import (
    StreamingATR,
    StreamingTRANGE,
)

from .volume import (
    StreamingOBV,
    StreamingAD,
)


__all__ = [
    # Base classes
    'StreamingIndicator',
    'StreamingOHLCVIndicator',
    'CircularBuffer',
    # Overlap
    'StreamingSMA',
    'StreamingEMA',
    'StreamingBBANDS',
    'StreamingDEMA',
    'StreamingTEMA',
    'StreamingWMA',
    # Momentum
    'StreamingRSI',
    'StreamingMACD',
    'StreamingSTOCH',
    'StreamingMOM',
    'StreamingROC',
    # Volatility
    'StreamingATR',
    'StreamingTRANGE',
    # Volume
    'StreamingOBV',
    'StreamingAD',
]
