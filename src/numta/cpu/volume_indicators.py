"""
Volume Indicators - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for volume indicators.
"""

import numpy as np
from numba import jit


__all__ = [
    "_ad_numba",
    "_adosc_numba",
    "_obv_numba",
]


@jit(nopython=True, cache=True)
def _ad_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              volume: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled Chaikin A/D Line calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    Formula:
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = Money Flow Multiplier * Volume
    AD Line = Cumulative sum of Money Flow Volume
    """
    n = len(high)
    ad_value = 0.0

    for i in range(n):
        # Calculate Money Flow Multiplier
        high_low_diff = high[i] - low[i]

        if high_low_diff == 0.0:
            # Avoid division by zero
            # When high == low, the multiplier is undefined, so we use 0
            mf_multiplier = 0.0
        else:
            mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / high_low_diff

        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume[i]

        # Accumulate AD Line
        ad_value += mf_volume
        output[i] = ad_value


@jit(nopython=True, cache=True)
def _adosc_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray, fastperiod: int, slowperiod: int,
                 output: np.ndarray) -> None:
    """
    Numba-compiled Chaikin A/D Oscillator calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    The oscillator is the difference between the fast EMA and slow EMA
    of the A/D Line.
    """
    n = len(high)

    # First, calculate the A/D Line
    ad_line = np.empty(n, dtype=np.float64)
    ad_value = 0.0

    for i in range(n):
        # Calculate Money Flow Multiplier
        high_low_diff = high[i] - low[i]

        if high_low_diff == 0.0:
            mf_multiplier = 0.0
        else:
            mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / high_low_diff

        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume[i]

        # Accumulate AD Line
        ad_value += mf_volume
        ad_line[i] = ad_value

    # Fill lookback period with NaN
    for i in range(slowperiod - 1):
        output[i] = np.nan

    # Calculate EMAs using arrays to track intermediate values
    fast_ema_array = np.empty(n, dtype=np.float64)
    slow_ema_array = np.empty(n, dtype=np.float64)

    # Fast EMA calculation
    fast_multiplier = 2.0 / (fastperiod + 1)
    for i in range(fastperiod - 1):
        fast_ema_array[i] = np.nan

    # Initialize first fast EMA as SMA
    sum_val = 0.0
    for i in range(fastperiod):
        sum_val += ad_line[i]
    fast_ema = sum_val / fastperiod
    fast_ema_array[fastperiod - 1] = fast_ema

    # Calculate remaining fast EMA values
    for i in range(fastperiod, n):
        fast_ema = (ad_line[i] - fast_ema) * fast_multiplier + fast_ema
        fast_ema_array[i] = fast_ema

    # Slow EMA calculation
    slow_multiplier = 2.0 / (slowperiod + 1)
    for i in range(slowperiod - 1):
        slow_ema_array[i] = np.nan

    # Initialize first slow EMA as SMA
    sum_val = 0.0
    for i in range(slowperiod):
        sum_val += ad_line[i]
    slow_ema = sum_val / slowperiod
    slow_ema_array[slowperiod - 1] = slow_ema

    # Calculate remaining slow EMA values
    for i in range(slowperiod, n):
        slow_ema = (ad_line[i] - slow_ema) * slow_multiplier + slow_ema
        slow_ema_array[i] = slow_ema

    # Calculate oscillator as difference
    for i in range(slowperiod - 1, n):
        output[i] = fast_ema_array[i] - slow_ema_array[i]


@jit(nopython=True, cache=True)
def _obv_numba(close: np.ndarray, volume: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled OBV calculation (in-place)

    Formula:
    If Close > Close[-1]: OBV = OBV[-1] + Volume
    If Close < Close[-1]: OBV = OBV[-1] - Volume
    If Close = Close[-1]: OBV = OBV[-1]
    """
    n = len(close)

    # Start with first volume (TA-Lib convention)
    output[0] = volume[0]

    # Calculate cumulative OBV
    for i in range(1, n):
        if close[i] > close[i - 1]:
            output[i] = output[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            output[i] = output[i - 1] - volume[i]
        else:
            output[i] = output[i - 1]
