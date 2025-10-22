"""
Volume Indicators - Indicators based on volume data
"""

import numpy as np
from typing import Union
from numba import jit


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


def AD(high: Union[np.ndarray, list],
       low: Union[np.ndarray, list],
       close: Union[np.ndarray, list],
       volume: Union[np.ndarray, list]) -> np.ndarray:
    """
    Chaikin A/D Line (Accumulation/Distribution Line)

    The Chaikin Accumulation/Distribution Line is a volume-based indicator
    designed to measure the cumulative flow of money into and out of a security.
    It relates the closing price to the high-low range and multiplies this by volume.

    The indicator attempts to determine whether a security is being accumulated
    (bought) or distributed (sold) by comparing the close price position within
    the period's range, weighted by volume.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    volume : array-like
        Volume array

    Returns
    -------
    np.ndarray
        Array of Chaikin A/D Line values

    Notes
    -----
    - Compatible with TA-Lib AD signature
    - Uses Numba JIT compilation for maximum performance
    - When high == low, the money flow multiplier is 0 (no change to AD line)

    Formula
    -------
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
                          = (2 * Close - High - Low) / (High - Low)
    Money Flow Volume = Money Flow Multiplier * Volume
    AD Line[i] = AD Line[i-1] + Money Flow Volume[i]

    The multiplier ranges from -1 to +1:
    - +1: Close = High (strong buying pressure)
    - 0: Close = Mid-point of range (neutral)
    - -1: Close = Low (strong selling pressure)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import AD
    >>> high = np.array([10, 11, 12, 11, 13])
    >>> low = np.array([9, 10, 10, 9, 11])
    >>> close = np.array([9.5, 10.5, 11, 10, 12])
    >>> volume = np.array([1000, 1100, 1200, 900, 1300])
    >>> ad = AD(high, low, close, volume)
    >>> print(ad)
    """
    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    # Validate input lengths
    n = len(high)
    if len(low) != n or len(close) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _ad_numba(high, low, close, volume, output)

    return output