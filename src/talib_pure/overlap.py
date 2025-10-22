"""
Overlap Studies - Indicators that overlay price charts
"""

import numpy as np
from typing import Union
from numba import jit


@jit(nopython=True, cache=True)
def _sma_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled SMA calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate first SMA value
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    output[timeperiod - 1] = sum_val / timeperiod

    # Use rolling window for subsequent values
    for i in range(timeperiod, n):
        sum_val = sum_val - close[i - timeperiod] + close[i]
        output[i] = sum_val / timeperiod


def SMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Simple Moving Average (SMA)

    The Simple Moving Average (SMA) is calculated by adding the closing prices
    of the last N periods and dividing by N. This indicator is used to smooth
    price data and identify trends.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 30)

    Returns
    -------
    np.ndarray
        Array of SMA values with NaN for the lookback period

    Notes
    -----
    - The first (timeperiod - 1) values will be NaN
    - Compatible with TA-Lib SMA signature
    - Uses Numba JIT compilation for maximum performance

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import SMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> sma = SMA(close, timeperiod=3)
    >>> print(sma)
    [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]
    """
    # Validate inputs (TA-Lib requires timeperiod >= 2)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2 (TA-Lib requirement)")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _sma_numba(close, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _ema_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled EMA calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    Formula:
    EMA = (Close - EMA_prev) * multiplier + EMA_prev
    where multiplier = 2 / (timeperiod + 1)

    The first EMA value is initialized as SMA of first timeperiod values.
    """
    n = len(close)
    multiplier = 2.0 / (timeperiod + 1)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Initialize first EMA value as SMA
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    ema = sum_val / timeperiod
    output[timeperiod - 1] = ema

    # Calculate EMA for remaining values
    for i in range(timeperiod, n):
        ema = (close[i] - ema) * multiplier + ema
        output[i] = ema


def EMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Exponential Moving Average (EMA)

    The Exponential Moving Average (EMA) is a type of moving average that places
    a greater weight and significance on the most recent data points. The EMA
    responds more quickly to recent price changes than a simple moving average.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 30)

    Returns
    -------
    np.ndarray
        Array of EMA values with NaN for the lookback period

    Notes
    -----
    - The first (timeperiod - 1) values will be NaN
    - Compatible with TA-Lib EMA signature
    - Uses Numba JIT compilation for maximum performance
    - The first EMA value is initialized as the SMA of the first timeperiod values
    - Smoothing factor: 2 / (timeperiod + 1)

    Formula
    -------
    Multiplier = 2 / (timeperiod + 1)
    EMA[0] = SMA(close[0:timeperiod])
    EMA[i] = (Close[i] - EMA[i-1]) * Multiplier + EMA[i-1]

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import EMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ema = EMA(close, timeperiod=3)
    >>> print(ema)
    [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]
    """
    # Validate inputs (TA-Lib requires timeperiod >= 2)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2 (TA-Lib requirement)")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _ema_numba(close, timeperiod, output)

    return output
