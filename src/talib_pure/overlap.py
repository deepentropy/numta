"""
Overlap Studies - Indicators that overlay price charts
"""

import numpy as np
from typing import Union


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

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import SMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> sma = SMA(close, timeperiod=3)
    >>> print(sma)
    [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]
    """
    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    if len(close) == 0:
        return np.array([])

    # Initialize output with NaN
    output = np.full(len(close), np.nan, dtype=np.float64)

    # Not enough data points
    if len(close) < timeperiod:
        return output

    # Calculate SMA using convolution for better performance
    # This is much faster than a rolling window approach for large arrays
    weights = np.ones(timeperiod) / timeperiod
    sma_values = np.convolve(close, weights, mode='valid')

    # Place the SMA values in the output array
    output[timeperiod - 1:] = sma_values

    return output
