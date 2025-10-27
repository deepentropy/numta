"""
Math Operators - Mathematical operations on price data

This module implements mathematical operations compatible with TA-Lib.
"""

import numpy as np
from typing import Union

# Import CPU implementations
from ..cpu.math_operators import (
    _max_numba,
    _maxindex_numba,
    _min_numba,
    _minindex_numba,
    _minmax_numba,
    _minmaxindex_numba,
    _sum_numba,
)


def MAX(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Highest Value Over a Specified Period (MAX)

    MAX finds the highest (maximum) value in a rolling window over the
    specified time period. This is useful for identifying recent peaks,
    resistance levels, and price extremes.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices, but can be any series)
    timeperiod : int, optional
        Number of periods for the rolling maximum (default: 30)

    Returns
    -------
    np.ndarray
        Array of maximum values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MAX signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Returns the highest value in each rolling window

    Formula
    -------
    For each position i:
    MAX[i] = maximum(data[i-timeperiod+1 : i+1])

    Lookback period: timeperiod - 1
    (For timeperiod=30, lookback=29)

    Interpretation:
    - Shows recent highest value
    - Can act as resistance level
    - Distance from MAX indicates potential upside
    - Breaking above MAX signals potential breakout
    - Declining MAX suggests weakening highs

    Common Uses:
    - Identify resistance levels
    - Calculate trading channels (MAX + MIN)
    - Measure price deviation from highs
    - Breakout detection
    - Risk management (stop loss placement)
    - Relative strength analysis

    Trading Applications:
    - Channel trading: Trade between MAX and MIN
    - Breakout: Enter when price exceeds MAX
    - Overbought: When price approaches MAX
    - Stop loss: Place below MAX for shorts
    - Support/Resistance: MAX acts as resistance

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MAX
    >>> close = np.array([100, 105, 103, 108, 106, 110, 107, 109, 104, 111])
    >>> max_val = MAX(close, timeperiod=5)
    >>> print(max_val)

    See Also
    --------
    MIN : Lowest value over a specified period
    MAXINDEX : Index of highest value
    MINMAX : Lowest and highest values
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _max_numba(data, timeperiod, output)

    return output


def MAXINDEX(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Index of Highest Value Over a Specified Period (MAXINDEX)

    MAXINDEX returns the number of periods ago when the highest value occurred
    within the rolling window. A value of 0 means the highest value is at the
    current bar, while larger values indicate the high was further in the past.

    This indicator is useful for identifying how recent the peak value is and
    whether new highs are being made.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices, but can be any series)
    timeperiod : int, optional
        Number of periods for the rolling window (default: 30)

    Returns
    -------
    np.ndarray
        Array of index values (distance in periods) with NaN for lookback period

    Notes
    -----
    - Compatible with TA-Lib MAXINDEX signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Returns 0 when current bar has the highest value
    - Returns positive integer for bars ago

    Formula
    -------
    For each position i:
    MAXINDEX[i] = i - argmax(data[i-timeperiod+1 : i+1])

    Where argmax returns the position of the maximum value.
    The result is the distance from current bar (0 = current bar).

    Lookback period: timeperiod - 1
    (For timeperiod=30, lookback=29)

    Interpretation:
    - 0: Current bar is the highest (new high)
    - Small values (1-5): Recent high (strong momentum)
    - Medium values (6-15): Moderate momentum
    - Large values (16+): Old high (weakening momentum)
    - Increasing MAXINDEX: Momentum fading
    - Decreasing MAXINDEX: Approaching new high

    Advantages:
    - Quantifies momentum timing
    - Identifies aging highs
    - Objective measure of strength
    - Early warning of weakness
    - Useful for divergence analysis

    Common Uses:
    - Momentum strength measurement
    - Identify stale highs (resistance)
    - Divergence detection
    - Breakout confirmation
    - Trend strength analysis
    - Entry/exit timing

    Trading Applications:
    - Enter long when MAXINDEX crosses below threshold (e.g., 3)
    - Exit when MAXINDEX rises above threshold (e.g., 10)
    - Confirm breakout: MAXINDEX = 0 after consolidation
    - Divergence: Price high but MAXINDEX increasing
    - Filter trades: Only trade when MAXINDEX < X

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MAXINDEX
    >>> close = np.array([100, 105, 103, 108, 106, 110, 107, 109, 104, 111])
    >>> maxidx = MAXINDEX(close, timeperiod=5)
    >>> print(maxidx)
    >>> # maxidx[-1] = 0 means current bar (111) is the highest

    See Also
    --------
    MAX : Highest value over a specified period
    MININDEX : Index of lowest value
    AROONOSC : Aroon Oscillator (similar concept)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _maxindex_numba(data, timeperiod, output)

    return output


def MIN(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Lowest Value Over a Specified Period (MIN)

    MIN finds the lowest (minimum) value in a rolling window over the
    specified time period.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods (default: 30)

    Returns
    -------
    np.ndarray
        Array of minimum values

    See Also
    --------
    MAX : Highest value over period
    MININDEX : Index of lowest value
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _min_numba(data, timeperiod, output)
    return output


def MININDEX(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Index of Lowest Value Over a Specified Period (MININDEX)

    MININDEX returns the number of periods ago when the lowest value occurred.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods (default: 30)

    Returns
    -------
    np.ndarray
        Array of index values (distance in periods)

    See Also
    --------
    MIN : Lowest value over period
    MAXINDEX : Index of highest value
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _minindex_numba(data, timeperiod, output)
    return output


def MINMAX(data: Union[np.ndarray, list], timeperiod: int = 30) -> tuple:
    """
    Lowest and Highest Values Over a Specified Period (MINMAX)

    MINMAX returns both the minimum and maximum values in a rolling window.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods (default: 30)

    Returns
    -------
    tuple of np.ndarray
        (min, max) - Two arrays with minimum and maximum values

    See Also
    --------
    MIN : Lowest value over period
    MAX : Highest value over period
    MINMAXINDEX : Indexes of lowest and highest values
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty
    if n < timeperiod:
        nans = np.full(n, np.nan, dtype=np.float64)
        return nans, nans
    min_out = np.empty(n, dtype=np.float64)
    max_out = np.empty(n, dtype=np.float64)
    _minmax_numba(data, timeperiod, min_out, max_out)
    return min_out, max_out


def MINMAXINDEX(data: Union[np.ndarray, list], timeperiod: int = 30) -> tuple:
    """
    Indexes of Lowest and Highest Values Over a Specified Period (MINMAXINDEX)

    MINMAXINDEX returns the number of periods ago when the lowest and highest
    values occurred.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods (default: 30)

    Returns
    -------
    tuple of np.ndarray
        (minidx, maxidx) - Two arrays with index values (distance in periods)

    See Also
    --------
    MININDEX : Index of lowest value
    MAXINDEX : Index of highest value
    MINMAX : Lowest and highest values
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty
    if n < timeperiod:
        nans = np.full(n, np.nan, dtype=np.float64)
        return nans, nans
    min_idx_out = np.empty(n, dtype=np.float64)
    max_idx_out = np.empty(n, dtype=np.float64)
    _minmaxindex_numba(data, timeperiod, min_idx_out, max_idx_out)
    return min_idx_out, max_idx_out


def SUM(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Summation (SUM)

    SUM calculates the sum of values over a specified period.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods for summation (default: 30)

    Returns
    -------
    np.ndarray
        Array of sum values

    See Also
    --------
    SMA : Simple Moving Average
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _sum_numba(data, timeperiod, output)
    return output
