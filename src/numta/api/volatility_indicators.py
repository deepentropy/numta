"""
Volatility Indicators - Measures of price volatility

This module implements volatility indicators compatible with TA-Lib.
"""

import numpy as np
from typing import Union

# Import CPU implementations
from ..cpu.volatility_indicators import _natr_numba, _trange_numba


def NATR(high: Union[np.ndarray, list],
         low: Union[np.ndarray, list],
         close: Union[np.ndarray, list],
         timeperiod: int = 14) -> np.ndarray:
    """
    Normalized Average True Range (NATR)

    NATR is a normalized version of the Average True Range (ATR) that expresses
    volatility as a percentage of the closing price. This normalization allows for
    better comparison of volatility across securities with different price levels.

    By expressing ATR as a percentage, NATR provides a standardized measure that
    can be compared across different instruments, timeframes, and price levels.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Period for ATR calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of NATR values (percentage)

    Notes
    -----
    - Compatible with TA-Lib NATR signature
    - Values expressed as percentages (0-100+)
    - Normalizes volatility for cross-security comparison
    - Lookback period same as ATR

    Formula
    -------
    NATR = (ATR / Close) * 100

    Where ATR is the Average True Range over the specified period.

    Interpretation:
    - Higher NATR: Higher volatility relative to price
    - Lower NATR: Lower volatility relative to price
    - Rising NATR: Increasing volatility
    - Falling NATR: Decreasing volatility
    - NATR useful for position sizing
    - Compare NATR across different securities

    Advantages:
    - Normalized for price level
    - Comparable across securities
    - Percentage-based (easier interpretation)
    - Accounts for gaps (uses True Range)
    - Smoothed measure (uses ATR)

    Advantages over ATR:
    - Can compare different securities
    - Not affected by absolute price
    - Better for multi-security analysis
    - Useful for percentage-based stops

    Common Uses:
    - Volatility comparison across securities
    - Position sizing (normalize by volatility)
    - Stop loss placement (percentage-based)
    - Volatility filtering (trade selection)
    - Risk management
    - Volatility breakout detection

    Trading Applications:
    - High NATR: Reduce position size (higher risk)
    - Low NATR: Increase position size (lower risk)
    - Stop loss: Place at NATR% below entry
    - Filter: Only trade when NATR in range
    - Breakout: Enter when NATR expands

    Position Sizing Example:
    Risk per trade = Account * 0.02 (2%)
    Position size = Risk / (NATR * Price / 100)

    Comparison with Related Indicators:
    - ATR: Absolute volatility measure
    - NATR: Percentage volatility measure
    - Bollinger Bands: Volatility bands
    - Standard Deviation: Statistical volatility

    Examples
    --------
    >>> import numpy as np
    >>> from numta import NATR
    >>> high = np.array([110, 112, 111, 113, 115, 114, 116, 118])
    >>> low = np.array([100, 102, 101, 103, 105, 104, 106, 108])
    >>> close = np.array([105, 107, 106, 108, 110, 109, 111, 113])
    >>> natr = NATR(high, low, close, timeperiod=14)
    >>> # Values are percentages representing volatility

    See Also
    --------
    ATR : Average True Range
    TRANGE : True Range
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Calculate using Numba implementation
    output = np.empty(n, dtype=np.float64)
    _natr_numba(high, low, close, timeperiod, output)

    return output


def TRANGE(high: Union[np.ndarray, list],
           low: Union[np.ndarray, list],
           close: Union[np.ndarray, list]) -> np.ndarray:
    """
    True Range (TRANGE)

    True Range measures volatility by taking the maximum of:
    - Current high - current low
    - Absolute value of current high - previous close
    - Absolute value of current low - previous close

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array

    Returns
    -------
    np.ndarray
        Array of true range values

    See Also
    --------
    ATR : Average True Range
    NATR : Normalized Average True Range
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _trange_numba(high, low, close, output)

    return output
