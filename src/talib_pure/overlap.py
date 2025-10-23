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


@jit(nopython=True, cache=True)
def _bbands_numba(close: np.ndarray, timeperiod: int, nbdevup: float, nbdevdn: float,
                  upperband: np.ndarray, middleband: np.ndarray, lowerband: np.ndarray) -> None:
    """
    Numba-compiled Bollinger Bands calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output arrays in-place.

    Formula:
    Middle Band = SMA(close, timeperiod)
    Upper Band = Middle Band + (nbdevup * StdDev)
    Lower Band = Middle Band - (nbdevdn * StdDev)
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        upperband[i] = np.nan
        middleband[i] = np.nan
        lowerband[i] = np.nan

    # Calculate first SMA value
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    sma = sum_val / timeperiod

    # Calculate first standard deviation
    variance = 0.0
    for i in range(timeperiod):
        diff = close[i] - sma
        variance += diff * diff
    stddev = np.sqrt(variance / timeperiod)

    # Set first values
    middleband[timeperiod - 1] = sma
    upperband[timeperiod - 1] = sma + nbdevup * stddev
    lowerband[timeperiod - 1] = sma - nbdevdn * stddev

    # Calculate remaining values using rolling window
    for i in range(timeperiod, n):
        # Update SMA (rolling window)
        sum_val = sum_val - close[i - timeperiod] + close[i]
        sma = sum_val / timeperiod

        # Calculate standard deviation for current window
        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = close[j] - sma
            variance += diff * diff
        stddev = np.sqrt(variance / timeperiod)

        # Calculate bands
        middleband[i] = sma
        upperband[i] = sma + nbdevup * stddev
        lowerband[i] = sma - nbdevdn * stddev


def BBANDS(close: Union[np.ndarray, list],
           timeperiod: int = 5,
           nbdevup: float = 2.0,
           nbdevdn: float = 2.0,
           matype: int = 0) -> tuple:
    """
    Bollinger Bands (BBANDS)

    Bollinger Bands are a volatility indicator that consists of three lines:
    a middle band (SMA), an upper band, and a lower band. The upper and lower
    bands are typically set 2 standard deviations away from the middle band.

    Developed by John Bollinger, these bands expand and contract based on
    market volatility. They are widely used for identifying overbought/oversold
    conditions and potential breakouts.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 5)
    nbdevup : float, optional
        Number of standard deviations for upper band (default: 2.0)
    nbdevdn : float, optional
        Number of standard deviations for lower band (default: 2.0)
    matype : int, optional
        Moving average type: 0 = SMA (default). Note: Only SMA is currently supported.

    Returns
    -------
    tuple of np.ndarray
        (upperband, middleband, lowerband) - Three arrays with the band values

    Notes
    -----
    - Compatible with TA-Lib BBANDS signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Currently only supports SMA (matype=0)
    - Bands widen during high volatility and narrow during low volatility

    Formula
    -------
    Middle Band = SMA(close, timeperiod)
    Upper Band = Middle Band + (nbdevup × StdDev)
    Lower Band = Middle Band - (nbdevdn × StdDev)

    Where StdDev is the population standard deviation over the timeperiod.

    Lookback period: timeperiod - 1
    (For timeperiod=20, lookback=19)

    Interpretation:
    - Price touching upper band: Potential overbought condition
    - Price touching lower band: Potential oversold condition
    - Band squeeze (narrow bands): Low volatility, potential breakout coming
    - Band expansion (wide bands): High volatility, trend in progress
    - Price breaking above upper band: Strong uptrend
    - Price breaking below lower band: Strong downtrend
    - Middle band acts as dynamic support/resistance

    Common Trading Strategies:
    - Bollinger Bounce: Buy at lower band, sell at upper band (ranging markets)
    - Bollinger Squeeze: Trade breakouts after period of low volatility
    - Walking the Bands: In strong trends, price "walks" along one band
    - %b Indicator: (Price - Lower Band) / (Upper Band - Lower Band)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import BBANDS
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> upper, middle, lower = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2)
    >>> print(upper)
    >>> print(middle)
    >>> print(lower)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    if nbdevup < 0:
        raise ValueError("nbdevup must be >= 0")
    if nbdevdn < 0:
        raise ValueError("nbdevdn must be >= 0")
    if matype != 0:
        raise ValueError("Only matype=0 (SMA) is currently supported")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # Not enough data points - return all NaN
    if n < timeperiod:
        nans = np.full(n, np.nan, dtype=np.float64)
        return nans, nans, nans

    # Pre-allocate output arrays and run Numba-optimized calculation
    upperband = np.empty(n, dtype=np.float64)
    middleband = np.empty(n, dtype=np.float64)
    lowerband = np.empty(n, dtype=np.float64)

    _bbands_numba(close, timeperiod, nbdevup, nbdevdn, upperband, middleband, lowerband)

    return upperband, middleband, lowerband
