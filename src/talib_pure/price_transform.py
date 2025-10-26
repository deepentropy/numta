"""
Price Transform - Price transformation functions

This module implements price transformation functions compatible with TA-Lib.
"""

import numpy as np
from typing import Union
from numba import jit


@jit(nopython=True, cache=True)
def _medprice_numba(high: np.ndarray, low: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled MEDPRICE calculation (in-place)

    Formula: MEDPRICE = (High + Low) / 2
    """
    n = len(high)
    for i in range(n):
        output[i] = (high[i] + low[i]) / 2.0


def MEDPRICE(high: Union[np.ndarray, list], low: Union[np.ndarray, list]) -> np.ndarray:
    """
    Median Price (MEDPRICE)

    MEDPRICE calculates the median price for each period as the average of the
    high and low prices. Despite its name, it calculates the arithmetic mean,
    not the statistical median. This provides a simple measure of the price
    level for each bar.

    The median price is useful for identifying the midpoint of the price range
    and can be used in various technical analysis calculations.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array

    Returns
    -------
    np.ndarray
        Array of median price values

    Notes
    -----
    - Compatible with TA-Lib MEDPRICE signature
    - Uses Numba JIT compilation for maximum performance
    - No lookback period (calculated for every bar)
    - Despite the name, calculates mean not statistical median
    - All input arrays must have the same length

    Formula
    -------
    MEDPRICE = (High + Low) / 2

    Interpretation:
    - Represents the midpoint of the day's range
    - Used as a simplified price level
    - Can act as pivot point
    - Smoother than using close alone
    - Reduces impact of opening/closing gaps

    Common Uses:
    - Input for other indicators
    - Pivot point calculations
    - Support/resistance identification
    - Price averaging for smoothing
    - Alternative to close prices

    Advantages:
    - Simple and intuitive
    - Reduces noise from close prices
    - Captures full range information
    - Fast to calculate
    - No lookback period required

    Related Prices:
    - Typical Price: (High + Low + Close) / 3
    - Weighted Close: (High + Low + Close * 2) / 4
    - Average Price: (Open + High + Low + Close) / 4

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MEDPRICE
    >>> high = np.array([105, 106, 108, 107, 109])
    >>> low = np.array([100, 101, 103, 102, 104])
    >>> medprice = MEDPRICE(high, low)
    >>> print(medprice)
    [102.5 103.5 105.5 104.5 106.5]

    See Also
    --------
    TYPPRICE : Typical Price
    WCLPRICE : Weighted Close Price
    AVGPRICE : Average Price
    MIDPOINT : MidPoint over period
    """
    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # Check arrays have the same length
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _medprice_numba(high, low, output)

    return output


@jit(nopython=True, cache=True)
def _midpoint_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MIDPOINT calculation (in-place)

    Formula: MIDPOINT = (MAX + MIN) / 2 over timeperiod
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate midpoint for each window
    for i in range(timeperiod - 1, n):
        max_val = data[i]
        min_val = data[i]
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] > max_val:
                max_val = data[j]
            if data[j] < min_val:
                min_val = data[j]
        output[i] = (max_val + min_val) / 2.0


def MIDPOINT(data: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    MidPoint Over Period (MIDPOINT)

    MIDPOINT calculates the midpoint (average) between the highest and lowest
    values over a specified time period. This provides a dynamic center line
    that adapts to the recent price range.

    Unlike MEDPRICE which uses high/low of each individual bar, MIDPOINT
    calculates the middle of the entire range over multiple periods, making
    it a smoothed indicator.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices, but can be any series)
    timeperiod : int, optional
        Number of periods for calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of midpoint values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MIDPOINT signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Can be applied to any data series (close, volume, etc.)

    Formula
    -------
    For each position i:
    MAX = maximum(data[i-timeperiod+1 : i+1])
    MIN = minimum(data[i-timeperiod+1 : i+1])
    MIDPOINT = (MAX + MIN) / 2

    Lookback period: timeperiod - 1
    (For timeperiod=14, lookback=13)

    Interpretation:
    - Acts as dynamic equilibrium level
    - Price above MIDPOINT: Bullish
    - Price below MIDPOINT: Bearish
    - MIDPOINT as support/resistance
    - Distance from MIDPOINT indicates momentum
    - MIDPOINT slope shows trend direction

    Advantages:
    - Adapts to recent price range
    - Provides objective center line
    - Smoother than moving averages
    - Less lag than longer MAs
    - Works in ranging and trending markets

    Common Uses:
    - Mean reversion trading
    - Dynamic support/resistance
    - Trend identification
    - Entry/exit signals
    - Channel midline
    - Deviation measurement

    Trading Applications:
    - Buy when price crosses above MIDPOINT
    - Sell when price crosses below MIDPOINT
    - Mean reversion: Fade extremes back to MIDPOINT
    - Breakout: Enter when price breaks range beyond MIDPOINT
    - Stop loss: Place on opposite side of MIDPOINT

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MIDPOINT
    >>> close = np.array([100, 105, 103, 108, 106, 110, 107, 109, 104, 111,
    ...                   113, 112, 115, 114, 116])
    >>> midpoint = MIDPOINT(close, timeperiod=14)
    >>> print(midpoint)

    See Also
    --------
    MEDPRICE : Median Price (High + Low) / 2
    MAX : Highest value over period
    MIN : Lowest value over period
    BBANDS : Bollinger Bands (similar concept with volatility)
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
    _midpoint_numba(data, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _midprice_numba(high: np.ndarray, low: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MIDPRICE calculation (in-place)

    Formula: MIDPRICE = (MAX(high, period) + MIN(low, period)) / 2
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate midprice for each window
    for i in range(timeperiod - 1, n):
        max_high = high[i]
        min_low = low[i]
        for j in range(i - timeperiod + 1, i + 1):
            if high[j] > max_high:
                max_high = high[j]
            if low[j] < min_low:
                min_low = low[j]
        output[i] = (max_high + min_low) / 2.0


def MIDPRICE(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             timeperiod: int = 14) -> np.ndarray:
    """
    Midpoint Price Over Period (MIDPRICE)

    MIDPRICE calculates the midpoint between the highest high and lowest low
    over a specified time period. This provides a dynamic center line that
    represents the middle of the price range over the lookback window.

    Unlike MEDPRICE which uses high/low of each individual bar, MIDPRICE
    uses the highest high and lowest low over multiple periods.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Number of periods for calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of midprice values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MIDPRICE signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - All input arrays must have the same length

    Formula
    -------
    For each position i:
    Highest High = max(high[i-timeperiod+1 : i+1])
    Lowest Low = min(low[i-timeperiod+1 : i+1])
    MIDPRICE = (Highest High + Lowest Low) / 2

    Lookback period: timeperiod - 1
    (For timeperiod=14, lookback=13)

    Interpretation:
    - Acts as dynamic equilibrium level
    - Price above MIDPRICE: Bullish bias
    - Price below MIDPRICE: Bearish bias
    - MIDPRICE as support/resistance zone
    - Distance from MIDPRICE indicates deviation
    - MIDPRICE slope shows range trend

    Advantages:
    - Incorporates full range information
    - Adapts to recent price extremes
    - Provides objective center line
    - Useful for channel trading
    - Less lag than moving averages

    Common Uses:
    - Mean reversion trading
    - Dynamic support/resistance
    - Channel midline
    - Entry/exit reference point
    - Stop loss placement
    - Trend strength measurement

    Comparison with Similar Indicators:
    - MEDPRICE: (High + Low) / 2 for single bar
    - MIDPOINT: (MAX + MIN) / 2 of close prices
    - MIDPRICE: (MAX(high) + MIN(low)) / 2 over period

    Trading Applications:
    - Buy when price crosses above MIDPRICE
    - Sell when price crosses below MIDPRICE
    - Mean reversion: Trade back to MIDPRICE
    - Channel trading: Use with MAX(high) and MIN(low)
    - Stop loss: Place on opposite side of MIDPRICE

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MIDPRICE
    >>> high = np.array([105, 106, 108, 107, 109, 110, 111, 112, 113, 114,
    ...                  115, 116, 117, 118, 119])
    >>> low = np.array([100, 101, 103, 102, 104, 105, 106, 107, 108, 109,
    ...                 110, 111, 112, 113, 114])
    >>> midprice = MIDPRICE(high, low, timeperiod=14)
    >>> print(midprice)

    See Also
    --------
    MEDPRICE : Median Price (High + Low) / 2 per bar
    MIDPOINT : MidPoint of close over period
    MAX : Highest value over period
    MIN : Lowest value over period
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # Check arrays have the same length
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _midprice_numba(high, low, timeperiod, output)

    return output


def TYPPRICE(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Typical Price (TYPPRICE)

    Typical Price is a simple average of the high, low, and close prices,
    providing a single value that represents the average price for a period.
    Also known as HLC/3.

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
        Array of typical price values

    Notes
    -----
    - Building block for Money Flow Index
    - Smooths price fluctuations
    - Simple arithmetic average
    - No lag or lookback period
    - Compatible with TA-Lib TYPPRICE signature

    Formula
    -------
    TYPPRICE = (High + Low + Close) / 3

    Interpretation:
    - Represents "typical" price for the period
    - More balanced than using close alone
    - Accounts for entire trading range
    - Useful for pivot point calculations
    - Used in volume-weighted indicators

    Applications:
    - Money Flow Index calculation
    - Pivot point calculations
    - Volume-weighted studies
    - Price smoothing
    - Support/resistance levels

    Comparison with Other Price Measures:
    - Close: Last traded price only
    - MEDPRICE: (High + Low) / 2
    - TYPPRICE: (High + Low + Close) / 3
    - WCLPRICE: (High + Low + 2*Close) / 4

    Advantages:
    - Simple and intuitive
    - No parameters required
    - No lag (current period calculation)
    - Smooths price action
    - Accounts for full range

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import TYPPRICE
    >>> high = np.array([110, 112, 111, 113, 115])
    >>> low = np.array([100, 102, 101, 103, 105])
    >>> close = np.array([105, 107, 106, 108, 110])
    >>> typprice = TYPPRICE(high, low, close)
    >>> # Result: [105.0, 107.0, 106.0, 108.0, 110.0]

    See Also
    --------
    MEDPRICE : Median Price (High + Low) / 2
    WCLPRICE : Weighted Close Price
    AVGPRICE : Average Price
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Check arrays have the same length
    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Calculate typical price: (H + L + C) / 3
    output = (high + low + close) / 3.0

    return output


def WCLPRICE(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Weighted Close Price (WCLPRICE)

    Calculates the weighted close price by giving double weight to the closing
    price. This provides a smoothed price measure that emphasizes the close.

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
        Array of weighted close price values

    Notes
    -----
    - Also known as HLC/4 weighted
    - Close price weighted twice
    - No lag or lookback period
    - Compatible with TA-Lib WCLPRICE signature
    - Smooths volatility better than close alone

    Formula
    -------
    WCLPRICE = (High + Low + 2*Close) / 4

    Interpretation:
    - Emphasizes closing price (weighted 2x)
    - Smooths intraday volatility
    - More stable than close alone
    - Accounts for full trading range

    Comparison with Other Price Measures:
    - Close: Last traded price only
    - MEDPRICE: (High + Low) / 2
    - TYPPRICE: (High + Low + Close) / 3
    - WCLPRICE: (High + Low + 2*Close) / 4
    - AVGPRICE: (High + Low + Close + Open) / 4

    Applications:
    - Moving average calculations
    - Trend analysis
    - Support/resistance levels
    - Price smoothing
    - Alternative to using close alone

    Advantages:
    - Simple calculation
    - No parameters needed
    - No lag (current period)
    - Emphasizes close (most important price)
    - Includes full range information

    Trading Usage:
    - Use in place of close for MAs
    - Smoother trend identification
    - Reduced noise vs close
    - Better represents day's action

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import WCLPRICE
    >>> high = np.array([110, 112, 111, 113, 115])
    >>> low = np.array([100, 102, 101, 103, 105])
    >>> close = np.array([105, 107, 106, 108, 110])
    >>> wclprice = WCLPRICE(high, low, close)
    >>> # Result: [103.75, 105.75, 104.75, 106.75, 108.75]

    See Also
    --------
    TYPPRICE : Typical Price (H+L+C)/3
    MEDPRICE : Median Price (H+L)/2
    AVGPRICE : Average Price (H+L+C+O)/4
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Check arrays have the same length
    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Calculate weighted close price: (H + L + 2*C) / 4
    output = (high + low + 2.0 * close) / 4.0

    return output
