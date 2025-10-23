"""
Momentum Indicators - Indicators that measure the rate of price change
"""

import numpy as np
from typing import Union
from numba import jit


@jit(nopython=True, cache=True)
def _adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ADX calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    ADX Formula:
    1. Calculate True Range (TR), +DM, -DM
    2. Smooth TR, +DM, -DM using Wilder's smoothing
    3. Calculate +DI = 100 * smoothed(+DM) / smoothed(TR)
    4. Calculate -DI = 100 * smoothed(-DM) / smoothed(TR)
    5. Calculate DX = 100 * |+DI - -DI| / (+DI + -DI)
    6. ADX = smoothed(DX) using Wilder's smoothing
    """
    n = len(high)

    # Fill lookback period with NaN (2 * timeperiod - 1)
    lookback = 2 * timeperiod - 1
    for i in range(lookback):
        output[i] = np.nan

    # Step 1: Calculate TR, +DM, -DM arrays
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.empty(n, dtype=np.float64)
    minus_dm = np.empty(n, dtype=np.float64)

    # First TR value (no previous close)
    tr[0] = high[0] - low[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        # Directional Movement
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Step 2: Smooth TR, +DM, -DM using Wilder's smoothing
    # First smoothed value is the sum of first timeperiod values
    smoothed_tr = 0.0
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0

    for i in range(timeperiod):
        smoothed_tr += tr[i]
        smoothed_plus_dm += plus_dm[i]
        smoothed_minus_dm += minus_dm[i]

    # Step 3-5: Calculate DI and DX
    dx = np.empty(n, dtype=np.float64)
    for i in range(timeperiod - 1):
        dx[i] = np.nan

    # Calculate +DI, -DI, DX for timeperiod onwards
    for i in range(timeperiod, n):
        # Wilder's smoothing: smooth[i] = smooth[i-1] - smooth[i-1]/n + new_value
        smoothed_tr = smoothed_tr - smoothed_tr / timeperiod + tr[i]
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / timeperiod + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / timeperiod + minus_dm[i]

        # Calculate directional indicators
        if smoothed_tr != 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum != 0:
            dx[i] = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx[i] = 0.0

    # Step 6: Smooth DX to get ADX
    # First ADX value is the average of first timeperiod DX values
    adx_sum = 0.0
    for i in range(timeperiod, 2 * timeperiod):
        adx_sum += dx[i]

    adx = adx_sum / timeperiod
    output[2 * timeperiod - 1] = adx

    # Subsequent ADX values use Wilder's smoothing
    for i in range(2 * timeperiod, n):
        adx = adx + (dx[i] - adx) / timeperiod
        output[i] = adx


def ADX(high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        timeperiod: int = 14) -> np.ndarray:
    """
    Average Directional Movement Index (ADX)

    The Average Directional Movement Index (ADX) is used to measure the strength
    of a trend. ADX is non-directional; it quantifies trend strength regardless
    of whether the trend is up or down.

    The ADX is derived from the relationship of the Directional Movement Index (DMI)
    and the Average True Range (ATR). It uses smoothed moving averages of the
    difference between two consecutive lows and highs.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the indicator (default: 14)

    Returns
    -------
    np.ndarray
        Array of ADX values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib ADX signature
    - Uses Numba JIT compilation for maximum performance
    - The first (2 * timeperiod - 1) values will be NaN
    - ADX values range from 0 to 100
    - ADX > 25 typically indicates a strong trend
    - ADX < 20 typically indicates a weak trend or ranging market

    Formula
    -------
    1. Calculate True Range (TR):
       TR = max(high - low, |high - prev_close|, |low - prev_close|)

    2. Calculate Directional Movement (+DM, -DM):
       +DM = high - prev_high (if positive and > down_move, else 0)
       -DM = prev_low - low (if positive and > up_move, else 0)

    3. Smooth TR, +DM, -DM using Wilder's smoothing:
       First value = sum of first timeperiod values
       Subsequent = prev_smooth - prev_smooth/timeperiod + current_value

    4. Calculate Directional Indicators:
       +DI = 100 * smoothed(+DM) / smoothed(TR)
       -DI = 100 * smoothed(-DM) / smoothed(TR)

    5. Calculate DX (Directional Index):
       DX = 100 * |+DI - -DI| / (+DI + -DI)

    6. Calculate ADX (smoothed DX):
       First ADX = average of first timeperiod DX values
       Subsequent ADX = prev_ADX + (DX - prev_ADX) / timeperiod

    Interpretation:
    - ADX measures trend strength (not direction)
    - Rising ADX indicates strengthening trend
    - Falling ADX indicates weakening trend
    - Use with +DI/-DI to determine trend direction

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ADX
    >>> high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    >>> close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5])
    >>> adx = ADX(high, low, close, timeperiod=5)
    >>> print(adx)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Validate input lengths
    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    lookback = 2 * timeperiod - 1
    if n < lookback + 1:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _adx_numba(high, low, close, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _adxr_numba(adx: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ADXR calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    ADXR Formula:
    ADXR[i] = (ADX[i] + ADX[i - (timeperiod - 1)]) / 2
    """
    n = len(adx)
    lookback = 3 * timeperiod - 2

    # Fill lookback period with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Calculate ADXR: average of current ADX and ADX from (timeperiod-1) ago
    lag = timeperiod - 1
    for i in range(lookback, n):
        output[i] = (adx[i] + adx[i - lag]) / 2.0


def ADXR(high: Union[np.ndarray, list],
         low: Union[np.ndarray, list],
         close: Union[np.ndarray, list],
         timeperiod: int = 14) -> np.ndarray:
    """
    Average Directional Movement Index Rating (ADXR)

    The Average Directional Movement Index Rating (ADXR) is a smoothed version
    of the ADX indicator. It is calculated as the average of the current ADX
    and the ADX from (timeperiod - 1) bars ago.

    ADXR provides a smoother trend strength measurement than ADX, reducing
    short-term volatility while maintaining sensitivity to longer-term trends.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the indicator (default: 14)

    Returns
    -------
    np.ndarray
        Array of ADXR values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib ADXR signature
    - Uses Numba JIT compilation for maximum performance
    - The first (3 * timeperiod - 2) values will be NaN
    - ADXR values range from 0 to 100
    - ADXR provides a smoother reading than ADX

    Formula
    -------
    ADXR[i] = (ADX[i] + ADX[i - (timeperiod - 1)]) / 2

    Where ADX is the Average Directional Movement Index.

    Lookback period: 3 * timeperiod - 2
    (For timeperiod=14, lookback=40)

    Interpretation:
    - ADXR measures trend strength with less noise than ADX
    - Similar thresholds as ADX apply:
      - ADXR > 25: Strong trend
      - ADXR < 20: Weak trend or ranging market
    - Rising ADXR: Strengthening trend
    - Falling ADXR: Weakening trend
    - ADXR lags ADX by approximately (timeperiod - 1) / 2 periods

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ADXR
    >>> high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    >>> close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5])
    >>> adxr = ADXR(high, low, close, timeperiod=5)
    >>> print(adxr)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Calculate ADX first
    adx = ADX(high, low, close, timeperiod)

    n = len(adx)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Lookback period: 3 * timeperiod - 2
    lookback = 3 * timeperiod - 2

    # Not enough data points - return all NaN
    if n < lookback + 1:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _adxr_numba(adx, timeperiod, output)

    return output