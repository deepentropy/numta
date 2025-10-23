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


@jit(nopython=True, cache=True)
def _sma_for_apo(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Helper function to calculate SMA for APO
    Returns SMA array (used internally by APO)
    """
    n = len(close)
    output = np.empty(n, dtype=np.float64)

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

    return output


@jit(nopython=True, cache=True)
def _ema_for_apo(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Helper function to calculate EMA for APO
    Returns EMA array (used internally by APO)
    """
    n = len(close)
    multiplier = 2.0 / (timeperiod + 1)

    output = np.empty(n, dtype=np.float64)

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

    return output


@jit(nopython=True, cache=True)
def _apo_numba_sma(close: np.ndarray, fastperiod: int, slowperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled APO calculation using SMA (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    APO Formula:
    APO = SMA(fastperiod) - SMA(slowperiod)
    """
    n = len(close)

    # Calculate fast and slow SMAs
    fast_ma = _sma_for_apo(close, fastperiod)
    slow_ma = _sma_for_apo(close, slowperiod)

    # Lookback is determined by the slower MA
    lookback = slowperiod - 1

    # Fill lookback period with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Calculate APO: fast MA - slow MA
    for i in range(lookback, n):
        output[i] = fast_ma[i] - slow_ma[i]


@jit(nopython=True, cache=True)
def _apo_numba_ema(close: np.ndarray, fastperiod: int, slowperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled APO calculation using EMA (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    APO Formula:
    APO = EMA(fastperiod) - EMA(slowperiod)
    """
    n = len(close)

    # Calculate fast and slow EMAs
    fast_ma = _ema_for_apo(close, fastperiod)
    slow_ma = _ema_for_apo(close, slowperiod)

    # Lookback is determined by the slower MA
    lookback = slowperiod - 1

    # Fill lookback period with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Calculate APO: fast MA - slow MA
    for i in range(lookback, n):
        output[i] = fast_ma[i] - slow_ma[i]


def APO(close: Union[np.ndarray, list],
        fastperiod: int = 12,
        slowperiod: int = 26,
        matype: int = 0) -> np.ndarray:
    """
    Absolute Price Oscillator (APO)

    The Absolute Price Oscillator (APO) displays the difference between two
    moving averages of a security's price. It is expressed in absolute terms
    (price points) rather than percentage terms.

    APO is similar to MACD but shows the absolute difference between the MAs,
    while MACD shows the same values but is typically used with a signal line
    and histogram.

    Parameters
    ----------
    close : array-like
        Close prices array
    fastperiod : int, optional
        Number of periods for the fast MA (default: 12)
    slowperiod : int, optional
        Number of periods for the slow MA (default: 26)
    matype : int, optional
        Moving average type: 0 = SMA (default), 1 = EMA

    Returns
    -------
    np.ndarray
        Array of APO values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib APO signature
    - Uses Numba JIT compilation for maximum performance
    - The first (slowperiod - 1) values will be NaN
    - APO values can be positive or negative
    - Positive APO indicates fast EMA > slow EMA (bullish)
    - Negative APO indicates fast EMA < slow EMA (bearish)

    Formula
    -------
    APO = MA(close, fastperiod) - MA(close, slowperiod)

    Where:
    - MA is either SMA (matype=0, default) or EMA (matype=1)
    - fastperiod < slowperiod (typically)

    Lookback period: slowperiod - 1
    (For fastperiod=12, slowperiod=26, lookback=25)

    Interpretation:
    - APO > 0: Fast MA above slow MA (bullish signal)
    - APO < 0: Fast MA below slow MA (bearish signal)
    - APO crossing above 0: Potential buy signal
    - APO crossing below 0: Potential sell signal
    - Increasing APO: Strengthening uptrend
    - Decreasing APO: Strengthening downtrend

    Difference from MACD:
    - APO shows absolute difference (price points)
    - MACD is essentially the same but typically includes signal line
    - APO = MACD line (without signal line or histogram)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import APO
    >>> close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    >>> apo = APO(close, fastperiod=3, slowperiod=5)
    >>> print(apo)
    """
    # Validate inputs
    if fastperiod < 2:
        raise ValueError("fastperiod must be >= 2")
    if slowperiod < 2:
        raise ValueError("slowperiod must be >= 2")
    if matype not in (0, 1):
        raise ValueError("Only matype=0 (SMA) and matype=1 (EMA) are currently supported")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    lookback = slowperiod - 1
    if n < lookback + 1:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)

    if matype == 0:
        # SMA (default)
        _apo_numba_sma(close, fastperiod, slowperiod, output)
    else:
        # EMA (matype == 1)
        _apo_numba_ema(close, fastperiod, slowperiod, output)

    return output


@jit(nopython=True, cache=True)
def _aroon_numba(high: np.ndarray, low: np.ndarray, timeperiod: int,
                 aroondown: np.ndarray, aroonup: np.ndarray) -> None:
    """
    Numba-compiled Aroon calculation (in-place) - Optimized O(n) version

    This function is JIT-compiled for maximum performance.
    It modifies the output arrays in-place.

    Uses a monotonic deque approach to efficiently track max/min in sliding window.
    Time complexity: O(n) instead of O(n*m)

    Aroon Formula:
    Aroon Up = ((timeperiod - periods since highest high) / timeperiod) * 100
    Aroon Down = ((timeperiod - periods since lowest low) / timeperiod) * 100
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        aroondown[i] = np.nan
        aroonup[i] = np.nan

    # We'll use a simple array-based deque for tracking indices
    # For high: we want monotonic decreasing (track potential maxima)
    # For low: we want monotonic increasing (track potential minima)
    # Make the deque larger than the window to handle edge cases safely
    max_window_size = n  # Just allocate enough space for worst case

    # Deques to store indices (using fixed-size arrays)
    high_deque = np.empty(max_window_size, dtype=np.int64)
    low_deque = np.empty(max_window_size, dtype=np.int64)
    high_front = 0  # Front index
    high_back = 0   # Back index (exclusive)
    low_front = 0
    low_back = 0

    # Process each window
    for i in range(n):
        # Window is from (i - timeperiod) to i (inclusive), so timeperiod+1 elements
        # But Aroon looks at the last timeperiod+1 bars (including current bar)
        window_start = i - timeperiod

        # Remove indices that are out of the current window for high deque
        while high_front < high_back and high_deque[high_front] < window_start:
            high_front += 1

        # Remove indices that are out of the current window for low deque
        while low_front < low_back and low_deque[low_front] < window_start:
            low_front += 1

        # Maintain monotonic decreasing deque for high (remove smaller or equal values)
        # When equal, keep the newer (rightmost) index for Aroon
        while high_front < high_back and high[high_deque[high_back - 1]] <= high[i]:
            high_back -= 1

        # Maintain monotonic increasing deque for low (remove larger or equal values)
        # When equal, keep the newer (rightmost) index for Aroon
        while low_front < low_back and low[low_deque[low_back - 1]] >= low[i]:
            low_back -= 1

        # Add current index to both deques
        high_deque[high_back] = i
        high_back += 1
        low_deque[low_back] = i
        low_back += 1

        # Calculate Aroon values only after we have enough data
        if i >= timeperiod:
            # The front of the deque contains the index of max/min in the window
            high_idx = high_deque[high_front]
            low_idx = low_deque[low_front]

            # Calculate periods since high/low
            periods_since_high = i - high_idx
            periods_since_low = i - low_idx

            # Calculate Aroon values
            aroonup[i] = ((timeperiod - periods_since_high) / timeperiod) * 100.0
            aroondown[i] = ((timeperiod - periods_since_low) / timeperiod) * 100.0


def AROON(high: Union[np.ndarray, list],
          low: Union[np.ndarray, list],
          timeperiod: int = 14) -> tuple:
    """
    Aroon (Aroon Indicator)

    The Aroon indicator is a technical indicator used to identify trend changes
    and the strength of a trend. It consists of two lines: Aroon Up and Aroon Down.

    Aroon Up measures the time elapsed since the highest high over the period.
    Aroon Down measures the time elapsed since the lowest low over the period.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Number of periods for the indicator (default: 14)

    Returns
    -------
    tuple of np.ndarray
        (aroondown, aroonup) - Two arrays with Aroon Down and Aroon Up values

    Notes
    -----
    - Compatible with TA-Lib AROON signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Aroon values range from 0 to 100
    - Aroon Up = 100 when current price is at the period high
    - Aroon Down = 100 when current price is at the period low

    Formula
    -------
    Aroon Up = ((timeperiod - periods_since_high) / timeperiod) * 100
    Aroon Down = ((timeperiod - periods_since_low) / timeperiod) * 100

    Where:
    - periods_since_high = periods since the highest high in the window
    - periods_since_low = periods since the lowest low in the window

    Lookback period: timeperiod
    (For timeperiod=14, lookback=14)

    Interpretation:
    - Aroon Up > 70 and Aroon Down < 30: Strong uptrend
    - Aroon Down > 70 and Aroon Up < 30: Strong downtrend
    - Aroon Up crossing above Aroon Down: Potential bullish signal
    - Aroon Down crossing above Aroon Up: Potential bearish signal
    - Both lines between 30-70: Consolidation/ranging market
    - Aroon Up = 100: New high reached
    - Aroon Down = 100: New low reached

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import AROON
    >>> high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    >>> aroondown, aroonup = AROON(high, low, timeperiod=5)
    >>> print(aroondown)
    >>> print(aroonup)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # Validate input lengths
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod + 1:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output arrays and run Numba-optimized calculation
    aroondown = np.empty(n, dtype=np.float64)
    aroonup = np.empty(n, dtype=np.float64)
    _aroon_numba(high, low, timeperiod, aroondown, aroonup)

    return aroondown, aroonup


def AROONOSC(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             timeperiod: int = 14) -> np.ndarray:
    """
    Aroon Oscillator (AROONOSC)

    The Aroon Oscillator is the difference between Aroon Up and Aroon Down.
    It oscillates between -100 and +100, with zero as the midpoint.

    The oscillator helps identify the strength and direction of a trend.
    Positive values indicate bullish momentum, while negative values
    indicate bearish momentum.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Number of periods for the indicator (default: 14)

    Returns
    -------
    np.ndarray
        Array of Aroon Oscillator values

    Notes
    -----
    - Compatible with TA-Lib AROONOSC signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Values range from -100 to +100
    - Zero line crossovers can signal trend changes

    Formula
    -------
    AROONOSC = Aroon Up - Aroon Down

    Where:
    - Aroon Up = ((timeperiod - periods_since_high) / timeperiod) * 100
    - Aroon Down = ((timeperiod - periods_since_low) / timeperiod) * 100

    Lookback period: timeperiod
    (For timeperiod=14, lookback=14)

    Interpretation:
    - AROONOSC > 0: Aroon Up > Aroon Down (bullish, uptrend dominant)
    - AROONOSC < 0: Aroon Down > Aroon Up (bearish, downtrend dominant)
    - AROONOSC > 50: Strong uptrend
    - AROONOSC < -50: Strong downtrend
    - AROONOSC near 0: Consolidation or weak trend
    - AROONOSC crossing above 0: Potential bullish signal
    - AROONOSC crossing below 0: Potential bearish signal
    - Extreme values (+100 or -100): Very strong trend

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import AROONOSC
    >>> high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    >>> aroonosc = AROONOSC(high, low, timeperiod=5)
    >>> print(aroonosc)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # Validate input lengths
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod + 1:
        return np.full(n, np.nan, dtype=np.float64)

    # Calculate Aroon Up and Aroon Down
    aroondown, aroonup = AROON(high, low, timeperiod)

    # Calculate oscillator: Aroon Up - Aroon Down
    aroonosc = aroonup - aroondown

    return aroonosc


@jit(nopython=True, cache=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ATR calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    ATR Formula:
    1. Calculate True Range (TR) for each bar
    2. Apply Wilder's smoothing to TR to get ATR
    """
    n = len(high)

    # Fill lookback period with NaN (lookback = timeperiod)
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate True Range for all bars
    tr = np.empty(n, dtype=np.float64)

    # First TR value (no previous close)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # We need at least timeperiod+1 bars to calculate the first ATR
    if n < timeperiod + 1:
        return

    # Calculate first ATR as simple average of TR[1] through TR[timeperiod]
    # Note: We skip TR[0] because it doesn't use previous close
    atr_sum = 0.0
    for i in range(1, timeperiod + 1):
        atr_sum += tr[i]

    atr = atr_sum / timeperiod
    output[timeperiod] = atr

    # Apply Wilder's smoothing for subsequent values
    # ATR[i] = ((ATR[i-1] * (timeperiod - 1)) + TR[i]) / timeperiod
    # Which is equivalent to: ATR[i] = ATR[i-1] - (ATR[i-1] / timeperiod) + (TR[i] / timeperiod)
    for i in range(timeperiod + 1, n):
        atr = atr - (atr / timeperiod) + (tr[i] / timeperiod)
        output[i] = atr


def ATR(high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        timeperiod: int = 14) -> np.ndarray:
    """
    Average True Range (ATR)

    The Average True Range (ATR) is a volatility indicator that measures
    the average range of price movement. It was developed by J. Welles Wilder
    and is widely used to assess market volatility.

    ATR is particularly useful for:
    - Setting stop-loss levels
    - Position sizing based on volatility
    - Identifying breakout potential
    - Comparing volatility across different instruments

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
        Array of ATR values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib ATR signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - ATR is always positive (measures absolute volatility, not direction)
    - Higher ATR indicates higher volatility
    - Lower ATR indicates lower volatility

    Formula
    -------
    1. Calculate True Range (TR) for each bar:
       TR = max(high - low, |high - prev_close|, |low - prev_close|)

    2. Calculate ATR using Wilder's smoothing:
       First ATR = average of first timeperiod TR values
       Subsequent ATR = ((prev_ATR × (timeperiod - 1)) + current_TR) / timeperiod

    Lookback period: timeperiod
    (For timeperiod=14, lookback=14)

    Interpretation:
    - ATR measures volatility, not direction
    - Rising ATR indicates increasing volatility
    - Falling ATR indicates decreasing volatility
    - ATR is often used with multiples (e.g., 2×ATR for stop-loss)
    - Compare current ATR to historical ATR for context
    - Higher timeframes generally have higher ATR values

    Common Uses:
    - Stop-loss placement: Close - (2 × ATR) for long positions
    - Position sizing: Risk a fixed dollar amount per ATR unit
    - Breakout confirmation: Look for ATR expansion on breakouts
    - Trend strength: Higher ATR often accompanies strong trends

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ATR
    >>> high = np.array([48, 49, 50, 51, 52, 51, 50, 49, 50, 51, 52])
    >>> low = np.array([46, 47, 48, 49, 50, 49, 48, 47, 48, 49, 50])
    >>> close = np.array([47, 48, 49, 50, 51, 50, 49, 48, 49, 50, 51])
    >>> atr = ATR(high, low, close, timeperiod=5)
    >>> print(atr)
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

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
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _atr_numba(high, low, close, timeperiod, output)

    return output


def BOP(open_price: Union[np.ndarray, list],
        high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Balance Of Power (BOP)

    Balance of Power is a momentum indicator that measures the strength of buying
    and selling pressure. It was introduced by Igor Levshin in the August 2001
    issue of Technical Analysis of Stocks & Commodities magazine.

    The indicator calculates the ratio between the close-open range and the
    high-low range, providing insight into which side (buyers or sellers) is
    winning the battle for price control.

    Formula
    -------
    BOP = (Close - Open) / (High - Low)

    When High equals Low (no price range), BOP is set to 0.

    Parameters
    ----------
    open_price : array-like
        Open prices array
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array

    Returns
    -------
    np.ndarray
        Array of Balance of Power values (range: -1 to +1)

    Notes
    -----
    - Compatible with TA-Lib BOP signature
    - No lookback period - returns values for all input bars
    - Range: -1.0 to +1.0
    - Positive values indicate buying pressure dominance (bulls in control)
    - Negative values indicate selling pressure dominance (bears in control)
    - Values near zero indicate balance between buyers and sellers
    - When High = Low (no range), BOP is set to 0 to avoid division by zero
    - All input arrays must have the same length

    Interpretation
    --------------
    - BOP > 0: Bulls are winning (close > open, buyers dominating)
    - BOP < 0: Bears are winning (close < open, sellers dominating)
    - BOP = 0: Balance or no price range
    - BOP near +1: Very strong buying pressure (close near high, open near low)
    - BOP near -1: Very strong selling pressure (close near low, open near high)
    - BOP oscillating around 0: Market indecision or consolidation

    Trading Signals
    ---------------
    - Crossing above 0: Potential bullish signal
    - Crossing below 0: Potential bearish signal
    - Divergence between BOP and price: Potential trend reversal
    - Sustained positive BOP: Uptrend confirmation
    - Sustained negative BOP: Downtrend confirmation

    Common Usage
    ------------
    BOP is often smoothed with a moving average (e.g., 14-period SMA) to reduce
    noise and make trends more visible. The raw BOP can be quite volatile.

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import BOP
    >>> open_price = np.array([100, 101, 102, 103, 104])
    >>> high = np.array([105, 106, 107, 108, 109])
    >>> low = np.array([99, 100, 101, 102, 103])
    >>> close = np.array([103, 104, 105, 106, 107])
    >>> bop = BOP(open_price, high, low, close)
    >>> print(bop)
    [0.5 0.5 0.5 0.5 0.5]

    See Also
    --------
    ADX : Average Directional Index (trend strength)
    AROON : Aroon Indicator (trend identification)
    MFI : Money Flow Index (volume-weighted momentum)
    """
    # Convert to numpy arrays if needed
    open_price = np.asarray(open_price, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Check all arrays have the same length
    n = len(open_price)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("open, high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Calculate BOP: (Close - Open) / (High - Low)
    # Handle division by zero when High == Low
    numerator = close - open_price
    denominator = high - low

    # Create output array
    output = np.zeros(n, dtype=np.float64)

    # Avoid division by zero
    non_zero_range = denominator != 0.0
    output[non_zero_range] = numerator[non_zero_range] / denominator[non_zero_range]

    return output


@jit(nopython=True, cache=True)
def _cci_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled CCI calculation (in-place)

    CCI = (Typical Price - SMA of TP) / (0.015 × Mean Absolute Deviation)

    Where:
    - Typical Price (TP) = (High + Low + Close) / 3
    - SMA of TP = Simple Moving Average of Typical Price over timeperiod
    - Mean Absolute Deviation = Mean of |TP - SMA of TP| over timeperiod
    - 0.015 is Lambert's constant to normalize the indicator
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate CCI for each window
    for i in range(timeperiod - 1, n):
        # Get window
        window_start = i - timeperiod + 1
        window_end = i + 1

        # Calculate typical prices for the window
        sum_tp = 0.0
        for j in range(window_start, window_end):
            tp = (high[j] + low[j] + close[j]) / 3.0
            sum_tp += tp

        # Calculate SMA of typical price
        sma_tp = sum_tp / timeperiod

        # Calculate current typical price
        current_tp = (high[i] + low[i] + close[i]) / 3.0

        # Calculate mean absolute deviation
        sum_abs_dev = 0.0
        for j in range(window_start, window_end):
            tp = (high[j] + low[j] + close[j]) / 3.0
            sum_abs_dev += abs(tp - sma_tp)

        mean_abs_dev = sum_abs_dev / timeperiod

        # Calculate CCI
        if mean_abs_dev == 0.0:
            output[i] = 0.0
        else:
            output[i] = (current_tp - sma_tp) / (0.015 * mean_abs_dev)


def CCI(high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        timeperiod: int = 14) -> np.ndarray:
    """
    Commodity Channel Index (CCI)

    The Commodity Channel Index (CCI) is a momentum-based oscillator developed by
    Donald Lambert and featured in Commodities magazine in 1980. CCI measures the
    variation of a security's price from its statistical mean.

    High CCI values indicate that prices are unusually high compared to average,
    while low values indicate that prices are unusually low. The indicator can be
    used to identify overbought and oversold levels, as well as divergences that
    may signal trend reversals.

    Formula
    -------
    1. Typical Price (TP) = (High + Low + Close) / 3
    2. SMA of TP = Simple Moving Average of Typical Price over timeperiod
    3. Mean Absolute Deviation = Mean of |TP - SMA of TP| over timeperiod
    4. CCI = (TP - SMA of TP) / (0.015 × Mean Absolute Deviation)

    The constant 0.015 was chosen by Lambert to ensure that approximately 70-80%
    of CCI values fall between -100 and +100.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of CCI values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib CCI signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - All input arrays must have the same length
    - When Mean Absolute Deviation is 0, CCI is set to 0

    Interpretation
    --------------
    - CCI > +100: Overbought condition (prices unusually high)
    - CCI < -100: Oversold condition (prices unusually low)
    - CCI crossing above +100: Buy signal
    - CCI crossing below -100: Sell signal
    - CCI returning to 0 line: Trend weakening
    - Divergence between CCI and price: Potential reversal

    Trading Levels
    --------------
    - +100: Overbought threshold
    - 0: Centerline (neutral)
    - -100: Oversold threshold
    - +200: Extremely overbought
    - -200: Extremely oversold

    Common Uses
    -----------
    - Identify cyclical trends in commodities
    - Detect overbought/oversold conditions
    - Confirm price breakouts
    - Spot divergences for reversal signals
    - Trade mean reversion strategies

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import CCI
    >>> high = np.array([83, 84, 85, 86, 87])
    >>> low = np.array([81, 82, 83, 84, 85])
    >>> close = np.array([82, 83, 84, 85, 86])
    >>> cci = CCI(high, low, close, timeperiod=5)
    >>> print(cci)

    See Also
    --------
    RSI : Relative Strength Index
    STOCH : Stochastic Oscillator
    MFI : Money Flow Index
    WILLR : Williams %R
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Check all arrays have the same length
    n = len(high)
    if not (len(low) == len(close) == n):
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _cci_numba(high, low, close, timeperiod, output)

    return output