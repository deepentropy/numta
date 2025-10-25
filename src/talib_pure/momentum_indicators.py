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

@jit(nopython=True, cache=True)
def _rsi_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled RSI calculation (in-place)

    Formula:
    1. Calculate price changes (gains and losses)
    2. Calculate average gain and average loss using Wilder's smoothing
    3. RS = Average Gain / Average Loss
    4. RSI = 100 - (100 / (1 + RS))
    """
    n = len(data)

    # Fill lookback period with NaN
    lookback = timeperiod
    for i in range(lookback):
        output[i] = np.nan

    # Calculate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        change = data[i] - data[i - 1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change

    # Calculate initial average gain and loss (simple average for first period)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, timeperiod + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= timeperiod
    avg_loss /= timeperiod

    # Calculate first RSI value
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            output[timeperiod] = 50.0  # No movement
        else:
            output[timeperiod] = 100.0  # All gains
    else:
        rs = avg_gain / avg_loss
        output[timeperiod] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(timeperiod + 1, n):
        avg_gain = (avg_gain * (timeperiod - 1) + gains[i]) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + losses[i]) / timeperiod

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                output[i] = 50.0
            else:
                output[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            output[i] = 100.0 - (100.0 / (1.0 + rs))




# GPU (CuPy) implementation
def _rsi_cupy(data: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based RSI calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    data_gpu = cp.asarray(data, dtype=cp.float64)
    n = len(data_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    lookback = timeperiod

    # Check if we have enough data
    if n <= lookback:
        return cp.asnumpy(output)

    # Calculate gains and losses
    gains = cp.zeros(n, dtype=cp.float64)
    losses = cp.zeros(n, dtype=cp.float64)

    for i in range(1, n):
        change = data_gpu[i] - data_gpu[i - 1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change

    # Calculate initial average gain and loss
    avg_gain = cp.sum(gains[1:timeperiod + 1]) / timeperiod
    avg_loss = cp.sum(losses[1:timeperiod + 1]) / timeperiod

    # Calculate first RSI value
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            output[timeperiod] = 50.0
        else:
            output[timeperiod] = 100.0
    else:
        rs = avg_gain / avg_loss
        output[timeperiod] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(timeperiod + 1, n):
        avg_gain = (avg_gain * (timeperiod - 1) + gains[i]) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + losses[i]) / timeperiod

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                output[i] = 50.0
            else:
                output[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            output[i] = 100.0 - (100.0 / (1.0 + rs))

    # Transfer back to CPU
    return cp.asnumpy(output)


@jit(nopython=True, cache=True)
def _roc_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROC calculation (in-place)

    Formula: ROC = ((price / prevPrice) - 1) * 100
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROC
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = ((data[i] / prev_price) - 1.0) * 100.0


def ROC(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change (ROC)

    ROC measures the percentage change in price over a specified time period.
    It is a momentum oscillator that oscillates above and below a zero line,
    showing the velocity of price changes.

    ROC is the percentage version of the Momentum (MOM) indicator, making it
    easier to compare across different securities regardless of price level.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROC values (percentage change)

    Notes
    -----
    - Compatible with TA-Lib ROC signature
    - Uses Numba JIT compilation for performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - Unbounded oscillator (no fixed range)

    Formula
    -------
    ROC[i] = ((Price[i] / Price[i - timeperiod]) - 1) * 100

    Equivalent to: ((Price[i] - Price[i - timeperiod]) / Price[i - timeperiod]) * 100

    Lookback period: timeperiod
    (For timeperiod=10, lookback=10)

    Interpretation:
    - Positive ROC: Price rising (bullish momentum)
    - Negative ROC: Price falling (bearish momentum)
    - ROC = 0: No change over period
    - Increasing ROC: Accelerating momentum
    - Decreasing ROC: Decelerating momentum
    - ROC crossing zero: Potential trend change

    Advantages:
    - Normalized for price level (percentage)
    - Comparable across different securities
    - Identifies momentum strength
    - Leading indicator
    - Simple and intuitive

    Common Uses:
    - Trend identification
    - Momentum strength measurement
    - Divergence analysis
    - Overbought/oversold detection
    - Confirmation signals
    - Cross-security comparison

    Trading Applications:
    - Buy when ROC crosses above zero
    - Sell when ROC crosses below zero
    - Divergence: Price makes new high but ROC doesn't (bearish)
    - Divergence: Price makes new low but ROC doesn't (bullish)
    - Overbought: ROC > +threshold (e.g., +10%)
    - Oversold: ROC < -threshold (e.g., -10%)

    Comparison with Related Indicators:
    - MOM: Absolute price difference (not percentage)
    - ROCP: Same as ROC but decimal form
    - ROCR: Ratio form (price/prevPrice)
    - RSI: Bounded version (0-100) with smoothing

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROC
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> roc = ROC(close, timeperiod=10)
    >>> print(roc)
    >>> # roc[10] = ((111 / 100) - 1) * 100 = 11%

    See Also
    --------
    MOM : Momentum
    ROCP : Rate of Change Percentage
    ROCR : Rate of Change Ratio
    RSI : Relative Strength Index
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _roc_numba(data, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _rocp_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROCP calculation (in-place)

    Formula: ROCP = (price - prevPrice) / prevPrice
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROCP
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = (data[i] - prev_price) / prev_price


def ROCP(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change Percentage (ROCP)

    ROCP calculates the percentage change in price as a decimal value (not multiplied
    by 100). It is mathematically equivalent to ROC / 100.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROCP values (decimal percentage change)

    Notes
    -----
    - Compatible with TA-Lib ROCP signature
    - Values are decimals (0.10 = 10%)
    - ROCP = ROC / 100

    Formula
    -------
    ROCP[i] = (Price[i] - Price[i - timeperiod]) / Price[i - timeperiod]

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROCP
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> rocp = ROCP(close, timeperiod=10)
    >>> # rocp[10] = (111 - 100) / 100 = 0.11 (11%)

    See Also
    --------
    ROC : Rate of Change (percentage form)
    ROCR : Rate of Change Ratio
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _rocp_numba(data, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _rocr_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROCR calculation (in-place)

    Formula: ROCR = price / prevPrice
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROCR
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = data[i] / prev_price


def ROCR(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change Ratio (ROCR)

    ROCR calculates the ratio of the current price to the price n periods ago.
    Values oscillate around 1.0, where 1.0 represents no change.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROCR values (ratio)

    Notes
    -----
    - Compatible with TA-Lib ROCR signature
    - Values oscillate around 1.0
    - ROCR > 1.0: Price increased
    - ROCR < 1.0: Price decreased
    - ROCR = 1.0: No change

    Formula
    -------
    ROCR[i] = Price[i] / Price[i - timeperiod]

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROCR
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> rocr = ROCR(close, timeperiod=10)
    >>> # rocr[10] = 111 / 100 = 1.11

    See Also
    --------
    ROC : Rate of Change (percentage)
    ROCR100 : Rate of Change Ratio * 100
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _rocr_numba(data, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _rocr100_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROCR100 calculation (in-place)

    Formula: ROCR100 = (price / prevPrice) * 100
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROCR100
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = (data[i] / prev_price) * 100.0


def ROCR100(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change Ratio 100 Scale (ROCR100)

    ROCR100 calculates the ratio of the current price to the price n periods ago,
    scaled to 100. Values oscillate around 100, where 100 represents no change.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROCR100 values (ratio * 100)

    Notes
    -----
    - Compatible with TA-Lib ROCR100 signature
    - Values oscillate around 100
    - ROCR100 > 100: Price increased
    - ROCR100 < 100: Price decreased
    - ROCR100 = 100: No change
    - ROCR100 = ROCR * 100

    Formula
    -------
    ROCR100[i] = (Price[i] / Price[i - timeperiod]) * 100

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROCR100
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> rocr100 = ROCR100(close, timeperiod=10)
    >>> # rocr100[10] = (111 / 100) * 100 = 111

    See Also
    --------
    ROCR : Rate of Change Ratio
    ROC : Rate of Change (percentage)
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _rocr100_numba(data, timeperiod, output)
    return output

def RSI(data: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI)

    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Developed by J. Welles Wilder in 1978, it oscillates between 0 and 100 and is
    primarily used to identify overbought and oversold conditions.

    RSI compares the magnitude of recent gains to recent losses to determine whether
    an asset is overbought or oversold. It uses Wilder's smoothing method for the
    average gains and losses.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for RSI calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of RSI values (0-100)

    Notes
    -----
    - Compatible with TA-Lib RSI signature
    - Uses Numba JIT compilation for performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - Bounded oscillator (0-100)
    - Uses Wilder's smoothing method

    Formula
    -------
    1. Calculate price changes:
       Gain = max(0, Price[i] - Price[i-1])
       Loss = max(0, Price[i-1] - Price[i])

    2. Calculate average gain and loss using Wilder's smoothing:
       First Average Gain = Sum(Gains over timeperiod) / timeperiod
       First Average Loss = Sum(Losses over timeperiod) / timeperiod

       Subsequent:
       Average Gain = ((Previous Avg Gain) * (timeperiod-1) + Current Gain) / timeperiod
       Average Loss = ((Previous Avg Loss) * (timeperiod-1) + Current Loss) / timeperiod

    3. Calculate Relative Strength (RS):
       RS = Average Gain / Average Loss

    4. Calculate RSI:
       RSI = 100 - (100 / (1 + RS))

    Alternative formula:
       RSI = 100 * (Average Gain / (Average Gain + Average Loss))

    Lookback period: timeperiod
    (For timeperiod=14, lookback=14)

    Interpretation:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI = 50: Neutral (equal buying/selling pressure)
    - RSI crossing 50: Trend change signal
    - Divergence: Price vs RSI moving in opposite directions

    Traditional Levels:
    - Overbought: RSI > 70
    - Oversold: RSI < 30
    - In strong trends, use 80/20 instead of 70/30

    Advantages:
    - Bounded (0-100) for easy interpretation
    - Works in trending and ranging markets
    - Identifies overbought/oversold conditions
    - Detects divergences
    - Widely used and understood
    - Multiple timeframes applicable

    Common Uses:
    - Overbought/oversold identification
    - Divergence detection
    - Trend strength measurement
    - Support/resistance levels
    - Centerline crossovers (50)
    - Failure swings (reversal patterns)

    Trading Signals:

    1. Overbought/Oversold:
       - Buy when RSI crosses above 30 (leaving oversold)
       - Sell when RSI crosses below 70 (leaving overbought)

    2. Centerline Crossover:
       - Buy when RSI crosses above 50 (bullish)
       - Sell when RSI crosses below 50 (bearish)

    3. Divergence:
       - Bullish: Price makes lower low, RSI makes higher low
       - Bearish: Price makes higher high, RSI makes lower high

    4. Failure Swings:
       - Top: RSI > 70, pullback, fails to exceed previous high, then breaks pullback low
       - Bottom: RSI < 30, bounce, fails to break previous low, then breaks bounce high

    Timeframe Adjustments:
    - Short-term: RSI(9) - More sensitive, more signals
    - Standard: RSI(14) - Wilder's original recommendation
    - Long-term: RSI(25) - Less sensitive, fewer signals

    Limitations:
    - Can remain overbought/oversold for extended periods
    - Less effective in strong trending markets
    - Whipsaw signals in choppy conditions
    - Lag due to smoothing

    Comparison with Related Indicators:
    - Stochastic: Similar concept, different calculation
    - MFI: Volume-weighted version of RSI
    - ROC: Unbounded momentum indicator
    - MACD: Trend-following momentum indicator

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import RSI
    >>> close = np.array([44, 44.5, 45, 45.5, 45, 44.5, 44, 44.5, 45, 45.5,
    ...                   46, 46.5, 47, 47.5, 48])
    >>> rsi = RSI(close, timeperiod=14)
    >>> print(rsi)
    >>> # Values will be between 0 and 100

    See Also
    --------
    MFI : Money Flow Index (volume-weighted RSI)
    ROC : Rate of Change
    STOCH : Stochastic Oscillator
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
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _rsi_cupy(data, timeperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output array and run Numba-optimized calculation
        output = np.empty(n, dtype=np.float64)
        _rsi_numba(data, timeperiod, output)

        return output
@jit(nopython=True, cache=True)
def _stoch_fastk_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       fastk_period: int, output: np.ndarray) -> None:
    """Numba-compiled Fast %K calculation for STOCH"""
    n = len(high)

    for i in range(fastk_period - 1):
        output[i] = np.nan

    for i in range(fastk_period - 1, n):
        highest = high[i]
        lowest = low[i]
        for j in range(i - fastk_period + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]

        if highest - lowest == 0:
            output[i] = 50.0
        else:
            output[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0


def STOCHF(high: Union[np.ndarray, list],
           low: Union[np.ndarray, list],
           close: Union[np.ndarray, list],
           fastk_period: int = 5,
           fastd_period: int = 3,
           fastd_matype: int = 0) -> tuple:
    """
    Stochastic Fast (STOCHF)

    Fast Stochastic oscillator that shows the position of the closing price
    relative to the high-low range over a set period. Returns Fast %K and Fast %D.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    fastk_period : int, optional
        Period for Fast %K calculation (default: 5)
    fastd_period : int, optional
        Period for Fast %D (SMA of Fast %K) (default: 3)
    fastd_matype : int, optional
        Type of moving average for Fast %D (default: 0 = SMA)

    Returns
    -------
    tuple of np.ndarray
        (fastk, fastd) - Fast %K and Fast %D arrays

    Notes
    -----
    Fast %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    Fast %D = SMA(Fast %K, fastd_period)
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    if fastk_period < 1:
        raise ValueError("fastk_period must be >= 1")
    if fastd_period < 1:
        raise ValueError("fastd_period must be >= 1")

    # Calculate Fast %K using Numba-optimized helper
    fastk = np.empty(n, dtype=np.float64)
    _stoch_fastk_numba(high, low, close, fastk_period, fastk)

    # Calculate Fast %D
    from .overlap import SMA
    fastd = SMA(fastk, timeperiod=fastd_period)

    return fastk, fastd


def STOCH(high: Union[np.ndarray, list],
          low: Union[np.ndarray, list],
          close: Union[np.ndarray, list],
          fastk_period: int = 5,
          slowk_period: int = 3,
          slowk_matype: int = 0,
          slowd_period: int = 3,
          slowd_matype: int = 0) -> tuple:
    """
    Stochastic (STOCH)

    Stochastic oscillator that shows the position of the closing price
    relative to the high-low range over a set period. The slow version applies
    additional smoothing to reduce noise.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    fastk_period : int, optional
        Period for initial %K calculation (default: 5)
    slowk_period : int, optional
        Period for smoothing fast %K to slow %K (default: 3)
    slowk_matype : int, optional
        Type of moving average for slow %K (default: 0 = SMA)
    slowd_period : int, optional
        Period for smoothing slow %K to slow %D (default: 3)
    slowd_matype : int, optional
        Type of moving average for slow %D (default: 0 = SMA)

    Returns
    -------
    tuple of np.ndarray
        (slowk, slowd) - Slow %K and Slow %D arrays

    Notes
    -----
    Fast %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    Slow %K = SMA(Fast %K, slowk_period)
    Slow %D = SMA(Slow %K, slowd_period)
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    if fastk_period < 1:
        raise ValueError("fastk_period must be >= 1")
    if slowk_period < 1:
        raise ValueError("slowk_period must be >= 1")
    if slowd_period < 1:
        raise ValueError("slowd_period must be >= 1")

    # Calculate Fast %K using Numba-optimized helper
    fastk = np.empty(n, dtype=np.float64)
    _stoch_fastk_numba(high, low, close, fastk_period, fastk)

    # Smooth Fast %K to get Slow %K
    from .overlap import SMA
    slowk = SMA(fastk, timeperiod=slowk_period)

    # Smooth Slow %K to get Slow %D
    slowd = SMA(slowk, timeperiod=slowd_period)

    return slowk, slowd


def STOCHRSI(data: Union[np.ndarray, list],
             timeperiod: int = 14,
             fastk_period: int = 5,
             fastd_period: int = 3,
             fastd_matype: int = 0) -> tuple:
    """
    Stochastic Relative Strength Index (STOCHRSI)

    Applies the Stochastic oscillator formula to RSI values instead of price.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Period for RSI calculation (default: 14)
    fastk_period : int, optional
        Period for Stochastic %K on RSI (default: 5)
    fastd_period : int, optional
        Period for %D (SMA of %K) (default: 3)
    fastd_matype : int, optional
        Type of moving average for %D (default: 0 = SMA)

    Returns
    -------
    tuple of np.ndarray
        (fastk, fastd) - Stochastic RSI %K and %D arrays
    """
    # Convert to numpy array
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    if fastk_period < 1:
        raise ValueError("fastk_period must be >= 1")
    if fastd_period < 1:
        raise ValueError("fastd_period must be >= 1")

    # Calculate RSI
    rsi = RSI(data, timeperiod=timeperiod)

    # Apply Stochastic to RSI
    # RSI is already in 0-100 range, treat it as "close" price
    # For high/low, we use the RSI values themselves
    fastk = np.empty(n, dtype=np.float64)
    _stoch_fastk_numba(rsi, rsi, rsi, fastk_period, fastk)

    # Calculate Fast %D
    from .overlap import SMA
    fastd = SMA(fastk, timeperiod=fastd_period)

    return fastk, fastd


@jit(nopython=True, cache=True)
def _cmo_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled CMO calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    CMO Formula:
    CMO = ((Sum of Gains - Sum of Losses) / (Sum of Gains + Sum of Losses)) * 100
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate CMO for each window
    for i in range(timeperiod, n):
        # Calculate price changes over the window
        sum_gains = 0.0
        sum_losses = 0.0

        for j in range(i - timeperiod + 1, i + 1):
            change = close[j] - close[j - 1]
            if change > 0:
                sum_gains += change
            elif change < 0:
                sum_losses += abs(change)

        # Calculate CMO
        total = sum_gains + sum_losses
        if total == 0.0:
            output[i] = 0.0
        else:
            output[i] = ((sum_gains - sum_losses) / total) * 100.0




# GPU (CuPy) implementation
def _cmo_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based CMO calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    close_gpu = cp.asarray(close, dtype=cp.float64)
    n = len(close_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    # Check if we have enough data
    if n <= timeperiod:
        return cp.asnumpy(output)

    # Calculate CMO for each window
    for i in range(timeperiod, n):
        # Calculate price changes over the window
        sum_gains = cp.float64(0.0)
        sum_losses = cp.float64(0.0)

        for j in range(i - timeperiod + 1, i + 1):
            change = close_gpu[j] - close_gpu[j - 1]
            if change > 0:
                sum_gains += change
            elif change < 0:
                sum_losses += cp.abs(change)

        # Calculate CMO
        total = sum_gains + sum_losses
        if total == 0.0:
            output[i] = 0.0
        else:
            output[i] = ((sum_gains - sum_losses) / total) * 100.0

    # Transfer back to CPU
    return cp.asnumpy(output)


def CMO(close: Union[np.ndarray, list],
        timeperiod: int = 14) -> np.ndarray:
    """
    Chande Momentum Oscillator (CMO)

    The Chande Momentum Oscillator (CMO) is a momentum indicator developed by
    Tushar Chande. It measures the difference between the sum of gains and the
    sum of losses over a specified period, normalized by the total price movement.

    Unlike RSI which uses an exponential moving average, CMO uses simple sums,
    making it more responsive to recent price changes. CMO oscillates between
    -100 and +100, with zero as the midpoint.

    Formula
    -------
    CMO = ((Sum of Gains - Sum of Losses) / (Sum of Gains + Sum of Losses)) × 100

    Where:
    - Sum of Gains = Sum of all up moves over timeperiod
    - Sum of Losses = Sum of all down moves over timeperiod
    - Gains and losses are calculated as absolute values

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of CMO values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib CMO signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - CMO values range from -100 to +100
    - More responsive than RSI due to simple summation vs. exponential smoothing

    Interpretation
    --------------
    - CMO > 0: Bullish momentum (gains exceed losses)
    - CMO < 0: Bearish momentum (losses exceed gains)
    - CMO > +50: Overbought condition
    - CMO < -50: Oversold condition
    - CMO crossing above 0: Potential buy signal
    - CMO crossing below 0: Potential sell signal
    - Extreme values (+100 or -100): Very strong momentum

    Trading Levels
    --------------
    - +50: Overbought threshold
    - 0: Neutral/centerline
    - -50: Oversold threshold
    - +100: Maximum bullish momentum (all gains, no losses)
    - -100: Maximum bearish momentum (all losses, no gains)

    Advantages over RSI
    -------------------
    - More responsive to price changes
    - Symmetric scale (-100 to +100 vs. 0 to 100)
    - Easier to identify overbought/oversold symmetrically
    - Does not use exponential smoothing

    Common Uses
    -----------
    - Identify overbought/oversold conditions
    - Generate buy/sell signals on centerline crossovers
    - Spot divergences for reversal signals
    - Confirm trend strength
    - Filter trades (only buy when CMO > 0, only sell when CMO < 0)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import CMO
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    ...                   111, 110, 112, 114, 113])
    >>> cmo = CMO(close, timeperiod=14)
    >>> print(cmo)

    See Also
    --------
    RSI : Relative Strength Index
    MFI : Money Flow Index
    WILLR : Williams %R
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    # Need timeperiod + 1 because we need previous close for first calculation
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cmo_cupy(close, timeperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output array and run Numba-optimized calculation
        output = np.empty(n, dtype=np.float64)
        _cmo_numba(close, timeperiod, output)

        return output


@jit(nopython=True, cache=True)
def _dx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled DX calculation (in-place)

    DX = 100 * |+DI - -DI| / (+DI + -DI)
    """
    n = len(high)

    # Fill lookback period with NaN
    lookback = timeperiod
    for i in range(lookback):
        output[i] = np.nan

    # Calculate TR, +DM, -DM arrays
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.empty(n, dtype=np.float64)
    minus_dm = np.empty(n, dtype=np.float64)

    # First TR value (no previous close)
    tr[0] = high[0] - low[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        # True Range
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

    # Smooth TR, +DM, -DM using Wilder's smoothing
    smoothed_tr = 0.0
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0

    for i in range(timeperiod):
        smoothed_tr += tr[i]
        smoothed_plus_dm += plus_dm[i]
        smoothed_minus_dm += minus_dm[i]

    # Calculate DX for timeperiod onwards
    for i in range(timeperiod, n):
        # Wilder's smoothing
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
            output[i] = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            output[i] = 0.0




# GPU (CuPy) implementation
def _dx_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based DX calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    high_gpu = cp.asarray(high, dtype=cp.float64)
    low_gpu = cp.asarray(low, dtype=cp.float64)
    close_gpu = cp.asarray(close, dtype=cp.float64)
    n = len(high_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    lookback = timeperiod

    # Check if we have enough data
    if n <= lookback:
        return cp.asnumpy(output)

    # Calculate TR, +DM, -DM arrays
    tr = cp.empty(n, dtype=cp.float64)
    plus_dm = cp.empty(n, dtype=cp.float64)
    minus_dm = cp.empty(n, dtype=cp.float64)

    # First TR value (no previous close)
    tr[0] = high_gpu[0] - low_gpu[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        # True Range
        hl = high_gpu[i] - low_gpu[i]
        hc = cp.abs(high_gpu[i] - close_gpu[i - 1])
        lc = cp.abs(low_gpu[i] - close_gpu[i - 1])
        tr[i] = cp.maximum(cp.maximum(hl, hc), lc)

        # Directional Movement
        up_move = high_gpu[i] - high_gpu[i - 1]
        down_move = low_gpu[i - 1] - low_gpu[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Smooth TR, +DM, -DM using Wilder's smoothing
    smoothed_tr = cp.sum(tr[:timeperiod])
    smoothed_plus_dm = cp.sum(plus_dm[:timeperiod])
    smoothed_minus_dm = cp.sum(minus_dm[:timeperiod])

    # Calculate DX for timeperiod onwards
    for i in range(timeperiod, n):
        # Wilder's smoothing
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
            output[i] = 100.0 * cp.abs(plus_di - minus_di) / di_sum
        else:
            output[i] = 0.0

    # Transfer back to CPU
    return cp.asnumpy(output)


def DX(high: Union[np.ndarray, list],
       low: Union[np.ndarray, list],
       close: Union[np.ndarray, list],
       timeperiod: int = 14) -> np.ndarray:
    """
    Directional Movement Index (DX)

    The Directional Movement Index (DX) measures the strength of directional
    movement in a market. It is derived from the relationship between the Plus
    Directional Indicator (+DI) and Minus Directional Indicator (-DI).

    DX is a component of the ADX (Average Directional Index) indicator and
    represents the absolute difference between +DI and -DI divided by their sum.

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
        Array of DX values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib DX signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - DX values range from 0 to 100
    - Higher values indicate stronger directional movement

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

    5. Calculate DX:
       DX = 100 * |+DI - -DI| / (+DI + -DI)

    Lookback period: timeperiod
    (For timeperiod=14, lookback=14)

    Interpretation:
    - DX = 0-25: Weak or absent trend
    - DX = 25-50: Moderate trend strength
    - DX = 50-75: Strong trend
    - DX = 75-100: Very strong trend
    - Rising DX: Trend strengthening
    - Falling DX: Trend weakening

    Relationship to ADX:
    - DX is the raw calculation before smoothing
    - ADX = smoothed average of DX
    - DX is more volatile than ADX
    - ADX smooths out DX fluctuations

    Common Uses:
    - Measure trend strength (not direction)
    - Identify potential trend reversals
    - Filter weak trends from strong trends
    - Component in ADX calculation
    - Confirm breakout strength

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import DX
    >>> high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    >>> close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5])
    >>> dx = DX(high, low, close, timeperiod=5)
    >>> print(dx)

    See Also
    --------
    ADX : Average Directional Index
    ADXR : Average Directional Index Rating
    PLUS_DI : Plus Directional Indicator
    MINUS_DI : Minus Directional Indicator
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
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _dx_cupy(high, low, close, timeperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output array and run Numba-optimized calculation
        output = np.empty(n, dtype=np.float64)
        _dx_numba(high, low, close, timeperiod, output)

        return output


@jit(nopython=True, cache=True)
def _macd_numba(close: np.ndarray, fastperiod: int, slowperiod: int, signalperiod: int,
                macd: np.ndarray, signal: np.ndarray, hist: np.ndarray) -> None:
    """
    Numba-compiled MACD calculation (in-place)

    Formula:
    MACD = EMA(close, fastperiod) - EMA(close, slowperiod)
    Signal = EMA(MACD, signalperiod)
    Histogram = MACD - Signal
    """
    n = len(close)
    fast_mult = 2.0 / (fastperiod + 1)
    slow_mult = 2.0 / (slowperiod + 1)
    signal_mult = 2.0 / (signalperiod + 1)

    # Lookback = slowperiod + signalperiod - 2
    lookback = slowperiod + signalperiod - 2
    for i in range(lookback):
        macd[i] = np.nan
        signal[i] = np.nan
        hist[i] = np.nan

    # Calculate fast EMA
    fast_ema = np.empty(n, dtype=np.float64)
    for i in range(fastperiod - 1):
        fast_ema[i] = np.nan

    # Initialize first fast EMA as SMA
    sum_val = 0.0
    for i in range(fastperiod):
        sum_val += close[i]
    fast_ema[fastperiod - 1] = sum_val / fastperiod

    # Calculate remaining fast EMA values
    for i in range(fastperiod, n):
        fast_ema[i] = (close[i] - fast_ema[i - 1]) * fast_mult + fast_ema[i - 1]

    # Calculate slow EMA
    slow_ema = np.empty(n, dtype=np.float64)
    for i in range(slowperiod - 1):
        slow_ema[i] = np.nan

    # Initialize first slow EMA as SMA
    sum_val = 0.0
    for i in range(slowperiod):
        sum_val += close[i]
    slow_ema[slowperiod - 1] = sum_val / slowperiod

    # Calculate remaining slow EMA values
    for i in range(slowperiod, n):
        slow_ema[i] = (close[i] - slow_ema[i - 1]) * slow_mult + slow_ema[i - 1]

    # Calculate MACD line
    for i in range(slowperiod - 1, n):
        macd[i] = fast_ema[i] - slow_ema[i]

    # Calculate signal line (EMA of MACD)
    signal_start_idx = slowperiod + signalperiod - 2

    # Initialize first signal value as SMA of MACD
    sum_val = 0.0
    for i in range(slowperiod - 1, slowperiod + signalperiod - 1):
        sum_val += macd[i]
    signal_ema = sum_val / signalperiod
    signal[signal_start_idx] = signal_ema

    # Calculate remaining signal values
    for i in range(signal_start_idx + 1, n):
        signal_ema = (macd[i] - signal_ema) * signal_mult + signal_ema
        signal[i] = signal_ema

    # Calculate histogram
    for i in range(signal_start_idx, n):
        hist[i] = macd[i] - signal[i]




# GPU (CuPy) implementation
def _macd_cupy(close: np.ndarray, fastperiod: int, slowperiod: int, signalperiod: int) -> tuple:
    """
    CuPy-based MACD calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Import EMA from overlap for GPU calculation
    from .overlap import _ema_cupy

    # Calculate fast and slow EMAs
    fast_ema = cp.asarray(_ema_cupy(close, fastperiod), dtype=cp.float64)
    slow_ema = cp.asarray(_ema_cupy(close, slowperiod), dtype=cp.float64)

    # Calculate MACD line
    macd = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD)
    macd_cpu = cp.asnumpy(macd)
    signal = cp.asarray(_ema_cupy(macd_cpu, signalperiod), dtype=cp.float64)

    # Calculate histogram
    hist = macd - signal

    # Transfer back to CPU
    return cp.asnumpy(macd), cp.asnumpy(signal), cp.asnumpy(hist)


def MACD(close: Union[np.ndarray, list],
         fastperiod: int = 12,
         slowperiod: int = 26,
         signalperiod: int = 9) -> tuple:
    """
    Moving Average Convergence/Divergence (MACD)

    MACD is one of the most popular and widely used technical indicators. It shows
    the relationship between two exponential moving averages (EMAs) of prices.
    Developed by Gerald Appel in the late 1970s, MACD is used to identify trend
    direction, strength, momentum, and potential reversal points.

    The MACD consists of three components:
    1. MACD Line: Difference between fast and slow EMAs
    2. Signal Line: EMA of the MACD line
    3. Histogram: Difference between MACD and Signal lines

    Parameters
    ----------
    close : array-like
        Close prices array
    fastperiod : int, optional
        Period for fast EMA (default: 12)
    slowperiod : int, optional
        Period for slow EMA (default: 26)
    signalperiod : int, optional
        Period for signal line EMA (default: 9)

    Returns
    -------
    tuple of np.ndarray
        (macd, signal, histogram) - Three arrays with the MACD components

    Notes
    -----
    - Compatible with TA-Lib MACD signature
    - Uses Numba JIT compilation for maximum performance
    - Lookback period: slowperiod + signalperiod - 2
    - Standard settings: 12, 26, 9
    - All three outputs have the same length as input

    Formula
    -------
    1. Fast EMA = EMA(close, fastperiod)
    2. Slow EMA = EMA(close, slowperiod)
    3. MACD Line = Fast EMA - Slow EMA
    4. Signal Line = EMA(MACD Line, signalperiod)
    5. Histogram = MACD Line - Signal Line

    Lookback period: slowperiod + signalperiod - 2
    (For default 12/26/9, lookback = 26 + 9 - 2 = 33)

    Interpretation:
    - MACD > 0: Price is above equilibrium (bullish)
    - MACD < 0: Price is below equilibrium (bearish)
    - MACD crossing above Signal: Bullish signal
    - MACD crossing below Signal: Bearish signal
    - Histogram > 0: MACD above Signal (bullish momentum)
    - Histogram < 0: MACD below Signal (bearish momentum)
    - Histogram increasing: Momentum strengthening
    - Histogram decreasing: Momentum weakening

    Trading Signals:
    - **Crossovers**: MACD crossing Signal line
      - Bullish: MACD crosses above Signal
      - Bearish: MACD crosses below Signal
    - **Centerline Crossovers**: MACD crossing zero line
      - Bullish: MACD crosses above 0
      - Bearish: MACD crosses below 0
    - **Divergences**: Price and MACD moving in opposite directions
      - Bullish: Price makes lower low, MACD makes higher low
      - Bearish: Price makes higher high, MACD makes lower high

    Histogram Analysis:
    - Histogram peak: Maximum momentum, potential reversal ahead
    - Histogram trough: Minimum momentum, potential reversal ahead
    - Histogram approaching zero: Momentum fading
    - Histogram expanding: Momentum accelerating

    Common Settings:
    - Default (12, 26, 9): Standard for daily charts
    - Faster (5, 35, 5): More responsive, more signals
    - Slower (19, 39, 9): Less sensitive, fewer signals
    - Weekly (12, 26, 9): Same periods on weekly charts

    Advantages:
    - Combines trend following and momentum
    - Shows both direction and strength
    - Multiple signal types (crossovers, divergences)
    - Widely recognized and tested
    - Works across different timeframes

    Limitations:
    - Lagging indicator (based on moving averages)
    - Can generate false signals in ranging markets
    - Requires confirmation from other indicators
    - Less effective in choppy conditions

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MACD
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    ...                   110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
    ...                   120, 122, 121, 123, 125, 124, 126, 128, 127, 129,
    ...                   130, 132, 131, 133, 135])
    >>> macd, signal, hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    >>> print(f"MACD: {macd[-1]:.4f}")
    >>> print(f"Signal: {signal[-1]:.4f}")
    >>> print(f"Histogram: {hist[-1]:.4f}")

    See Also
    --------
    MACDEXT : MACD with controllable MA type
    MACDFIX : MACD with fixed 12/26 periods
    EMA : Exponential Moving Average
    """
    # Validate inputs
    if fastperiod < 2:
        raise ValueError("fastperiod must be >= 2")
    if slowperiod < 2:
        raise ValueError("slowperiod must be >= 2")
    if signalperiod < 1:
        raise ValueError("signalperiod must be >= 1")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be < slowperiod")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _macd_cupy(close, fastperiod, slowperiod, signalperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output arrays
        macd = np.empty(n, dtype=np.float64)
        signal = np.empty(n, dtype=np.float64)
        hist = np.empty(n, dtype=np.float64)

        _macd_numba(close, fastperiod, slowperiod, signalperiod, macd, signal, hist)

        return macd, signal, hist


def MACDEXT(close: Union[np.ndarray, list],
            fastperiod: int = 12,
            fastmatype: int = 0,
            slowperiod: int = 26,
            slowmatype: int = 0,
            signalperiod: int = 9,
            signalmatype: int = 0) -> tuple:
    """
    MACD with Controllable MA Type (MACDEXT)

    MACDEXT is an extended version of MACD that allows you to specify different
    types of moving averages for each component (fast MA, slow MA, and signal line).
    This provides flexibility to experiment with different smoothing methods beyond
    the standard exponential moving average.

    Parameters
    ----------
    close : array-like
        Close prices array
    fastperiod : int, optional
        Period for fast MA (default: 12)
    fastmatype : int, optional
        Type of MA for fast line (default: 0)
    slowperiod : int, optional
        Period for slow MA (default: 26)
    slowmatype : int, optional
        Type of MA for slow line (default: 0)
    signalperiod : int, optional
        Period for signal line (default: 9)
    signalmatype : int, optional
        Type of MA for signal line (default: 0)

    MA Types:
        - 0: SMA (Simple Moving Average)
        - 1: EMA (Exponential Moving Average)
        - 2: WMA (Weighted Moving Average) [Not yet implemented]
        - 3: DEMA (Double Exponential Moving Average)
        - 4: TEMA (Triple Exponential Moving Average) [Not yet implemented]
        - 5: TRIMA (Triangular Moving Average) [Not yet implemented]
        - 6: KAMA (Kaufman Adaptive Moving Average)
        - 7: MAMA (Mesa Adaptive Moving Average) [Not yet implemented]
        - 8: T3 (Triple Exponential T3) [Not yet implemented]

    Returns
    -------
    tuple of np.ndarray
        (macd, signal, histogram) - Three arrays with the MACD components

    Notes
    -----
    - Compatible with TA-Lib MACDEXT signature
    - Lookback period varies based on MA types used
    - Default matype=0 (EMA) matches standard MACD
    - Allows mixing different MA types

    Formula
    -------
    1. Fast MA = MA(close, fastperiod, fastmatype)
    2. Slow MA = MA(close, slowperiod, slowmatype)
    3. MACD Line = Fast MA - Slow MA
    4. Signal Line = MA(MACD Line, signalperiod, signalmatype)
    5. Histogram = MACD Line - Signal Line

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MACDEXT
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    ...                   110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
    ...                   120, 122, 121, 123, 125, 124, 126, 128, 127, 129,
    ...                   130, 132, 131, 133, 135])
    >>> # Use DEMA for fast, EMA for slow, SMA for signal
    >>> macd, signal, hist = MACDEXT(close, fastperiod=12, fastmatype=3,
    ...                               slowperiod=26, slowmatype=1,
    ...                               signalperiod=9, signalmatype=0)

    See Also
    --------
    MACD : Standard MACD with EMA
    MACDFIX : MACD with fixed 12/26 periods
    MA : Generic Moving Average function
    """
    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    from .overlap import MA

    backend = get_backend()

    if backend == "gpu":
        # For GPU, use _macd_cupy with MA routing
        # MACDEXT uses MA which will respect the backend setting
        pass

    # Import MA function from overlap module

    # Validate inputs
    if fastperiod < 2:
        raise ValueError("fastperiod must be >= 2")
    if slowperiod < 2:
        raise ValueError("slowperiod must be >= 2")
    if signalperiod < 1:
        raise ValueError("signalperiod must be >= 1")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be < slowperiod")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # Calculate fast and slow MAs
    fast_ma = MA(close, timeperiod=fastperiod, matype=fastmatype)
    slow_ma = MA(close, timeperiod=slowperiod, matype=slowmatype)

    # Calculate MACD line
    macd = fast_ma - slow_ma

    # Calculate signal line (MA of MACD)
    signal = MA(macd, timeperiod=signalperiod, matype=signalmatype)

    # Calculate histogram
    hist = macd - signal

    return macd, signal, hist


def MACDFIX(close: Union[np.ndarray, list], signalperiod: int = 9) -> tuple:
    """
    Moving Average Convergence/Divergence Fix 12/26 (MACDFIX)

    MACDFIX is a simplified version of MACD with fixed periods of 12 and 26 for
    the fast and slow EMAs. Only the signal period can be adjusted. This provides
    a standardized MACD calculation with minimal parameters.

    The "Fix 12/26" refers to the standard MACD periods that have been widely
    used since Gerald Appel introduced the indicator in the 1970s.

    Parameters
    ----------
    close : array-like
        Close prices array
    signalperiod : int, optional
        Period for signal line EMA (default: 9)

    Returns
    -------
    tuple of np.ndarray
        (macd, signal, histogram) - Three arrays with the MACD components

    Notes
    -----
    - Compatible with TA-Lib MACDFIX signature
    - Fixed fastperiod = 12
    - Fixed slowperiod = 26
    - Uses Numba JIT compilation for maximum performance
    - Lookback period: 26 + signalperiod - 2

    Formula
    -------
    1. Fast EMA = EMA(close, 12)
    2. Slow EMA = EMA(close, 26)
    3. MACD Line = Fast EMA - Slow EMA
    4. Signal Line = EMA(MACD Line, signalperiod)
    5. Histogram = MACD Line - Signal Line

    Lookback period: 26 + signalperiod - 2
    (For default signalperiod=9, lookback = 33)

    Interpretation:
    Same as standard MACD - see MACD function documentation for details.

    Advantages:
    - Standardized calculation (12/26 periods)
    - Fewer parameters to tune
    - Widely recognized settings
    - Faster execution (fixed periods)
    - Historical standard since 1970s

    Use Cases:
    - When you want standard MACD settings
    - Backtesting with fixed parameters
    - Comparing across different securities with same settings
    - When you only need to adjust signal period

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MACDFIX
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    ...                   110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
    ...                   120, 122, 121, 123, 125, 124, 126, 128, 127, 129,
    ...                   130, 132, 131, 133, 135])
    >>> macd, signal, hist = MACDFIX(close, signalperiod=9)
    >>> print(f"MACD: {macd[-1]:.4f}")
    >>> print(f"Signal: {signal[-1]:.4f}")
    >>> print(f"Histogram: {hist[-1]:.4f}")

    See Also
    --------
    MACD : MACD with adjustable periods
    MACDEXT : MACD with controllable MA type
    EMA : Exponential Moving Average
    """
    # MACDFIX uses fixed 12/26 periods
    return MACD(close, fastperiod=12, slowperiod=26, signalperiod=signalperiod)


@jit(nopython=True, cache=True)
def _mfi_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
               timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MFI calculation (in-place)

    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price * Volume
    3. Positive/Negative Money Flow based on Typical Price direction
    4. Money Flow Ratio = Sum(Positive MF) / Sum(Negative MF)
    5. MFI = 100 - [100 / (1 + Money Flow Ratio)]
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate typical price and raw money flow
    typical_price = np.empty(n, dtype=np.float64)
    money_flow = np.empty(n, dtype=np.float64)

    for i in range(n):
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0
        money_flow[i] = typical_price[i] * volume[i]

    # Calculate MFI for each window
    for i in range(timeperiod, n):
        positive_mf = 0.0
        negative_mf = 0.0

        for j in range(i - timeperiod + 1, i + 1):
            if j > 0:  # Need previous typical price for comparison
                if typical_price[j] > typical_price[j - 1]:
                    positive_mf += money_flow[j]
                elif typical_price[j] < typical_price[j - 1]:
                    negative_mf += money_flow[j]
                # If equal, don't add to either (neutral)

        # Calculate MFI
        if negative_mf == 0.0:
            # All positive flow
            output[i] = 100.0
        else:
            mf_ratio = positive_mf / negative_mf
            output[i] = 100.0 - (100.0 / (1.0 + mf_ratio))


def MFI(high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        volume: Union[np.ndarray, list],
        timeperiod: int = 14) -> np.ndarray:
    """
    Money Flow Index (MFI)

    The Money Flow Index (MFI) is a volume-weighted momentum indicator that
    measures the strength of money flowing in and out of a security. Often
    called "volume-weighted RSI," it combines price and volume data to identify
    overbought and oversold conditions.

    Developed by Gene Quong and Avrum Soudack, MFI oscillates between 0 and 100,
    with readings above 80 typically indicating overbought conditions and readings
    below 20 indicating oversold conditions.

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
    timeperiod : int, optional
        Number of periods for the calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of MFI values (0-100) with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MFI signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - Values range from 0 to 100
    - All input arrays must have the same length

    Formula
    -------
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price × Volume
    3. Money Flow Direction:
       - Positive MF: When Typical Price > Previous Typical Price
       - Negative MF: When Typical Price < Previous Typical Price
    4. Money Flow Ratio = Sum(Positive MF, timeperiod) / Sum(Negative MF, timeperiod)
    5. MFI = 100 - [100 / (1 + Money Flow Ratio)]

    Or equivalently:
    MFI = 100 × (Positive Money Flow) / (Positive MF + Negative MF)

    Lookback period: timeperiod
    (For timeperiod=14, lookback=14)

    Interpretation:
    - 0-20: Oversold (potential buy signal)
    - 20-80: Normal range
    - 80-100: Overbought (potential sell signal)
    - Rising MFI: Increasing buying pressure
    - Falling MFI: Increasing selling pressure
    - Divergence with price: Potential reversal

    Trading Signals:
    - **Overbought/Oversold**:
      - Buy: MFI crosses above 20 (leaving oversold)
      - Sell: MFI crosses below 80 (leaving overbought)

    - **Divergences**:
      - Bullish: Price makes lower low, MFI makes higher low
      - Bearish: Price makes higher high, MFI makes lower high

    - **Failure Swings**:
      - Bullish: MFI drops below 20, rallies, pulls back above 20, then breaks rally high
      - Bearish: MFI rises above 80, declines, bounces below 80, then breaks decline low

    Advantages:
    - Incorporates volume data
    - Identifies overbought/oversold levels
    - Early warning of reversals
    - Works well with divergence analysis
    - Provides confirmation signals

    Limitations:
    - Can stay overbought/oversold for extended periods
    - False signals in strong trends
    - Requires significant volume data
    - Less effective in low-volume markets

    Common Uses:
    - Overbought/oversold identification
    - Divergence analysis for reversals
    - Confirmation of price trends
    - Volume-based momentum measurement
    - Entry/exit timing
    - Risk management

    Comparison with RSI:
    - MFI includes volume, RSI only uses price
    - MFI can provide earlier signals
    - MFI more sensitive to volume surges
    - Both range from 0 to 100
    - Similar overbought/oversold thresholds

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MFI
    >>> high = np.array([105, 106, 108, 107, 109, 110, 111, 112, 113, 114,
    ...                  115, 116, 117, 118, 119])
    >>> low = np.array([100, 101, 103, 102, 104, 105, 106, 107, 108, 109,
    ...                 110, 111, 112, 113, 114])
    >>> close = np.array([102, 103, 105, 104, 106, 107, 108, 109, 110, 111,
    ...                   112, 113, 114, 115, 116])
    >>> volume = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
    ...                    1800, 1900, 2000, 2100, 2200, 2300, 2400])
    >>> mfi = MFI(high, low, close, volume, timeperiod=14)
    >>> print(f"MFI: {mfi[-1]:.2f}")

    See Also
    --------
    RSI : Relative Strength Index
    ADX : Average Directional Index
    ADOSC : Chaikin A/D Oscillator
    OBV : On Balance Volume

    References
    ----------
    Quong, G. & Soudack, A. "Money Flow Index"
    Technical Analysis of Stocks & Commodities Magazine
    """
    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    # Check all arrays have the same length
    n = len(high)
    if len(low) != n or len(close) != n or len(volume) != n:
        raise ValueError("high, low, close, and volume must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Validate timeperiod
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _mfi_numba(high, low, close, volume, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _minus_dm_numba(high: np.ndarray, low: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MINUS_DM calculation (in-place)

    Formula:
    -DM = Previous Low - Current Low (when this > +DM calculation)
    Otherwise -DM = 0
    Then apply Wilder's smoothing
    """
    n = len(high)

    # Fill lookback period with NaN
    lookback = timeperiod - 1
    for i in range(lookback):
        output[i] = np.nan

    # Calculate raw -DM
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Apply Wilder's smoothing
    smoothed = 0.0
    for i in range(1, timeperiod):
        smoothed += minus_dm[i]
    output[timeperiod - 1] = smoothed

    for i in range(timeperiod, n):
        smoothed = (smoothed * (timeperiod - 1) + minus_dm[i]) / timeperiod
        output[i] = smoothed


def MINUS_DM(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             timeperiod: int = 14) -> np.ndarray:
    """
    Minus Directional Movement (MINUS_DM)

    MINUS_DM is part of the Directional Movement System developed by J. Welles Wilder.
    It measures downward price movement and is used in calculating the ADX indicator.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Period for Wilder's smoothing (default: 14)

    Returns
    -------
    np.ndarray
        Array of smoothed minus directional movement values

    See Also
    --------
    PLUS_DM : Plus Directional Movement
    MINUS_DI : Minus Directional Indicator
    ADX : Average Directional Index
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")
    if n == 0:
        return np.array([], dtype=np.float64)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    output = np.empty(n, dtype=np.float64)
    _minus_dm_numba(high, low, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _minus_di_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MINUS_DI calculation (in-place)

    Formula:
    -DI = 100 * (Smoothed -DM / Smoothed TR)
    """
    n = len(high)

    # Fill lookback period with NaN
    lookback = timeperiod - 1
    for i in range(lookback):
        output[i] = np.nan

    # Calculate TR
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Calculate -DM
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Apply Wilder's smoothing to both
    smoothed_tr = 0.0
    smoothed_dm = 0.0
    for i in range(1, timeperiod):
        smoothed_tr += tr[i]
        smoothed_dm += minus_dm[i]

    if smoothed_tr != 0.0:
        output[timeperiod - 1] = 100.0 * smoothed_dm / smoothed_tr
    else:
        output[timeperiod - 1] = 0.0

    for i in range(timeperiod, n):
        smoothed_tr = (smoothed_tr * (timeperiod - 1) + tr[i]) / timeperiod
        smoothed_dm = (smoothed_dm * (timeperiod - 1) + minus_dm[i]) / timeperiod

        if smoothed_tr != 0.0:
            output[i] = 100.0 * smoothed_dm / smoothed_tr
        else:
            output[i] = 0.0


def MINUS_DI(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             close: Union[np.ndarray, list],
             timeperiod: int = 14) -> np.ndarray:
    """
    Minus Directional Indicator (MINUS_DI)

    MINUS_DI, part of Wilder's Directional Movement System, measures the strength
    of downward price movement. It is calculated as 100 times the ratio of smoothed
    minus directional movement to smoothed true range.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Period for Wilder's smoothing (default: 14)

    Returns
    -------
    np.ndarray
        Array of minus directional indicator values (0-100)

    Notes
    -----
    - Values range from 0 to 100
    - Higher values indicate stronger downward movement
    - Used with PLUS_DI for crossover signals
    - Component of the ADX calculation

    See Also
    --------
    PLUS_DI : Plus Directional Indicator
    MINUS_DM : Minus Directional Movement
    ADX : Average Directional Index
    DX : Directional Movement Index
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")
    if n == 0:
        return np.array([], dtype=np.float64)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    output = np.empty(n, dtype=np.float64)
    _minus_di_numba(high, low, close, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _mom_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MOM calculation (in-place)

    Formula: MOM = Current Price - Price n periods ago
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate momentum
    for i in range(timeperiod, n):
        output[i] = data[i] - data[i - timeperiod]


def MOM(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Momentum (MOM)

    MOM measures the rate of change in price over a specified time period by calculating
    the difference between the current price and the price n periods ago. It is one of
    the most fundamental momentum indicators in technical analysis.

    Positive values indicate upward momentum (price is higher than n periods ago),
    while negative values indicate downward momentum (price is lower than n periods ago).

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for momentum calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of momentum values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MOM signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - Unbounded oscillator (no fixed range)

    Formula
    -------
    MOM[i] = Price[i] - Price[i - timeperiod]

    Lookback period: timeperiod
    (For timeperiod=10, lookback=10)

    Interpretation:
    - Positive MOM: Price rising (bullish momentum)
    - Negative MOM: Price falling (bearish momentum)
    - MOM = 0: No change over period
    - Increasing MOM: Accelerating momentum
    - Decreasing MOM: Decelerating momentum
    - MOM crossing zero: Potential trend change

    Advantages:
    - Simple and intuitive
    - Identifies momentum strength
    - Early trend change signals
    - No lag compared to moving averages
    - Works across all timeframes

    Common Uses:
    - Trend identification
    - Momentum strength measurement
    - Divergence analysis (price vs momentum)
    - Overbought/oversold signals (with thresholds)
    - Confirmation of price movements
    - Filter for other indicators

    Trading Applications:
    - Buy when MOM crosses above zero
    - Sell when MOM crosses below zero
    - Divergence: Price makes new high but MOM doesn't (bearish)
    - Divergence: Price makes new low but MOM doesn't (bullish)
    - Exit when MOM momentum slows (MOM peaks)

    Comparison with Related Indicators:
    - ROC (Rate of Change): MOM expressed as percentage
    - RSI: Bounded version (0-100) with smoothing
    - MACD: Uses moving averages instead of raw prices

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MOM
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> mom = MOM(close, timeperiod=10)
    >>> print(mom)
    >>> # mom[10] = 111 - 100 = 11 (positive momentum)

    See Also
    --------
    ROC : Rate of Change (percentage)
    RSI : Relative Strength Index
    MACD : Moving Average Convergence Divergence
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _mom_numba(data, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _plus_dm_numba(high: np.ndarray, low: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled PLUS_DM calculation (in-place)

    Formula:
    +DM = Current High - Previous High (when this > -DM calculation)
    Otherwise +DM = 0
    Then apply Wilder's smoothing
    """
    n = len(high)

    # Fill lookback period with NaN
    lookback = timeperiod - 1
    for i in range(lookback):
        output[i] = np.nan

    # Calculate raw +DM
    plus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move

    # Apply Wilder's smoothing
    smoothed = 0.0
    for i in range(1, timeperiod):
        smoothed += plus_dm[i]
    output[timeperiod - 1] = smoothed

    for i in range(timeperiod, n):
        smoothed = (smoothed * (timeperiod - 1) + plus_dm[i]) / timeperiod
        output[i] = smoothed


def PLUS_DM(high: Union[np.ndarray, list],
            low: Union[np.ndarray, list],
            timeperiod: int = 14) -> np.ndarray:
    """
    Plus Directional Movement (PLUS_DM)

    PLUS_DM is part of the Directional Movement System developed by J. Welles Wilder.
    It measures upward price movement and is used in calculating the ADX indicator.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Period for Wilder's smoothing (default: 14)

    Returns
    -------
    np.ndarray
        Array of smoothed plus directional movement values

    See Also
    --------
    MINUS_DM : Minus Directional Movement
    PLUS_DI : Plus Directional Indicator
    ADX : Average Directional Index
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")
    if n == 0:
        return np.array([], dtype=np.float64)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    output = np.empty(n, dtype=np.float64)
    _plus_dm_numba(high, low, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _plus_di_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled PLUS_DI calculation (in-place)

    Formula:
    +DI = 100 * (Smoothed +DM / Smoothed TR)
    """
    n = len(high)

    # Fill lookback period with NaN
    lookback = timeperiod - 1
    for i in range(lookback):
        output[i] = np.nan

    # Calculate TR
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Calculate +DM
    plus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move

    # Apply Wilder's smoothing to both
    smoothed_tr = 0.0
    smoothed_dm = 0.0
    for i in range(1, timeperiod):
        smoothed_tr += tr[i]
        smoothed_dm += plus_dm[i]

    if smoothed_tr != 0.0:
        output[timeperiod - 1] = 100.0 * smoothed_dm / smoothed_tr
    else:
        output[timeperiod - 1] = 0.0

    for i in range(timeperiod, n):
        smoothed_tr = (smoothed_tr * (timeperiod - 1) + tr[i]) / timeperiod
        smoothed_dm = (smoothed_dm * (timeperiod - 1) + plus_dm[i]) / timeperiod

        if smoothed_tr != 0.0:
            output[i] = 100.0 * smoothed_dm / smoothed_tr
        else:
            output[i] = 0.0


def PLUS_DI(high: Union[np.ndarray, list],
            low: Union[np.ndarray, list],
            close: Union[np.ndarray, list],
            timeperiod: int = 14) -> np.ndarray:
    """
    Plus Directional Indicator (PLUS_DI)

    PLUS_DI, part of Wilder's Directional Movement System, measures the strength
    of upward price movement. It is calculated as 100 times the ratio of smoothed
    plus directional movement to smoothed true range.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Period for Wilder's smoothing (default: 14)

    Returns
    -------
    np.ndarray
        Array of plus directional indicator values (0-100)

    Notes
    -----
    - Values range from 0 to 100
    - Higher values indicate stronger upward movement
    - Used with MINUS_DI for crossover signals
    - Component of the ADX calculation

    See Also
    --------
    MINUS_DI : Minus Directional Indicator
    PLUS_DM : Plus Directional Movement
    ADX : Average Directional Index
    DX : Directional Movement Index
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")
    if n == 0:
        return np.array([], dtype=np.float64)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    output = np.empty(n, dtype=np.float64)
    _plus_di_numba(high, low, close, timeperiod, output)
    return output


def PPO(data: Union[np.ndarray, list],
        fastperiod: int = 12,
        slowperiod: int = 26,
        matype: int = 0) -> np.ndarray:
    """
    Percentage Price Oscillator (PPO)

    PPO measures the difference between two exponential moving averages (EMAs) as a
    percentage of the larger (slower) EMA. It is similar to MACD but expressed as a
    percentage, making it useful for comparing momentum across securities with different
    price levels.

    Developed by Gerald Appel in the late 1970s (also creator of MACD), PPO provides
    normalized momentum readings that can be compared across different instruments.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    fastperiod : int, optional
        Period for the fast EMA (default: 12)
    slowperiod : int, optional
        Period for the slow EMA (default: 26)
    matype : int, optional
        Moving average type (default: 0 for EMA, currently only EMA supported)

    Returns
    -------
    np.ndarray
        Array of PPO values (percentage difference)

    Notes
    -----
    - Compatible with TA-Lib PPO signature
    - Values are expressed as percentages
    - Unbounded oscillator (can be any percentage)
    - Lookback period depends on slowperiod EMA

    Formula
    -------
    PPO = [(Fast EMA - Slow EMA) / Slow EMA] * 100

    Where:
    - Fast EMA: Exponential Moving Average with fastperiod
    - Slow EMA: Exponential Moving Average with slowperiod

    Standard settings: fastperiod=12, slowperiod=26

    Interpretation:
    - Positive PPO: Fast EMA > Slow EMA (bullish momentum)
    - Negative PPO: Fast EMA < Slow EMA (bearish momentum)
    - PPO = 0: Fast EMA = Slow EMA (neutral)
    - Rising PPO: Increasing bullish momentum
    - Falling PPO: Increasing bearish momentum
    - PPO crossing zero: Potential trend change

    Advantages:
    - Normalized for price level (percentage-based)
    - Comparable across different securities
    - Identifies momentum strength
    - Smoothed compared to raw price changes
    - Less noise than ROC or MOM

    Advantages over MACD:
    - Can compare different securities directly
    - Not affected by absolute price levels
    - Better for multi-security analysis
    - Percentage makes interpretation easier

    Common Uses:
    - Trend identification
    - Momentum strength measurement
    - Signal line crossovers (with 9-period EMA of PPO)
    - Centerline crossovers
    - Divergence analysis
    - Multi-security comparison

    Trading Applications:
    - Buy when PPO crosses above zero
    - Sell when PPO crosses below zero
    - Buy when PPO crosses above signal line
    - Sell when PPO crosses below signal line
    - Divergence: Price makes new high but PPO doesn't (bearish)
    - Divergence: Price makes new low but PPO doesn't (bullish)

    Signal Line:
    Typically, a 9-period EMA of PPO is used as a signal line:
    Signal = EMA(PPO, 9)
    Buy when PPO crosses above signal
    Sell when PPO crosses below signal

    Comparison with Related Indicators:
    - MACD: Absolute difference (not percentage)
    - ROC: Single period percentage change
    - MOM: Absolute price difference
    - RSI: Bounded oscillator (0-100)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import PPO
    >>> close = np.random.randn(100).cumsum() + 100
    >>> ppo = PPO(close, fastperiod=12, slowperiod=26)
    >>> # Positive values indicate upward momentum
    >>> # Signal line: ppo_signal = EMA(ppo, 9)

    See Also
    --------
    MACD : Moving Average Convergence Divergence
    ROC : Rate of Change
    MOM : Momentum
    APO : Absolute Price Oscillator
    """
    # Validate inputs
    if fastperiod < 2:
        raise ValueError("fastperiod must be >= 2")
    if slowperiod < 2:
        raise ValueError("slowperiod must be >= 2")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be < slowperiod")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Import EMA from overlap module
    from .overlap import EMA

    # Calculate EMAs
    fast_ema = EMA(data, timeperiod=fastperiod)
    slow_ema = EMA(data, timeperiod=slowperiod)

    # Calculate PPO
    output = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
            output[i] = np.nan
        elif slow_ema[i] == 0.0:
            output[i] = 0.0
        else:
            output[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100.0

    return output


@jit(nopython=True, cache=True)
def _roc_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROC calculation (in-place)

    Formula: ROC = ((price / prevPrice) - 1) * 100
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROC
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = ((data[i] / prev_price) - 1.0) * 100.0


def ROC(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change (ROC)

    ROC measures the percentage change in price over a specified time period.
    It is a momentum oscillator that oscillates above and below a zero line,
    showing the velocity of price changes.

    ROC is the percentage version of the Momentum (MOM) indicator, making it
    easier to compare across different securities regardless of price level.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROC values (percentage change)

    Notes
    -----
    - Compatible with TA-Lib ROC signature
    - Uses Numba JIT compilation for performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - Unbounded oscillator (no fixed range)

    Formula
    -------
    ROC[i] = ((Price[i] / Price[i - timeperiod]) - 1) * 100

    Equivalent to: ((Price[i] - Price[i - timeperiod]) / Price[i - timeperiod]) * 100

    Lookback period: timeperiod
    (For timeperiod=10, lookback=10)

    Interpretation:
    - Positive ROC: Price rising (bullish momentum)
    - Negative ROC: Price falling (bearish momentum)
    - ROC = 0: No change over period
    - Increasing ROC: Accelerating momentum
    - Decreasing ROC: Decelerating momentum
    - ROC crossing zero: Potential trend change

    Advantages:
    - Normalized for price level (percentage)
    - Comparable across different securities
    - Identifies momentum strength
    - Leading indicator
    - Simple and intuitive

    Common Uses:
    - Trend identification
    - Momentum strength measurement
    - Divergence analysis
    - Overbought/oversold detection
    - Confirmation signals
    - Cross-security comparison

    Trading Applications:
    - Buy when ROC crosses above zero
    - Sell when ROC crosses below zero
    - Divergence: Price makes new high but ROC doesn't (bearish)
    - Divergence: Price makes new low but ROC doesn't (bullish)
    - Overbought: ROC > +threshold (e.g., +10%)
    - Oversold: ROC < -threshold (e.g., -10%)

    Comparison with Related Indicators:
    - MOM: Absolute price difference (not percentage)
    - ROCP: Same as ROC but decimal form
    - ROCR: Ratio form (price/prevPrice)
    - RSI: Bounded version (0-100) with smoothing

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROC
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> roc = ROC(close, timeperiod=10)
    >>> print(roc)
    >>> # roc[10] = ((111 / 100) - 1) * 100 = 11%

    See Also
    --------
    MOM : Momentum
    ROCP : Rate of Change Percentage
    ROCR : Rate of Change Ratio
    RSI : Relative Strength Index
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _roc_numba(data, timeperiod, output)

    return output


@jit(nopython=True, cache=True)
def _rocp_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROCP calculation (in-place)

    Formula: ROCP = (price - prevPrice) / prevPrice
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROCP
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = (data[i] - prev_price) / prev_price


def ROCP(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change Percentage (ROCP)

    ROCP calculates the percentage change in price as a decimal value (not multiplied
    by 100). It is mathematically equivalent to ROC / 100.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROCP values (decimal percentage change)

    Notes
    -----
    - Compatible with TA-Lib ROCP signature
    - Values are decimals (0.10 = 10%)
    - ROCP = ROC / 100

    Formula
    -------
    ROCP[i] = (Price[i] - Price[i - timeperiod]) / Price[i - timeperiod]

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROCP
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> rocp = ROCP(close, timeperiod=10)
    >>> # rocp[10] = (111 - 100) / 100 = 0.11 (11%)

    See Also
    --------
    ROC : Rate of Change (percentage form)
    ROCR : Rate of Change Ratio
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _rocp_numba(data, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _rocr_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROCR calculation (in-place)

    Formula: ROCR = price / prevPrice
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROCR
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = data[i] / prev_price


def ROCR(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change Ratio (ROCR)

    ROCR calculates the ratio of the current price to the price n periods ago.
    Values oscillate around 1.0, where 1.0 represents no change.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROCR values (ratio)

    Notes
    -----
    - Compatible with TA-Lib ROCR signature
    - Values oscillate around 1.0
    - ROCR > 1.0: Price increased
    - ROCR < 1.0: Price decreased
    - ROCR = 1.0: No change

    Formula
    -------
    ROCR[i] = Price[i] / Price[i - timeperiod]

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROCR
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> rocr = ROCR(close, timeperiod=10)
    >>> # rocr[10] = 111 / 100 = 1.11

    See Also
    --------
    ROC : Rate of Change (percentage)
    ROCR100 : Rate of Change Ratio * 100
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _rocr_numba(data, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _rocr100_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled ROCR100 calculation (in-place)

    Formula: ROCR100 = (price / prevPrice) * 100
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate ROCR100
    for i in range(timeperiod, n):
        prev_price = data[i - timeperiod]
        if prev_price == 0.0:
            output[i] = np.nan
        else:
            output[i] = (data[i] / prev_price) * 100.0


def ROCR100(data: Union[np.ndarray, list], timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change Ratio 100 Scale (ROCR100)

    ROCR100 calculates the ratio of the current price to the price n periods ago,
    scaled to 100. Values oscillate around 100, where 100 represents no change.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 10)

    Returns
    -------
    np.ndarray
        Array of ROCR100 values (ratio * 100)

    Notes
    -----
    - Compatible with TA-Lib ROCR100 signature
    - Values oscillate around 100
    - ROCR100 > 100: Price increased
    - ROCR100 < 100: Price decreased
    - ROCR100 = 100: No change
    - ROCR100 = ROCR * 100

    Formula
    -------
    ROCR100[i] = (Price[i] / Price[i - timeperiod]) * 100

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ROCR100
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    >>> rocr100 = ROCR100(close, timeperiod=10)
    >>> # rocr100[10] = (111 / 100) * 100 = 111

    See Also
    --------
    ROCR : Rate of Change Ratio
    ROC : Rate of Change (percentage)
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    _rocr100_numba(data, timeperiod, output)
    return output


@jit(nopython=True, cache=True)
def _rsi_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled RSI calculation (in-place)

    Formula:
    1. Calculate price changes (gains and losses)
    2. Calculate average gain and average loss using Wilder's smoothing
    3. RS = Average Gain / Average Loss
    4. RSI = 100 - (100 / (1 + RS))
    """
    n = len(data)

    # Fill lookback period with NaN
    lookback = timeperiod
    for i in range(lookback):
        output[i] = np.nan

    # Calculate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        change = data[i] - data[i - 1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change

    # Calculate initial average gain and loss (simple average for first period)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, timeperiod + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= timeperiod
    avg_loss /= timeperiod

    # Calculate first RSI value
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            output[timeperiod] = 50.0  # No movement
        else:
            output[timeperiod] = 100.0  # All gains
    else:
        rs = avg_gain / avg_loss
        output[timeperiod] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(timeperiod + 1, n):
        avg_gain = (avg_gain * (timeperiod - 1) + gains[i]) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + losses[i]) / timeperiod

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                output[i] = 50.0
            else:
                output[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            output[i] = 100.0 - (100.0 / (1.0 + rs))


def STOCHF(high: Union[np.ndarray, list],
           low: Union[np.ndarray, list],
           close: Union[np.ndarray, list],
           fastk_period: int = 5,
           fastd_period: int = 3,
           fastd_matype: int = 0) -> tuple:
    """
    Stochastic Fast (STOCHF)

    Stochastic Fast is a momentum oscillator that shows the position of the closing price
    relative to the high-low range over a set period. It consists of two lines:
    %K (fast) and %D (signal line).

    Developed by George Lane in the 1950s, the Stochastic oscillator is based on the
    observation that in uptrends, prices close near the high, and in downtrends, prices
    close near the low.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    fastk_period : int, optional
        Period for %K calculation (default: 5)
    fastd_period : int, optional
        Period for %D (moving average of %K) (default: 3)
    fastd_matype : int, optional
        Type of moving average for %D (default: 0 = SMA)

    Returns
    -------
    tuple of np.ndarray
        (fastk, fastd) - Fast %K and Fast %D arrays

    Notes
    -----
    - Compatible with TA-Lib STOCHF signature
    - Fast version (no smoothing of %K)
    - Values range from 0 to 100
    - %D is moving average of %K

    Formula
    -------
    Fast %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    Fast %D = SMA(Fast %K, fastd_period)

    Where:
    - Lowest Low = Minimum low over fastk_period
    - Highest High = Maximum high over fastk_period

    Interpretation:
    - %K > 80: Overbought
    - %K < 20: Oversold
    - %K crosses above %D: Bullish signal
    - %K crosses below %D: Bearish signal
    - Divergence: Price vs stochastic moving in opposite directions

    See Also
    --------
    STOCH : Stochastic Slow (smoothed version)
    RSI : Relative Strength Index
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    if fastk_period < 1:
        raise ValueError("fastk_period must be >= 1")
    if fastd_period < 1:
        raise ValueError("fastd_period must be >= 1")

    # Calculate Fast %K
    fastk = np.empty(n, dtype=np.float64)

    # Fill lookback period with NaN
    for i in range(fastk_period - 1):
        fastk[i] = np.nan

    # Calculate %K for each period
    for i in range(fastk_period - 1, n):
        # Find highest high and lowest low over period
        highest = high[i]
        lowest = low[i]
        for j in range(i - fastk_period + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]

        # Calculate %K
        if highest - lowest == 0:
            fastk[i] = 50.0  # Avoid division by zero
        else:
            fastk[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0

    # Calculate Fast %D (SMA of Fast %K)
    from .overlap import SMA
    fastd = SMA(fastk, timeperiod=fastd_period)

    return fastk, fastd


def STOCH(high: Union[np.ndarray, list],
          low: Union[np.ndarray, list],
          close: Union[np.ndarray, list],
          fastk_period: int = 5,
          slowk_period: int = 3,
          slowk_matype: int = 0,
          slowd_period: int = 3,
          slowd_matype: int = 0) -> tuple:
    """
    Stochastic (STOCH)

    Stochastic is a momentum oscillator that shows the position of the closing price
    relative to the high-low range over a set period. The slow version applies additional
    smoothing to reduce noise and provide more reliable signals.

    It consists of two lines: %K (slow) and %D (signal line). The slow version smooths
    the fast %K to create slow %K, then smooths that to create slow %D.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    fastk_period : int, optional
        Period for initial %K calculation (default: 5)
    slowk_period : int, optional
        Period for smoothing fast %K to slow %K (default: 3)
    slowk_matype : int, optional
        Type of moving average for slow %K (default: 0 = SMA)
    slowd_period : int, optional
        Period for smoothing slow %K to slow %D (default: 3)
    slowd_matype : int, optional
        Type of moving average for slow %D (default: 0 = SMA)

    Returns
    -------
    tuple of np.ndarray
        (slowk, slowd) - Slow %K and Slow %D arrays

    Notes
    -----
    - Compatible with TA-Lib STOCH signature
    - Slow version (additional smoothing)
    - Values range from 0 to 100
    - Less sensitive than STOCHF
    - Provides more reliable signals

    Formula
    -------
    1. Calculate Fast %K (same as STOCHF)
    2. Slow %K = SMA(Fast %K, slowk_period)
    3. Slow %D = SMA(Slow %K, slowd_period)

    Interpretation:
    - %K > 80: Overbought (consider selling)
    - %K < 20: Oversold (consider buying)
    - %K crosses above %D: Bullish signal (buy)
    - %K crosses below %D: Bearish signal (sell)
    - Divergence: Price makes new high/low but stochastic doesn't

    Trading Signals:
    1. Overbought/Oversold:
       - Buy when %K crosses above 20 from below
       - Sell when %K crosses below 80 from above

    2. Crossovers:
       - Buy when %K crosses above %D
       - Sell when %K crosses below %D

    3. Divergence:
       - Bullish: Price makes lower low, stochastic makes higher low
       - Bearish: Price makes higher high, stochastic makes lower high

    4. Combined:
       - Best signals: Crossover in overbought/oversold zone
       - Buy: %K crosses above %D while both < 20
       - Sell: %K crosses below %D while both > 80

    Parameter Adjustment:
    - fastk_period (default 5):
      - Shorter (3): More sensitive, more signals
      - Longer (14): Less sensitive, smoother

    - slowk_period (default 3):
      - Controls smoothing of %K
      - Longer = smoother, less whipsaw

    Common Settings:
    - Fast Stochastic: (5, 3, 3)
    - Slow Stochastic: (14, 3, 3)
    - Lane's Original: (5, 3, 3)

    Advantages:
    - Clear overbought/oversold levels
    - Bounded (0-100) for easy interpretation
    - Works well in ranging markets
    - Multiple signal types
    - Divergence detection

    Disadvantages:
    - Whipsaw in trending markets
    - Can stay overbought/oversold for long periods
    - Requires confirmation
    - Less effective in strong trends

    Comparison with Related Indicators:
    - STOCHF: Faster, more signals, more noise
    - RSI: Similar concept, different calculation
    - CCI: Unbounded oscillator
    - Williams %R: Inverted stochastic

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import STOCH
    >>> high = np.array([110, 112, 111, 113, 115, 114, 116, 118])
    >>> low = np.array([100, 102, 101, 103, 105, 104, 106, 108])
    >>> close = np.array([105, 107, 106, 108, 110, 109, 111, 113])
    >>> slowk, slowd = STOCH(high, low, close)
    >>> # slowk and slowd range from 0 to 100

    See Also
    --------
    STOCHF : Stochastic Fast
    RSI : Relative Strength Index
    CCI : Commodity Channel Index
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    if fastk_period < 1:
        raise ValueError("fastk_period must be >= 1")
    if slowk_period < 1:
        raise ValueError("slowk_period must be >= 1")
    if slowd_period < 1:
        raise ValueError("slowd_period must be >= 1")

    # Calculate Fast %K directly
    fastk = np.empty(n, dtype=np.float64)
    for i in range(fastk_period - 1):
        fastk[i] = np.nan
    for i in range(fastk_period - 1, n):
        highest = high[i]
        lowest = low[i]
        for j in range(i - fastk_period + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        if highest - lowest == 0:
            fastk[i] = 50.0
        else:
            fastk[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0

    # Smooth Fast %K to get Slow %K
    from .overlap import SMA
    slowk = SMA(fastk, timeperiod=slowk_period)

    # Smooth Slow %K to get Slow %D
    slowd = SMA(slowk, timeperiod=slowd_period)

    return slowk, slowd


def STOCHRSI(data: Union[np.ndarray, list],
             timeperiod: int = 14,
             fastk_period: int = 5,
             fastd_period: int = 3,
             fastd_matype: int = 0) -> tuple:
    """
    Stochastic Relative Strength Index (STOCHRSI)

    STOCHRSI applies the Stochastic oscillator formula to RSI values instead of price.
    It measures the level of RSI relative to its high-low range over a set period,
    providing a more sensitive momentum indicator that oscillates between 0 and 100.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Period for RSI calculation (default: 14)
    fastk_period : int, optional
        Period for Stochastic %K calculation (default: 5)
    fastd_period : int, optional
        Period for %D (moving average of %K) (default: 3)
    fastd_matype : int, optional
        Type of moving average for %D (default: 0 = SMA)

    Returns
    -------
    tuple of np.ndarray
        (fastk, fastd) - StochRSI %K and %D arrays

    Notes
    -----
    - More sensitive than regular Stochastic
    - Combines RSI and Stochastic concepts
    - Values range from 0 to 100
    - Prone to more whipsaw signals

    Formula
    -------
    1. Calculate RSI(timeperiod)
    2. Apply Stochastic formula to RSI:
       StochRSI %K = ((RSI - Lowest RSI) / (Highest RSI - Lowest RSI)) * 100
    3. StochRSI %D = SMA(StochRSI %K, fastd_period)

    See Also
    --------
    RSI : Relative Strength Index
    STOCH : Stochastic Oscillator
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    # Calculate RSI
    rsi = RSI(data, timeperiod=timeperiod)

    # Apply Stochastic formula to RSI
    fastk = np.empty(n, dtype=np.float64)

    lookback = fastk_period - 1
    for i in range(lookback):
        fastk[i] = np.nan

    for i in range(lookback, n):
        # Find highest and lowest RSI over fastk_period
        if i < timeperiod + lookback:
            fastk[i] = np.nan
            continue

        highest_rsi = -np.inf
        lowest_rsi = np.inf
        valid_count = 0

        for j in range(i - fastk_period + 1, i + 1):
            if not np.isnan(rsi[j]):
                highest_rsi = max(highest_rsi, rsi[j])
                lowest_rsi = min(lowest_rsi, rsi[j])
                valid_count += 1

        if valid_count == 0 or highest_rsi == lowest_rsi:
            fastk[i] = 50.0
        else:
            fastk[i] = ((rsi[i] - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100.0

    # Calculate %D (SMA of %K)
    from .overlap import SMA
    fastd = SMA(fastk, timeperiod=fastd_period)

    return fastk, fastd


def TRIX(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (TRIX)

    TRIX shows the percent rate of change of a triple exponentially smoothed
    moving average. It was developed by Jack Hutson in the early 1980s as a
    filtered momentum oscillator.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Period for EMA calculations (default: 30)

    Returns
    -------
    np.ndarray
        Array of TRIX values (percentage)

    Notes
    -----
    - Shows momentum of triple-smoothed EMA
    - Filters out insignificant price movements
    - Values around zero indicate no trend
    - Positive values indicate bullish momentum
    - Negative values indicate bearish momentum
    - Compatible with TA-Lib TRIX signature

    Formula
    -------
    1. EMA1 = EMA(data, timeperiod)
    2. EMA2 = EMA(EMA1, timeperiod)
    3. EMA3 = EMA(EMA2, timeperiod)
    4. TRIX = 1-period ROC of EMA3
         TRIX = ((EMA3[i] - EMA3[i-1]) / EMA3[i-1]) * 100

    Interpretation:
    - TRIX > 0: Bullish momentum
    - TRIX < 0: Bearish momentum
    - TRIX crossing zero: Trend change
    - Divergence: Potential reversal signal
    - Rising TRIX: Strengthening uptrend
    - Falling TRIX: Strengthening downtrend

    Trading Signals:
    - Buy: TRIX crosses above zero
    - Sell: TRIX crosses below zero
    - Buy: TRIX crosses above signal line
    - Sell: TRIX crosses below signal line
    - Bullish divergence: Price lower low, TRIX higher low
    - Bearish divergence: Price higher high, TRIX lower high

    Advantages:
    - Triple smoothing filters noise
    - Fewer whipsaw signals
    - Good for trend identification
    - Works well in trending markets

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import TRIX
    >>> close = np.linspace(100, 120, 100)
    >>> trix = TRIX(close, timeperiod=15)

    See Also
    --------
    EMA : Exponential Moving Average
    ROC : Rate of Change
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n == 0:
        return np.array([], dtype=np.float64)

    # Import EMA from overlap
    from .overlap import EMA

    # Calculate triple EMA
    ema1 = EMA(data, timeperiod=timeperiod)
    ema2 = EMA(ema1, timeperiod=timeperiod)
    ema3 = EMA(ema2, timeperiod=timeperiod)

    # Calculate 1-period ROC of triple EMA
    output = np.empty(n, dtype=np.float64)
    output[0] = np.nan

    for i in range(1, n):
        if np.isnan(ema3[i]) or np.isnan(ema3[i - 1]) or ema3[i - 1] == 0.0:
            output[i] = np.nan
        else:
            output[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100.0

    return output


def ULTOSC(high: Union[np.ndarray, list],
           low: Union[np.ndarray, list],
           close: Union[np.ndarray, list],
           timeperiod1: int = 7,
           timeperiod2: int = 14,
           timeperiod3: int = 28) -> np.ndarray:
    """
    Ultimate Oscillator (ULTOSC)

    The Ultimate Oscillator was developed by Larry Williams in 1976 to measure
    momentum across three different timeframes. It addresses the problem of
    traditional oscillators using only one timeframe.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod1 : int, optional
        First (short) period (default: 7)
    timeperiod2 : int, optional
        Second (medium) period (default: 14)
    timeperiod3 : int, optional
        Third (long) period (default: 28)

    Returns
    -------
    np.ndarray
        Array of Ultimate Oscillator values (0-100 range)

    Notes
    -----
    - Combines three timeframes for better signals
    - Oscillates between 0 and 100
    - Overbought: > 70
    - Oversold: < 30
    - Compatible with TA-Lib ULTOSC signature

    Formula
    -------
    For each period:
    1. Buying Pressure (BP) = Close - True Low
       where True Low = min(Low, Previous Close)

    2. True Range (TR) = True High - True Low
       where True High = max(High, Previous Close)
              True Low = min(Low, Previous Close)

    3. Average = sum(BP over period) / sum(TR over period)

    4. Ultimate Oscillator = 100 * [(4*Avg7 + 2*Avg14 + Avg28) / (4 + 2 + 1)]

    Interpretation:
    - Range: 0 to 100
    - > 70: Overbought condition
    - < 30: Oversold condition
    - 50: Neutral
    - Rising: Bullish momentum
    - Falling: Bearish momentum

    Trading Signals:
    - Buy: Bullish divergence below 30
    - Sell: Bearish divergence above 70
    - Buy: Cross above 30 from below
    - Sell: Cross below 70 from above
    - Confirmation: Use with trend indicators

    Divergence Signals:
    - Bullish: Price makes lower low, UO higher low
    - Bearish: Price makes higher high, UO lower high
    - Most reliable when UO in extreme zones

    Advantages:
    - Multiple timeframes reduce whipsaws
    - Good for divergence trading
    - Objective overbought/oversold levels
    - Works in trending and ranging markets

    Larry Williams' Original Rules:
    1. Bullish divergence forms below 30
    2. UO rises above 50
    3. UO pulls back but stays above 30
    4. Buy on break above pullback high

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import ULTOSC
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> close = np.array([105, 107, 109, 108, 110, 112, 111])
    >>> ultosc = ULTOSC(high, low, close)

    See Also
    --------
    RSI : Relative Strength Index
    STOCH : Stochastic Oscillator
    MFI : Money Flow Index
    """
    # Validate inputs
    if timeperiod1 < 1 or timeperiod2 < 1 or timeperiod3 < 1:
        raise ValueError("All timeperiods must be >= 1")

    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Calculate Buying Pressure and True Range
    bp = np.empty(n, dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)

    # First bar
    bp[0] = close[0] - low[0]
    tr[0] = high[0] - low[0]

    # Subsequent bars
    for i in range(1, n):
        true_low = min(low[i], close[i - 1])
        true_high = max(high[i], close[i - 1])

        bp[i] = close[i] - true_low
        tr[i] = true_high - true_low

    # Calculate output
    output = np.empty(n, dtype=np.float64)

    # Determine lookback period (need data for longest period)
    lookback = max(timeperiod1, timeperiod2, timeperiod3) - 1

    for i in range(lookback):
        output[i] = np.nan

    # Calculate Ultimate Oscillator for each bar
    for i in range(lookback, n):
        # Calculate averages for each timeframe
        # Avg = Sum(BP) / Sum(TR) over period

        # Period 1 (shortest)
        if i >= timeperiod1 - 1:
            sum_bp1 = 0.0
            sum_tr1 = 0.0
            for j in range(i - timeperiod1 + 1, i + 1):
                sum_bp1 += bp[j]
                sum_tr1 += tr[j]
            avg1 = sum_bp1 / sum_tr1 if sum_tr1 != 0 else 0.0
        else:
            output[i] = np.nan
            continue

        # Period 2 (medium)
        if i >= timeperiod2 - 1:
            sum_bp2 = 0.0
            sum_tr2 = 0.0
            for j in range(i - timeperiod2 + 1, i + 1):
                sum_bp2 += bp[j]
                sum_tr2 += tr[j]
            avg2 = sum_bp2 / sum_tr2 if sum_tr2 != 0 else 0.0
        else:
            output[i] = np.nan
            continue

        # Period 3 (longest)
        if i >= timeperiod3 - 1:
            sum_bp3 = 0.0
            sum_tr3 = 0.0
            for j in range(i - timeperiod3 + 1, i + 1):
                sum_bp3 += bp[j]
                sum_tr3 += tr[j]
            avg3 = sum_bp3 / sum_tr3 if sum_tr3 != 0 else 0.0
        else:
            output[i] = np.nan
            continue

        # Ultimate Oscillator = 100 * [(4*Avg1 + 2*Avg2 + Avg3) / 7]
        output[i] = 100.0 * ((4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0)

    return output


@jit(nopython=True, cache=True)
def _willr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 timeperiod: int, output: np.ndarray) -> None:
    """Numba-compiled Williams %R calculation"""
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate Williams %R for each window
    for i in range(timeperiod - 1, n):
        # Find highest high and lowest low in window
        highest = high[i]
        lowest = low[i]

        for j in range(i - timeperiod + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]

        # Calculate Williams %R
        if highest == lowest:
            output[i] = -50.0  # Neutral when range is zero
        else:
            output[i] = ((highest - close[i]) / (highest - lowest)) * -100.0


def WILLR(high: Union[np.ndarray, list],
          low: Union[np.ndarray, list],
          close: Union[np.ndarray, list],
          timeperiod: int = 14) -> np.ndarray:
    """
    Williams' %R (WILLR)

    Williams %R is a momentum indicator created by Larry Williams that measures
    overbought and oversold levels. It is the inverse of the Fast Stochastic
    Oscillator.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for %R calculation (default: 14)

    Returns
    -------
    np.ndarray
        Array of Williams %R values (range: 0 to -100)

    Notes
    -----
    - Range: 0 to -100
    - Overbought: 0 to -20
    - Oversold: -80 to -100
    - Compatible with TA-Lib WILLR signature
    - Inverse of Fast Stochastic %K

    Formula
    -------
    %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

    Where:
    - Highest High = Highest high over the lookback period
    - Lowest Low = Lowest low over the lookback period

    Interpretation:
    - 0 to -20: Overbought (potential sell signal)
    - -80 to -100: Oversold (potential buy signal)
    - -50: Neutral/midpoint
    - Rising %R: Strengthening momentum
    - Falling %R: Weakening momentum

    Trading Signals:
    - Buy: %R crosses above -80 from below
    - Sell: %R crosses below -20 from above
    - Buy: Bullish divergence at oversold levels
    - Sell: Bearish divergence at overbought levels

    Relationship to Stochastic:
    - Williams %R = Fast Stochastic %K * -1 - 100
    - Both use same high/low/close data
    - Different scaling: Stoch (0-100), %R (0 to -100)
    - Same interpretation, opposite scale

    Larry Williams' Original:
    - Larry Williams originally used 10-day period
    - Modern default is 14 periods
    - Can be used on any timeframe
    - Works best in ranging markets

    Advantages:
    - Clear overbought/oversold levels
    - Easy to interpret
    - Works on all timeframes
    - Good for divergence trading
    - Fewer parameters than Stochastic

    Disadvantages:
    - Can stay overbought/oversold in trends
    - Whipsaw signals in choppy markets
    - Best combined with trend indicators
    - Less smoothing than Slow Stochastic

    Common Periods:
    - 10: Original Larry Williams setting
    - 14: Modern default (more common)
    - 20: Slower, fewer signals
    - 7: Faster, more signals

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import WILLR
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> close = np.array([105, 107, 109, 108, 110, 112, 111])
    >>> willr = WILLR(high, low, close, timeperiod=14)

    See Also
    --------
    STOCHF : Fast Stochastic Oscillator
    STOCH : Slow Stochastic Oscillator
    RSI : Relative Strength Index
    CCI : Commodity Channel Index
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(high)
    if len(low) != n or len(close) != n:
        raise ValueError("high, low, and close must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _willr_numba(high, low, close, timeperiod, output)

    return output