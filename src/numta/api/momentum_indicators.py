"""
Momentum Indicators - Indicators that measure the rate of price change
"""

"""Public API for momentum_indicators"""

import numpy as np
from typing import Union

# Import backend implementations
from ..cpu.momentum_indicators import *
from ..backend import get_backend


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
    >>> from numta import ADX
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
    >>> from numta import ADXR
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
    >>> from numta import APO
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
    >>> from numta import AROON
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
    >>> from numta import AROONOSC
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

    # Use Numba-optimized implementation
    output = np.empty(n, dtype=np.float64)
    _aroonosc_numba(high, low, timeperiod, output)

    return output


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
    >>> from numta import ATR
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
    >>> from numta import BOP
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

    # Use Numba-optimized implementation
    output = np.empty(n, dtype=np.float64)
    _bop_numba(open_price, high, low, close, output)

    return output


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
    >>> from numta import CCI
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
    >>> from numta import ROC
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
    >>> from numta import ROCP
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
    >>> from numta import ROCR
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
    >>> from numta import ROCR100
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
    >>> from numta import RSI
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

    backend = get_backend()

    # Use CPU implementation (default)
    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _rsi_numba(data, timeperiod, output)

    return output


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
    >>> from numta import CMO
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

    backend = get_backend()

    # Use CPU implementation (default)
    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _cmo_numba(close, timeperiod, output)

    return output


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
    >>> from numta import DX
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

    backend = get_backend()

    # Use CPU implementation (default)
    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _dx_numba(high, low, close, timeperiod, output)

    return output


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
    >>> from numta import MACD
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

    backend = get_backend()

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
    >>> from numta import MACDEXT
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
    from .overlap import MA

    backend = get_backend()


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
    >>> from numta import TRIX
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

    # Calculate 1-period ROC of triple EMA using Numba-optimized implementation
    from ..cpu.momentum_indicators import _trix_numba
    output = np.empty(n, dtype=np.float64)
    _trix_numba(ema3, output)

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
    >>> from numta import ULTOSC
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

    # Calculate Ultimate Oscillator using Numba-optimized implementation
    from ..cpu.momentum_indicators import _ultosc_numba
    output = np.empty(n, dtype=np.float64)
    _ultosc_numba(bp, tr, timeperiod1, timeperiod2, timeperiod3, output)

    return output


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
    >>> from numta import WILLR
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


def MACDFIX(close: Union[np.ndarray, list],
            signalperiod: int = 9) -> tuple:
    """
    Moving Average Convergence/Divergence Fix (MACDFIX)

    MACDFIX is a variant of MACD with fixed fast and slow periods (12 and 26).
    This optimized version provides faster computation by hard-coding the
    standard MACD periods while still allowing customization of the signal period.

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
    - Uses Numba JIT compilation for maximum performance
    - Fixed fast period: 12
    - Fixed slow period: 26
    - Lookback period: 26 + signalperiod - 2
    - All three outputs have the same length as input

    Formula
    -------
    1. Fast EMA = EMA(close, 12)
    2. Slow EMA = EMA(close, 26)
    3. MACD Line = Fast EMA - Slow EMA
    4. Signal Line = EMA(MACD Line, signalperiod)
    5. Histogram = MACD Line - Signal Line

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MACDFIX
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    ...                   110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
    ...                   120, 122, 121, 123, 125, 124, 126, 128, 127, 129,
    ...                   130, 132, 131, 133, 135])
    >>> macd, signal, hist = MACDFIX(close, signalperiod=9)

    See Also
    --------
    MACD : MACD with customizable fast/slow periods
    MACDEXT : MACD with controllable MA type
    """
    # Validate inputs
    if signalperiod < 1:
        raise ValueError("signalperiod must be >= 1")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # Pre-allocate output arrays
    macd = np.empty(n, dtype=np.float64)
    signal = np.empty(n, dtype=np.float64)
    hist = np.empty(n, dtype=np.float64)

    _macdfix_numba(close, signalperiod, macd, signal, hist)

    return macd, signal, hist


def MFI(high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        volume: Union[np.ndarray, list],
        timeperiod: int = 14) -> np.ndarray:
    """
    Money Flow Index (MFI)

    The Money Flow Index is a momentum indicator that uses both price and volume
    to measure buying and selling pressure. Often called the "volume-weighted RSI",
    MFI oscillates between 0 and 100 and is used to identify overbought and
    oversold conditions.

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
        Number of periods for the indicator (default: 14)

    Returns
    -------
    np.ndarray
        Array of MFI values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MFI signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - MFI values range from 0 to 100
    - MFI > 80 typically indicates overbought conditions
    - MFI < 20 typically indicates oversold conditions

    Formula
    -------
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price * Volume
    3. Money Flow is positive when Typical Price increases, negative when it decreases
    4. Money Flow Ratio = (Sum of Positive Money Flow) / (Sum of Negative Money Flow) over timeperiod
    5. MFI = 100 - [100 / (1 + Money Flow Ratio)]

    Interpretation:
    - MFI > 80: Overbought (potential sell signal)
    - MFI < 20: Oversold (potential buy signal)
    - Divergences between MFI and price indicate potential reversals

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MFI
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> close = np.array([105, 107, 109, 108, 110, 112, 111])
    >>> volume = np.array([1000, 1200, 1100, 1300, 1400, 1250, 1350])
    >>> mfi = MFI(high, low, close, volume, timeperiod=14)

    See Also
    --------
    RSI : Relative Strength Index
    OBV : On Balance Volume
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays with float64 dtype and ensure contiguous memory layout
    # This prevents potential segfaults in numba JIT-compiled functions
    high = np.ascontiguousarray(np.asarray(high, dtype=np.float64))
    low = np.ascontiguousarray(np.asarray(low, dtype=np.float64))
    close = np.ascontiguousarray(np.asarray(close, dtype=np.float64))
    volume = np.ascontiguousarray(np.asarray(volume, dtype=np.float64))

    n = len(high)
    if len(low) != n or len(close) != n or len(volume) != n:
        raise ValueError("high, low, close, and volume must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _mfi_numba(high, low, close, volume, timeperiod, output)

    return output


def MINUS_DI(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             close: Union[np.ndarray, list],
             timeperiod: int = 14) -> np.ndarray:
    """
    Minus Directional Indicator (MINUS_DI)

    The Minus Directional Indicator is a component of the Directional Movement
    System developed by J. Welles Wilder. It measures the strength of downward
    price movement and is used in conjunction with PLUS_DI to determine trend
    direction and strength.

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
        Array of Minus Directional Indicator values

    Notes
    -----
    - Compatible with TA-Lib MINUS_DI signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Values range from 0 to 100
    - Higher values indicate stronger downward movement

    Formula
    -------
    1. Calculate True Range (TR) and Minus Directional Movement (-DM)
    2. Apply Wilder's smoothing to both TR and -DM
    3. -DI = 100 * (Smoothed -DM / Smoothed TR)

    Interpretation:
    - -DI > +DI: Downtrend dominates
    - -DI < +DI: Uptrend dominates
    - Use with +DI and ADX for complete trend analysis

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MINUS_DI
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> close = np.array([105, 107, 109, 108, 110, 112, 111])
    >>> minus_di = MINUS_DI(high, low, close, timeperiod=14)

    See Also
    --------
    PLUS_DI : Plus Directional Indicator
    ADX : Average Directional Movement Index
    DX : Directional Movement Index
    """
    # Validate inputs
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
    _minus_di_numba(high, low, close, timeperiod, output)

    return output


def MINUS_DM(high: Union[np.ndarray, list],
             low: Union[np.ndarray, list],
             timeperiod: int = 14) -> np.ndarray:
    """
    Minus Directional Movement (MINUS_DM)

    The Minus Directional Movement is a component of the Directional Movement
    System developed by J. Welles Wilder. It represents the smoothed value of
    downward price movement.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Number of periods for smoothing (default: 14)

    Returns
    -------
    np.ndarray
        Array of smoothed Minus Directional Movement values

    Notes
    -----
    - Compatible with TA-Lib MINUS_DM signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Uses Wilder's smoothing method

    Formula
    -------
    1. Raw -DM = Previous Low - Current Low (when down move > up move and > 0)
    2. Otherwise -DM = 0
    3. Apply Wilder's smoothing over timeperiod

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MINUS_DM
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> minus_dm = MINUS_DM(high, low, timeperiod=14)

    See Also
    --------
    PLUS_DM : Plus Directional Movement
    MINUS_DI : Minus Directional Indicator
    ADX : Average Directional Movement Index
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _minus_dm_numba(high, low, timeperiod, output)

    return output


def MOM(real: Union[np.ndarray, list],
        timeperiod: int = 10) -> np.ndarray:
    """
    Momentum (MOM)

    Momentum is a simple and direct measure of the rate of price change. It
    calculates the difference between the current price and the price n periods
    ago. Positive values indicate upward momentum, while negative values indicate
    downward momentum.

    Parameters
    ----------
    real : array-like
        Array of real values (typically close prices)
    timeperiod : int, optional
        Number of periods to look back (default: 10)

    Returns
    -------
    np.ndarray
        Array of momentum values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MOM signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Unbounded indicator (can be any positive or negative value)

    Formula
    -------
    MOM = Current Price - Price n periods ago

    Interpretation:
    - MOM > 0: Upward momentum (price increased)
    - MOM < 0: Downward momentum (price decreased)
    - MOM = 0: No momentum (price unchanged)
    - Increasing MOM: Accelerating trend
    - Decreasing MOM: Decelerating trend

    Trading Signals:
    - Crossover above 0: Bullish signal
    - Crossover below 0: Bearish signal
    - Divergences indicate potential reversals

    Examples
    --------
    >>> import numpy as np
    >>> from numta import MOM
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110])
    >>> mom = MOM(close, timeperiod=10)

    See Also
    --------
    ROC : Rate of Change (percentage-based momentum)
    RSI : Relative Strength Index
    MACD : Moving Average Convergence/Divergence
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    # Convert to numpy array
    real = np.asarray(real, dtype=np.float64)

    n = len(real)
    if n == 0:
        return np.array([], dtype=np.float64)

    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _mom_numba(real, timeperiod, output)

    return output


def PLUS_DI(high: Union[np.ndarray, list],
            low: Union[np.ndarray, list],
            close: Union[np.ndarray, list],
            timeperiod: int = 14) -> np.ndarray:
    """
    Plus Directional Indicator (PLUS_DI)

    The Plus Directional Indicator is a component of the Directional Movement
    System developed by J. Welles Wilder. It measures the strength of upward
    price movement and is used in conjunction with MINUS_DI to determine trend
    direction and strength.

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
        Array of Plus Directional Indicator values

    Notes
    -----
    - Compatible with TA-Lib PLUS_DI signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Values range from 0 to 100
    - Higher values indicate stronger upward movement

    Formula
    -------
    1. Calculate True Range (TR) and Plus Directional Movement (+DM)
    2. Apply Wilder's smoothing to both TR and +DM
    3. +DI = 100 * (Smoothed +DM / Smoothed TR)

    Interpretation:
    - +DI > -DI: Uptrend dominates
    - +DI < -DI: Downtrend dominates
    - Use with -DI and ADX for complete trend analysis

    Examples
    --------
    >>> import numpy as np
    >>> from numta import PLUS_DI
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> close = np.array([105, 107, 109, 108, 110, 112, 111])
    >>> plus_di = PLUS_DI(high, low, close, timeperiod=14)

    See Also
    --------
    MINUS_DI : Minus Directional Indicator
    ADX : Average Directional Movement Index
    DX : Directional Movement Index
    """
    # Validate inputs
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
    _plus_di_numba(high, low, close, timeperiod, output)

    return output


def PLUS_DM(high: Union[np.ndarray, list],
            low: Union[np.ndarray, list],
            timeperiod: int = 14) -> np.ndarray:
    """
    Plus Directional Movement (PLUS_DM)

    The Plus Directional Movement is a component of the Directional Movement
    System developed by J. Welles Wilder. It represents the smoothed value of
    upward price movement.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    timeperiod : int, optional
        Number of periods for smoothing (default: 14)

    Returns
    -------
    np.ndarray
        Array of smoothed Plus Directional Movement values

    Notes
    -----
    - Compatible with TA-Lib PLUS_DM signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Uses Wilder's smoothing method

    Formula
    -------
    1. Raw +DM = Current High - Previous High (when up move > down move and > 0)
    2. Otherwise +DM = 0
    3. Apply Wilder's smoothing over timeperiod

    Examples
    --------
    >>> import numpy as np
    >>> from numta import PLUS_DM
    >>> high = np.array([110, 112, 114, 113, 115, 117, 116])
    >>> low = np.array([100, 102, 104, 103, 105, 107, 106])
    >>> plus_dm = PLUS_DM(high, low, timeperiod=14)

    See Also
    --------
    MINUS_DM : Minus Directional Movement
    PLUS_DI : Plus Directional Indicator
    ADX : Average Directional Movement Index
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _plus_dm_numba(high, low, timeperiod, output)

    return output


def PPO(close: Union[np.ndarray, list],
        fastperiod: int = 12,
        slowperiod: int = 26,
        matype: int = 0) -> np.ndarray:
    """
    Percentage Price Oscillator (PPO)

    The Percentage Price Oscillator is a momentum oscillator that measures the
    difference between two moving averages as a percentage of the larger moving
    average. It is similar to MACD but expressed in percentage terms, making it
    more suitable for comparing securities with different price levels.

    Parameters
    ----------
    close : array-like
        Close prices array
    fastperiod : int, optional
        Period for fast moving average (default: 12)
    slowperiod : int, optional
        Period for slow moving average (default: 26)
    matype : int, optional
        Type of moving average (default: 0 = EMA)
        Currently only EMA (0) is supported

    Returns
    -------
    np.ndarray
        Array of PPO values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib PPO signature
    - Uses Numba JIT compilation for maximum performance
    - Lookback period: slowperiod - 1
    - Unbounded indicator (can be any percentage value)
    - More suitable than MACD for comparing different securities

    Formula
    -------
    PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100

    Where:
    - Fast EMA = EMA(close, fastperiod)
    - Slow EMA = EMA(close, slowperiod)

    Interpretation:
    - PPO > 0: Fast MA above slow MA (bullish)
    - PPO < 0: Fast MA below slow MA (bearish)
    - Rising PPO: Increasing bullish momentum
    - Falling PPO: Increasing bearish momentum

    Trading Signals:
    - Crossover above 0: Bullish signal
    - Crossover below 0: Bearish signal
    - Divergences indicate potential reversals

    Advantages over MACD:
    - Percentage-based allows comparison across securities
    - Not affected by absolute price levels
    - Same interpretation regardless of price scale

    Examples
    --------
    >>> import numpy as np
    >>> from numta import PPO
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    ...                   110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
    ...                   120, 122, 121, 123, 125, 124, 126, 128, 127, 129])
    >>> ppo = PPO(close, fastperiod=12, slowperiod=26)

    See Also
    --------
    MACD : Moving Average Convergence/Divergence
    APO : Absolute Price Oscillator
    EMA : Exponential Moving Average
    """
    # Validate inputs
    if fastperiod < 2:
        raise ValueError("fastperiod must be >= 2")
    if slowperiod < 2:
        raise ValueError("slowperiod must be >= 2")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be < slowperiod")
    if matype != 0:
        raise ValueError("Currently only matype=0 (EMA) is supported")

    # Convert to numpy array
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Import EMA from overlap module
    from ..api.overlap import EMA

    # Calculate fast and slow EMAs
    fast_ema = EMA(close, fastperiod)
    slow_ema = EMA(close, slowperiod)

    # Pre-allocate output array
    output = np.empty(n, dtype=np.float64)

    # Calculate PPO using numba function
    _ppo_numba(fast_ema, slow_ema, output)

    return output



