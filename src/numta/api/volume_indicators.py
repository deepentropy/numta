"""
Volume Indicators - Indicators based on volume data
"""

import numpy as np
from typing import Union

# Import CPU implementations
from ..cpu.volume_indicators import _ad_numba, _adosc_numba, _obv_numba


def AD(high: Union[np.ndarray, list],
       low: Union[np.ndarray, list],
       close: Union[np.ndarray, list],
       volume: Union[np.ndarray, list]) -> np.ndarray:
    """
    Chaikin A/D Line (Accumulation/Distribution Line)

    The Chaikin Accumulation/Distribution Line is a volume-based indicator
    designed to measure the cumulative flow of money into and out of a security.
    It relates the closing price to the high-low range and multiplies this by volume.

    The indicator attempts to determine whether a security is being accumulated
    (bought) or distributed (sold) by comparing the close price position within
    the period's range, weighted by volume.

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

    Returns
    -------
    np.ndarray
        Array of Chaikin A/D Line values

    Notes
    -----
    - Compatible with TA-Lib AD signature
    - Uses Numba JIT compilation for maximum performance
    - When high == low, the money flow multiplier is 0 (no change to AD line)

    Formula
    -------
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
                          = (2 * Close - High - Low) / (High - Low)
    Money Flow Volume = Money Flow Multiplier * Volume
    AD Line[i] = AD Line[i-1] + Money Flow Volume[i]

    The multiplier ranges from -1 to +1:
    - +1: Close = High (strong buying pressure)
    - 0: Close = Mid-point of range (neutral)
    - -1: Close = Low (strong selling pressure)

    Examples
    --------
    >>> import numpy as np
    >>> from numta import AD
    >>> high = np.array([10, 11, 12, 11, 13])
    >>> low = np.array([9, 10, 10, 9, 11])
    >>> close = np.array([9.5, 10.5, 11, 10, 12])
    >>> volume = np.array([1000, 1100, 1200, 900, 1300])
    >>> ad = AD(high, low, close, volume)
    >>> print(ad)
    """
    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    # Validate input lengths
    n = len(high)
    if len(low) != n or len(close) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _ad_numba(high, low, close, volume, output)

    return output


def ADOSC(high: Union[np.ndarray, list],
          low: Union[np.ndarray, list],
          close: Union[np.ndarray, list],
          volume: Union[np.ndarray, list],
          fastperiod: int = 3,
          slowperiod: int = 10) -> np.ndarray:
    """
    Chaikin A/D Oscillator

    The Chaikin A/D Oscillator is a momentum indicator that measures the
    accumulation-distribution line of a moving average convergence-divergence (MACD).
    It takes the difference between a 3-day and 10-day exponential moving average
    of the Accumulation/Distribution Line.

    The oscillator is designed to anticipate directional changes in the A/D Line
    by measuring the momentum behind the movements. A positive value indicates
    that the security is being accumulated (buying pressure), while a negative
    value indicates distribution (selling pressure).

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
    fastperiod : int, optional
        Number of periods for the fast EMA (default: 3)
    slowperiod : int, optional
        Number of periods for the slow EMA (default: 10)

    Returns
    -------
    np.ndarray
        Array of Chaikin A/D Oscillator values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib ADOSC signature
    - Uses Numba JIT compilation for maximum performance
    - The first (slowperiod - 1) values will be NaN
    - fastperiod should be less than slowperiod for meaningful results

    Formula
    -------
    AD Line = Cumulative sum of Money Flow Volume
    Fast EMA = EMA(AD Line, fastperiod)
    Slow EMA = EMA(AD Line, slowperiod)
    ADOSC = Fast EMA - Slow EMA

    Interpretation:
    - Positive ADOSC: Accumulation (buying pressure)
    - Negative ADOSC: Distribution (selling pressure)
    - Rising ADOSC: Increasing buying pressure
    - Falling ADOSC: Increasing selling pressure

    Examples
    --------
    >>> import numpy as np
    >>> from numta import ADOSC
    >>> high = np.array([10, 11, 12, 11, 13, 14, 15])
    >>> low = np.array([9, 10, 10, 9, 11, 12, 13])
    >>> close = np.array([9.5, 10.5, 11, 10, 12, 13, 14])
    >>> volume = np.array([1000, 1100, 1200, 900, 1300, 1400, 1500])
    >>> adosc = ADOSC(high, low, close, volume)
    >>> print(adosc)
    """
    # Validate inputs
    if fastperiod < 2:
        raise ValueError("fastperiod must be >= 2")
    if slowperiod < 2:
        raise ValueError("slowperiod must be >= 2")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be less than slowperiod")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    # Validate input lengths
    n = len(high)
    if len(low) != n or len(close) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < slowperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _adosc_numba(high, low, close, volume, fastperiod, slowperiod, output)

    return output


def OBV(close: Union[np.ndarray, list],
        volume: Union[np.ndarray, list]) -> np.ndarray:
    """
    On Balance Volume (OBV)

    OBV is a momentum indicator that uses volume flow to predict changes in stock price.
    Developed by Joe Granville in 1963, it measures buying and selling pressure as a
    cumulative indicator by adding volume on up days and subtracting volume on down days.

    The theory is that volume precedes price movement. If a security is seeing an
    increasing OBV, it shows that buyers are willing to step in and push the price higher.

    Parameters
    ----------
    close : array-like
        Close prices array
    volume : array-like
        Volume array

    Returns
    -------
    np.ndarray
        Array of OBV values (cumulative volume)

    Notes
    -----
    - Compatible with TA-Lib OBV signature
    - Uses Numba JIT compilation for performance
    - No lookback period (starts from first bar)
    - Values are cumulative (running total)
    - Absolute OBV value is not important

    Formula
    -------
    If Close[i] > Close[i-1]:
        OBV[i] = OBV[i-1] + Volume[i]
    If Close[i] < Close[i-1]:
        OBV[i] = OBV[i-1] - Volume[i]
    If Close[i] = Close[i-1]:
        OBV[i] = OBV[i-1]

    Starting value: OBV[0] = 0

    Interpretation:
    - Rising OBV: Buying pressure (bullish)
    - Falling OBV: Selling pressure (bearish)
    - OBV confirms price trend when moving same direction
    - Divergence signals potential reversal
    - Focus on direction, not absolute value

    Advantages:
    - Simple and intuitive
    - Leading indicator (volume leads price)
    - Confirms price trends
    - Identifies divergences
    - Works across timeframes
    - No parameters to optimize

    Common Uses:
    - Trend confirmation
    - Divergence detection
    - Breakout confirmation
    - Support/resistance levels
    - Accumulation/distribution
    - Volume analysis

    Trading Signals:
    1. Trend Confirmation:
       - Price up + OBV up = Strong uptrend
       - Price down + OBV down = Strong downtrend

    2. Divergence (Reversal Signal):
       - Price makes new high but OBV doesn't = Bearish divergence
       - Price makes new low but OBV doesn't = Bullish divergence

    3. Breakout Confirmation:
       - OBV breaks resistance before price = Bullish
       - OBV breaks support before price = Bearish

    4. Trendline Analysis:
       - Draw trendlines on OBV
       - OBV trendline break signals price reversal

    Trading Applications:
    - Enter long when OBV breaks above resistance
    - Exit long when OBV shows divergence
    - Enter short when OBV breaks below support
    - Use OBV trendlines for entry/exit signals

    Limitations:
    - Can give false signals in choppy markets
    - Large volume spikes can distort OBV
    - Doesn't account for volume quality
    - Works best with other indicators

    Comparison with Related Indicators:
    - AD Line: Uses high-low range, not just close
    - Money Flow Index: Bounded version (0-100)
    - Volume: OBV is cumulative volume
    - Chaikin Money Flow: Weighted by price location

    Examples
    --------
    >>> import numpy as np
    >>> from numta import OBV
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106])
    >>> volume = np.array([1000, 1500, 1200, 1800, 2000, 1100, 1900])
    >>> obv = OBV(close, volume)
    >>> print(obv)
    [0, 1500, 300, 2100, 4100, 3000, 4900]
    >>> # Rising OBV indicates accumulation

    See Also
    --------
    AD : Accumulation/Distribution Line
    ADOSC : Chaikin A/D Oscillator
    MFI : Money Flow Index
    """
    # Convert to numpy arrays
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    n = len(close)
    if len(volume) != n:
        raise ValueError("close and volume must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _obv_numba(close, volume, output)

    return output
