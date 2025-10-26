"""
Pattern Recognition - Candlestick pattern recognition functions

This module implements candlestick pattern recognition compatible with TA-Lib.
Patterns return integer values:
    100: Bullish pattern
   -100: Bearish pattern
      0: No pattern
"""

"""Public API for pattern_recognition"""

import numpy as np
from typing import Union

# Import backend implementations
from ..cpu.pattern_recognition import *
from ..gpu.pattern_recognition import *
from ..backend import get_backend


def CDL2CROWS(open_: Union[np.ndarray, list],
              high: Union[np.ndarray, list],
              low: Union[np.ndarray, list],
              close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Two Crows - 3-candle bearish reversal pattern

    The Two Crows pattern is a bearish reversal that appears after an uptrend.
    It consists of a long white candle followed by two black candles that gap up
    but close progressively lower, warning of a potential reversal.

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        Array of pattern signals:
        -100: Bearish Two Crows pattern detected
           0: No pattern

    Notes
    -----
    Pattern Requirements:
    1. Long white candle (uptrend continuation)
    2. Black candle that gaps up but closes within first candle body
    3. Black candle that opens within second body, closes lower

    The pattern suggests that bulls are losing control after the initial gap up,
    warning of a potential bearish reversal.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdl2crows_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdl2crows_numba(open_, high, low, close, output)
        return output


def CDL3BLACKCROWS(open_: Union[np.ndarray, list],
                   high: Union[np.ndarray, list],
                   low: Union[np.ndarray, list],
                   close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Three Black Crows - 3-candle bearish reversal pattern

    The Three Black Crows is a powerful bearish reversal pattern consisting of
    three consecutive long black candles. Each candle opens within the previous
    candle's body and closes progressively lower, signaling strong selling pressure.

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        Array of pattern signals:
        -100: Bearish Three Black Crows pattern detected
           0: No pattern

    Notes
    -----
    Pattern Requirements:
    1. Three consecutive long black (bearish) candles
    2. Each candle opens within the previous candle's body
    3. Each candle closes progressively lower
    4. Small or no upper shadows (showing no bullish pressure)

    This pattern is one of the most reliable bearish reversal indicators and
    suggests a strong shift from bullish to bearish sentiment.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdl3blackcrows_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdl3blackcrows_numba(open_, high, low, close, output)
        return output


def CDL3INSIDE(open_: Union[np.ndarray, list],
               high: Union[np.ndarray, list],
               low: Union[np.ndarray, list],
               close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Three Inside Up/Down - 3-candle reversal pattern

    The Three Inside pattern is a reversal pattern that combines a harami pattern
    with a confirmation candle. It comes in bullish (Three Inside Up) and bearish
    (Three Inside Down) versions.

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        Array of pattern signals:
        +100: Bullish Three Inside Up pattern detected
        -100: Bearish Three Inside Down pattern detected
           0: No pattern

    Notes
    -----
    Three Inside Up (Bullish):
    1. Long black candle (downtrend)
    2. White candle inside first candle (bullish harami)
    3. White candle closing above first candle's high (confirmation)

    Three Inside Down (Bearish):
    1. Long white candle (uptrend)
    2. Black candle inside first candle (bearish harami)
    3. Black candle closing below first candle's low (confirmation)

    The pattern provides stronger reversal signal than harami alone due to
    the confirmation candle.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdl3inside_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdl3inside_numba(open_, high, low, close, output)
        return output


def CDL3OUTSIDE(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Three Outside Up/Down - 3-candle reversal pattern

    The Three Outside pattern is a reversal pattern that combines an engulfing pattern
    with a confirmation candle. It comes in bullish (Three Outside Up) and bearish
    (Three Outside Down) versions.

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        Array of pattern signals:
        +100: Bullish Three Outside Up pattern detected
        -100: Bearish Three Outside Down pattern detected
           0: No pattern

    Notes
    -----
    Three Outside Up (Bullish):
    1. Black candle (downtrend)
    2. White candle that engulfs first candle (bullish engulfing)
    3. White candle closing higher than second (confirmation)

    Three Outside Down (Bearish):
    1. White candle (uptrend)
    2. Black candle that engulfs first candle (bearish engulfing)
    3. Black candle closing lower than second (confirmation)

    The pattern provides stronger reversal signal than engulfing alone due to
    the confirmation candle.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdl3outside_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdl3outside_numba(open_, high, low, close, output)
        return output


def CDLMARUBOZU(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Marubozu Candlestick Pattern

    The Marubozu is a candlestick pattern characterized by having little to no
    wicks/shadows. It indicates strong directional momentum.

    A Marubozu forms when the opening and closing prices are at or very near
    the high and low of the period, showing that buyers (bullish) or sellers
    (bearish) were in complete control throughout the entire period.

    Pattern Characteristics:
    - Bullish Marubozu: Long white/green body with minimal shadows
      * Open near/at the low
      * Close near/at the high
      * Indicates strong buying pressure
    - Bearish Marubozu: Long black/red body with minimal shadows
      * Open near/at the high
      * Close near/at the low
      * Indicates strong selling pressure

    Parameters
    ----------
    open_ : array-like
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
        Integer array with pattern signals:
        100: Bullish Marubozu
       -100: Bearish Marubozu
          0: No pattern

    Notes
    -----
    - Compatible with TA-Lib CDLMARUBOZU signature
    - Uses Numba JIT compilation (CPU) or CuPy (GPU) for performance
    - Body must comprise at least 95% of the high-low range
    - No lookback period - each candle is evaluated independently

    Implementation Details:
    - Automatically uses CPU (Numba) or GPU (CuPy) backend based on
      current backend setting (see set_backend)
    - Pattern recognition criteria:
      * Body ratio = |Close - Open| / (High - Low)
      * Marubozu if body_ratio >= 0.95
      * Sign determined by Close vs Open relationship

    Trading Interpretation:
    - Bullish Marubozu:
      * Strong buying signal, especially at support levels
      * Suggests continuation if in uptrend
      * Suggests reversal if after downtrend
    - Bearish Marubozu:
      * Strong selling signal, especially at resistance levels
      * Suggests continuation if in downtrend
      * Suggests reversal if after uptrend
    - Best used with:
      * Volume confirmation (high volume strengthens signal)
      * Trend context (more reliable with trend)
      * Support/resistance levels

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import CDLMARUBOZU
    >>> # Bullish Marubozu example
    >>> open_ = np.array([100.0, 102.0, 104.0])
    >>> high = np.array([105.0, 107.0, 109.0])
    >>> low = np.array([100.0, 102.0, 104.0])
    >>> close = np.array([105.0, 107.0, 109.0])
    >>> pattern = CDLMARUBOZU(open_, high, low, close)
    >>> print(pattern)  # [100, 100, 100] - All bullish Marubozu

    >>> # Bearish Marubozu example
    >>> open_ = np.array([105.0, 107.0, 109.0])
    >>> high = np.array([105.0, 107.0, 109.0])
    >>> low = np.array([100.0, 102.0, 104.0])
    >>> close = np.array([100.0, 102.0, 104.0])
    >>> pattern = CDLMARUBOZU(open_, high, low, close)
    >>> print(pattern)  # [-100, -100, -100] - All bearish Marubozu

    >>> # No pattern (has wicks)
    >>> open_ = np.array([102.0])
    >>> high = np.array([105.0])
    >>> low = np.array([100.0])
    >>> close = np.array([103.0])
    >>> pattern = CDLMARUBOZU(open_, high, low, close)
    >>> print(pattern)  # [0] - No pattern (body only 20% of range)

    See Also
    --------
    CDLCLOSINGMARUBOZU : Closing Marubozu pattern
    CDLDOJI : Doji pattern (opposite of Marubozu)

    References
    ----------
    - Japanese Candlestick Charting Techniques by Steve Nison
    - Encyclopedia of Candlestick Charts by Thomas Bulkowski
    """
    # Convert to numpy arrays if needed
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Validate input lengths
    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.int32)

    # Check backend and dispatch to appropriate implementation

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cdlmarubozu_cupy(open_, high, low, close)
    else:
        # Use CPU implementation (default)
        output = np.zeros(n, dtype=np.int32)
        _cdlmarubozu_numba(open_, high, low, close, output)
        return output


def CDLMATCHINGLOW(open_: Union[np.ndarray, list],
                   high: Union[np.ndarray, list],
                   low: Union[np.ndarray, list],
                   close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Matching Low Candlestick Pattern

    The Matching Low is a bullish reversal pattern that forms during a downtrend,
    consisting of two consecutive bearish candles with matching closing prices.

    Pattern Characteristics:
    - Context: Appears in downtrend or during price decline
    - First Candle: Black (bearish) candlestick
    - Second Candle: Black candlestick closing at same level as first
    - Signal: Bullish reversal - failure to make new lows suggests support

    The matching lows suggest that sellers are unable to push prices lower,
    indicating potential exhaustion of selling pressure and formation of support.

    Parameters
    ----------
    open_ : array-like
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
        Integer array with pattern signals:
        100: Bullish Matching Low pattern
          0: No pattern

    Notes
    -----
    - Compatible with TA-Lib CDLMATCHINGLOW signature
    - Uses Numba JIT compilation (CPU) or CuPy (GPU) for performance
    - Requires minimum 2 candles
    - Tolerance for "matching" closes is 0.1% of average range

    Trading Interpretation:
    - Bullish Reversal Signal:
      * Two consecutive bearish candles fail to make new lows
      * Suggests support level formation
      * More reliable when:
        - Appears after significant downtrend
        - Occurs at known support levels
        - Confirmed by volume or other indicators
    - Best used with:
      * Trend context (reversal patterns more reliable after trends)
      * Volume confirmation
      * Support/resistance levels
      * Other technical indicators

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import CDLMATCHINGLOW
    >>> # Matching Low example - two black candles with matching closes
    >>> open_ = np.array([110.0, 105.0, 102.0])
    >>> high = np.array([112.0, 106.0, 103.0])
    >>> low = np.array([104.0, 100.0, 98.0])
    >>> close = np.array([105.0, 100.0, 100.0])  # Last two closes match
    >>> pattern = CDLMATCHINGLOW(open_, high, low, close)
    >>> print(pattern)  # [0, 0, 100] - Pattern detected on third candle

    See Also
    --------
    CDLENGULFING : Engulfing pattern
    CDLHARAMI : Harami pattern

    References
    ----------
    - Japanese Candlestick Charting Techniques by Steve Nison
    - Encyclopedia of Candlestick Charts by Thomas Bulkowski
    """
    # Convert to numpy arrays if needed
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Validate input lengths
    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.int32)

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Check backend and dispatch to appropriate implementation

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cdlmatchinglow_cupy(open_, high, low, close)
    else:
        # Use CPU implementation (default)
        output = np.zeros(n, dtype=np.int32)
        _cdlmatchinglow_numba(open_, high, low, close, output)
        return output


def CDLMATHOLD(open_: Union[np.ndarray, list],
               high: Union[np.ndarray, list],
               low: Union[np.ndarray, list],
               close: Union[np.ndarray, list],
               penetration: float = 0.5) -> np.ndarray:
    """
    Mat Hold Candlestick Pattern

    The Mat Hold is a bullish continuation pattern that appears during an uptrend.
    It consists of five candles showing a brief consolidation before trend continuation.

    Pattern Structure:
    - Candle 1: Long white (bullish) candle indicating strong upward momentum
    - Candle 2: Small black candle gapping up from candle 1
    - Candles 3-4: Small candles (any color) continuing downward but staying
                   within the body of candle 1
    - Candle 5: White candle closing above the highs of candles 2-4

    The pattern is similar to Rising Three Methods but with a gap up on the second
    candle, making it a stronger bullish continuation signal.

    Parameters
    ----------
    open_ : array-like
        Open prices array
    high : array-like
        High prices array
    low : array-like
        Low prices array
    close : array-like
        Close prices array
    penetration : float, optional
        Percentage of penetration of candles 2-4 into candle 1's body.
        Default is 0.5 (50%). Valid range: 0.0 to 1.0.
        Lower values require tighter consolidation within first candle's body.

    Returns
    -------
    np.ndarray
        Integer array with pattern signals:
        100: Bullish Mat Hold pattern
          0: No pattern

    Notes
    -----
    - Compatible with TA-Lib CDLMATHOLD signature
    - Uses Numba JIT compilation (CPU) or CuPy (GPU) for performance
    - Requires minimum 5 candles
    - Long candle = body > 1.5x average body
    - Small candle = body < 0.5x average body
    - Average calculated over previous 10 periods

    Trading Interpretation:
    - Bullish Continuation Signal:
      * Brief consolidation in established uptrend
      * Pullback stays within first candle's body
      * Fifth candle confirms continuation with strong close
      * Pattern validity increases when:
        - Appears in strong uptrend
        - First candle shows strong momentum
        - Consolidation volume decreases
        - Fifth candle has high volume
    - Strategy:
      * Enter on close of fifth candle
      * Stop loss below low of consolidation
      * Target: measure of first candle added to breakout point

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import CDLMATHOLD
    >>> # Mat Hold example
    >>> open_ = np.array([100.0, 110.0, 112.0, 111.0, 110.0, 113.0])
    >>> high = np.array([110.0, 112.0, 113.0, 112.0, 111.0, 115.0])
    >>> low = np.array([100.0, 111.0, 110.0, 109.0, 108.0, 112.0])
    >>> close = np.array([110.0, 111.0, 110.5, 109.5, 109.0, 114.0])
    >>> pattern = CDLMATHOLD(open_, high, low, close, penetration=0.5)
    >>> print(pattern)  # [0, 0, 0, 0, 0, 100] - Pattern on 6th candle

    >>> # With different penetration
    >>> pattern = CDLMATHOLD(open_, high, low, close, penetration=0.3)
    >>> print(pattern)  # Stricter penetration requirement

    See Also
    --------
    CDLRISEFALL3METHODS : Rising/Falling Three Methods
    CDL3WHITESOLDIERS : Three White Soldiers

    References
    ----------
    - Japanese Candlestick Charting Techniques by Steve Nison
    - Encyclopedia of Candlestick Charts by Thomas Bulkowski
    """
    # Convert to numpy arrays if needed
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Validate penetration parameter
    if not 0.0 <= penetration <= 1.0:
        raise ValueError("penetration must be between 0.0 and 1.0")

    # Validate input lengths
    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.array([], dtype=np.int32)

    if n < 5:
        return np.zeros(n, dtype=np.int32)

    # Check backend and dispatch to appropriate implementation

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cdlmathold_cupy(open_, high, low, close, penetration)
    else:
        # Use CPU implementation (default)
        output = np.zeros(n, dtype=np.int32)
        _cdlmathold_numba(open_, high, low, close, penetration, output)
        return output


def CDLMORNINGDOJISTAR(open_: Union[np.ndarray, list],
                       high: Union[np.ndarray, list],
                       low: Union[np.ndarray, list],
                       close: Union[np.ndarray, list],
                       penetration: float = 0.3) -> np.ndarray:
    """
    Morning Doji Star Candlestick Pattern

    A bullish reversal pattern consisting of three candles:
    1. Long black candle (continuing downtrend)
    2. Doji gapping down (indecision)
    3. White candle closing well into first candle's body

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices
    penetration : float, optional
        Percentage of penetration into first candle's body (default 0.3)

    Returns
    -------
    np.ndarray
        100: Bullish Morning Doji Star
          0: No pattern

    Notes
    -----
    Similar to Morning Star but requires a Doji on the second candle,
    indicating stronger indecision before the reversal.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n < 3:
        return np.zeros(n, dtype=np.int32)

    backend = get_backend()

    if backend == "gpu":
        return _cdlmorningdojistar_cupy(open_, high, low, close, penetration)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlmorningdojistar_numba(open_, high, low, close, penetration, output)
        return output


def CDLMORNINGSTAR(open_: Union[np.ndarray, list],
                   high: Union[np.ndarray, list],
                   low: Union[np.ndarray, list],
                   close: Union[np.ndarray, list],
                   penetration: float = 0.3) -> np.ndarray:
    """
    Morning Star Candlestick Pattern

    A bullish reversal pattern consisting of three candles:
    1. Long black candle (continuing downtrend)
    2. Short-bodied candle gapping down (star position, any color)
    3. White candle closing well into first candle's body

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices
    penetration : float, optional
        Percentage of penetration into first candle's body (default 0.3)

    Returns
    -------
    np.ndarray
        100: Bullish Morning Star
          0: No pattern

    Notes
    -----
    More common than Morning Doji Star. The second candle can be any color
    but must have a small body and gap down from the first candle.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n < 3:
        return np.zeros(n, dtype=np.int32)

    backend = get_backend()

    if backend == "gpu":
        return _cdlmorningstar_cupy(open_, high, low, close, penetration)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlmorningstar_numba(open_, high, low, close, penetration, output)
        return output


def CDLONNECK(open_: Union[np.ndarray, list],
              high: Union[np.ndarray, list],
              low: Union[np.ndarray, list],
              close: Union[np.ndarray, list]) -> np.ndarray:
    """
    On-Neck Candlestick Pattern

    A bearish continuation pattern consisting of two candles:
    1. Long black candle (continuing downtrend)
    2. White candle opening below prior low, closing at prior low

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        -100: Bearish On-Neck pattern
           0: No pattern

    Notes
    -----
    The pattern suggests continuation of the downtrend. The white candle
    fails to close above the prior candle's low, showing weakness in buying.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    backend = get_backend()

    if backend == "gpu":
        return _cdlonneck_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlonneck_numba(open_, high, low, close, output)
        return output


def CDLPIERCING(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Piercing Pattern Candlestick

    A bullish reversal pattern consisting of two candles:
    1. Long black candle (continuing downtrend)
    2. Long white candle opening below prior low, closing above 50% of prior body

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        100: Bullish Piercing Pattern
          0: No pattern

    Notes
    -----
    The pattern shows strong buying after a gap down, piercing more than
    halfway into the prior black candle. Distinguished from Bullish Engulfing
    by not completely covering the prior candle's body.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    n = len(open_)
    if len(high) != n or len(low) != n or len(close) != n:
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    backend = get_backend()

    if backend == "gpu":
        return _cdlpiercing_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlpiercing_numba(open_, high, low, close, output)
        return output


def CDLRICKSHAWMAN(open_: Union[np.ndarray, list], high: Union[np.ndarray, list],
                   low: Union[np.ndarray, list], close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Rickshaw Man - Neutral doji with long shadows on both sides

    Returns 100 when pattern detected (indicates indecision), 0 otherwise.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlrickshawman_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlrickshawman_numba(open_, high, low, close, output)
        return output


def CDLRISEFALL3METHODS(open_: Union[np.ndarray, list], high: Union[np.ndarray, list],
                        low: Union[np.ndarray, list], close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Rising/Falling Three Methods - 5-candle continuation pattern

    Returns +100 (rising), -100 (falling), or 0 (no pattern).
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 5:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlrisefall3methods_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlrisefall3methods_numba(open_, high, low, close, output)
        return output


def CDLSEPARATINGLINES(open_: Union[np.ndarray, list], high: Union[np.ndarray, list],
                       low: Union[np.ndarray, list], close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Separating Lines - 2-candle continuation with matching opens

    Returns +100 (bullish), -100 (bearish), or 0 (no pattern).
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 2:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlseparatinglines_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlseparatinglines_numba(open_, high, low, close, output)
        return output


def CDLSHOOTINGSTAR(open_: Union[np.ndarray, list], high: Union[np.ndarray, list],
                    low: Union[np.ndarray, list], close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Shooting Star - Bearish reversal with small body and long upper shadow

    Returns -100 (bearish), or 0 (no pattern).
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 2:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlshootingstar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlshootingstar_numba(open_, high, low, close, output)
        return output


def CDLSHORTLINE(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> np.ndarray:
    """Short Line Candle pattern recognition

    A 1-bar pattern with short real body and short upper and lower shadows.
    Returns 100 for white candles, -100 for black candles, 0 otherwise.
    """
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlshortline_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlshortline_numba(open_, high, low, close, output)
        return output


def CDLSPINNINGTOP(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray) -> np.ndarray:
    """Spinning Top pattern recognition

    A 1-bar pattern with small real body and long upper/lower shadows (both > body).
    Indicates indecision. Returns 100 for white, -100 for black, 0 otherwise.
    """
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlspinningtop_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlspinningtop_numba(open_, high, low, close, output)
        return output


def CDLSTALLEDPATTERN(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray) -> np.ndarray:
    """Stalled Pattern recognition

    A 3-bar bearish pattern: three white soldiers with weakening momentum.
    Third candle is small and rides on second's shoulder. Returns -100 or 0.
    """
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlstalledpattern_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlstalledpattern_numba(open_, high, low, close, output)
        return output


def CDLSTICKSANDWICH(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> np.ndarray:
    """Stick Sandwich pattern recognition

    A 3-bar bullish pattern: black candle, white candle (gap up), black candle.
    First and third candles have matching closes. Returns 100 or 0.
    """
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlsticksandwich_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlsticksandwich_numba(open_, high, low, close, output)
        return output


def CDLTAKURI(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
              close: np.ndarray) -> np.ndarray:
    """Takuri (Dragonfly Doji with very long lower shadow) pattern recognition

    A 1-bar bullish reversal: doji body, very short upper shadow, very long lower shadow.
    Returns 100 or 0.
    """
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdltakuri_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdltakuri_numba(open_, high, low, close, output)
        return output


def CDLTASUKIGAP(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> np.ndarray:
    """Tasuki Gap pattern recognition

    A 3-bar continuation pattern with gap followed by pullback that doesn't close gap.
    Returns 100 for upside, -100 for downside, 0 otherwise.
    """
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdltasukigap_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdltasukigap_numba(open_, high, low, close, output)
        return output


def CDLTHRUSTING(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> np.ndarray:
    """Thrusting Pattern: Bearish, white closes below midpoint of prior black"""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlthrusting_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlthrusting_numba(open_, high, low, close, output)
        return output


def CDLTRISTAR(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
               close: np.ndarray) -> np.ndarray:
    """Tristar Pattern: Three dojis with second gapping. Returns ±100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdltristar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdltristar_numba(open_, high, low, close, output)
        return output


def CDLUNIQUE3RIVER(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray) -> np.ndarray:
    """Unique 3 River: Bullish reversal, 3 candles. Returns 100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlunique3river_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlunique3river_numba(open_, high, low, close, output)
        return output


def CDLUPSIDEGAP2CROWS(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> np.ndarray:
    """Upside Gap Two Crows: Bearish reversal. Returns -100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlupsidegap2crows_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlupsidegap2crows_numba(open_, high, low, close, output)
        return output


def CDLXSIDEGAP3METHODS(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray) -> np.ndarray:
    """Upside/Downside Gap 3 Methods: Continuation. Returns ±100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlxsidegap3methods_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlxsidegap3methods_numba(open_, high, low, close, output)
        return output


def CDL3STARSINSOUTH(*args, **kwargs):
    raise NotImplementedError("CDL3STARSINSOUTH not yet implemented")


def CDL3WHITESOLDIERS(*args, **kwargs):
    raise NotImplementedError("CDL3WHITESOLDIERS not yet implemented")


def CDLABANDONEDBABY(*args, **kwargs):
    raise NotImplementedError("CDLABANDONEDBABY not yet implemented")


def CDLADVANCEBLOCK(*args, **kwargs):
    raise NotImplementedError("CDLADVANCEBLOCK not yet implemented")


def CDLBELTHOLD(*args, **kwargs):
    raise NotImplementedError("CDLBELTHOLD not yet implemented")


def CDLBREAKAWAY(*args, **kwargs):
    raise NotImplementedError("CDLBREAKAWAY not yet implemented")


def CDLCLOSINGMARUBOZU(*args, **kwargs):
    raise NotImplementedError("CDLCLOSINGMARUBOZU not yet implemented")


def CDLCONCEALBABYSWALL(*args, **kwargs):
    raise NotImplementedError("CDLCONCEALBABYSWALL not yet implemented")


def CDLCOUNTERATTACK(*args, **kwargs):
    raise NotImplementedError("CDLCOUNTERATTACK not yet implemented")


def CDLDARKCLOUDCOVER(*args, **kwargs):
    raise NotImplementedError("CDLDARKCLOUDCOVER not yet implemented")


def CDLDOJI(*args, **kwargs):
    raise NotImplementedError("CDLDOJI not yet implemented")


def CDLDOJISTAR(*args, **kwargs):
    raise NotImplementedError("CDLDOJISTAR not yet implemented")


def CDLDRAGONFLYDOJI(*args, **kwargs):
    raise NotImplementedError("CDLDRAGONFLYDOJI not yet implemented")


def CDLENGULFING(*args, **kwargs):
    raise NotImplementedError("CDLENGULFING not yet implemented")


def CDLEVENINGDOJISTAR(*args, **kwargs):
    raise NotImplementedError("CDLEVENINGDOJISTAR not yet implemented")


def CDLEVENINGSTAR(*args, **kwargs):
    raise NotImplementedError("CDLEVENINGSTAR not yet implemented")


def CDLGAPSIDESIDEWHITE(*args, **kwargs):
    raise NotImplementedError("CDLGAPSIDESIDEWHITE not yet implemented")


def CDLGRAVESTONEDOJI(*args, **kwargs):
    raise NotImplementedError("CDLGRAVESTONEDOJI not yet implemented")


def CDLHAMMER(*args, **kwargs):
    raise NotImplementedError("CDLHAMMER not yet implemented")


def CDLHANGINGMAN(*args, **kwargs):
    raise NotImplementedError("CDLHANGINGMAN not yet implemented")


def CDLHARAMI(*args, **kwargs):
    raise NotImplementedError("CDLHARAMI not yet implemented")


def CDLHARAMICROSS(*args, **kwargs):
    raise NotImplementedError("CDLHARAMICROSS not yet implemented")


def CDLHIGHWAVE(*args, **kwargs):
    raise NotImplementedError("CDLHIGHWAVE not yet implemented")


def CDLHIKKAKE(*args, **kwargs):
    raise NotImplementedError("CDLHIKKAKE not yet implemented")


def CDLHIKKAKEMOD(*args, **kwargs):
    raise NotImplementedError("CDLHIKKAKEMOD not yet implemented")


def CDLHOMINGPIGEON(*args, **kwargs):
    raise NotImplementedError("CDLHOMINGPIGEON not yet implemented")


def CDLIDENTICAL3CROWS(*args, **kwargs):
    raise NotImplementedError("CDLIDENTICAL3CROWS not yet implemented")


def CDLINNECK(*args, **kwargs):
    raise NotImplementedError("CDLINNECK not yet implemented")


def CDLINVERTEDHAMMER(*args, **kwargs):
    raise NotImplementedError("CDLINVERTEDHAMMER not yet implemented")


def CDLKICKING(*args, **kwargs):
    raise NotImplementedError("CDLKICKING not yet implemented")


def CDLKICKINGBYLENGTH(*args, **kwargs):
    raise NotImplementedError("CDLKICKINGBYLENGTH not yet implemented")


def CDLLADDERBOTTOM(*args, **kwargs):
    raise NotImplementedError("CDLLADDERBOTTOM not yet implemented")


def CDLLONGLEGGEDDOJI(*args, **kwargs):
    raise NotImplementedError("CDLLONGLEGGEDDOJI not yet implemented")


def CDLLONGLINE(*args, **kwargs):
    raise NotImplementedError("CDLLONGLINE not yet implemented")


