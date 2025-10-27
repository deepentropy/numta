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


def CDL3STARSINSOUTH(open_: Union[np.ndarray, list],
                     high: Union[np.ndarray, list],
                     low: Union[np.ndarray, list],
                     close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Three Stars in the South - 3-candle bullish reversal pattern

    A rare bullish reversal pattern consisting of three black candles with
    progressively smaller bodies and decreasing lower shadows, signaling
    weakening bearish pressure and potential reversal.

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
        +100: Bullish Three Stars in the South pattern detected
           0: No pattern

    Notes
    -----
    Pattern Requirements:
    1. Long black candle with long lower shadow
    2. Smaller black candle within first candle's range
    3. Short black marubozu within second candle's range

    This pattern indicates that sellers are losing momentum despite continued
    bearish candles, suggesting a potential bullish reversal.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdl3starsinsouth_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdl3starsinsouth_numba(open_, high, low, close, output)
        return output


def CDL3WHITESOLDIERS(open_: Union[np.ndarray, list],
                      high: Union[np.ndarray, list],
                      low: Union[np.ndarray, list],
                      close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Three White Soldiers - 3-candle bullish reversal pattern

    A powerful bullish reversal pattern consisting of three consecutive long
    white candles. Each candle opens within the previous candle's body and
    closes progressively higher, signaling strong buying pressure.

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
        +100: Bullish Three White Soldiers pattern detected
           0: No pattern

    Notes
    -----
    Pattern Requirements:
    1. Three consecutive long white (bullish) candles
    2. Each candle opens within the previous candle's body
    3. Each candle closes progressively higher
    4. Small or no lower shadows (showing no bearish pressure)

    This pattern is the bullish counterpart to Three Black Crows and is
    one of the most reliable bullish reversal indicators.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdl3whitesoldiers_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdl3whitesoldiers_numba(open_, high, low, close, output)
        return output


def CDLABANDONEDBABY(open_: Union[np.ndarray, list],
                     high: Union[np.ndarray, list],
                     low: Union[np.ndarray, list],
                     close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Abandoned Baby - 3-candle reversal pattern with isolated doji

    A rare and powerful reversal pattern featuring a doji that is isolated
    from the surrounding candles by gaps on both sides, resembling an
    "abandoned" baby.

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
        +100: Bullish Abandoned Baby pattern detected
        -100: Bearish Abandoned Baby pattern detected
           0: No pattern

    Notes
    -----
    Bullish Abandoned Baby:
    1. Black candle in downtrend
    2. Gap down to a doji (isolated)
    3. Gap up to a white candle

    Bearish Abandoned Baby:
    1. White candle in uptrend
    2. Gap up to a doji (isolated)
    3. Gap down to a black candle

    The doji represents indecision and the gaps show a dramatic shift in
    sentiment, making this a strong reversal signal.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlabandonedbaby_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlabandonedbaby_numba(open_, high, low, close, output)
        return output


def CDLADVANCEBLOCK(open_: Union[np.ndarray, list],
                    high: Union[np.ndarray, list],
                    low: Union[np.ndarray, list],
                    close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Advance Block - 3-candle bearish reversal warning pattern

    A warning pattern that appears during an uptrend, consisting of three
    white candles with progressively smaller bodies and increasing upper
    shadows, signaling weakening bullish momentum.

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
        -100: Bearish Advance Block warning detected
           0: No pattern

    Notes
    -----
    Pattern Requirements:
    1. Three white candles in uptrend
    2. Each candle has progressively smaller body
    3. Increasing upper shadows
    4. Each opens within previous candle's body

    This pattern warns that despite continued upward movement, buyers are
    losing strength, suggesting a potential reversal or consolidation.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdladvanceblock_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdladvanceblock_numba(open_, high, low, close, output)
        return output


def CDLBELTHOLD(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Belt Hold - Single candle reversal pattern

    A strong single-candle reversal pattern characterized by a marubozu
    candle opening at the extreme of the range.

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
        +100: Bullish Belt Hold pattern detected
        -100: Bearish Belt Hold pattern detected
           0: No pattern

    Notes
    -----
    Bullish Belt Hold (Yorikiri):
    - White marubozu opening on or near the low
    - No or minimal lower shadow
    - Strong buying from the open

    Bearish Belt Hold (Yorikiri):
    - Black marubozu opening on or near the high
    - No or minimal upper shadow
    - Strong selling from the open

    This pattern shows strong directional conviction from the opening price.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlbelthold_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlbelthold_numba(open_, high, low, close, output)
        return output


def CDLBREAKAWAY(open_: Union[np.ndarray, list],
                 high: Union[np.ndarray, list],
                 low: Union[np.ndarray, list],
                 close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Breakaway - 5-candle continuation pattern

    A continuation pattern that appears after a trend, featuring a gap
    followed by continuation candles, then a strong candle closing back
    into the initial gap.

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
        +100: Bullish Breakaway pattern detected
        -100: Bearish Breakaway pattern detected
           0: No pattern

    Notes
    -----
    Bullish Breakaway:
    1. Long black candle
    2. Gap down
    3. 2-3 candles continuing downward
    4. Long white candle closing within the original gap

    Bearish Breakaway:
    1. Long white candle
    2. Gap up
    3. 2-3 candles continuing upward
    4. Long black candle closing within the original gap

    Despite the reversal-looking final candle, this is actually a
    continuation pattern, suggesting the trend will resume.
    """
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")
    if n < 5:
        return np.zeros(n, dtype=np.int32)

    if get_backend() == "gpu":
        return _cdlbreakaway_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlbreakaway_numba(open_, high, low, close, output)
        return output


def CDLCLOSINGMARUBOZU(open_: Union[np.ndarray, list],
                       high: Union[np.ndarray, list],
                       low: Union[np.ndarray, list],
                       close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Closing Marubozu - Single candle pattern with no shadow at close

    A marubozu candle with a shadow on the opening side but none on the closing side.
    Shows strong momentum in the closing direction.

    Bullish version: White candle that opens below the low and closes at the high,
    indicating strong buying pressure throughout the period.

    Bearish version: Black candle that opens above the high and closes at the low,
    indicating strong selling pressure throughout the period.

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
         100: Bullish Closing Marubozu (white candle closing at high)
        -100: Bearish Closing Marubozu (black candle closing at low)
           0: No pattern

    Notes
    -----
    The pattern requires a significant body (>60% of range) and a shadow
    only on the opening side (>10% of range), with virtually no shadow
    on the closing side (<5% of range).
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlclosingmarubozu_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlclosingmarubozu_numba(open_, high, low, close, output)
        return output


def CDLCONCEALBABYSWALL(open_: Union[np.ndarray, list],
                        high: Union[np.ndarray, list],
                        low: Union[np.ndarray, list],
                        close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Concealing Baby Swallow - 4-candle bullish reversal pattern

    A rare bullish reversal pattern that appears in a downtrend, signaling
    potential trend exhaustion and reversal.

    The pattern consists of:
    1. First candle: Black Marubozu
    2. Second candle: Black Marubozu
    3. Third candle: Black candle that gaps down
    4. Fourth candle: Large black candle that engulfs the third candle

    Despite all candles being black, the engulfing of the gap-down candle
    suggests that selling pressure is being absorbed, indicating a potential
    bullish reversal.

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
         100: Bullish Concealing Baby Swallow pattern detected
           0: No pattern

    Notes
    -----
    This is a rare pattern that requires strict criteria including two
    consecutive Black Marubozu candles followed by a gap-down that is
    then engulfed by a larger black candle.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 4:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlconcealbabyswall_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlconcealbabyswall_numba(open_, high, low, close, output)
        return output


def CDLCOUNTERATTACK(open_: Union[np.ndarray, list],
                     high: Union[np.ndarray, list],
                     low: Union[np.ndarray, list],
                     close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Counterattack - 2-candle reversal pattern with matching closes

    A reversal pattern where the second candle opens in the direction of
    the trend but closes at the same level as the previous candle, suggesting
    a potential trend reversal.

    Bullish version: Long black candle followed by a white candle that opens
    significantly lower but rallies to close at the same level as the black candle.

    Bearish version: Long white candle followed by a black candle that opens
    significantly higher but sells off to close at the same level as the white candle.

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
         100: Bullish Counterattack pattern detected
        -100: Bearish Counterattack pattern detected
           0: No pattern

    Notes
    -----
    The pattern requires both candles to have significant bodies (>80% of average)
    and the second candle must open beyond the first candle's range. The closes
    must match within 10% of average body size.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlcounterattack_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlcounterattack_numba(open_, high, low, close, output)
        return output


def CDLDARKCLOUDCOVER(open_: Union[np.ndarray, list],
                      high: Union[np.ndarray, list],
                      low: Union[np.ndarray, list],
                      close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Dark Cloud Cover - 2-candle bearish reversal pattern

    A bearish reversal pattern that appears at the top of an uptrend.
    The pattern consists of a long white candle followed by a black candle
    that opens above the previous high but closes well into the white candle's body.

    The pattern suggests that bulls pushed prices higher on the open, but bears
    took control and pushed prices down, closing below the midpoint of the
    previous white candle. This shift in momentum warns of a potential reversal.

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
        -100: Bearish Dark Cloud Cover pattern detected
           0: No pattern

    Notes
    -----
    The second candle must:
    - Open above the previous candle's high
    - Close below the midpoint of the previous candle's body
    - Close above the previous candle's open (not fully engulfing)

    The deeper the penetration into the first candle's body, the more
    significant the bearish signal.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdldarkcloudcover_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdldarkcloudcover_numba(open_, high, low, close, output)
        return output


def CDLDOJI(open_: Union[np.ndarray, list],
            high: Union[np.ndarray, list],
            low: Union[np.ndarray, list],
            close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Doji - Single candle pattern with very small body

    A candle where the open and close prices are virtually equal, creating
    a very small or non-existent body. The doji represents indecision and
    equilibrium between buyers and sellers.

    Dojis are considered neutral patterns that can signal potential reversals
    when they appear after a strong trend. The significance of a doji depends
    on the preceding trend and subsequent price action.

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
         100: Doji pattern detected (indicates indecision/potential reversal)
           0: No pattern

    Notes
    -----
    A doji is identified when:
    - The body is less than 10% of the total range
    - The body is less than 10% of the average body size

    Different types of dojis (Dragonfly, Gravestone, Long-legged) have
    separate pattern functions with more specific criteria.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdldoji_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdldoji_numba(open_, high, low, close, output)
        return output


def CDLDOJISTAR(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Doji Star - 2-candle reversal pattern with doji gapping away

    A reversal pattern consisting of a long candle followed by a doji
    that gaps away from the first candle's body. The gap and the doji
    together signal potential trend exhaustion and reversal.

    Bullish version: Long black candle followed by a doji that gaps down,
    suggesting selling pressure may be exhausted.

    Bearish version: Long white candle followed by a doji that gaps up,
    suggesting buying pressure may be exhausted.

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
         100: Bullish Doji Star pattern detected
        -100: Bearish Doji Star pattern detected
           0: No pattern

    Notes
    -----
    The pattern requires:
    - First candle: Significant body (>70% of average)
    - Second candle: Doji (body <10% of range and <10% of average)
    - Gap between the two candles (doji does not overlap first candle)

    The pattern is often followed by a third candle that confirms the reversal.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdldojistar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdldojistar_numba(open_, high, low, close, output)
        return output


def CDLDRAGONFLYDOJI(open_: Union[np.ndarray, list],
                     high: Union[np.ndarray, list],
                     low: Union[np.ndarray, list],
                     close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Dragonfly Doji - Single candle doji with long lower shadow

    A doji pattern where the open and close are at or very near the high
    of the candle, creating a long lower shadow. The shape resembles a
    dragonfly with wings extending downward.

    This pattern indicates that sellers pushed prices significantly lower
    during the period, but buyers regained control and pushed prices back
    to the opening level. When appearing after a downtrend, it suggests
    potential bullish reversal.

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
         100: Dragonfly Doji pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Body must be very small (<10% of range and average body)
    - Lower shadow must be significant (>60% of range)
    - Upper shadow must be minimal (<10% of range)

    The significance is greater when:
    - Appearing after a downtrend
    - The lower shadow is very long relative to recent candles
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdldragonflydoji_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdldragonflydoji_numba(open_, high, low, close, output)
        return output


def CDLENGULFING(open_: Union[np.ndarray, list],
                 high: Union[np.ndarray, list],
                 low: Union[np.ndarray, list],
                 close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Engulfing Pattern - 2-candle reversal where second candle engulfs first

    A powerful reversal pattern where the second candle's body completely
    engulfs the first candle's body.

    Bullish Engulfing: Occurs after a downtrend. A small black candle is
    followed by a larger white candle that opens at or below the first candle's
    close and closes at or above the first candle's open.

    Bearish Engulfing: Occurs after an uptrend. A small white candle is
    followed by a larger black candle that opens at or above the first candle's
    close and closes at or below the first candle's open.

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
         100: Bullish Engulfing pattern detected
        -100: Bearish Engulfing pattern detected
           0: No pattern

    Notes
    -----
    The pattern is more significant when:
    - The first candle is small and the second is large
    - Appearing at the end of a clear trend
    - Accompanied by high volume on the engulfing candle

    The second candle must have a body >50% of average body size.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlengulfing_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlengulfing_numba(open_, high, low, close, output)
        return output


def CDLEVENINGDOJISTAR(open_: Union[np.ndarray, list],
                       high: Union[np.ndarray, list],
                       low: Union[np.ndarray, list],
                       close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Evening Doji Star - 3-candle bearish reversal pattern with doji

    A bearish reversal pattern that appears at the top of an uptrend.
    The pattern consists of three candles:

    1. First candle: Long white candle continuing the uptrend
    2. Second candle: Doji that gaps up from the first candle
    3. Third candle: Black candle that closes well into the first candle's body

    The doji in the middle represents indecision at the top, and the third
    candle confirms the reversal.

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
        -100: Evening Doji Star pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - First candle: White body >70% of average
    - Second candle: Doji that gaps up (doesn't overlap first candle)
    - Third candle: Black body >50% of average, closes below midpoint of first

    This pattern is more reliable than the regular Evening Star due to
    the strong indecision signal from the doji.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 3:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdleveningdojistar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdleveningdojistar_numba(open_, high, low, close, output)
        return output


def CDLEVENINGSTAR(open_: Union[np.ndarray, list],
                   high: Union[np.ndarray, list],
                   low: Union[np.ndarray, list],
                   close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Evening Star - 3-candle bearish reversal pattern

    A bearish reversal pattern that appears at the top of an uptrend,
    signaling potential trend change. The pattern consists of three candles:

    1. First candle: Long white candle continuing the uptrend
    2. Second candle: Small-bodied candle (star) that gaps up
    3. Third candle: Black candle that closes well into the first candle's body

    The small star candle represents indecision at the top, and the third
    candle confirms bearish reversal.

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
        -100: Evening Star pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - First candle: White body >70% of average
    - Second candle: Small body (<30% of average) that gaps up
    - Third candle: Black body >50% of average, closes below midpoint of first

    The pattern is the bearish counterpart to the Morning Star pattern.
    The smaller the second candle and the larger the third candle,
    the more significant the reversal signal.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 3:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdleveningstar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdleveningstar_numba(open_, high, low, close, output)
        return output


def CDLGAPSIDESIDEWHITE(open_: Union[np.ndarray, list],
                        high: Union[np.ndarray, list],
                        low: Union[np.ndarray, list],
                        close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Gap Side-by-Side White Lines - 3-candle continuation pattern

    A continuation pattern consisting of two white candles appearing
    side-by-side after a gap in the direction of the trend.

    Bullish version (in uptrend):
    - First candle: White candle
    - Second candle: White candle that gaps up
    - Third candle: White candle similar in size to second, opening near
      second's open (side-by-side)

    Bearish version (in downtrend):
    - First candle: Black candle
    - Second candle: White candle that gaps down
    - Third candle: White candle similar in size to second, opening near
      second's open

    Despite the white candles in the bearish version, the pattern confirms
    downtrend continuation when the gap holds.

    Parameters
    ----------
    open_ : array-like
        Open prices
    high : array-like
        Low prices
    low : array-like
        Low prices
    close : array-like
        Close prices

    Returns
    -------
    np.ndarray
        Array of pattern signals:
         100: Bullish Gap Side-by-Side White Lines
        -100: Bearish Gap Side-by-Side White Lines
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Second and third candles must be white with bodies >30% of average
    - Second and third candles must be similar in size (<30% difference)
    - Second and third candles must open at similar levels (<20% difference)
    - Gap must be maintained
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 3:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlgapsidesidewhite_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlgapsidesidewhite_numba(open_, high, low, close, output)
        return output


def CDLGRAVESTONEDOJI(open_: Union[np.ndarray, list],
                      high: Union[np.ndarray, list],
                      low: Union[np.ndarray, list],
                      close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Gravestone Doji - Single candle doji with long upper shadow

    A doji pattern where the open and close are at or very near the low
    of the candle, creating a long upper shadow. The shape resembles an
    inverted T or a gravestone.

    This pattern indicates that buyers pushed prices significantly higher
    during the period, but sellers regained control and pushed prices back
    down to the opening level. When appearing after an uptrend, it suggests
    potential bearish reversal.

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
        -100: Gravestone Doji pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Body must be very small (<10% of range and average body)
    - Upper shadow must be significant (>60% of range)
    - Lower shadow must be minimal (<10% of range)

    The significance is greater when:
    - Appearing after an uptrend
    - The upper shadow is very long relative to recent candles
    - Followed by a gap down on the next candle

    This pattern is the bearish counterpart to the Dragonfly Doji.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlgravestonedoji_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlgravestonedoji_numba(open_, high, low, close, output)
        return output


def CDLHAMMER(open_: Union[np.ndarray, list],
              high: Union[np.ndarray, list],
              low: Union[np.ndarray, list],
              close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Hammer - Single candle bullish reversal with small body and long lower shadow

    A single candlestick pattern with a small body at the upper end of the
    trading range and a long lower shadow (at least twice the body length).
    The shape resembles a hammer.

    This pattern indicates that although sellers pushed prices significantly lower
    during the session, buyers regained control and pushed prices back up near
    the open. When appearing after a downtrend, it signals potential bullish
    reversal.

    The color of the body is less important (can be white or black), though
    white hammers are considered slightly more bullish.

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
         100: Hammer pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Small body (<30% of range)
    - Long lower shadow (>2x body size and >50% of range)
    - Small upper shadow (<20% of range)
    - Body in upper portion of range

    The pattern is most significant when:
    - Appearing after a sustained downtrend
    - Followed by a gap up or strong white candle
    - The lower shadow is very long relative to recent candles
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlhammer_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlhammer_numba(open_, high, low, close, output)
        return output


def CDLHANGINGMAN(open_: Union[np.ndarray, list],
                  high: Union[np.ndarray, list],
                  low: Union[np.ndarray, list],
                  close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Hanging Man - Single candle bearish reversal with small body and long lower shadow

    A single candlestick pattern that is visually identical to the Hammer pattern
    (small body at upper end, long lower shadow), but appears at the top of an
    uptrend rather than after a downtrend, signaling potential bearish reversal.

    The long lower shadow indicates that sellers pushed prices significantly lower
    during the session, but buyers managed to push prices back up. When this occurs
    after an uptrend, it suggests that buying pressure may be weakening and that
    sellers are beginning to gain strength.

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
        -100: Hanging Man pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Small body (<30% of range)
    - Long lower shadow (>2x body size and >50% of range)
    - Small upper shadow (<20% of range)
    - Body in upper portion of range

    The pattern is most significant when:
    - Appearing after a sustained uptrend
    - Followed by a gap down or strong black candle
    - Accompanied by high volume

    Despite appearing at opposite trend positions, Hammer and Hanging Man
    have identical visual characteristics. Context determines the interpretation.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlhangingman_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlhangingman_numba(open_, high, low, close, output)
        return output


def CDLHARAMI(open_: Union[np.ndarray, list],
              high: Union[np.ndarray, list],
              low: Union[np.ndarray, list],
              close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Harami - 2-candle reversal where second candle is contained within first

    A two-candle reversal pattern where the second candle's body is completely
    contained within the first candle's body. The name "harami" is Japanese for
    "pregnant," referring to the visual appearance of the pattern.

    Bullish Harami: Occurs after a downtrend. A large black candle is followed
    by a smaller white candle whose body is completely within the black candle's
    body, suggesting potential upward reversal.

    Bearish Harami: Occurs after an uptrend. A large white candle is followed
    by a smaller black candle whose body is completely within the white candle's
    body, suggesting potential downward reversal.

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
         100: Bullish Harami pattern detected
        -100: Bearish Harami pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - First candle: Large body (>70% of average)
    - Second candle: Body completely contained within first body
    - Bullish: Black first, white second
    - Bearish: White first, black second

    The pattern signals indecision and potential reversal. The smaller the
    second candle, the more significant the pattern. The Harami Cross
    (where second candle is a doji) is an even stronger signal.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlharami_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlharami_numba(open_, high, low, close, output)
        return output


def CDLHARAMICROSS(open_: Union[np.ndarray, list],
                   high: Union[np.ndarray, list],
                   low: Union[np.ndarray, list],
                   close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Harami Cross - 2-candle reversal where second candle is a doji within first

    A variation of the Harami pattern where the second candle is a doji (cross)
    completely contained within the first candle's body. This pattern is more
    significant than a regular Harami because the doji represents stronger
    indecision and equilibrium between buyers and sellers.

    Bullish Harami Cross: Large black candle followed by a doji completely
    within its body, signaling potential bullish reversal after downtrend.

    Bearish Harami Cross: Large white candle followed by a doji completely
    within its body, signaling potential bearish reversal after uptrend.

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
         100: Bullish Harami Cross pattern detected
        -100: Bearish Harami Cross pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - First candle: Large body (>70% of average)
    - Second candle: Doji (body <10% of range and average)
    - Second candle (open and close) completely within first body
    - Bullish: Black first, doji second
    - Bearish: White first, doji second

    This pattern is considered more reliable than a regular Harami due to
    the strong indecision signal from the doji. It often precedes significant
    reversals when appearing at the end of established trends.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlharamicross_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlharamicross_numba(open_, high, low, close, output)
        return output


def CDLHIGHWAVE(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    High Wave - Doji with very long upper and lower shadows

    A doji candlestick pattern with exceptionally long shadows in both
    directions, resembling a wave. This pattern indicates extreme volatility
    and strong indecision in the market.

    The long shadows show that both buyers and sellers made significant moves
    during the session, but neither group maintained control, resulting in a
    close near the open. This extreme indecision can signal potential reversal
    or continuation depending on the preceding trend and subsequent confirmation.

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
         100: High Wave pattern detected (indicates high volatility/indecision)
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Body must be very small (doji: <10% of range and average)
    - Both upper and lower shadows must be significant (>30% of range each)
    - Total shadows must dominate (>80% of range)

    The pattern indicates:
    - Extreme market indecision
    - High volatility
    - Battle between buyers and sellers
    - Potential for significant move in either direction

    This pattern is most significant when appearing at major support/resistance
    levels or at the end of extended trends. Subsequent price action provides
    the directional clue.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlhighwave_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlhighwave_numba(open_, high, low, close, output)
        return output


def CDLHIKKAKE(open_: Union[np.ndarray, list],
               high: Union[np.ndarray, list],
               low: Union[np.ndarray, list],
               close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Hikkake - 3-bar pattern with false inside day followed by breakout

    The Hikkake pattern is a clever reversal pattern that exploits failed inside bars.
    An inside bar (day 2) is followed by a breakout in the opposite direction on day 3,
    trapping traders who anticipated a continuation of the inside bar pattern.

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
         100: Bullish Hikkake (break below inside bar low)
        -100: Bearish Hikkake (break above inside bar high)
           0: No pattern

    Notes
    -----
    Pattern structure:
    - Day 1: Normal candle (reference bar)
    - Day 2: Inside day (high < day1 high AND low > day1 low)
    - Day 3: Breakout (bullish breaks below day 2 low, bearish breaks above day 2 high)

    The pattern is counterintuitive - a downward break is bullish because it's a
    "false breakdown" that reverses. This catches short-sellers off guard.

    The pattern works because:
    - Inside bars suggest consolidation
    - Initial breakout direction attracts momentum traders
    - Reversal traps those traders, forcing them to cover
    - This creates momentum in the opposite direction
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlhikkake_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlhikkake_numba(open_, high, low, close, output)
        return output


def CDLHIKKAKEMOD(open_: Union[np.ndarray, list],
                  high: Union[np.ndarray, list],
                  low: Union[np.ndarray, list],
                  close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Modified Hikkake - Hikkake pattern with delayed confirmation window

    The Modified Hikkake is a more reliable variation of the standard Hikkake pattern.
    Instead of requiring immediate day 3 confirmation, it allows a confirmation window
    of several days (typically 3-8 days) after the inside bar appears.

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
         100: Bullish Modified Hikkake
        -100: Bearish Modified Hikkake
           0: No pattern

    Notes
    -----
    Pattern structure:
    - Day 1: Normal candle (reference bar)
    - Day 2: Inside day (high < day1 high AND low > day1 low)
    - Day 3-8: Confirmation within this window

    Advantages over standard Hikkake:
    - More reliable due to delayed confirmation
    - Filters out false signals
    - Allows for market context to develop
    - Higher success rate but slower signal

    This pattern is particularly useful in choppy markets where immediate
    breakouts often fail. The delay requirement ensures the move has conviction.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlhikkakemod_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlhikkakemod_numba(open_, high, low, close, output)
        return output


def CDLHOMINGPIGEON(open_: Union[np.ndarray, list],
                    high: Union[np.ndarray, list],
                    low: Union[np.ndarray, list],
                    close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Homing Pigeon - Bullish reversal with two black candles, second contained in first

    The Homing Pigeon is a bullish reversal pattern that appears in downtrends.
    It consists of two consecutive black candles where the second candle's body
    is completely contained within the first candle's body, signaling weakening
    selling pressure.

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
         100: Bullish Homing Pigeon pattern
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Both candles must be black (bearish)
    - First candle has significant body (>70% of average)
    - Second candle's body completely contained within first body
    - Second candle is smaller than first

    The pattern indicates:
    - Selling momentum is decreasing
    - Bears are losing control
    - Potential bullish reversal ahead
    - Similar to Harami but both candles are black

    The name comes from the image of a pigeon returning home (to higher prices).
    The smaller second candle suggests sellers are exhausted after the strong
    down move, creating an opportunity for bulls to take control.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlhomingpigeon_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlhomingpigeon_numba(open_, high, low, close, output)
        return output


def CDLIDENTICAL3CROWS(open_: Union[np.ndarray, list],
                       high: Union[np.ndarray, list],
                       low: Union[np.ndarray, list],
                       close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Identical Three Crows - Bearish reversal with three black candles and similar closes

    The Identical Three Crows pattern is a strong bearish reversal signal that appears
    at market tops. It consists of three consecutive black candles with very similar
    closing prices, indicating persistent and consistent selling pressure.

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
        -100: Bearish Identical Three Crows pattern
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Three consecutive black candles with significant bodies
    - Each candle opens within the prior candle's body
    - Closes are progressively lower
    - Closing prices are very similar (identical within 0.5% tolerance)

    The pattern indicates:
    - Strong, persistent selling pressure
    - Sellers are in complete control
    - High probability of continued downtrend
    - More bearish than regular Three Black Crows

    The "identical" closes suggest methodical, persistent selling rather than
    panic. This organized selling is often more bearish because it indicates
    large institutional distribution rather than emotional retail selling.

    This pattern is most reliable when it appears after an extended uptrend or
    at a major resistance level.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlidentical3crows_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlidentical3crows_numba(open_, high, low, close, output)
        return output


def CDLINNECK(open_: Union[np.ndarray, list],
              high: Union[np.ndarray, list],
              low: Union[np.ndarray, list],
              close: Union[np.ndarray, list]) -> np.ndarray:
    """
    In-Neck - Bearish continuation with white candle closing at prior low

    The In-Neck pattern is a bearish continuation pattern that appears in downtrends.
    After a black candle, a white candle opens below the prior low but closes right
    at the prior candle's low (the "neck" level), showing that buying pressure was
    unable to push prices higher and selling is likely to resume.

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
        -100: Bearish In-Neck pattern
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - First candle: Black with significant body
    - Second candle: White, opens below first candle's low (gap down)
    - Second candle closes at/near first candle's low (within 1% tolerance)

    The pattern indicates:
    - Failed rally attempt
    - Selling pressure resumes quickly
    - Bearish continuation likely
    - Similar to On-Neck but closes exactly at low

    The pattern is bearish because:
    - Despite gap down, bulls couldn't push prices up
    - Close at prior low shows strong resistance
    - Trapped longs will likely exit, adding to selling pressure

    This pattern is most reliable in established downtrends and should be
    confirmed with subsequent price action.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlinneck_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlinneck_numba(open_, high, low, close, output)
        return output


def CDLINVERTEDHAMMER(open_: Union[np.ndarray, list],
                      high: Union[np.ndarray, list],
                      low: Union[np.ndarray, list],
                      close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Inverted Hammer - Single candle bullish reversal with small body and long upper shadow

    The Inverted Hammer is a bullish reversal pattern that appears at market bottoms.
    Despite its bearish appearance (long upper shadow), it suggests buyers pushed prices
    higher during the session, indicating potential reversal from downtrend.

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
         100: Bullish Inverted Hammer pattern
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Small body (<30% of range) at lower end of candle
    - Long upper shadow (>2x body, >50% of range)
    - Small lower shadow (<20% of range)
    - Body at bottom (lower shadow + body < 50% of range)

    The pattern indicates:
    - Buyers testing higher prices (upper shadow)
    - Potential bullish reversal from downtrend
    - Opposite of Shooting Star pattern
    - Requires confirmation from next candle

    The upper shadow shows that buyers pushed prices significantly higher during
    the session, even though sellers pushed them back down by the close. This
    buyer activity after a downtrend suggests potential reversal.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlinvertedhammer_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlinvertedhammer_numba(open_, high, low, close, output)
        return output


def CDLKICKING(open_: Union[np.ndarray, list],
               high: Union[np.ndarray, list],
               low: Union[np.ndarray, list],
               close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Kicking - Strong reversal with two opposite marubozu candles that gap

    The Kicking pattern is one of the strongest reversal signals in candlestick
    analysis. It consists of two consecutive marubozu candles of opposite colors
    that gap away from each other, showing a sudden and dramatic shift in sentiment.

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
         100: Bullish Kicking (black marubozu followed by gapping white marubozu)
        -100: Bearish Kicking (white marubozu followed by gapping black marubozu)
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Both candles are marubozu (body >95% of range)
    - Opposite colors (first black, second white OR first white, second black)
    - Gap between candles (second opens beyond first's close)

    The pattern indicates:
    - Sudden shift in market sentiment
    - Very strong reversal signal
    - Gap shows conviction in new direction
    - Both sides (bulls and bears) showing extreme strength

    The name "kicking" refers to how the market kicks away from its previous
    direction with violent momentum. This is one of the most reliable reversal
    patterns when it appears.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlkicking_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlkicking_numba(open_, high, low, close, output)
        return output


def CDLKICKINGBYLENGTH(open_: Union[np.ndarray, list],
                       high: Union[np.ndarray, list],
                       low: Union[np.ndarray, list],
                       close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Kicking By Length - Kicking pattern where second marubozu is longer than first

    This is an enhanced version of the Kicking pattern with an additional requirement:
    the second candle's body must be longer than the first. This shows not just a
    reversal, but accelerating momentum in the new direction.

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
         100: Bullish Kicking By Length
        -100: Bearish Kicking By Length
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Both candles are marubozu (body >95% of range)
    - Opposite colors
    - Gap between candles
    - Second candle body longer than first (additional requirement)

    The pattern indicates:
    - Even stronger reversal than regular Kicking
    - Accelerating momentum in new direction
    - High confidence reversal signal
    - Second wave stronger than first

    The length requirement filters for the most powerful reversals where
    the new directional move has more force than the previous one. This
    makes it an even more reliable signal than standard Kicking.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlkickingbylength_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlkickingbylength_numba(open_, high, low, close, output)
        return output


def CDLLADDERBOTTOM(open_: Union[np.ndarray, list],
                    high: Union[np.ndarray, list],
                    low: Union[np.ndarray, list],
                    close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Ladder Bottom - Rare 5-candle bullish reversal pattern

    The Ladder Bottom is a rare but powerful bullish reversal pattern that appears
    after extended downtrends. It shows seller exhaustion through a sequence of
    black candles that progressively weaken, followed by a strong white candle.

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
         100: Bullish Ladder Bottom pattern
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - First three candles: Black with significant bodies, declining closes
    - Fourth candle: Black, opens lower than third (exhaustion gap)
    - Fifth candle: White, closes above fourth's open

    The pattern indicates:
    - Seller exhaustion after extended decline
    - Strong bullish reversal signal
    - The "ladder" of declining closes shows organized selling
    - The gap down then reversal shows capitulation

    The name comes from the visual appearance of three descending black candles
    forming a ladder down, with the fourth candle's gap showing exhaustion,
    and the fifth white candle reversing the trend. Rare but highly reliable.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdlladderbottom_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlladderbottom_numba(open_, high, low, close, output)
        return output


def CDLLONGLEGGEDDOJI(open_: Union[np.ndarray, list],
                      high: Union[np.ndarray, list],
                      low: Union[np.ndarray, list],
                      close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Long-Legged Doji - Doji with very long upper and lower shadows

    The Long-Legged Doji is a powerful indecision pattern that shows extreme
    volatility and battle between buyers and sellers. Both sides made strong
    moves but neither could maintain control, resulting in a close near the open.

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
         100: Long-Legged Doji pattern detected
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Very small body (doji: <10% of range and average)
    - Both shadows very long (>40% of range each)
    - Total shadows dominate (>85% of range)

    The pattern indicates:
    - Extreme market indecision
    - High volatility during session
    - Major battle between bulls and bears
    - Potential reversal or continuation
    - Requires next-candle confirmation

    Similar to High Wave Candle but with stricter requirements for shadow length.
    The "long legs" refer to the extended upper and lower shadows. Most significant
    at major support/resistance levels or after extended trends.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdllongleggeddoji_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdllongleggeddoji_numba(open_, high, low, close, output)
        return output


def CDLLONGLINE(open_: Union[np.ndarray, list],
                high: Union[np.ndarray, list],
                low: Union[np.ndarray, list],
                close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Long Line - Single candle with very long body showing strong conviction

    The Long Line pattern identifies candles with very long bodies (>130% of average),
    indicating strong directional movement and conviction by either buyers or sellers.

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
         100: White Long Line (strong buying pressure)
        -100: Black Long Line (strong selling pressure)
           0: No pattern

    Notes
    -----
    Pattern requirements:
    - Body must be significantly longer than average (>130% of 10-bar average)

    The pattern indicates:
    - Strong directional movement
    - High conviction from market participants
    - Opposite of Short Line pattern
    - White Long Line: Sustained buying pressure
    - Black Long Line: Sustained selling pressure

    A long body shows that one side (bulls or bears) was in complete control
    during the session, creating strong momentum in one direction. This is often
    a continuation signal in the direction of the long candle.
    """
    # Convert inputs to numpy arrays
    open_, high, low, close = [np.asarray(x, dtype=np.float64) for x in [open_, high, low, close]]

    # Validate input
    n = len(open_)
    if not all(len(x) == n for x in [high, low, close]):
        raise ValueError("All input arrays must have the same length")

    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # Use GPU or CPU backend
    if get_backend() == "gpu":
        return _cdllongline_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdllongline_numba(open_, high, low, close, output)
        return output


