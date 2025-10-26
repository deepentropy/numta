"""
Pattern Recognition - Candlestick pattern recognition functions

This module implements candlestick pattern recognition compatible with TA-Lib.
Patterns return integer values:
    100: Bullish pattern
   -100: Bearish pattern
      0: No pattern
"""

import numpy as np
from typing import Union
from numba import jit


# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlmarubozu_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled Marubozu pattern recognition (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    Marubozu is a candlestick with little to no wicks/shadows.
    - Bullish Marubozu: Close near high, Open near low (strong buying)
    - Bearish Marubozu: Close near low, Open near high (strong selling)

    The pattern is identified when the body comprises at least 95% of the
    high-low range, indicating very little upper or lower shadow.
    """
    n = len(open_)

    for i in range(n):
        # Calculate body and total range
        body = abs(close[i] - open_[i])
        total_range = high[i] - low[i]

        # Avoid division by zero
        if total_range == 0.0:
            output[i] = 0
            continue

        # Check if body is at least 95% of total range (very small wicks)
        body_ratio = body / total_range

        if body_ratio >= 0.95:
            # Check if bullish (close > open) or bearish (close < open)
            if close[i] > open_[i]:
                # Bullish Marubozu: white/green candle
                output[i] = 100
            elif close[i] < open_[i]:
                # Bearish Marubozu: black/red candle
                output[i] = -100
            else:
                # Close == Open (doji-like), not a Marubozu
                output[i] = 0
        else:
            # Body ratio too small, not a Marubozu
            output[i] = 0


# GPU (CuPy) implementation
def _cdlmarubozu_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray) -> np.ndarray:
    """
    CuPy-based Marubozu pattern recognition for GPU

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
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    # Calculate body and total range
    body = cp.abs(close_gpu - open_gpu)
    total_range = high_gpu - low_gpu

    # Initialize output
    output = cp.zeros(len(open_), dtype=cp.int32)

    # Calculate body ratio (avoid division by zero)
    # Use where to handle division by zero
    body_ratio = cp.where(total_range > 0, body / total_range, 0.0)

    # Identify Marubozu patterns (body >= 95% of range)
    is_marubozu = body_ratio >= 0.95

    # Bullish Marubozu (close > open)
    is_bullish = (close_gpu > open_gpu) & is_marubozu
    output = cp.where(is_bullish, 100, output)

    # Bearish Marubozu (close < open)
    is_bearish = (close_gpu < open_gpu) & is_marubozu
    output = cp.where(is_bearish, -100, output)

    # Transfer back to CPU
    return cp.asnumpy(output)


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
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cdlmarubozu_cupy(open_, high, low, close)
    else:
        # Use CPU implementation (default)
        output = np.zeros(n, dtype=np.int32)
        _cdlmarubozu_numba(open_, high, low, close, output)
        return output


# ==================== CDLMATCHINGLOW ====================

# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlmatchinglow_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled Matching Low pattern recognition (in-place)

    The Matching Low is a bullish reversal pattern consisting of two consecutive
    bearish candles where the closing prices match within a tolerance.

    Pattern Requirements:
    - First candle: Black (bearish) candle
    - Second candle: Black candle with close equal to previous close
    - Both candles should have similar lows (matching lows)
    """
    n = len(open_)

    # Need at least 2 candles
    if n < 2:
        return

    # Calculate average true range for tolerance (using last 10 periods or available)
    lookback = min(10, n)
    avg_range = 0.0
    for i in range(max(0, n - lookback), n):
        avg_range += high[i] - low[i]
    if lookback > 0:
        avg_range /= lookback

    # Tolerance is 0.1% of average range (can be adjusted)
    tolerance = avg_range * 0.001

    # First candle cannot form pattern
    output[0] = 0

    for i in range(1, n):
        # Check if both candles are black (bearish)
        is_black_1 = close[i-1] < open_[i-1]
        is_black_2 = close[i] < open_[i]

        if is_black_1 and is_black_2:
            # Check if closes match within tolerance
            close_diff = abs(close[i] - close[i-1])

            if close_diff <= tolerance:
                # Matching Low pattern found
                output[i] = 100
            else:
                output[i] = 0
        else:
            output[i] = 0


# GPU (CuPy) implementation
def _cdlmatchinglow_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                         close: np.ndarray) -> np.ndarray:
    """
    CuPy-based Matching Low pattern recognition for GPU
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    if n < 2:
        return cp.asnumpy(output)

    # Calculate average true range for tolerance
    lookback = min(10, n)
    avg_range = cp.mean(high_gpu[-lookback:] - low_gpu[-lookback:])
    tolerance = avg_range * 0.001

    # Check if candles are black (bearish)
    is_black = close_gpu < open_gpu

    # Shift arrays to compare with previous candle
    is_black_prev = cp.roll(is_black, 1)
    close_prev = cp.roll(close_gpu, 1)

    # Check if closes match within tolerance
    close_match = cp.abs(close_gpu - close_prev) <= tolerance

    # Pattern: both candles black and closes match
    pattern = is_black & is_black_prev & close_match

    # Set output (skip first candle)
    output[1:] = cp.where(pattern[1:], 100, 0)

    # Transfer back to CPU
    return cp.asnumpy(output)


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
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cdlmatchinglow_cupy(open_, high, low, close)
    else:
        # Use CPU implementation (default)
        output = np.zeros(n, dtype=np.int32)
        _cdlmatchinglow_numba(open_, high, low, close, output)
        return output


# ==================== CDLMATHOLD ====================

# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlmathold_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, penetration: float, output: np.ndarray) -> None:
    """
    Numba-compiled Mat Hold pattern recognition (in-place)

    The Mat Hold is a bullish continuation pattern consisting of five candles:
    1. Long white (bullish) candle
    2. Small black candle gapping up from first
    3-4. Small candles (any color) continuing downward but staying within first candle's body
    5. White candle closing above the highs of candles 2-4

    The pattern indicates a brief consolidation/pullback in an uptrend before continuation.
    """
    n = len(open_)

    # Need at least 5 candles
    if n < 5:
        return

    # Calculate average body sizes for classification
    for i in range(4, n):
        # First candle: long white candle
        body_1 = close[i-4] - open_[i-4]
        is_white_1 = close[i-4] > open_[i-4]

        # Calculate average body for "long" determination
        # Use candles BEFORE the pattern (i-5 backwards) to avoid including pattern candles
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-15), max(0, i-5)):
            avg_body += abs(close[j] - open_[j])
            count += 1
        if count > 0:
            avg_body /= count
        else:
            # If not enough history, use a default
            avg_body = abs(body_1) / 2.0  # Assume pattern candle is larger than average

        # First candle must be long white
        if not (is_white_1 and body_1 > avg_body * 1.5):
            output[i] = 0
            continue

        # Second candle: small black candle with upside gap
        is_black_2 = close[i-3] < open_[i-3]
        body_2 = abs(close[i-3] - open_[i-3])
        has_gap_up = open_[i-3] > close[i-4]

        if not (is_black_2 and body_2 < avg_body * 0.5 and has_gap_up):
            output[i] = 0
            continue

        # Third and fourth candles: small bodies, falling pattern
        body_3 = abs(close[i-2] - open_[i-2])
        body_4 = abs(close[i-1] - open_[i-1])

        if not (body_3 < avg_body * 0.5 and body_4 < avg_body * 0.5):
            output[i] = 0
            continue

        # Check that candles 3-4 are falling (each close lower than previous open)
        falling_pattern = close[i-2] < open_[i-3] and high[i-1] < high[i-2]

        if not falling_pattern:
            output[i] = 0
            continue

        # Check penetration: lows of candles 2-4 should stay above penetration level
        penetration_level = close[i-4] - (body_1 * penetration)
        within_body = (low[i-3] >= penetration_level and
                      low[i-2] >= penetration_level and
                      low[i-1] >= penetration_level)

        if not within_body:
            output[i] = 0
            continue

        # Fifth candle: white candle closing above highs of candles 2-4
        is_white_5 = close[i] > open_[i]
        max_high_234 = max(high[i-3], max(high[i-2], high[i-1]))
        closes_above = close[i] > max_high_234
        opens_above = open_[i] > close[i-1]

        if is_white_5 and opens_above and closes_above:
            # Mat Hold pattern found
            output[i] = 100
        else:
            output[i] = 0


# GPU (CuPy) implementation
def _cdlmathold_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, penetration: float) -> np.ndarray:
    """
    CuPy-based Mat Hold pattern recognition for GPU
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    if n < 5:
        return cp.asnumpy(output)

    # Calculate bodies
    body = close_gpu - open_gpu
    abs_body = cp.abs(body)

    # Rolling average of body size (last 10 periods)
    # Using simple convolution for rolling mean
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)  # Avoid division by zero

    # For each position i (representing the 5th candle)
    for i in range(4, n):
        # Check first candle: long white
        is_white_1 = body[i-4] > 0
        is_long_1 = abs_body[i-4] > avg_body[i-4] * 1.5

        # Second candle: small black with gap up
        is_black_2 = body[i-3] < 0
        is_small_2 = abs_body[i-3] < avg_body[i-3] * 0.5
        has_gap_up = open_gpu[i-3] > close_gpu[i-4]

        # Third and fourth: small bodies
        is_small_3 = abs_body[i-2] < avg_body[i-2] * 0.5
        is_small_4 = abs_body[i-1] < avg_body[i-1] * 0.5

        # Falling pattern
        falling = (close_gpu[i-2] < open_gpu[i-3]) and (high_gpu[i-1] < high_gpu[i-2])

        # Penetration check
        penetration_level = close_gpu[i-4] - (abs_body[i-4] * penetration)
        within_body = ((low_gpu[i-3] >= penetration_level) and
                      (low_gpu[i-2] >= penetration_level) and
                      (low_gpu[i-1] >= penetration_level))

        # Fifth candle: white, opens above 4th close, closes above max high of 2-4
        is_white_5 = body[i] > 0
        opens_above = open_gpu[i] > close_gpu[i-1]
        max_high_234 = cp.maximum(high_gpu[i-3], cp.maximum(high_gpu[i-2], high_gpu[i-1]))
        closes_above = close_gpu[i] > max_high_234

        # Combine all conditions
        pattern = (is_white_1 and is_long_1 and
                  is_black_2 and is_small_2 and has_gap_up and
                  is_small_3 and is_small_4 and
                  falling and within_body and
                  is_white_5 and opens_above and closes_above)

        if pattern:
            output[i] = 100

    # Transfer back to CPU
    return cp.asnumpy(output)


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
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cdlmathold_cupy(open_, high, low, close, penetration)
    else:
        # Use CPU implementation (default)
        output = np.zeros(n, dtype=np.int32)
        _cdlmathold_numba(open_, high, low, close, penetration, output)
        return output


# ==================== CDLMORNINGDOJISTAR ====================

# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlmorningdojistar_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray, penetration: float, output: np.ndarray) -> None:
    """
    Numba-compiled Morning Doji Star pattern recognition (in-place)

    3-candle bullish reversal pattern:
    1. Long black candle
    2. Doji gapping down
    3. White candle closing well into first candle's body
    """
    n = len(open_)

    if n < 3:
        return

    # Calculate average body sizes
    for i in range(2, n):
        # Calculate average body for context
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-12), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        if count > 0:
            avg_body /= count
        else:
            avg_body = abs(close[i-2] - open_[i-2])

        # First candle: long black
        body_1 = close[i-2] - open_[i-2]
        is_black_1 = close[i-2] < open_[i-2]
        is_long_1 = abs(body_1) > avg_body * 1.2

        if not (is_black_1 and is_long_1):
            output[i] = 0
            continue

        # Second candle: doji with gap down
        body_2 = abs(close[i-1] - open_[i-1])
        is_doji = body_2 <= avg_body * 0.1
        has_gap_down = open_[i-1] < close[i-2]

        if not (is_doji and has_gap_down):
            output[i] = 0
            continue

        # Third candle: white with penetration
        is_white_3 = close[i] > open_[i]
        body_3 = abs(close[i] - open_[i])
        is_significant_3 = body_3 > avg_body * 0.3
        penetrates = close[i] > close[i-2] + abs(body_1) * penetration

        if is_white_3 and is_significant_3 and penetrates:
            output[i] = 100
        else:
            output[i] = 0


# GPU (CuPy) implementation
def _cdlmorningdojistar_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, penetration: float) -> np.ndarray:
    """CuPy-based Morning Doji Star pattern recognition for GPU"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    if n < 3:
        return cp.asnumpy(output)

    # Calculate bodies
    body = close_gpu - open_gpu
    abs_body = cp.abs(body)

    # Rolling average
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        # First candle: long black
        is_black_1 = body[i-2] < 0
        is_long_1 = abs_body[i-2] > avg_body[i-2] * 1.2

        # Second candle: doji with gap down
        is_doji = abs_body[i-1] <= avg_body[i-1] * 0.1
        has_gap_down = open_gpu[i-1] < close_gpu[i-2]

        # Third candle: white with penetration
        is_white_3 = body[i] > 0
        is_significant_3 = abs_body[i] > avg_body[i] * 0.3
        penetrates = close_gpu[i] > close_gpu[i-2] + abs_body[i-2] * penetration

        if (is_black_1 and is_long_1 and is_doji and has_gap_down and
            is_white_3 and is_significant_3 and penetrates):
            output[i] = 100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    backend = get_backend()

    if backend == "gpu":
        return _cdlmorningdojistar_cupy(open_, high, low, close, penetration)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlmorningdojistar_numba(open_, high, low, close, penetration, output)
        return output


# ==================== CDLMORNINGSTAR ====================

# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlmorningstar_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                         close: np.ndarray, penetration: float, output: np.ndarray) -> None:
    """
    Numba-compiled Morning Star pattern recognition (in-place)

    3-candle bullish reversal pattern:
    1. Long black candle
    2. Short body (any color) gapping down
    3. White candle closing well into first candle's body
    """
    n = len(open_)

    if n < 3:
        return

    # Calculate average body sizes
    for i in range(2, n):
        # Calculate average body for context
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-12), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        if count > 0:
            avg_body /= count
        else:
            avg_body = abs(close[i-2] - open_[i-2])

        # First candle: long black
        body_1 = close[i-2] - open_[i-2]
        is_black_1 = close[i-2] < open_[i-2]
        is_long_1 = abs(body_1) > avg_body * 1.2

        if not (is_black_1 and is_long_1):
            output[i] = 0
            continue

        # Second candle: short body with gap down
        body_2 = abs(close[i-1] - open_[i-1])
        is_short_2 = body_2 <= avg_body * 0.5
        has_gap_down = max(open_[i-1], close[i-1]) < close[i-2]

        if not (is_short_2 and has_gap_down):
            output[i] = 0
            continue

        # Third candle: white with penetration
        is_white_3 = close[i] > open_[i]
        body_3 = abs(close[i] - open_[i])
        is_significant_3 = body_3 > avg_body * 0.3
        penetrates = close[i] > close[i-2] + abs(body_1) * penetration

        if is_white_3 and is_significant_3 and penetrates:
            output[i] = 100
        else:
            output[i] = 0


# GPU (CuPy) implementation
def _cdlmorningstar_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, penetration: float) -> np.ndarray:
    """CuPy-based Morning Star pattern recognition for GPU"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    if n < 3:
        return cp.asnumpy(output)

    # Calculate bodies
    body = close_gpu - open_gpu
    abs_body = cp.abs(body)

    # Rolling average
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        # First candle: long black
        is_black_1 = body[i-2] < 0
        is_long_1 = abs_body[i-2] > avg_body[i-2] * 1.2

        # Second candle: short body with gap down
        is_short_2 = abs_body[i-1] <= avg_body[i-1] * 0.5
        has_gap_down = cp.maximum(open_gpu[i-1], close_gpu[i-1]) < close_gpu[i-2]

        # Third candle: white with penetration
        is_white_3 = body[i] > 0
        is_significant_3 = abs_body[i] > avg_body[i] * 0.3
        penetrates = close_gpu[i] > close_gpu[i-2] + abs_body[i-2] * penetration

        if (is_black_1 and is_long_1 and is_short_2 and has_gap_down and
            is_white_3 and is_significant_3 and penetrates):
            output[i] = 100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    backend = get_backend()

    if backend == "gpu":
        return _cdlmorningstar_cupy(open_, high, low, close, penetration)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlmorningstar_numba(open_, high, low, close, penetration, output)
        return output


# ==================== CDLONNECK ====================

# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlonneck_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled On-Neck pattern recognition (in-place)

    2-candle bearish continuation pattern:
    1. Long black candle
    2. White candle opening below prior low, closing at prior low
    """
    n = len(open_)

    if n < 2:
        return

    # Calculate average body sizes
    for i in range(1, n):
        # Calculate average body for context
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-12), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        if count > 0:
            avg_body /= count
        else:
            avg_body = abs(close[i-1] - open_[i-1])

        # Calculate average range for tolerance
        avg_range = 0.0
        for j in range(max(0, i-10), i):
            avg_range += high[j] - low[j]
        if count > 0:
            avg_range /= min(10, i)
        else:
            avg_range = high[i-1] - low[i-1]

        tolerance = avg_range * 0.002  # 0.2% tolerance

        # First candle: long black
        is_black_1 = close[i-1] < open_[i-1]
        body_1 = abs(close[i-1] - open_[i-1])
        is_long_1 = body_1 > avg_body * 1.2

        if not (is_black_1 and is_long_1):
            output[i] = 0
            continue

        # Second candle: white opening below prior low
        is_white_2 = close[i] > open_[i]
        opens_below = open_[i] < low[i-1]

        # Close matches prior low (within tolerance)
        close_matches_low = (close[i] >= low[i-1] - tolerance and
                            close[i] <= low[i-1] + tolerance)

        if is_white_2 and opens_below and close_matches_low:
            output[i] = -100  # Bearish continuation
        else:
            output[i] = 0


# GPU (CuPy) implementation
def _cdlonneck_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray) -> np.ndarray:
    """CuPy-based On-Neck pattern recognition for GPU"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    if n < 2:
        return cp.asnumpy(output)

    # Calculate bodies and ranges
    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    price_range = high_gpu - low_gpu

    # Rolling averages
    kernel_body = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel_body, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    avg_range = cp.convolve(price_range, kernel_body, mode='same')
    avg_range = cp.where(avg_range > 0, avg_range, 1.0)

    tolerance = avg_range * 0.002

    for i in range(1, n):
        # First candle: long black
        is_black_1 = body[i-1] < 0
        is_long_1 = abs_body[i-1] > avg_body[i-1] * 1.2

        # Second candle: white opening below prior low
        is_white_2 = body[i] > 0
        opens_below = open_gpu[i] < low_gpu[i-1]

        # Close matches prior low
        close_matches_low = ((close_gpu[i] >= low_gpu[i-1] - tolerance[i]) and
                            (close_gpu[i] <= low_gpu[i-1] + tolerance[i]))

        if is_black_1 and is_long_1 and is_white_2 and opens_below and close_matches_low:
            output[i] = -100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    backend = get_backend()

    if backend == "gpu":
        return _cdlonneck_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlonneck_numba(open_, high, low, close, output)
        return output


# ==================== CDLPIERCING ====================

# CPU (Numba) implementation
@jit(nopython=True, cache=True)
def _cdlpiercing_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled Piercing Pattern recognition (in-place)

    2-candle bullish reversal pattern:
    1. Long black candle
    2. Long white candle opening below prior low, closing above 50% of prior body
    """
    n = len(open_)

    if n < 2:
        return

    # Calculate average body sizes
    for i in range(1, n):
        # Calculate average body for context
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-12), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        if count > 0:
            avg_body /= count
        else:
            avg_body = abs(close[i-1] - open_[i-1])

        # First candle: long black
        is_black_1 = close[i-1] < open_[i-1]
        body_1 = abs(close[i-1] - open_[i-1])
        is_long_1 = body_1 > avg_body * 1.2

        if not (is_black_1 and is_long_1):
            output[i] = 0
            continue

        # Second candle: long white
        is_white_2 = close[i] > open_[i]
        body_2 = abs(close[i] - open_[i])
        is_long_2 = body_2 > avg_body * 1.2

        if not (is_white_2 and is_long_2):
            output[i] = 0
            continue

        # Gap down opening
        opens_below_low = open_[i] < low[i-1]

        # Close above midpoint of prior body
        midpoint = close[i-1] + body_1 * 0.5
        closes_above_mid = close[i] > midpoint

        # But not above prior open (distinguishes from engulfing)
        closes_below_open = close[i] < open_[i-1]

        if opens_below_low and closes_above_mid and closes_below_open:
            output[i] = 100  # Bullish reversal
        else:
            output[i] = 0


# GPU (CuPy) implementation
def _cdlpiercing_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> np.ndarray:
    """CuPy-based Piercing Pattern recognition for GPU"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    open_gpu = cp.asarray(open_)
    high_gpu = cp.asarray(high)
    low_gpu = cp.asarray(low)
    close_gpu = cp.asarray(close)

    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    if n < 2:
        return cp.asnumpy(output)

    # Calculate bodies
    body = close_gpu - open_gpu
    abs_body = cp.abs(body)

    # Rolling average
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(1, n):
        # First candle: long black
        is_black_1 = body[i-1] < 0
        is_long_1 = abs_body[i-1] > avg_body[i-1] * 1.2

        # Second candle: long white
        is_white_2 = body[i] > 0
        is_long_2 = abs_body[i] > avg_body[i] * 1.2

        # Gap down and penetration
        opens_below_low = open_gpu[i] < low_gpu[i-1]
        midpoint = close_gpu[i-1] + abs_body[i-1] * 0.5
        closes_above_mid = close_gpu[i] > midpoint
        closes_below_open = close_gpu[i] < open_gpu[i-1]

        if (is_black_1 and is_long_1 and is_white_2 and is_long_2 and
            opens_below_low and closes_above_mid and closes_below_open):
            output[i] = 100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    backend = get_backend()

    if backend == "gpu":
        return _cdlpiercing_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlpiercing_numba(open_, high, low, close, output)
        return output


# ==================== CDLRICKSHAWMAN ====================

@jit(nopython=True, cache=True)
def _cdlrickshawman_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                         close: np.ndarray, output: np.ndarray) -> None:
    """Rickshaw Man: Doji with long upper and lower shadows"""
    n = len(open_)

    for i in range(n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else abs(close[i] - open_[i])

        body = abs(close[i] - open_[i])
        upper_shadow = high[i] - max(open_[i], close[i])
        lower_shadow = min(open_[i], close[i]) - low[i]
        total_range = high[i] - low[i]

        # Doji body
        is_doji = body <= avg_body * 0.1

        # Long shadows on both sides
        has_long_shadows = (upper_shadow > avg_body * 1.0 and
                           lower_shadow > avg_body * 1.0)

        # Body near midpoint
        midpoint = low[i] + total_range / 2
        tolerance = total_range * 0.15
        body_near_mid = (min(open_[i], close[i]) <= midpoint + tolerance and
                        max(open_[i], close[i]) >= midpoint - tolerance)

        if is_doji and has_long_shadows and body_near_mid:
            output[i] = 100  # Neutral/indecision
        else:
            output[i] = 0


def _cdlrickshawman_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray) -> np.ndarray:
    """CuPy-based Rickshaw Man pattern recognition"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    body = cp.abs(close_gpu - open_gpu)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(n):
        upper_shadow = high_gpu[i] - cp.maximum(open_gpu[i], close_gpu[i])
        lower_shadow = cp.minimum(open_gpu[i], close_gpu[i]) - low_gpu[i]
        total_range = high_gpu[i] - low_gpu[i]

        is_doji = body[i] <= avg_body[i] * 0.1
        has_long_shadows = (upper_shadow > avg_body[i] * 1.0 and lower_shadow > avg_body[i] * 1.0)

        midpoint = low_gpu[i] + total_range / 2
        tolerance = total_range * 0.15
        body_near_mid = (cp.minimum(open_gpu[i], close_gpu[i]) <= midpoint + tolerance and
                        cp.maximum(open_gpu[i], close_gpu[i]) >= midpoint - tolerance)

        if is_doji and has_long_shadows and body_near_mid:
            output[i] = 100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlrickshawman_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlrickshawman_numba(open_, high, low, close, output)
        return output


# ==================== CDLRISEFALL3METHODS ====================

@jit(nopython=True, cache=True)
def _cdlrisefall3methods_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, output: np.ndarray) -> None:
    """Rising/Falling Three Methods: 5-candle continuation pattern"""
    n = len(open_)
    if n < 5:
        return

    for i in range(4, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-12), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        # 1st candle: long
        body_1 = close[i-4] - open_[i-4]
        is_long_1 = abs(body_1) > avg_body * 1.2

        # Candles 2-4: small, opposite color
        body_2, body_3, body_4 = close[i-3] - open_[i-3], close[i-2] - open_[i-2], close[i-1] - open_[i-1]
        all_small = (abs(body_2) < avg_body * 0.5 and
                    abs(body_3) < avg_body * 0.5 and
                    abs(body_4) < avg_body * 0.5)

        # 5th candle: long, same color as 1st
        body_5 = close[i] - open_[i]
        is_long_5 = abs(body_5) > avg_body * 1.2
        same_direction = (body_1 > 0 and body_5 > 0) or (body_1 < 0 and body_5 < 0)

        # Rising Three Methods
        if (body_1 > 0 and is_long_1 and all_small and
            body_2 < 0 and body_3 < 0 and body_4 < 0 and
            high[i-3] < high[i-4] and high[i-2] < high[i-4] and high[i-1] < high[i-4] and
            low[i-3] > low[i-4] and low[i-2] > low[i-4] and low[i-1] > low[i-4] and
            is_long_5 and body_5 > 0 and close[i] > close[i-4]):
            output[i] = 100
        # Falling Three Methods
        elif (body_1 < 0 and is_long_1 and all_small and
              body_2 > 0 and body_3 > 0 and body_4 > 0 and
              low[i-3] > low[i-4] and low[i-2] > low[i-4] and low[i-1] > low[i-4] and
              high[i-3] < high[i-4] and high[i-2] < high[i-4] and high[i-1] < high[i-4] and
              is_long_5 and body_5 < 0 and close[i] < close[i-4]):
            output[i] = -100
        else:
            output[i] = 0


def _cdlrisefall3methods_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray) -> np.ndarray:
    """CuPy-based Rising/Falling Three Methods"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 5:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(4, n):
        is_long_1 = abs_body[i-4] > avg_body[i-4] * 1.2
        all_small = (abs_body[i-3] < avg_body[i-3] * 0.5 and
                    abs_body[i-2] < avg_body[i-2] * 0.5 and
                    abs_body[i-1] < avg_body[i-1] * 0.5)
        is_long_5 = abs_body[i] > avg_body[i] * 1.2

        # Rising Three Methods
        if (body[i-4] > 0 and is_long_1 and all_small and
            body[i-3] < 0 and body[i-2] < 0 and body[i-1] < 0 and
            high_gpu[i-3] < high_gpu[i-4] and high_gpu[i-2] < high_gpu[i-4] and high_gpu[i-1] < high_gpu[i-4] and
            low_gpu[i-3] > low_gpu[i-4] and low_gpu[i-2] > low_gpu[i-4] and low_gpu[i-1] > low_gpu[i-4] and
            is_long_5 and body[i] > 0 and close_gpu[i] > close_gpu[i-4]):
            output[i] = 100
        # Falling Three Methods
        elif (body[i-4] < 0 and is_long_1 and all_small and
              body[i-3] > 0 and body[i-2] > 0 and body[i-1] > 0 and
              low_gpu[i-3] > low_gpu[i-4] and low_gpu[i-2] > low_gpu[i-4] and low_gpu[i-1] > low_gpu[i-4] and
              high_gpu[i-3] < high_gpu[i-4] and high_gpu[i-2] < high_gpu[i-4] and high_gpu[i-1] < high_gpu[i-4] and
              is_long_5 and body[i] < 0 and close_gpu[i] < close_gpu[i-4]):
            output[i] = -100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlrisefall3methods_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlrisefall3methods_numba(open_, high, low, close, output)
        return output


# ==================== CDLSEPARATINGLINES ====================

@jit(nopython=True, cache=True)
def _cdlseparatinglines_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray, output: np.ndarray) -> None:
    """Separating Lines: 2-candle continuation with matching opens"""
    n = len(open_)
    if n < 2:
        return

    for i in range(1, n):
        avg_body, avg_range = 0.0, 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            avg_range += high[j] - low[j]
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0
        avg_range = avg_range / count if count > 0 else 1.0

        # Opposite colors
        is_opposite = (close[i-1] > open_[i-1] and close[i] < open_[i]) or (close[i-1] < open_[i-1] and close[i] > open_[i])

        # Matching opens
        open_match = abs(open_[i] - open_[i-1]) <= avg_range * 0.002

        # Long body on second candle
        is_long_2 = abs(close[i] - open_[i]) > avg_body * 1.2

        upper_shadow = high[i] - max(open_[i], close[i])
        lower_shadow = min(open_[i], close[i]) - low[i]

        # Bullish: white candle with small lower shadow
        if is_opposite and open_match and is_long_2 and close[i] > open_[i] and lower_shadow < avg_body * 0.1:
            output[i] = 100
        # Bearish: black candle with small upper shadow
        elif is_opposite and open_match and is_long_2 and close[i] < open_[i] and upper_shadow < avg_body * 0.1:
            output[i] = -100
        else:
            output[i] = 0


def _cdlseparatinglines_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray) -> np.ndarray:
    """CuPy-based Separating Lines"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 2:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    price_range = high_gpu - low_gpu

    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)
    avg_range = cp.convolve(price_range, kernel, mode='same')
    avg_range = cp.where(avg_range > 0, avg_range, 1.0)

    for i in range(1, n):
        is_opposite = (body[i-1] > 0 and body[i] < 0) or (body[i-1] < 0 and body[i] > 0)
        open_match = cp.abs(open_gpu[i] - open_gpu[i-1]) <= avg_range[i] * 0.002
        is_long_2 = abs_body[i] > avg_body[i] * 1.2

        upper_shadow = high_gpu[i] - cp.maximum(open_gpu[i], close_gpu[i])
        lower_shadow = cp.minimum(open_gpu[i], close_gpu[i]) - low_gpu[i]

        if is_opposite and open_match and is_long_2 and body[i] > 0 and lower_shadow < avg_body[i] * 0.1:
            output[i] = 100
        elif is_opposite and open_match and is_long_2 and body[i] < 0 and upper_shadow < avg_body[i] * 0.1:
            output[i] = -100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlseparatinglines_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlseparatinglines_numba(open_, high, low, close, output)
        return output


# ==================== CDLSHOOTINGSTAR ====================

@jit(nopython=True, cache=True)
def _cdlshootingstar_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, output: np.ndarray) -> None:
    """Shooting Star: Small body, long upper shadow, gap up"""
    n = len(open_)
    if n < 2:
        return

    for i in range(1, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        body = abs(close[i] - open_[i])
        upper_shadow = high[i] - max(open_[i], close[i])
        lower_shadow = min(open_[i], close[i]) - low[i]

        # Small body
        is_small_body = body < avg_body * 0.3

        # Long upper shadow (at least 2x body)
        has_long_upper = upper_shadow > body * 2.0 and upper_shadow > avg_body * 1.0

        # Very short lower shadow
        has_short_lower = lower_shadow < avg_body * 0.1

        # Gap up from previous close
        has_gap_up = min(open_[i], close[i]) > max(open_[i-1], close[i-1])

        if is_small_body and has_long_upper and has_short_lower and has_gap_up:
            output[i] = -100  # Bearish
        else:
            output[i] = 0


def _cdlshootingstar_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                         close: np.ndarray) -> np.ndarray:
    """CuPy-based Shooting Star"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 2:
        return cp.asnumpy(output)

    body = cp.abs(close_gpu - open_gpu)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(1, n):
        upper_shadow = high_gpu[i] - cp.maximum(open_gpu[i], close_gpu[i])
        lower_shadow = cp.minimum(open_gpu[i], close_gpu[i]) - low_gpu[i]

        is_small_body = body[i] < avg_body[i] * 0.3
        has_long_upper = upper_shadow > body[i] * 2.0 and upper_shadow > avg_body[i] * 1.0
        has_short_lower = lower_shadow < avg_body[i] * 0.1
        has_gap_up = cp.minimum(open_gpu[i], close_gpu[i]) > cp.maximum(open_gpu[i-1], close_gpu[i-1])

        if is_small_body and has_long_upper and has_short_lower and has_gap_up:
            output[i] = -100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlshootingstar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlshootingstar_numba(open_, high, low, close, output)
        return output


# ==================== CDLSHORTLINE ====================

@jit(nopython=True, cache=True)
def _cdlshortline_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, output: np.ndarray) -> None:
    """Short Line Candle: Short real body with short shadows"""
    n = len(open_)

    for i in range(n):
        avg_body = 0.0
        avg_shadow = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            upper_shadow = high[j] - max(open_[j], close[j])
            lower_shadow = min(open_[j], close[j]) - low[j]
            avg_shadow += (upper_shadow + lower_shadow) / 2.0
            count += 1
        avg_body = avg_body / count if count > 0 else abs(close[i] - open_[i])
        avg_shadow = avg_shadow / count if count > 0 else 1.0

        body = abs(close[i] - open_[i])
        upper_shadow = high[i] - max(open_[i], close[i])
        lower_shadow = min(open_[i], close[i]) - low[i]

        # Short body and short shadows
        is_short_body = body < avg_body * 1.0
        is_short_upper = upper_shadow < avg_shadow * 1.0
        is_short_lower = lower_shadow < avg_shadow * 1.0

        if is_short_body and is_short_upper and is_short_lower:
            # Return based on candle color
            if close[i] >= open_[i]:
                output[i] = 100  # White/bullish
            else:
                output[i] = -100  # Black/bearish
        else:
            output[i] = 0


def _cdlshortline_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> np.ndarray:
    """CuPy-based Short Line Candle"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    body = cp.abs(close_gpu - open_gpu)
    upper_shadow = high_gpu - cp.maximum(open_gpu, close_gpu)
    lower_shadow = cp.minimum(open_gpu, close_gpu) - low_gpu
    avg_shadow = (upper_shadow + lower_shadow) / 2.0

    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(body, kernel, mode='same')
    avg_shadow_conv = cp.convolve(avg_shadow, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, body)
    avg_shadow_conv = cp.where(avg_shadow_conv > 0, avg_shadow_conv, 1.0)

    is_short_body = body < avg_body * 1.0
    is_short_upper = upper_shadow < avg_shadow_conv * 1.0
    is_short_lower = lower_shadow < avg_shadow_conv * 1.0
    is_white = close_gpu >= open_gpu

    pattern = is_short_body & is_short_upper & is_short_lower
    output = cp.where(pattern & is_white, 100, output)
    output = cp.where(pattern & ~is_white, -100, output)

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlshortline_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlshortline_numba(open_, high, low, close, output)
        return output


# ==================== CDLSPINNINGTOP ====================

@jit(nopython=True, cache=True)
def _cdlspinningtop_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, output: np.ndarray) -> None:
    """Spinning Top: Small body with long upper and lower shadows"""
    n = len(open_)

    for i in range(n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else abs(close[i] - open_[i])

        body = abs(close[i] - open_[i])
        upper_shadow = high[i] - max(open_[i], close[i])
        lower_shadow = min(open_[i], close[i]) - low[i]

        # Small body with both shadows > body
        is_small_body = body < avg_body * 1.0
        has_long_shadows = upper_shadow > body and lower_shadow > body

        if is_small_body and has_long_shadows:
            if close[i] >= open_[i]:
                output[i] = 100  # White
            else:
                output[i] = -100  # Black
        else:
            output[i] = 0


def _cdlspinningtop_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                         close: np.ndarray) -> np.ndarray:
    """CuPy-based Spinning Top"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    body = cp.abs(close_gpu - open_gpu)
    upper_shadow = high_gpu - cp.maximum(open_gpu, close_gpu)
    lower_shadow = cp.minimum(open_gpu, close_gpu) - low_gpu

    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, body)

    is_small_body = body < avg_body * 1.0
    has_long_shadows = (upper_shadow > body) & (lower_shadow > body)
    is_white = close_gpu >= open_gpu

    pattern = is_small_body & has_long_shadows
    output = cp.where(pattern & is_white, 100, output)
    output = cp.where(pattern & ~is_white, -100, output)

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlspinningtop_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlspinningtop_numba(open_, high, low, close, output)
        return output


# ==================== CDLSTALLEDPATTERN ====================

@jit(nopython=True, cache=True)
def _cdlstalledpattern_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, output: np.ndarray) -> None:
    """Stalled Pattern: Three white soldiers with weakening momentum"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-12), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        # All three must be white with consecutively higher closes
        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        all_white = body_1 > 0 and body_2 > 0 and body_3 > 0
        higher_closes = close[i] > close[i-1] > close[i-2]

        # First two candles: long bodies
        is_long_1 = abs(body_1) > avg_body * 1.0
        is_long_2 = abs(body_2) > avg_body * 1.0

        # Second candle: very short upper shadow
        upper_shadow_2 = high[i-1] - max(open_[i-1], close[i-1])
        short_upper_2 = upper_shadow_2 < avg_body * 0.1

        # Second opens within first body
        opens_near_1 = open_[i-1] > open_[i-2] and open_[i-1] <= close[i-2] + avg_body * 0.2

        # Third candle: small body
        is_small_3 = abs(body_3) < avg_body * 1.0

        # Third opens near top of second (riding on shoulder)
        rides_shoulder = open_[i] >= close[i-1] - abs(body_3) - avg_body * 0.2

        if (all_white and higher_closes and is_long_1 and is_long_2 and
            short_upper_2 and opens_near_1 and is_small_3 and rides_shoulder):
            output[i] = -100  # Bearish signal (momentum weakening)
        else:
            output[i] = 0


def _cdlstalledpattern_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                            close: np.ndarray) -> np.ndarray:
    """CuPy-based Stalled Pattern"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        all_white = body[i-2] > 0 and body[i-1] > 0 and body[i] > 0
        higher_closes = close_gpu[i] > close_gpu[i-1] > close_gpu[i-2]

        is_long_1 = abs_body[i-2] > avg_body[i-2] * 1.0
        is_long_2 = abs_body[i-1] > avg_body[i-1] * 1.0

        upper_shadow_2 = high_gpu[i-1] - cp.maximum(open_gpu[i-1], close_gpu[i-1])
        short_upper_2 = upper_shadow_2 < avg_body[i-1] * 0.1

        opens_near_1 = open_gpu[i-1] > open_gpu[i-2] and open_gpu[i-1] <= close_gpu[i-2] + avg_body[i-2] * 0.2

        is_small_3 = abs_body[i] < avg_body[i] * 1.0
        rides_shoulder = open_gpu[i] >= close_gpu[i-1] - abs_body[i] - avg_body[i] * 0.2

        if (all_white and higher_closes and is_long_1 and is_long_2 and
            short_upper_2 and opens_near_1 and is_small_3 and rides_shoulder):
            output[i] = -100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlstalledpattern_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlstalledpattern_numba(open_, high, low, close, output)
        return output


# ==================== CDLSTICKSANDWICH ====================

@jit(nopython=True, cache=True)
def _cdlsticksandwich_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                            close: np.ndarray, output: np.ndarray) -> None:
    """Stick Sandwich: Black-white-black with matching closes"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        # Pattern: black, white, black
        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        is_black_1 = body_1 < 0
        is_white_2 = body_2 > 0
        is_black_3 = body_3 < 0

        # Second candle's low > first's close
        gap_up = low[i-1] > close[i-2]

        # First and third closes match (within tolerance)
        tolerance = avg_body * 0.05
        closes_match = abs(close[i] - close[i-2]) <= tolerance

        if is_black_1 and is_white_2 and is_black_3 and gap_up and closes_match:
            output[i] = 100  # Bullish reversal
        else:
            output[i] = 0


def _cdlsticksandwich_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                           close: np.ndarray) -> np.ndarray:
    """CuPy-based Stick Sandwich"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        is_black_1 = body[i-2] < 0
        is_white_2 = body[i-1] > 0
        is_black_3 = body[i] < 0

        gap_up = low_gpu[i-1] > close_gpu[i-2]

        tolerance = avg_body[i] * 0.05
        closes_match = cp.abs(close_gpu[i] - close_gpu[i-2]) <= tolerance

        if is_black_1 and is_white_2 and is_black_3 and gap_up and closes_match:
            output[i] = 100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlsticksandwich_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlsticksandwich_numba(open_, high, low, close, output)
        return output


# ==================== CDLTAKURI ====================

@jit(nopython=True, cache=True)
def _cdltakuri_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, output: np.ndarray) -> None:
    """Takuri: Dragonfly Doji with very long lower shadow"""
    n = len(open_)

    for i in range(n):
        avg_body = 0.0
        avg_shadow = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            upper_shadow = high[j] - max(open_[j], close[j])
            lower_shadow = min(open_[j], close[j]) - low[j]
            avg_shadow += max(upper_shadow, lower_shadow)
            count += 1
        avg_body = avg_body / count if count > 0 else abs(close[i] - open_[i])
        avg_shadow = avg_shadow / count if count > 0 else 1.0

        body = abs(close[i] - open_[i])
        upper_shadow = high[i] - max(open_[i], close[i])
        lower_shadow = min(open_[i], close[i]) - low[i]

        # Doji body, very short upper shadow, very long lower shadow
        is_doji = body <= avg_body * 0.1
        short_upper = upper_shadow < avg_shadow * 0.1
        long_lower = lower_shadow > avg_shadow * 2.0

        if is_doji and short_upper and long_lower:
            output[i] = 100  # Bullish reversal
        else:
            output[i] = 0


def _cdltakuri_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray) -> np.ndarray:
    """CuPy-based Takuri"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)

    body = cp.abs(close_gpu - open_gpu)
    upper_shadow = high_gpu - cp.maximum(open_gpu, close_gpu)
    lower_shadow = cp.minimum(open_gpu, close_gpu) - low_gpu
    max_shadow = cp.maximum(upper_shadow, lower_shadow)

    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(body, kernel, mode='same')
    avg_shadow = cp.convolve(max_shadow, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, body)
    avg_shadow = cp.where(avg_shadow > 0, avg_shadow, 1.0)

    is_doji = body <= avg_body * 0.1
    short_upper = upper_shadow < avg_shadow * 0.1
    long_lower = lower_shadow > avg_shadow * 2.0

    pattern = is_doji & short_upper & long_lower
    output = cp.where(pattern, 100, 0)

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdltakuri_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdltakuri_numba(open_, high, low, close, output)
        return output


# ==================== CDLTASUKIGAP ====================

@jit(nopython=True, cache=True)
def _cdltasukigap_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, output: np.ndarray) -> None:
    """Tasuki Gap: Gap followed by pullback that doesn't close gap"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        # Bodies approximately equal
        tolerance = avg_body * 0.2
        bodies_similar = abs(abs(body_2) - abs(body_3)) <= tolerance

        # Upside Tasuki Gap
        gap_up = low[i-1] > high[i-2]
        is_white_1 = body_1 > 0
        is_white_2 = body_2 > 0
        is_black_3 = body_3 < 0
        opens_in_body_2 = open_[i] < close[i-1] and open_[i] > open_[i-1]
        closes_above_gap = close[i] > high[i-2]

        if (gap_up and is_white_1 and is_white_2 and is_black_3 and
            bodies_similar and opens_in_body_2 and closes_above_gap):
            output[i] = 100  # Bullish continuation
            continue

        # Downside Tasuki Gap
        gap_down = high[i-1] < low[i-2]
        is_black_1 = body_1 < 0
        is_black_2 = body_2 < 0
        is_white_3 = body_3 > 0
        opens_in_body_2_down = open_[i] > close[i-1] and open_[i] < open_[i-1]
        closes_below_gap = close[i] < low[i-2]

        if (gap_down and is_black_1 and is_black_2 and is_white_3 and
            bodies_similar and opens_in_body_2_down and closes_below_gap):
            output[i] = -100  # Bearish continuation
        else:
            output[i] = 0


def _cdltasukigap_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> np.ndarray:
    """CuPy-based Tasuki Gap"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        tolerance = avg_body[i] * 0.2
        bodies_similar = cp.abs(abs_body[i-1] - abs_body[i]) <= tolerance

        # Upside
        gap_up = low_gpu[i-1] > high_gpu[i-2]
        is_white_1 = body[i-2] > 0
        is_white_2 = body[i-1] > 0
        is_black_3 = body[i] < 0
        opens_in_body_2 = open_gpu[i] < close_gpu[i-1] and open_gpu[i] > open_gpu[i-1]
        closes_above_gap = close_gpu[i] > high_gpu[i-2]

        if (gap_up and is_white_1 and is_white_2 and is_black_3 and
            bodies_similar and opens_in_body_2 and closes_above_gap):
            output[i] = 100
            continue

        # Downside
        gap_down = high_gpu[i-1] < low_gpu[i-2]
        is_black_1 = body[i-2] < 0
        is_black_2 = body[i-1] < 0
        is_white_3 = body[i] > 0
        opens_in_body_2_down = open_gpu[i] > close_gpu[i-1] and open_gpu[i] < open_gpu[i-1]
        closes_below_gap = close_gpu[i] < low_gpu[i-2]

        if (gap_down and is_black_1 and is_black_2 and is_white_3 and
            bodies_similar and opens_in_body_2_down and closes_below_gap):
            output[i] = -100

    return cp.asnumpy(output)


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

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdltasukigap_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdltasukigap_numba(open_, high, low, close, output)
        return output


# ==================== CDLTHRUSTING ====================

@jit(nopython=True, cache=True)
def _cdlthrusting_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, output: np.ndarray) -> None:
    """Thrusting: Black candle, white opens below low, closes below midpoint"""
    n = len(open_)
    if n < 2:
        return

    for i in range(1, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        # Pattern: long black, white opens below prior low, closes under midpoint
        body_1 = close[i-1] - open_[i-1]
        body_2 = close[i] - open_[i]

        is_black_1 = body_1 < 0
        is_long_1 = abs(body_1) > avg_body * 1.0
        is_white_2 = body_2 > 0
        opens_below = open_[i] < low[i-1]

        # Closes into body but below midpoint
        midpoint = close[i-1] + abs(body_1) * 0.5
        closes_in_body = close[i] > close[i-1]
        closes_below_mid = close[i] <= midpoint

        if is_black_1 and is_long_1 and is_white_2 and opens_below and closes_in_body and closes_below_mid:
            output[i] = -100  # Bearish
        else:
            output[i] = 0


def _cdlthrusting_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> np.ndarray:
    """CuPy Thrusting"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 2:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(1, n):
        is_black_1 = body[i-1] < 0
        is_long_1 = abs_body[i-1] > avg_body[i-1] * 1.0
        is_white_2 = body[i] > 0
        opens_below = open_gpu[i] < low_gpu[i-1]
        midpoint = close_gpu[i-1] + abs_body[i-1] * 0.5
        closes_in_body = close_gpu[i] > close_gpu[i-1]
        closes_below_mid = close_gpu[i] <= midpoint

        if is_black_1 and is_long_1 and is_white_2 and opens_below and closes_in_body and closes_below_mid:
            output[i] = -100

    return cp.asnumpy(output)


def CDLTHRUSTING(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> np.ndarray:
    """Thrusting Pattern: Bearish, white closes below midpoint of prior black"""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlthrusting_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlthrusting_numba(open_, high, low, close, output)
        return output


# ==================== CDLTRISTAR ====================

@jit(nopython=True, cache=True)
def _cdltristar_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, output: np.ndarray) -> None:
    """Tristar: Three consecutive dojis with second gapping"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        body_1 = abs(close[i-2] - open_[i-2])
        body_2 = abs(close[i-1] - open_[i-1])
        body_3 = abs(close[i] - open_[i])

        # All three must be dojis
        is_doji_1 = body_1 <= avg_body * 0.1
        is_doji_2 = body_2 <= avg_body * 0.1
        is_doji_3 = body_3 <= avg_body * 0.1

        if not (is_doji_1 and is_doji_2 and is_doji_3):
            output[i] = 0
            continue

        # Bearish: 2nd gaps up, 3rd not higher
        gap_up = low[i-1] > high[i-2]
        third_not_higher = high[i] <= high[i-1]
        if gap_up and third_not_higher:
            output[i] = -100
            continue

        # Bullish: 2nd gaps down, 3rd not lower
        gap_down = high[i-1] < low[i-2]
        third_not_lower = low[i] >= low[i-1]
        if gap_down and third_not_lower:
            output[i] = 100
        else:
            output[i] = 0


def _cdltristar_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> np.ndarray:
    """CuPy Tristar"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = cp.abs(close_gpu - open_gpu)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        is_doji_1 = body[i-2] <= avg_body[i-2] * 0.1
        is_doji_2 = body[i-1] <= avg_body[i-1] * 0.1
        is_doji_3 = body[i] <= avg_body[i] * 0.1

        if not (is_doji_1 and is_doji_2 and is_doji_3):
            continue

        gap_up = low_gpu[i-1] > high_gpu[i-2]
        third_not_higher = high_gpu[i] <= high_gpu[i-1]
        if gap_up and third_not_higher:
            output[i] = -100
            continue

        gap_down = high_gpu[i-1] < low_gpu[i-2]
        third_not_lower = low_gpu[i] >= low_gpu[i-1]
        if gap_down and third_not_lower:
            output[i] = 100

    return cp.asnumpy(output)


def CDLTRISTAR(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
               close: np.ndarray) -> np.ndarray:
    """Tristar Pattern: Three dojis with second gapping. Returns ±100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdltristar_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdltristar_numba(open_, high, low, close, output)
        return output


# ==================== CDLUNIQUE3RIVER ====================

@jit(nopython=True, cache=True)
def _cdlunique3river_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                           close: np.ndarray, output: np.ndarray) -> None:
    """Unique 3 River: Long black, black harami, small white"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        # 1st: long black
        is_black_1 = body_1 < 0
        is_long_1 = abs(body_1) > avg_body * 1.0

        # 2nd: black harami (close > 1st close, open <= 1st open, low < 1st low)
        is_black_2 = body_2 < 0
        close_higher = close[i-1] > close[i-2]
        open_within = open_[i-1] <= open_[i-2]
        low_below = low[i-1] < low[i-2]

        # 3rd: small white (open > 2nd low)
        is_white_3 = body_3 > 0
        is_small_3 = abs(body_3) < avg_body * 1.0
        opens_above = open_[i] > low[i-1]

        if (is_black_1 and is_long_1 and is_black_2 and close_higher and
            open_within and low_below and is_white_3 and is_small_3 and opens_above):
            output[i] = 100  # Bullish
        else:
            output[i] = 0


def _cdlunique3river_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray) -> np.ndarray:
    """CuPy Unique 3 River"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        is_black_1 = body[i-2] < 0
        is_long_1 = abs_body[i-2] > avg_body[i-2] * 1.0
        is_black_2 = body[i-1] < 0
        close_higher = close_gpu[i-1] > close_gpu[i-2]
        open_within = open_gpu[i-1] <= open_gpu[i-2]
        low_below = low_gpu[i-1] < low_gpu[i-2]
        is_white_3 = body[i] > 0
        is_small_3 = abs_body[i] < avg_body[i] * 1.0
        opens_above = open_gpu[i] > low_gpu[i-1]

        if (is_black_1 and is_long_1 and is_black_2 and close_higher and
            open_within and low_below and is_white_3 and is_small_3 and opens_above):
            output[i] = 100

    return cp.asnumpy(output)


def CDLUNIQUE3RIVER(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray) -> np.ndarray:
    """Unique 3 River: Bullish reversal, 3 candles. Returns 100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlunique3river_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlunique3river_numba(open_, high, low, close, output)
        return output


# ==================== CDLUPSIDEGAP2CROWS ====================

@jit(nopython=True, cache=True)
def _cdlupsidegap2crows_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray, output: np.ndarray) -> None:
    """Upside Gap Two Crows: Long white, black gaps up, black engulfs"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        # 1st: long white
        is_white_1 = body_1 > 0
        is_long_1 = abs(body_1) > avg_body * 1.0

        # 2nd: short black, gaps up
        is_black_2 = body_2 < 0
        is_short_2 = abs(body_2) <= avg_body * 1.0
        gap_up = low[i-1] > high[i-2]

        # 3rd: black, opens > 2nd open, closes < 2nd close, closes > 1st close
        is_black_3 = body_3 < 0
        opens_higher = open_[i] > open_[i-1]
        closes_lower = close[i] < close[i-1]
        closes_above_1st = close[i] > close[i-2]

        if (is_white_1 and is_long_1 and is_black_2 and is_short_2 and gap_up and
            is_black_3 and opens_higher and closes_lower and closes_above_1st):
            output[i] = -100  # Bearish
        else:
            output[i] = 0


def _cdlupsidegap2crows_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray) -> np.ndarray:
    """CuPy Upside Gap Two Crows"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu
    abs_body = cp.abs(body)
    kernel = cp.ones(10) / 10.0
    avg_body = cp.convolve(abs_body, kernel, mode='same')
    avg_body = cp.where(avg_body > 0, avg_body, 1.0)

    for i in range(2, n):
        is_white_1 = body[i-2] > 0
        is_long_1 = abs_body[i-2] > avg_body[i-2] * 1.0
        is_black_2 = body[i-1] < 0
        is_short_2 = abs_body[i-1] <= avg_body[i-1] * 1.0
        gap_up = low_gpu[i-1] > high_gpu[i-2]
        is_black_3 = body[i] < 0
        opens_higher = open_gpu[i] > open_gpu[i-1]
        closes_lower = close_gpu[i] < close_gpu[i-1]
        closes_above_1st = close_gpu[i] > close_gpu[i-2]

        if (is_white_1 and is_long_1 and is_black_2 and is_short_2 and gap_up and
            is_black_3 and opens_higher and closes_lower and closes_above_1st):
            output[i] = -100

    return cp.asnumpy(output)


def CDLUPSIDEGAP2CROWS(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> np.ndarray:
    """Upside Gap Two Crows: Bearish reversal. Returns -100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlupsidegap2crows_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlupsidegap2crows_numba(open_, high, low, close, output)
        return output


# ==================== CDLXSIDEGAP3METHODS ====================

@jit(nopython=True, cache=True)
def _cdlxsidegap3methods_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, output: np.ndarray) -> None:
    """Upside/Downside Gap 3 Methods: Gap followed by opposite that closes gap"""
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        # Upside gap 3 methods
        is_white_1 = body_1 > 0
        is_white_2 = body_2 > 0
        is_black_3 = body_3 < 0
        gap_up = low[i-1] > high[i-2]
        opens_in_body_2 = open_[i] < close[i-1] and open_[i] > open_[i-1]
        closes_in_body_1 = close[i] < close[i-2] and close[i] > open_[i-2]

        if is_white_1 and is_white_2 and gap_up and is_black_3 and opens_in_body_2 and closes_in_body_1:
            output[i] = 100  # Bullish continuation
            continue

        # Downside gap 3 methods
        is_black_1 = body_1 < 0
        is_black_2 = body_2 < 0
        is_white_3 = body_3 > 0
        gap_down = high[i-1] < low[i-2]
        opens_in_body_2_down = open_[i] > close[i-1] and open_[i] < open_[i-1]
        closes_in_body_1_down = close[i] > close[i-2] and close[i] < open_[i-2]

        if is_black_1 and is_black_2 and gap_down and is_white_3 and opens_in_body_2_down and closes_in_body_1_down:
            output[i] = -100  # Bearish continuation
        else:
            output[i] = 0


def _cdlxsidegap3methods_cupy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray) -> np.ndarray:
    """CuPy Upside/Downside Gap 3 Methods"""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("CuPy required for GPU backend")

    open_gpu, high_gpu, low_gpu, close_gpu = cp.asarray(open_), cp.asarray(high), cp.asarray(low), cp.asarray(close)
    n = len(open_)
    output = cp.zeros(n, dtype=cp.int32)
    if n < 3:
        return cp.asnumpy(output)

    body = close_gpu - open_gpu

    for i in range(2, n):
        # Upside
        is_white_1 = body[i-2] > 0
        is_white_2 = body[i-1] > 0
        is_black_3 = body[i] < 0
        gap_up = low_gpu[i-1] > high_gpu[i-2]
        opens_in_body_2 = open_gpu[i] < close_gpu[i-1] and open_gpu[i] > open_gpu[i-1]
        closes_in_body_1 = close_gpu[i] < close_gpu[i-2] and close_gpu[i] > open_gpu[i-2]

        if is_white_1 and is_white_2 and gap_up and is_black_3 and opens_in_body_2 and closes_in_body_1:
            output[i] = 100
            continue

        # Downside
        is_black_1 = body[i-2] < 0
        is_black_2 = body[i-1] < 0
        is_white_3 = body[i] > 0
        gap_down = high_gpu[i-1] < low_gpu[i-2]
        opens_in_body_2_down = open_gpu[i] > close_gpu[i-1] and open_gpu[i] < open_gpu[i-1]
        closes_in_body_1_down = close_gpu[i] > close_gpu[i-2] and close_gpu[i] < open_gpu[i-2]

        if is_black_1 and is_black_2 and gap_down and is_white_3 and opens_in_body_2_down and closes_in_body_1_down:
            output[i] = -100

    return cp.asnumpy(output)


def CDLXSIDEGAP3METHODS(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray) -> np.ndarray:
    """Upside/Downside Gap 3 Methods: Continuation. Returns ±100 or 0."""
    open_, high, low, close = map(np.asarray, [open_, high, low, close])
    n = len(open_)
    if not (len(high) == len(low) == len(close) == n):
        raise ValueError("All input arrays must have the same length")
    if n == 0:
        return np.array([], dtype=np.int32)

    from .backend import get_backend
    if get_backend() == "gpu":
        return _cdlxsidegap3methods_cupy(open_, high, low, close)
    else:
        output = np.zeros(n, dtype=np.int32)
        _cdlxsidegap3methods_numba(open_, high, low, close, output)
        return output


# Placeholder functions for other patterns referenced in __init__.py
# These would be implemented similarly to CDLMARUBOZU

def CDL2CROWS(*args, **kwargs):
    raise NotImplementedError("CDL2CROWS not yet implemented")

def CDL3BLACKCROWS(*args, **kwargs):
    raise NotImplementedError("CDL3BLACKCROWS not yet implemented")

def CDL3INSIDE(*args, **kwargs):
    raise NotImplementedError("CDL3INSIDE not yet implemented")

def CDL3OUTSIDE(*args, **kwargs):
    raise NotImplementedError("CDL3OUTSIDE not yet implemented")

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
