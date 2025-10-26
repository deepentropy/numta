"""
Pattern Recognition - Candlestick pattern recognition functions

This module implements candlestick pattern recognition compatible with TA-Lib.
Patterns return integer values:
    100: Bullish pattern
   -100: Bearish pattern
      0: No pattern
"""

"""CPU implementations using Numba JIT compilation"""

import numpy as np
from numba import jit


__all__ = [
    "_cdl2crows_numba",
    "_cdl3blackcrows_numba",
    "_cdl3inside_numba",
    "_cdl3outside_numba",
    "_cdlmarubozu_numba",
    "_cdlmatchinglow_numba",
    "_cdlmathold_numba",
    "_cdlmorningdojistar_numba",
    "_cdlmorningstar_numba",
    "_cdlonneck_numba",
    "_cdlpiercing_numba",
    "_cdlrickshawman_numba",
    "_cdlrisefall3methods_numba",
    "_cdlseparatinglines_numba",
    "_cdlshootingstar_numba",
    "_cdlshortline_numba",
    "_cdlspinningtop_numba",
    "_cdlstalledpattern_numba",
    "_cdlsticksandwich_numba",
    "_cdltakuri_numba",
    "_cdltasukigap_numba",
    "_cdlthrusting_numba",
    "_cdltristar_numba",
    "_cdlunique3river_numba",
    "_cdlupsidegap2crows_numba",
    "_cdlxsidegap3methods_numba",
]


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


@jit(nopython=True, cache=True)
def _cdl2crows_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, output: np.ndarray) -> None:
    """
    Two Crows: 3-candle bearish reversal pattern

    Pattern appears after an uptrend with:
    1. First candle: Long white candle (uptrend continuation)
    2. Second candle: Black candle that gaps up, opens above first close, closes below first close
    3. Third candle: Black candle that opens above second candle body, closes lower than second

    Returns -100 for bearish pattern, 0 otherwise.
    """
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        # Calculate average body size
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        # First candle: Long white candle
        body_1 = close[i-2] - open_[i-2]
        is_long_white_1 = body_1 > avg_body * 1.0 and body_1 > 0

        # Second candle: Black candle that gaps up
        body_2 = close[i-1] - open_[i-1]
        is_black_2 = body_2 < 0
        gaps_up = open_[i-1] > close[i-2]
        closes_in_body_1 = close[i-1] < close[i-2] and close[i-1] > open_[i-2]

        # Third candle: Black candle
        body_3 = close[i] - open_[i]
        is_black_3 = body_3 < 0
        opens_above_body_2 = open_[i] > close[i-1] and open_[i] < open_[i-1]
        closes_lower = close[i] < close[i-1]

        if (is_long_white_1 and is_black_2 and gaps_up and closes_in_body_1 and
            is_black_3 and opens_above_body_2 and closes_lower):
            output[i] = -100
        else:
            output[i] = 0


@jit(nopython=True, cache=True)
def _cdl3blackcrows_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, output: np.ndarray) -> None:
    """
    Three Black Crows: 3-candle bearish reversal pattern

    Pattern consists of three consecutive long black candles with:
    - Each candle opens within the previous candle's body
    - Each candle closes progressively lower
    - Small or no upper shadows

    Returns -100 for bearish pattern, 0 otherwise.
    """
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        # Calculate average body size
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        # All three candles must be black (bearish)
        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        are_all_black = body_1 < 0 and body_2 < 0 and body_3 < 0

        # All candles should be reasonably long
        are_all_long = (abs(body_1) > avg_body * 0.7 and
                       abs(body_2) > avg_body * 0.7 and
                       abs(body_3) > avg_body * 0.7)

        # Each opens within previous body and closes lower
        opens_in_body_2 = open_[i-1] < open_[i-2] and open_[i-1] > close[i-2]
        opens_in_body_3 = open_[i] < open_[i-1] and open_[i] > close[i-1]

        # Progressive lower closes
        lower_closes = close[i-1] < close[i-2] and close[i] < close[i-1]

        # Small upper shadows (characteristic of the pattern)
        upper_shadow_1 = high[i-2] - open_[i-2]
        upper_shadow_2 = high[i-1] - open_[i-1]
        upper_shadow_3 = high[i] - open_[i]
        small_shadows = (upper_shadow_1 < abs(body_1) * 0.3 and
                        upper_shadow_2 < abs(body_2) * 0.3 and
                        upper_shadow_3 < abs(body_3) * 0.3)

        if (are_all_black and are_all_long and opens_in_body_2 and opens_in_body_3 and
            lower_closes and small_shadows):
            output[i] = -100
        else:
            output[i] = 0


@jit(nopython=True, cache=True)
def _cdl3inside_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, output: np.ndarray) -> None:
    """
    Three Inside Up/Down: 3-candle reversal pattern

    Three Inside Up (bullish):
    1. First candle: Long black candle
    2. Second candle: White candle inside first (harami)
    3. Third candle: White candle that closes above first high

    Three Inside Down (bearish):
    1. First candle: Long white candle
    2. Second candle: Black candle inside first (harami)
    3. Third candle: Black candle that closes below first low

    Returns +100 for bullish, -100 for bearish, 0 otherwise.
    """
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        # Calculate average body size
        avg_body = 0.0
        count = 0
        for j in range(max(0, i-10), i):
            avg_body += abs(close[j] - open_[j])
            count += 1
        avg_body = avg_body / count if count > 0 else 1.0

        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        # Three Inside Up (bullish)
        is_long_black_1 = body_1 < 0 and abs(body_1) > avg_body * 0.8
        is_white_2 = body_2 > 0
        inside_body_1 = (open_[i-1] < open_[i-2] and open_[i-1] > close[i-2] and
                        close[i-1] > close[i-2] and close[i-1] < open_[i-2])
        is_white_3 = body_3 > 0
        closes_above_high_1 = close[i] > high[i-2]

        if (is_long_black_1 and is_white_2 and inside_body_1 and
            is_white_3 and closes_above_high_1):
            output[i] = 100
            continue

        # Three Inside Down (bearish)
        is_long_white_1 = body_1 > 0 and abs(body_1) > avg_body * 0.8
        is_black_2 = body_2 < 0
        inside_body_1_bear = (open_[i-1] > close[i-2] and open_[i-1] < open_[i-2] and
                             close[i-1] < close[i-2] and close[i-1] > open_[i-2])
        is_black_3 = body_3 < 0
        closes_below_low_1 = close[i] < low[i-2]

        if (is_long_white_1 and is_black_2 and inside_body_1_bear and
            is_black_3 and closes_below_low_1):
            output[i] = -100
        else:
            output[i] = 0


@jit(nopython=True, cache=True)
def _cdl3outside_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, output: np.ndarray) -> None:
    """
    Three Outside Up/Down: 3-candle reversal pattern

    Three Outside Up (bullish):
    1. First candle: Black candle
    2. Second candle: White candle that engulfs first (bullish engulfing)
    3. Third candle: White candle that closes higher than second

    Three Outside Down (bearish):
    1. First candle: White candle
    2. Second candle: Black candle that engulfs first (bearish engulfing)
    3. Third candle: Black candle that closes lower than second

    Returns +100 for bullish, -100 for bearish, 0 otherwise.
    """
    n = len(open_)
    if n < 3:
        return

    for i in range(2, n):
        body_1 = close[i-2] - open_[i-2]
        body_2 = close[i-1] - open_[i-1]
        body_3 = close[i] - open_[i]

        # Three Outside Up (bullish)
        is_black_1 = body_1 < 0
        is_white_2 = body_2 > 0
        # Engulfing: second opens below first close, closes above first open
        engulfs_bullish = (open_[i-1] < close[i-2] and close[i-1] > open_[i-2])
        is_white_3 = body_3 > 0
        closes_higher = close[i] > close[i-1]

        if (is_black_1 and is_white_2 and engulfs_bullish and
            is_white_3 and closes_higher):
            output[i] = 100
            continue

        # Three Outside Down (bearish)
        is_white_1 = body_1 > 0
        is_black_2 = body_2 < 0
        # Engulfing: second opens above first close, closes below first open
        engulfs_bearish = (open_[i-1] > close[i-2] and close[i-1] < open_[i-2])
        is_black_3 = body_3 < 0
        closes_lower = close[i] < close[i-1]

        if (is_white_1 and is_black_2 and engulfs_bearish and
            is_black_3 and closes_lower):
            output[i] = -100
        else:
            output[i] = 0


