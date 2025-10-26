"""
Pattern Recognition - Candlestick pattern recognition functions

This module implements candlestick pattern recognition compatible with TA-Lib.
Patterns return integer values:
    100: Bullish pattern
   -100: Bearish pattern
      0: No pattern
"""

"""GPU implementations using CuPy"""

import numpy as np

# Try to import cupy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


__all__ = [
    "_cdlmarubozu_cupy",
    "_cdlmatchinglow_cupy",
    "_cdlmathold_cupy",
    "_cdlmorningdojistar_cupy",
    "_cdlmorningstar_cupy",
    "_cdlonneck_cupy",
    "_cdlpiercing_cupy",
    "_cdlrickshawman_cupy",
    "_cdlrisefall3methods_cupy",
    "_cdlseparatinglines_cupy",
    "_cdlshootingstar_cupy",
    "_cdlshortline_cupy",
    "_cdlspinningtop_cupy",
    "_cdlstalledpattern_cupy",
    "_cdlsticksandwich_cupy",
    "_cdltakuri_cupy",
    "_cdltasukigap_cupy",
    "_cdlthrusting_cupy",
    "_cdltristar_cupy",
    "_cdlunique3river_cupy",
    "_cdlupsidegap2crows_cupy",
    "_cdlxsidegap3methods_cupy",
]


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


