"""
Momentum Indicators - Indicators that measure the rate of price change
"""

"""CPU implementations using Numba JIT compilation"""

import numpy as np
from numba import jit


__all__ = [
    "_adx_numba",
    "_adxr_numba",
    "_apo_numba_ema",
    "_apo_numba_sma",
    "_aroon_numba",
    "_aroonosc_numba",
    "_atr_numba",
    "_bop_numba",
    "_cci_numba",
    "_cmo_numba",
    "_dx_numba",
    "_ema_for_apo",
    "_macd_numba",
    "_macdext_numba",
    "_macdfix_numba",
    "_mfi_numba",
    "_minus_di_numba",
    "_minus_dm_numba",
    "_mom_numba",
    "_plus_di_numba",
    "_plus_dm_numba",
    "_ppo_numba",
    "_roc_numba",
    "_rocp_numba",
    "_rocr100_numba",
    "_rocr_numba",
    "_rsi_numba",
    "_sma_for_apo",
    "_stoch_fastk_numba",
    "_stochrsi_numba",
    "_trix_numba",
    "_ultosc_numba",
    "_willr_numba",
]


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




@jit(nopython=True, cache=True)
def _aroonosc_numba(high: np.ndarray, low: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled AROONOSC calculation (in-place)
    
    AROONOSC = Aroon Up - Aroon Down
    
    This optimized version calculates both Aroon Up and Aroon Down in a single pass
    and computes the oscillator directly.
    """
    n = len(high)
    
    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan
    
    # Calculate Aroon Oscillator for each bar
    for i in range(timeperiod, n):
        # Find periods since highest high and lowest low in the lookback window
        periods_since_high = 0
        periods_since_low = 0
        highest = high[i]
        lowest = low[i]
        
        for j in range(i - timeperiod + 1, i + 1):
            if high[j] >= highest:
                highest = high[j]
                periods_since_high = i - j
            if low[j] <= lowest:
                lowest = low[j]
                periods_since_low = i - j
        
        # Calculate Aroon Up and Aroon Down
        aroon_up = ((timeperiod - periods_since_high) / timeperiod) * 100.0
        aroon_down = ((timeperiod - periods_since_low) / timeperiod) * 100.0
        
        # AROONOSC = Aroon Up - Aroon Down
        output[i] = aroon_up - aroon_down


@jit(nopython=True, cache=True)
def _bop_numba(open_price: np.ndarray, high: np.ndarray, low: np.ndarray, 
               close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled BOP (Balance of Power) calculation (in-place)
    
    Formula: BOP = (Close - Open) / (High - Low)
    
    When High equals Low (no price range), BOP is set to 0.
    """
    n = len(open_price)
    
    for i in range(n):
        numerator = close[i] - open_price[i]
        denominator = high[i] - low[i]
        
        # Handle division by zero when High == Low
        if denominator == 0.0:
            output[i] = 0.0
        else:
            output[i] = numerator / denominator


@jit(nopython=True, cache=True)
def _ppo_numba(fast_ema: np.ndarray, slow_ema: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled PPO calculation (in-place)
    
    Formula: PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100
    
    This function calculates the percentage price oscillator from pre-computed EMAs.
    """
    n = len(fast_ema)
    
    for i in range(n):
        if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
            output[i] = np.nan
        elif slow_ema[i] == 0.0:
            output[i] = 0.0
        else:
            output[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100.0


@jit(nopython=True, cache=True)
def _macdext_numba(fast_ma: np.ndarray, slow_ma: np.ndarray, signal_ma: np.ndarray,
                   macd_out: np.ndarray, signal_out: np.ndarray, hist_out: np.ndarray) -> None:
    """
    Numba-compiled MACDEXT calculation (in-place)
    
    Calculates MACD line, signal line, and histogram from pre-computed MAs.
    
    Formula:
    - MACD Line = Fast MA - Slow MA
    - Signal Line = MA(MACD Line)
    - Histogram = MACD Line - Signal Line
    """
    n = len(fast_ma)
    
    # Calculate MACD line
    for i in range(n):
        macd_out[i] = fast_ma[i] - slow_ma[i]
        signal_out[i] = signal_ma[i]
        hist_out[i] = macd_out[i] - signal_ma[i]


@jit(nopython=True, cache=True)
def _macdfix_numba(close: np.ndarray, signalperiod: int,
                   macd_out: np.ndarray, signal_out: np.ndarray, hist_out: np.ndarray) -> None:
    """
    Numba-compiled MACDFIX calculation (in-place)
    
    Optimized version with hardcoded 12/26 periods for maximum performance.
    
    Formula:
    1. Fast EMA = EMA(close, 12)
    2. Slow EMA = EMA(close, 26)
    3. MACD Line = Fast EMA - Slow EMA
    4. Signal Line = EMA(MACD Line, signalperiod)
    5. Histogram = MACD Line - Signal Line
    """
    n = len(close)
    
    # Fixed periods for MACDFIX
    fastperiod = 12
    slowperiod = 26
    
    # Calculate multipliers
    fast_multiplier = 2.0 / (fastperiod + 1)
    slow_multiplier = 2.0 / (slowperiod + 1)
    signal_multiplier = 2.0 / (signalperiod + 1)
    
    # Temporary arrays for EMAs
    fast_ema = np.empty(n, dtype=np.float64)
    slow_ema = np.empty(n, dtype=np.float64)
    
    # Initialize fast EMA with NaN
    for i in range(fastperiod - 1):
        fast_ema[i] = np.nan
    
    # Calculate first fast EMA value as SMA
    fast_sum = 0.0
    for i in range(fastperiod):
        fast_sum += close[i]
    fast_val = fast_sum / fastperiod
    fast_ema[fastperiod - 1] = fast_val
    
    # Calculate remaining fast EMA values
    for i in range(fastperiod, n):
        fast_val = (close[i] - fast_val) * fast_multiplier + fast_val
        fast_ema[i] = fast_val
    
    # Initialize slow EMA with NaN
    for i in range(slowperiod - 1):
        slow_ema[i] = np.nan
    
    # Calculate first slow EMA value as SMA
    slow_sum = 0.0
    for i in range(slowperiod):
        slow_sum += close[i]
    slow_val = slow_sum / slowperiod
    slow_ema[slowperiod - 1] = slow_val
    
    # Calculate remaining slow EMA values
    for i in range(slowperiod, n):
        slow_val = (close[i] - slow_val) * slow_multiplier + slow_val
        slow_ema[i] = slow_val
    
    # Calculate MACD line
    for i in range(n):
        macd_out[i] = fast_ema[i] - slow_ema[i]
    
    # Calculate signal line (EMA of MACD)
    # Initialize with NaN
    for i in range(slowperiod + signalperiod - 2):
        signal_out[i] = np.nan
    
    # Calculate first signal value as SMA of first signalperiod MACD values
    signal_sum = 0.0
    start_idx = slowperiod - 1
    for i in range(start_idx, start_idx + signalperiod):
        signal_sum += macd_out[i]
    signal_val = signal_sum / signalperiod
    signal_out[start_idx + signalperiod - 1] = signal_val
    
    # Calculate remaining signal values as EMA
    for i in range(start_idx + signalperiod, n):
        signal_val = (macd_out[i] - signal_val) * signal_multiplier + signal_val
        signal_out[i] = signal_val
    
    # Calculate histogram
    for i in range(n):
        hist_out[i] = macd_out[i] - signal_out[i]


@jit(nopython=True, cache=True)
def _stochrsi_numba(rsi: np.ndarray, fastk_period: int, output: np.ndarray) -> None:
    """
    Numba-compiled StochRSI %K calculation (in-place)

    Applies Stochastic formula to RSI values.

    Formula: StochRSI %K = ((RSI - Lowest RSI) / (Highest RSI - Lowest RSI)) * 100
    """
    n = len(rsi)
    lookback = fastk_period - 1

    # Initialize with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Calculate StochRSI for each bar
    for i in range(lookback, n):
        # Find highest and lowest RSI over fastk_period
        highest_rsi = -np.inf
        lowest_rsi = np.inf
        valid_count = 0

        for j in range(i - fastk_period + 1, i + 1):
            if not np.isnan(rsi[j]):
                if rsi[j] > highest_rsi:
                    highest_rsi = rsi[j]
                if rsi[j] < lowest_rsi:
                    lowest_rsi = rsi[j]
                valid_count += 1

        if valid_count == 0 or highest_rsi == lowest_rsi:
            output[i] = 50.0
        else:
            output[i] = ((rsi[i] - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100.0


@jit(nopython=True, cache=True)
def _trix_numba(ema3: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled TRIX calculation (in-place)

    Calculates 1-period ROC of triple EMA.

    Formula: TRIX = ((EMA3[i] - EMA3[i-1]) / EMA3[i-1]) * 100
    """
    n = len(ema3)

    # First value is NaN (no previous value)
    output[0] = np.nan

    # Calculate 1-period ROC
    for i in range(1, n):
        if np.isnan(ema3[i]) or np.isnan(ema3[i - 1]) or ema3[i - 1] == 0.0:
            output[i] = np.nan
        else:
            output[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100.0


@jit(nopython=True, cache=True)
def _ultosc_numba(bp: np.ndarray, tr: np.ndarray,
                  timeperiod1: int, timeperiod2: int, timeperiod3: int,
                  output: np.ndarray) -> None:
    """
    Numba-compiled Ultimate Oscillator calculation (in-place)

    Calculates UO from pre-computed Buying Pressure and True Range arrays.

    Formula: UO = 100 * [(4*Avg1 + 2*Avg2 + Avg3) / 7]
    where Avg = Sum(BP over period) / Sum(TR over period)
    """
    n = len(bp)

    # Determine lookback period
    lookback = max(timeperiod1, timeperiod2, timeperiod3) - 1

    # Initialize with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Calculate Ultimate Oscillator for each bar
    for i in range(lookback, n):
        # Period 1 (shortest)
        if i >= timeperiod1 - 1:
            sum_bp1 = 0.0
            sum_tr1 = 0.0
            for j in range(i - timeperiod1 + 1, i + 1):
                sum_bp1 += bp[j]
                sum_tr1 += tr[j]
            avg1 = sum_bp1 / sum_tr1 if sum_tr1 != 0.0 else 0.0
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
            avg2 = sum_bp2 / sum_tr2 if sum_tr2 != 0.0 else 0.0
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
            avg3 = sum_bp3 / sum_tr3 if sum_tr3 != 0.0 else 0.0
        else:
            output[i] = np.nan
            continue

        # Ultimate Oscillator = 100 * [(4*Avg1 + 2*Avg2 + Avg3) / 7]
        output[i] = 100.0 * ((4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0)
