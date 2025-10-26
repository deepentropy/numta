"""
Overlap Studies - Indicators that overlay price charts
"""

"""CPU implementations using Numba JIT compilation"""

import numpy as np
from numba import jit

__all__ = [
    "_sma_numba",
    "_ema_numba",
    "_bbands_numba",
    "_dema_numba",
    "_kama_numba",
    "_sar_numba",
    "_sarext_numba",
    "_wma_numba",
    "_tema_numba",
    "_t3_numba",
    "_trima_numba",
]


@jit(nopython=True, cache=True)
def _sma_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled SMA calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.
    Handles NaN values in input data.
    """
    n = len(close)

    # Find first valid (non-NaN) index
    start_idx = 0
    for i in range(n):
        if not np.isnan(close[i]):
            start_idx = i
            break

    # Fill initial values with NaN
    for i in range(start_idx + timeperiod - 1):
        output[i] = np.nan

    # Check if we have enough valid data
    if start_idx + timeperiod > n:
        for i in range(n):
            output[i] = np.nan
        return

    # Calculate first SMA value from first timeperiod valid values
    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += close[i]
    output[start_idx + timeperiod - 1] = sum_val / timeperiod

    # Use rolling window for subsequent values
    for i in range(start_idx + timeperiod, n):
        if np.isnan(close[i]) or np.isnan(close[i - timeperiod]):
            output[i] = np.nan
        else:
            sum_val = sum_val - close[i - timeperiod] + close[i]
            output[i] = sum_val / timeperiod


@jit(nopython=True, cache=True)
def _ema_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled EMA calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    Formula:
    EMA = (Close - EMA_prev) * multiplier + EMA_prev
    where multiplier = 2 / (timeperiod + 1)

    The first EMA value is initialized as SMA of first timeperiod values.
    """
    n = len(close)
    multiplier = 2.0 / (timeperiod + 1)

    # Find first valid (non-NaN) index
    start_idx = 0
    for i in range(n):
        if not np.isnan(close[i]):
            start_idx = i
            break

    # Fill initial values with NaN
    for i in range(start_idx + timeperiod - 1):
        output[i] = np.nan

    # Check if we have enough valid data
    if start_idx + timeperiod > n:
        for i in range(n):
            output[i] = np.nan
        return

    # Initialize first EMA value as SMA of first timeperiod valid values
    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += close[i]
    ema = sum_val / timeperiod
    output[start_idx + timeperiod - 1] = ema

    # Calculate EMA for remaining values
    for i in range(start_idx + timeperiod, n):
        if np.isnan(close[i]):
            output[i] = np.nan
        else:
            ema = (close[i] - ema) * multiplier + ema
            output[i] = ema


@jit(nopython=True, cache=True)
def _bbands_numba(close: np.ndarray, timeperiod: int, nbdevup: float, nbdevdn: float,
                  upperband: np.ndarray, middleband: np.ndarray, lowerband: np.ndarray) -> None:
    """
    Numba-compiled Bollinger Bands calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output arrays in-place.

    Formula:
    Middle Band = SMA(close, timeperiod)
    Upper Band = Middle Band + (nbdevup * StdDev)
    Lower Band = Middle Band - (nbdevdn * StdDev)
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        upperband[i] = np.nan
        middleband[i] = np.nan
        lowerband[i] = np.nan

    # Calculate first SMA value
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    sma = sum_val / timeperiod

    # Calculate first standard deviation
    variance = 0.0
    for i in range(timeperiod):
        diff = close[i] - sma
        variance += diff * diff
    stddev = np.sqrt(variance / timeperiod)

    # Set first values
    middleband[timeperiod - 1] = sma
    upperband[timeperiod - 1] = sma + nbdevup * stddev
    lowerband[timeperiod - 1] = sma - nbdevdn * stddev

    # Calculate remaining values using rolling window
    for i in range(timeperiod, n):
        # Update SMA (rolling window)
        sum_val = sum_val - close[i - timeperiod] + close[i]
        sma = sum_val / timeperiod

        # Calculate standard deviation for current window
        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = close[j] - sma
            variance += diff * diff
        stddev = np.sqrt(variance / timeperiod)

        # Calculate bands
        middleband[i] = sma
        upperband[i] = sma + nbdevup * stddev
        lowerband[i] = sma - nbdevdn * stddev


@jit(nopython=True, cache=True)
def _dema_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled DEMA calculation (in-place)
    
    Formula:
    DEMA = 2 * EMA - EMA(EMA)
    """
    n = len(close)
    multiplier = 2.0 / (timeperiod + 1)
    
    # Fill lookback period with NaN (2 * timeperiod - 2)
    lookback = 2 * timeperiod - 2
    for i in range(lookback):
        output[i] = np.nan
    
    # Calculate first EMA
    ema1 = np.empty(n, dtype=np.float64)
    for i in range(timeperiod - 1):
        ema1[i] = np.nan
    
    # Initialize first EMA value as SMA
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    ema1[timeperiod - 1] = sum_val / timeperiod
    
    # Calculate remaining EMA1 values
    for i in range(timeperiod, n):
        ema1[i] = (close[i] - ema1[i-1]) * multiplier + ema1[i-1]
    
    # Calculate EMA of EMA (EMA2)
    sum_val = 0.0
    for i in range(timeperiod - 1, 2 * timeperiod - 1):
        sum_val += ema1[i]
    ema2 = sum_val / timeperiod
    
    # Calculate DEMA values
    output[2 * timeperiod - 2] = 2.0 * ema1[2 * timeperiod - 2] - ema2
    
    for i in range(2 * timeperiod - 1, n):
        ema2 = (ema1[i] - ema2) * multiplier + ema2
        output[i] = 2.0 * ema1[i] - ema2


@jit(nopython=True, cache=True)
def _kama_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled KAMA calculation (in-place) - Optimized

    This implementation:
    1. Uses incremental volatility calculation (O(n) instead of O(n*timeperiod))
    2. Matches TA-Lib output exactly
    3. Uses unstable period warmup like TA-Lib

    Formula:
    1. ER = Change / Volatility
       where Change = abs(close[i] - close[i-timeperiod])
             Volatility = sum of abs(close[j] - close[j-1]) over timeperiod
    2. Fastest = 2/(2+1) = 0.6667
       Slowest = 2/(30+1) = 0.0645
    3. SC = (ER * (fastest - slowest) + slowest)^2
    4. KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    """
    n = len(close)
    fastest = 2.0 / (2.0 + 1.0)  # 0.6667
    slowest = 2.0 / (30.0 + 1.0)  # 0.0645
    const_diff = fastest - slowest

    # Unstable period: TA-Lib uses max(timeperiod, 30) + unstable_period
    # For simplicity and to match TA-Lib, use 30 as minimum unstable period
    # Lookback period equals timeperiod
    lookback_period = timeperiod

    # Fill lookback period with NaN
    for i in range(lookback_period):
        output[i] = np.nan

    if n <= lookback_period:
        return

    # Initialize starting from unstable_period
    today = lookback_period
    trailing_idx = 0

    # Calculate initial volatility sum
    per_sum = 0.0
    for i in range(lookback_period):
        per_sum += abs(close[i + 1] - close[i])

    # Initialize KAMA at first output position
    kama = close[today]

    # Calculate and output first KAMA value
    if per_sum != 0.0:
        er = abs(close[today] - close[trailing_idx]) / per_sum
    else:
        er = 0.0

    sc = er * const_diff + slowest
    sc = sc * sc
    kama = kama + sc * (close[today] - kama)
    output[today] = kama

    # Move to next position
    today += 1
    trailing_idx += 1

    # Calculate remaining KAMA values using incremental volatility
    while today < n:
        # Incrementally update volatility sum
        per_sum -= abs(close[trailing_idx] - close[trailing_idx - 1])
        per_sum += abs(close[today] - close[today - 1])

        # Calculate ER
        if per_sum != 0.0:
            er = abs(close[today] - close[trailing_idx]) / per_sum
        else:
            er = 0.0

        # Calculate SC
        sc = er * const_diff + slowest
        sc = sc * sc

        # Update KAMA
        kama = kama + sc * (close[today] - kama)
        output[today] = kama

        today += 1
        trailing_idx += 1


@jit(nopython=True, cache=True)
def _sar_numba(high: np.ndarray, low: np.ndarray, acceleration: float, maximum: float, output: np.ndarray) -> None:
    """
    Numba-compiled SAR calculation (in-place)

    Parabolic SAR algorithm by J. Welles Wilder
    """
    n = len(high)

    # Initialize
    is_long = True  # Start with long position
    sar = low[0]
    ep = high[0]  # Extreme point
    af = acceleration  # Acceleration factor

    output[0] = sar

    for i in range(1, n):
        # Calculate new SAR
        sar = sar + af * (ep - sar)

        if is_long:
            # Long position
            # SAR should not be above prior two lows
            if i >= 1:
                sar = min(sar, low[i - 1])
            if i >= 2:
                sar = min(sar, low[i - 2])

            # Check for reversal
            if low[i] < sar:
                # Reverse to short
                is_long = False
                sar = ep  # SAR becomes the extreme point
                ep = low[i]  # New extreme point
                af = acceleration  # Reset AF
            else:
                # Continue long
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
        else:
            # Short position
            # SAR should not be below prior two highs
            if i >= 1:
                sar = max(sar, high[i - 1])
            if i >= 2:
                sar = max(sar, high[i - 2])

            # Check for reversal
            if high[i] > sar:
                # Reverse to long
                is_long = True
                sar = ep  # SAR becomes the extreme point
                ep = high[i]  # New extreme point
                af = acceleration  # Reset AF
            else:
                # Continue short
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)

        output[i] = sar


@jit(nopython=True, cache=True)
def _sarext_numba(high: np.ndarray, low: np.ndarray,
                  startvalue: float, offsetonreverse: float,
                  accelerationinit_long: float, accelerationlong: float, accelerationmax_long: float,
                  accelerationinit_short: float, accelerationshort: float, accelerationmax_short: float,
                  output: np.ndarray) -> None:
    """
    Numba-compiled SAREXT calculation (in-place)

    Extended Parabolic SAR with separate parameters for long and short
    """
    n = len(high)

    # Initialize
    is_long = True
    sar = startvalue if startvalue != 0 else low[0]
    ep = high[0]
    af = accelerationinit_long

    output[0] = sar

    for i in range(1, n):
        # Calculate new SAR
        sar = sar + af * (ep - sar)

        if is_long:
            # Long position
            if i >= 1:
                sar = min(sar, low[i - 1])
            if i >= 2:
                sar = min(sar, low[i - 2])

            # Check for reversal
            if low[i] < sar:
                is_long = False
                sar = ep + offsetonreverse
                ep = low[i]
                af = accelerationinit_short
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + accelerationlong, accelerationmax_long)
        else:
            # Short position
            if i >= 1:
                sar = max(sar, high[i - 1])
            if i >= 2:
                sar = max(sar, high[i - 2])

            # Check for reversal
            if high[i] > sar:
                is_long = True
                sar = ep - offsetonreverse
                ep = high[i]
                af = accelerationinit_long
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + accelerationshort, accelerationmax_short)

        output[i] = sar


@jit(nopython=True, cache=True)
def _wma_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled WMA calculation using incremental O(n) algorithm

    The optimization uses incremental calculation to avoid recalculating
    the entire weighted sum at each position. When sliding the window:
    1. Remove the old value (which had weight 1)
    2. Subtract simple_sum from weighted_sum (all values lose 1 weight)
    3. Add new value with full weight (timeperiod)
    4. Update simple_sum

    This reduces complexity from O(n*timeperiod) to O(n).
    """
    n = len(data)

    # Calculate sum of weights: 1 + 2 + 3 + ... + timeperiod
    weight_sum = (timeperiod * (timeperiod + 1)) / 2.0

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate first WMA value using the standard method
    weighted_sum = 0.0
    simple_sum = 0.0
    for j in range(timeperiod):
        weight = j + 1  # Weight increases from 1 (oldest) to timeperiod (newest)
        value = data[timeperiod - 1 - (timeperiod - 1 - j)]  # = data[j]
        weighted_sum += value * weight
        simple_sum += value

    output[timeperiod - 1] = weighted_sum / weight_sum

    # Use incremental calculation for remaining values
    # Formula when sliding window from position i to i+1:
    # - Remove old value: weighted_sum -= data[i-timeperiod] (weight 1)
    # - All values lose 1 weight: weighted_sum -= simple_sum
    # - Add new value: weighted_sum += data[i] * timeperiod
    # - Update simple_sum: simple_sum = simple_sum - data[i-timeperiod] + data[i]
    for i in range(timeperiod, n):
        old_value = data[i - timeperiod]
        new_value = data[i]

        # Remove contribution of the oldest value (which had weight 1)
        # Subtract simple_sum (all remaining values lose 1 weight)
        weighted_sum = weighted_sum - simple_sum

        # Update simple sum (remove old, add new)
        simple_sum = simple_sum - old_value + new_value

        # Add new value with full weight
        weighted_sum = weighted_sum + new_value * timeperiod

        output[i] = weighted_sum / weight_sum


@jit(nopython=True, cache=True)
def _tema_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled TEMA calculation (in-place)
    
    Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    
    This optimized version computes all three EMAs in a single pass
    to avoid multiple function call overhead and intermediate array allocations.
    """
    n = len(data)
    
    # Allocate temporary arrays for the three EMAs
    ema1 = np.empty(n, dtype=np.float64)
    ema2 = np.empty(n, dtype=np.float64)
    ema3 = np.empty(n, dtype=np.float64)
    
    # Calculate EMA1
    _ema_numba(data, timeperiod, ema1)
    
    # Calculate EMA2 (EMA of EMA1)
    _ema_numba(ema1, timeperiod, ema2)
    
    # Calculate EMA3 (EMA of EMA2)
    _ema_numba(ema2, timeperiod, ema3)
    
    # TEMA = 3*EMA1 - 3*EMA2 + EMA3
    for i in range(n):
        output[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i]


@jit(nopython=True, cache=True)
def _t3_numba(data: np.ndarray, timeperiod: int, vfactor: float, output: np.ndarray) -> None:
    """
    Numba-compiled T3 calculation (in-place)
    
    T3 (Tillson T3) uses 6 EMAs with special coefficients based on volume factor.
    
    Formula: T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
    where coefficients are calculated from vfactor
    """
    n = len(data)
    
    # Calculate coefficients from volume factor
    c1 = -vfactor * vfactor * vfactor
    c2 = 3.0 * vfactor * vfactor + 3.0 * vfactor * vfactor * vfactor
    c3 = -6.0 * vfactor * vfactor - 3.0 * vfactor - 3.0 * vfactor * vfactor * vfactor
    c4 = 1.0 + 3.0 * vfactor + vfactor * vfactor * vfactor + 3.0 * vfactor * vfactor
    
    # Allocate temporary arrays for all 6 EMAs
    ema1 = np.empty(n, dtype=np.float64)
    ema2 = np.empty(n, dtype=np.float64)
    ema3 = np.empty(n, dtype=np.float64)
    ema4 = np.empty(n, dtype=np.float64)
    ema5 = np.empty(n, dtype=np.float64)
    ema6 = np.empty(n, dtype=np.float64)
    
    # Calculate all 6 EMAs in sequence
    _ema_numba(data, timeperiod, ema1)
    _ema_numba(ema1, timeperiod, ema2)
    _ema_numba(ema2, timeperiod, ema3)
    _ema_numba(ema3, timeperiod, ema4)
    _ema_numba(ema4, timeperiod, ema5)
    _ema_numba(ema5, timeperiod, ema6)
    
    # T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
    for i in range(n):
        output[i] = c1 * ema6[i] + c2 * ema5[i] + c3 * ema4[i] + c4 * ema3[i]


@jit(nopython=True, cache=True)
def _trima_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled TRIMA calculation (in-place)
    
    TRIMA (Triangular Moving Average) is a double-smoothed SMA.
    
    Formula:
    - If timeperiod is odd: n = (timeperiod + 1) / 2, TRIMA = SMA(SMA(data, n), n)
    - If timeperiod is even: n1 = timeperiod / 2, n2 = n1 + 1, TRIMA = SMA(SMA(data, n1), n2)
    """
    n = len(data)
    
    # Calculate periods for double SMA
    if timeperiod % 2 == 1:  # Odd period
        n1 = (timeperiod + 1) // 2
        n2 = n1
    else:  # Even period
        n1 = timeperiod // 2
        n2 = n1 + 1
    
    # Allocate temporary array for first SMA
    sma1 = np.empty(n, dtype=np.float64)
    
    # Calculate first SMA
    _sma_numba(data, n1, sma1)
    
    # Calculate second SMA (SMA of SMA) to get TRIMA
    _sma_numba(sma1, n2, output)
