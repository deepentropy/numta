"""
Overlap Studies - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread, mirroring the CPU kernel logic.
"""

import math
from numba import cuda


@cuda.jit
def _sma_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    if n < timeperiod:
        for i in range(n):
            output_2d[tid, i] = math.nan
        return

    start_idx = 0
    for i in range(n):
        if not math.isnan(close_2d[tid, i]):
            start_idx = i
            break

    if start_idx + timeperiod > n:
        for i in range(n):
            output_2d[tid, i] = math.nan
        return

    for i in range(start_idx + timeperiod - 1):
        output_2d[tid, i] = math.nan

    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += close_2d[tid, i]
    output_2d[tid, start_idx + timeperiod - 1] = sum_val / timeperiod

    for i in range(start_idx + timeperiod, n):
        if math.isnan(close_2d[tid, i]) or math.isnan(close_2d[tid, i - timeperiod]):
            output_2d[tid, i] = math.nan
        else:
            sum_val = sum_val - close_2d[tid, i - timeperiod] + close_2d[tid, i]
            output_2d[tid, i] = sum_val / timeperiod


@cuda.jit
def _ema_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    multiplier = 2.0 / (timeperiod + 1)

    start_idx = 0
    for i in range(n):
        if not math.isnan(close_2d[tid, i]):
            start_idx = i
            break

    if start_idx + timeperiod > n:
        for i in range(n):
            output_2d[tid, i] = math.nan
        return

    for i in range(start_idx + timeperiod - 1):
        output_2d[tid, i] = math.nan

    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += close_2d[tid, i]
    ema = sum_val / timeperiod
    output_2d[tid, start_idx + timeperiod - 1] = ema

    for i in range(start_idx + timeperiod, n):
        if math.isnan(close_2d[tid, i]):
            output_2d[tid, i] = math.nan
        else:
            ema = (close_2d[tid, i] - ema) * multiplier + ema
            output_2d[tid, i] = ema


@cuda.jit
def _bbands_batch_cuda(close_2d, timeperiod, nbdevup, nbdevdn,
                       upper_2d, middle_2d, lower_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    for i in range(timeperiod - 1):
        upper_2d[tid, i] = math.nan
        middle_2d[tid, i] = math.nan
        lower_2d[tid, i] = math.nan

    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close_2d[tid, i]
    sma = sum_val / timeperiod

    variance = 0.0
    for i in range(timeperiod):
        diff = close_2d[tid, i] - sma
        variance += diff * diff
    stddev = math.sqrt(variance / timeperiod)

    middle_2d[tid, timeperiod - 1] = sma
    upper_2d[tid, timeperiod - 1] = sma + nbdevup * stddev
    lower_2d[tid, timeperiod - 1] = sma - nbdevdn * stddev

    for i in range(timeperiod, n):
        sum_val = sum_val - close_2d[tid, i - timeperiod] + close_2d[tid, i]
        sma = sum_val / timeperiod

        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = close_2d[tid, j] - sma
            variance += diff * diff
        stddev = math.sqrt(variance / timeperiod)

        middle_2d[tid, i] = sma
        upper_2d[tid, i] = sma + nbdevup * stddev
        lower_2d[tid, i] = sma - nbdevdn * stddev


@cuda.jit
def _dema_batch_cuda(close_2d, timeperiod, ema1_2d, output_2d):
    """DEMA batch. ema1_2d is a pre-allocated temp array (num_tickers, num_bars)."""
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    multiplier = 2.0 / (timeperiod + 1)
    lookback = 2 * timeperiod - 2

    for i in range(lookback):
        output_2d[tid, i] = math.nan

    # EMA1
    for i in range(timeperiod - 1):
        ema1_2d[tid, i] = math.nan

    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close_2d[tid, i]
    ema1_2d[tid, timeperiod - 1] = sum_val / timeperiod

    for i in range(timeperiod, n):
        ema1_2d[tid, i] = (close_2d[tid, i] - ema1_2d[tid, i - 1]) * multiplier + ema1_2d[tid, i - 1]

    # EMA2
    sum_val = 0.0
    for i in range(timeperiod - 1, 2 * timeperiod - 1):
        sum_val += ema1_2d[tid, i]
    ema2 = sum_val / timeperiod

    output_2d[tid, 2 * timeperiod - 2] = 2.0 * ema1_2d[tid, 2 * timeperiod - 2] - ema2

    for i in range(2 * timeperiod - 1, n):
        ema2 = (ema1_2d[tid, i] - ema2) * multiplier + ema2
        output_2d[tid, i] = 2.0 * ema1_2d[tid, i] - ema2


@cuda.jit
def _kama_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    fastest = 2.0 / (2.0 + 1.0)
    slowest = 2.0 / (30.0 + 1.0)
    const_diff = fastest - slowest

    lookback_period = timeperiod

    for i in range(lookback_period):
        output_2d[tid, i] = math.nan

    if n <= lookback_period:
        return

    today = lookback_period
    trailing_idx = 0

    per_sum = 0.0
    for i in range(lookback_period):
        per_sum += abs(close_2d[tid, i + 1] - close_2d[tid, i])

    kama = close_2d[tid, today]

    if per_sum != 0.0:
        er = abs(close_2d[tid, today] - close_2d[tid, trailing_idx]) / per_sum
    else:
        er = 0.0

    sc = er * const_diff + slowest
    sc = sc * sc
    kama = kama + sc * (close_2d[tid, today] - kama)
    output_2d[tid, today] = kama

    today += 1
    trailing_idx += 1

    while today < n:
        per_sum -= abs(close_2d[tid, trailing_idx] - close_2d[tid, trailing_idx - 1])
        per_sum += abs(close_2d[tid, today] - close_2d[tid, today - 1])

        if per_sum != 0.0:
            er = abs(close_2d[tid, today] - close_2d[tid, trailing_idx]) / per_sum
        else:
            er = 0.0

        sc = er * const_diff + slowest
        sc = sc * sc
        kama = kama + sc * (close_2d[tid, today] - kama)
        output_2d[tid, today] = kama

        today += 1
        trailing_idx += 1


@cuda.jit
def _wma_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    weight_sum = (timeperiod * (timeperiod + 1)) / 2.0

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    weighted_sum = 0.0
    simple_sum = 0.0
    for j in range(timeperiod):
        weight = j + 1
        value = data_2d[tid, j]
        weighted_sum += value * weight
        simple_sum += value

    output_2d[tid, timeperiod - 1] = weighted_sum / weight_sum

    for i in range(timeperiod, n):
        old_value = data_2d[tid, i - timeperiod]
        new_value = data_2d[tid, i]

        weighted_sum = weighted_sum - simple_sum
        simple_sum = simple_sum - old_value + new_value
        weighted_sum = weighted_sum + new_value * timeperiod

        output_2d[tid, i] = weighted_sum / weight_sum


@cuda.jit
def _ema_into_row(data_2d, tid, timeperiod, multiplier, out_2d):
    """Helper: compute EMA of row tid from data_2d into out_2d (device-side, called from kernel)."""
    n = data_2d.shape[1]

    start_idx = 0
    for i in range(n):
        if not math.isnan(data_2d[tid, i]):
            start_idx = i
            break

    if start_idx + timeperiod > n:
        for i in range(n):
            out_2d[tid, i] = math.nan
        return

    for i in range(start_idx + timeperiod - 1):
        out_2d[tid, i] = math.nan

    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += data_2d[tid, i]
    ema = sum_val / timeperiod
    out_2d[tid, start_idx + timeperiod - 1] = ema

    for i in range(start_idx + timeperiod, n):
        if math.isnan(data_2d[tid, i]):
            out_2d[tid, i] = math.nan
        else:
            ema = (data_2d[tid, i] - ema) * multiplier + ema
            out_2d[tid, i] = ema


@cuda.jit
def _tema_batch_cuda(data_2d, timeperiod, ema1_2d, ema2_2d, ema3_2d, output_2d):
    """TEMA batch. Requires 3 pre-allocated temp arrays."""
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]
    multiplier = 2.0 / (timeperiod + 1)

    # EMA1 of data
    _ema_into_row(data_2d, tid, timeperiod, multiplier, ema1_2d)
    # EMA2 of EMA1
    _ema_into_row(ema1_2d, tid, timeperiod, multiplier, ema2_2d)
    # EMA3 of EMA2
    _ema_into_row(ema2_2d, tid, timeperiod, multiplier, ema3_2d)

    for i in range(n):
        output_2d[tid, i] = 3.0 * ema1_2d[tid, i] - 3.0 * ema2_2d[tid, i] + ema3_2d[tid, i]


@cuda.jit
def _t3_batch_cuda(data_2d, timeperiod, vfactor,
                   ema1_2d, ema2_2d, ema3_2d, ema4_2d, ema5_2d, ema6_2d, output_2d):
    """T3 batch. Requires 6 pre-allocated temp arrays."""
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]
    multiplier = 2.0 / (timeperiod + 1)

    c1 = -vfactor * vfactor * vfactor
    c2 = 3.0 * vfactor * vfactor + 3.0 * vfactor * vfactor * vfactor
    c3 = -6.0 * vfactor * vfactor - 3.0 * vfactor - 3.0 * vfactor * vfactor * vfactor
    c4 = 1.0 + 3.0 * vfactor + vfactor * vfactor * vfactor + 3.0 * vfactor * vfactor

    _ema_into_row(data_2d, tid, timeperiod, multiplier, ema1_2d)
    _ema_into_row(ema1_2d, tid, timeperiod, multiplier, ema2_2d)
    _ema_into_row(ema2_2d, tid, timeperiod, multiplier, ema3_2d)
    _ema_into_row(ema3_2d, tid, timeperiod, multiplier, ema4_2d)
    _ema_into_row(ema4_2d, tid, timeperiod, multiplier, ema5_2d)
    _ema_into_row(ema5_2d, tid, timeperiod, multiplier, ema6_2d)

    for i in range(n):
        output_2d[tid, i] = c1 * ema6_2d[tid, i] + c2 * ema5_2d[tid, i] + c3 * ema4_2d[tid, i] + c4 * ema3_2d[tid, i]


@cuda.jit
def _sma_into_row(data_2d, tid, timeperiod, out_2d):
    """Helper: compute SMA of row tid from data_2d into out_2d (device-side)."""
    n = data_2d.shape[1]

    if n < timeperiod:
        for i in range(n):
            out_2d[tid, i] = math.nan
        return

    start_idx = 0
    for i in range(n):
        if not math.isnan(data_2d[tid, i]):
            start_idx = i
            break

    if start_idx + timeperiod > n:
        for i in range(n):
            out_2d[tid, i] = math.nan
        return

    for i in range(start_idx + timeperiod - 1):
        out_2d[tid, i] = math.nan

    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += data_2d[tid, i]
    out_2d[tid, start_idx + timeperiod - 1] = sum_val / timeperiod

    for i in range(start_idx + timeperiod, n):
        if math.isnan(data_2d[tid, i]) or math.isnan(data_2d[tid, i - timeperiod]):
            out_2d[tid, i] = math.nan
        else:
            sum_val = sum_val - data_2d[tid, i - timeperiod] + data_2d[tid, i]
            out_2d[tid, i] = sum_val / timeperiod


@cuda.jit
def _trima_batch_cuda(data_2d, timeperiod, sma1_2d, output_2d):
    """TRIMA batch. Requires 1 pre-allocated temp array."""
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return

    if timeperiod % 2 == 1:
        n1 = (timeperiod + 1) // 2
        n2 = n1
    else:
        n1 = timeperiod // 2
        n2 = n1 + 1

    _sma_into_row(data_2d, tid, n1, sma1_2d)
    _sma_into_row(sma1_2d, tid, n2, output_2d)


@cuda.jit
def _mama_batch_cuda(close_2d, fastlimit, slowlimit, mama_2d, fama_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    lookback = 32

    for i in range(lookback):
        mama_2d[tid, i] = math.nan
        fama_2d[tid, i] = math.nan

    if n <= lookback:
        return

    mama_2d[tid, lookback] = close_2d[tid, lookback]
    fama_2d[tid, lookback] = close_2d[tid, lookback]

    for i in range(lookback + 1, n):
        price_change = abs(close_2d[tid, i] - close_2d[tid, i - 1])

        avg_change = 0.0
        lookback_window = min(10, i)
        for j in range(i - lookback_window, i):
            avg_change += abs(close_2d[tid, j] - close_2d[tid, j - 1])
        if lookback_window > 0:
            avg_change = avg_change / lookback_window
        else:
            avg_change = 1.0

        if avg_change > 0.0:
            alpha = price_change / avg_change * slowlimit
            if alpha < slowlimit:
                alpha = slowlimit
            elif alpha > fastlimit:
                alpha = fastlimit
        else:
            alpha = slowlimit

        mama_2d[tid, i] = alpha * close_2d[tid, i] + (1.0 - alpha) * mama_2d[tid, i - 1]

        fama_alpha = alpha * 0.5
        fama_2d[tid, i] = fama_alpha * mama_2d[tid, i] + (1.0 - fama_alpha) * fama_2d[tid, i - 1]


@cuda.jit
def _sar_batch_cuda(high_2d, low_2d, acceleration, maximum, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    is_long = True
    sar = low_2d[tid, 0]
    ep = high_2d[tid, 0]
    af = acceleration

    output_2d[tid, 0] = sar

    for i in range(1, n):
        sar = sar + af * (ep - sar)

        if is_long:
            if i >= 1:
                sar = min(sar, low_2d[tid, i - 1])
            if i >= 2:
                sar = min(sar, low_2d[tid, i - 2])

            if low_2d[tid, i] < sar:
                is_long = False
                sar = ep
                ep = low_2d[tid, i]
                af = acceleration
            else:
                if high_2d[tid, i] > ep:
                    ep = high_2d[tid, i]
                    af = min(af + acceleration, maximum)
        else:
            if i >= 1:
                sar = max(sar, high_2d[tid, i - 1])
            if i >= 2:
                sar = max(sar, high_2d[tid, i - 2])

            if high_2d[tid, i] > sar:
                is_long = True
                sar = ep
                ep = high_2d[tid, i]
                af = acceleration
            else:
                if low_2d[tid, i] < ep:
                    ep = low_2d[tid, i]
                    af = min(af + acceleration, maximum)

        output_2d[tid, i] = sar


@cuda.jit
def _sarext_batch_cuda(high_2d, low_2d,
                       startvalue, offsetonreverse,
                       accinit_long, acclong, accmax_long,
                       accinit_short, accshort, accmax_short,
                       output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    is_long = True
    if startvalue != 0:
        sar = startvalue
    else:
        sar = low_2d[tid, 0]
    ep = high_2d[tid, 0]
    af = accinit_long

    output_2d[tid, 0] = sar

    for i in range(1, n):
        sar = sar + af * (ep - sar)

        if is_long:
            if i >= 1:
                sar = min(sar, low_2d[tid, i - 1])
            if i >= 2:
                sar = min(sar, low_2d[tid, i - 2])

            if low_2d[tid, i] < sar:
                is_long = False
                sar = ep + offsetonreverse
                ep = low_2d[tid, i]
                af = accinit_short
            else:
                if high_2d[tid, i] > ep:
                    ep = high_2d[tid, i]
                    af = min(af + acclong, accmax_long)
        else:
            if i >= 1:
                sar = max(sar, high_2d[tid, i - 1])
            if i >= 2:
                sar = max(sar, high_2d[tid, i - 2])

            if high_2d[tid, i] > sar:
                is_long = True
                sar = ep - offsetonreverse
                ep = high_2d[tid, i]
                af = accinit_long
            else:
                if low_2d[tid, i] < ep:
                    ep = low_2d[tid, i]
                    af = min(af + accshort, accmax_short)

        output_2d[tid, i] = sar
