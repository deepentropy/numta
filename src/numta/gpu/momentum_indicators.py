"""
Momentum Indicators - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _rsi_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    lookback = timeperiod
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= timeperiod:
        return

    # Calculate initial average gain/loss
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, timeperiod + 1):
        change = data_2d[tid, i] - data_2d[tid, i - 1]
        if change > 0:
            avg_gain += change
        elif change < 0:
            avg_loss += -change
    avg_gain /= timeperiod
    avg_loss /= timeperiod

    if avg_loss == 0.0:
        if avg_gain == 0.0:
            output_2d[tid, timeperiod] = 50.0
        else:
            output_2d[tid, timeperiod] = 100.0
    else:
        rs = avg_gain / avg_loss
        output_2d[tid, timeperiod] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(timeperiod + 1, n):
        change = data_2d[tid, i] - data_2d[tid, i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (timeperiod - 1) + gain) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + loss) / timeperiod

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                output_2d[tid, i] = 50.0
            else:
                output_2d[tid, i] = 100.0
        else:
            rs = avg_gain / avg_loss
            output_2d[tid, i] = 100.0 - (100.0 / (1.0 + rs))


@cuda.jit
def _macd_batch_cuda(close_2d, fastperiod, slowperiod, signalperiod,
                     macd_2d, signal_2d, hist_2d):
    """Mirrors CPU _macd_numba: build fast_ema/slow_ema arrays, then signal."""
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    lookback = slowperiod + signalperiod - 2

    if n <= lookback:
        for i in range(n):
            macd_2d[tid, i] = math.nan
            signal_2d[tid, i] = math.nan
            hist_2d[tid, i] = math.nan
        return

    fast_mult = 2.0 / (fastperiod + 1)
    slow_mult = 2.0 / (slowperiod + 1)
    signal_mult = 2.0 / (signalperiod + 1)

    for i in range(lookback):
        macd_2d[tid, i] = math.nan
        signal_2d[tid, i] = math.nan
        hist_2d[tid, i] = math.nan

    # Fast EMA: init as SMA at fastperiod-1, then EMA forward
    sum_val = 0.0
    for i in range(fastperiod):
        sum_val += close_2d[tid, i]
    fast_ema = sum_val / fastperiod
    # fast_ema now = fast_ema[fastperiod-1]

    # Advance fast EMA to slowperiod-1
    for i in range(fastperiod, slowperiod):
        fast_ema = (close_2d[tid, i] - fast_ema) * fast_mult + fast_ema
    # fast_ema now = fast_ema[slowperiod-1]

    # Slow EMA: init as SMA at slowperiod-1
    sum_val = 0.0
    for i in range(slowperiod):
        sum_val += close_2d[tid, i]
    slow_ema = sum_val / slowperiod
    # slow_ema now = slow_ema[slowperiod-1]

    # MACD line from slowperiod-1
    macd_2d[tid, slowperiod - 1] = fast_ema - slow_ema

    for i in range(slowperiod, n):
        fast_ema = (close_2d[tid, i] - fast_ema) * fast_mult + fast_ema
        slow_ema = (close_2d[tid, i] - slow_ema) * slow_mult + slow_ema
        macd_2d[tid, i] = fast_ema - slow_ema

    # Signal: EMA of MACD starting at slowperiod-1
    signal_start = slowperiod + signalperiod - 2
    sum_val = 0.0
    for i in range(slowperiod - 1, slowperiod - 1 + signalperiod):
        sum_val += macd_2d[tid, i]
    signal_ema = sum_val / signalperiod
    signal_2d[tid, signal_start] = signal_ema

    for i in range(signal_start + 1, n):
        signal_ema = (macd_2d[tid, i] - signal_ema) * signal_mult + signal_ema
        signal_2d[tid, i] = signal_ema

    # Histogram
    for i in range(signal_start, n):
        hist_2d[tid, i] = macd_2d[tid, i] - signal_2d[tid, i]


@cuda.jit
def _adx_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    lookback = 2 * timeperiod - 1
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= lookback:
        return

    # TR, +DM, -DM + smoothing
    # First TR
    prev_tr = high_2d[tid, 0] - low_2d[tid, 0]
    smoothed_tr = prev_tr
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0

    for i in range(1, timeperiod):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr_val = max(hl, max(hc, lc))
        smoothed_tr += tr_val

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        if up_move > down_move and up_move > 0:
            smoothed_plus_dm += up_move
        if down_move > up_move and down_move > 0:
            smoothed_minus_dm += down_move

    # Compute DX from timeperiod to 2*timeperiod-1
    # We need to store DX values for averaging
    # Use local accumulator for first timeperiod DX values
    dx_sum = 0.0
    for i in range(timeperiod, 2 * timeperiod):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr_val = max(hl, max(hc, lc))

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        plus_dm_val = 0.0
        minus_dm_val = 0.0
        if up_move > down_move and up_move > 0:
            plus_dm_val = up_move
        if down_move > up_move and down_move > 0:
            minus_dm_val = down_move

        smoothed_tr = smoothed_tr - smoothed_tr / timeperiod + tr_val
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / timeperiod + plus_dm_val
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / timeperiod + minus_dm_val

        if smoothed_tr != 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        if di_sum != 0:
            dx_val = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx_val = 0.0

        dx_sum += dx_val

    adx = dx_sum / timeperiod
    output_2d[tid, 2 * timeperiod - 1] = adx

    for i in range(2 * timeperiod, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr_val = max(hl, max(hc, lc))

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        plus_dm_val = 0.0
        minus_dm_val = 0.0
        if up_move > down_move and up_move > 0:
            plus_dm_val = up_move
        if down_move > up_move and down_move > 0:
            minus_dm_val = down_move

        smoothed_tr = smoothed_tr - smoothed_tr / timeperiod + tr_val
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / timeperiod + plus_dm_val
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / timeperiod + minus_dm_val

        if smoothed_tr != 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        if di_sum != 0:
            dx_val = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx_val = 0.0

        adx = adx + (dx_val - adx) / timeperiod
        output_2d[tid, i] = adx


@cuda.jit
def _atr_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    if n < timeperiod + 1:
        return

    atr_sum = 0.0
    for i in range(1, timeperiod + 1):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        atr_sum += max(hl, max(hc, lc))

    atr = atr_sum / timeperiod
    output_2d[tid, timeperiod] = atr

    for i in range(timeperiod + 1, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr = max(hl, max(hc, lc))
        atr = atr - (atr / timeperiod) + (tr / timeperiod)
        output_2d[tid, i] = atr


@cuda.jit
def _cci_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        sum_tp = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            sum_tp += (high_2d[tid, j] + low_2d[tid, j] + close_2d[tid, j]) / 3.0

        sma_tp = sum_tp / timeperiod
        current_tp = (high_2d[tid, i] + low_2d[tid, i] + close_2d[tid, i]) / 3.0

        sum_abs_dev = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            tp = (high_2d[tid, j] + low_2d[tid, j] + close_2d[tid, j]) / 3.0
            sum_abs_dev += abs(tp - sma_tp)

        mean_abs_dev = sum_abs_dev / timeperiod

        if mean_abs_dev == 0.0:
            output_2d[tid, i] = 0.0
        else:
            output_2d[tid, i] = (current_tp - sma_tp) / (0.015 * mean_abs_dev)


@cuda.jit
def _cmo_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        sum_gains = 0.0
        sum_losses = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            change = close_2d[tid, j] - close_2d[tid, j - 1]
            if change > 0:
                sum_gains += change
            elif change < 0:
                sum_losses += abs(change)

        total = sum_gains + sum_losses
        if total == 0.0:
            output_2d[tid, i] = 0.0
        else:
            output_2d[tid, i] = ((sum_gains - sum_losses) / total) * 100.0


@cuda.jit
def _dx_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    lookback = timeperiod
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= timeperiod:
        return

    smoothed_tr = high_2d[tid, 0] - low_2d[tid, 0]
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0

    for i in range(1, timeperiod):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        smoothed_tr += max(hl, max(hc, lc))

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        if up_move > down_move and up_move > 0:
            smoothed_plus_dm += up_move
        if down_move > up_move and down_move > 0:
            smoothed_minus_dm += down_move

    for i in range(timeperiod, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr_val = max(hl, max(hc, lc))

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        plus_dm_val = 0.0
        minus_dm_val = 0.0
        if up_move > down_move and up_move > 0:
            plus_dm_val = up_move
        if down_move > up_move and down_move > 0:
            minus_dm_val = down_move

        smoothed_tr = smoothed_tr - smoothed_tr / timeperiod + tr_val
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / timeperiod + plus_dm_val
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / timeperiod + minus_dm_val

        if smoothed_tr != 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        if di_sum != 0:
            output_2d[tid, i] = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _mom_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        output_2d[tid, i] = data_2d[tid, i] - data_2d[tid, i - timeperiod]


@cuda.jit
def _roc_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        prev = data_2d[tid, i - timeperiod]
        if prev == 0.0:
            output_2d[tid, i] = math.nan
        else:
            output_2d[tid, i] = ((data_2d[tid, i] / prev) - 1.0) * 100.0


@cuda.jit
def _rocp_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        prev = data_2d[tid, i - timeperiod]
        if prev == 0.0:
            output_2d[tid, i] = math.nan
        else:
            output_2d[tid, i] = (data_2d[tid, i] - prev) / prev


@cuda.jit
def _rocr_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        prev = data_2d[tid, i - timeperiod]
        if prev == 0.0:
            output_2d[tid, i] = math.nan
        else:
            output_2d[tid, i] = data_2d[tid, i] / prev


@cuda.jit
def _rocr100_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        prev = data_2d[tid, i - timeperiod]
        if prev == 0.0:
            output_2d[tid, i] = math.nan
        else:
            output_2d[tid, i] = (data_2d[tid, i] / prev) * 100.0


@cuda.jit
def _willr_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        highest = high_2d[tid, i]
        lowest = low_2d[tid, i]
        for j in range(i - timeperiod + 1, i + 1):
            if high_2d[tid, j] > highest:
                highest = high_2d[tid, j]
            if low_2d[tid, j] < lowest:
                lowest = low_2d[tid, j]

        if highest == lowest:
            output_2d[tid, i] = -50.0
        else:
            output_2d[tid, i] = ((highest - close_2d[tid, i]) / (highest - lowest)) * -100.0


@cuda.jit
def _stoch_fastk_batch_cuda(high_2d, low_2d, close_2d, fastk_period, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(fastk_period - 1):
        output_2d[tid, i] = math.nan

    for i in range(fastk_period - 1, n):
        highest = high_2d[tid, i]
        lowest = low_2d[tid, i]
        for j in range(i - fastk_period + 1, i + 1):
            if high_2d[tid, j] > highest:
                highest = high_2d[tid, j]
            if low_2d[tid, j] < lowest:
                lowest = low_2d[tid, j]

        if highest - lowest == 0:
            output_2d[tid, i] = 50.0
        else:
            output_2d[tid, i] = ((close_2d[tid, i] - lowest) / (highest - lowest)) * 100.0


@cuda.jit
def _bop_batch_cuda(open_2d, high_2d, low_2d, close_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= open_2d.shape[0]:
        return
    n = open_2d.shape[1]

    for i in range(n):
        denom = high_2d[tid, i] - low_2d[tid, i]
        if denom == 0.0:
            output_2d[tid, i] = 0.0
        else:
            output_2d[tid, i] = (close_2d[tid, i] - open_2d[tid, i]) / denom


@cuda.jit
def _aroon_batch_cuda(high_2d, low_2d, timeperiod, aroondown_2d, aroonup_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod):
        aroondown_2d[tid, i] = math.nan
        aroonup_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        highest = high_2d[tid, i]
        lowest = low_2d[tid, i]
        high_idx = i
        low_idx = i
        for j in range(i - timeperiod, i + 1):
            if high_2d[tid, j] >= highest:
                highest = high_2d[tid, j]
                high_idx = j
            if low_2d[tid, j] <= lowest:
                lowest = low_2d[tid, j]
                low_idx = j

        aroonup_2d[tid, i] = ((timeperiod - (i - high_idx)) / timeperiod) * 100.0
        aroondown_2d[tid, i] = ((timeperiod - (i - low_idx)) / timeperiod) * 100.0


@cuda.jit
def _aroonosc_batch_cuda(high_2d, low_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        periods_since_high = 0
        periods_since_low = 0
        highest = high_2d[tid, i]
        lowest = low_2d[tid, i]

        for j in range(i - timeperiod + 1, i + 1):
            if high_2d[tid, j] >= highest:
                highest = high_2d[tid, j]
                periods_since_high = i - j
            if low_2d[tid, j] <= lowest:
                lowest = low_2d[tid, j]
                periods_since_low = i - j

        aroon_up = ((timeperiod - periods_since_high) / timeperiod) * 100.0
        aroon_down = ((timeperiod - periods_since_low) / timeperiod) * 100.0
        output_2d[tid, i] = aroon_up - aroon_down


@cuda.jit
def _mfi_batch_cuda(high_2d, low_2d, close_2d, volume_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod, n):
        positive_mf = 0.0
        negative_mf = 0.0

        for j in range(i - timeperiod + 1, i + 1):
            tp = (high_2d[tid, j] + low_2d[tid, j] + close_2d[tid, j]) / 3.0
            mf = tp * volume_2d[tid, j]
            if j > 0:
                prev_tp = (high_2d[tid, j - 1] + low_2d[tid, j - 1] + close_2d[tid, j - 1]) / 3.0
                if tp > prev_tp:
                    positive_mf += mf
                elif tp < prev_tp:
                    negative_mf += mf

        if negative_mf == 0.0:
            output_2d[tid, i] = 100.0
        else:
            mf_ratio = positive_mf / negative_mf
            output_2d[tid, i] = 100.0 - (100.0 / (1.0 + mf_ratio))


@cuda.jit
def _minus_dm_batch_cuda(high_2d, low_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    lookback = timeperiod - 1
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= lookback:
        return

    smoothed = 0.0
    for i in range(1, timeperiod):
        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        if down_move > up_move and down_move > 0:
            smoothed += down_move
    output_2d[tid, timeperiod - 1] = smoothed

    for i in range(timeperiod, n):
        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        dm_val = 0.0
        if down_move > up_move and down_move > 0:
            dm_val = down_move
        smoothed = (smoothed * (timeperiod - 1) + dm_val) / timeperiod
        output_2d[tid, i] = smoothed


@cuda.jit
def _plus_dm_batch_cuda(high_2d, low_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    lookback = timeperiod - 1
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= lookback:
        return

    smoothed = 0.0
    for i in range(1, timeperiod):
        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        if up_move > down_move and up_move > 0:
            smoothed += up_move
    output_2d[tid, timeperiod - 1] = smoothed

    for i in range(timeperiod, n):
        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        dm_val = 0.0
        if up_move > down_move and up_move > 0:
            dm_val = up_move
        smoothed = (smoothed * (timeperiod - 1) + dm_val) / timeperiod
        output_2d[tid, i] = smoothed


@cuda.jit
def _minus_di_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    lookback = timeperiod - 1
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= lookback:
        return

    smoothed_tr = 0.0
    smoothed_dm = 0.0
    for i in range(1, timeperiod):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        smoothed_tr += max(hl, max(hc, lc))

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        if down_move > up_move and down_move > 0:
            smoothed_dm += down_move

    if smoothed_tr != 0.0:
        output_2d[tid, timeperiod - 1] = 100.0 * smoothed_dm / smoothed_tr
    else:
        output_2d[tid, timeperiod - 1] = 0.0

    for i in range(timeperiod, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr_val = max(hl, max(hc, lc))
        smoothed_tr = (smoothed_tr * (timeperiod - 1) + tr_val) / timeperiod

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        dm_val = 0.0
        if down_move > up_move and down_move > 0:
            dm_val = down_move
        smoothed_dm = (smoothed_dm * (timeperiod - 1) + dm_val) / timeperiod

        if smoothed_tr != 0.0:
            output_2d[tid, i] = 100.0 * smoothed_dm / smoothed_tr
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _plus_di_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    lookback = timeperiod - 1
    for i in range(lookback):
        output_2d[tid, i] = math.nan

    if n <= lookback:
        return

    smoothed_tr = 0.0
    smoothed_dm = 0.0
    for i in range(1, timeperiod):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        smoothed_tr += max(hl, max(hc, lc))

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        if up_move > down_move and up_move > 0:
            smoothed_dm += up_move

    if smoothed_tr != 0.0:
        output_2d[tid, timeperiod - 1] = 100.0 * smoothed_dm / smoothed_tr
    else:
        output_2d[tid, timeperiod - 1] = 0.0

    for i in range(timeperiod, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr_val = max(hl, max(hc, lc))
        smoothed_tr = (smoothed_tr * (timeperiod - 1) + tr_val) / timeperiod

        up_move = high_2d[tid, i] - high_2d[tid, i - 1]
        down_move = low_2d[tid, i - 1] - low_2d[tid, i]
        dm_val = 0.0
        if up_move > down_move and up_move > 0:
            dm_val = up_move
        smoothed_dm = (smoothed_dm * (timeperiod - 1) + dm_val) / timeperiod

        if smoothed_tr != 0.0:
            output_2d[tid, i] = 100.0 * smoothed_dm / smoothed_tr
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _ultosc_batch_cuda(high_2d, low_2d, close_2d,
                       timeperiod1, timeperiod2, timeperiod3, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    max_period = max(timeperiod1, max(timeperiod2, timeperiod3))
    lookback = max_period - 1

    for i in range(lookback):
        output_2d[tid, i] = math.nan

    for i in range(lookback, n):
        # Calculate BP and TR sums for each period
        sum_bp1 = 0.0
        sum_tr1 = 0.0
        for j in range(i - timeperiod1 + 1, i + 1):
            prev_close = close_2d[tid, j - 1] if j > 0 else close_2d[tid, 0]
            bp = close_2d[tid, j] - min(low_2d[tid, j], prev_close)
            hl = high_2d[tid, j] - low_2d[tid, j]
            hc = abs(high_2d[tid, j] - prev_close)
            lc = abs(low_2d[tid, j] - prev_close)
            tr = max(hl, max(hc, lc))
            sum_bp1 += bp
            sum_tr1 += tr
        avg1 = sum_bp1 / sum_tr1 if sum_tr1 != 0.0 else 0.0

        sum_bp2 = 0.0
        sum_tr2 = 0.0
        for j in range(i - timeperiod2 + 1, i + 1):
            prev_close = close_2d[tid, j - 1] if j > 0 else close_2d[tid, 0]
            bp = close_2d[tid, j] - min(low_2d[tid, j], prev_close)
            hl = high_2d[tid, j] - low_2d[tid, j]
            hc = abs(high_2d[tid, j] - prev_close)
            lc = abs(low_2d[tid, j] - prev_close)
            tr = max(hl, max(hc, lc))
            sum_bp2 += bp
            sum_tr2 += tr
        avg2 = sum_bp2 / sum_tr2 if sum_tr2 != 0.0 else 0.0

        sum_bp3 = 0.0
        sum_tr3 = 0.0
        for j in range(i - timeperiod3 + 1, i + 1):
            prev_close = close_2d[tid, j - 1] if j > 0 else close_2d[tid, 0]
            bp = close_2d[tid, j] - min(low_2d[tid, j], prev_close)
            hl = high_2d[tid, j] - low_2d[tid, j]
            hc = abs(high_2d[tid, j] - prev_close)
            lc = abs(low_2d[tid, j] - prev_close)
            tr = max(hl, max(hc, lc))
            sum_bp3 += bp
            sum_tr3 += tr
        avg3 = sum_bp3 / sum_tr3 if sum_tr3 != 0.0 else 0.0

        output_2d[tid, i] = 100.0 * ((4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0)
