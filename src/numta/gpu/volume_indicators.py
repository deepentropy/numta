"""
Volume Indicators - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _ad_batch_cuda(high_2d, low_2d, close_2d, volume_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    ad_value = 0.0
    for i in range(n):
        hl_diff = high_2d[tid, i] - low_2d[tid, i]
        if hl_diff == 0.0:
            mf_mult = 0.0
        else:
            mf_mult = ((close_2d[tid, i] - low_2d[tid, i]) - (high_2d[tid, i] - close_2d[tid, i])) / hl_diff
        ad_value += mf_mult * volume_2d[tid, i]
        output_2d[tid, i] = ad_value


@cuda.jit
def _obv_batch_cuda(close_2d, volume_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    output_2d[tid, 0] = volume_2d[tid, 0]
    for i in range(1, n):
        if close_2d[tid, i] > close_2d[tid, i - 1]:
            output_2d[tid, i] = output_2d[tid, i - 1] + volume_2d[tid, i]
        elif close_2d[tid, i] < close_2d[tid, i - 1]:
            output_2d[tid, i] = output_2d[tid, i - 1] - volume_2d[tid, i]
        else:
            output_2d[tid, i] = output_2d[tid, i - 1]


@cuda.jit
def _adosc_batch_cuda(high_2d, low_2d, close_2d, volume_2d,
                      fastperiod, slowperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    # Fill lookback with NaN
    for i in range(slowperiod - 1):
        output_2d[tid, i] = math.nan

    # Compute AD line inline
    # Then compute fast EMA and slow EMA of AD line
    # Need full AD line first
    ad_value = 0.0

    # Fast EMA init: SMA of first fastperiod AD values
    fast_mult = 2.0 / (fastperiod + 1)
    slow_mult = 2.0 / (slowperiod + 1)

    # Compute AD values and accumulate SMA seeds
    fast_sum = 0.0
    slow_sum = 0.0
    fast_ema = 0.0
    slow_ema = 0.0
    fast_init = False
    slow_init = False

    for i in range(n):
        hl_diff = high_2d[tid, i] - low_2d[tid, i]
        if hl_diff == 0.0:
            mf_mult = 0.0
        else:
            mf_mult = ((close_2d[tid, i] - low_2d[tid, i]) - (high_2d[tid, i] - close_2d[tid, i])) / hl_diff
        ad_value += mf_mult * volume_2d[tid, i]
        ad_val = ad_value

        if i < fastperiod:
            fast_sum += ad_val
        if i == fastperiod - 1:
            fast_ema = fast_sum / fastperiod
            fast_init = True
        elif fast_init:
            fast_ema = (ad_val - fast_ema) * fast_mult + fast_ema

        if i < slowperiod:
            slow_sum += ad_val
        if i == slowperiod - 1:
            slow_ema = slow_sum / slowperiod
            slow_init = True
        elif slow_init:
            slow_ema = (ad_val - slow_ema) * slow_mult + slow_ema

        if i >= slowperiod - 1 and fast_init and slow_init:
            output_2d[tid, i] = fast_ema - slow_ema
