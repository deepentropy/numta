"""
Volatility Indicators - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _trange_batch_cuda(high_2d, low_2d, close_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    # First bar: just high - low
    output_2d[tid, 0] = high_2d[tid, 0] - low_2d[tid, 0]

    for i in range(1, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        output_2d[tid, i] = max(hl, max(hc, lc))


@cuda.jit
def _natr_batch_cuda(high_2d, low_2d, close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod):
        output_2d[tid, i] = math.nan

    if n < timeperiod + 1:
        return

    # Compute TR inline, accumulate first ATR
    atr_sum = 0.0
    for i in range(1, timeperiod + 1):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        atr_sum += max(hl, max(hc, lc))

    atr = atr_sum / timeperiod
    if close_2d[tid, timeperiod] != 0.0:
        output_2d[tid, timeperiod] = (atr / close_2d[tid, timeperiod]) * 100.0
    else:
        output_2d[tid, timeperiod] = 0.0

    for i in range(timeperiod + 1, n):
        hl = high_2d[tid, i] - low_2d[tid, i]
        hc = abs(high_2d[tid, i] - close_2d[tid, i - 1])
        lc = abs(low_2d[tid, i] - close_2d[tid, i - 1])
        tr = max(hl, max(hc, lc))
        atr = atr - (atr / timeperiod) + (tr / timeperiod)
        if close_2d[tid, i] != 0.0:
            output_2d[tid, i] = (atr / close_2d[tid, i]) * 100.0
        else:
            output_2d[tid, i] = 0.0
