"""
Price Transform - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _medprice_batch_cuda(high_2d, low_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]
    for i in range(n):
        output_2d[tid, i] = (high_2d[tid, i] + low_2d[tid, i]) / 2.0


@cuda.jit
def _midpoint_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        max_val = data_2d[tid, i]
        min_val = data_2d[tid, i]
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] > max_val:
                max_val = data_2d[tid, j]
            if data_2d[tid, j] < min_val:
                min_val = data_2d[tid, j]
        output_2d[tid, i] = (max_val + min_val) / 2.0


@cuda.jit
def _midprice_batch_cuda(high_2d, low_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        max_high = high_2d[tid, i]
        min_low = low_2d[tid, i]
        for j in range(i - timeperiod + 1, i + 1):
            if high_2d[tid, j] > max_high:
                max_high = high_2d[tid, j]
            if low_2d[tid, j] < min_low:
                min_low = low_2d[tid, j]
        output_2d[tid, i] = (max_high + min_low) / 2.0


@cuda.jit
def _typprice_batch_cuda(high_2d, low_2d, close_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]
    for i in range(n):
        output_2d[tid, i] = (high_2d[tid, i] + low_2d[tid, i] + close_2d[tid, i]) / 3.0


@cuda.jit
def _wclprice_batch_cuda(high_2d, low_2d, close_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]
    for i in range(n):
        output_2d[tid, i] = (high_2d[tid, i] + low_2d[tid, i] + close_2d[tid, i] * 2.0) / 4.0
