"""
Math Operators - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _max_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        max_val = data_2d[tid, i]
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] > max_val:
                max_val = data_2d[tid, j]
        output_2d[tid, i] = max_val


@cuda.jit
def _maxindex_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        max_val = data_2d[tid, i]
        max_idx = 0
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] >= max_val:
                max_val = data_2d[tid, j]
                max_idx = i - j
        output_2d[tid, i] = float(max_idx)


@cuda.jit
def _min_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        min_val = data_2d[tid, i]
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] < min_val:
                min_val = data_2d[tid, j]
        output_2d[tid, i] = min_val


@cuda.jit
def _minindex_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        min_val = data_2d[tid, i]
        min_idx = 0
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] <= min_val:
                min_val = data_2d[tid, j]
                min_idx = i - j
        output_2d[tid, i] = float(min_idx)


@cuda.jit
def _minmax_batch_cuda(data_2d, timeperiod, min_out_2d, max_out_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        min_out_2d[tid, i] = math.nan
        max_out_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        min_val = data_2d[tid, i]
        max_val = data_2d[tid, i]
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] < min_val:
                min_val = data_2d[tid, j]
            if data_2d[tid, j] > max_val:
                max_val = data_2d[tid, j]
        min_out_2d[tid, i] = min_val
        max_out_2d[tid, i] = max_val


@cuda.jit
def _minmaxindex_batch_cuda(data_2d, timeperiod, min_idx_out_2d, max_idx_out_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        min_idx_out_2d[tid, i] = math.nan
        max_idx_out_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        min_val = data_2d[tid, i]
        max_val = data_2d[tid, i]
        min_idx = 0
        max_idx = 0
        for j in range(i - timeperiod + 1, i + 1):
            if data_2d[tid, j] <= min_val:
                min_val = data_2d[tid, j]
                min_idx = i - j
            if data_2d[tid, j] >= max_val:
                max_val = data_2d[tid, j]
                max_idx = i - j
        min_idx_out_2d[tid, i] = float(min_idx)
        max_idx_out_2d[tid, i] = float(max_idx)


@cuda.jit
def _sum_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    # First sum
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += data_2d[tid, i]
    output_2d[tid, timeperiod - 1] = sum_val

    # Rolling sum
    for i in range(timeperiod, n):
        sum_val = sum_val - data_2d[tid, i - timeperiod] + data_2d[tid, i]
        output_2d[tid, i] = sum_val
