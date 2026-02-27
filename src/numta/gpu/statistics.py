"""
Statistics - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _stddev_batch_cuda(data_2d, timeperiod, nbdev, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        mean_val = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            mean_val += data_2d[tid, j]
        mean_val /= timeperiod

        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = data_2d[tid, j] - mean_val
            variance += diff * diff
        variance /= timeperiod

        output_2d[tid, i] = math.sqrt(variance) * nbdev


@cuda.jit
def _var_batch_cuda(data_2d, timeperiod, nbdev, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        mean_val = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            mean_val += data_2d[tid, j]
        mean_val /= timeperiod

        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = data_2d[tid, j] - mean_val
            variance += diff * diff
        variance /= timeperiod

        output_2d[tid, i] = variance * nbdev


@cuda.jit
def _tsf_batch_cuda(data_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= data_2d.shape[0]:
        return
    n = data_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    sum_x = timeperiod * (timeperiod - 1) / 2.0
    sum_xx = timeperiod * (timeperiod - 1) * (2 * timeperiod - 1) / 6.0

    for i in range(timeperiod - 1, n):
        sum_y = 0.0
        sum_xy = 0.0
        for j in range(timeperiod):
            y = data_2d[tid, i - timeperiod + 1 + j]
            sum_y += y
            sum_xy += j * y

        denominator = timeperiod * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            output_2d[tid, i] = sum_y / timeperiod
        else:
            b = (timeperiod * sum_xy - sum_x * sum_y) / denominator
            a = (sum_y - b * sum_x) / timeperiod
            output_2d[tid, i] = a + b * timeperiod
