"""
Statistic Functions - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
"""

import math
from numba import cuda


@cuda.jit
def _beta_batch_cuda(high_2d, low_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        ws = i - timeperiod + 1
        mean_h = 0.0
        mean_l = 0.0
        for j in range(ws, i + 1):
            mean_h += high_2d[tid, j]
            mean_l += low_2d[tid, j]
        mean_h /= timeperiod
        mean_l /= timeperiod

        cov = 0.0
        var = 0.0
        for j in range(ws, i + 1):
            hd = high_2d[tid, j] - mean_h
            ld = low_2d[tid, j] - mean_l
            cov += hd * ld
            var += ld * ld

        if var > 1e-10:
            output_2d[tid, i] = cov / var
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _correl_batch_cuda(high_2d, low_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= high_2d.shape[0]:
        return
    n = high_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        ws = i - timeperiod + 1
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for j in range(ws, i + 1):
            x = high_2d[tid, j]
            y = low_2d[tid, j]
            sx += x
            sy += y
            sxx += x * x
            syy += y * y
            sxy += x * y

        num = timeperiod * sxy - sx * sy
        dx = timeperiod * sxx - sx * sx
        dy = timeperiod * syy - sy * sy

        if dx <= 0.0 or dy <= 0.0:
            output_2d[tid, i] = 0.0
        else:
            denom = (dx * dy) ** 0.5
            if denom == 0.0:
                output_2d[tid, i] = 0.0
            else:
                output_2d[tid, i] = num / denom


@cuda.jit
def _linearreg_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        ws = i - timeperiod + 1
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for j in range(ws, i + 1):
            x = float(j - ws)
            y = close_2d[tid, j]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y

        np_ = float(timeperiod)
        denom = np_ * sxx - sx * sx
        if denom != 0.0:
            m = (np_ * sxy - sx * sy) / denom
            b = (sy - m * sx) / np_
            output_2d[tid, i] = b + m * (timeperiod - 1)
        else:
            output_2d[tid, i] = sy / np_


@cuda.jit
def _linearreg_angle_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        ws = i - timeperiod + 1
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for j in range(ws, i + 1):
            x = float(j - ws)
            y = close_2d[tid, j]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y

        np_ = float(timeperiod)
        denom = np_ * sxx - sx * sx
        if denom != 0.0:
            m = (np_ * sxy - sx * sy) / denom
            output_2d[tid, i] = math.atan(m) * (180.0 / math.pi)
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _linearreg_intercept_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        ws = i - timeperiod + 1
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for j in range(ws, i + 1):
            x = float(j - ws)
            y = close_2d[tid, j]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y

        np_ = float(timeperiod)
        denom = np_ * sxx - sx * sx
        if denom != 0.0:
            m = (np_ * sxy - sx * sy) / denom
            output_2d[tid, i] = (sy - m * sx) / np_
        else:
            output_2d[tid, i] = sy / np_


@cuda.jit
def _linearreg_slope_batch_cuda(close_2d, timeperiod, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]

    for i in range(timeperiod - 1):
        output_2d[tid, i] = math.nan

    for i in range(timeperiod - 1, n):
        ws = i - timeperiod + 1
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for j in range(ws, i + 1):
            x = float(j - ws)
            y = close_2d[tid, j]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y

        np_ = float(timeperiod)
        denom = np_ * sxx - sx * sx
        if denom != 0.0:
            output_2d[tid, i] = (np_ * sxy - sx * sy) / denom
        else:
            output_2d[tid, i] = 0.0
