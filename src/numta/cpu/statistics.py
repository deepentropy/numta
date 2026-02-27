"""
Statistics Functions - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for statistical functions.
"""

import numpy as np
from numba import jit


__all__ = [
    "_stddev_numba",
    "_tsf_numba",
    "_var_numba",
]


@jit(nopython=True, cache=True)
def _stddev_numba(data: np.ndarray, timeperiod: int, nbdev: float, output: np.ndarray) -> None:
    """
    Numba-compiled STDDEV calculation (in-place)

    Formula: STDDEV = sqrt(sum((x - mean)^2) / n) * nbdev
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate standard deviation for each window
    for i in range(timeperiod - 1, n):
        # Calculate mean
        mean_val = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            mean_val += data[j]
        mean_val /= timeperiod

        # Calculate variance
        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = data[j] - mean_val
            variance += diff * diff
        variance /= timeperiod

        # Calculate standard deviation
        output[i] = np.sqrt(variance) * nbdev


@jit(nopython=True, cache=True)
def _tsf_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """Numba-compiled TSF calculation using linear regression"""
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Precalculate sums for x values (0, 1, 2, ..., timeperiod-1)
    # Closed-form: sum(0..n-1) = n*(n-1)/2, sum_sq(0..n-1) = n*(n-1)*(2n-1)/6
    sum_x = timeperiod * (timeperiod - 1) / 2.0
    sum_xx = timeperiod * (timeperiod - 1) * (2 * timeperiod - 1) / 6.0

    # Calculate TSF for each window
    for i in range(timeperiod - 1, n):
        # Calculate sums for current window
        sum_y = 0.0
        sum_xy = 0.0

        for j in range(timeperiod):
            y = data[i - timeperiod + 1 + j]
            sum_y += y
            sum_xy += j * y

        # Linear regression: y = a + b*x
        # b = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
        # a = (sum_y - b*sum_x) / n

        denominator = timeperiod * sum_xx - sum_x * sum_x

        if abs(denominator) < 1e-10:
            # Degenerate case: all x values the same (shouldn't happen)
            output[i] = sum_y / timeperiod
        else:
            b = (timeperiod * sum_xy - sum_x * sum_y) / denominator
            a = (sum_y - b * sum_x) / timeperiod

            # Forecast next value: x = timeperiod (one step ahead)
            output[i] = a + b * timeperiod


@jit(nopython=True, cache=True)
def _var_numba(data: np.ndarray, timeperiod: int, nbdev: float, output: np.ndarray) -> None:
    """Numba-compiled VAR calculation (population variance)"""
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate variance for each window
    for i in range(timeperiod - 1, n):
        # Calculate mean for window
        sum_val = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            sum_val += data[j]
        mean = sum_val / timeperiod

        # Calculate sum of squared deviations
        sum_sq_dev = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            dev = data[j] - mean
            sum_sq_dev += dev * dev

        # Population variance (divide by N, not N-1)
        variance = sum_sq_dev / timeperiod
        output[i] = variance * nbdev
