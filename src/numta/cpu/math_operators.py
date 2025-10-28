"""
Math Operators - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for mathematical operations.
"""

import numpy as np
from numba import jit


__all__ = [
    "_max_numba",
    "_maxindex_numba",
    "_min_numba",
    "_minindex_numba",
    "_minmax_numba",
    "_minmaxindex_numba",
    "_sum_numba",
]


@jit(nopython=True, cache=True)
def _max_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MAX calculation (in-place)

    Finds the highest value over a rolling window.
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate max for each window
    for i in range(timeperiod - 1, n):
        max_val = data[i]
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] > max_val:
                max_val = data[j]
        output[i] = max_val


@jit(nopython=True, cache=True)
def _maxindex_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MAXINDEX calculation (in-place)

    Finds the index of the highest value over a rolling window.
    Returns the distance from current position (0 = current bar).
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate maxindex for each window
    for i in range(timeperiod - 1, n):
        max_val = data[i]
        max_idx = 0  # Distance from current position
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] >= max_val:  # >= to get the most recent if tied
                max_val = data[j]
                max_idx = i - j  # Distance from current
        output[i] = float(max_idx)


@jit(nopython=True, cache=True)
def _min_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """Numba-compiled MIN calculation"""
    n = len(data)
    for i in range(timeperiod - 1):
        output[i] = np.nan
    for i in range(timeperiod - 1, n):
        min_val = data[i]
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] < min_val:
                min_val = data[j]
        output[i] = min_val


@jit(nopython=True, cache=True)
def _minindex_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """Numba-compiled MININDEX calculation"""
    n = len(data)
    for i in range(timeperiod - 1):
        output[i] = np.nan
    for i in range(timeperiod - 1, n):
        min_val = data[i]
        min_idx = 0
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] <= min_val:
                min_val = data[j]
                min_idx = i - j
        output[i] = float(min_idx)


@jit(nopython=True, cache=True)
def _minmax_numba(data: np.ndarray, timeperiod: int, min_out: np.ndarray, max_out: np.ndarray) -> None:
    """Numba-compiled MINMAX calculation"""
    n = len(data)
    for i in range(timeperiod - 1):
        min_out[i] = np.nan
        max_out[i] = np.nan
    for i in range(timeperiod - 1, n):
        min_val = data[i]
        max_val = data[i]
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] < min_val:
                min_val = data[j]
            if data[j] > max_val:
                max_val = data[j]
        min_out[i] = min_val
        max_out[i] = max_val


@jit(nopython=True, cache=True)
def _minmaxindex_numba(data: np.ndarray, timeperiod: int, min_idx_out: np.ndarray, max_idx_out: np.ndarray) -> None:
    """Numba-compiled MINMAXINDEX calculation"""
    n = len(data)
    for i in range(timeperiod - 1):
        min_idx_out[i] = np.nan
        max_idx_out[i] = np.nan
    for i in range(timeperiod - 1, n):
        min_val = data[i]
        max_val = data[i]
        min_idx = 0
        max_idx = 0
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] <= min_val:
                min_val = data[j]
                min_idx = i - j
            if data[j] >= max_val:
                max_val = data[j]
                max_idx = i - j
        min_idx_out[i] = float(min_idx)
        max_idx_out[i] = float(max_idx)


@jit(nopython=True, cache=True)
def _sum_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """Numba-compiled SUM calculation (in-place)"""
    n = len(data)

    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate first sum
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += data[i]
    output[timeperiod - 1] = sum_val

    # Calculate subsequent sums using rolling technique
    for i in range(timeperiod, n):
        sum_val = sum_val - data[i - timeperiod] + data[i]
        output[i] = sum_val
