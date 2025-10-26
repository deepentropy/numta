"""
Statistic Functions - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for statistic functions.
"""

import numpy as np
from numba import jit


__all__ = [
    "_correl_numba",
    "_linearreg_angle_numba",
    "_linearreg_intercept_numba",
    "_linearreg_numba",
    "_linearreg_slope_numba",
]


@jit(nopython=True, cache=True)
def _correl_numba(high: np.ndarray, low: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled CORREL calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    CORREL Formula (Pearson's Correlation Coefficient):
    r = (n * Σ(xy) - Σx * Σy) / sqrt((n * Σ(x²) - (Σx)²) * (n * Σ(y²) - (Σy)²))
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate correlation for each window
    for i in range(timeperiod - 1, n):
        # Get window
        window_start = i - timeperiod + 1
        window_end = i + 1

        # Calculate sums for Pearson correlation
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0

        for j in range(window_start, window_end):
            x = high[j]
            y = low[j]
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_yy += y * y
            sum_xy += x * y

        # Calculate Pearson correlation coefficient
        # r = (n * Σxy - Σx * Σy) / sqrt((n * Σx² - (Σx)²) * (n * Σy² - (Σy)²))
        numerator = timeperiod * sum_xy - sum_x * sum_y
        denominator_x = timeperiod * sum_xx - sum_x * sum_x
        denominator_y = timeperiod * sum_yy - sum_y * sum_y

        # Handle edge cases
        if denominator_x <= 0.0 or denominator_y <= 0.0:
            output[i] = 0.0
        else:
            denominator = (denominator_x * denominator_y) ** 0.5
            if denominator == 0.0:
                output[i] = 0.0
            else:
                output[i] = numerator / denominator


@jit(nopython=True, cache=True)
def _linearreg_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled LINEARREG calculation (in-place)

    Linear Regression using least squares method:
    y = b + m*x

    For each window:
    - Calculate slope (m) and intercept (b)
    - Return value at end of window: b + m*(timeperiod-1)
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate linear regression for each window
    for i in range(timeperiod - 1, n):
        window_start = i - timeperiod + 1
        window_end = i + 1

        # Calculate sums for least squares
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0

        for j in range(window_start, window_end):
            x = float(j - window_start)  # x values: 0, 1, 2, ..., timeperiod-1
            y = close[j]
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y

        # Calculate slope (m) and intercept (b)
        # m = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
        # b = (sum_y - m*sum_x) / n
        n_period = float(timeperiod)
        denominator = n_period * sum_xx - sum_x * sum_x

        if denominator != 0.0:
            m = (n_period * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y - m * sum_x) / n_period

            # Return value at end of window: y = b + m*(timeperiod-1)
            output[i] = b + m * (timeperiod - 1)
        else:
            # All x values are the same (shouldn't happen with our x = 0,1,2,...)
            output[i] = sum_y / n_period


@jit(nopython=True, cache=True)
def _linearreg_angle_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled LINEARREG_ANGLE calculation (in-place)

    Calculates the angle of the linear regression line in degrees.
    Angle = arctan(slope) * (180 / pi)
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate linear regression angle for each window
    for i in range(timeperiod - 1, n):
        window_start = i - timeperiod + 1
        window_end = i + 1

        # Calculate sums for least squares
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0

        for j in range(window_start, window_end):
            x = float(j - window_start)  # x values: 0, 1, 2, ..., timeperiod-1
            y = close[j]
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y

        # Calculate slope (m)
        n_period = float(timeperiod)
        denominator = n_period * sum_xx - sum_x * sum_x

        if denominator != 0.0:
            m = (n_period * sum_xy - sum_x * sum_y) / denominator

            # Convert slope to angle in degrees
            # angle = arctan(slope) * (180 / pi)
            angle = np.arctan(m) * (180.0 / np.pi)
            output[i] = angle
        else:
            # No slope (flat line)
            output[i] = 0.0


@jit(nopython=True, cache=True)
def _linearreg_intercept_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled LINEARREG_INTERCEPT calculation (in-place)

    Calculates the intercept (b) of the linear regression line.
    y = b + m*x
    b = (Σy - m*Σx) / n
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate linear regression intercept for each window
    for i in range(timeperiod - 1, n):
        window_start = i - timeperiod + 1
        window_end = i + 1

        # Calculate sums for least squares
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0

        for j in range(window_start, window_end):
            x = float(j - window_start)  # x values: 0, 1, 2, ..., timeperiod-1
            y = close[j]
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y

        # Calculate slope (m) and intercept (b)
        n_period = float(timeperiod)
        denominator = n_period * sum_xx - sum_x * sum_x

        if denominator != 0.0:
            m = (n_period * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y - m * sum_x) / n_period
            output[i] = b
        else:
            # Flat line - intercept is the mean
            output[i] = sum_y / n_period


@jit(nopython=True, cache=True)
def _linearreg_slope_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled LINEARREG_SLOPE calculation (in-place)

    Calculates the slope (m) of the linear regression line.
    y = b + m*x
    m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate linear regression slope for each window
    for i in range(timeperiod - 1, n):
        window_start = i - timeperiod + 1
        window_end = i + 1

        # Calculate sums for least squares
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0

        for j in range(window_start, window_end):
            x = float(j - window_start)  # x values: 0, 1, 2, ..., timeperiod-1
            y = close[j]
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y

        # Calculate slope (m)
        n_period = float(timeperiod)
        denominator = n_period * sum_xx - sum_x * sum_x

        if denominator != 0.0:
            m = (n_period * sum_xy - sum_x * sum_y) / denominator
            output[i] = m
        else:
            # No slope (flat line)
            output[i] = 0.0
