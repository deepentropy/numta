"""
Volatility Indicators - GPU/CuPy implementations

This module contains CuPy GPU implementations for volatility indicators.
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


__all__ = [
    "_trange_cupy",
    "_natr_cupy",
]


def _trange_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    CuPy-based True Range calculation for GPU

    True Range is the maximum of:
    - Current high - current low
    - Absolute value of current high - previous close
    - Absolute value of current low - previous close

    Parameters
    ----------
    high : np.ndarray
        High prices array
    low : np.ndarray
        Low prices array
    close : np.ndarray
        Close prices array

    Returns
    -------
    np.ndarray
        Array of true range values
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    high_gpu = cp.asarray(high, dtype=cp.float64)
    low_gpu = cp.asarray(low, dtype=cp.float64)
    close_gpu = cp.asarray(close, dtype=cp.float64)

    n = len(high_gpu)
    output = cp.empty(n, dtype=cp.float64)

    # First bar: just high - low
    output[0] = high_gpu[0] - low_gpu[0]

    # Subsequent bars: max of three ranges
    # Vectorized operations for better GPU performance
    if n > 1:
        hl = high_gpu[1:] - low_gpu[1:]
        hc = cp.abs(high_gpu[1:] - close_gpu[:-1])
        lc = cp.abs(low_gpu[1:] - close_gpu[:-1])

        # Stack and take maximum across axis 0
        stacked = cp.stack([hl, hc, lc], axis=0)
        output[1:] = cp.max(stacked, axis=0)

    # Transfer back to CPU
    return cp.asnumpy(output)


def _natr_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               timeperiod: int) -> np.ndarray:
    """
    CuPy-based NATR (Normalized Average True Range) calculation for GPU

    NATR = (ATR / Close) * 100

    Parameters
    ----------
    high : np.ndarray
        High prices array
    low : np.ndarray
        Low prices array
    close : np.ndarray
        Close prices array
    timeperiod : int
        Period for ATR calculation

    Returns
    -------
    np.ndarray
        Array of NATR values (percentage)
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # First, calculate True Range on GPU
    trange_values = _trange_cupy(high, low, close)

    # Transfer to GPU for ATR calculation
    trange_gpu = cp.asarray(trange_values, dtype=cp.float64)
    close_gpu = cp.asarray(close, dtype=cp.float64)

    n = len(trange_gpu)
    output = cp.full(n, cp.nan, dtype=cp.float64)

    if n < timeperiod:
        return cp.asnumpy(output)

    # Calculate initial ATR as simple average
    atr = cp.mean(trange_gpu[:timeperiod])
    output[timeperiod - 1] = (atr / close_gpu[timeperiod - 1]) * 100.0 if close_gpu[timeperiod - 1] != 0.0 else cp.nan

    # Calculate subsequent ATR values using Wilder's smoothing
    # ATR = ((Prior ATR * (timeperiod - 1)) + Current TR) / timeperiod
    for i in range(timeperiod, n):
        atr = ((atr * (timeperiod - 1)) + trange_gpu[i]) / timeperiod
        if close_gpu[i] != 0.0:
            output[i] = (atr / close_gpu[i]) * 100.0
        else:
            output[i] = cp.nan

    # Transfer back to CPU
    return cp.asnumpy(output)
