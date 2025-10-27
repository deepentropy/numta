"""
Optimized implementations of SMA with different backends
"""

import numpy as np
from typing import Union, Literal

# Optional dependencies
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def SMA_cumsum(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """
    Optimized SMA using cumulative sum - O(n) time complexity

    This is faster than convolution for most cases because:
    - Cumulative sum is O(n)
    - No convolution overhead
    - Better cache locality

    Parameters
    ----------
    close : np.ndarray
        Close prices array
    timeperiod : int, optional
        Number of periods (default: 30)

    Returns
    -------
    np.ndarray
        SMA values with NaN for lookback period
    """
    close = np.asarray(close, dtype=np.float64)

    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    if len(close) == 0:
        return np.array([])

    output = np.full(len(close), np.nan, dtype=np.float64)

    if len(close) < timeperiod:
        return output

    # Use cumulative sum for O(n) performance
    cumsum = np.cumsum(close)

    # First SMA value
    output[timeperiod - 1] = cumsum[timeperiod - 1] / timeperiod

    # Remaining values: (cumsum[i] - cumsum[i-timeperiod]) / timeperiod
    output[timeperiod:] = (cumsum[timeperiod:] - cumsum[:-timeperiod]) / timeperiod

    return output


if HAS_NUMBA:
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _sma_numba_kernel(close: np.ndarray, timeperiod: int) -> np.ndarray:
        """
        Numba-optimized SMA kernel using cumulative sum

        JIT compiled with:
        - nopython=True: Full compilation, no Python overhead
        - cache=True: Cache compiled code for faster subsequent runs
        - fastmath=True: Use faster but less precise math operations
        """
        n = len(close)
        output = np.full(n, np.nan, dtype=np.float64)

        if n < timeperiod:
            return output

        # Calculate first SMA
        window_sum = 0.0
        for i in range(timeperiod):
            window_sum += close[i]
        output[timeperiod - 1] = window_sum / timeperiod

        # Rolling window: subtract old, add new
        for i in range(timeperiod, n):
            window_sum = window_sum - close[i - timeperiod] + close[i]
            output[i] = window_sum / timeperiod

        return output


    def SMA_numba(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
        """
        Numba-optimized SMA implementation

        Uses JIT compilation for maximum performance. First call will be slower
        due to compilation, but subsequent calls are very fast.

        Parameters
        ----------
        close : array-like
            Close prices
        timeperiod : int, optional
            Number of periods (default: 30)

        Returns
        -------
        np.ndarray
            SMA values with NaN for lookback period
        """
        close = np.asarray(close, dtype=np.float64)

        if timeperiod < 1:
            raise ValueError("timeperiod must be >= 1")

        if len(close) == 0:
            return np.array([])

        return _sma_numba_kernel(close, timeperiod)


def SMA_auto(
    close: Union[np.ndarray, list],
    timeperiod: int = 30,
    backend: Literal["auto", "numpy", "cumsum", "numba"] = "auto"
) -> np.ndarray:
    """
    Automatically choose the best SMA implementation based on:
    - Available libraries (Numba)
    - Data size
    - User preference

    Parameters
    ----------
    close : array-like
        Close prices
    timeperiod : int, optional
        Number of periods (default: 30)
    backend : str, optional
        Backend to use:
        - "auto": Automatically choose best backend
        - "numpy": Use np.convolve (original implementation)
        - "cumsum": Use cumulative sum (faster for most cases)
        - "numba": Use Numba JIT (if available)

    Returns
    -------
    np.ndarray
        SMA values with NaN for lookback period

    Examples
    --------
    >>> import numpy as np
    >>> close = np.random.uniform(100, 200, 10000)
    >>>
    >>> # Auto-select best backend
    >>> sma = SMA_auto(close, timeperiod=30)
    >>>
    >>> # Force specific backend
    >>> sma_numba = SMA_auto(close, timeperiod=30, backend="numba")
    """
    from .api.overlap import SMA  # Original implementation

    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)

    # Handle explicit backend selection
    if backend == "numpy":
        return SMA(close_arr, timeperiod)
    elif backend == "cumsum":
        return SMA_cumsum(close_arr, timeperiod)
    elif backend == "numba":
        if not HAS_NUMBA:
            raise ImportError("Numba is not installed. Install with: pip install numba")
        return SMA_numba(close_arr, timeperiod)

    # Auto-select best backend
    if backend == "auto":
        # For medium to large datasets, prefer Numba if available
        if HAS_NUMBA and n > 1000:
            return SMA_numba(close_arr, timeperiod)

        # For all other cases, use cumsum (faster than numpy convolution)
        return SMA_cumsum(close_arr, timeperiod)

    raise ValueError(f"Unknown backend: {backend}")


# Convenience function for checking available backends
def get_available_backends():
    """
    Get list of available performance backends

    Returns
    -------
    dict
        Dictionary with backend availability and descriptions
    """
    return {
        "numpy": {
            "available": True,
            "description": "NumPy convolve (original implementation)"
        },
        "cumsum": {
            "available": True,
            "description": "Cumulative sum (O(n), faster for most cases)"
        },
        "numba": {
            "available": HAS_NUMBA,
            "description": "Numba JIT compilation (fastest for CPU)"
        }
    }
