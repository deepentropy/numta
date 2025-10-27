"""
Volume Indicators - GPU/CuPy implementations

This module contains CuPy GPU implementations for volume indicators.
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


__all__ = [
    "_ad_cupy",
    "_adosc_cupy",
    "_obv_cupy",
]


def _ad_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
    """
    CuPy-based Chaikin A/D Line calculation for GPU

    Formula:
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = Money Flow Multiplier * Volume
    AD Line = Cumulative sum of Money Flow Volume

    Parameters
    ----------
    high : np.ndarray
        High prices array
    low : np.ndarray
        Low prices array
    close : np.ndarray
        Close prices array
    volume : np.ndarray
        Volume array

    Returns
    -------
    np.ndarray
        Array of Chaikin A/D Line values
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
    volume_gpu = cp.asarray(volume, dtype=cp.float64)

    n = len(high_gpu)

    # Calculate Money Flow Multiplier
    # MFM = ((Close - Low) - (High - Close)) / (High - Low)
    #     = (2*Close - High - Low) / (High - Low)
    high_low_diff = high_gpu - low_gpu

    # Handle division by zero: when high == low, multiplier is 0
    mf_multiplier = cp.where(
        high_low_diff == 0.0,
        0.0,
        ((close_gpu - low_gpu) - (high_gpu - close_gpu)) / high_low_diff
    )

    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * volume_gpu

    # Calculate AD Line as cumulative sum
    output = cp.cumsum(mf_volume)

    # Transfer back to CPU
    return cp.asnumpy(output)


def _adosc_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                volume: np.ndarray, fastperiod: int, slowperiod: int) -> np.ndarray:
    """
    CuPy-based Chaikin A/D Oscillator calculation for GPU

    The oscillator is the difference between the fast EMA and slow EMA
    of the A/D Line.

    Parameters
    ----------
    high : np.ndarray
        High prices array
    low : np.ndarray
        Low prices array
    close : np.ndarray
        Close prices array
    volume : np.ndarray
        Volume array
    fastperiod : int
        Number of periods for the fast EMA
    slowperiod : int
        Number of periods for the slow EMA

    Returns
    -------
    np.ndarray
        Array of Chaikin A/D Oscillator values
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # First, calculate the A/D Line using GPU
    ad_line = _ad_cupy(high, low, close, volume)

    # Transfer AD line to GPU
    ad_line_gpu = cp.asarray(ad_line, dtype=cp.float64)
    n = len(ad_line_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    if n < slowperiod:
        return cp.asnumpy(output)

    # Calculate fast EMA
    fast_multiplier = 2.0 / (fastperiod + 1)
    fast_ema = cp.mean(ad_line_gpu[:fastperiod])

    # Calculate slow EMA
    slow_multiplier = 2.0 / (slowperiod + 1)
    slow_ema = cp.mean(ad_line_gpu[:slowperiod])

    # Calculate oscillator values
    # First valid value is at slowperiod - 1
    for i in range(fastperiod, slowperiod):
        fast_ema = (ad_line_gpu[i] - fast_ema) * fast_multiplier + fast_ema

    for i in range(slowperiod - 1, n):
        if i >= fastperiod:
            fast_ema = (ad_line_gpu[i] - fast_ema) * fast_multiplier + fast_ema
        slow_ema = (ad_line_gpu[i] - slow_ema) * slow_multiplier + slow_ema
        output[i] = fast_ema - slow_ema

    # Transfer back to CPU
    return cp.asnumpy(output)


def _obv_cupy(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    CuPy-based On Balance Volume (OBV) calculation for GPU

    Formula:
    If Close > Close[-1]: OBV = OBV[-1] + Volume
    If Close < Close[-1]: OBV = OBV[-1] - Volume
    If Close = Close[-1]: OBV = OBV[-1]

    Parameters
    ----------
    close : np.ndarray
        Close prices array
    volume : np.ndarray
        Volume array

    Returns
    -------
    np.ndarray
        Array of OBV values (cumulative volume)
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    close_gpu = cp.asarray(close, dtype=cp.float64)
    volume_gpu = cp.asarray(volume, dtype=cp.float64)

    n = len(close_gpu)
    output = cp.empty(n, dtype=cp.float64)

    # Start with first volume (TA-Lib convention)
    output[0] = volume_gpu[0]

    if n > 1:
        # Calculate price changes
        price_change = close_gpu[1:] - close_gpu[:-1]

        # Create volume adjustments based on price direction
        # Positive change: add volume, Negative change: subtract volume, No change: keep same
        volume_adjustment = cp.where(
            price_change > 0,
            volume_gpu[1:],
            cp.where(price_change < 0, -volume_gpu[1:], 0.0)
        )

        # Calculate OBV as cumulative sum
        output[1:] = output[0] + cp.cumsum(volume_adjustment)

    # Transfer back to CPU
    return cp.asnumpy(output)
