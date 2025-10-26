"""
Momentum Indicators - Indicators that measure the rate of price change
"""

"""GPU implementations using CuPy"""

import numpy as np

# Try to import cupy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


__all__ = [
    "_cmo_cupy",
    "_dx_cupy",
    "_macd_cupy",
    "_rsi_cupy",
]


def _rsi_cupy(data: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based RSI calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    data_gpu = cp.asarray(data, dtype=cp.float64)
    n = len(data_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    lookback = timeperiod

    # Check if we have enough data
    if n <= lookback:
        return cp.asnumpy(output)

    # Calculate gains and losses
    gains = cp.zeros(n, dtype=cp.float64)
    losses = cp.zeros(n, dtype=cp.float64)

    for i in range(1, n):
        change = data_gpu[i] - data_gpu[i - 1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change

    # Calculate initial average gain and loss
    avg_gain = cp.sum(gains[1:timeperiod + 1]) / timeperiod
    avg_loss = cp.sum(losses[1:timeperiod + 1]) / timeperiod

    # Calculate first RSI value
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            output[timeperiod] = 50.0
        else:
            output[timeperiod] = 100.0
    else:
        rs = avg_gain / avg_loss
        output[timeperiod] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(timeperiod + 1, n):
        avg_gain = (avg_gain * (timeperiod - 1) + gains[i]) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + losses[i]) / timeperiod

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                output[i] = 50.0
            else:
                output[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            output[i] = 100.0 - (100.0 / (1.0 + rs))

    # Transfer back to CPU
    return cp.asnumpy(output)


def _cmo_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based CMO calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
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
    n = len(close_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    # Check if we have enough data
    if n <= timeperiod:
        return cp.asnumpy(output)

    # Calculate CMO for each window
    for i in range(timeperiod, n):
        # Calculate price changes over the window
        sum_gains = cp.float64(0.0)
        sum_losses = cp.float64(0.0)

        for j in range(i - timeperiod + 1, i + 1):
            change = close_gpu[j] - close_gpu[j - 1]
            if change > 0:
                sum_gains += change
            elif change < 0:
                sum_losses += cp.abs(change)

        # Calculate CMO
        total = sum_gains + sum_losses
        if total == 0.0:
            output[i] = 0.0
        else:
            output[i] = ((sum_gains - sum_losses) / total) * 100.0

    # Transfer back to CPU
    return cp.asnumpy(output)


def _dx_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based DX calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
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

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    lookback = timeperiod

    # Check if we have enough data
    if n <= lookback:
        return cp.asnumpy(output)

    # Calculate TR, +DM, -DM arrays
    tr = cp.empty(n, dtype=cp.float64)
    plus_dm = cp.empty(n, dtype=cp.float64)
    minus_dm = cp.empty(n, dtype=cp.float64)

    # First TR value (no previous close)
    tr[0] = high_gpu[0] - low_gpu[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        # True Range
        hl = high_gpu[i] - low_gpu[i]
        hc = cp.abs(high_gpu[i] - close_gpu[i - 1])
        lc = cp.abs(low_gpu[i] - close_gpu[i - 1])
        tr[i] = cp.maximum(cp.maximum(hl, hc), lc)

        # Directional Movement
        up_move = high_gpu[i] - high_gpu[i - 1]
        down_move = low_gpu[i - 1] - low_gpu[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Smooth TR, +DM, -DM using Wilder's smoothing
    smoothed_tr = cp.sum(tr[:timeperiod])
    smoothed_plus_dm = cp.sum(plus_dm[:timeperiod])
    smoothed_minus_dm = cp.sum(minus_dm[:timeperiod])

    # Calculate DX for timeperiod onwards
    for i in range(timeperiod, n):
        # Wilder's smoothing
        smoothed_tr = smoothed_tr - smoothed_tr / timeperiod + tr[i]
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / timeperiod + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / timeperiod + minus_dm[i]

        # Calculate directional indicators
        if smoothed_tr != 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum != 0:
            output[i] = 100.0 * cp.abs(plus_di - minus_di) / di_sum
        else:
            output[i] = 0.0

    # Transfer back to CPU
    return cp.asnumpy(output)


def _macd_cupy(close: np.ndarray, fastperiod: int, slowperiod: int, signalperiod: int) -> tuple:
    """
    CuPy-based MACD calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Import EMA from overlap for GPU calculation
    from .overlap import _ema_cupy

    # Calculate fast and slow EMAs
    fast_ema = cp.asarray(_ema_cupy(close, fastperiod), dtype=cp.float64)
    slow_ema = cp.asarray(_ema_cupy(close, slowperiod), dtype=cp.float64)

    # Calculate MACD line
    macd = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD)
    macd_cpu = cp.asnumpy(macd)
    signal = cp.asarray(_ema_cupy(macd_cpu, signalperiod), dtype=cp.float64)

    # Calculate histogram
    hist = macd - signal

    # Transfer back to CPU
    return cp.asnumpy(macd), cp.asnumpy(signal), cp.asnumpy(hist)


