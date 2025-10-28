"""
Overlap Studies - Indicators that overlay price charts
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
    "_dema_cupy",
    "_ema_cupy",
    "_kama_cupy",
    "_ma_cupy",
    "_mama_cupy",
    "_sar_cupy",
    "_sarext_cupy",
    "_sma_cupy",
    "_t3_cupy",
    "_tema_cupy",
    "_trima_cupy",
    "_wma_cupy",
]


def _sma_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based SMA calculation for GPU
    
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
    
    # Find first valid index
    valid_mask = ~cp.isnan(close_gpu)
    if not cp.any(valid_mask):
        return cp.asnumpy(output)
    
    valid_indices = cp.where(valid_mask)[0]
    start_idx = int(cp.asnumpy(valid_indices[0]))
    
    # Check if we have enough data
    if start_idx + timeperiod > n:
        return cp.asnumpy(output)
    
    # Calculate first SMA
    sum_val = cp.sum(close_gpu[start_idx:start_idx + timeperiod])
    output[start_idx + timeperiod - 1] = sum_val / timeperiod
    
    # Rolling window for subsequent values
    for i in range(start_idx + timeperiod, n):
        if cp.isnan(close_gpu[i]) or cp.isnan(close_gpu[i - timeperiod]):
            output[i] = cp.nan
        else:
            sum_val = sum_val - close_gpu[i - timeperiod] + close_gpu[i]
            output[i] = sum_val / timeperiod
    
    # Transfer back to CPU
    return cp.asnumpy(output)


def _ema_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based EMA calculation for GPU
    
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
    multiplier = 2.0 / (timeperiod + 1)
    
    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)
    
    # Find first valid index
    valid_mask = ~cp.isnan(close_gpu)
    if not cp.any(valid_mask):
        return cp.asnumpy(output)
    
    valid_indices = cp.where(valid_mask)[0]
    start_idx = int(cp.asnumpy(valid_indices[0]))
    
    # Check if we have enough data
    if start_idx + timeperiod > n:
        return cp.asnumpy(output)
    
    # Initialize first EMA as SMA
    sum_val = cp.sum(close_gpu[start_idx:start_idx + timeperiod])
    ema = sum_val / timeperiod
    output[start_idx + timeperiod - 1] = ema
    
    # Calculate EMA for remaining values
    for i in range(start_idx + timeperiod, n):
        if cp.isnan(close_gpu[i]):
            output[i] = cp.nan
        else:
            ema = (close_gpu[i] - ema) * multiplier + ema
            output[i] = ema
    
    # Transfer back to CPU
    return cp.asnumpy(output)


def _dema_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based DEMA calculation for GPU
    
    This function uses CuPy for GPU-accelerated computation.
    DEMA = 2 * EMA - EMA(EMA)
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )
    
    # Calculate first EMA on GPU
    ema1 = cp.asarray(_ema_cupy(close, timeperiod), dtype=cp.float64)
    
    # Calculate EMA of EMA
    ema1_cpu = cp.asnumpy(ema1)
    ema2 = cp.asarray(_ema_cupy(ema1_cpu, timeperiod), dtype=cp.float64)
    
    # DEMA = 2 * EMA - EMA(EMA)
    dema = 2 * ema1 - ema2
    
    # Transfer back to CPU
    return cp.asnumpy(dema)


def _kama_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based KAMA calculation for GPU
    
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
    if n < timeperiod:
        return cp.asnumpy(output)
    
    # Constants for KAMA
    fastest = 2.0 / (2.0 + 1.0)   # Fastest SC for 2-period EMA
    slowest = 2.0 / (30.0 + 1.0)  # Slowest SC for 30-period EMA
    
    # Calculate first KAMA value
    for i in range(timeperiod, n):
        # Calculate efficiency ratio (ER)
        change = cp.abs(close_gpu[i] - close_gpu[i - timeperiod])
        
        volatility = cp.float64(0.0)
        for j in range(i - timeperiod + 1, i + 1):
            volatility += cp.abs(close_gpu[j] - close_gpu[j - 1])
        
        if volatility == 0:
            er = cp.float64(0.0)
        else:
            er = change / volatility
        
        # Calculate smoothing constant (SC)
        sc = ((er * (fastest - slowest)) + slowest) ** 2
        
        # Calculate KAMA
        if i == timeperiod:
            # First KAMA = close price
            kama = close_gpu[i]
        else:
            # KAMA = KAMA_prev + SC * (Price - KAMA_prev)
            kama = kama + sc * (close_gpu[i] - kama)
        
        output[i] = kama
    
    # Transfer back to CPU
    return cp.asnumpy(output)


def _ma_cupy(close: np.ndarray, timeperiod: int, matype: int) -> np.ndarray:
    """
    CuPy-based MA calculation for GPU
    
    Routes to appropriate GPU-accelerated MA implementation.
    """
    # MA is just a router, so we call the appropriate GPU function
    if matype == 0:
        return _sma_cupy(close, timeperiod)
    elif matype == 1:
        return _ema_cupy(close, timeperiod)
    elif matype == 3:
        return _dema_cupy(close, timeperiod)
    elif matype == 6:
        return _kama_cupy(close, timeperiod)
    else:
        # For unsupported types, fall back to CPU
        from .backend import set_backend, get_backend
        old_backend = get_backend()
        set_backend('cpu')
        
        if matype == 0:
            result = SMA(close, timeperiod)
        elif matype == 1:
            result = EMA(close, timeperiod)
        elif matype == 3:
            result = DEMA(close, timeperiod)
        elif matype == 6:
            result = KAMA(close, timeperiod)
        else:
            raise NotImplementedError(f"MA type {matype} not implemented")
        
        set_backend(old_backend)
        return result


def _mama_cupy(close: np.ndarray, fastlimit: float, slowlimit: float) -> tuple:
    """
    CuPy-based MAMA calculation for GPU
    
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
    
    # Initialize arrays
    mama = cp.full(n, cp.nan, dtype=cp.float64)
    fama = cp.full(n, cp.nan, dtype=cp.float64)
    
    # Lookback period
    lookback = 32
    
    # Initialize with first valid value
    if n > lookback:
        mama[lookback] = close_gpu[lookback]
        fama[lookback] = close_gpu[lookback]
        
        # Simplified adaptive calculation
        for i in range(lookback + 1, n):
            # Calculate price change rate (simplified adaptation)
            price_change = cp.abs(close_gpu[i] - close_gpu[i-1])
            
            # Calculate average change
            start_j = max(0, i-10)
            avg_change = cp.float64(0.0)
            for j in range(start_j, i):
                avg_change += cp.abs(close_gpu[j] - close_gpu[j-1])
            avg_change = avg_change / min(10, i) if i > 0 else cp.float64(1.0)
            
            # Adaptive alpha based on price volatility
            if avg_change > 0:
                alpha = min(fastlimit, max(slowlimit, float(price_change / avg_change) * slowlimit))
            else:
                alpha = slowlimit
            
            # MAMA calculation (adaptive EMA)
            mama[i] = alpha * close_gpu[i] + (1 - alpha) * mama[i-1]
            
            # FAMA follows MAMA with half the alpha
            fama_alpha = alpha * 0.5
            fama[i] = fama_alpha * mama[i] + (1 - fama_alpha) * fama[i-1]
    
    # Transfer back to CPU
    return cp.asnumpy(mama), cp.asnumpy(fama)


def _sar_cupy(high: np.ndarray, low: np.ndarray, acceleration: float, maximum: float) -> np.ndarray:
    """
    CuPy-based SAR calculation for GPU
    
    This function uses CuPy for GPU-accelerated computation.
    Note: SAR is inherently sequential, so GPU benefit is limited.
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
    n = len(high_gpu)
    
    output = cp.empty(n, dtype=cp.float64)
    
    # Initialize
    is_long = True
    sar = low_gpu[0]
    ep = high_gpu[0]  # Extreme point
    af = acceleration
    
    output[0] = sar
    
    # Sequential calculation (cannot be easily parallelized)
    for i in range(1, n):
        # Calculate new SAR
        sar = sar + af * (ep - sar)
        
        if is_long:
            # Long position
            # SAR should not be above prior two lows
            if i >= 1:
                sar = cp.minimum(sar, low_gpu[i - 1])
            if i >= 2:
                sar = cp.minimum(sar, low_gpu[i - 2])
            
            # Check for reversal
            if low_gpu[i] < sar:
                # Reverse to short
                is_long = False
                sar = ep
                ep = low_gpu[i]
                af = acceleration
            else:
                # Continue long
                if high_gpu[i] > ep:
                    ep = high_gpu[i]
                    af = min(af + acceleration, maximum)
        else:
            # Short position
            # SAR should not be below prior two highs
            if i >= 1:
                sar = cp.maximum(sar, high_gpu[i - 1])
            if i >= 2:
                sar = cp.maximum(sar, high_gpu[i - 2])
            
            # Check for reversal
            if high_gpu[i] > sar:
                # Reverse to long
                is_long = True
                sar = ep
                ep = high_gpu[i]
                af = acceleration
            else:
                # Continue short
                if low_gpu[i] < ep:
                    ep = low_gpu[i]
                    af = min(af + acceleration, maximum)
        
        output[i] = sar
    
    # Transfer back to CPU
    return cp.asnumpy(output)


def _sarext_cupy(high: np.ndarray, low: np.ndarray,
                 startvalue: float, offsetonreverse: float,
                 accelerationinit_long: float, accelerationlong: float, accelerationmax_long: float,
                 accelerationinit_short: float, accelerationshort: float, accelerationmax_short: float) -> np.ndarray:
    """
    CuPy-based SAREXT calculation for GPU

    Extended Parabolic SAR with separate parameters for long and short.
    Note: Sequential algorithm limits GPU parallelization.
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
    n = len(high_gpu)

    output = cp.empty(n, dtype=cp.float64)

    # Initialize
    is_long = True
    sar = startvalue if startvalue != 0 else low_gpu[0]
    ep = high_gpu[0]
    af = accelerationinit_long

    output[0] = sar

    # Sequential calculation
    for i in range(1, n):
        # Calculate new SAR
        sar = sar + af * (ep - sar)

        if is_long:
            # Long position
            if i >= 1:
                sar = cp.minimum(sar, low_gpu[i - 1])
            if i >= 2:
                sar = cp.minimum(sar, low_gpu[i - 2])

            # Check for reversal
            if low_gpu[i] < sar:
                is_long = False
                sar = ep + offsetonreverse
                ep = low_gpu[i]
                af = accelerationinit_short
            else:
                if high_gpu[i] > ep:
                    ep = high_gpu[i]
                    af = min(af + accelerationlong, accelerationmax_long)
        else:
            # Short position
            if i >= 1:
                sar = cp.maximum(sar, high_gpu[i - 1])
            if i >= 2:
                sar = cp.maximum(sar, high_gpu[i - 2])

            # Check for reversal
            if high_gpu[i] > sar:
                is_long = True
                sar = ep - offsetonreverse
                ep = high_gpu[i]
                af = accelerationinit_long
            else:
                if low_gpu[i] < ep:
                    ep = low_gpu[i]
                    af = min(af + accelerationshort, accelerationmax_short)

        output[i] = sar

    # Transfer back to CPU
    return cp.asnumpy(output)


def _tema_cupy(data: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based TEMA calculation for GPU

    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Calculate EMAs using GPU
    ema1 = cp.asarray(_ema_cupy(data, timeperiod), dtype=cp.float64)
    ema1_cpu = cp.asnumpy(ema1)

    ema2 = cp.asarray(_ema_cupy(ema1_cpu, timeperiod), dtype=cp.float64)
    ema2_cpu = cp.asnumpy(ema2)

    ema3 = cp.asarray(_ema_cupy(ema2_cpu, timeperiod), dtype=cp.float64)

    # TEMA = 3*EMA1 - 3*EMA2 + EMA3
    output = 3.0 * ema1 - 3.0 * ema2 + ema3

    # Transfer back to CPU
    return cp.asnumpy(output)


def _t3_cupy(data: np.ndarray, timeperiod: int, vfactor: float) -> np.ndarray:
    """
    CuPy-based T3 calculation for GPU

    T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Calculate coefficients
    c1 = -vfactor * vfactor * vfactor
    c2 = 3.0 * vfactor * vfactor + 3.0 * vfactor * vfactor * vfactor
    c3 = -6.0 * vfactor * vfactor - 3.0 * vfactor - 3.0 * vfactor * vfactor * vfactor
    c4 = 1.0 + 3.0 * vfactor + vfactor * vfactor * vfactor + 3.0 * vfactor * vfactor

    # Calculate 6 EMAs using GPU
    ema1 = _ema_cupy(data, timeperiod)
    ema2 = _ema_cupy(ema1, timeperiod)
    ema3 = _ema_cupy(ema2, timeperiod)
    ema4 = _ema_cupy(ema3, timeperiod)
    ema5 = _ema_cupy(ema4, timeperiod)
    ema6 = _ema_cupy(ema5, timeperiod)

    # Convert to GPU for final calculation
    ema3_gpu = cp.asarray(ema3, dtype=cp.float64)
    ema4_gpu = cp.asarray(ema4, dtype=cp.float64)
    ema5_gpu = cp.asarray(ema5, dtype=cp.float64)
    ema6_gpu = cp.asarray(ema6, dtype=cp.float64)

    # T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
    output = c1 * ema6_gpu + c2 * ema5_gpu + c3 * ema4_gpu + c4 * ema3_gpu

    # Transfer back to CPU
    return cp.asnumpy(output)


def _trima_cupy(data: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based TRIMA calculation for GPU

    TRIMA = SMA(SMA(data))
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Calculate periods for double SMA
    if timeperiod % 2 == 1:
        # Odd period
        n = (timeperiod + 1) // 2
        sma1 = _sma_cupy(data, n)
        sma2 = _sma_cupy(sma1, n)
    else:
        # Even period
        n1 = timeperiod // 2
        n2 = n1 + 1
        sma1 = _sma_cupy(data, n1)
        sma2 = _sma_cupy(sma1, n2)

    return sma2


def _wma_cupy(data: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based WMA calculation for GPU

    Uses incremental O(n) algorithm on GPU.
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

    # Calculate sum of weights
    weight_sum = (timeperiod * (timeperiod + 1)) / 2.0

    output = cp.full(n, cp.nan, dtype=cp.float64)

    # Calculate first WMA value
    weighted_sum = cp.float64(0.0)
    simple_sum = cp.float64(0.0)
    for j in range(timeperiod):
        weight = j + 1
        value = data_gpu[j]
        weighted_sum += value * weight
        simple_sum += value

    output[timeperiod - 1] = weighted_sum / weight_sum

    # Use incremental calculation for remaining values
    for i in range(timeperiod, n):
        old_value = data_gpu[i - timeperiod]
        new_value = data_gpu[i]

        # Remove contribution of oldest value and subtract simple_sum
        weighted_sum = weighted_sum - simple_sum

        # Update simple sum
        simple_sum = simple_sum - old_value + new_value

        # Add new value with full weight
        weighted_sum = weighted_sum + new_value * timeperiod

        output[i] = weighted_sum / weight_sum

    # Transfer back to CPU
    return cp.asnumpy(output)


