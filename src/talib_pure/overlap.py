"""
Overlap Studies - Indicators that overlay price charts
"""

import numpy as np
from typing import Union
from numba import jit


@jit(nopython=True, cache=True)
def _sma_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled SMA calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.
    Handles NaN values in input data.
    """
    n = len(close)

    # Find first valid (non-NaN) index
    start_idx = 0
    for i in range(n):
        if not np.isnan(close[i]):
            start_idx = i
            break

    # Fill initial values with NaN
    for i in range(start_idx + timeperiod - 1):
        output[i] = np.nan

    # Check if we have enough valid data
    if start_idx + timeperiod > n:
        for i in range(n):
            output[i] = np.nan
        return

    # Calculate first SMA value from first timeperiod valid values
    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += close[i]
    output[start_idx + timeperiod - 1] = sum_val / timeperiod

    # Use rolling window for subsequent values
    for i in range(start_idx + timeperiod, n):
        if np.isnan(close[i]) or np.isnan(close[i - timeperiod]):
            output[i] = np.nan
        else:
            sum_val = sum_val - close[i - timeperiod] + close[i]
            output[i] = sum_val / timeperiod


# GPU (CuPy) implementation
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



def SMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Simple Moving Average (SMA)

    The Simple Moving Average (SMA) is calculated by adding the closing prices
    of the last N periods and dividing by N. This indicator is used to smooth
    price data and identify trends.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 30)

    Returns
    -------
    np.ndarray
        Array of SMA values with NaN for the lookback period

    Notes
    -----
    - The first (timeperiod - 1) values will be NaN
    - Compatible with TA-Lib SMA signature
    - Uses Numba JIT compilation for maximum performance

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import SMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> sma = SMA(close, timeperiod=3)
    >>> print(sma)
    [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]
    """
    # Validate inputs (TA-Lib requires timeperiod >= 2)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2 (TA-Lib requirement)")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    
    backend = get_backend()
    
    if backend == "gpu":
        # Use GPU implementation
        return _sma_cupy(close, timeperiod)
    else:
        # Use CPU implementation (default)
    # The Numba function handles NaN values and insufficient data
        output = np.empty(n, dtype=np.float64)
        _sma_numba(close, timeperiod, output)

        return output


@jit(nopython=True, cache=True)
def _ema_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled EMA calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output array in-place.

    Formula:
    EMA = (Close - EMA_prev) * multiplier + EMA_prev
    where multiplier = 2 / (timeperiod + 1)

    The first EMA value is initialized as SMA of first timeperiod values.
    """
    n = len(close)
    multiplier = 2.0 / (timeperiod + 1)

    # Find first valid (non-NaN) index
    start_idx = 0
    for i in range(n):
        if not np.isnan(close[i]):
            start_idx = i
            break

    # Fill initial values with NaN
    for i in range(start_idx + timeperiod - 1):
        output[i] = np.nan

    # Check if we have enough valid data
    if start_idx + timeperiod > n:
        for i in range(n):
            output[i] = np.nan
        return

    # Initialize first EMA value as SMA of first timeperiod valid values
    sum_val = 0.0
    for i in range(start_idx, start_idx + timeperiod):
        sum_val += close[i]
    ema = sum_val / timeperiod
    output[start_idx + timeperiod - 1] = ema

    # Calculate EMA for remaining values
    for i in range(start_idx + timeperiod, n):
        if np.isnan(close[i]):
            output[i] = np.nan
        else:
            ema = (close[i] - ema) * multiplier + ema
            output[i] = ema


# GPU (CuPy) implementation  
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



def EMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Exponential Moving Average (EMA)

    The Exponential Moving Average (EMA) is a type of moving average that places
    a greater weight and significance on the most recent data points. The EMA
    responds more quickly to recent price changes than a simple moving average.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 30)

    Returns
    -------
    np.ndarray
        Array of EMA values with NaN for the lookback period

    Notes
    -----
    - The first (timeperiod - 1) values will be NaN
    - Compatible with TA-Lib EMA signature
    - Uses Numba JIT compilation for maximum performance
    - The first EMA value is initialized as the SMA of the first timeperiod values
    - Smoothing factor: 2 / (timeperiod + 1)

    Formula
    -------
    Multiplier = 2 / (timeperiod + 1)
    EMA[0] = SMA(close[0:timeperiod])
    EMA[i] = (Close[i] - EMA[i-1]) * Multiplier + EMA[i-1]

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import EMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ema = EMA(close, timeperiod=3)
    >>> print(ema)
    [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]
    """
    # Validate inputs (TA-Lib requires timeperiod >= 2)
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2 (TA-Lib requirement)")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    
    backend = get_backend()
    
    if backend == "gpu":
        # Use GPU implementation
        return _ema_cupy(close, timeperiod)
    else:
        # Use CPU implementation (default)
        output = np.empty(n, dtype=np.float64)
        _ema_numba(close, timeperiod, output)
        return output


@jit(nopython=True, cache=True)
def _bbands_numba(close: np.ndarray, timeperiod: int, nbdevup: float, nbdevdn: float,
                  upperband: np.ndarray, middleband: np.ndarray, lowerband: np.ndarray) -> None:
    """
    Numba-compiled Bollinger Bands calculation (in-place)

    This function is JIT-compiled for maximum performance.
    It modifies the output arrays in-place.

    Formula:
    Middle Band = SMA(close, timeperiod)
    Upper Band = Middle Band + (nbdevup * StdDev)
    Lower Band = Middle Band - (nbdevdn * StdDev)
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        upperband[i] = np.nan
        middleband[i] = np.nan
        lowerband[i] = np.nan

    # Calculate first SMA value
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    sma = sum_val / timeperiod

    # Calculate first standard deviation
    variance = 0.0
    for i in range(timeperiod):
        diff = close[i] - sma
        variance += diff * diff
    stddev = np.sqrt(variance / timeperiod)

    # Set first values
    middleband[timeperiod - 1] = sma
    upperband[timeperiod - 1] = sma + nbdevup * stddev
    lowerband[timeperiod - 1] = sma - nbdevdn * stddev

    # Calculate remaining values using rolling window
    for i in range(timeperiod, n):
        # Update SMA (rolling window)
        sum_val = sum_val - close[i - timeperiod] + close[i]
        sma = sum_val / timeperiod

        # Calculate standard deviation for current window
        variance = 0.0
        for j in range(i - timeperiod + 1, i + 1):
            diff = close[j] - sma
            variance += diff * diff
        stddev = np.sqrt(variance / timeperiod)

        # Calculate bands
        middleband[i] = sma
        upperband[i] = sma + nbdevup * stddev
        lowerband[i] = sma - nbdevdn * stddev


def BBANDS(close: Union[np.ndarray, list],
           timeperiod: int = 5,
           nbdevup: float = 2.0,
           nbdevdn: float = 2.0,
           matype: int = 0) -> tuple:
    """
    Bollinger Bands (BBANDS)

    Bollinger Bands are a volatility indicator that consists of three lines:
    a middle band (SMA), an upper band, and a lower band. The upper and lower
    bands are typically set 2 standard deviations away from the middle band.

    Developed by John Bollinger, these bands expand and contract based on
    market volatility. They are widely used for identifying overbought/oversold
    conditions and potential breakouts.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 5)
    nbdevup : float, optional
        Number of standard deviations for upper band (default: 2.0)
    nbdevdn : float, optional
        Number of standard deviations for lower band (default: 2.0)
    matype : int, optional
        Moving average type: 0 = SMA (default). Note: Only SMA is currently supported.

    Returns
    -------
    tuple of np.ndarray
        (upperband, middleband, lowerband) - Three arrays with the band values

    Notes
    -----
    - Compatible with TA-Lib BBANDS signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Currently only supports SMA (matype=0)
    - Bands widen during high volatility and narrow during low volatility

    Formula
    -------
    Middle Band = SMA(close, timeperiod)
    Upper Band = Middle Band + (nbdevup × StdDev)
    Lower Band = Middle Band - (nbdevdn × StdDev)

    Where StdDev is the population standard deviation over the timeperiod.

    Lookback period: timeperiod - 1
    (For timeperiod=20, lookback=19)

    Interpretation:
    - Price touching upper band: Potential overbought condition
    - Price touching lower band: Potential oversold condition
    - Band squeeze (narrow bands): Low volatility, potential breakout coming
    - Band expansion (wide bands): High volatility, trend in progress
    - Price breaking above upper band: Strong uptrend
    - Price breaking below lower band: Strong downtrend
    - Middle band acts as dynamic support/resistance

    Common Trading Strategies:
    - Bollinger Bounce: Buy at lower band, sell at upper band (ranging markets)
    - Bollinger Squeeze: Trade breakouts after period of low volatility
    - Walking the Bands: In strong trends, price "walks" along one band
    - %b Indicator: (Price - Lower Band) / (Upper Band - Lower Band)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import BBANDS
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> upper, middle, lower = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2)
    >>> print(upper)
    >>> print(middle)
    >>> print(lower)
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    if nbdevup < 0:
        raise ValueError("nbdevup must be >= 0")
    if nbdevdn < 0:
        raise ValueError("nbdevdn must be >= 0")
    if matype != 0:
        raise ValueError("Only matype=0 (SMA) is currently supported")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # Not enough data points - return all NaN
    if n < timeperiod:
        nans = np.full(n, np.nan, dtype=np.float64)
        return nans, nans, nans

    # Pre-allocate output arrays and run Numba-optimized calculation
    upperband = np.empty(n, dtype=np.float64)
    middleband = np.empty(n, dtype=np.float64)
    lowerband = np.empty(n, dtype=np.float64)

    _bbands_numba(close, timeperiod, nbdevup, nbdevdn, upperband, middleband, lowerband)

    return upperband, middleband, lowerband


@jit(nopython=True, cache=True)
def _dema_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled DEMA calculation (in-place)
    
    Formula:
    DEMA = 2 * EMA - EMA(EMA)
    """
    n = len(close)
    multiplier = 2.0 / (timeperiod + 1)
    
    # Fill lookback period with NaN (2 * timeperiod - 2)
    lookback = 2 * timeperiod - 2
    for i in range(lookback):
        output[i] = np.nan
    
    # Calculate first EMA
    ema1 = np.empty(n, dtype=np.float64)
    for i in range(timeperiod - 1):
        ema1[i] = np.nan
    
    # Initialize first EMA value as SMA
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    ema1[timeperiod - 1] = sum_val / timeperiod
    
    # Calculate remaining EMA1 values
    for i in range(timeperiod, n):
        ema1[i] = (close[i] - ema1[i-1]) * multiplier + ema1[i-1]
    
    # Calculate EMA of EMA (EMA2)
    sum_val = 0.0
    for i in range(timeperiod - 1, 2 * timeperiod - 1):
        sum_val += ema1[i]
    ema2 = sum_val / timeperiod
    
    # Calculate DEMA values
    output[2 * timeperiod - 2] = 2.0 * ema1[2 * timeperiod - 2] - ema2
    
    for i in range(2 * timeperiod - 1, n):
        ema2 = (ema1[i] - ema2) * multiplier + ema2
        output[i] = 2.0 * ema1[i] - ema2


# GPU (CuPy) implementation
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



def DEMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Double Exponential Moving Average (DEMA)
    
    The Double Exponential Moving Average (DEMA) is a smoothing indicator that
    reduces lag compared to traditional EMAs. Despite its name, it's not simply
    two EMAs but rather a composite of single and double EMAs designed to follow
    prices more closely.
    
    Developed by Patrick Mulloy and published in February 1994, DEMA aims to
    provide a moving average with less lag than traditional moving averages.
    
    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the calculation (default: 30)
    
    Returns
    -------
    np.ndarray
        Array of DEMA values with NaN for the lookback period
    
    Notes
    -----
    - Compatible with TA-Lib DEMA signature
    - Uses Numba JIT compilation for maximum performance
    - The first (2 * timeperiod - 2) values will be NaN
    - Lookback period: 2 * timeperiod - 2
    - More responsive than standard EMA
    
    Formula
    -------
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    DEMA = 2 * EMA1 - EMA2
    
    The formula removes lag by subtracting the "EMA of EMA" from double the
    original EMA. This creates a faster-reacting moving average.
    
    Lookback period: 2 * timeperiod - 2
    (For timeperiod=30, lookback=58)
    
    Interpretation:
    - DEMA follows prices more closely than SMA or EMA
    - Crossovers generate earlier signals than traditional MAs
    - Steeper slope indicates stronger trend
    - Use for trend following and dynamic support/resistance
    - Less prone to whipsaws in trending markets
    
    Advantages:
    - Reduced lag compared to SMA and EMA
    - Smoother than short-period EMAs
    - Earlier trend change signals
    - Better tracking of price movements
    
    Common Uses:
    - Trend identification and confirmation
    - Dynamic support/resistance levels
    - Crossover trading systems
    - Identifying entry/exit points
    - Filtering market noise
    
    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import DEMA
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> dema = DEMA(close, timeperiod=5)
    >>> print(dema)
    
    See Also
    --------
    EMA : Exponential Moving Average
    TEMA : Triple Exponential Moving Average
    SMA : Simple Moving Average
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    
    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)
    
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Not enough data points - return all NaN
    lookback = 2 * timeperiod - 2
    if n <= lookback:
        return np.full(n, np.nan, dtype=np.float64)
    
    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _dema_numba(close, timeperiod, output)
    
    return output


@jit(nopython=True, cache=True)
def _kama_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled KAMA calculation (in-place) - Optimized

    This implementation:
    1. Uses incremental volatility calculation (O(n) instead of O(n*timeperiod))
    2. Matches TA-Lib output exactly
    3. Uses unstable period warmup like TA-Lib

    Formula:
    1. ER = Change / Volatility
       where Change = abs(close[i] - close[i-timeperiod])
             Volatility = sum of abs(close[j] - close[j-1]) over timeperiod
    2. Fastest = 2/(2+1) = 0.6667
       Slowest = 2/(30+1) = 0.0645
    3. SC = (ER * (fastest - slowest) + slowest)^2
    4. KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    """
    n = len(close)
    fastest = 2.0 / (2.0 + 1.0)  # 0.6667
    slowest = 2.0 / (30.0 + 1.0)  # 0.0645
    const_diff = fastest - slowest

    # Unstable period: TA-Lib uses max(timeperiod, 30) + unstable_period
    # For simplicity and to match TA-Lib, use 30 as minimum unstable period
    # Lookback period equals timeperiod
    lookback_period = timeperiod

    # Fill lookback period with NaN
    for i in range(lookback_period):
        output[i] = np.nan

    if n <= lookback_period:
        return

    # Initialize starting from unstable_period
    today = lookback_period
    trailing_idx = 0

    # Calculate initial volatility sum
    per_sum = 0.0
    for i in range(lookback_period):
        per_sum += abs(close[i + 1] - close[i])

    # Initialize KAMA at first output position
    kama = close[today]

    # Calculate and output first KAMA value
    if per_sum != 0.0:
        er = abs(close[today] - close[trailing_idx]) / per_sum
    else:
        er = 0.0

    sc = er * const_diff + slowest
    sc = sc * sc
    kama = kama + sc * (close[today] - kama)
    output[today] = kama

    # Move to next position
    today += 1
    trailing_idx += 1

    # Calculate remaining KAMA values using incremental volatility
    while today < n:
        # Incrementally update volatility sum
        per_sum -= abs(close[trailing_idx] - close[trailing_idx - 1])
        per_sum += abs(close[today] - close[today - 1])

        # Calculate ER
        if per_sum != 0.0:
            er = abs(close[today] - close[trailing_idx]) / per_sum
        else:
            er = 0.0

        # Calculate SC
        sc = er * const_diff + slowest
        sc = sc * sc

        # Update KAMA
        kama = kama + sc * (close[today] - kama)
        output[today] = kama

        today += 1
        trailing_idx += 1


# GPU (CuPy) implementation
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



def KAMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average (KAMA)

    The Kaufman Adaptive Moving Average (KAMA) is an intelligent moving average
    developed by Perry Kaufman. Unlike traditional moving averages with fixed
    smoothing, KAMA adapts its smoothing constant based on market efficiency.

    KAMA uses an Efficiency Ratio (ER) to measure the directional movement
    relative to volatility. When prices move efficiently in one direction, KAMA
    responds quickly (like a fast EMA). During choppy, sideways markets, KAMA
    slows down (like a slow EMA), reducing noise and false signals.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the efficiency ratio calculation (default: 30)

    Returns
    -------
    np.ndarray
        Array of KAMA values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib KAMA signature
    - Uses Numba JIT compilation for maximum performance
    - The first timeperiod values will be NaN
    - Lookback period: timeperiod
    - Adapts to market conditions automatically

    Formula
    -------
    1. Efficiency Ratio (ER):
       ER = Change / Volatility
       where:
       - Change = |Close[i] - Close[i - timeperiod]|
       - Volatility = Sum of |Close[j] - Close[j-1]| over timeperiod

    2. Smoothing Constant (SC):
       SC = [ER × (Fastest - Slowest) + Slowest]²
       where:
       - Fastest = 2/(2+1) = 0.6667 (2-period EMA constant)
       - Slowest = 2/(30+1) = 0.0645 (30-period EMA constant)

    3. KAMA:
       KAMA[0] = Close[timeperiod]
       KAMA[i] = KAMA[i-1] + SC × (Close[i] - KAMA[i-1])

    Lookback period: timeperiod
    (For timeperiod=30, lookback=30)

    Interpretation:
    - ER near 1.0: Strong directional move (low noise) → KAMA reacts quickly
    - ER near 0.0: Choppy market (high noise) → KAMA smooths heavily
    - KAMA above price: Potential resistance / bearish signal
    - KAMA below price: Potential support / bullish signal
    - KAMA slope indicates trend strength and direction
    - Price crossing KAMA: Potential trend change signal

    Advantages:
    - Self-adjusting to market conditions
    - Reduces whipsaws in ranging markets
    - Responsive during trending markets
    - Better signal-to-noise ratio than fixed MAs
    - Fewer false signals than traditional MAs

    Common Uses:
    - Trend identification in various market conditions
    - Dynamic support/resistance levels
    - Crossover trading systems with price or other MAs
    - Trailing stop placement
    - Market regime detection (trending vs ranging)

    Trading Signals:
    - Buy: Price crosses above KAMA with positive slope
    - Sell: Price crosses below KAMA with negative slope
    - Strong trend: KAMA shows smooth, consistent slope
    - Consolidation: KAMA flattens, indicating market indecision

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import KAMA
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> kama = KAMA(close, timeperiod=5)
    >>> print(kama)

    See Also
    --------
    EMA : Exponential Moving Average
    DEMA : Double Exponential Moving Average
    SMA : Simple Moving Average

    References
    ----------
    Kaufman, P. J. (1995). "Smarter Trading: Improving Performance in Changing Markets"
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    
    backend = get_backend()
    
    if backend == "gpu":
        # Use GPU implementation
        return _kama_cupy(close, timeperiod)
    else:
        # Use CPU implementation (default)
        output = np.empty(n, dtype=np.float64)
        _kama_numba(close, timeperiod, output)
        return output


# GPU (CuPy) implementation
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



def MA(close: Union[np.ndarray, list], timeperiod: int = 30, matype: int = 0) -> np.ndarray:
    """
    Moving Average (MA)

    Generic moving average function that can calculate different types of
    moving averages based on the matype parameter. This provides a unified
    interface for accessing various moving average implementations.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the moving average (default: 30)
    matype : int, optional
        Type of moving average (default: 0)
        - 0: SMA (Simple Moving Average)
        - 1: EMA (Exponential Moving Average)
        - 2: WMA (Weighted Moving Average) [Not yet implemented]
        - 3: DEMA (Double Exponential Moving Average)
        - 4: TEMA (Triple Exponential Moving Average) [Not yet implemented]
        - 5: TRIMA (Triangular Moving Average) [Not yet implemented]
        - 6: KAMA (Kaufman Adaptive Moving Average)
        - 7: MAMA (Mesa Adaptive Moving Average) [Not yet implemented]
        - 8: T3 (Triple Exponential T3) [Not yet implemented]

    Returns
    -------
    np.ndarray
        Array of moving average values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib MA signature
    - Uses Numba JIT compilation for maximum performance (where implemented)
    - Lookback period varies by MA type
    - Currently implements: SMA, EMA, DEMA, KAMA
    - Other MA types will be added in future releases

    Supported Moving Average Types
    -------------------------------
    **SMA (matype=0)**: Simple Moving Average
    - Arithmetic mean of prices over timeperiod
    - Equal weight to all data points
    - Lookback: timeperiod - 1

    **EMA (matype=1)**: Exponential Moving Average
    - Exponentially weighted moving average
    - More weight to recent prices
    - Responds faster to price changes than SMA
    - Lookback: timeperiod - 1

    **DEMA (matype=3)**: Double Exponential Moving Average
    - Reduced lag compared to EMA
    - Formula: 2*EMA - EMA(EMA)
    - Lookback: 2*timeperiod - 2

    **KAMA (matype=6)**: Kaufman Adaptive Moving Average
    - Adapts to market conditions
    - Fast in trending markets, slow in ranging markets
    - Uses Efficiency Ratio for adaptation
    - Lookback: timeperiod

    Interpretation:
    - MA smooths price action to reveal underlying trend
    - Price above MA: Potential uptrend
    - Price below MA: Potential downtrend
    - MA slope indicates trend direction
    - Different MA types offer different lag vs smoothness tradeoffs

    Common Uses:
    - Trend identification and confirmation
    - Dynamic support/resistance levels
    - Crossover trading systems
    - Filtering market noise
    - Identifying entry/exit points

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MA
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>>
    >>> # Simple Moving Average
    >>> sma = MA(close, timeperiod=5, matype=0)
    >>>
    >>> # Exponential Moving Average
    >>> ema = MA(close, timeperiod=5, matype=1)
    >>>
    >>> # Double Exponential Moving Average
    >>> dema = MA(close, timeperiod=5, matype=3)
    >>>
    >>> # Kaufman Adaptive Moving Average
    >>> kama = MA(close, timeperiod=5, matype=6)

    See Also
    --------
    SMA : Simple Moving Average
    EMA : Exponential Moving Average
    DEMA : Double Exponential Moving Average
    KAMA : Kaufman Adaptive Moving Average
    """
    # Validate matype
    if matype not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError(f"Invalid matype: {matype}. Must be between 0 and 8.")

    # Route to appropriate MA implementation
    if matype == 0:
        # SMA - Simple Moving Average
        return SMA(close, timeperiod=timeperiod)
    elif matype == 1:
        # EMA - Exponential Moving Average
        return EMA(close, timeperiod=timeperiod)
    elif matype == 2:
        # WMA - Weighted Moving Average
        raise NotImplementedError("WMA (matype=2) not yet implemented")
    elif matype == 3:
        # DEMA - Double Exponential Moving Average
        return DEMA(close, timeperiod=timeperiod)
    elif matype == 4:
        # TEMA - Triple Exponential Moving Average
        raise NotImplementedError("TEMA (matype=4) not yet implemented")
    elif matype == 5:
        # TRIMA - Triangular Moving Average
        raise NotImplementedError("TRIMA (matype=5) not yet implemented")
    elif matype == 6:
        # KAMA - Kaufman Adaptive Moving Average
        return KAMA(close, timeperiod=timeperiod)
    elif matype == 7:
        # MAMA - Mesa Adaptive Moving Average
        raise NotImplementedError("MAMA (matype=7) not yet implemented")
    elif matype == 8:
        # T3 - Triple Exponential T3
        raise NotImplementedError("T3 (matype=8) not yet implemented")


# GPU (CuPy) implementation
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



def MAMA(close: Union[np.ndarray, list],
         fastlimit: float = 0.5,
         slowlimit: float = 0.05) -> tuple:
    """
    MESA Adaptive Moving Average (MAMA)

    MAMA (MESA Adaptive Moving Average) was developed by John Ehlers and uses
    the Hilbert Transform to adapt to price movement. It features a fast attack
    average and a slow decay average, allowing it to quickly respond to price
    changes while holding its value during consolidations.

    Returns both MAMA and FAMA (Following Adaptive Moving Average) lines.

    Parameters
    ----------
    close : array-like
        Close prices array
    fastlimit : float, optional
        Upper limit for the adaptive alpha (default: 0.5)
    slowlimit : float, optional
        Lower limit for the adaptive alpha (default: 0.05)

    Returns
    -------
    tuple of np.ndarray
        (mama, fama) - Two arrays with the MAMA and FAMA values

    Notes
    -----
    - Compatible with TA-Lib MAMA signature
    - This is a simplified implementation
    - Uses EMA-based adaptation instead of full Hilbert Transform
    - Lookback period: approximately 32 bars
    - MAMA responds faster than FAMA to price changes

    Interpretation:
    - MAMA > FAMA: Bullish signal (uptrend)
    - MAMA < FAMA: Bearish signal (downtrend)
    - MAMA crossing above FAMA: Buy signal
    - MAMA crossing below FAMA: Sell signal
    - Distance between MAMA and FAMA indicates trend strength

    Advantages:
    - Adaptive to market conditions
    - Reduces whipsaw trades
    - Fast response to trend changes
    - Holds value during consolidations
    - Dual lines provide crossover signals

    Common Uses:
    - Trend following and identification
    - Entry/exit signals via crossovers
    - Dynamic support/resistance levels
    - Trend strength measurement
    - Filter for trading systems

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import MAMA
    >>> close = np.linspace(100, 150, 50)
    >>> mama, fama = MAMA(close, fastlimit=0.5, slowlimit=0.05)
    >>> print(f"MAMA: {mama[-1]:.2f}, FAMA: {fama[-1]:.2f}")

    See Also
    --------
    KAMA : Kaufman Adaptive Moving Average
    EMA : Exponential Moving Average
    HT_TRENDLINE : Hilbert Transform - Instantaneous Trendline

    References
    ----------
    Ehlers, J. F. (2001). "MAMA - The Mother of Adaptive Moving Averages"
    Stocks & Commodities Magazine, September 2001
    """
    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    
    backend = get_backend()
    
    if backend == "gpu":
        # Use GPU implementation
        return _mama_cupy(close, fastlimit, slowlimit)
    else:
        # Use CPU implementation (default)
        # Initialize arrays
        mama = np.empty(n, dtype=np.float64)
        fama = np.empty(n, dtype=np.float64)

    # Lookback period (simplified - using ~32 like other HT indicators)
    lookback = 32
    for i in range(lookback):
        mama[i] = np.nan
        fama[i] = np.nan

    # Initialize with first valid value
    mama[lookback] = close[lookback]
    fama[lookback] = close[lookback]

    # Simplified adaptive calculation
    for i in range(lookback + 1, n):
        # Calculate price change rate (simplified adaptation)
        price_change = abs(close[i] - close[i-1])
        avg_change = 0.0
        for j in range(max(0, i-10), i):
            avg_change += abs(close[j] - close[j-1])
        avg_change = avg_change / min(10, i) if i > 0 else 1.0

        # Adaptive alpha based on price volatility
        if avg_change > 0:
            alpha = min(fastlimit, max(slowlimit, price_change / avg_change * slowlimit))
        else:
            alpha = slowlimit

        # MAMA calculation (adaptive EMA)
        mama[i] = alpha * close[i] + (1 - alpha) * mama[i-1]

        # FAMA follows MAMA with half the alpha
        fama_alpha = alpha * 0.5
        fama[i] = fama_alpha * mama[i] + (1 - fama_alpha) * fama[i-1]

    return mama, fama



@jit(nopython=True, cache=True)
def _sar_numba(high: np.ndarray, low: np.ndarray, acceleration: float, maximum: float, output: np.ndarray) -> None:
    """
    Numba-compiled SAR calculation (in-place)

    Parabolic SAR algorithm by J. Welles Wilder
    """
    n = len(high)

    # Initialize
    is_long = True  # Start with long position
    sar = low[0]
    ep = high[0]  # Extreme point
    af = acceleration  # Acceleration factor

    output[0] = sar

    for i in range(1, n):
        # Calculate new SAR
        sar = sar + af * (ep - sar)

        if is_long:
            # Long position
            # SAR should not be above prior two lows
            if i >= 1:
                sar = min(sar, low[i - 1])
            if i >= 2:
                sar = min(sar, low[i - 2])

            # Check for reversal
            if low[i] < sar:
                # Reverse to short
                is_long = False
                sar = ep  # SAR becomes the extreme point
                ep = low[i]  # New extreme point
                af = acceleration  # Reset AF
            else:
                # Continue long
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
        else:
            # Short position
            # SAR should not be below prior two highs
            if i >= 1:
                sar = max(sar, high[i - 1])
            if i >= 2:
                sar = max(sar, high[i - 2])

            # Check for reversal
            if high[i] > sar:
                # Reverse to long
                is_long = True
                sar = ep  # SAR becomes the extreme point
                ep = high[i]  # New extreme point
                af = acceleration  # Reset AF
            else:
                # Continue short
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)

        output[i] = sar


# GPU (CuPy) implementation
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



def SAR(high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        acceleration: float = 0.02,
        maximum: float = 0.2) -> np.ndarray:
    """
    Parabolic SAR (Stop and Reverse)

    Parabolic SAR, developed by J. Welles Wilder, is a trend-following indicator that
    provides entry and exit points. It appears as a series of dots placed above or below
    price bars, indicating the direction of the trend and potential reversal points.

    The indicator accelerates with the trend, moving closer to price as the trend continues,
    providing trailing stop levels.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    acceleration : float, optional
        Acceleration factor (default: 0.02)
    maximum : float, optional
        Maximum acceleration factor (default: 0.2)

    Returns
    -------
    np.ndarray
        Array of SAR values

    Notes
    -----
    - Compatible with TA-Lib SAR signature
    - Uses Numba JIT compilation for performance
    - No lookback period (starts from first bar)
    - Default values from Wilder's recommendation

    Algorithm
    ---------
    1. Start with initial SAR and trend direction
    2. Each period:
       - Calculate new SAR = Prior SAR + AF * (EP - Prior SAR)
       - If long: SAR must be below prior 2 lows
       - If short: SAR must be above prior 2 highs
    3. Check for reversal:
       - Long: If low < SAR, reverse to short
       - Short: If high > SAR, reverse to long
    4. Update EP (extreme point) and AF if trend continues

    Where:
    - SAR: Stop and Reverse point
    - EP: Extreme Point (highest high or lowest low in current trend)
    - AF: Acceleration Factor (starts at 'acceleration', increases by 'acceleration'
          each time EP is updated, capped at 'maximum')

    Interpretation:
    - SAR below price: Uptrend (bullish)
    - SAR above price: Downtrend (bearish)
    - SAR reversal: Trend change signal
    - Distance from price: Trend strength
    - AF increase: Trend acceleration

    Advantages:
    - Clear buy/sell signals
    - Automatic trailing stop
    - Works well in trending markets
    - Objective entry/exit points
    - No lag (price-based, not average-based)

    Disadvantages:
    - Whipsaw in sideways markets
    - Always in the market (long or short)
    - Cannot signal "no position"
    - Less effective in choppy conditions

    Common Uses:
    - Trend direction identification
    - Trailing stop placement
    - Entry/exit signals
    - Stop loss management
    - Trend reversal detection

    Trading Signals:
    1. Basic:
       - Buy when SAR flips from above to below price
       - Sell when SAR flips from below to above price

    2. Trailing Stop:
       - Long position: Use SAR as trailing stop loss
       - Short position: Use SAR as trailing stop loss

    3. Combination:
       - Use with trend filter (e.g., ADX)
       - Confirm with other indicators
       - Avoid in low ADX (choppy) conditions

    Parameter Adjustment:
    - Acceleration (default 0.02):
      - Lower (0.01): More conservative, fewer signals
      - Higher (0.05): More aggressive, more signals

    - Maximum (default 0.2):
      - Lower (0.1): Slower acceleration
      - Higher (0.3): Faster acceleration

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import SAR
    >>> high = np.array([110, 112, 111, 113, 115, 114, 116, 118])
    >>> low = np.array([100, 102, 101, 103, 105, 104, 106, 108])
    >>> sar = SAR(high, low, acceleration=0.02, maximum=0.2)
    >>> # SAR values indicate stop and reverse points

    See Also
    --------
    SAREXT : Parabolic SAR Extended
    ADX : Average Directional Index (trend strength)
    """
    # Validate inputs
    if acceleration <= 0:
        raise ValueError("acceleration must be > 0")
    if maximum <= 0:
        raise ValueError("maximum must be > 0")
    if acceleration > maximum:
        raise ValueError("acceleration must be <= maximum")

    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    
    backend = get_backend()
    
    if backend == "gpu":
        # Use GPU implementation
        return _sar_cupy(high, low, acceleration, maximum)
    else:
        # Use CPU implementation (default)
        output = np.empty(n, dtype=np.float64)
        _sar_numba(high, low, acceleration, maximum, output)
        return output


@jit(nopython=True, cache=True)
def _sarext_numba(high: np.ndarray, low: np.ndarray,
                  startvalue: float, offsetonreverse: float,
                  accelerationinit_long: float, accelerationlong: float, accelerationmax_long: float,
                  accelerationinit_short: float, accelerationshort: float, accelerationmax_short: float,
                  output: np.ndarray) -> None:
    """
    Numba-compiled SAREXT calculation (in-place)

    Extended Parabolic SAR with separate parameters for long and short
    """
    n = len(high)

    # Initialize
    is_long = True
    sar = startvalue if startvalue != 0 else low[0]
    ep = high[0]
    af = accelerationinit_long

    output[0] = sar

    for i in range(1, n):
        # Calculate new SAR
        sar = sar + af * (ep - sar)

        if is_long:
            # Long position
            if i >= 1:
                sar = min(sar, low[i - 1])
            if i >= 2:
                sar = min(sar, low[i - 2])

            # Check for reversal
            if low[i] < sar:
                is_long = False
                sar = ep + offsetonreverse
                ep = low[i]
                af = accelerationinit_short
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + accelerationlong, accelerationmax_long)
        else:
            # Short position
            if i >= 1:
                sar = max(sar, high[i - 1])
            if i >= 2:
                sar = max(sar, high[i - 2])

            # Check for reversal
            if high[i] > sar:
                is_long = True
                sar = ep - offsetonreverse
                ep = high[i]
                af = accelerationinit_long
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + accelerationshort, accelerationmax_short)

        output[i] = sar


# GPU (CuPy) implementation
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



def SAREXT(high: Union[np.ndarray, list],
           low: Union[np.ndarray, list],
           startvalue: float = 0.0,
           offsetonreverse: float = 0.0,
           accelerationinit_long: float = 0.02,
           accelerationlong: float = 0.02,
           accelerationmax_long: float = 0.2,
           accelerationinit_short: float = 0.02,
           accelerationshort: float = 0.02,
           accelerationmax_short: float = 0.2) -> np.ndarray:
    """
    Parabolic SAR - Extended (SAREXT)

    Extended version of Parabolic SAR with separate parameters for long and short positions.
    This allows for asymmetric acceleration and more fine-tuned control over the indicator's
    behavior in different market conditions.

    Parameters
    ----------
    high : array-like
        High prices array
    low : array-like
        Low prices array
    startvalue : float, optional
        Starting value for SAR (default: 0.0, auto-calculated)
    offsetonreverse : float, optional
        Offset on reversal (default: 0.0)
    accelerationinit_long : float, optional
        Initial acceleration factor for long (default: 0.02)
    accelerationlong : float, optional
        Acceleration increment for long (default: 0.02)
    accelerationmax_long : float, optional
        Maximum acceleration for long (default: 0.2)
    accelerationinit_short : float, optional
        Initial acceleration factor for short (default: 0.02)
    accelerationshort : float, optional
        Acceleration increment for short (default: 0.02)
    accelerationmax_short : float, optional
        Maximum acceleration for short (default: 0.2)

    Returns
    -------
    np.ndarray
        Array of SAR values

    Notes
    -----
    - Compatible with TA-Lib SAREXT signature
    - Allows asymmetric parameters for long/short
    - More flexible than standard SAR
    - Useful for markets with directional bias

    See Also
    --------
    SAR : Standard Parabolic SAR
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _sarext_cupy(high, low, startvalue, offsetonreverse,
                           accelerationinit_long, accelerationlong, accelerationmax_long,
                           accelerationinit_short, accelerationshort, accelerationmax_short)
    else:
        # Use CPU implementation (default)
        output = np.empty(n, dtype=np.float64)
        _sarext_numba(high, low, startvalue, offsetonreverse,
                     accelerationinit_long, accelerationlong, accelerationmax_long,
                     accelerationinit_short, accelerationshort, accelerationmax_short,
                     output)
        return output


# GPU (CuPy) implementation
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



def TEMA(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Triple Exponential Moving Average (TEMA)

    TEMA uses multiple EMAs to reduce lag and provide a smoother trend indicator.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Period for EMA calculations (default: 30)

    Returns
    -------
    np.ndarray
        Array of TEMA values

    Formula
    -------
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    See Also
    --------
    EMA : Exponential Moving Average
    DEMA : Double Exponential Moving Average
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _tema_cupy(data, timeperiod)
    else:
        # Use CPU implementation (default)
        # Calculate EMAs
        ema1 = EMA(data, timeperiod=timeperiod)
        ema2 = EMA(ema1, timeperiod=timeperiod)
        ema3 = EMA(ema2, timeperiod=timeperiod)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        output = 3.0 * ema1 - 3.0 * ema2 + ema3

        return output


# GPU (CuPy) implementation
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



def T3(data: Union[np.ndarray, list],
       timeperiod: int = 5,
       vfactor: float = 0.7) -> np.ndarray:
    """
    Triple Exponential Moving Average (T3)

    T3 is a smoothed moving average developed by Tim Tillson that uses multiple
    EMAs with a volume factor for improved smoothness and reduced lag.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Period for EMA calculations (default: 5)
    vfactor : float, optional
        Volume factor (default: 0.7, range: 0 to 1)

    Returns
    -------
    np.ndarray
        Array of T3 values

    Notes
    -----
    T3 applies 6 EMAs with coefficients based on vfactor
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _t3_cupy(data, timeperiod, vfactor)
    else:
        # Use CPU implementation (default)
        # Calculate coefficients
        c1 = -vfactor * vfactor * vfactor
        c2 = 3.0 * vfactor * vfactor + 3.0 * vfactor * vfactor * vfactor
        c3 = -6.0 * vfactor * vfactor - 3.0 * vfactor - 3.0 * vfactor * vfactor * vfactor
        c4 = 1.0 + 3.0 * vfactor + vfactor * vfactor * vfactor + 3.0 * vfactor * vfactor

        # Calculate 6 EMAs
        ema1 = EMA(data, timeperiod=timeperiod)
        ema2 = EMA(ema1, timeperiod=timeperiod)
        ema3 = EMA(ema2, timeperiod=timeperiod)
        ema4 = EMA(ema3, timeperiod=timeperiod)
        ema5 = EMA(ema4, timeperiod=timeperiod)
        ema6 = EMA(ema5, timeperiod=timeperiod)

        # T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
        output = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

        return output


# GPU (CuPy) implementation
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



def TRIMA(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Triangular Moving Average (TRIMA)

    TRIMA is a double-smoothed simple moving average that places more weight
    on the middle portion of the data series. It's calculated as an SMA of an SMA,
    creating a triangular weighting pattern.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Period for the moving average (default: 30)

    Returns
    -------
    np.ndarray
        Array of TRIMA values

    Notes
    -----
    - Provides extra smoothing compared to SMA
    - Lags more than SMA but filters noise better
    - Compatible with TA-Lib TRIMA signature
    - More weight on middle data points
    
    Formula
    -------
    When period is odd:
        n = (period + 1) / 2
        TRIMA = SMA(SMA(data, n), n)
    
    When period is even:
        n1 = period / 2
        n2 = n1 + 1
        TRIMA = SMA(SMA(data, n1), n2)

    Interpretation:
    - Smoother than SMA due to double averaging
    - Good for identifying long-term trends
    - Filters out short-term noise
    - Slower to react to price changes

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import TRIMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> trima = TRIMA(close, timeperiod=5)

    See Also
    --------
    SMA : Simple Moving Average
    EMA : Exponential Moving Average
    DEMA : Double Exponential Moving Average
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    
    if n == 0:
        return np.array([], dtype=np.float64)

    # Calculate the periods for double SMA
    if timeperiod % 2 == 1:  # Odd period
        n1 = (timeperiod + 1) // 2
        n2 = n1
    else:  # Even period
        n1 = timeperiod // 2
        n2 = n1 + 1

    # Double SMA
    sma1 = SMA(data, timeperiod=n1)
    trima = SMA(sma1, timeperiod=n2)

    return trima


@jit(nopython=True, cache=True)
def _wma_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled WMA calculation using incremental O(n) algorithm

    The optimization uses incremental calculation to avoid recalculating
    the entire weighted sum at each position. When sliding the window:
    1. Remove the old value (which had weight 1)
    2. Subtract simple_sum from weighted_sum (all values lose 1 weight)
    3. Add new value with full weight (timeperiod)
    4. Update simple_sum

    This reduces complexity from O(n*timeperiod) to O(n).
    """
    n = len(data)

    # Calculate sum of weights: 1 + 2 + 3 + ... + timeperiod
    weight_sum = (timeperiod * (timeperiod + 1)) / 2.0

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate first WMA value using the standard method
    weighted_sum = 0.0
    simple_sum = 0.0
    for j in range(timeperiod):
        weight = j + 1  # Weight increases from 1 (oldest) to timeperiod (newest)
        value = data[timeperiod - 1 - (timeperiod - 1 - j)]  # = data[j]
        weighted_sum += value * weight
        simple_sum += value

    output[timeperiod - 1] = weighted_sum / weight_sum

    # Use incremental calculation for remaining values
    # Formula when sliding window from position i to i+1:
    # - Remove old value: weighted_sum -= data[i-timeperiod] (weight 1)
    # - All values lose 1 weight: weighted_sum -= simple_sum
    # - Add new value: weighted_sum += data[i] * timeperiod
    # - Update simple_sum: simple_sum = simple_sum - data[i-timeperiod] + data[i]
    for i in range(timeperiod, n):
        old_value = data[i - timeperiod]
        new_value = data[i]

        # Remove contribution of the oldest value (which had weight 1)
        # Subtract simple_sum (all remaining values lose 1 weight)
        weighted_sum = weighted_sum - simple_sum

        # Update simple sum (remove old, add new)
        simple_sum = simple_sum - old_value + new_value

        # Add new value with full weight
        weighted_sum = weighted_sum + new_value * timeperiod

        output[i] = weighted_sum / weight_sum


# GPU (CuPy) implementation
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



def WMA(data: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Weighted Moving Average (WMA)

    WMA assigns greater weight to recent data points and less weight to older
    data points. The weights decrease linearly from the most recent to the
    oldest data point.

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods for the moving average (default: 30)

    Returns
    -------
    np.ndarray
        Array of WMA values

    Notes
    -----
    - More weight on recent data
    - Linear weight decrease
    - More responsive than SMA
    - Less responsive than EMA
    - Compatible with TA-Lib WMA signature

    Formula
    -------
    WMA = (P1*n + P2*(n-1) + P3*(n-2) + ... + Pn*1) / (n*(n+1)/2)

    Where:
    - P1 = most recent price
    - P2 = second most recent price
    - Pn = oldest price in the window
    - n = timeperiod

    Weight Calculation:
    - Sum of weights = 1 + 2 + 3 + ... + n = n*(n+1)/2
    - Most recent: weight = n
    - Second most recent: weight = n-1
    - Oldest: weight = 1

    Example with period=4:
    - Most recent: weight 4/10 = 40%
    - Previous: weight 3/10 = 30%
    - Next: weight 2/10 = 20%
    - Oldest: weight 1/10 = 10%
    - Sum = 4+3+2+1 = 10

    Interpretation:
    - Emphasizes recent price action
    - Smoother than EMA
    - Less lag than SMA
    - Good balance of smoothness and responsiveness

    Comparison with Other MAs:
    - SMA: Equal weights
    - WMA: Linear decreasing weights
    - EMA: Exponential decreasing weights
    - Response: EMA > WMA > SMA

    Applications:
    - Trend identification
    - Support/resistance levels
    - Crossover systems
    - Price filtering
    - Momentum confirmation

    Trading Signals:
    - Price above WMA: Uptrend
    - Price below WMA: Downtrend
    - WMA slope up: Bullish
    - WMA slope down: Bearish
    - Short/Long WMA cross: Trend change

    Advantages:
    - More responsive than SMA
    - Smoother than short-period EMA
    - Clear weighting scheme
    - Good for intermediate trends

    Disadvantages:
    - More lag than EMA
    - Complex than SMA
    - Can overshoot in volatile markets
    - Weights somewhat arbitrary

    Common Periods:
    - 10: Short-term (day trading)
    - 20: Medium-term (swing trading)
    - 50: Long-term (position trading)
    - 200: Very long-term (investors)

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import WMA
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> wma = WMA(close, timeperiod=5)

    See Also
    --------
    SMA : Simple Moving Average
    EMA : Exponential Moving Average
    DEMA : Double Exponential Moving Average
    TEMA : Triple Exponential Moving Average
    """
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n == 0:
        return np.array([], dtype=np.float64)

    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _wma_numba(data, timeperiod, output)

    return output
