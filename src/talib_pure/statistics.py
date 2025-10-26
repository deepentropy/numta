"""
Statistics Functions - Statistical analysis functions

This module implements statistical functions compatible with TA-Lib.
"""

import numpy as np
from typing import Union
from numba import jit


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


def STDDEV(data: Union[np.ndarray, list],
           timeperiod: int = 5,
           nbdev: float = 1.0) -> np.ndarray:
    """
    Standard Deviation (STDDEV)

    STDDEV calculates the statistical standard deviation of price over a specified
    period. It measures the dispersion of prices from their mean, providing insight
    into volatility and price variability.

    Standard deviation is a key component of Bollinger Bands and is widely used
    in risk management and volatility analysis.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for calculation (default: 5)
    nbdev : float, optional
        Number of deviations (multiplier) (default: 1.0)

    Returns
    -------
    np.ndarray
        Array of standard deviation values

    Notes
    -----
    - Compatible with TA-Lib STDDEV signature
    - Uses Numba JIT compilation for performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Uses population standard deviation (division by n, not n-1)

    Formula
    -------
    For each position i:
    1. Calculate mean: Mean = Sum(data[i-timeperiod+1 : i+1]) / timeperiod
    2. Calculate variance: Var = Sum((data - mean)^2) / timeperiod
    3. Calculate std dev: STDDEV = sqrt(Var) * nbdev

    Lookback period: timeperiod - 1
    (For timeperiod=5, lookback=4)

    Interpretation:
    - High STDDEV: High volatility, wider price swings
    - Low STDDEV: Low volatility, tighter price range
    - Increasing STDDEV: Rising volatility
    - Decreasing STDDEV: Falling volatility
    - STDDEV expansion: Potential breakout
    - STDDEV contraction: Consolidation/squeeze

    Advantages:
    - Objective volatility measure
    - Statistical foundation
    - Widely understood
    - Key component of other indicators
    - Useful for risk management

    Common Uses:
    - Volatility measurement
    - Bollinger Bands calculation
    - Position sizing (risk-based)
    - Volatility breakout detection
    - Risk assessment
    - Market regime identification

    Trading Applications:
    1. Bollinger Bands:
       - Upper Band = SMA + (2 * STDDEV)
       - Lower Band = SMA - (2 * STDDEV)

    2. Volatility Breakout:
       - Low STDDEV followed by expansion = Breakout
       - Trade in direction of breakout

    3. Position Sizing:
       - Size = Risk / (STDDEV * multiplier)
       - Reduce size when volatility high

    4. Market Regimes:
       - High STDDEV: Trending market
       - Low STDDEV: Range-bound market

    Parameter Adjustment:
    - timeperiod (default 5):
      - Shorter: More sensitive to recent changes
      - Longer: Smoother, less reactive
      - Common: 20 (Bollinger Bands default)

    - nbdev (default 1.0):
      - 1.0: One standard deviation
      - 2.0: Two standard deviations (Bollinger Bands)
      - 3.0: Three standard deviations

    Statistical Interpretation:
    Assuming normal distribution:
    - ±1 STDDEV: ~68% of values
    - ±2 STDDEV: ~95% of values
    - ±3 STDDEV: ~99.7% of values

    Comparison with Related Indicators:
    - ATR: Average True Range (uses true range, not price)
    - NATR: Normalized ATR (percentage)
    - Variance: STDDEV squared
    - Beta: Volatility relative to benchmark

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import STDDEV
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    >>> stddev = STDDEV(close, timeperiod=5, nbdev=1.0)
    >>> # Higher values indicate higher volatility

    >>> # Bollinger Bands calculation
    >>> from talib_pure import SMA
    >>> middle = SMA(close, timeperiod=20)
    >>> std = STDDEV(close, timeperiod=20, nbdev=2.0)
    >>> upper = middle + std
    >>> lower = middle - std

    See Also
    --------
    VAR : Variance
    ATR : Average True Range
    NATR : Normalized Average True Range
    BBANDS : Bollinger Bands
    """
    # Validate inputs
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    if nbdev <= 0:
        raise ValueError("nbdev must be > 0")

    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)

    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _stddev_numba(data, timeperiod, nbdev, output)

    return output


@jit(nopython=True, cache=True)
def _tsf_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """Numba-compiled TSF calculation using linear regression"""
    n = len(data)
    
    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan
    
    # Precalculate sums for x values (0, 1, 2, ..., timeperiod-1)
    sum_x = 0.0
    sum_xx = 0.0
    for x in range(timeperiod):
        sum_x += x
        sum_xx += x * x
    
    # Calculate TSF for each window
    for i in range(timeperiod - 1, n):
        # Calculate sums for current window
        sum_y = 0.0
        sum_xy = 0.0
        
        for j in range(timeperiod):
            y = data[i - timeperiod + 1 + j]
            x = float(j)
            sum_y += y
            sum_xy += x * y
        
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


def TSF(data: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    Time Series Forecast (TSF)

    TSF calculates the forecasted value based on linear regression analysis.
    It performs linear least squares regression over a moving window and predicts
    the value for the next bar.

    Parameters
    ----------
    data : array-like
        Input data array (typically close prices)
    timeperiod : int, optional
        Number of periods for regression (default: 14)

    Returns
    -------
    np.ndarray
        Array of forecasted values

    Notes
    -----
    - Based on linear regression
    - Forecasts one period ahead
    - Reacts faster than moving averages
    - Compatible with TA-Lib TSF signature
    - Also known as "moving linear regression"

    Formula
    -------
    For each window of 'timeperiod' bars:
    1. Fit linear regression: y = a + b*x
    2. Calculate slope: b = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    3. Calculate intercept: a = (Σy - b*Σx) / n
    4. Forecast: TSF = a + b*timeperiod

    Where x = 0, 1, 2, ..., timeperiod-1
    And y = price values in the window

    Interpretation:
    - Price above TSF: Bullish trend
    - Price below TSF: Bearish trend
    - TSF slope up: Uptrend
    - TSF slope down: Downtrend
    - TSF crossing price: Potential reversal

    Trading Signals:
    - Buy: Price crosses above TSF
    - Sell: Price crosses below TSF
    - Combine with moving average for confirmation
    - TSF/MA crossover indicates trend change

    Advantages:
    - Faster than moving averages
    - Provides forward-looking forecast
    - Smoother than price
    - Statistical basis (least squares)

    Comparison with Moving Averages:
    - TSF: Projects future value
    - MA: Average of past values
    - TSF reacts faster to trends
    - MA smoother but lags more

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import TSF
    >>> close = np.linspace(100, 120, 50)
    >>> tsf = TSF(close, timeperiod=14)

    See Also
    --------
    LINEARREG : Linear Regression
    LINEARREG_SLOPE : Linear Regression Slope
    SMA : Simple Moving Average
    EMA : Exponential Moving Average
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
    _tsf_numba(data, timeperiod, output)

    return output


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


def VAR(data: Union[np.ndarray, list], timeperiod: int = 5, nbdev: float = 1.0) -> np.ndarray:
    """
    Variance (VAR)

    Calculates the population variance over a specified period. This is the
    biased variance estimator that divides by N (population size) rather than
    N-1 (sample variance).

    Parameters
    ----------
    data : array-like
        Input data array
    timeperiod : int, optional
        Number of periods for variance calculation (default: 5)
    nbdev : float, optional
        Number of deviations to multiply by (default: 1.0)

    Returns
    -------
    np.ndarray
        Array of variance values

    Notes
    -----
    - Uses population variance formula (divide by N)
    - Compatible with TA-Lib VAR signature
    - STDDEV = sqrt(VAR) when nbdev=1
    - Uses Numba JIT for performance

    Formula
    -------
    VAR = (Σ(x - mean)²) / N * nbdev

    Where:
    - x = individual data points in the window
    - mean = average of the window
    - N = timeperiod (population size)
    - nbdev = multiplier (default 1.0)

    Interpretation:
    - Higher variance: More price volatility
    - Lower variance: Less price volatility
    - Variance squared units (price²)
    - Take square root to get standard deviation

    Population vs Sample Variance:
    - Population VAR: divide by N (this function)
    - Sample VAR: divide by (N-1)
    - TA-Lib uses population variance (ddof=0)

    Applications:
    - Volatility measurement
    - Risk assessment
    - Bollinger Bands calculation (uses STDDEV)
    - Statistical analysis
    - Pattern recognition

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import VAR
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> var = VAR(data, timeperiod=5)

    See Also
    --------
    STDDEV : Standard Deviation (sqrt of variance)
    ATR : Average True Range
    NATR : Normalized Average True Range
    """
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")
    if nbdev <= 0:
        raise ValueError("nbdev must be > 0")

    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n == 0:
        return np.array([], dtype=np.float64)

    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _var_numba(data, timeperiod, nbdev, output)

    return output
