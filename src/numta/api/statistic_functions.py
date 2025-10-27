"""
Statistic Functions - Statistical analysis functions

This module implements statistical functions compatible with TA-Lib.
"""

import numpy as np
from typing import Union

# Import CPU implementations
from ..cpu.statistic_functions import (
    _beta_numba,
    _correl_numba,
    _linearreg_numba,
    _linearreg_angle_numba,
    _linearreg_intercept_numba,
    _linearreg_slope_numba,
)


def BETA(high: Union[np.ndarray, list],
         low: Union[np.ndarray, list],
         timeperiod: int = 5) -> np.ndarray:
    """
    Beta - Volatility measure relative to a benchmark

    Beta measures how volatile a security is relative to a benchmark.
    It's calculated as the covariance between the security and benchmark
    divided by the variance of the benchmark.

    Parameters
    ----------
    high : array-like
        High prices (treated as the asset/security)
    low : array-like
        Low prices (treated as the benchmark)
    timeperiod : int, default 5
        Number of periods to use for calculation

    Returns
    -------
    np.ndarray
        Array of beta values:
        - Beta > 1: Asset is more volatile than benchmark
        - Beta = 1: Asset has same volatility as benchmark
        - Beta < 1: Asset is less volatile than benchmark
        - Beta < 0: Asset moves inversely to benchmark

    Notes
    -----
    Beta Formula:
        Beta = Covariance(asset, benchmark) / Variance(benchmark)
        Beta = Σ((asset - asset_mean) * (benchmark - benchmark_mean)) / Σ((benchmark - benchmark_mean)²)

    Common uses:
    - Portfolio risk assessment
    - Capital Asset Pricing Model (CAPM)
    - Hedge ratio calculation
    - Understanding systematic risk

    A higher beta indicates the asset amplifies market movements, while
    a lower beta suggests the asset is more stable relative to the market.
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # Validate inputs
    if len(high) != len(low):
        raise ValueError("high and low must have the same length")

    if len(high) == 0:
        return np.zeros(0, dtype=np.float64)

    if timeperiod < 1:
        raise ValueError("timeperiod must be at least 1")

    # Calculate using Numba implementation
    output = np.empty(len(high), dtype=np.float64)
    _beta_numba(high, low, timeperiod, output)

    return output


def CORREL(high: Union[np.ndarray, list],
           low: Union[np.ndarray, list],
           timeperiod: int = 30) -> np.ndarray:
    """
    Pearson's Correlation Coefficient (r)

    Pearson's Correlation Coefficient measures the linear correlation between
    two datasets. In TA-Lib, it calculates the correlation between high and low
    prices over a specified period.

    The correlation coefficient ranges from -1 to +1:
    - +1 indicates perfect positive correlation
    - 0 indicates no correlation
    - -1 indicates perfect negative correlation

    Formula
    -------
    r = (n × Σ(xy) - Σx × Σy) / sqrt((n × Σ(x²) - (Σx)²) × (n × Σ(y²) - (Σy)²))

    Where:
    - x = high prices
    - y = low prices
    - n = timeperiod
    - Σ = sum over the timeperiod window

    Parameters
    ----------
    high : array-like
        High prices array (first variable)
    low : array-like
        Low prices array (second variable)
    timeperiod : int, optional
        Number of periods for the calculation (default: 30)

    Returns
    -------
    np.ndarray
        Array of correlation coefficients with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib CORREL signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Correlation values range from -1.0 to +1.0
    - All input arrays must have the same length
    - When variance is zero (constant prices), correlation is set to 0

    Interpretation
    --------------
    - r = +1.0: Perfect positive correlation (high and low move together)
    - r = 0.0: No linear correlation
    - r = -1.0: Perfect negative correlation (high and low move opposite)
    - |r| > 0.7: Strong correlation
    - |r| < 0.3: Weak correlation
    - 0.3 ≤ |r| ≤ 0.7: Moderate correlation

    Common Uses
    -----------
    - Measure relationship between high and low prices
    - Identify range compression (high correlation = tight range)
    - Detect volatility changes (correlation patterns)
    - Compare price series relationships
    - Validate trading pair correlations

    Statistical Properties
    ----------------------
    - Measures only linear relationships
    - Sensitive to outliers
    - Assumes bivariate normality for significance testing
    - Scale-independent (normalized measure)

    Examples
    --------
    >>> import numpy as np
    >>> from numta import CORREL
    >>> # Positively correlated high and low
    >>> high = np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
    >>> low = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    >>> correl = CORREL(high, low, timeperiod=5)
    >>> print(correl)  # Should be close to +1.0

    See Also
    --------
    BETA : Beta (volatility measure)
    VAR : Variance
    STDDEV : Standard Deviation
    """
    # Validate inputs
    if timeperiod < 1:
        raise ValueError("timeperiod must be >= 1")

    # Convert to numpy arrays if needed
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # Check all arrays have the same length
    n = len(high)
    if len(low) != n:
        raise ValueError("high and low must have the same length")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Not enough data points - return all NaN
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _correl_numba(high, low, timeperiod, output)

    return output


def LINEARREG(close: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    Linear Regression

    Linear Regression fits a straight line through the price data using the least
    squares method. This function returns the endpoint value of the regression line
    for each window, which represents the "fair value" or trend-following estimate
    of the current price.

    The regression line minimizes the sum of squared distances between the line
    and the actual price points. The returned value is the projection of the
    regression line at the most recent bar.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the regression (default: 14)

    Returns
    -------
    np.ndarray
        Array of linear regression values with NaN for the lookback period

    Notes
    -----
    - Compatible with TA-Lib LINEARREG signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Uses least squares method for optimal fit

    Formula
    -------
    For each window of length timeperiod:
    1. Fit line: y = b + m*x
       where x = 0, 1, 2, ..., timeperiod-1
             y = close prices in window

    2. Calculate slope (m) and intercept (b):
       m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
       b = (Σy - m*Σx) / n

    3. Return endpoint: LINEARREG = b + m*(timeperiod-1)

    Lookback period: timeperiod - 1
    (For timeperiod=14, lookback=13)

    Interpretation:
    - Value above price: Downtrend (price below trend line)
    - Value below price: Uptrend (price above trend line)
    - Price crossing above LINEARREG: Bullish signal
    - Price crossing below LINEARREG: Bearish signal
    - Steeper slope indicates stronger trend
    - Can be used as dynamic support/resistance

    Advantages:
    - Mathematically optimal trend line (least squares)
    - Smooths price action to reveal underlying trend
    - Objective measure of trend direction and strength
    - Can be used for forecasting next period
    - Less lag than moving averages

    Common Uses:
    - Trend identification and confirmation
    - Dynamic support/resistance levels
    - Price deviation analysis (distance from regression line)
    - Breakout detection (price crossing regression line)
    - Combine with LINEARREG_ANGLE for trend strength
    - Use with LINEARREG_SLOPE for trend direction

    Comparison with Moving Averages:
    - More responsive to recent price changes
    - Better at identifying trend changes
    - Provides "fair value" estimate based on linear trend
    - Can project forward beyond current bar

    Examples
    --------
    >>> import numpy as np
    >>> from numta import LINEARREG
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113])
    >>> linreg = LINEARREG(close, timeperiod=14)
    >>> print(linreg)

    See Also
    --------
    LINEARREG_ANGLE : Linear Regression Angle
    LINEARREG_SLOPE : Linear Regression Slope
    LINEARREG_INTERCEPT : Linear Regression Intercept
    TSF : Time Series Forecast
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
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _linearreg_numba(close, timeperiod, output)

    return output


def LINEARREG_ANGLE(close: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    Linear Regression Angle

    Linear Regression Angle calculates the angle (in degrees) of the linear
    regression line fitted through the price data. This provides a quantitative
    measure of trend direction and steepness.

    The angle is calculated as arctan(slope) × (180/π), converting the regression
    line's slope into an intuitive angle measurement. Positive angles indicate
    uptrends, negative angles indicate downtrends, and angles near zero indicate
    sideways movement.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the regression (default: 14)

    Returns
    -------
    np.ndarray
        Array of linear regression angles in degrees with NaN for lookback period

    Notes
    -----
    - Compatible with TA-Lib LINEARREG_ANGLE signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Returns angle in degrees (not radians)
    - Positive values = uptrend, negative values = downtrend

    Formula
    -------
    For each window of length timeperiod:
    1. Calculate regression slope (m):
       m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
       where x = 0, 1, 2, ..., timeperiod-1
             y = close prices in window

    2. Convert slope to angle:
       LINEARREG_ANGLE = arctan(m) × (180/π)

    Lookback period: timeperiod - 1
    (For timeperiod=14, lookback=13)

    Interpretation:
    - Angle > 0: Uptrend (bullish)
    - Angle < 0: Downtrend (bearish)
    - Angle ≈ 0: Sideways/ranging market
    - |Angle| > 45°: Very steep trend (may be unsustainable)
    - |Angle| 15-45°: Strong trending market
    - |Angle| 5-15°: Moderate trend
    - |Angle| < 5°: Weak trend or consolidation
    - Increasing angle: Trend strengthening
    - Decreasing angle: Trend weakening

    Angle Ranges:
    - 0° to 90°: Uptrend (positive slope)
    - -90° to 0°: Downtrend (negative slope)
    - Near ±90°: Nearly vertical (extreme trend)
    - Near 0°: Nearly horizontal (no trend)

    Advantages:
    - Intuitive measure of trend strength
    - Easy to quantify trend steepness
    - Can be used for objective trading rules
    - Normalized across different securities
    - Helps identify unsustainable trends

    Common Uses:
    - Trend strength measurement and comparison
    - Filter trades (only trade when angle > threshold)
    - Identify overbought/oversold conditions (extreme angles)
    - Detect trend changes (angle crossing zero)
    - Combine with price action for confirmation
    - Set stop-loss based on angle thresholds

    Trading Applications:
    - Enter long when angle crosses above +15°
    - Enter short when angle crosses below -15°
    - Exit when angle approaches ±45° (overextended)
    - Avoid trading when |angle| < 5° (choppy market)
    - Use angle divergence with price for reversals

    Limitations:
    - Lagging indicator (based on historical data)
    - Can give false signals in choppy markets
    - Extreme angles may indicate trend exhaustion
    - Shorter timeperiods = more noise and false signals

    Examples
    --------
    >>> import numpy as np
    >>> from numta import LINEARREG_ANGLE
    >>> # Uptrending prices
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113])
    >>> angle = LINEARREG_ANGLE(close, timeperiod=14)
    >>> print(angle)  # Should show positive angles

    See Also
    --------
    LINEARREG : Linear Regression
    LINEARREG_SLOPE : Linear Regression Slope
    LINEARREG_INTERCEPT : Linear Regression Intercept
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
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _linearreg_angle_numba(close, timeperiod, output)

    return output


def LINEARREG_INTERCEPT(close: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    Linear Regression Intercept

    Linear Regression Intercept calculates the y-intercept (b) of the linear
    regression line fitted through the price data. The intercept represents
    where the regression line crosses the y-axis (at x=0).

    The regression line equation is y = b + m*x, where b is the intercept
    and m is the slope. The intercept can be used to understand the baseline
    level of the trend.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the regression (default: 14)

    Returns
    -------
    np.ndarray
        Array of linear regression intercept values with NaN for lookback period

    Notes
    -----
    - Compatible with TA-Lib LINEARREG_INTERCEPT signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Uses least squares method for optimal fit

    Formula
    -------
    For each window of length timeperiod:
    1. Fit line: y = b + m*x
       where x = 0, 1, 2, ..., timeperiod-1
             y = close prices in window

    2. Calculate slope (m):
       m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)

    3. Calculate intercept (b):
       b = (Σy - m*Σx) / n

    Lookback period: timeperiod - 1
    (For timeperiod=14, lookback=13)

    Interpretation:
    - Represents the y-intercept of the regression line
    - Shows the baseline level at the beginning of the window
    - Changes indicate shifts in the trend baseline
    - Can be combined with LINEARREG_SLOPE for full line equation
    - Higher intercept values indicate higher baseline prices

    Relationship with Other Indicators:
    - LINEARREG = b + m*(timeperiod-1)
    - TSF = b + m*timeperiod (Time Series Forecast)
    - Use with LINEARREG_SLOPE: y = intercept + slope*x

    Common Uses:
    - Determine baseline level of trend
    - Reconstruct full regression line equation
    - Analyze trend starting points
    - Compare baseline shifts across different periods
    - Combine with slope for trend projections

    Mathematical Properties:
    - Minimizes sum of squared residuals
    - Represents regression line at x=0
    - Independent of slope calculation
    - Sensitive to price level changes

    Examples
    --------
    >>> import numpy as np
    >>> from numta import LINEARREG_INTERCEPT
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113])
    >>> intercept = LINEARREG_INTERCEPT(close, timeperiod=14)
    >>> print(intercept)

    See Also
    --------
    LINEARREG : Linear Regression
    LINEARREG_SLOPE : Linear Regression Slope
    LINEARREG_ANGLE : Linear Regression Angle
    TSF : Time Series Forecast
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
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _linearreg_intercept_numba(close, timeperiod, output)

    return output


def LINEARREG_SLOPE(close: Union[np.ndarray, list], timeperiod: int = 14) -> np.ndarray:
    """
    Linear Regression Slope

    Linear Regression Slope calculates the slope (m) of the linear regression
    line fitted through the price data. The slope represents the rate of change
    of the trend line and indicates the strength and direction of the trend.

    The regression line equation is y = b + m*x, where m is the slope and b is
    the intercept. The slope is measured in price units per period, providing
    a direct measure of trend momentum.

    Parameters
    ----------
    close : array-like
        Close prices array
    timeperiod : int, optional
        Number of periods for the regression (default: 14)

    Returns
    -------
    np.ndarray
        Array of linear regression slope values with NaN for lookback period

    Notes
    -----
    - Compatible with TA-Lib LINEARREG_SLOPE signature
    - Uses Numba JIT compilation for maximum performance
    - The first (timeperiod - 1) values will be NaN
    - Lookback period: timeperiod - 1
    - Uses least squares method for optimal fit
    - Slope is in price units per period

    Formula
    -------
    For each window of length timeperiod:
    1. Fit line: y = b + m*x
       where x = 0, 1, 2, ..., timeperiod-1
             y = close prices in window

    2. Calculate slope (m):
       m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)

    Lookback period: timeperiod - 1
    (For timeperiod=14, lookback=13)

    Interpretation:
    - Positive slope: Uptrend (bullish)
    - Negative slope: Downtrend (bearish)
    - Slope ≈ 0: Sideways/ranging market
    - Larger absolute slope: Stronger trend
    - Slope units: Price change per period
    - Increasing slope: Trend accelerating
    - Decreasing slope: Trend decelerating

    Advantages:
    - Direct measure of trend strength in price units
    - Not dependent on angle conversion
    - Easy to interpret across different securities
    - Can be used to project future prices
    - Provides rate of change information

    Common Uses:
    - Measure trend strength and direction
    - Calculate expected price change per period
    - Project future price levels: price + slope*periods
    - Filter trades based on minimum slope requirement
    - Detect trend acceleration/deceleration
    - Compare trend strength across different timeframes

    Relationship with Other Indicators:
    - LINEARREG_ANGLE = arctan(LINEARREG_SLOPE) * 180/π
    - LINEARREG = intercept + slope*(timeperiod-1)
    - TSF = intercept + slope*timeperiod
    - Use with LINEARREG_INTERCEPT for full line equation

    Trading Applications:
    - Enter long when slope crosses above threshold (e.g., 0.5)
    - Enter short when slope crosses below threshold (e.g., -0.5)
    - Exit when slope approaches zero (trend weakening)
    - Use slope divergence with price for reversals
    - Filter: only trade when |slope| > minimum value

    Examples
    --------
    >>> import numpy as np
    >>> from numta import LINEARREG_SLOPE
    >>> # Uptrending prices
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113])
    >>> slope = LINEARREG_SLOPE(close, timeperiod=14)
    >>> print(slope)  # Should show positive slope values

    See Also
    --------
    LINEARREG : Linear Regression
    LINEARREG_ANGLE : Linear Regression Angle
    LINEARREG_INTERCEPT : Linear Regression Intercept
    TSF : Time Series Forecast
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
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _linearreg_slope_numba(close, timeperiod, output)

    return output
