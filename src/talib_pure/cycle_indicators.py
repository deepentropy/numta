"""
Cycle Indicators - Hilbert Transform based cycle analysis functions

These indicators use the Hilbert Transform to analyze market cycles.
The Hilbert Transform is a mathematical technique that produces in-phase
and quadrature components of a price series, allowing cycle analysis.
"""

import numpy as np
from typing import Union, Tuple
from numba import jit


@jit(nopython=True, cache=True)
def _ht_dcperiod_numba(close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled HT_DCPERIOD calculation

    Hilbert Transform - Dominant Cycle Period
    Uses Hilbert Transform to determine the dominant cycle period
    """
    n = len(close)

    # Constants
    a = 0.0962
    b = 0.5769

    # Minimum lookback for Hilbert Transform
    lookback = 32

    # Fill lookback period with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Initialize arrays
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    im = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    smooth_period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        # Smooth price
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        # Detrend
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature components
        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        # Advance the phase of I1 and Q1 by 90 degrees
        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition for 3-bar averaging
        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]

            # Smooth I and Q
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        # Homodyne Discriminator
        if i >= 11:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]

            re[i] = 0.2 * re[i] + 0.8 * re[i-1]
            im[i] = 0.2 * im[i] + 0.8 * im[i-1]

            # Compute period
            if im[i] != 0.0 and re[i] != 0.0:
                period[i] = 360.0 / (np.arctan(im[i] / re[i]) * 180.0 / np.pi)

            # Constrain period
            if period[i] > 1.5 * period[i-1]:
                period[i] = 1.5 * period[i-1]
            if period[i] < 0.67 * period[i-1]:
                period[i] = 0.67 * period[i-1]
            if period[i] < 6:
                period[i] = 6
            if period[i] > 50:
                period[i] = 50

            # Smooth period
            period[i] = 0.2 * period[i] + 0.8 * period[i-1]
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]
        else:
            period[i] = 0.0
            smooth_period[i] = 0.0

        output[i] = smooth_period[i]


def HT_DCPERIOD(close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Hilbert Transform - Dominant Cycle Period

    This function uses the Hilbert Transform to identify the dominant cycle
    period in a price series. It analyzes market cycles and returns the
    period length of the dominant cycle.

    Parameters
    ----------
    close : array-like
        Close prices array

    Returns
    -------
    np.ndarray
        Array of dominant cycle period values

    Notes
    -----
    - Compatible with TA-Lib HT_DCPERIOD signature
    - Uses Numba JIT compilation for performance
    - The first 32 values will be NaN (unstable period)
    - Period values typically range from 6 to 50
    - Based on John Ehlers' work on MESA (Maximum Entropy Spectrum Analysis)

    Interpretation:
    - Values represent the length in bars of the dominant cycle
    - Lower values indicate shorter, faster cycles
    - Higher values indicate longer, slower cycles
    - Use to adapt other indicators to current market conditions
    - Helps identify cycle changes and market regime shifts

    Common Uses:
    - Adaptive indicator periods
    - Cycle-based trading systems
    - Market regime identification
    - Optimizing moving average periods

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import HT_DCPERIOD
    >>> close = np.random.randn(100) + 100
    >>> period = HT_DCPERIOD(close)
    >>> print(period)

    See Also
    --------
    HT_DCPHASE : Hilbert Transform - Dominant Cycle Phase
    HT_PHASOR : Hilbert Transform - Phasor Components
    HT_TRENDMODE : Hilbert Transform - Trend vs Cycle Mode
    """
    # Convert to numpy array if needed
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Pre-allocate output array
    output = np.empty(n, dtype=np.float64)
    _ht_dcperiod_numba(close, output)

    return output


@jit(nopython=True, cache=True)
def _ht_dcphase_numba(close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled HT_DCPHASE calculation

    Hilbert Transform - Dominant Cycle Phase
    """
    n = len(close)
    lookback = 32

    # Fill lookback with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Initialize arrays (similar to DCPERIOD)
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        # Smooth price
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        # Detrend
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature
        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        # Advance phase by 90 degrees
        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition
        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]

            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        # Compute DC Phase
        if i >= 11:
            if i2[i] != 0.0:
                dc_phase = np.arctan(q2[i] / i2[i]) * 180.0 / np.pi
            else:
                dc_phase = 0.0

            # Adjust for quadrant
            if i2[i] < 0.0:
                dc_phase += 180.0
            if dc_phase < 0.0:
                dc_phase += 360.0

            output[i] = dc_phase
        else:
            output[i] = 0.0


def HT_DCPHASE(close: Union[np.ndarray, list]) -> np.ndarray:
    """
    Hilbert Transform - Dominant Cycle Phase

    This function calculates the phase of the dominant cycle using the
    Hilbert Transform. The phase represents where in the cycle the market
    currently is, measured in degrees (0-360).

    Parameters
    ----------
    close : array-like
        Close prices array

    Returns
    -------
    np.ndarray
        Array of dominant cycle phase values (in degrees)

    Notes
    -----
    - Compatible with TA-Lib HT_DCPHASE signature
    - Uses Numba JIT compilation for performance
    - The first 32 values will be NaN (unstable period)
    - Phase values range from 0 to 360 degrees
    - Based on John Ehlers' MESA techniques

    Interpretation:
    - 0-90 degrees: Early cycle, potential accumulation
    - 90-180 degrees: Mid cycle, trending phase
    - 180-270 degrees: Late cycle, potential distribution
    - 270-360 degrees: Cycle completion, reversal zone
    - Phase crossing 0/360: Cycle restart signal

    Common Uses:
    - Cycle position identification
    - Timing entry/exit points
    - Predicting cycle turns
    - Confirming trend changes

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import HT_DCPHASE
    >>> close = np.random.randn(100) + 100
    >>> phase = HT_DCPHASE(close)
    >>> print(phase)

    See Also
    --------
    HT_DCPERIOD : Hilbert Transform - Dominant Cycle Period
    HT_PHASOR : Hilbert Transform - Phasor Components
    HT_SINE : Hilbert Transform - SineWave
    """
    # Convert to numpy array
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Pre-allocate output
    output = np.empty(n, dtype=np.float64)
    _ht_dcphase_numba(close, output)

    return output


@jit(nopython=True, cache=True)
def _ht_phasor_numba(close: np.ndarray, inphase: np.ndarray, quadrature: np.ndarray) -> None:
    """
    Numba-compiled HT_PHASOR calculation

    Returns InPhase and Quadrature components
    """
    n = len(close)
    lookback = 32

    # Fill lookback with NaN
    for i in range(lookback):
        inphase[i] = np.nan
        quadrature[i] = np.nan

    # Initialize arrays
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        # Smooth price
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        # Detrend
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature
        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        # Advance phase
        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition
        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]

            # Smooth
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

            inphase[i] = i2[i]
            quadrature[i] = q2[i]
        else:
            inphase[i] = 0.0
            quadrature[i] = 0.0


def HT_PHASOR(close: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hilbert Transform - Phasor Components

    This function returns the InPhase and Quadrature components of the
    Hilbert Transform. These components form a phasor representation of
    the market cycle.

    Parameters
    ----------
    close : array-like
        Close prices array

    Returns
    -------
    tuple of np.ndarray
        (inphase, quadrature) - Two arrays with the phasor components

    Notes
    -----
    - Compatible with TA-Lib HT_PHASOR signature
    - Uses Numba JIT compilation for performance
    - The first 32 values will be NaN (unstable period)
    - Returns two arrays: InPhase and Quadrature
    - Based on John Ehlers' MESA analysis

    Components:
    - InPhase: Real component of the phasor
    - Quadrature: Imaginary component of the phasor (90° phase shift)
    - Together they define the cycle's position and amplitude

    Interpretation:
    - Magnitude = sqrt(InPhase² + Quadrature²)
    - Phase Angle = arctan(Quadrature / InPhase)
    - InPhase > 0, Quadrature > 0: First quadrant (0-90°)
    - InPhase < 0, Quadrature > 0: Second quadrant (90-180°)
    - InPhase < 0, Quadrature < 0: Third quadrant (180-270°)
    - InPhase > 0, Quadrature < 0: Fourth quadrant (270-360°)

    Common Uses:
    - Advanced cycle analysis
    - Building custom cycle indicators
    - Phase and amplitude extraction
    - Cycle prediction algorithms
    - Component for other Hilbert Transform indicators

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure import HT_PHASOR
    >>> close = np.random.randn(100) + 100
    >>> inphase, quadrature = HT_PHASOR(close)
    >>> magnitude = np.sqrt(inphase**2 + quadrature**2)
    >>> phase = np.arctan2(quadrature, inphase) * 180 / np.pi

    See Also
    --------
    HT_DCPERIOD : Hilbert Transform - Dominant Cycle Period
    HT_DCPHASE : Hilbert Transform - Dominant Cycle Phase
    HT_SINE : Hilbert Transform - SineWave
    """
    # Convert to numpy array
    close = np.asarray(close, dtype=np.float64)

    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    # Pre-allocate outputs
    inphase = np.empty(n, dtype=np.float64)
    quadrature = np.empty(n, dtype=np.float64)
    _ht_phasor_numba(close, inphase, quadrature)

    return inphase, quadrature


@jit(nopython=True, cache=True)
def _ht_sine_numba(close: np.ndarray, sine: np.ndarray, leadsine: np.ndarray) -> None:
    """Numba-compiled HT_SINE calculation"""
    n = len(close)
    lookback = 32

    for i in range(lookback):
        sine[i] = np.nan
        leadsine[i] = np.nan

    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    im = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    smooth_period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        if i >= 11:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            re[i] = 0.2 * re[i] + 0.8 * re[i-1]
            im[i] = 0.2 * im[i] + 0.8 * im[i-1]

            if im[i] != 0.0 and re[i] != 0.0:
                period[i] = 360.0 / (np.arctan(im[i] / re[i]) * 180.0 / np.pi)

            if period[i] > 1.5 * period[i-1]:
                period[i] = 1.5 * period[i-1]
            if period[i] < 0.67 * period[i-1]:
                period[i] = 0.67 * period[i-1]
            if period[i] < 6:
                period[i] = 6
            if period[i] > 50:
                period[i] = 50

            period[i] = 0.2 * period[i] + 0.8 * period[i-1]
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]

            if i2[i] != 0.0:
                dc_phase = np.arctan(q2[i] / i2[i]) * 180.0 / np.pi
            else:
                dc_phase = 0.0

            if i2[i] < 0.0:
                dc_phase += 180.0
            if dc_phase < 0.0:
                dc_phase += 360.0

            dc_phase_rad = dc_phase * np.pi / 180.0
            sine[i] = np.sin(dc_phase_rad)
            leadsine[i] = np.sin(dc_phase_rad + 45.0 * np.pi / 180.0)


def HT_SINE(close: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """Hilbert Transform - SineWave"""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    sine = np.empty(n, dtype=np.float64)
    leadsine = np.empty(n, dtype=np.float64)
    _ht_sine_numba(close, sine, leadsine)
    return sine, leadsine


@jit(nopython=True, cache=True)
def _ht_trendline_numba(close: np.ndarray, output: np.ndarray) -> None:
    """Numba-compiled HT_TRENDLINE calculation"""
    n = len(close)
    lookback = 32

    for i in range(lookback):
        output[i] = np.nan

    for i in range(lookback, n):
        if i >= 3:
            output[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            output[i] = close[i]


def HT_TRENDLINE(close: Union[np.ndarray, list]) -> np.ndarray:
    """Hilbert Transform - Instantaneous Trendline"""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _ht_trendline_numba(close, output)
    return output


@jit(nopython=True, cache=True)
def _ht_trendmode_numba(close: np.ndarray, output: np.ndarray) -> None:
    """Numba-compiled HT_TRENDMODE calculation"""
    n = len(close)
    lookback = 63

    for i in range(lookback):
        output[i] = np.nan

    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    dc_period = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        if i >= 12:
            dc_period[i] = 15.0  # Simplified: use average period
            deviation = abs(close[i] - smooth[i])
            avg_deviation = 0.0
            count = 0
            for j in range(max(0, i-19), i+1):
                avg_deviation += abs(close[j] - smooth[j])
                count += 1
            avg_deviation = avg_deviation / count if count > 0 else 1.0

            if dc_period[i] > 30 or deviation > 2.0 * avg_deviation:
                trend[i] = 1.0
            else:
                trend[i] = 0.0

            if i > lookback:
                trend[i] = 0.8 * trend[i-1] + 0.2 * trend[i]

            output[i] = 1 if trend[i] > 0.5 else 0
        else:
            output[i] = 0


def HT_TRENDMODE(close: Union[np.ndarray, list]) -> np.ndarray:
    """Hilbert Transform - Trend vs Cycle Mode"""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _ht_trendmode_numba(close, output)
    return output
