"""
Cycle Indicators - Hilbert Transform based cycle analysis functions

These indicators use the Hilbert Transform to analyze market cycles.
The Hilbert Transform is a mathematical technique that produces in-phase
and quadrature components of a price series, allowing cycle analysis.
"""

import numpy as np
from typing import Union, Tuple

# Import CPU implementations
from ..cpu.cycle_indicators import (
    _ht_dcperiod_numba,
    _ht_dcphase_numba,
    _ht_phasor_numba,
    _ht_sine_numba,
    _ht_trendline_numba,
    _ht_trendmode_numba,
)


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


def HT_TRENDLINE(close: Union[np.ndarray, list]) -> np.ndarray:
    """Hilbert Transform - Instantaneous Trendline"""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _ht_trendline_numba(close, output)
    return output


def HT_TRENDMODE(close: Union[np.ndarray, list]) -> np.ndarray:
    """Hilbert Transform - Trend vs Cycle Mode"""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _ht_trendmode_numba(close, output)
    return output
